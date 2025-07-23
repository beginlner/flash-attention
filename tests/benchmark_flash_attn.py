import triton
import torch
from torch.nn.functional import scaled_dot_product_attention

from flash_attn import flash_attn_varlen_func

b, s, h, d = 1, 4096, 128, 192
v_dim = 128
dtype = torch.bfloat16
device = torch.device("cuda:0")
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.cuda.set_device(device)
torch.manual_seed(0)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator < 1e-12:
        return 0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item()


def assert_close(x, y, name=""):
    diff = calc_diff(x, y)
    amax = (x - y).abs().max()
    print(f"{name} diff: {diff}, amax: {amax}")
    assert diff < 2e-5


def timer(func):
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    e = 100
    for _ in range(e):
        func()
    en.record()
    torch.cuda.synchronize()
    t = st.elapsed_time(en) / e
    FLOPS = b * s * s * h * (d + v_dim) * 6
    bytes = b * s * h * (d + v_dim) * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {FLOPS / 10 ** 9 / t} tflops, {bytes / 10 ** 6 / t} GB/s")
    return t


def test_flash_attention(b, s, h, d):
    print(b, h, s, d, v_dim)

    q = torch.randn(b, s, h, d)
    k = torch.randn(b, s, h, d)
    v = torch.randn(b, s, h, v_dim)
    cu_seqlens = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32)

    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()

    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()

    def flash_attn():
        q1.grad = k1.grad = v1.grad = None
        return flash_attn_varlen_func(
            q1.view(b * s, h, d), k1.view(b * s, h, d), v1.view(b * s, h, v_dim), cu_seqlens, cu_seqlens, s, s, causal=True).view(b, s, h, v_dim)

    def torch_attn():
        q2.grad = k2.grad = v2.grad = None
        return scaled_dot_product_attention(
            q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2), is_causal=True).transpose(1, 2)

    out_flash_attn = flash_attn()
    out_torch_attn = torch_attn()
    assert_close(out_flash_attn, out_torch_attn, "out")

    grad_triton = torch.randn_like(out_flash_attn)
    out_flash_attn.backward(grad_triton)
    out_torch_attn.backward(grad_triton)
    assert_close(q1.grad, q2.grad, "dq")
    assert_close(k1.grad, k2.grad, "dk")
    assert_close(v1.grad, v2.grad, "dv")

    grad_triton = torch.randn_like(out_flash_attn)
    timer(lambda: flash_attn().backward(grad_triton))
    timer(lambda: flash_attn().backward(grad_triton))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        flash_attn().backward(grad_triton)
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))
    # prof.export_chrome_trace("tests/flash_attn_trace.json")

def test_flash_attention_max_logits(b, s, h, d):
    print(b, h, s, d, v_dim)

    q = torch.randn(b, s, h, d)
    k = torch.randn(b, s, h, d)
    v = torch.randn(b, s, h, v_dim)
    cu_seqlens = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32)

    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()

    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()

    def flash_attn():
        q1.grad = k1.grad = v1.grad = None
        o, max_logits = flash_attn_varlen_func(
            q1.view(b * s, h, d), k1.view(b * s, h, d), v1.view(b * s, h, v_dim), cu_seqlens, cu_seqlens, s, s, causal=True, return_max_logits=True)
        o = o.view(b, s, h, v_dim)
        return o, max_logits
    
    def torch_attn_with_max_logits(q2, k2, v2):
        assert b == 1 # ref attn does not support b > 1
        q2.grad = k2.grad = v2.grad = None
        q2, k2, v2 = q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2)
        attn = (q2 @ k2.transpose(-2, -1)) / (d ** 0.5)
        causal = True
        if causal:
            mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float('-inf'))
        max_logits = attn.max(dim=-1, keepdim=True)[0]
        attn_weights = torch.softmax(attn, dim=-1)
        output = attn_weights @ v2
        return output, max_logits

    out_flash_attn, max_logits = flash_attn()
    out_torch_attn, max_logits_torch = torch_attn_with_max_logits(q2, k2, v2)
    max_logits_torch = max_logits_torch.reshape((b, h, s))
    out_torch_attn = out_torch_attn.transpose(1, 2)
    assert_close(out_flash_attn, out_torch_attn, "out")
    assert_close(max_logits, max_logits_torch, "max_logits")

if __name__ == "__main__":
    test_flash_attention_max_logits(b, s, h, d)
