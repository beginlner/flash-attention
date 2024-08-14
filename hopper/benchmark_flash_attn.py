import math
import triton
import torch

from flash_attn_interface import flash_attn_func as flash_attn_func_hopper
from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_hopper
from flash_attn.flash_attn_interface import flash_attn_varlen_func

b = 4
s_q = 4096
s_k = 4096
h = 32
h_k = 32
d = 192
v_dim = 128
causal = True
has_bwd = False
# dtype = torch.float8_e4m3fn
dtype = torch.bfloat16
default_dtype = torch.bfloat16
torch.set_default_dtype(default_dtype)
device = torch.device("cuda:0")
torch.set_default_device(device)
torch.cuda.set_device(device)
torch.manual_seed(0)


def calc_diff(x, y):
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    denominator = (x * x + y * y).sum()
    if denominator < 1e-12:
        return 0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item(), RMSE


def assert_close(x, y, name=""):
    x, y = x.double(), y.double()
    diff = calc_diff(x, y)
    amax = (x - y).abs().max()
    print(f"{name} diff: {diff}, amax: {amax}")
    assert diff[0] < 1e-2


def timer(func):
    t = triton.testing.do_bench(func, fast_flush=False)
    FLOPS = b * h * (d + v_dim) * ((s_k * s_k - max(s_k - s_q, 0) * max(s_k - s_q, 0)) if causal else s_q * s_k * 2) * (3.5 if has_bwd else 1)
    BYTES = b * s_k * h_k * (d + v_dim) * (torch.finfo(dtype).bits / 8)
    print(f"{t} ms, {FLOPS / 10 ** 9 / t} TFLOP/s, {BYTES / 10 ** 6 / t} GB/s")
    return t


def scaled_dot_product_attention(query, key, value, is_causal=False) -> torch.Tensor:
    key = key.repeat_interleave(h // h_k, dim=1)
    value = value.repeat_interleave(h // h_k, dim=1)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight.to(query.dtype) @ value


def test_flash_attention():
    print(b, s_q, s_k, h, h_k, d, v_dim, causal)

    q = torch.randn(b * s_q, h, d)
    k = torch.randn(b * s_k, h_k, d)
    v = torch.randn(b * s_k, h_k, v_dim)
    grad_out = torch.randn(b * s_q, h, v_dim)

    q1 = q.clone().to(dtype).requires_grad_()
    k1 = k.clone().to(dtype).requires_grad_()
    v1 = v.clone().to(dtype).requires_grad_()
    # v1 = k1
    cu_seqlens_q = torch.arange(0, (b + 1) * s_q, s_q, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, (b + 1) * s_k, s_k, dtype=torch.int32)

    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()

    def flash_attn():
        q1.grad = k1.grad = v1.grad = None
        return flash_attn_func_hopper(q1.unflatten(0, (b, s_q)), k1.unflatten(0, (b, s_k)), v1.unflatten(0, (b, s_k)), causal=causal)[0].flatten(0, 1)
        # return flash_attn_varlen_func_hopper(q1, k1, v1, cu_seqlens_q, cu_seqlens_k, s_q, s_k, causal=causal)[0]
        # return flash_attn_varlen_func(q1, k1, v1, cu_seqlens_q, cu_seqlens_k, s_q, s_k, causal=causal)

    def torch_attn():
        q2.grad = k2.grad = v2.grad = None
        return scaled_dot_product_attention(
            q2.unflatten(0, (b, s_q)).float().transpose(1, 2),
            k2.unflatten(0, (b, s_k)).float().transpose(1, 2),
            v2.unflatten(0, (b, s_k)).float().transpose(1, 2),
            is_causal=causal,
        ).to(dtype).transpose(1, 2).flatten(0, 1)

    def fn():
        flash_attn().backward(grad_out) if has_bwd else flash_attn()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))

    out_flash_attn = flash_attn()
    out_torch_attn = torch_attn()
    assert_close(out_flash_attn, out_torch_attn, "out")

    if has_bwd:
        out_flash_attn.backward(grad_out)
        out_torch_attn.backward(grad_out)
        assert_close(q1.grad, q2.grad, "dq")
        assert_close(k1.grad, k2.grad, "dk")
        assert_close(v1.grad, v2.grad, "dv")

    timer(fn)
    timer(fn)


if __name__ == "__main__":
    test_flash_attention()
