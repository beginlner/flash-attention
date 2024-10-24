import triton
import torch
from torch.nn.functional import scaled_dot_product_attention

from flash_attn import flash_attn_varlen_func

b, s_q, s_k, h = 4, 4096, 4096, 32
causal = True
dtype = torch.bfloat16
device = torch.device("cuda:0")
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.cuda.set_device(device)
torch.manual_seed(0)

cache_seqlens_q = torch.full((b,), s_q, dtype=torch.int32)
cache_seqlens_k = torch.full((b,), s_k, dtype=torch.int32)
cu_seqlens_q = torch.nn.functional.pad(cache_seqlens_q, (1, 0)).cumsum(0, dtype=torch.int32)
cu_seqlens_k = torch.nn.functional.pad(cache_seqlens_k, (1, 0)).cumsum(0, dtype=torch.int32)
print(cu_seqlens_q, cu_seqlens_k)


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
    # assert diff[0] < 1e-5


def timer(func):
    t = triton.testing.do_bench(func, fast_flush=False)
    FLOPS = b * h * (d + v_dim) * ((s_k * s_k - max(s_k - s_q, 0) * max(s_k - s_q, 0)) if causal else s_q * s_k * 2) * 3.5
    print(f"{t} ms, {FLOPS / 10 ** 9 / t} tflops")
    return t


def test_flash_attention(d, v_dim):
    print(b, h, s_q, s_k, d, v_dim)

    q = torch.randn(b, s_q, h, d)
    k = torch.randn(b, s_k, h, d)
    v = torch.randn(b, s_k, h, v_dim)
    grad_out = torch.randn(b, s_q, h, v_dim)

    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()

    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()

    def flash_attn():
        q1.grad = k1.grad = v1.grad = None
        return flash_attn_varlen_func(
            q1.view(b * s_q, h, d), k1.view(b * s_k, h, d), v1.view(b * s_k, h, v_dim), cu_seqlens_q, cu_seqlens_k, s_q, s_k, causal=causal).view(b, s_q, h, v_dim)

    def torch_attn():
        q2.grad = k2.grad = v2.grad = None
        return scaled_dot_product_attention(
            q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2), is_causal=causal).transpose(1, 2)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        flash_attn().backward(grad_out)
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))
    # prof.export_chrome_trace("tests/flash_attn_trace.json")

    out_flash_attn = flash_attn()
    out_torch_attn = torch_attn()
    assert_close(out_flash_attn, out_torch_attn, "out")

    out_flash_attn.backward(grad_out)
    out_torch_attn.backward(grad_out)
    assert_close(q1.grad, q2.grad, "dq")
    assert_close(k1.grad, k2.grad, "dk")
    assert_close(v1.grad, v2.grad, "dv")

    timer(lambda: flash_attn().backward(grad_out))
    timer(lambda: flash_attn().backward(grad_out))


if __name__ == "__main__":
    # for d, v_dim in [(32, 32), (64, 64), (96, 96), (128, 128), (128, 64), (160, 160), (192, 192), (192, 128), (192, 64), (224, 224), (256, 256)]:
    for d, v_dim in [(192, 128)]:
        test_flash_attention(d, v_dim)
