import math
import triton
import torch

from flash_attn import flash_attn_varlen_func

b, s_q, s_k, h = 1, 4096, 4096, 128
dtype = torch.bfloat16
device = torch.device("cuda:0")
torch.set_default_dtype(dtype)
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
    # assert diff < 1e-5


def timer(func):
    t = triton.testing.do_bench(func)
    FLOPS = b * s_q * s_k * h * (d + v_dim) * 3
    bytes = b * s_q * h * (d + v_dim) * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {FLOPS / 10 ** 9 / t} tflops, {bytes / 10 ** 6 / t} GB/s")
    return t


def scaled_dot_product_attention(query, key, value, is_causal=False) -> torch.Tensor:
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    # print("attn_weight:", attn_weight)
    return attn_weight.to(dtype) @ value


def test_flash_attention(d, v_dim):
    print(b, h, s_q, s_k, d, v_dim)

    q = (torch.randn(b, s_q, h, d) - 0.5).to(torch.float8_e4m3fn).to(dtype)
    k = (torch.randn(b, s_k, h, d) - 0.5).to(torch.float8_e4m3fn).to(dtype)
    v = (torch.randn(b, s_k, h, v_dim) - 0.5).to(torch.float8_e4m3fn).to(dtype)
    grad_output = (torch.randn(b, s_q, h, v_dim) - 0.5).to(torch.float8_e4m3fn).to(dtype)
    # print("q:", q, "k:", k, "v:", v, "do:", grad_output)
    cu_seqlens_q = torch.arange(0, (b + 1) * s_q, step=s_q, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, (b + 1) * s_k, step=s_k, dtype=torch.int32)

    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()

    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()

    def flash_attn():
        q1.grad = k1.grad = v1.grad = None
        return flash_attn_varlen_func(
            q1.view(b * s_q, h, d),
            k1.view(b * s_k, h, d),
            v1.view(b * s_k, h, v_dim),
            cu_seqlens_q, cu_seqlens_k, s_q, s_k,
            causal=True,
            use_fp8=True,
        ).view(b, s_q, h, v_dim)

    def torch_attn():
        q2.grad = k2.grad = v2.grad = None
        return scaled_dot_product_attention(
            q2.transpose(1, 2),
            k2.transpose(1, 2),
            v2.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)

    out_flash_attn = flash_attn()
    out_torch_attn = torch_attn()
    assert_close(out_flash_attn, out_torch_attn, "out")

    out_flash_attn.backward(grad_output)
    out_torch_attn.backward(grad_output)
    assert_close(q1.grad, q2.grad, "dq")
    assert_close(k1.grad, k2.grad, "dk")
    assert_close(v1.grad, v2.grad, "dv")

    timer(lambda: flash_attn().backward(grad_output))
    timer(lambda: flash_attn().backward(grad_output))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        flash_attn().backward(grad_output)
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))
    # prof.export_chrome_trace("tests/flash_attn_trace.json")


if __name__ == "__main__":
    # for d, v_dim in [(32, 32), (64, 64), (96, 96), (128, 128), (128, 64), (160, 160), (192, 192), (192, 128), (192, 64), (224, 224), (256, 256)]:
    for d, v_dim in [(192, 128)]:
        test_flash_attention(d, v_dim)
