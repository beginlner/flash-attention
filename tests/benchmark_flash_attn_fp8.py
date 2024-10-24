import math
import triton
import torch

from flash_attn import *

b, s_q, s_k, h, h_kv = 4, 4096, 4096, 16, 16
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
    t = triton.testing.do_bench(func)
    FLOPS = b * h * (d + v_dim) * ((s_k * s_k - max(s_k - s_q, 0) * max(s_k - s_q, 0)) if causal else s_q * s_k * 2)
    bytes = b * s_q * h * (d + v_dim) * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {FLOPS / 10 ** 9 / t} tflops, {bytes / 10 ** 6 / t} GB/s")
    return t

def scaled_dot_product_attention(query, key, value, is_causal=False) -> torch.Tensor:
    key = key.repeat_interleave(h // h_kv, dim=0)
    value = value.repeat_interleave(h // h_kv, dim=0)
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
    return attn_weight.to(dtype) @ value


def test_flash_attention(d, v_dim):
    print(b, h, h_kv, s_q, s_k, d, v_dim)

    q = torch.randn(cu_seqlens_q[-1].item(), h, d)
    k = torch.randn(cu_seqlens_k[-1].item(), h_kv, d)
    v = torch.randn(cu_seqlens_k[-1].item(), h_kv, v_dim)
    grad_output = torch.randn(cu_seqlens_q[-1].item(), h, v_dim)
    # print("q:", q, "k:", k, "v:", v, "do:", grad_output)

    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()

    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()

    def flash_attn(fp8_type=None):
        q1.grad = k1.grad = v1.grad = None
        # return flash_attn_varlen_qkvpacked_func(
        #     torch.stack([q1, k1, v1], dim=1),
        #     cu_seqlens_q, q1.shape[0],
        #     causal=True,
        #     fp8_type=fp8_type,
        # )
        # return flash_attn_varlen_kvpacked_func(
        #     q1, torch.stack([k1, v1], dim=1),
        #     cu_seqlens_q, cu_seqlens_k, q1.shape[0], k1.shape[0],
        #     causal=True,
        #     fp8_type=fp8_type,
        # )
        return flash_attn_varlen_func(
            q1, k1, v1,
            cu_seqlens_q, cu_seqlens_k, q1.shape[0], k1.shape[0],
            causal=True,
            fp8_type=fp8_type,
        )

    def torch_attn():
        q2.grad = k2.grad = v2.grad = None
        out = torch.empty(cu_seqlens_q[-1].item(), h, v_dim)
        for i in range(b):
            out[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = scaled_dot_product_attention(
                q2[cu_seqlens_q[i]:cu_seqlens_q[i + 1]].transpose(0, 1),
                k2[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].transpose(0, 1),
                v2[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].transpose(0, 1),
                is_causal=True,
            ).transpose(0, 1)
        return out

    out_flash_attn = flash_attn()
    out_torch_attn = torch_attn()
    assert_close(out_flash_attn, out_torch_attn, "out_bf16")

    # out_flash_attn.backward(grad_output)
    # out_torch_attn.backward(grad_output)
    # assert_close(q1.grad, q2.grad, "dq_bf16")
    # assert_close(k1.grad, k2.grad, "dk_bf16")
    # assert_close(v1.grad, v2.grad, "dv_bf16")

    out_flash_attn = flash_attn(("e4m3", "bf16"))
    out_torch_attn = torch_attn()
    assert_close(out_flash_attn, out_torch_attn, "out")

    # out_flash_attn.backward(grad_output)
    # out_torch_attn.backward(grad_output)
    # assert_close(q1.grad, q2.grad, "dq")
    # assert_close(k1.grad, k2.grad, "dk")
    # assert_close(v1.grad, v2.grad, "dv")

    timer(lambda: flash_attn())#.backward(grad_output))
    timer(lambda: flash_attn())#.backward(grad_output))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        flash_attn()#.backward(grad_output)
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))

    timer(lambda: flash_attn(("e4m3", "bf16")))#.backward(grad_output))
    timer(lambda: flash_attn(("e4m3", "bf16")))#.backward(grad_output))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        flash_attn(("e4m3", "bf16"))#.backward(grad_output)
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))


if __name__ == "__main__":
    # for d, v_dim in [(128, 128), (192, 128)]:
    for d, v_dim in [(192, 128)]:
        test_flash_attention(d, v_dim)
