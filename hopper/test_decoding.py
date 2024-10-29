import math
import triton
import torch
import random
from flash_attn_interface import flash_attn_with_kvcache

b, s, h_q, h_kv = 132, 2048, 64, 1
dtype = torch.bfloat16
device = torch.device("cuda:1")
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.cuda.set_device(device)
torch.manual_seed(0)
random.seed(0)

cache_seqlens = torch.full((b,), s, dtype=torch.int32)
# for i in range(b):
#     cache_seqlens[i] = min(max(random.normalvariate(3000, 1000), 1), 20480)
# cache_seqlens[0] = 131072
cache_seqlens = torch.sort(cache_seqlens, descending=True).values
total_seqlens = cache_seqlens.sum().item()
max_seqlen = cache_seqlens.max().item()
max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
print("max_seqlen:", max_seqlen, "sum_seqlen:", total_seqlens)


def calc_diff(x, y):
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator < 1e-12:
        return 0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item(), RMSE


def assert_close(x, y, name=""):
    x, y = x.double(), y.double()
    diff = calc_diff(x, y)
    amax = (x - y).abs().max()
    print(f"diff {diff}, amax {amax} {name}")
    # assert diff < 1e-5


def timer(func, name=""):
    t = triton.testing.do_bench(func, fast_flush=False)
    FLOPS = total_seqlens * h_q * (d + v_dim) * 2
    bytes = total_seqlens * h_kv * (d + v_dim) * (torch.finfo(dtype_real).bits // 8)

    print(f"{t:.3f} ms, {bytes / 10 ** 6 / t:.0f} GB/s {name}")
    return t


def scaled_dot_product_attention(query, key, value) -> torch.Tensor:
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


@torch.inference_mode()
def test_flash_attention(d, v_dim):
    print(b, s, h_q, h_kv, d, v_dim, dtype_real)

    s_q = 1
    q = torch.randn(b, s_q, h_q, d)
    k = torch.randn(b, s, h_kv, d)
    v = torch.randn(b, s, h_kv, v_dim)
    q_real = q.to(dtype_real)
    k_real = k.to(dtype_real)
    v_real = v.to(dtype_real)

    def fa3_decoding():
        return flash_attn_with_kvcache(
            q_real, k_real, v_real, cache_seqlens, causal=True,
            gqa_parallel=False,
            num_splits=1,
        )

    def torch_attn():
        out = torch.empty(b, s_q, h_q, v_dim)
        for i in range(b):
            out[i] = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                k[i, :cache_seqlens[i]].transpose(0, 1),
                v[i, :cache_seqlens[i]].transpose(0, 1),
            ).transpose(0, 1)
        return out

    out_blocked_flash = fa3_decoding()
    out_torch_attn = torch_attn()
    assert_close(out_blocked_flash, out_torch_attn)

    timer(fa3_decoding)
    timer(fa3_decoding)

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    #     fa3_decoding()
    # print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))
    # prof.export_chrome_trace("tests/flash_attn_trace.json")


if __name__ == "__main__":
    # for d, v_dim in [(32, 32), (64, 64), (96, 96), (128, 128), (160, 160), (192, 192), (224, 224), (256, 256), (512, 512), (576, 512)]:
    for d, v_dim in [(256, 256)]:
        torch.cuda.empty_cache()
        dtype_real = torch.bfloat16
        test_flash_attention(d, v_dim)
        dtype_real = torch.float8_e4m3fn
        test_flash_attention(d, v_dim)
