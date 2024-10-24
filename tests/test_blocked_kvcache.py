import math
import triton
import torch
import random
from flash_attn.flash_attn_interface import get_kvcache_block_size, flash_attn_with_blocked_kvcache

b, s, h_q, h_kv = 132, 4096, 64, 1
dtype = torch.bfloat16
device = torch.device("cuda:0")
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
print("max:", max_seqlen, "sum:", total_seqlens, cache_seqlens.tolist())


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
    print(f"{name}: diff {diff}, amax {amax}")
    # assert diff < 1e-5


def timer(func, name=""):
    t = triton.testing.do_bench(func, fast_flush=False)
    FLOPS = total_seqlens * h_q * (d + v_dim) * 2
    bytes = total_seqlens * h_kv * d * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {FLOPS / 10 ** 9 / t} tflops, {bytes / 10 ** 6 / t} GB/s")
    return t


def scaled_dot_product_attention(query, key, value) -> torch.Tensor:
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


@torch.inference_mode()
def test_flash_attention(d, v_dim):
    block_size = get_kvcache_block_size(d)
    print(b, s, h_q, h_kv, d, v_dim)

    s_q = 1
    q = torch.randn(b, s_q, h_q, d)

    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    # block_table = torch.nn.functional.pad(block_table, (0, 32768 // block_size))
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    blocked_v = blocked_k[..., :v_dim]

    def blocked_flash_attn():
        return flash_attn_with_blocked_kvcache(
            q, blocked_k, blocked_v, block_table, cache_seqlens, causal=True,
            # window_size=(511, 0),
        )

    def torch_attn():
        out = torch.empty(b, s_q, h_q, v_dim)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            out[i] = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, v_dim)[begin:end].transpose(0, 1),
            ).transpose(0, 1)
        return out

    out_blocked_flash = blocked_flash_attn()
    out_torch_attn = torch_attn()
    assert_close(out_blocked_flash, out_torch_attn, "blocked_flash_attn")

    timer(blocked_flash_attn)
    timer(blocked_flash_attn)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        blocked_flash_attn()
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=50))
    # prof.export_chrome_trace("tests/flash_attn_trace.json")


if __name__ == "__main__":
    # for d, v_dim in [(32, 32), (64, 64), (96, 96), (128, 128), (160, 160), (192, 192), (224, 224), (256, 256), (512, 512), (576, 512)]:
    for d, v_dim in [(576, 512)]:
        torch.cuda.empty_cache()
        test_flash_attention(d, v_dim)
