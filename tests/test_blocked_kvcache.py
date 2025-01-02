import math
import triton
import torch
import random
from flash_attn.flash_attn_interface import *

b, s, h_q, h_kv = 64, 4096, 64, 1
s_q = 3
causal = True
shared_kv = True
dtype = torch.bfloat16
device = torch.device("cuda:5")
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.cuda.set_device(device)
torch.manual_seed(0)
random.seed(0)

cache_seqlens = torch.full((b,), s, dtype=torch.int32)
for i in range(b):
    cache_seqlens[i] = min(max(random.normalvariate(1000, 1000), s_q), 20480)
cache_seqlens[0] = 65536
cache_seqlens = torch.sort(cache_seqlens, descending=True).values
total_seqlens = cache_seqlens.sum().item()
mean_seqlens = cache_seqlens.float().mean().int().item()
max_seqlen = cache_seqlens.max().item()
max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
cu_seqlens_q = torch.arange(0, (b + 1) * s_q, step=s_q, dtype=torch.int32)
cu_seqlens_k = torch.cumsum(torch.nn.functional.pad(cache_seqlens, (1, 0)), 0).int()
print("max:", max_seqlen, "sum:", total_seqlens, "mean:", mean_seqlens)


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
    FLOPS = s_q * total_seqlens * h_q * (d + v_dim) * 2
    bytes = total_seqlens * h_kv * (d + (v_dim if not shared_kv else 0)) * (torch.finfo(dtype).bits // 8)

    print(f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} tflops, {bytes / 10 ** 6 / t:.0f} GB/s")
    return t


def scaled_dot_product_attention(query, key, value, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


@torch.inference_mode()
def test_flash_attention(d, v_dim):
    block_size = get_kvcache_block_size(d)
    print(b, s_q, s, h_q, h_kv, d, v_dim, causal)

    q = torch.randn(b, s_q, h_q, d)

    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    blocked_v = torch.randn(block_table.numel(), block_size, h_kv, v_dim) if not shared_kv else blocked_k[..., :v_dim]
    try:
        tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens.cpu(), (h_q // h_kv) * s_q, h_kv, block_size)
        tile_scheduler_metadata = tile_scheduler_metadata.cuda()
        num_splits = num_splits.cuda()
        # print(tile_scheduler_metadata)
        # print(num_splits)
        # print(tile_scheduler_metadata.shape, num_splits.shape)
        # print(cache_seqlens)
    except:
        pass

    def blocked_flash_attn():
        try:
            return flash_attn_with_blocked_kvcache_mla(
                q, blocked_k, block_table, cache_seqlens, v_dim,
                tile_scheduler_metadata, num_splits, causal=causal, return_softmax_lse=True,
            )
        except:
            return flash_attn_with_blocked_kvcache(
                q, blocked_k, blocked_v, block_table, cache_seqlens, causal=causal, return_softmax_lse=True,
            )

    def torch_attn():
        out = torch.empty(b, s_q, h_q, v_dim, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, v_dim)[begin:end].transpose(0, 1),
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_flash, lse_flash = blocked_flash_attn()
    out_torch, lse_torch = torch_attn()
    assert_close(out_flash, out_torch, "out")
    assert_close(lse_flash, lse_torch, "lse")

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
