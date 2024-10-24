import random
import math
import triton
import torch
from flash_attn.flash_attn_interface import get_kvcache_block_size, flash_attn_with_blocked_kvcache

b, s, h_q, h_kv, d = 256, 2048, 128, 1, 576
v_dim = 512
k0_bits, k1_bits = 4, 16
k0_dtype, k1_dtype = "int4", "bfloat16"
split_length = 384
assert (split_length * k0_bits + (d - split_length) * k1_bits) % 32 == 0
compressed_head_size = (split_length * k0_bits + (d - split_length) * k1_bits) // 32
block_size = get_kvcache_block_size(d)
s_pad = triton.cdiv(s, block_size) * block_size
dtype = torch.bfloat16
device = torch.device("cuda:0")
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.cuda.set_device(device)
torch.manual_seed(0)
random.seed(0)

cache_seqlens = torch.full((b,), s, dtype=torch.int32)
for i in range(b):
    cache_seqlens[i] = min(max(random.normalvariate(1100, 500), 1), 2048)
cache_seqlens = torch.sort(cache_seqlens, descending=True).values
total_seqlens = cache_seqlens.sum().item()
max_seqlen = cache_seqlens.max().item()
max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
print("max:", max_seqlen, "sum:", total_seqlens, cache_seqlens.tolist())


def timer(func, name=""):
    t = triton.testing.do_bench(func, fast_flush=False)
    FLOPS = total_seqlens * h_q * (d + v_dim) * 2
    bytes = total_seqlens * h_kv * d * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {FLOPS / 10 ** 9 / t} tflops, {bytes / 10 ** 6 / t} GB/s")
    return t


def create_k():
    x0 = torch.randint(low=-(1 << (k0_bits - 1)), high=(1 << (k0_bits - 1)), size=(b, max_seqlen_pad, h_kv, split_length), dtype=torch.int32)
    if k1_dtype == "bfloat16":
        x1 = torch.randn((b, max_seqlen_pad, h_kv, d - split_length), dtype=torch.bfloat16)
        y1 = x1.view(torch.int16).to(torch.int32)
    else:
        x1 = torch.randint(low=-(1 << (k1_bits - 1)), high=(1 << (k1_bits - 1)), size=(b, max_seqlen_pad, h_kv, d - split_length), dtype=torch.int32)
        y1 = x1

    compressed = torch.zeros(b, max_seqlen_pad, h_kv, compressed_head_size, dtype=torch.int32)
    for i in range(0, split_length):
        compressed[..., i // (32 // k0_bits)] |= (
            (x0[..., i] & ((1 << k0_bits) - 1)) << (i % (32 // k0_bits) * k0_bits))
    for i in range(split_length, d):
        compressed[..., split_length // (32 // k0_bits) + (i - split_length) // (32 // k1_bits)] |= (
            (y1[..., i - split_length] & ((1 << k1_bits) - 1)) << (i % (32 // k1_bits) * k1_bits))

    decompressed = torch.cat((x0.bfloat16(), x1.bfloat16()), dim=-1)

    return compressed, decompressed


@torch.inference_mode()
def test_flash_attention():
    print(b, s, h_q, h_kv, d, v_dim, split_length)

    s_q = 1
    q = torch.randn(b, s_q, h_q, d)
    compressed_k, k = create_k()
    compressed_blocked_k = compressed_k.view(-1, block_size, h_kv, compressed_head_size)
    blocked_k = k.view(-1, block_size, h_kv, d)
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)

    def blocked_quant_flash_attn():
        return flash_attn_with_blocked_kvcache(
            q, compressed_blocked_k, None, block_table, cache_seqlens, head_size_v=v_dim, causal=True,
            kvcache_quantization_dtypes=(k0_dtype, k1_dtype), kvcache_quantization_split_length=split_length,
        )

    def blocked_flash_attn():
        return flash_attn_with_blocked_kvcache(
            q, blocked_k, None, block_table, cache_seqlens, head_size_v=v_dim, causal=True,
        )

    out_blocked_quant_flash = blocked_quant_flash_attn()
    out_blocked_flash = blocked_flash_attn()
    assert torch.equal(out_blocked_quant_flash, out_blocked_flash)

    timer(blocked_quant_flash_attn)
    timer(blocked_flash_attn)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        blocked_quant_flash_attn()
        blocked_flash_attn()
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))
    # prof.export_chrome_trace("tests/flash_attn_trace.json")


if __name__ == "__main__":
    test_flash_attention()
