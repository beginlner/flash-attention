import random

import torch
import triton

from flash_attn.flash_attn_interface import flash_attn_with_blocked_kvcache


def scaled_dot_product_attention(query, key, value, is_causal=False, softmax_scale=None):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** (-0.5)
    attn_weight = query @ key.transpose(-2, -1) * softmax_scale
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    assert cos_diff < 1e-5, f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}"


@torch.inference_mode()
def test_flash_decode(b, s_q, mean_sk, h_q, h_kv, d, dv, block_size, causal, varlen):
    print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {block_size=}, {causal=}, {varlen=}")

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 128) * 128
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}, {cache_seqlens.tolist()=}")

    q = torch.randn(b, s_q, h_q, d + 128)[..., :d]
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), 9, block_size, h_kv, d)[:, 1]
    blocked_v = torch.randn(block_table.numel(), 9, block_size, h_kv, dv)[:, 1]
    softmax_scale = (192 + 100) ** (-0.5)

    def flash_decode():
        return flash_attn_with_blocked_kvcache(q, blocked_k, blocked_v, block_table, cache_seqlens, softmax_scale=softmax_scale, causal=causal)

    def ref_decode():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.reshape(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.reshape(-1, h_kv, dv)[begin:end].transpose(0, 1),
                is_causal=causal,
                softmax_scale=softmax_scale,
            )
            out[i] = O.transpose(0, 1)
        return out

    out_flash = flash_decode()
    out_torch = ref_decode()
    cal_diff(out_flash, out_torch, "out")

    for _ in range(5):
        out = flash_decode()
        assert torch.equal(out, out_flash), "out deterministic check failed!"

    t = triton.testing.do_bench(flash_decode)
    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    print(f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s")


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    block_size = 128
    h_kv = 8
    d, dv = 128, 128
    causal = True

    for b in [128]:
        for s in [4096, 8192]:
            for h_q in [32, 64, 128]:
                for s_q in [1, 2]:
                    for varlen in [False, True]:
                        test_flash_decode(b, s_q, s, h_q, h_kv, d, dv, block_size, causal, varlen)
