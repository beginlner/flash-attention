import math
import triton
import torch
from flash_attn.flash_attn_interface import get_kvcache_block_size, flash_attn_with_blocked_kvcache

b, s, h_q, h_kv, d = 132, 4096, 128, 1, 576
v_dim = 512
block_size = get_kvcache_block_size(d)
dtype = torch.bfloat16
device = torch.device("cuda:5")
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
    print(f"{name}: diff {diff}, amax {amax}")
    assert diff < 1e-5


def timer(func, name=""):
    t = triton.testing.do_bench(func)
    FLOPS = b * s * h_q * (d + v_dim) * 2
    bytes = b * s * h_kv * d * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {FLOPS / 10 ** 9 / t} tflops, {bytes / 10 ** 6 / t} GB/s")
    return t


def scaled_dot_product_attention(query, key, value) -> torch.Tensor:
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


@torch.inference_mode()
def test_flash_attention(b, s, h_q, h_kv, d):
    print(b, s, h_q, h_kv, d, v_dim)

    s_q = 1
    q = torch.randn(b, s_q, h_q, d)
    k = torch.randn(b, s, h_kv, d)
    v = k[..., :v_dim]
    full_k = k[:, :, :, None, :].expand(b, s, h_kv, h_q // h_kv, d).reshape(b, s, h_q, d)
    full_v = v[:, :, :, None, :].expand(b, s, h_kv, h_q // h_kv, v_dim).reshape(b, s, h_q, v_dim)
    blocked_k = k.view(-1, block_size, h_kv, d)
    blocked_v = v.view(-1, block_size, h_kv, v_dim)
    block_table = torch.arange(b * s // block_size, dtype=torch.int32).view(b, s // block_size)
    cache_seqlens = torch.full((b,), s, dtype=torch.int32)

    def blocked_flash_attn(): return flash_attn_with_blocked_kvcache(q, blocked_k, None, block_table, cache_seqlens, head_size_v=v_dim, causal=True)

    def torch_attn(): return scaled_dot_product_attention(q.transpose(1, 2), full_k.transpose(1, 2), full_v.transpose(1, 2)).transpose(1, 2)

    out_blocked_flash = blocked_flash_attn()
    out_torch_attn = torch_attn()
    assert_close(out_blocked_flash, out_torch_attn, "blocked_flash_attn")

    timer(blocked_flash_attn)
    timer(blocked_flash_attn)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        blocked_flash_attn()
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))
    # prof.export_chrome_trace("tests/flash_attn_trace.json")


if __name__ == "__main__":
    test_flash_attention(b, s, h_q, h_kv, d)
