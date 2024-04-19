import math
import triton
import torch
from flash_attn.flash_attn_interface import get_kvcache_block_size, flash_attn_with_blocked_kvcache

b, s, h_q, h_kv, d = 132, 4096, 128, 1, 576
v_dim = 512
k0_bits, k1_bits = 4, 8
k0_dtype, k1_dtype = "int4", "int8"
split_length = 512
assert (split_length * k0_bits + (d - split_length) * k1_bits) % 32 == 0
compressed_head_size = (split_length * k0_bits + (d - split_length) * k1_bits) // 32
block_size = get_kvcache_block_size(d)
dtype = torch.bfloat16
device = torch.device("cuda:7")
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
    amax = (x.double() - y.double()).abs().max()
    print(f"{name}: diff {diff}, amax {amax}")
    assert diff < 2e-5


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


def create_k():
    # 步骤 1: 创建一个大小为 576 的 tensor，前 512 个元素模拟 int4，后 64 个元素为 int8
    x_int4 = torch.randint(low=-8, high=8, size=(b, s, h_kv, split_length), dtype=torch.int32)
    x_int8 = torch.randint(low=-128, high=128, size=(b, s, h_kv, d - split_length), dtype=torch.int32)
    # x_int4 = torch.ones((b, s, h_kv, split_length), dtype=torch.int32) * 1
    # x_int8 = torch.ones((b, s, h_kv, d - split_length), dtype=torch.int32) * 1
    x = torch.cat((x_int4, x_int8), dim=-1)

    # 步骤 2: 将 x 压缩存放到 torch.int32 类型的 tensor 中
    # 创建一个足够大的 int32 tensor 来存储压缩后的数据
    compressed = torch.zeros(b, s, h_kv, compressed_head_size, dtype=torch.int32)

    # 填充 compressed tensor
    # 对于 int4，每个 int32 可以存储 8 个 int4 值
    for i in range(0, split_length):
        compressed[..., i // 8] |= (x[..., i] & 0xF) << (i % 8 * 4)

    # 对于 int8，每个 int32 可以存储 4 个 int8 值
    for i in range(split_length, d):
        compressed[..., split_length // 8 + (i - split_length) // 4] |= (x[..., i] & 0xFF) << (i % 4 * 8)

    # 步骤 3: 将压缩后的 tensor 转换回大小为 576 的 float 类型的 tensor
    decompressed = x.bfloat16()

    return compressed, decompressed


@torch.inference_mode()
def test_flash_attention():
    print(b, s, h_q, h_kv, d, v_dim)

    s_q = 1
    q = torch.randn(b, s_q, h_q, d)
    compressed_k, k = create_k()
    compressed_blocked_k = compressed_k.view(-1, block_size, h_kv, compressed_head_size)
    blocked_k = k.view(-1, block_size, h_kv, d)
    block_table = torch.arange(b * s // block_size, dtype=torch.int32).view(b, s // block_size)
    cache_seqlens = torch.full((b,), s, dtype=torch.int32)

    def blocked_quant_flash_attn():
        return flash_attn_with_blocked_kvcache(
            q, compressed_blocked_k, None, block_table, cache_seqlens, head_size_v=v_dim, causal=True,
            kvcahe_quantization_dtypes=(k0_dtype, k1_dtype), kvcahe_quantization_split_length=split_length,
        )

    def blocked_flash_attn():
        return flash_attn_with_blocked_kvcache(
            q, blocked_k, None, block_table, cache_seqlens, head_size_v=v_dim, causal=True,
        )

    out_blocked_quant_flash = blocked_quant_flash_attn()
    out_blocked_flash = blocked_flash_attn()
    assert_close(out_blocked_quant_flash, out_blocked_flash, "blocked_flash_attn")

    timer(blocked_quant_flash_attn)
    timer(blocked_quant_flash_attn)
    timer(blocked_flash_attn)
    timer(blocked_flash_attn)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        blocked_quant_flash_attn()
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))
    # prof.export_chrome_trace("tests/flash_attn_trace.json")


if __name__ == "__main__":
    test_flash_attention()
