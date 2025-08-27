import math
import torch

import triton
import triton.language as tl

@triton.jit
def topk_index_to_mask_kernel(
    index_ptr,
    mask_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    num_blocks_q,
    num_blocks_k,
    index_stride,
    index_topk: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0)
    cu_seqlen_q = tl.load(cu_seqlens_q + batch_idx)
    cu_seqlen_k = tl.load(cu_seqlens_k + batch_idx)
    seqlen_q = tl.load(cu_seqlens_q + batch_idx + 1) - cu_seqlen_q
    seqlen_k = tl.load(cu_seqlens_k + batch_idx + 1) - cu_seqlen_k
    seq_idx = tl.program_id(axis=1)
    if seq_idx >= seqlen_q:
        return
    index_ptr += (cu_seqlen_q + seq_idx) * index_stride

    block_q_idx = seq_idx // 128
    index = tl.load(index_ptr + tl.arange(0, index_topk)) - cu_seqlen_k
    mask_ptr += batch_idx * num_blocks_q * num_blocks_k * 256 + block_q_idx * num_blocks_k * 256

    mask = index >= 0 and index < seqlen_k

    block_k_idx = index // 128
    in_block_thread_idx = seq_idx % 128 // 16 * 32 + seq_idx % 8 * 4 + index % 8 // 2

    add_index = block_k_idx * 256 + in_block_thread_idx
    add_val = 1 << (seq_idx % 16 // 8 * 32 + index % 128 // 8 * 2 + index % 2).to(tl.int64)

    tl.atomic_or(mask_ptr + add_index, add_val, mask=mask)


def next_power_of_2(n: int) -> int:
    if n < 1:
        return 1
    if (n & (n - 1)) == 0:
        return n
    return 1 << n.bit_length()


def topk_index_to_mask_triton(
    index: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    _, index_topk = index.shape
    # calc stride
    index_stride = index.stride(0)
    assert index.stride(1) == 1
    assert cu_seqlens_q.stride(0) == 1
    assert cu_seqlens_k.stride(0) == 1
    batch_size = cu_seqlens_q.shape[0] - 1
    num_blocks_q = math.ceil(max_seqlen_q / 128)
    num_blocks_k = math.ceil(max_seqlen_k / 128)
    mask = torch.zeros((batch_size, num_blocks_q, num_blocks_k, 256), dtype=torch.int64, device=index.device)

    topk_index_to_mask_kernel[(batch_size, max_seqlen_q)](
        index,
        mask,
        cu_seqlens_q,
        cu_seqlens_k,
        num_blocks_q,
        num_blocks_k,
        index_stride,
        index_topk,
    )
    return mask
