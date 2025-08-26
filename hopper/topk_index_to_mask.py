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
        batch_size,
        num_blocks_q,
        num_blocks_k,
        PADDED_BATCH_SIZE: tl.constexpr,
        max_seqlen_k,
        index_stride,
        index_topk: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    index_ptr += pid * index_stride

    batch_range = tl.arange(0, PADDED_BATCH_SIZE)

    cu_seqlens_q_buf = tl.load(cu_seqlens_q + batch_range, mask=batch_range < batch_size, other=0)
    batch_idx = tl.max(tl.where(cu_seqlens_q_buf <= pid and batch_range < batch_size, batch_range, -1), axis=0)
    index_offset = tl.load(cu_seqlens_k + batch_idx)
    pid_offset = tl.load(cu_seqlens_q + batch_idx)

    block_q_idx = (pid - pid_offset) // 128
    index = tl.load(index_ptr + tl.arange(0, index_topk)) - index_offset
    mask_ptr += batch_idx * num_blocks_q * num_blocks_k * 256 + block_q_idx * num_blocks_k * 256

    mask = index >= 0 and index < max_seqlen_k

    block_k_idx = index // 128
    in_block_thread_idx = (pid - pid_offset) % 128 // 16 * 32 + (pid - pid_offset) % 8 * 4 + index % 8 // 2

    add_index = block_k_idx * 256 + in_block_thread_idx
    add_val = 1 << ((pid - pid_offset) % 16 // 8 * 32 + index % 128 // 8 * 2 + index % 2)

    tl.atomic_or(mask_ptr + add_index, add_val, mask=mask)


def next_power_of_2(n: int) -> int:
    if n < 1:
        return 1
    if (n & (n - 1)) == 0:
        return n
    return 1 << n.bit_length()


def topk_index_to_mask_triton(
        index,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
):
    total_q, index_topk = index.shape
    # calc stride
    index_stride = index.stride(0)
    assert index.stride(1) == 1
    assert cu_seqlens_q.stride(0) == 1
    assert cu_seqlens_k.stride(0) == 1
    batch_size = cu_seqlens_q.shape[0] - 1
    num_blocks_q = math.ceil(max_seqlen_q / 128)
    num_blocks_k = math.ceil(max_seqlen_k / 128)
    mask = torch.zeros((batch_size, num_blocks_q, num_blocks_k, 256), dtype=torch.int64, device=index.device)

    topk_index_to_mask_kernel[(total_q,)](
        index,
        mask,
        cu_seqlens_q,
        cu_seqlens_k,
        batch_size,
        num_blocks_q,
        num_blocks_k,
        next_power_of_2(batch_size),
        max_seqlen_k,
        index_stride,
        index_topk,
    )
    return mask
