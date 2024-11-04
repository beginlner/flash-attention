#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_mla_params {
    using index_t = int64_t;

    int b, seqlen_q, d, d_v;
    int h, h_h_k_ratio;
    float scale_softmax, scale_softmax_log2;
    int * __restrict__ cu_seqlens_k;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void * __restrict__ o_ptr;
    void * __restrict__ softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;

    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    int * __restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int * __restrict__ num_splits_ptr;

    void * __restrict__ softmax_lseaccum_ptr;
    void * __restrict__ oaccum_ptr;

    int kvcache_quantization_type;
    int kvcache_quantization_split_length;
};

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim> void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream);
