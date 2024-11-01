/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#endif

#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "named_barrier.h"
#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Split, bool Is_local, typename Params>
__forceinline__ __device__ void store_zero(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;
    const BlockInfo</*Varlen=*/true> binfo(params, bidb);

    if (Split && !Is_local) { return; }
    if (tidx >= kNThreadsS) {
        return;
    }
    // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
    // Otherwise we might read OOB elements from gK and gV,
    // or get wrong results when we combine gOaccum from different blocks.
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
                                 + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                       + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    clear(tOrOaccum);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
#pragma unroll
    for (int m = 0; m < size<1>(tOgOaccum); ++m) {
        const int row = get<0>(tOcO(0, m, 0));
        if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
    }
}

template<typename Kernel_traits, bool Split, typename Params, typename AccO, typename Softmax>
__forceinline__ __device__ void store(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx,
                                      void* smem_, AccO acc_o, Softmax softmax) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;
    const BlockInfo</*Varlen=*/true> binfo(params, bidb);

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);
    // if (cute::thread0()) { print(lse); }

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
            !Split,
            typename Kernel_traits::SmemCopyAtomO,
            typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum is larger than sQ, so we need to syncthreads here
    if constexpr (Split || kHeadDimV > kHeadDim) { __syncthreads(); }

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
                                 + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                       + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    if (tidx >= kNThreadsS) {
        return;
    }

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // ((MMA=4, X), MMA_M, MMA_K=1)
    Tensor taccOcO_row = taccOcO(make_coord(0, _, 0), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Split, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx) {
    static_assert(!Has_alibi);
    static_assert(Kernel_traits::Share_KV);
    static_assert(Kernel_traits::Blocked_KV);
    static_assert(!Kernel_traits::QKCooperative);
    static_assert(std::is_same_v<typename Kernel_traits::KV_type1, typename Kernel_traits::Element>);

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarpsS = Kernel_traits::kNWarpsS;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    static_assert(kNThreads == 256 and kNThreadsS == 128);
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;
    const BlockInfo</*Varlen=*/true> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    constexpr int n_blocks_per_split = PARTITION_SIZE / kBlockN;
    const int n_block_min = !Is_local
                            ? n_split_idx * n_blocks_per_split
                            : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        store_zero<Kernel_traits, Split, Is_local>(params, bidb, bidh, m_block, n_split_idx);
        return;
    }
    int n_block = n_block_max - 1;

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});

    Tensor sP = make_tensor(sK.data() + 2 * size(sK), typename Kernel_traits::SmemLayoutP{});
    Tensor tPsP = sP(_, tidx % kNThreadsS, _, _);
    Tensor sScale_o = make_tensor(recast_ptr<float>(sP.data() + size(sP)), typename Kernel_traits::SmemLayoutRow{});
    Tensor tScale_osScale_o = sScale_o(_, tidx % kNThreadsS);
    Tensor sRow_max = make_tensor(sScale_o.data() + size(sScale_o), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_maxsRow_max = sRow_max(_, tidx % kNThreadsS);
    Tensor sRow_sum = make_tensor(sRow_max.data() + size(sRow_max), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_sumsRow_sum = sRow_sum(_, tidx % kNThreadsS);

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrO = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
    Tensor acc_o = make_tensor(tOrO.data(), flash::convert_gmma_to_mma_tensor(tOrO.layout()));  // (4, MMA_M, X)
    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    int warp_group_idx = cutlass::canonical_warp_group_idx();
    if (warp_group_idx == 0) {
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);
        Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
        Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

        const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
        flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

        if (n_block % 2 == 1) {
            // Double buffer for sK
            constexpr int sK_offset = size(sK);
            tSrK.data() = tSrK.data() + sK_offset / 8;
            tOrVt.data() = tOrVt.data() + sK_offset / 8;
        }

        // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
        // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
        // We will have at least 1 "masking" iteration.
        // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
        // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
        constexpr int n_masking_steps = (!Is_causal && !Is_local) ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
#pragma unroll 1
        for (int masking_step = n_masking_steps; n_block >= n_block_min; --masking_step, --n_block) {
            __syncthreads();

            Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
            Tensor acc_s = make_tensor(tSrS.data(), flash::convert_gmma_to_mma_tensor(tSrS.layout()));  // (4, MMA_M, X)
            flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma, tSrQ, tSrK, tSrS);

            const bool is_masking_step = masking_step > 0;
            const bool is_first_masking_step = masking_step == n_masking_steps;

            if (is_masking_step) {
                mask.template apply_mask<Is_causal, /*Is_even_MN*/false>(
                        acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarpsS * 16
                );
            } else {
                mask.template apply_mask</*Causal_mask=*/false, /*Is_even_MN*/true>(
                        acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarpsS * 16
                );
            }

            // We have key_padding_mask so we'll need to Check_inf
            Tensor scale_o = is_first_masking_step
                             ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/true, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2)
                             : is_masking_step ?
                               softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2)
                                               : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2);

            Tensor rP = make_tensor<Element>(acc_s.layout());
            cute::copy(flash::convert_type<Element>(acc_s), rP);
            cute::copy(rP, tPsP);
            cute::copy(scale_o, tScale_osScale_o);

            cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::SReady));

            flash::rescale_o(acc_o, scale_o);

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);

            // Double buffer for sK
            const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
            tSrK.data() = tSrK.data() + sK_offset / 8;
            tOrVt.data() = tOrVt.data() + sK_offset / 8;
        }

        cute::copy(softmax.row_max, tRow_maxsRow_max);
        cute::copy(softmax.row_sum, tRow_sumsRow_sum);
        cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::SoftmaxReady));
    } else {
        assert(params.cache_batch_idx == nullptr);
        const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        {
            // Load block_table from global memory to shared memory
            int *block_table_shared = reinterpret_cast<int *>(sScale_o.data().get().get() + size(sScale_o));
            int n_page_min = n_block_min * kBlockN / params.page_block_size;
            int n_page_max = (n_block_max - 1) * kBlockN / params.page_block_size;
            for (int i = tidx - kNThreadsS; i <= n_page_max - n_page_min; i += kNThreads - kNThreadsS) {
                SM80_CP_ASYNC_CACHEALWAYS<int>::copy(block_table[i + n_page_min], block_table_shared[i]);
            }
            cp_async_fence();
            block_table = block_table_shared - n_page_min;
        }
        // We move K and V to the last block.
        const int block_table_idx = (n_block_max - 1) * kBlockN / params.page_block_size;
        const int block_table_offset = (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;

        const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
        Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopyK gmem_tiled_copy_Q;
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx - kNThreadsS);
        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                                                              binfo.actual_seqlen_q - m_block * kBlockM);
        cp_async_fence();

        flash::cp_async_wait<1>();  // Wait for block_table ready.
        cutlass::arch::NamedBarrier::sync(kNThreads - kNThreadsS, static_cast<int>(NamedBarriers::BlockTableReady));

        const index_t row_offset_k = block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.k_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopyK gmem_tiled_copy_K;
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx - kNThreadsS);
        Tensor tKgK = gmem_thr_copy_K.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
        Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_K.partition_S(cK);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));

        using KV_type0 = typename Kernel_traits::KV_type0;
        using KV_type1 = typename Kernel_traits::KV_type1;
        constexpr int KV_type0_bits = sizeof_bits<KV_type0>::value;
        constexpr int KV_type1_bits = sizeof_bits<KV_type1>::value;
        assert(params.k_row_stride * 32 % KV_type0_bits == 0);
        const int64_t kquant0_row_stride = params.k_row_stride * 32 / KV_type0_bits;
        Tensor gKQuant0 = make_tensor(make_gmem_ptr(recast_ptr<KV_type0>(recast_ptr<int32_t>(params.k_ptr) + row_offset_k)),
                                      Shape<Int<kBlockN>, Int<Kernel_traits::SplitLength>>{},
                                      make_stride(kquant0_row_stride, _1{}));
        static_assert(Kernel_traits::SplitLength * KV_type0_bits % 32 == 0);
        assert(params.k_row_stride * 32 % KV_type1_bits == 0);
        const int64_t kquant1_row_stride = params.k_row_stride * 32 / KV_type1_bits;
        Tensor gKQuant1 = make_tensor(make_gmem_ptr(recast_ptr<KV_type1>(recast_ptr<int32_t>(params.k_ptr) + row_offset_k + Kernel_traits::SplitLength * KV_type0_bits / 32)),
                                      Shape<Int<kBlockN>, Int<kHeadDim - Kernel_traits::SplitLength>>{},
                                      make_stride(kquant1_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopyKQuant0 gmem_tiled_copy_KQuant0;
        auto gmem_thr_copy_KQuant0 = gmem_tiled_copy_KQuant0.get_thread_slice(tidx - kNThreadsS);
        Tensor tKQuant0gKQuant0 = gmem_thr_copy_KQuant0.partition_S(gKQuant0);
        Tensor tKQuant0rKQuant0 = make_fragment_like(tKQuant0gKQuant0);
        Tensor tKQuant0rKQuant0_high = make_tensor<Element>(tKQuant0rKQuant0.layout());
        typename Kernel_traits::GmemTiledCopyKQuant1 gmem_tiled_copy_KQuant1;
        auto gmem_thr_copy_KQuant1 = gmem_tiled_copy_KQuant1.get_thread_slice(tidx - kNThreadsS);
        Tensor tKQuant1gKQuant1 = gmem_thr_copy_KQuant1.partition_S(gKQuant1);
        typename Kernel_traits::SmemTiledCopyK smem_tiled_copy_K;
        Tensor tKQuant1rKQuant1 = make_fragment_like(tKQuant1gKQuant1);
        Tensor tKQuant1rKQuant1_high = make_tensor<Element>(tKQuant1rKQuant1.layout());
        auto LDG_K = [&] (const int n) {
#pragma unroll
            for (int k = 0; k < size<2>(tKQuant0gKQuant0); ++k) {
                copy(gmem_tiled_copy_KQuant0, tKQuant0gKQuant0(_, n, k), tKQuant0rKQuant0(_, n, k));
            }
            for (int k = 0; k < size<2>(tKQuant1gKQuant1); ++k) {
                copy(gmem_tiled_copy_KQuant1, tKQuant1gKQuant1(_, n, k), tKsK(_, n, size<2>(tKQuant0gKQuant0) + k));
            }
        };
        auto Cast_K = [&] (const int n) {
#pragma unroll
            for (int k = 0; k < size<2>(tKQuant0gKQuant0); ++k) {
                convert_type_out(tKQuant0rKQuant0(_, n, k), tKQuant0rKQuant0_high(_, n, k));
            }
        };
        auto STS_K = [&] (const int n) {
#pragma unroll
            for (int k = 0; k < size<2>(tKQuant0gKQuant0); ++k) {
                copy(smem_tiled_copy_K, tKQuant0rKQuant0_high(_, n, k), tKsK(_, n, k));
            }
        };
        auto LoadK = [&](int n_block) {
            if (n_block <= n_block_min) { return; }
            // Advance gK
            const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
            const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
            const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
            const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
            const index_t offset = (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;

            tKgK.data() = tKgK.data() + offset;
            tKQuant0gKQuant0.data() = recast_ptr<KV_type0>(recast_ptr<int32_t>(tKQuant0gKQuant0.data()) + offset);
            tKQuant1gKQuant1.data() = recast_ptr<KV_type1>(recast_ptr<int32_t>(tKQuant1gKQuant1.data()) + offset);

            // Double buffer for sK
            const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
            tKsK.data() = tKsK.data() + sK_offset;

            if (Kernel_traits::SplitLength == 0) {
                flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK);
            } else {
#pragma unroll
                for (int n = 0; n < size<1>(tKsK); ++n) {
                    LDG_K(n);
                }
            }
            cute::cp_async_fence();
        };

        if (n_block % 2 == 1) {
            // Double buffer for sK
            constexpr int sK_offset = size(sK);
            tKsK.data() = tKsK.data() + sK_offset;
            tOrVt.data() = tOrVt.data() + sK_offset / 8;
        }

        // We need to clear the sK smem tiles because K is V.
        if (Kernel_traits::SplitLength == 0) {
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK,
                                                                                         binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
#pragma unroll
            for (int n = 0; n < size<1>(tKsK); ++n) {
                if (get<0>(tKcK(0, n, 0)) < binfo.actual_seqlen_k - n_block * kBlockN) {
                    LDG_K(n);
                    Cast_K(n);
                    STS_K(n);
                } else {
                    clear(tKsK(_, n, _));
                }
            }
        }
        cute::cp_async_fence();

#pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
            flash::cp_async_wait<0>();
            __syncthreads();

            LoadK(n_block);

            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::SReady));

            if (Kernel_traits::SplitLength > 0 && n_block > n_block_min) {
#pragma unroll
                for (int n = 0; n < size<1>(tKsK); ++n) {
                    Cast_K(n);
                }
#pragma unroll
                for (int n = 0; n < size<1>(tKsK); ++n) {
                    STS_K(n);
                }
            }

            typename Kernel_traits::TiledMma tiled_mma;
            auto acc_s_layout = flash::convert_gmma_to_mma_tensor(partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}).layout());
            Tensor rP = make_tensor<Element>(acc_s_layout);
            Tensor scale_o = make_tensor<float>(Shape<_2>{});
            cute::copy(tScale_osScale_o, scale_o);
            cute::copy(tPsP, rP);

            flash::rescale_o(acc_o, scale_o);

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);

            // Double buffer for sK
            const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
            tOrVt.data() = tOrVt.data() + sK_offset / 8;
        }

        cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::SoftmaxReady));
        cute::copy(tRow_maxsRow_max, softmax.row_max);
        cute::copy(tRow_sumsRow_sum, softmax.row_sum);
    }

    store<Kernel_traits, Split>(params, bidb, bidh, m_block, n_split_idx, (void*)smem_, acc_o, softmax);
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Split, typename Params>
__forceinline__ __device__ void compute_attn_splitkv_mla(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.z / params.h;
    // The block index for the head.
    const int bidh = blockIdx.z - bidb * params.h;
    const int n_split_idx = blockIdx.y;
    const BlockInfo</*Varlen=*/true> binfo(params, bidb);
    const bool NoSplit = binfo.actual_seqlen_k <= PARTITION_SIZE;
    if (Split == NoSplit) return;
    flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal, Is_local, Has_alibi, Split>(params, bidb, bidh, m_block, n_split_idx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int kBlockM, int Log_max_splits, typename Params>
__forceinline__ __device__ void combine_attn_seqk_parallel_mla(const Params &params) {
    using OutElement = typename Kernel_traits::OutElement;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = 128;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    // TODO: Assume Varlen
    const BlockInfo</*Varlen=*/true> binfo(params, bidx * kBlockM / (params.h * params.seqlen_q));
    const int actual_num_splits = std::min(params.num_splits, cute::ceil_div(binfo.actual_seqlen_k, PARTITION_SIZE));
    if (actual_num_splits == 1) return;
    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(params.b * params.h * params.seqlen_q, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then tranpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
#pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < actual_num_splits && col < params.b * params.h * params.seqlen_q - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
    }
    // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
    __syncthreads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
    // static_assert(kThreadsPerSplit <= 32);
    static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
#pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }

    // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
#pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = expf(lse_accum(0) - lse_max);
#pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM && tidx / kRowsPerLoadTranspose < params.b * params.h * params.seqlen_q - bidx * kBlockM) { gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum; }
    // Store the scales exp(lse - lse_logsum) in shared memory.
#pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < actual_num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
    }
    __syncthreads();

    const index_t row_offset_oaccum = bidx * kBlockM * params.d_v;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 Stride<Int<kHeadDimV>, _1>{});
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
            make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                            GmemLayoutAtomOaccum{},
                            Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    // Predicates
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    // Repeat the partitioning with identity layouts
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Load Oaccum in then scale and accumulate to O
    for (int split = 0; split < actual_num_splits; ++split) {
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(
                gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
        );
#pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            int row = get<0>(tOcOaccum(0, m, 0));
            ElementAccum lse_scale = sLSE[split][row];
#pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
#pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                }
            }
            // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]); print(tOrOaccum); }
        }
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_v;
    }
    // if (cute::thread0()) { print_tensor(tOrO); }

    Tensor rO = flash::convert_type<OutElement>(tOrO);
    // Write to gO
#pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        if (idx < params.b * params.h * params.seqlen_q) {
            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
            // The index to the rows of Q
            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            auto o_ptr = reinterpret_cast<OutElement *>(params.o_ptr) + batch_idx * params.o_batch_stride
                         + head_idx * params.o_head_stride + row * params.o_row_stride;
#pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                const int col = get<1>(tOcOaccum(0, m, k));
                Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
                // TODO: Should check if this is using vectorized store, but it seems pretty fast
                copy(rO(_, m, k), gO);
                // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
                // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
            }
        }
    }
}

} // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Split>
__global__ void __launch_bounds__(256, 1, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const Flash_fwd_params params) {
    flash::compute_attn_splitkv_mla<Kernel_traits, Is_causal, Is_local, Has_alibi, Split>(params);
}

template<typename Kernel_traits, int kBlockM, int Log_max_splits>
__global__ void __launch_bounds__(256, 1, 1)
flash_fwd_splitkv_mla_combine_kernel(__grid_constant__ const Flash_fwd_params params) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel_mla<Kernel_traits, kBlockM, Log_max_splits>(params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
void run_flash_splitkv_fwd_mla(Flash_fwd_params &params, cudaStream_t stream) {
    TORCH_CHECK(!params.unpadded_lse);
    if (extra_stream_ptr == nullptr) extra_stream_ptr = std::make_shared<cudaStream_t>(at::cuda::getStreamFromPool(true).stream());
    auto stream1 = *extra_stream_ptr;
    size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                if (params.num_splits > 1) {
                    // Launch the split kernel in another stream.
                    wait_stream(stream1, stream);
                    auto split_kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, true>;
                    smem_size = std::max(smem_size, size(typename Kernel_traits::SmemLayoutO{}) * sizeof(typename Kernel_traits::ElementAccum));
                    C10_CUDA_CHECK(cudaFuncSetAttribute(split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    split_kernel<<<dim3(num_m_block, params.num_splits, params.b * params.h), Kernel_traits::kNThreads, smem_size, stream1>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
                auto kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, false>;
                C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                kernel<<<dim3(num_m_block, 1, params.b * params.h), Kernel_traits::kNThreads, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDimV % 128 == 0 ? 4 : (Kernel_traits::kHeadDimV % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        NUM_SPLITS_SWITCH(params.num_splits, kLogMaxSplits, [&] {
            flash_fwd_splitkv_mla_combine_kernel<Kernel_traits, kBlockM, kLogMaxSplits><<<grid_combine, 128, 0, stream1>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
        wait_stream(stream, stream1);
    }
}
