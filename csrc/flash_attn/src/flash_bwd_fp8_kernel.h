/***************************************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits_fp8.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"

#include "alibi.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<class... Args, class TiledGMMA>
__forceinline__ __device__ auto custom_tiled_copy(Copy_Atom<Args...> const& copy_atom, TiledGMMA const& tiled_gmma) {
    auto tv_layout = tiled_gmma.get_layoutC_TV();
    if constexpr (rank<0>(tv_layout) == _4{}) {
        auto new_tv_stride = make_stride(make_stride(stride<0, 0>(tv_layout), _2{}, stride<0, 2>(tv_layout), stride<0, 3>(tv_layout)),
                                         make_stride(make_stride(_1{}, stride<1, 0, 0>(tv_layout), stride<1, 0, 2>(tv_layout)), stride<1, 1>(tv_layout)));
        auto new_tv_layout = make_layout(shape(tv_layout), new_tv_stride);
        return make_tiled_copy_impl(copy_atom, new_tv_layout, make_shape(tile_size<0>(tiled_gmma),tile_size<1>(tiled_gmma)));
    } else {
        static_assert(rank<0>(tv_layout) == _3{});
        auto new_tv_stride = make_stride(make_stride(stride<0, 0>(tv_layout), _2{}, stride<0, 2>(tv_layout)),
                                         make_stride(make_stride(_1{}, stride<1, 0, 0>(tv_layout), stride<1, 0, 2>(tv_layout)), stride<1, 1>(tv_layout)));
        auto new_tv_layout = make_layout(shape(tv_layout), new_tv_stride);
        return make_tiled_copy_impl(copy_atom, new_tv_layout, make_shape(tile_size<0>(tiled_gmma),tile_size<1>(tiled_gmma)));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
__forceinline__ __device__ void compute_dq_dk_dv_1colblock_fp8(const Params &params, const int bidb, const int bidh, const int n_block) {
    static_assert(!Is_dropout);
    static_assert(!Is_first);
    static_assert(!Is_last);
    static_assert(Seq_parallel);

    using Element = typename Kernel_traits::Element;
    using GradElement = typename Kernel_traits::GradElement;
    using OutElement = typename Kernel_traits::OutElement;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + ((m_block_max - 1) * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded
        // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q
        + (m_block_max - 1) * kBlockM;
    const index_t row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded
        + (m_block_max - 1) * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<GradElement *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sQt = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutQt{});
    // Double buffer for sQ
    Tensor sK = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ) * 2, typename Kernel_traits::SmemLayoutK{});
    Tensor sKt = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKt{});
    Tensor sV = make_tensor(sKt.data() + size(sKt), typename Kernel_traits::SmemLayoutV{});
    Tensor sdO = make_tensor(recast_ptr<GradElement>(sV.data() + size(sV)), typename Kernel_traits::SmemLayoutdO{});
    Tensor sdOt = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutdOt{});
    Tensor sPt = make_tensor(recast_ptr<Element>(sdOt.data() + size(sdOt)), typename Kernel_traits::SmemLayoutPt{});
    Tensor sdS = make_tensor(recast_ptr<GradElement>(sPt.data() + size(sPt)), typename Kernel_traits::SmemLayoutdS{});
    Tensor sdSt = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutdSt{});

    // View of shared memory
    Tensor sPt_t = make_tensor(sPt.data(), typename Kernel_traits::SmemLayoutPt_t{});
    Tensor sdSt_t = make_tensor(sdSt.data(), typename Kernel_traits::SmemLayoutdSt_t{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = typename Kernel_traits::GmemTiledCopyQKV;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);

    typename Kernel_traits::TiledGMmaS tiled_gmma_s;
    auto thr_gmma_s = tiled_gmma_s.get_thread_slice(tidx);
    Tensor tSrQ = thr_gmma_s.partition_fragment_A(sQ);
    Tensor tSrK = thr_gmma_s.partition_fragment_B(sK);

    typename Kernel_traits::TiledGMmadP tiled_gmma_dp;
    auto thr_gmma_dp = tiled_gmma_dp.get_thread_slice(tidx);
    Tensor tdPrdO = thr_gmma_dp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_gmma_dp.partition_fragment_B(sV);

    typename Kernel_traits::TiledGMmadK tiled_gmma_dk;
    auto thr_gmma_dk = tiled_gmma_dk.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_gmma_dk.partition_fragment_A(sdSt);
    Tensor tdKrQt = thr_gmma_dk.partition_fragment_B(sQt);

    typename Kernel_traits::TiledGMmadV tiled_gmma_dv;
    auto thr_gmma_dv = tiled_gmma_dv.get_thread_slice(tidx);
    Tensor tdVrPt = thr_gmma_dv.partition_fragment_A(sPt);
    Tensor tdVrdOt = thr_gmma_dv.partition_fragment_B(sdOt);

    typename Kernel_traits::TiledGMmadQ tiled_gmma_dq;
    auto thr_gmma_dq = tiled_gmma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_gmma_dq.partition_fragment_A(sdS);
    Tensor tdQrKt = thr_gmma_dq.partition_fragment_B(sKt);

    Tensor tdKrdK_gmma = partition_fragment_C(tiled_gmma_dk, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dk = make_tensor(tdKrdK_gmma.data(), flash::convert_gmma_to_mma_tensor(tdKrdK_gmma.layout()));  // MMA, MMA_N, MMA_K
    Tensor tdVrdV_gmma = partition_fragment_C(tiled_gmma_dv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dv = make_tensor(tdVrdV_gmma.data(), flash::convert_gmma_to_mma_tensor(tdVrdV_gmma.layout()));  // MMA, MMA_N, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_PdS = custom_tiled_copy(Copy_Atom<SM90_U16x8_STSM_T, Element>{}, tiled_gmma_s);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsPt_t = smem_thr_copy_PdS.partition_D(sPt_t);
    Tensor tdSsdSt_t = smem_thr_copy_PdS.partition_D(sdSt_t);

    auto reg2reg = ReorgCFp8toAFp8SharedTransposed();

    Tensor tdSsdS = thr_gmma_dp.partition_C(sdS);

    // Transpose Q, K and dO in shared memory
    auto smem_tiled_copy_X = typename Kernel_traits::SmemTiledCopyX{};
    auto smem_thr_copy_X = smem_tiled_copy_X.get_thread_slice(tidx);
    Tensor tcQsQ = smem_thr_copy_X.partition_S(sQ);
    Tensor tcKsK = smem_thr_copy_X.partition_S(sK);
    Tensor tcdOsdO = smem_thr_copy_X.partition_S(sdO);

    auto smem_tiled_copy_Xt = typename Kernel_traits::SmemTiledCopyXt{};
    auto smem_thr_copy_Xt = smem_tiled_copy_Xt.get_thread_slice(tidx);
    Tensor tcQsQt = smem_thr_copy_Xt.partition_D(sQt);
    Tensor tcKsKt = smem_thr_copy_Xt.partition_D(sKt);
    Tensor tcdOsdOt = smem_thr_copy_Xt.partition_D(sdOt);

    Tensor tcQrQ = make_tensor<Element>(make_shape(Shape<_8, _1>{}, shape<1>(tcQsQ), shape<2>(tcQsQ)));
    Tensor tcKrK = make_tensor<Element>(make_shape(Shape<_8, _1>{}, shape<1>(tcKsK), shape<2>(tcKsK)));
    Tensor tcdOrdO = make_tensor<GradElement>(make_shape(Shape<_8, _1>{}, shape<1>(tcdOsdO), shape<2>(tcdOsdO)));

    Tensor tcQrQt = make_tensor(tcQrQ.data(), make_layout(layout<0>(tcQrQ), layout<2>(tcQrQ), layout<1>(tcQrQ)));
    Tensor tcKrKt = make_tensor(tcKrK.data(), make_layout(layout<0>(tcKrK), layout<2>(tcKrK), layout<1>(tcKrK)));
    Tensor tcdOrdOt = make_tensor(tcdOrdO.data(), make_layout(layout<0>(tcdOrdO), layout<2>(tcdOrdO), layout<1>(tcdOrdO)));

    // FP8 scales
    float Descale_DO = reinterpret_cast<float *>(params.descale_do_ptr)[0];
    float Descale_Q = reinterpret_cast<float *>(params.descale_q_ptr)[0];
    float Descale_K = reinterpret_cast<float *>(params.descale_k_ptr)[0];
    float Descale_V = reinterpret_cast<float *>(params.descale_v_ptr)[0];
    float scale_softmax = params.scale_softmax * (Descale_Q * Descale_K);
    float scale_softmax_log2 = params.scale_softmax_log2 * (Descale_Q * Descale_K);
    float Scale_S = std::is_same_v<Element, cutlass::float_e4m3_t> ? 448.0f : 57344.0f;
    float Descale_S = 1.0f / Scale_S;
    float Descale_DO_V = Descale_DO * Descale_V;
    float Descale_DO_S = Descale_DO * Descale_S;

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We'll advance gdQaccum before the 1st read/write.
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.h * params.d_rounded;

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
          + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
          + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<OutElement *>(params.dk_ptr) + row_offset_dk),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<OutElement *>(params.dv_ptr) + row_offset_dv),
                                 Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                 make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<OutElement>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<OutElement>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }

    if (Double_buffer && m_block % 2 == 1) {
        // Double buffer for sQ
        const int sQ_offset = size(sQ) * 2;
        tQsQ.data() = tQsQ.data() + sQ_offset;
        tSrQ.data() = tSrQ.data() + sQ_offset / 16;
        tdKrQt.data() = tdKrQt.data() + sQ_offset / 16;
        tcQsQ.data() = tcQsQ.data() + sQ_offset;
        tcQsQt.data() = tcQsQt.data() + sQ_offset;
    }

    if (params.deterministic) { __syncthreads(); }

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    // Clear the smem tiles to account for predicated off loads
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((2, 2), MMA_N, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
    // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
    // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
    // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

    // Tensor tKrK = make_fragment_like(tKsK);
    // // cute::copy(gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKrK);
    // cute::copy(gmem_tiled_copy_QKV, tKgK, tKrK);
    // // if (cute::thread(1, 0)) { print(tKrK); }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::cp_async_fence();

    clear(acc_dv);
    clear(acc_dk);

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    for (; m_block >= m_block_min; --m_block) {
        Tensor tSrS_gmma = partition_fragment_C(tiled_gmma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor acc_s = make_tensor(tSrS_gmma.data(), flash::convert_gmma_to_mma_tensor(tSrS_gmma.layout()));  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

        flash::gemm(tiled_gmma_s, tSrQ, tSrK, tSrS_gmma);

        if (m_block == m_block_max - 1) {
            cute::copy(smem_tiled_copy_X, tcKsK, tcKrK);
            permute_fp8(tcKrK);
            cute::copy(smem_tiled_copy_Xt, tcKrKt, tcKsKt);
        }
        cute::copy(smem_tiled_copy_X, tcQsQ, tcQrQ);
        permute_fp8(tcQrQ);
        cute::copy(smem_tiled_copy_Xt, tcQrQt, tcQsQt);
        cute::copy(smem_tiled_copy_X, tcdOsdO, tcdOrdO);
        permute_fp8(tcdOrdO);
        cute::copy(smem_tiled_copy_Xt, tcdOrdOt, tcdOsdOt);

        // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        // if (cute::thread(32, 0)) { print(scores); }

        if (Has_alibi) {
            alibi.apply_alibi(scores, n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                              m_block * kBlockM + get<0>(taccScS_row(0)), AtomLayoutMS * 16);
        }

        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp8, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.
        if (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                flash::apply_mask(scores, binfo.actual_seqlen_k,
                                  n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16);
            }
        } else if (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                flash::apply_mask_causal(scores, n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                                         binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                         binfo.actual_seqlen_q,
                                         // binfo.actual_seqlen_k, m_block * kBlockM + (tidx / 32) % AtomLayoutMS * 16 + (tidx % 32) / 4,
                                         AtomLayoutMS * 16);
            }
        } else if (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                flash::apply_mask_local(scores, n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }

        }

        // Compute the exponential value.
        Tensor scores_scaled = flash::scale_apply_exp2</*scale_max=*/false, /*inplace_Scale_S*/false>(scores, lse, scale_softmax_log2, Scale_S);
        // Convert scores from fp32 to fp8
        Tensor tSrS_gmma_scaled = make_tensor(scores_scaled.data(), tSrS_gmma.layout());
        Tensor tSrS_gmma_fp8 = flash::convert_type<Element>(tSrS_gmma_scaled);
        reg2reg(tSrS_gmma_fp8);
        Tensor tPrP = smem_thr_copy_PdS.retile_S(tSrS_gmma_fp8);
        cute::copy(smem_tiled_copy_PdS, tPrP, tPsPt_t);

        Tensor tdPrdP_gmma = partition_fragment_C(tiled_gmma_dp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor acc_dp = make_tensor(tdPrdP_gmma.data(), flash::convert_gmma_to_mma_tensor(tdPrdP_gmma.layout()));  // (MMA=4, MMA_N, MMA_N)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);

        flash::gemm(tiled_gmma_dp, tdPrdO, tdPrV, tdPrdP_gmma);

        // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
        Tensor dS = make_tensor(acc_dp.data(), flash::convert_layout_acc_rowcol(acc_dp.layout()));
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) * Descale_DO_V - dP_sum(mi));
            }
        }

        Tensor tdQrdQ_gmma = partition_fragment_C(tiled_gmma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor acc_dq = make_tensor(tdQrdQ_gmma.data(), flash::convert_gmma_to_mma_tensor(tdQrdQ_gmma.layout()));  // MMA, MMA_N, MMA_K
        tdQgdQaccum.data() = tdQgdQaccum.data() + (-int(kBlockM * params.h * params.d_rounded));
        clear(acc_dq);

        if (Double_buffer && m_block > m_block_min) {
            // Double buffer for sQ
            const int sQ_offset = m_block % 2 == 0 ? size(sQ) * 2 : -size(sQ) * 2;
            tQsQ.data() = tQsQ.data() + sQ_offset;
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            flash::cp_async_fence();
        }

        // Convert dS from fp32 to fp8
        Tensor tdPrdP_gmma_fp8 = flash::convert_type<GradElement>(tdPrdP_gmma);
        cute::copy(tdPrdP_gmma_fp8, tdSsdS);
        __syncthreads();

        flash::gemm(tiled_gmma_dv, tdVrPt, tdVrdOt, tdVrdV_gmma);

        reg2reg(tdPrdP_gmma_fp8);
        Tensor tdSrdS = smem_thr_copy_PdS.retile_S(tdPrdP_gmma_fp8);
        cute::copy(smem_tiled_copy_PdS, tdSrdS, tdSsdSt_t);
        __syncthreads(); // Need syncthreads since we're writing to the same sdO location

        if (m_block > m_block_min) {
            // Advance gdO
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
            flash::cp_async_fence();
        }

        flash::gemm(tiled_gmma_dq, tdQrdS, tdQrKt, tdQrdQ_gmma);

        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }

        CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
        #pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }

        flash::gemm(tiled_gmma_dk, tdKrdSt, tdKrQt, tdKrdK_gmma);
        if (!Double_buffer && m_block > m_block_min) {
            __syncthreads();
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            flash::cp_async_fence();
        }

        if (Double_buffer) {
            // Double buffer for sQ
            const int sQ_offset = m_block % 2 == 0 ? size(sQ) * 2 : -size(sQ) * 2;
            tSrQ.data() = tSrQ.data() + sQ_offset / 16;
            tdKrQt.data() = tdKrQt.data() + sQ_offset / 16;
            tcQsQ.data() = tcQsQ.data() + sQ_offset;
            tcQsQt.data() = tcQsQt.data() + sQ_offset;
        }
    }

    // Epilogue

    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= Descale_Q * params.scale_softmax_rp_dropout; }
    #pragma unroll
    for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= Descale_DO_S; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<OutElement>(acc_dk);
    Tensor rdV = flash::convert_type<OutElement>(acc_dv);

    Tensor sdK = make_tensor(make_smem_ptr(recast_ptr<OutElement>(smem_)), typename Kernel_traits::SmemLayoutdK{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dK = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_gmma_dk);
    auto smem_thr_copy_dK = smem_tiled_copy_dK.get_thread_slice(tidx);
    auto smem_tiled_copy_dV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_gmma_dv);
    auto smem_thr_copy_dV = smem_tiled_copy_dV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dK.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dK.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    __syncthreads();

    cute::copy(smem_tiled_copy_dK, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dK, taccdVrdV, taccdVsdV);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
       + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
       + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<OutElement *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<OutElement *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                             make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<OutElement>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<OutElement>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );

}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace flash
