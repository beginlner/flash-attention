/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, class TileShapeV_MNK_, int AtomLayoutNdKV, class Element_, int NumEpilogueThreads_, bool Varlen_>
struct CollectiveEpilogueBwd {

    using TileShape_MNK = TileShape_MNK_;
    using TileShapeV_MNK = TileShapeV_MNK_;
    using Element = Element_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;

    using GmemTiledCopydKVTMA = cute::SM90_TMA_STORE;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydK = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store
    static_assert(get<2>(TileShapeV_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kHeadDimV = get<2>(TileShapeV_MNK{});
    static constexpr int kGmemThreadsPerRowdV = cutlass::gcd(kHeadDimV / kGmemElemsPerLoad, NumEpilogueThreads);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRowdV == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRowV");
    using GmemLayoutAtomdV = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRowdV>, Int<kGmemThreadsPerRowdV>>,
                                  Stride<Int<kGmemThreadsPerRowdV>, _1>>;
    using GmemTiledCopydV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomdV{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    static constexpr int MmadKShapeN = get<2>(TileShape_MNK{}) / (2 / AtomLayoutNdKV);
    static constexpr int MmadVShapeN = get<2>(TileShapeV_MNK{}) / (2 / AtomLayoutNdKV);
    using SmemLayoutAtomdKTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), Int<MmadKShapeN>>());
    using SmemLayoutAtomdVTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShapeV_MNK{})), Int<MmadVShapeN>>());
    using SmemLayoutdKTMA = decltype(tile_to_shape(SmemLayoutAtomdKTMA{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdVTMA = decltype(tile_to_shape(SmemLayoutAtomdVTMA{}, select<1, 2>(TileShapeV_MNK{})));

    // If we don't use TMA
    static constexpr int kBlockKSmem = MmadKShapeN % 64 == 0 ? 64 : (MmadKShapeN % 32 == 0 ? 32 : 16);
    static constexpr int kSwizzle = kBlockKSmem == 64 ? 3 : (kBlockKSmem == 32 ? 2 : 1);
    using SmemLayoutAtomdKSTG =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                             Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                             Stride<Int<kBlockKSmem>, _1>>{}));
    static constexpr int kBlockKSmemV = MmadVShapeN % 64 == 0 ? 64 : (MmadVShapeN % 32 == 0 ? 32 : 16);
    static constexpr int kSwizzleV = kBlockKSmemV == 64 ? 3 : (kBlockKSmemV == 32 ? 2 : 1);
    using SmemLayoutAtomdVSTG =
        decltype(composition(Swizzle<kSwizzleV, 3, 3>{},
                             Layout<Shape<Int<8>, Int<kBlockKSmemV>>,
                             Stride<Int<kBlockKSmemV>, _1>>{}));

    using SmemLayoutAtomdK = std::conditional_t<!Varlen, SmemLayoutAtomdKTMA, SmemLayoutAtomdKSTG>;
    using SmemLayoutAtomdV = std::conditional_t<!Varlen, SmemLayoutAtomdVTMA, SmemLayoutAtomdVSTG>;
    using SmemLayoutdK = decltype(tile_to_shape(SmemLayoutAtomdK{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdV = decltype(tile_to_shape(SmemLayoutAtomdV{}, select<1, 2>(TileShapeV_MNK{})));

    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, Element>;

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdK>> smem_dk;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdV>> smem_dv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using LayoutdKV = cute::Layout<ShapedKV, StridedKV>;

    using TMA_dK = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
        SmemLayoutdKTMA{},
        select<1, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for dK

    using TMA_dV = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
        SmemLayoutdVTMA{},
        select<1, 2>(TileShapeV_MNK{}),
        _1{}));  // no mcast for dV

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        ShapedKV const shape_dV;
        StridedKV const stride_dV;
        int const* cu_seqlens = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        ShapedKV const shape_dV;
        StridedKV const stride_dV;
        TMA_dK tma_store_dK;
        TMA_dV tma_store_dV;
        int const* cu_seqlens = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Varlen) {
            assert (args.cu_seqlens != nullptr);
        }
        Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
        Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dV, args.stride_dV);
        TMA_dK tma_store_dK = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdK,
            SmemLayoutdKTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dK
        TMA_dV tma_store_dV = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdV,
            SmemLayoutdVTMA{},
            select<1, 2>(TileShapeV_MNK{}),
            _1{}); // no mcast for dV
        return {args.ptr_dK, args.shape_dK, args.stride_dK, args.ptr_dV, args.shape_dV, args.stride_dV,
                tma_store_dK, tma_store_dV, args.cu_seqlens};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (!Varlen) {
            cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
        }
    }

    template <typename SharedStorage, typename FrgTensordK, typename FrgTensordV, typename TiledMmadK, typename TiledMmadV>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensordK const& tdKrdK,
          FrgTensordV const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMmadK tiled_mma_dK,
          TiledMmadV tiled_mma_dV,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb] = block_coord;
        Tensor sdK = make_tensor(make_smem_ptr(shared_storage.epilogue.smem_dk.data()), SmemLayoutdK{});
        Tensor sdV = make_tensor(make_smem_ptr(shared_storage.epilogue.smem_dv.data()), SmemLayoutdV{});
        auto smem_tiled_copy_dK = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma_dK);
        auto smem_tiled_copy_dV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma_dV);
        auto smem_thr_copy_dK = smem_tiled_copy_dK.get_thread_slice(thread_idx);
        auto smem_thr_copy_dV = smem_tiled_copy_dV.get_thread_slice(thread_idx);

        Tensor tdVrdV_out = flash::convert_type<Element>(tdVrdV);
        Tensor tdKrdK_out = flash::convert_type<Element>(tdKrdK);
        Tensor taccdKrdK = smem_thr_copy_dK.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdKsdK = smem_thr_copy_dK.partition_D(sdK);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dV.partition_D(sdV);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading K and V

        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);
        cute::copy(smem_tiled_copy_dK, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dV, taccdKrdK, taccdKsdK);
        if constexpr (!Varlen) {
            cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
            cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

            Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK);
            Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dV);
            Tensor gdK = local_tile(mdK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            Tensor gdV = local_tile(mdV(_, _, bidh, bidb), select<1, 2>(TileShapeV_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
            auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
            Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
            Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
            Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                int const lane_predicate = cute::elect_one_sync();
                if (lane_predicate) {
                    cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
                    cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
                    tma_store_arrive();
                }
            }

        } else {
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            bool const is_varlen = params.cu_seqlens != nullptr;
            int const offset = !is_varlen ? 0 : params.cu_seqlens[bidb];
            int const seqlen = !is_varlen ? get<0>(params.shape_dK) : params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];

            Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
            Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
            Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShapeV_MNK{}), make_coord(n_block, _0{}));  // (M, K)

            GmemTiledCopydK gmem_tiled_copy_dK;
            GmemTiledCopydV gmem_tiled_copy_dV;
            auto gmem_thr_copy_dK = gmem_tiled_copy_dK.get_thread_slice(thread_idx);
            auto gmem_thr_copy_dV = gmem_tiled_copy_dV.get_thread_slice(thread_idx);
            Tensor tdKgdK = gmem_thr_copy_dK.partition_D(gdK);
            Tensor tdVgdV = gmem_thr_copy_dV.partition_D(gdV);
            Tensor tdKsdK = gmem_thr_copy_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
            Tensor tdVsdV = gmem_thr_copy_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
            Tensor tdKrdK = make_fragment_like(tdKgdK);
            Tensor tdVrdV = make_fragment_like(tdVgdV);
            cute::copy(gmem_tiled_copy_dK, tdKsdK, tdKrdK);
            cute::copy(gmem_tiled_copy_dV, tdVsdV, tdVrdV);
            // Construct identity layout for gdK and gdV
            Tensor cdK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
            Tensor cdV = cute::make_identity_tensor(select<1, 2>(TileShapeV_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
            // Repeat the partitioning with identity layouts
            Tensor tdKcdK = gmem_thr_copy_dK.partition_D(cdK);
            Tensor tdVcdV = gmem_thr_copy_dV.partition_D(cdV);
            Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
            Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVgdV)));
            #pragma unroll
            for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
            #pragma unroll
            for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(_0{}, _0{}, k)) < get<1>(params.shape_dV); }
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dK, tdKrdK, tdKgdK, tdKcdK, tdKpdK, seqlen - n_block * kBlockN
            );
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, seqlen - n_block * kBlockN
            );
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        if constexpr (!Varlen) { tma_store_wait<0>(); }
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        auto [n_block, bidh, bidb] = block_coord;
        bool const is_varlen = Varlen && params.cu_seqlens != nullptr;
        int const offset = !is_varlen ? 0 : params.cu_seqlens[bidb];
        int const seqlen = !is_varlen ? get<0>(params.shape_dK) : params.cu_seqlens[bidb + 1] - offset;

        Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShapeV_MNK{}), make_coord(n_block, _0{}));  // (M, K)

        GmemTiledCopydK gmem_tiled_copy_dK;
        GmemTiledCopydV gmem_tiled_copy_dV;
        auto gmem_thr_copy_dK = gmem_tiled_copy_dK.get_thread_slice(thread_idx);
        auto gmem_thr_copy_dV = gmem_tiled_copy_dV.get_thread_slice(thread_idx);
        Tensor tdKgdK = gmem_thr_copy_dK.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dV.partition_D(gdV);
        Tensor tdKrdK = make_fragment_like(tdKgdK);
        Tensor tdVrdV = make_fragment_like(tdVgdV);
        clear(tdKrdK);
        clear(tdVrdV);
        // Construct identity layout for gdK and gdV
        Tensor cdK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor cdV = cute::make_identity_tensor(select<1, 2>(TileShapeV_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tdKcdK = gmem_thr_copy_dK.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVgdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(_0{}, _0{}, k)) < get<1>(params.shape_dV); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dK, tdKrdK, tdKgdK, tdKcdK, tdKpdK, seqlen - n_block * kBlockN
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, seqlen - n_block * kBlockN
        );
    }

};

} // namespace flash
