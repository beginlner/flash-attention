/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#endif

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"

using namespace cute;

template <typename PrecType, int DIM> constexpr auto getSmemLayoutK() {
    constexpr int headSizeBytes = sizeof(PrecType) * DIM;

    if constexpr (headSizeBytes % 128 == 0) {
        return GMMA::Layout_K_SW128_Atom<PrecType>{};
    } else if constexpr (headSizeBytes % 64 == 0) {
        return GMMA::Layout_K_SW64_Atom<PrecType>{};
    } else {
        return GMMA::Layout_K_SW32_Atom<PrecType>{};
    }
}

template <typename PrecType, int DIM> constexpr auto getSmemLayoutMN() {

    constexpr int headSizeBytes = sizeof(PrecType) * DIM;

    if constexpr (headSizeBytes % 128 == 0) {
        return GMMA::Layout_MN_SW128_Atom<PrecType>{};
    } else if constexpr (headSizeBytes % 64 == 0) {
        return GMMA::Layout_MN_SW64_Atom<PrecType>{};
    } else {
        return GMMA::Layout_MN_SW32_Atom<PrecType>{};
    }
}

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
        typename out_type=cutlass::half_t,
        int kHeadDimV_=0,
        bool Share_KV_=false,
        int kNWarpsS_=0,
        bool Blocked_KV_=true,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_fp8_kernel_traits : public Base {
    using Element = elem_type;
    using OutElement = out_type;
    static_assert(sizeof_bits_v<Element> == 8);
    static_assert(sizeof_bits_v<OutElement> == 16);
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;
    static_assert(!Share_Q_K_smem);
    static_assert(!Is_Q_in_regs);
    static constexpr bool Share_KV = Share_KV_;
    static constexpr bool Blocked_KV = Blocked_KV_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kNWarpsS = kNWarpsS_ == 0 ? kNWarps : kNWarpsS_;
    static constexpr int kNThreadsS = kNWarpsS * 32;
    static_assert(kNThreads % kNThreadsS == 0);

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<kNWarpsS / 4>, _1, _1>>{}));

    static constexpr int AtomLayoutNO = kNThreads / kNThreadsS;
    using TiledMmaO = decltype(make_tiled_mma(
            cute::GMMA::rs_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kHeadDimV / AtomLayoutNO>, Int<kBlockN>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<kNWarpsS / 4>, Int<AtomLayoutNO>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    using SmemLayoutVt = decltype(tile_to_shape(
            getSmemLayoutK<Element, kBlockN>(),
            Shape<Int<kHeadDimV>, Int<kBlockN>>{}));

    using SmemLayoutP = Layout<Shape<Shape<_2, _2>, Int<kNThreadsS>, _1, Int<kBlockN / 8>>>;
    using SmemLayoutRow = Layout<Shape<_2, Int<kNThreadsS>>, Stride<Int<kNThreadsS>, _1>>;

    using SmemLayoutAtomO = decltype(
    composition(Swizzle<kSwizzle, 3, 3>{},
                Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, OutElement>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDimV % (4 * kNWarps) == 0);
    using SmemTiledCopyV = decltype(make_tiled_copy(
            Copy_Atom<SM75_U16x4_LDSM_T, Element>{},
            Layout<Shape<Shape<_4, _4>, Shape<_8, Int<kNWarps / 4>>>, Stride<Stride<_1, _32>, Stride<_4, _128>>>{},
            Layout<Shape<_4, _2>, Stride<_2, _1>>{}));
    using SmemTiledCopyVt = decltype(make_tiled_copy(
            Copy_Atom<SM90_U32x2_STSM_N, Element>{},
            Layout<Shape<Shape<_8, Int<kNWarps / 4>>, Shape<_4, _4>>, Stride<Stride<_4, _128>, Stride<_1, _32>>>{},
            Layout<Shape<_2, _4>, Stride<_4, _1>>{}));

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = (Share_KV ? size(SmemLayoutK{}) * 2 : size(SmemLayoutK{}) + size(SmemLayoutV{})) * sizeof(Element);
    static constexpr int kSmemPSize = kNThreadsS == kNThreads ? 0 : size(SmemLayoutP{}) * sizeof(Element);
    static constexpr int kSmemRowSize = kNThreadsS == kNThreads ? 0 : size(SmemLayoutRow{}) * 3 * sizeof(float);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(OutElement);
    static constexpr int kSmemSize = std::max(kSmemQSize + kSmemKVSize + kSmemPSize + kSmemRowSize, kSmemOSize);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
            Has_cp_async,
            SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
            DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 vals per read

    static constexpr int kGmemElemsPerLoadO = sizeof(cute::uint128_t) / sizeof(OutElement);
    static_assert(kHeadDimV % kGmemElemsPerLoadO == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadO");
    static constexpr int kGmemThreadsPerRowO = kBlockKSmem / kGmemElemsPerLoadO;
    static_assert(kNThreads % kGmemThreadsPerRowO == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtomO = Layout<Shape <Int<kNThreadsS / kGmemThreadsPerRowO>, Int<kGmemThreadsPerRowO>>,
            Stride<Int<kGmemThreadsPerRowO>, _1>>;
    using GmemTiledCopyO = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, OutElement>{},
                    GmemLayoutAtomO{},
                    Layout<Shape<_1, Int<kGmemElemsPerLoadO>>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
            kBlockKSmem == 32,
            Layout<Shape<_16, _8>,  // Thread layout, 8 threads per row
                    Stride< _8, _1>>,
            Layout<Shape<_8, _16>,  // Thread layout, 16 threads per row
                    Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                    GmemLayoutAtomOaccum{},
                    Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
        int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::float_e4m3_t,
        typename out_type=cutlass::half_t,
        int kHeadDimV_=0,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_fp8_kernel_traits : public Base {
    using Element = elem_type;
    using OutElement = out_type;
    static_assert(sizeof_bits_v<Element> == 8);
    static_assert(sizeof_bits_v<OutElement> == 16);
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;

    static_assert(!Is_V_in_regs_);
    static constexpr bool No_double_buffer = No_double_buffer_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(AtomLayoutMSdP % 4 == 0);
    static_assert(AtomLayoutNdKV % 4 == 0);
    static_assert(AtomLayoutMdQ % 4 == 0);
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    using TiledMmaSdP = TiledMMA<
            typename Base::MMA_Atom_Arch,
            Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
            Tile<Int<16 * AtomLayoutMSdP>, Int<16 * kNWarps / AtomLayoutMSdP>, _16>>;

    using TiledGMmaS = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM / (AtomLayoutMSdP / 4)>, Int<kBlockN / (kNWarps / AtomLayoutMSdP)>, Int<kHeadDim>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<AtomLayoutMSdP / 4>, Int<kNWarps / AtomLayoutMSdP>, _1>>{}));

    using TiledGMmadP = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM / (AtomLayoutMSdP / 4)>, Int<kBlockN / (kNWarps / AtomLayoutMSdP)>, Int<kHeadDimV>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<AtomLayoutMSdP / 4>, Int<kNWarps / AtomLayoutMSdP>, _1>>{}));

    using TiledGMmadK = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockN / (AtomLayoutNdKV / 4)>, Int<kHeadDim / (kNWarps / AtomLayoutNdKV)>, Int<kBlockM>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<AtomLayoutNdKV / 4>, Int<kNWarps / AtomLayoutNdKV>, _1>>{}));

    using TiledGMmadV = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockN / (AtomLayoutNdKV / 4)>, Int<kHeadDimV / (kNWarps / AtomLayoutNdKV)>, Int<kBlockM>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<AtomLayoutNdKV / 4>, Int<kNWarps / AtomLayoutNdKV>, _1>>{}));

    using TiledGMmadQ = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM / (AtomLayoutMdQ / 4)>, Int<kHeadDim / (kNWarps / AtomLayoutMdQ)>, Int<kBlockN>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<AtomLayoutMdQ / 4>, Int<kNWarps / AtomLayoutMdQ>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutQt = decltype(tile_to_shape(
            getSmemLayoutK<Element, kBlockM>(),
            make_shape(Int<kHeadDim>{}, Int<kBlockM>{})));

    using SmemLayoutK = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKt = decltype(tile_to_shape(
            getSmemLayoutK<Element, kBlockN>(),
            make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDimV>(),
            make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

    using SmemLayoutdO = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDimV>(),
            make_shape(Int<kBlockM>{}, Int<kHeadDimV>{})));

    using SmemLayoutdOt = decltype(tile_to_shape(
            getSmemLayoutK<Element, kBlockM>(),
            make_shape(Int<kHeadDimV>{}, Int<kBlockM>{})));

    using SmemLayoutPt = decltype(tile_to_shape(
            getSmemLayoutK<Element, kBlockM>(),
            make_shape(Int<kBlockN>{}, Int<kBlockM>{})));
    using SmemLayoutPt_t = decltype(
    composition(SmemLayoutPt{}, make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutdS = decltype(tile_to_shape(
            getSmemLayoutK<Element, kBlockN>(),
            make_shape(Int<kBlockM>{}, Int<kBlockN>{})));

    using SmemLayoutdSt = decltype(tile_to_shape(
            getSmemLayoutK<Element, kBlockM>(),
            make_shape(Int<kBlockN>{}, Int<kBlockM>{})));
    using SmemLayoutdSt_t = decltype(
    composition(SmemLayoutPt{}, make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutdK = decltype(tile_to_shape(
            getSmemLayoutK<OutElement, kHeadDim>(),
            make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutdV = decltype(tile_to_shape(
            getSmemLayoutK<OutElement, kHeadDimV>(),
            make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

    using SmemCopyAtomdKV = Copy_Atom<SM90_U32x4_STSM_N, OutElement>;

    // Transpose Q, K and dO in shared memory
    static_assert(kBlockM % 64 == 0);
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % (4 * kNWarps) == 0);
    static_assert(kHeadDimV % (4 * kNWarps) == 0);
    using SmemTiledCopyX = decltype(make_tiled_copy(
            Copy_Atom<SM75_U16x4_LDSM_T, Element>{},
            Layout<Shape<Shape<_4, _4>, Shape<_8, Int<kNWarps / 4>>>, Stride<Stride<_1, _32>, Stride<_4, _128>>>{},
            Layout<Shape<_4, _2>, Stride<_2, _1>>{}));
    using SmemTiledCopyXt = decltype(make_tiled_copy(
            Copy_Atom<SM90_U32x2_STSM_N, Element>{},
            Layout<Shape<Shape<_8, Int<kNWarps / 4>>, Shape<_4, _4>>, Stride<Stride<_4, _128>, Stride<_1, _32>>>{},
            Layout<Shape<_2, _4>, Stride<_4, _1>>{}));

    // Double buffer for sQ
    static constexpr int kSmemQSize = (size(SmemLayoutQ{}) + size(SmemLayoutQt{})) * (No_double_buffer ? 1 : 2) * sizeof_bytes_v<Element>;
    static constexpr int kSmemKSize = (size(SmemLayoutK{}) + size(SmemLayoutKt{})) * sizeof_bytes_v<Element>;
    static constexpr int kSmemVSize = size(SmemLayoutV{}) * sizeof_bytes_v<Element>;
    static constexpr int kSmemdOSize = (size(SmemLayoutdO{}) + size(SmemLayoutdOt{})) * sizeof_bytes_v<Element>;
    static constexpr int kSmemPSize = size(SmemLayoutPt{}) * sizeof_bytes_v<Element>;
    static constexpr int kSmemdSSize = (size(SmemLayoutdS{}) + size(SmemLayoutdSt{})) * sizeof_bytes_v<Element>;
    // kSmemdKVSize is always smaller than others.
    static constexpr int kSmemSize1colblock = kSmemQSize + kSmemKSize + kSmemVSize + kSmemdOSize + kSmemPSize + kSmemdSSize;

    using GmemLayoutAtom8Bit = Layout<Shape<Int<kNThreads / (kBlockKSmem / 16)>, Int<kBlockKSmem / 16>>, Stride<Int<kBlockKSmem / 16>, _1>>;
    using GmemLayoutAtom16Bit = Layout<Shape<Int<kNThreads / (kBlockKSmem / 8)>, Int<kBlockKSmem / 8>>, Stride<Int<kBlockKSmem / 8>, _1>>;
    using GmemLayoutAtom32Bit = Layout<Shape<Int<kNThreads / (kBlockKSmem / 4)>, Int<kBlockKSmem / 4>>, Stride<Int<kBlockKSmem / 4>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
            Has_cp_async,
            SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
            DefaultCopy
    >;

    using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom8Bit{},
                    Layout<Shape<_1, _16>>{}));

    using GmemTiledCopydKV = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, OutElement>{},
                    GmemLayoutAtom16Bit{},
                    Layout<Shape<_1, _8>>{}));

    using GmemTiledCopydQaccumAtomicAdd = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                    Layout<Shape<_8, _32>, Stride<_32, _1>>{},
                    Layout<Shape<_1, _1>>{}));

    // For preprocess

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / 8;

    using GmemTiledCopydO = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                    GmemLayoutAtom16Bit{},
                    Layout<Shape<_1, _8>>{}));

    using GmemTiledCopyO = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, OutElement>{},
                    GmemLayoutAtom16Bit{},
                    Layout<Shape<_1, _8>>{}));

    using GmemTiledCopydQaccum = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                    GmemLayoutAtom32Bit{},
                    Layout<Shape<_1, _4>>{}));

    // For postprocess

    using SmemLayoutdQ = decltype(tile_to_shape(
            GMMA::Layout_K_SW64_Atom<OutElement>{},
            make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<SM90_U32x4_STSM_N, OutElement>;
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(OutElement);

    using GmemTiledCopydQ = decltype(
    make_tiled_copy(Copy_Atom<DefaultCopy, OutElement>{},
                    GmemLayoutAtom16Bit{},
                    Layout<Shape<_1, _8>>{}));

};

////////////////////////////////////////////////////////////////////////////////////////////////////
