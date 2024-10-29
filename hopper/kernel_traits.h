/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    union {
        cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterBarrier barrier_O;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        int tile_count_semaphore;
    };
};

// Use if Oaccum is too large for SharedStorageQKVO
template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOaccum {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;    
    union {    
        struct {    
            cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
            cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
        };
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterBarrier barrier_O;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        int tile_count_semaphore;
    };
};

// SharedStorage struct with no smem for O
template <int kStages, class Gemm1Type, class Gemm2Type, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV>
struct SharedStorageQKV {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        int tile_count_semaphore;
    };
};

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;  
    union {
        cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
  };
  struct {    
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
    float softmax_scale_qk_log2;
    float descale_v;
    bool seqlen_init_k;
  };
};

// Use if Oaccum is too large for SharedStorageQKVOVt
template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOVtaccum {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    union {
        struct {
            cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;  
            cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
        };
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
  };
  struct {    
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
    float softmax_scale_qk_log2;
    float descale_v;
    bool seqlen_init_k;
  };
};

template <int kStages, class Gemm1Type, class Gemm2Type, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV>
struct SharedStorageQKVVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;  
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
  };
  struct {    
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
    float softmax_scale_qk_log2;
    float descale_v;
    bool seqlen_init_k;
  };
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_, bool Is_Q_in_regs_=false,
         int kClusterM_ = 1, typename elem_type=cutlass::half_t, bool Is_split_=false, int kBlockH_ = 1, int kHeadDimV_=0>
struct Flash_fwd_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using FinalOutputType = elem_type;
    using OutputType = std::conditional_t<Is_split_, float, FinalOutputType>;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;

    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
    static_assert(kNWarps_ == 8 || kNWarps_ == 12 || kNWarps_ == 16);
    static constexpr bool Is_WS = true;
    static_assert(!(Is_WS && Is_Q_in_regs), "Warp-specialization does not support Q in registers");

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockH = kBlockH_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_ > 0 ? kHeadDimV_ : kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kBlockM % kBlockH == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShapeV_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDimV>>;

    static constexpr int kClusterM = kClusterM_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_;

    static constexpr bool Is_split = Is_split_;
    static constexpr bool No_smem_O = Is_split;

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using TiledMma0 = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Is_Q_in_regs,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>())
        >{},
        AtomLayoutMNK{}));
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShapeV_MNK{})),
                                   GMMA::Major::K, GMMA::Major::MN>(),
        AtomLayoutMNK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    // for gmem -> smem Q copy 
    using FactoringLayoutQ = Layout<Shape<Int<kBlockM/kBlockH>, Int<kBlockH>, Int<kHeadDim>>,
        Stride<Int<kBlockH>, _1, Int<kBlockM>>>;
    using TileShapeQCopy = std::conditional_t<(kBlockH > 1),
        decltype(shape(FactoringLayoutQ{})), decltype(select<0, 2>(TileShape_MNK{}))>;
    using SmemLayoutQCopy = std::conditional_t<(kBlockH > 1),
        decltype(composition(SmemLayoutQ{}, FactoringLayoutQ{})), SmemLayoutQ>;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShapeV_MNK{})), decltype(cute::get<2>(TileShapeV_MNK{}))>());
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(get<1>(TileShapeV_MNK{}), get<2>(TileShapeV_MNK{}), Int<kStages>{})));

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutVt =
        decltype(composition(SmemLayoutV{},
                    make_ordered_layout(
                        make_shape(get<2>(TileShapeV_MNK{}), get<1>(TileShapeV_MNK{}), Int<kStages>{}),
                        Step<_2, _1, _3>{})));
    
    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShapeV_MNK{})), decltype(cute::get<2>(TileShapeV_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShapeV_MNK{})));
    // for smem -> gmem O copy
    using TileShapeOCopy = TileShapeQCopy;
    using SmemLayoutOCopy = std::conditional_t<(kBlockH > 1),
        decltype(composition(SmemLayoutO{}, FactoringLayoutQ{})), SmemLayoutO>;

    using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    using SharedStorage = std::conditional_t<!No_smem_O,
        SharedStorageQKVO<kStages, Element, Element, OutputType, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>,
        SharedStorageQKV<kStages, Element, Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    // using BarrierType = typename MainloopPipeline::ProducerBarrierType;

};

// Traits struct for fp8 kernel with in-kernel transpose
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_, bool Is_Q_in_regs_=false,
         int kClusterM_ = 1, typename elem_type=cutlass::float_e4m3_t, bool Is_split_ = false, int kBlockH_ = 1, int kHeadDimV_=0>
struct Flash_fwd_kernel_traits_fp8 {
    using Element = elem_type;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);
    using ElementAccum = float;
    using FinalOutputType = cutlass::bfloat16_t;
    using OutputType = std::conditional_t<Is_split_, float, FinalOutputType>;
    using index_t = int64_t;

    static constexpr bool Is_split = Is_split_;
    static constexpr bool No_smem_O = false;
    // NOTE: not using smem for epilogue degrades perf substantially.
    // static constexpr bool No_smem_O = Is_split;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;

    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
    static_assert(kNWarps_ == 8 || kNWarps_ == 12 || kNWarps_ == 16);
    static constexpr bool Is_WS = true;    
    static_assert(!Is_Q_in_regs, "Warp-specialization does not support Q in registers");    

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockH = kBlockH_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_ > 0 ? kHeadDimV_ : kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kBlockM % kBlockH == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShapeV_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDimV>>;

    static constexpr int kClusterM = kClusterM_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_;
    static_assert(kStages > 1);

    // Use this to save enough smem when writing out in float precision.
    static constexpr bool VO_union_all = Is_split && (kBlockM != 64) && (kHeadDim == 256);

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;    
    using TiledMma0 = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutMNK{}));
    
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShapeV_MNK{}))>(),
        AtomLayoutMNK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    // for gmem -> smem Q copy
    using FactoringLayoutQ = Layout<Shape<Int<kBlockM/kBlockH>, Int<kBlockH>, Int<kHeadDim>>,
        Stride<Int<kBlockH>, _1, Int<kBlockM>>>;
    using TileShapeQCopy = std::conditional_t<(kBlockH > 1),
        decltype(shape(FactoringLayoutQ{})), decltype(select<0, 2>(TileShape_MNK{}))>;
    using SmemLayoutQCopy = std::conditional_t<(kBlockH > 1),
        decltype(composition(SmemLayoutQ{}, FactoringLayoutQ{})), SmemLayoutQ>;

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using TransposeShapeAtomV = Shape<_64, _64>;    
    using SmemLayoutAtomV = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShapeV_MNK{}), shape<2>(TileShapeV_MNK{}), Int<kStages>{})));
    
    // for fp8 in-kernel transpose -- src layout
    using SmemLayoutDivideV = decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtomV{}));
    using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
    using FactoringShapeV = decltype(make_shape(SmemShapeLDSM{},
        shape<1>(SmemLayoutDivideV{}), shape<2>(SmemLayoutDivideV{}), shape<3>(SmemLayoutDivideV{})));
    using SmemLayoutTransposeV = decltype(composition(SmemLayoutDivideV{}, make_layout(FactoringShapeV{})));

    // For fp8, this is the memory transpose.
    using SmemLayoutAtomVt = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShapeV_MNK{}), shape<1>(TileShapeV_MNK{}), Int<kStages>{})));

    // for fp8 in-kernel transpose -- dst layout
    using SmemLayoutVtTrans =
        decltype(composition(SmemLayoutVt{},
                             make_ordered_layout(product_each(shape(SmemLayoutV{})), Step<_2, _1, _3>{})));
    using SmemLayoutDivideVt = decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtomV{}));
#ifndef NO_FP8_COLUMN_PERMUTE
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_8, _8>>;
#else
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
#endif
    using FactoringShapeVt = decltype(make_shape(SmemShapeSTSM{},
        shape<1>(SmemLayoutDivideVt{}), shape<2>(SmemLayoutDivideVt{}), shape<3>(SmemLayoutDivideVt{})));
    using SmemLayoutTransposeVt = decltype(composition(SmemLayoutDivideVt{}, make_layout(FactoringShapeVt{})));

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShapeV_MNK{})), decltype(cute::get<2>(TileShapeV_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShapeV_MNK{})));
    // for smem -> gmem O copy
    using TileShapeOCopy = TileShapeQCopy;
    using SmemLayoutOCopy = std::conditional_t<(kBlockH > 1),
        decltype(composition(SmemLayoutO{}, FactoringLayoutQ{})), SmemLayoutO>;

    // used for rmem -> smem O copy in fp8 kernel to undo column permutation
    using ThreadLayoutrO = Layout<Shape<_8, Int<kBlockM/16>, _4, _1>,
                                 Stride<_4, _32, _1, _0>>;
    using ValueLayoutrO = Layout<Shape<_1, _2, Shape<_2, _2>, Int<kHeadDimV/16>>,
                                Stride<_0, _2, Stride<_4, _1>, _8>>;
    using TiledCopyrO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, OutputType>{},
                      ThreadLayoutrO{}, ValueLayoutrO{}));

    using TiledCopyShaperO = Shape<_8, Int<kBlockM/8>, _16, Int<kHeadDimV/16>>;
    using SmemLayoutrO = decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

    using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    using SharedStorage = std::conditional_t<!No_smem_O,
        std::conditional_t<!VO_union_all,
            SharedStorageQKVOVt<kStages, Element, Element, OutputType, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>,
            SharedStorageQKVOVtaccum<kStages, Element, Element, OutputType, SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>>,
        SharedStorageQKVVt<kStages, Element, Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    // using BarrierType = typename MainloopPipeline::ProducerBarrierType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
