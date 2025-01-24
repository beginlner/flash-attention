#pragma once

#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#endif

#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

using namespace cute;

#include "named_barrier.h"
#include "block_info.h"
#include "utils.h"
#include "softmax.h"
#include "static_switch.h"
#include "flash_mla.h"


template<typename PrecType, int DIM, int DIM2 = DIM>
constexpr auto getSmemLayoutK() {
    constexpr int headSizeBytes = sizeof(PrecType) * DIM;
    constexpr int headSizeBytes2 = sizeof(PrecType) * DIM2;

    if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
        return GMMA::Layout_K_SW128_Atom<PrecType>{};
    } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
        return GMMA::Layout_K_SW64_Atom<PrecType>{};
    } else {
        return GMMA::Layout_K_SW32_Atom<PrecType>{};
    }
}

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t,
        int kHeadDimV_ = 0,
        bool Shared_KV_ = true,
        int SplitLength_ = 0, typename KV_type0_=cutlass::bfloat16_t, typename KV_type1_=cutlass::bfloat16_t>
struct Flash_fwd_kernel_traits_mla {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr bool Shared_KV = Shared_KV_;
    static constexpr bool Blocked_KV = true;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kNWarpsS = 4;
    static constexpr int kNThreadsS = kNWarpsS * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kHeadDimV <= kHeadDim);
    static constexpr int kBlockKSmem = (kHeadDim % 64 == 0 && SplitLength_ % 64 == 0) ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<kNWarpsS / 4>, _1, _1>>{}));

    static constexpr int AtomLayoutNO = kNThreads / kNThreadsS;
    using TiledMmaO = decltype(make_tiled_mma(
            cute::GMMA::rs_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kHeadDimV / AtomLayoutNO>, Int<kBlockN>>,
                    GMMA::Major::K, GMMA::Major::MN>(),
            Layout<Shape<Int<kNWarpsS / 4>, Int<AtomLayoutNO>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutP = Layout<Shape<Shape<_2, _2>, Int<kNThreadsS>, _1, Int<kBlockN / 8>>>;
    using SmemLayoutRow = Layout<Shape<_2, Int<kNThreadsS>>, Stride<_1, _2>>;

    using SmemLayoutAtomO = decltype(composition(
            Swizzle<kSwizzle, 3, 3>{},
            Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    static constexpr int kNThreadsLoad = Shared_KV ? kNThreads - kNThreadsS : kNThreads;
    static_assert(kNThreadsLoad % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<
            Shape<Int<kNThreadsLoad / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(make_tiled_copy(
            Copy_Atom<Gmem_copy_struct, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemLayoutAtomO = Layout<
            Shape<Int<kNThreadsS / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
            Copy_Atom<DefaultCopy, Element>{},
            GmemLayoutAtomO{},
            Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowAccum = kBlockKSmem / kGmemElemsPerLoadAccum;
    using GmemLayoutAtomOaccum = Layout<
            Shape<Int<kNThreadsS / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
            Stride<Int<kGmemThreadsPerRowAccum>, _1>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<DefaultCopy, ElementAccum>{},
            GmemLayoutAtomOaccum{},
            Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store

    static constexpr int SplitLength = SplitLength_;
    static_assert(SplitLength % kBlockKSmem == 0);
    using KV_type0 = std::conditional_t<(SplitLength > 0), KV_type0_, Element>;
    using KV_type1 = std::conditional_t<(SplitLength > 0), KV_type1_, Element>;
    using GmemTiledCopyKQuant0 = decltype(make_tiled_copy(
            Copy_Atom<DefaultCopy, KV_type0>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));
    using GmemTiledCopyKQuant1 = decltype(make_tiled_copy(
            Copy_Atom<std::conditional_t<std::is_same_v<KV_type1, Element>, Gmem_copy_struct, DefaultCopy>, KV_type1>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));
    using SmemTiledCopyK = decltype(make_tiled_copy(
            Copy_Atom<DefaultCopy, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));
};

namespace flash {

using namespace cute;

static constexpr int MaxNumPagesPerBlock = 256;

template<typename Kernel_traits>
struct SharedStorageMLA {
    union {
        struct {
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK> * 2> smem_k;  // Double buffer
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutP>> smem_p;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_scale;
            cute::array_aligned<int, MaxNumPagesPerBlock> smem_block_table;
        };
        struct {
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_max;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_sum;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;
        };
    };
};

template<typename Kernel_traits>
struct SharedStorageMHA {
    union {
        struct {
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK>> smem_k;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutV>> smem_v;
            cute::array_aligned<int, MaxNumPagesPerBlock> smem_block_table;
        };
        struct {
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;
        };
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Split, typename SharedStorage, typename AccO, typename Softmax>
__forceinline__ __device__ void store(const Flash_fwd_mla_params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx,
                                      SharedStorage &shared_storage, AccO acc_o, Softmax softmax) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    // Epilogue

    const int split_offset = __ldg(params.num_splits_ptr + bidb);

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);
    // if (cute::thread0()) { print(lse); }

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shared_storage.smem_o.data())), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
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

    __syncthreads();

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (Split ? row_offset_lseaccum : row_offset_lse)),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    if (tidx >= kNThreadsS) { return; }

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
            if (row < params.seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla(const Flash_fwd_mla_params &params,
                                                                   const int bidb, const int bidh, const int m_block,
                                                                   const int n_split_idx, const int seqlen_k,
                                                                   const int n_block_min, const int n_block_max, const bool NoSplit,
                                                                   SharedStorage &shared_storage) {
    static_assert(Kernel_traits::Shared_KV);
    static_assert(Kernel_traits::Blocked_KV);
    static_assert(std::is_same_v<typename Kernel_traits::KV_type1, typename Kernel_traits::Element>);

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    static_assert(kNThreads == 256 and kNThreadsS == 128);
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;
    int n_block = n_block_max - 1;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutVtransposed{});

    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Kernel_traits::SmemLayoutP{});
    Tensor tPsP = sP(_, tidx % kNThreadsS, _, _);
    Tensor sScale_o = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tScale_osScale_o = sScale_o(_, tidx % kNThreadsS);
    Tensor sRow_max = make_tensor(make_smem_ptr(shared_storage.smem_max.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_maxsRow_max = sRow_max(_, tidx % kNThreadsS);
    Tensor sRow_sum = make_tensor(make_smem_ptr(shared_storage.smem_sum.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_sumsRow_sum = sRow_sum(_, tidx % kNThreadsS);

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrO = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
    Tensor acc_o = make_tensor(tOrO.data(), flash::convert_gmma_to_mma_tensor(tOrO.layout()));  // (4, MMA_M, X)
    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    int warp_group_idx = cutlass::canonical_warp_group_idx();
    if (warp_group_idx == 0) {
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
        Tensor tSrK = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

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
        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
#pragma unroll 1
        for (int masking_step = n_masking_steps; n_block >= n_block_min; --masking_step, --n_block) {
            __syncthreads();

            Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
            Tensor acc_s = make_tensor(tSrS.data(), flash::convert_gmma_to_mma_tensor(tSrS.layout()));  // (4, MMA_M, X)
            flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma, tSrQ, tSrK, tSrS);

            const bool is_masking_step = masking_step > 0;
            const bool is_first_masking_step = masking_step == n_masking_steps;

            if (is_masking_step) {
                Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
                Tensor tScS = thr_mma.partition_C(cS);
#pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    if constexpr (!Is_causal) {  // Just masking based on col
                        if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) tSrS(i) = -INFINITY;
                    } else {
                        // Ensure seqlen_k - 1 - (n_block * kBlockN + col) >= (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                        // col <= seqlen_k - 1 - n_block * kBlockN - (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                        int row = int(get<0>(tScS(i)));
                        int col_limit_right = seqlen_k - 1 - n_block * kBlockN - (params.seqlen_q - 1 - (m_block * kBlockM + row)) / params.ngroups;
                        if (int(get<1>(tScS(i))) > col_limit_right) tSrS(i) = -INFINITY;
                    }
                }
            }

            // We have key_padding_mask so we'll need to Check_inf
            Tensor scale_o = is_first_masking_step
                             ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2)
                             : is_masking_step ?
                               softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2)
                                               : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*//*Is_local=*/false, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2);

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
        const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        {
            assert(n_block_max - n_block_min <= MaxNumPagesPerBlock);
            // Load block_table from global memory to shared memory
            int *block_table_shared = reinterpret_cast<int *>(shared_storage.smem_block_table.data());
            for (int i = tidx - kNThreadsS; i <= n_block_max - n_block_min - 1; i += kNThreads - kNThreadsS) {
                SM80_CP_ASYNC_CACHEALWAYS<int>::copy(block_table[i + n_block_min], block_table_shared[i]);
            }
            cp_async_fence();
            block_table = block_table_shared - n_block_min;
        }

        const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
        Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_Q;
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx - kNThreadsS);
        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                                                              params.seqlen_q - m_block * kBlockM);
        cp_async_fence();

        flash::cp_async_wait<1>();  // Wait for block_table ready.
        cutlass::arch::NamedBarrier::sync(kNThreads - kNThreadsS, static_cast<int>(NamedBarriers::BlockTableReady));

        // We move K and V to the last block.
        const index_t row_offset_k = block_table[n_block_max - 1] * params.k_batch_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.k_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_K;
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
        auto LDG_K = [&](const int n) {
#pragma unroll
            for (int k = 0; k < size<2>(tKQuant0gKQuant0); ++k) {
                copy(gmem_tiled_copy_KQuant0, tKQuant0gKQuant0(_, n, k), tKQuant0rKQuant0(_, n, k));
            }
            for (int k = 0; k < size<2>(tKQuant1gKQuant1); ++k) {
                copy(gmem_tiled_copy_KQuant1, tKQuant1gKQuant1(_, n, k), tKsK(_, n, size<2>(tKQuant0gKQuant0) + k));
            }
        };
        auto Cast_K = [&](const int n) {
#pragma unroll
            for (int k = 0; k < size<2>(tKQuant0gKQuant0); ++k) {
                convert_type_out(tKQuant0rKQuant0(_, n, k), tKQuant0rKQuant0_high(_, n, k));
            }
        };
        auto STS_K = [&](const int n) {
#pragma unroll
            for (int k = 0; k < size<2>(tKQuant0gKQuant0); ++k) {
                copy(smem_tiled_copy_K, tKQuant0rKQuant0_high(_, n, k), tKsK(_, n, k));
            }
        };
        auto LoadK = [&](int n_block) {
            if (n_block <= n_block_min) { return; }
            // Advance gK
            const index_t offset = (block_table[n_block - 1] - block_table[n_block]) * params.k_batch_stride;

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
#pragma unroll
                for (int n = 0; n < size<1>(tKsK); ++n) {
                    Cast_K(n);
                    STS_K(n);
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
                                                                                         seqlen_k - n_block * kBlockN);
        } else {
#pragma unroll
            for (int n = 0; n < size<1>(tKsK); ++n) {
                if (get<0>(tKcK(0, n, 0)) < seqlen_k - n_block * kBlockN) {
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

    if (NoSplit)
        store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, shared_storage, acc_o, softmax);
    else
        store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, shared_storage, acc_o, softmax);
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mha(const Flash_fwd_mla_params &params,
                                                                   const int bidb, const int bidh, const int m_block,
                                                                   const int n_split_idx, const int seqlen_k,
                                                                   const int n_block_min, const int n_block_max, const bool NoSplit,
                                                                   SharedStorage &shared_storage) {
    static_assert(!Kernel_traits::Shared_KV);
    static_assert(Kernel_traits::Blocked_KV);
    static_assert(Kernel_traits::SplitLength == 0);

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    static_assert(kNThreads == 128 and kNThreadsS == 128);
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;
    int n_block = n_block_max - 1;

    const int *block_table = params.block_table + bidb * params.block_table_batch_stride;

    const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = block_table[n_block] * params.k_batch_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table[n_block] * params.v_batch_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), typename Kernel_traits::SmemLayoutVtransposed{});
    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKcK = gmem_thr_copy_QKV.partition_S(cK);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = gmem_thr_copy_QKV.partition_S(cV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));

    {
        assert(n_block_max - n_block_min <= MaxNumPagesPerBlock);
        // Load block_table from global memory to shared memory
        int *block_table_shared = reinterpret_cast<int *>(shared_storage.smem_block_table.data());
        for (int i = tidx; i <= n_block_max - n_block_min - 1; i += kNThreads) {
            SM80_CP_ASYNC_CACHEALWAYS<int>::copy(block_table[i + n_block_min], block_table_shared[i]);
        }
        block_table = block_table_shared - n_block_min;
    }

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(
            gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, params.seqlen_q - m_block * kBlockM);
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(
            gmem_tiled_copy_QKV, tKgK, tKsK, tKcK, tKpK, seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    auto LoadK = [&](int n_block) {
        // Advance gK
        const index_t offset = (block_table[n_block] - block_table[n_block + 1]) * params.k_batch_stride;
        tKgK.data() = tKgK.data() + offset;

        flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKcK, tKpK);
        cute::cp_async_fence();
    };
    auto LoadV = [&](int n_block) {
        // Advance gV
        const index_t offset = (block_table[n_block] - block_table[n_block + 1]) * params.v_batch_stride;
        tVgV.data() = tVgV.data() + offset;

        flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy_QKV, tVgV, tVsV, tVcV, tVpV);
        cute::cp_async_fence();
    };

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
    Tensor acc_s = make_tensor(tSrS.data(), flash::convert_gmma_to_mma_tensor(tSrS.layout()));  // (4, MMA_M, X)
    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrO = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
    Tensor acc_o = make_tensor(tOrO.data(), flash::convert_gmma_to_mma_tensor(tOrO.layout()));  // (4, MMA_M, X)
    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.
    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
#pragma unroll 1
    for (int masking_step = n_masking_steps; n_block >= n_block_min; --masking_step, --n_block) {
        flash::cp_async_wait<0>();
        __syncthreads();

        const bool is_first_masking_step = masking_step == n_masking_steps;
        if (is_first_masking_step) {
            // Clear the smem tiles to account for predicated off loads
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, tVgV, tVsV, tVcV, tVpV, seqlen_k - n_block * kBlockN);
            cute::cp_async_fence();
        } else {
            LoadV(n_block);
        }

        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma, tSrQ, tSrK, tSrS);

        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block > n_block_min) { LoadK(n_block - 1); }

        const bool is_masking_step = masking_step > 0;
        if (is_masking_step) {
            Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
            Tensor tScS = thr_mma.partition_C(cS);
#pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal) {  // Just masking based on col
                    if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) tSrS(i) = -INFINITY;
                } else {
                    // Ensure seqlen_k - 1 - (n_block * kBlockN + col) >= (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                    // col <= seqlen_k - 1 - n_block * kBlockN - (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                    int row = int(get<0>(tScS(i)));
                    int col_limit_right = seqlen_k - 1 - n_block * kBlockN - (params.seqlen_q - 1 - (m_block * kBlockM + row)) / params.ngroups;
                    if (int(get<1>(tScS(i))) > col_limit_right) tSrS(i) = -INFINITY;
                }
            }
        }

        // We have key_padding_mask so we'll need to Check_inf
        Tensor scale_o = is_first_masking_step
                         ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2)
                         : is_masking_step ?
                           softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2)
                                           : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*//*Is_local=*/false, /*rescale_o=*/false>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        flash::rescale_o(acc_o, scale_o);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);
    }

    if (NoSplit)
        store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, shared_storage, acc_o, softmax);
    else
        store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, shared_storage, acc_o, softmax);
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;
    if (begin_idx >= params.b) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = __ldg(params.cu_seqlens_k + batch_id);
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        const int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == cute::ceil_div(seqlen_k, kBlockN);
        if (batch_id > begin_idx) {
            __syncthreads();  // Barrier between two tiles.
        }
        if constexpr (Kernel_traits::Shared_KV)
            flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, NoSplit, shared_storage);
        else
            flash::compute_attn_1rowblock_splitkv_mha<Kernel_traits, Is_causal>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, NoSplit, shared_storage);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Element, typename ElementAccum, typename index_t, int kHeadDimV, int kMaxSplits>
__global__ void __launch_bounds__(256, 1, 1)
flash_fwd_splitkv_mla_combine_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kNThreads = 128;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int hs = params.h * params.seqlen_q;
    const int batch_idx = bidx / hs;
    const int hs_idx = bidx % hs;

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int actual_num_splits = __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    assert(actual_num_splits <= kMaxSplits);
    if (actual_num_splits == 1) return;

    __shared__ ElementAccum sLseScale[kMaxSplits];

    const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
    const index_t row_offset_lse = bidx;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                   Shape<Int<kMaxSplits>>{}, make_stride(hs));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
        constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

        float local_lse[kNLsePerThread];
        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            local_lse[i] = split < actual_num_splits ? gLSEaccum(split) : -INFINITY;
        }

        float max_lse = -INFINITY;
        for (int i = 0; i < kNLsePerThread; ++i) max_lse = max(max_lse, local_lse[i]);
        for (int offset = 16; offset >= 1; offset /= 2) max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;
        for (int i = 0; i < kNLsePerThread; ++i) sum_lse = sum_lse + expf(local_lse[i] - max_lse);
        for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : logf(sum_lse) + max_lse;
        if (tidx == 0) gLSE(0) = global_lse;

        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            if (split < actual_num_splits) sLseScale[split] = expf(local_lse[i] - global_lse);
        }
    }
    __syncthreads();

    static_assert(kHeadDimV % kNThreads == 0);
    constexpr int Elements = kHeadDimV / kNThreads;
    const index_t row_offset_oaccum = (split_offset * hs + hs_idx) * kHeadDimV;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kHeadDimV>>{}, Stride<_1>{});
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            Layout<Shape<Int<kNThreads>>>{},
            Layout<Shape<Int<Elements>>>{}));
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    for (int split = 0; split < actual_num_splits; ++split) {
        cute::copy(tOgOaccum, tOrOaccum);
        ElementAccum lse_scale = sLseScale[split];
        for (int i = 0; i < size(tOrO); ++i) {
            tOrO(i) += lse_scale * tOrOaccum(i);
        }
        tOgOaccum.data() = tOgOaccum.data() + hs * kHeadDimV;
    }

    Tensor rO = flash::convert_type<Element>(tOrO);
    const int head_idx = (bidx - batch_idx * hs) / params.seqlen_q;
    const int row = bidx - batch_idx * hs - head_idx * params.seqlen_q;
    auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride + row * params.o_row_stride;
    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + tidx * Elements), Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
    cute::copy(rO, gO);
}

} // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename SharedStorage>
void run_flash_splitkv_fwd_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.page_block_size == Kernel_traits::kBlockN);
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage>;
        constexpr size_t smem_size = sizeof(SharedStorage);
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<
                typename Kernel_traits::Element, typename Kernel_traits::ElementAccum, typename Kernel_traits::index_t, Kernel_traits::kHeadDimV, kMaxSplits>;
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    static_assert(Headdim == 576);
    FLASH_ASSERT(params.d_v == 512);
    FLASH_ASSERT(params.k_ptr == params.v_ptr);  // Shared_KV
    if (params.kvcache_quantization_type == 0) {
        using Kernel_traits = Flash_fwd_kernel_traits_mla<576, 64, 64, 8, T, 512, true>;
        run_flash_splitkv_fwd_mla<Kernel_traits, flash::SharedStorageMLA<Kernel_traits>>(params, stream);
    } else {
        KVCACHE_QUANTIZATION_TYPE_SWITCH(params.kvcache_quantization_type, [&] {
            KVCACHE_QUANTIZATION_SPLIT_LENGTH_SWITCH(params.kvcache_quantization_split_length, [&] {
                using Kernel_traits = Flash_fwd_kernel_traits_mla<576, 64, 64, 8, T, 512, true, SplitLength, quant_type0, quant_type1>;
                run_flash_splitkv_fwd_mla<Kernel_traits, flash::SharedStorageMLA<Kernel_traits>>(params, stream);
            });
        });
    }
}

template<typename T>
void run_mha_fwd_splitkv_mha_128(Flash_fwd_mla_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.d == 128);
    FLASH_ASSERT(params.d_v == 128);
    FLASH_ASSERT(params.kvcache_quantization_type == 0);
    using Kernel_traits = Flash_fwd_kernel_traits_mla<128, 64, 128, 4, T, 128, false>;
    run_flash_splitkv_fwd_mla<Kernel_traits, flash::SharedStorageMHA<Kernel_traits>>(params, stream);
}

__global__ void __launch_bounds__(256, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const Mla_metadata_params params) {
    if (threadIdx.x > 0) return;

    int *seqlens_k_ptr = params.seqlens_k_ptr;
    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.batch_size;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    int total_num_blocks = 0;
    for (int i = 0; i < batch_size; ++i) {
        int num_blocks = cutlass::ceil_div(seqlens_k_ptr[i], block_size_n);
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
    }
    int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;

    int now_idx = 0, now_block = 0, now_n_split_idx = 0;
    num_splits_ptr[0] = 0;
    for (int i = 0; i < num_sm_parts; ++i) {
        tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 0] = now_idx;
        tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 1] = now_block * block_size_n;
        tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = now_n_split_idx;
        int remain_payload = payload;
        while (now_idx < batch_size) {
            int num_blocks = cutlass::ceil_div(seqlens_k_ptr[now_idx], block_size_n);
            int now_remain_blocks = num_blocks - now_block;
            if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                num_splits_ptr[now_idx + 1] = num_splits_ptr[now_idx] + (now_n_split_idx + 1);
                remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                ++now_idx;
                now_block = 0;
                now_n_split_idx = 0;
            } else {
                if (remain_payload - fixed_overhead_num_blocks > 0) {
                    now_block += remain_payload - fixed_overhead_num_blocks;
                    ++now_n_split_idx;
                    remain_payload = 0;
                }
                break;
            }
        }
        tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 2] = now_block > 0 ? now_idx : now_idx - 1;
        tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 3] = now_block > 0 ? now_block * block_size_n : seqlens_k_ptr[now_idx - 1];
    }
    assert(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
}

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream) {
    get_mla_metadata_kernel<<<256, 1, 0, stream>>>(params);
}
