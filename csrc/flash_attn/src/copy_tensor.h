#pragma once

#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#endif

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>

#include <cutlass/arch/barrier.h>

#include "cute/arch/cluster_sm90.hpp"

using namespace cute;

namespace flash {

__forceinline__ __device__ void barrierInit(uint64_t &tma_load_mbar, int numThreads, bool predict) {
    if (predict) {
        /// Initialize shared memory barrier
        tma_load_mbar = 0;
        cute::initialize_barrier(tma_load_mbar, numThreads);
    }
    __syncthreads();
    cutlass::arch::fence_barrier_init();
}

template<typename SrcEngine, typename SrcLayout, typename DstEngine,
        typename DstLayout, typename AtomX, class... ArgsX>
__forceinline__ __device__ void tma_copy(const Tensor<SrcEngine, SrcLayout> &gX,
                                         Tensor<DstEngine, DstLayout> &&sX,
                                         const TiledCopy<AtomX, ArgsX...> &tma_load_x,
                                         uint64_t &tma_load_mbar, uint16_t mcast_mask_a, bool predict) {
    using SrcType = typename AtomX::ValType;
// Set the bytes transferred in this TMX transaction (may involve multiple
// issues)
    constexpr int kTmaTransactionBytes =
            size(SrcLayout{}) * sizeof_bits_v<SrcType> / 8;

    if (predict) {
        cute::set_barrier_transaction_bytes(tma_load_mbar, kTmaTransactionBytes);
        copy(tma_load_x.with(tma_load_mbar, mcast_mask_a), gX, sX);
    }
}

} // namespace flash

