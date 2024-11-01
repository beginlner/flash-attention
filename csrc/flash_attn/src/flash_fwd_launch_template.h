/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#endif

#include <ATen/cuda/CUDAContext.h>

#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_kernel.h"
#include "flash_fwd_mla_kernel.h"

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void __launch_bounds__(256, 1, 1) kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Split, Append_KV>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!Kernel_traits::Is_Q_in_regs, "sm90 implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "sm90 implementation does not support Share_Q_K_smem");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVEN_MN_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                RETURN_SOFTMAX_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        // Will only return softmax if dropout, to reduce compilation time.
                        // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                        // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                        // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                        // If Is_local, set Is_causal to false
                        auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, ReturnSoftmaxConst && Is_dropout>;
                        // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, false, true, true, false>;
                        // printf("IsEvenMNConst = %d, IsEvenKConst = %d, Is_local = %d, Is_causal = %d, ReturnSoftmaxConst = %d, Is_dropout = %d\n", int(IsEvenMNConst), int(IsEvenKConst), int(Is_local), int(Is_causal), int(ReturnSoftmaxConst), int(Is_dropout));
                        // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, true, true, false>;
                        if (smem_size >= 48 * 1024) {
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                        }
                        // int ctas_per_sm;
                        // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                        // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    TORCH_CHECK(!params.unpadded_lse);
    if (extra_stream_ptr == nullptr) extra_stream_ptr = std::make_shared<cudaStream_t>(at::cuda::getStreamFromPool(true).stream());
    auto stream1 = *extra_stream_ptr;
    size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        EVEN_MN_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                    APPEND_KV_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
                            // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                            // If Is_local, set Is_causal to false
                            if (params.num_splits > 1) {
                                // Launch the split kernel in another stream.
                                wait_stream(stream1, stream);
                                auto split_kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, true, Append_KV>;
                                smem_size = std::max(smem_size, size(typename Kernel_traits::SmemLayoutO{}) * sizeof(typename Kernel_traits::ElementAccum));
                                if (smem_size >= 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                split_kernel<<<dim3(num_m_block, params.num_splits, params.b * params.h), Kernel_traits::kNThreads, smem_size, stream1>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                            }
                            auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, false, Append_KV>;
                            if (smem_size >= 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                            kernel<<<dim3(num_m_block, 1, params.b * params.h), Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDimV % 128 == 0 ? 4 : (Kernel_traits::kHeadDimV % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            NUM_SPLITS_SWITCH(params.num_splits, kLogMaxSplits, [&] {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, kLogMaxSplits, IsEvenKConst><<<grid_combine, 128, 0, stream1>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
        wait_stream(stream, stream1);
    }
}

template<typename T, int Headdim>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    // Also for headdim 160 with block size 64 x 128 after the rotary addition.
    if constexpr (Headdim == 576) {
        TORCH_CHECK(params.d_v == 512);
        // Shared KV
        if (params.kvcache_quantization_type == 0) {
            run_flash_splitkv_fwd_mla<Flash_fwd_kernel_traits<576, 64, 64, 8, false, false, T, 512, true, 4>>(params, stream);
        } else {
            KVCACHE_QUANTIZATION_TYPE_SWITCH(params.kvcache_quantization_type, [&] {
                KVCACHE_QUANTIZATION_SPLIT_LENGTH_SWITCH(params.kvcache_quantization_split_length, [&] {
                    run_flash_splitkv_fwd_mla<Flash_fwd_kernel_traits<576, 64, 64, 8, false, false, T, 512, true, 4, true, SplitLength, quant_type0, quant_type1>>(params, stream);
                });
            });
        }
        return;
    } else {
        constexpr static int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : (Headdim <= 256 ? 64 : 32));
        if (params.block_table != nullptr) TORCH_CHECK(params.page_block_size % kBlockN == 0);
        run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>>(params, stream);
    }
}

template<typename T>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr(!Is_dropout) {
                // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
                // Using block size (64 x 256) is 27% slower for seqlen=2k
                // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    });
}

template<typename T>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // These two are always slower
            // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    HEADDIMV_SWITCH((params.d_v == params.d ? 0 : params.d_v), [&] {
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr(!Is_dropout) {
                // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
                // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
                if (is_sm8x) {
                    if constexpr(!Is_causal) {
                        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
                    } else {
                        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
                    }
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
                }
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // 1st ones are good for H100, A100
                // 2nd one is good for A6000 bc we get slightly better occupancy
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
                // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            }
        });
    });
    });
}

template<typename T>
void run_mha_fwd_hdim160(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // For A100, H100, 128 x 32 is the fastest.
            // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
            // and 128 x 64 with 8 warps is the fastest for non-causal.
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    HEADDIMV_SWITCH((params.d_v == params.d ? 0 : params.d_v), [&] {
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if constexpr(!Is_dropout) {
                if (is_sm90) {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
        });
    });
    });
}

template<typename T>
void run_mha_fwd_hdim224(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_block = %d\n", max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64)) {  // 112 KB
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // We can't do 128 x 32 with 8 warps because with headdim 224, kBlockKSmem = 32.
            // If we have N = 32, there are only 1024 elements to load at once, where each load
            // is 8 elements. This means we can only use 128 threads and not 256 threads.
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

template<typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            // For A100, we want to run with 128 x 64 (128KB smem).
            // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
            if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // 64 KB
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 96 KB
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    });
}

// fp8

template<typename T, int Headdim>
void run_mha_fwd_fp8_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    TORCH_CHECK(Headdim == 576);
    TORCH_CHECK(params.d_v == 512);
    TORCH_CHECK(params.kvcache_quantization_type == 0);
    run_flash_splitkv_fwd<Flash_fwd_fp8_kernel_traits<576, 64, 64, 8, false, false, T, cutlass::bfloat16_t, 512, true, 4>>(params, stream);
}

template<typename T>
void run_mha_fwd_fp8_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    HEADDIMV_SWITCH((params.d_v == params.d ? 0 : params.d_v), [&] {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 128, 8, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 64, 64, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 64, 128, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 64, 256, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 64, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 64, 8, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 128, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 256, 8, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 256, 64, 8, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 256, 128, 8, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
            });
        });
    });
}

template<typename T>
void run_mha_fwd_fp8_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    HEADDIMV_SWITCH((params.d_v == params.d ? 0 : params.d_v), [&] {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 128, 8, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 64, 128, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 64, 64, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 64, 256, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 64, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 64, 8, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 128, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 256, 4, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
//                run_flash_fwd<Flash_fwd_fp8_kernel_traits<Headdim, 128, 256, 8, false, false, T, cutlass::bfloat16_t, kHeadDimV>, Is_dropout, Is_causal>(params, stream);
            });
        });
    });
}
