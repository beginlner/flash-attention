// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_fp8_<cutlass::float_e4m3_t, cutlass::float_e4m3_t, 192>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_fp8_hdim192<cutlass::float_e4m3_t, cutlass::float_e4m3_t>(params, stream);
}

template<>
void run_mha_bwd_fp8_<cutlass::float_e4m3_t, cutlass::float_e5m2_t, 192>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_fp8_hdim192<cutlass::float_e4m3_t, cutlass::float_e5m2_t>(params, stream);
}
