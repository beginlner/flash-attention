// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template void run_mha_fwd_fp8_splitkv_dispatch<cutlass::float_e4m3_t, 576>(Flash_fwd_params &params, cudaStream_t stream);
