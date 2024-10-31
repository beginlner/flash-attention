// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_APPEND_KV
#define APPEND_KV_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                           \
    TORCH_CHECK(!(COND));                         \
    constexpr static bool CONST_NAME = false;     \
    return __VA_ARGS__();                         \
  }()
#else
#define APPEND_KV_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_EVEN_MN
#define EVEN_MN_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                       \
    TORCH_CHECK(!(COND));                     \
    constexpr static bool CONST_NAME = false; \
    return __VA_ARGS__();                     \
  }()
#else
#define EVEN_MN_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_RETURN_SOFTMAX
#define RETURN_SOFTMAX_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                              \
    TORCH_CHECK(!(COND));                            \
    constexpr static bool CONST_NAME = false;        \
    return __VA_ARGS__();                            \
  }()
#else
#define RETURN_SOFTMAX_SWITCH BOOL_SWITCH
#endif

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define FP8_SWITCH(COND, ...) \
  [&] {                       \
    if (COND) {               \
      using elem_type = cutlass::float_e4m3_t; \
      return __VA_ARGS__();   \
    } else {                  \
      using elem_type = cutlass::float_e5m2_t; \
      return __VA_ARGS__();   \
    }                         \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = 32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kHeadDim = 96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 160) {           \
      constexpr static int kHeadDim = 160; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 192) {           \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 224) {           \
      constexpr static int kHeadDim = 224; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    } else if (HEADDIM == 512) {           \
      constexpr static int kHeadDim = 512; \
      return __VA_ARGS__();                \
    } else if (HEADDIM == 576) {           \
      constexpr static int kHeadDim = 576; \
      return __VA_ARGS__();                \
    } else {                               \
      TORCH_CHECK(                         \
        false, "Unsupported HeadDim");     \
    }                                      \
  }()

#define HEADDIMV_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM == 0) {                    \
      constexpr static int kHeadDimV = 0;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM == 64) {            \
      constexpr static int kHeadDimV = 64; \
      return __VA_ARGS__();                \
    } else if (HEADDIM == 128) {           \
      constexpr static int kHeadDimV = 128;\
      return __VA_ARGS__();                \
    } else {                               \
      TORCH_CHECK(                         \
        false, "Unsupported HeadDimV");    \
    }                                      \
  }()

#ifdef FLASHATTENTION_DISABLE_KVCACHE_QUANTIZATION
#define KVCACHE_QUANTIZATION_TYPE_SWITCH(TYPE, ...) \
  [&] {}
#else
#define KVCACHE_QUANTIZATION_TYPE_SWITCH(TYPE, ...) \
  [&] {                                             \
    if (TYPE == 4) {                                \
      using quant_type0 = int8_t;                   \
      using quant_type1 = cutlass::bfloat16_t;      \
      return __VA_ARGS__();                         \
    } else {                                        \
      TORCH_CHECK(                                  \
        false, "Unsupported TYPE");                 \
    }                                               \
  }()
#endif

#ifdef FLASHATTENTION_DISABLE_KVCACHE_QUANTIZATION
#define KVCACHE_QUANTIZATION_SPLIT_LENGTH_SWITCH(TYPE, ...)         \
  [&] {}
#else
#define KVCACHE_QUANTIZATION_SPLIT_LENGTH_SWITCH(SPLIT_LENGTH, ...) \
  [&] {                                                             \
    if (SPLIT_LENGTH == 512) {                                      \
      constexpr static int SplitLength = 512;                       \
      return __VA_ARGS__();                                         \
    } else {                                                        \
      TORCH_CHECK(                                                  \
        false, "Unsupported SplitLength");                          \
    }                                                               \
  }()
#endif

#define NUM_SPLITS_SWITCH(NUM_SPLITS, LOG_MAX_SPLITS, ...)                     \
  [&] {                                                                        \
    if (NUM_SPLITS <= 2) {                                                     \
      constexpr static int LOG_MAX_SPLITS = 1;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 4) {                                              \
      constexpr static int LOG_MAX_SPLITS = 2;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 8) {                                              \
      constexpr static int LOG_MAX_SPLITS = 3;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 16) {                                             \
      constexpr static int LOG_MAX_SPLITS = 4;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 32) {                                             \
      constexpr static int LOG_MAX_SPLITS = 5;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 64) {                                             \
      constexpr static int LOG_MAX_SPLITS = 6;                                 \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS <= 128) {                                            \
      constexpr static int LOG_MAX_SPLITS = 7;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      TORCH_CHECK(false, "Only support num_splits <= 128");                    \
    }                                                                          \
  }()
