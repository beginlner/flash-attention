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
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define PREC_SWITCH(PRECTYPE, NAME, ...)                                       \
  [&] {                                                                        \
    if (PRECTYPE == 1) {                                                       \
      using NAME = cutlass::bfloat16_t;                                        \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported PRECTYPE");                              \
    }                                                                          \
  }()

#define HEADDIM_SWITCH(HEADDIM, CONST_NAME, ...)                               \
  [&] {                                                                        \
    if (HEADDIM == 192) {                                                      \
      constexpr static int CONST_NAME = 192;                                   \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported HeadDim");                               \
    }                                                                          \
  }()

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
    } else {                                                                   \
      constexpr static int LOG_MAX_SPLITS = 7;                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()
