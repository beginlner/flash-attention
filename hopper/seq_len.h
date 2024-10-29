/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <array>
#include <algorithm>

#include <cutlass/cutlass.h>
#include <cute/layout.hpp>

namespace flash {

static constexpr int kMaxTileSize = 128;

template <bool UseVarSeqLen_, bool UseGQAPacking_> class SeqLenTraits {
public:
  static_assert(!(UseVarSeqLen_ && UseGQAPacking_),
    "Variable sequence length with GQA parallelization not implemented yet.");

  // Total number of queries / keys. Unpadded.
  int sum_s = 0;
  // seq len offsets.
  int *cu_seq_len = nullptr;
  // actual seq len array.
  int *seq_used = nullptr;
  // seq len of the current batch.
  int actual_seq_len = -1;

  // Whether this is for fixed-seq-len or var-seq-len.
  static constexpr bool UseVarSeqLen = UseVarSeqLen_;
  static constexpr bool UseGQAPacking = UseGQAPacking_;

  using ShapeT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int32_t, int32_t, int32_t>,
      std::conditional_t<
        UseGQAPacking,
        cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>,
        cute::Shape<int32_t, int32_t, int32_t, int32_t>
      >
  >;
  using StrideT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int64_t, _1, int64_t>, 
      std::conditional_t<
        UseGQAPacking,
        cute::Shape<int64_t, int64_t, _1, int64_t, int64_t>,
        cute::Shape<int64_t, _1, int64_t, int64_t>
      >
  >;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeLseT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int32_t, int32_t>, 
      cute::Shape<int32_t, int32_t, int32_t>
  >;
  using StrideLseT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int64_t, _1>, 
      cute::Shape<int64_t, int64_t, _1>
  >;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  // Not used for varseqlen
  using ShapeOAccumT = std::conditional_t<
    UseGQAPacking,
    cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>,
    cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>
  >;
  using StrideOAccumT = std::conditional_t<
    UseGQAPacking,
    cute::Shape<int64_t, int64_t, _1, int64_t, int64_t, int64_t>,
    cute::Shape<int64_t, _1, int64_t, int64_t, int64_t>
  >;
  using LayoutOAccumT = cute::Layout<ShapeOAccumT, StrideOAccumT>;

  using ShapeLseAccumT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideLseAccumT = cute::Shape<int64_t, int64_t, int64_t, _1>;
  using LayoutLseAccumT = cute::Layout<ShapeLseAccumT, StrideLseAccumT>;

  CUTLASS_HOST SeqLenTraits() {}

  CUTLASS_HOST SeqLenTraits(
      int sum_s, int max_seq_len, int *cu_seq_len = nullptr, int *seq_used = nullptr): 
      sum_s(sum_s), cu_seq_len(cu_seq_len), seq_used(seq_used), actual_seq_len(max_seq_len) {}

  CUTLASS_DEVICE void init(int bidb) {
    // TODO: add leftpad, seqlen_new for kv cache support
    if (seq_used) {
      actual_seq_len = seq_used[bidb];
    }
  }

  CUTLASS_DEVICE void init_no_guard(int bidb) {
    actual_seq_len = seq_used[bidb];
  }

  // Returns the layout of a tensor in MKHB format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  CUTLASS_HOST_DEVICE auto get_gmem_layout(
      int m, int k, int h, int b, 
      int64_t m_stride, int64_t h_stride, int64_t b_stride,
      bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    // static_assert(!UseGQAPacking, "Specialize default implementation for UseGQAPacking.");
    return make_layout(make_shape(m, k, h, b),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride));
  }

  // Returns the layout of a tensor in MKHB format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  // Overload that separates h into h_k and h/h_k.
  CUTLASS_HOST_DEVICE auto get_gmem_layout(
      int m, int k, int h_k, int b, int h_h_k_ratio,
      int64_t m_stride, int64_t h_stride, int64_t b_stride,
      bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    static_assert(!UseGQAPacking, "Specialize default implementation for UseGQAPacking.");
    return make_layout(make_shape(m, k, h_k * h_h_k_ratio, b),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride));    
  }

  // Returns the layout of a tensor in MKHBT format in global memory,
  // where T is number of splits.
  CUTLASS_HOST_DEVICE auto get_oaccum_gmem_layout(
      int m, int k, int h, int b, int num_splits,
      int64_t m_stride, int64_t h_stride, int64_t b_stride, int64_t split_stride,
      bool padded = false) const {
    return make_layout(make_shape(m, k, h, b, num_splits),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride, split_stride));
  }

  // Returns the layout of a tensor in MKHBT format in global memory,
  // where T is number of splits.
  // Overload that separates h into h_k and h/h_k.
  CUTLASS_HOST_DEVICE auto get_oaccum_gmem_layout(
      int m, int k, int h_k, int b, int h_h_k_ratio, int num_splits,
      int64_t m_stride, int64_t h_stride, int64_t b_stride, int64_t split_stride,
      bool padded = false) const {
    return make_layout(make_shape(m, k, h_k * h_h_k_ratio, b, num_splits),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride, split_stride));
  }

  // Returns the layout of lse tensor in BHM format in global memory.
  // padded: only useful for var-seq-len for dq_accum and softmax_d.
  CUTLASS_HOST_DEVICE auto get_lse_gmem_layout(
      int m, int h, int b, bool padded = false) const {
    static_assert(!UseVarSeqLen, "Specialize default implementation for VarSeqLen.");
    return make_layout(make_shape(b, h, m),
                       make_stride(int64_t(h * m), int64_t(m), cute::_1()));
  }

  // Returns the layout of lse tensor in TBHM format in global memory,
  // where T is number of splits.
  CUTLASS_HOST_DEVICE auto get_lseaccum_gmem_layout(
      int m, int h, int b, int num_splits, bool padded = false) const {
    return make_layout(make_shape(num_splits, b, h, m),
                       make_stride(int64_t(b * h * m), int64_t(h * m), int64_t(m), cute::_1()));
  }

  template <typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape, 
      int bidh, int bidb, bool padded = false) const {
    auto g_tensor = local_tile(
      m_tensor(_, _, bidh, bidb), tile_shape, make_coord(_, _0{}));
    return g_tensor;
  }

  template <bool Is_split, typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_lse_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape, 
      int bidh, int bidb, int n_split_idx, bool padded = false) const {
    // m_tensor has shape (B, H, M) or (splits, B, H, M)
    // Expect tile shape (bM)
    // Returns g_tensor of shape = (bM, ceil_div(M,bM))
    if constexpr(!Is_split) {
      auto g_tensor = local_tile(m_tensor(bidb, bidh, _), tile_shape, make_coord(_));
      return g_tensor;
    } else {
      auto g_tensor = local_tile(m_tensor(n_split_idx, bidb, bidh, _), tile_shape, make_coord(_));
      return g_tensor;
    }
  }

  template <bool Is_split, typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_o_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape,
      int bidh, int bidb, int split_idx, bool padded = false) const {
    // static_assert(!UseVarSeqLen, "Don't use get_o_local_tile_tensor with VarSeqLen.");
    // m_tensor has shape (M, K, H, B) or (M, K, H, B, splits)
    // Expect tile shape (bM, K)
    // Returns g_tensor of shape = (bM, K, ceil_div(M,bM))
    if constexpr(!Is_split) {
      auto g_tensor = local_tile(
        m_tensor(_, _, bidh, bidb), tile_shape, make_coord(_, _0{}));
      return g_tensor;
    } else {
      auto g_tensor = local_tile(
        m_tensor(_, _, bidh, bidb, split_idx), tile_shape, make_coord(_, _0{}));
      return g_tensor;
    }
  }
  
};

using FixedSeqLenTraits = SeqLenTraits<false, false>;

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
