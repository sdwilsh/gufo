// SPDX-License-Identifier: MIT
// Block layouts and 32-element sub-block dequantisation for the routed-expert
// types of Ling-3.0-flash. Layouts and the IQ4_NL codebook derive from
// llama.cpp ggml-common.h / ggml-quants.c (MIT, The ggml authors); see
// VENDOR.md. Every type is addressed as super-blocks of 256 elements made of
// eight 32-element "units"; `Unit<T>::Dequant` expands one unit to floats.
#ifndef GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_MMQ_QUANT_UNITS_HIP_HPP_
#define GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_MMQ_QUANT_UNITS_HIP_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"

namespace gufo::models::ling_3_flash::rocm::detail {

#pragma pack(push, 1)
struct BlockQ4K {
  std::uint16_t d, dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
struct BlockQ5K {
  std::uint16_t d, dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
struct BlockQ6K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};
struct BlockIq4Xs {
  std::uint16_t d;
  std::uint16_t scales_h;
  std::uint8_t scales_l[4];
  std::uint8_t qs[128];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4K) == 144 && sizeof(BlockQ5K) == 176 &&
              sizeof(BlockQ6K) == 210 && sizeof(BlockIq4Xs) == 136);

__device__ __forceinline__ float HalfBits(std::uint16_t bits) {
  return static_cast<float>(__builtin_bit_cast(_Float16, bits));
}

__device__ __constant__ const std::int8_t kIq4Values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template<WeightType T>
struct Unit;

template<>
struct Unit<WeightType::kIQ4_XS> {
  using Block = BlockIq4Xs;
  __device__ static void Dequant(const Block& b, int s, float* out) {
    const int ls = ((b.scales_l[s >> 1] >> (4 * (s & 1))) & 0xF) |
                   (((b.scales_h >> (2 * s)) & 3) << 4);
    const float dl = HalfBits(b.d) * static_cast<float>(ls - 32);
    const std::uint8_t* q = b.qs + 16 * s;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      out[j] = dl * static_cast<float>(kIq4Values[q[j] & 0xF]);
      out[j + 16] = dl * static_cast<float>(kIq4Values[q[j] >> 4]);
    }
  }
};

__device__ __forceinline__ void ScaleMinK4(int j, const std::uint8_t* q,
                                           int* sc, int* m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

template<>
struct Unit<WeightType::kQ4_K> {
  using Block = BlockQ4K;
  __device__ static void Dequant(const Block& b, int s, float* out) {
    int sc, m;
    ScaleMinK4(s, b.scales, &sc, &m);
    const float d1 = HalfBits(b.d) * static_cast<float>(sc);
    const float m1 = HalfBits(b.dmin) * static_cast<float>(m);
    const std::uint8_t* q = b.qs + 32 * (s >> 1);
    const int shift = 4 * (s & 1);
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      out[l] = d1 * static_cast<float>((q[l] >> shift) & 0xF) - m1;
    }
  }
};

template<>
struct Unit<WeightType::kQ5_K> {
  using Block = BlockQ5K;
  __device__ static void Dequant(const Block& b, int s, float* out) {
    int sc, m;
    ScaleMinK4(s, b.scales, &sc, &m);
    const float d1 = HalfBits(b.d) * static_cast<float>(sc);
    const float m1 = HalfBits(b.dmin) * static_cast<float>(m);
    const std::uint8_t* q = b.qs + 32 * (s >> 1);
    const int shift = 4 * (s & 1);
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      const int hi = (b.qh[l] >> s) & 1;
      out[l] = d1 * static_cast<float>(((q[l] >> shift) & 0xF) + 16 * hi) - m1;
    }
  }
};

template<>
struct Unit<WeightType::kQ6_K> {
  using Block = BlockQ6K;
  __device__ static void Dequant(const Block& b, int s, float* out) {
    const int h = s >> 2;  // 128-element half
    const int j = s & 3;   // 32-element quarter inside the half
    const std::uint8_t* ql = b.ql + 64 * h + 32 * (j & 1);
    const std::uint8_t* qh = b.qh + 32 * h;
    const float d = HalfBits(b.d);
    const int lo_shift = 4 * (j >> 1);
    const int hi_shift = 2 * j;
    const float d0 = d * static_cast<float>(b.scales[8 * h + 2 * j]);
    const float d1 = d * static_cast<float>(b.scales[8 * h + 2 * j + 1]);
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      const int q =
          ((ql[l] >> lo_shift) & 0xF) | (((qh[l] >> hi_shift) & 3) << 4);
      out[l] = (l < 16 ? d0 : d1) * static_cast<float>(q - 32);
    }
  }
};

template<WeightType T>
inline constexpr std::size_t kBlockBytes = sizeof(typename Unit<T>::Block);

/// Bytes of one weight row of `k` columns (k % 256 == 0).
template<WeightType T>
constexpr std::size_t RowBytes(std::uint32_t k) {
  return (k / 256) * kBlockBytes<T>;
}

}  // namespace gufo::models::ling_3_flash::rocm::detail

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_MMQ_QUANT_UNITS_HIP_HPP_
