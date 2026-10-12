// Ling-3.0-flash dense/linear kernels: Q8_0 / F16 / F32 projections, Q4_K
// embedding gather, RMSNorm, residual add and clamped SwiGLU. Wave32 (gfx11).
//
// Design summary
//  * GemvQ8_0 (decode, 1..8 tokens): activations are first quantised to two
//    int8 planes (value and residual) with one F32 scale per 32 values (a
//    single plane, llama.cpp's Q8_1 scheme, was too noisy for this model). Each
//    wave streams whole rows of the weights exactly once with dword loads (a
//    pair of 34-byte Q8_0 blocks is 17 aligned dwords), multiplies with
//    v_dot4_i32_iu8 against the int8 activations held in LDS, and applies the
//    two scales per block.
//  * GemmQ8_0 / GemmF16 (prefill): tiles of 16..128 tokens x 128 weight rows
//    (shape picked from n and rows), K step 32 through double-buffered LDS, F16
//    WMMA with F32 accumulation. Q8_0 codes are dequantised to F16 on the fly
//    while staging to LDS; activations are rounded to F16 once up front.
//    No activation scale quantisation.
//  * GemvF16 / GemvF32: one wave per row (RPW rows interleaved), x slabs of
//    1024 floats staged in LDS, F32 accumulation, 16-byte weight loads.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "src/models/ling_3_flash/kernels/rocm/act_quant.hip.hpp"
#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"

namespace gufo::models::ling_3_flash::rocm::dense {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr std::uint32_t kMaxGemvTokens = 8;
constexpr std::size_t kGemvLdsBytes = 60 * 1024;
constexpr int kSlab = 1024;  // floats of x staged per slab in F16/F32 GEMVs

constexpr std::size_t Align256(std::size_t v) {
  return (v + 255) & ~255ULL;
}

__device__ __forceinline__ float HalfBitsToFloat(std::uint32_t bits) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

__device__ __forceinline__ int Dot4(int a, int b, int c) {
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// ---------------------------------------------------------------------------
// Activation quantisation: x[t][k] -> q int8 [t][k], d F32 [t][k / 32].
// ---------------------------------------------------------------------------
// Two int8 planes: q1 = round(x / s), q2 = round((x - q1 s) * 127 / s) with
// s = absmax / 127, so x ~ s * (q1 + q2 / 127). One plane alone (llama.cpp's
// Q8_1) leaves ~0.3% rms noise per projection; for this model that noise,
// compounded over 42 layers, moved logit margins by up to ~1 nat. The second
// plane brings the activation error to ~3e-5 of the block maximum.
__global__ __launch_bounds__(256) void QuantizeActKernel(
    const float* __restrict__ x, std::int8_t* __restrict__ q,
    std::int8_t* __restrict__ q2, float* __restrict__ d, std::size_t total) {
  const std::size_t i = std::size_t{blockIdx.x} * 256 + threadIdx.x;
  const float v = i < total ? x[i] : 0.0F;
  float m = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor(m, off, 32));
  if (i < total) {
    QuantizeValue(v, m, &q[i], &q2[i]);
    if ((threadIdx.x & 31) == 0)
      d[i >> 5] = QuantScale(m);
  }
}

// The quantisation of QuantizeActKernel for one element whose wave32 holds 32
// consecutive elements of one row (so the xor-shuffle max is the block max).
// Shared by the fused producers below so their planes equal the standalone
// kernel's bit for bit.
__device__ __forceinline__ void QuantizeLane(float v, std::size_t i,
                                             std::int8_t* __restrict__ q,
                                             std::int8_t* __restrict__ q2,
                                             float* __restrict__ d) {
  float m = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor(m, off, 32));
  QuantizeValue(v, m, &q[i], &q2[i]);
  if ((threadIdx.x & 31) == 0)
    d[i >> 5] = QuantScale(m);
}

// ---------------------------------------------------------------------------
// F16 -> Q8_0 weight re-encoding (load time, MTP draft head).
// ---------------------------------------------------------------------------
// One thread per 32-value block: ggml's quantize_row_q8_0_ref (d = absmax /
// 127 in F32, id = 1 / d, q = roundf(x * id), d stored as F16). The 34-byte
// blocks are only 2-byte aligned, so the block is written as ushorts.
__global__ __launch_bounds__(256) void QuantizeF16ToQ8_0Kernel(
    const __half* __restrict__ w, std::uint16_t* __restrict__ out,
    std::uint64_t blocks) {
  const std::uint64_t b = std::uint64_t{blockIdx.x} * 256 + threadIdx.x;
  if (b >= blocks)
    return;
  const __half* src = w + b * 32;
  float v[32];
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    v[i] = __half2float(src[i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float d = amax / 127.0F;
  const float id = d != 0.0F ? 1.0F / d : 0.0F;
  std::uint16_t* dst = out + b * 17;
  dst[0] = __half_as_ushort(__float2half(d));
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const auto lo = static_cast<std::uint8_t>(
        static_cast<std::int8_t>(roundf(v[2 * i] * id)));
    const auto hi = static_cast<std::uint8_t>(
        static_cast<std::int8_t>(roundf(v[2 * i + 1] * id)));
    dst[1 + i] = static_cast<std::uint16_t>(lo | (hi << 8));
  }
}

// ---------------------------------------------------------------------------
// Q8_0 GEMV. y[t * rows + r]; k % 64 == 0.
// ---------------------------------------------------------------------------
template<int NT>
__device__ __forceinline__ void WaveReduceWrite(float (&acc)[NT], float* y,
                                                std::uint32_t row,
                                                std::uint32_t rows, bool live) {
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    float v = acc[t];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_xor(v, off, 32);
    if ((threadIdx.x & 31) == 0 && live)
      y[std::size_t{static_cast<unsigned>(t)} * rows + row] = v;
  }
}

// Up to three GEMVs over the same activations in one launch (grouped by block
// range): every row is computed exactly as in a launch of its own.
struct Q8Groups {
  const std::uint8_t* w[3];
  float* y[3];
  std::uint32_t rows[3];
  std::uint32_t first_block[3];  // first_block[0] == 0; unused: UINT32_MAX
};

template<int NT, int RPW>
__global__ __launch_bounds__(256) void GemvQ8Kernel(
    const Q8Groups groups, const std::int8_t* __restrict__ xq,
    const std::int8_t* __restrict__ xq2, const float* __restrict__ xd,
    std::uint32_t k) {
  const std::uint32_t gi = blockIdx.x >= groups.first_block[2]   ? 2U
                           : blockIdx.x >= groups.first_block[1] ? 1U
                                                                 : 0U;
  const std::uint8_t* __restrict__ w = groups.w[gi];
  float* __restrict__ y = groups.y[gi];
  const std::uint32_t rows = groups.rows[gi];
  const std::uint32_t block_in_group = blockIdx.x - groups.first_block[gi];
  extern __shared__ __attribute__((aligned(16))) std::uint8_t smem[];
  std::int8_t* s_q = reinterpret_cast<std::int8_t*>(smem);
  std::int8_t* s_q2 =
      reinterpret_cast<std::int8_t*>(smem + std::size_t{NT} * k);
  float* s_d = reinterpret_cast<float*>(smem + std::size_t{2 * NT} * k);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  {
    const auto* src = reinterpret_cast<const uint4*>(xq);
    auto* dst = reinterpret_cast<uint4*>(s_q);
    const auto* src2 = reinterpret_cast<const uint4*>(xq2);
    auto* dst2 = reinterpret_cast<uint4*>(s_q2);
    for (std::uint32_t i = tid; i < NT * k / 16; i += 256) {
      dst[i] = src[i];
      dst2[i] = src2[i];
    }
    const std::uint32_t nd = NT * (k / 32);
    for (std::uint32_t i = tid; i < nd; i += 256)
      s_d[i] = xd[i];
  }
  __syncthreads();

  const std::uint32_t npairs = k / 64;
  const std::size_t row_bytes = std::size_t{npairs} * 68;
  const std::uint32_t row0 = (block_in_group * 8 + wave) * RPW;
  const std::uint8_t* wrow[RPW];
#pragma unroll
  for (int j = 0; j < RPW; ++j) {
    const std::uint32_t r = min(row0 + j, rows - 1);
    wrow[j] = w + std::size_t{r} * row_bytes;
  }
  float acc[RPW][NT];
#pragma unroll
  for (int j = 0; j < RPW; ++j)
#pragma unroll
    for (int t = 0; t < NT; ++t)
      acc[j][t] = 0.0F;

  for (std::uint32_t p = lane; p < npairs; p += 32) {
    std::uint32_t u[RPW][17];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
      const auto* wp =
          reinterpret_cast<const std::uint32_t*>(wrow[j] + std::size_t{p} * 68);
#pragma unroll
      for (int i = 0; i < 17; ++i)
        u[j][i] = wp[i];
    }
    float d0[RPW], d1[RPW];
    int qa[RPW][8];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
      d0[j] = HalfBitsToFloat(u[j][0] & 0xFFFFU);
      d1[j] = HalfBitsToFloat(u[j][8] >> 16);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        qa[j][i] = static_cast<int>(
            __builtin_amdgcn_alignbit(u[j][i + 1], u[j][i], 16));
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const auto* xp = reinterpret_cast<const uint4*>(
          s_q + std::size_t{static_cast<unsigned>(t)} * k +
          std::size_t{p} * 64);
      const uint4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
      const int xa[8] = {static_cast<int>(x0.x), static_cast<int>(x0.y),
                         static_cast<int>(x0.z), static_cast<int>(x0.w),
                         static_cast<int>(x1.x), static_cast<int>(x1.y),
                         static_cast<int>(x1.z), static_cast<int>(x1.w)};
      const int xb[8] = {static_cast<int>(x2.x), static_cast<int>(x2.y),
                         static_cast<int>(x2.z), static_cast<int>(x2.w),
                         static_cast<int>(x3.x), static_cast<int>(x3.y),
                         static_cast<int>(x3.z), static_cast<int>(x3.w)};
      const auto* xp2 = reinterpret_cast<const uint4*>(
          s_q2 + std::size_t{static_cast<unsigned>(t)} * k +
          std::size_t{p} * 64);
      const uint4 y0 = xp2[0], y1 = xp2[1], y2 = xp2[2], y3 = xp2[3];
      const int ya[8] = {static_cast<int>(y0.x), static_cast<int>(y0.y),
                         static_cast<int>(y0.z), static_cast<int>(y0.w),
                         static_cast<int>(y1.x), static_cast<int>(y1.y),
                         static_cast<int>(y1.z), static_cast<int>(y1.w)};
      const int yb[8] = {static_cast<int>(y2.x), static_cast<int>(y2.y),
                         static_cast<int>(y2.z), static_cast<int>(y2.w),
                         static_cast<int>(y3.x), static_cast<int>(y3.y),
                         static_cast<int>(y3.z), static_cast<int>(y3.w)};
      const float sx0 = s_d[t * (k / 32) + 2 * p];
      const float sx1 = s_d[t * (k / 32) + 2 * p + 1];
#pragma unroll
      for (int j = 0; j < RPW; ++j) {
        int s0 = 0, s1 = 0, r0 = 0, r1 = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s0 = Dot4(qa[j][i], xa[i], s0);
          s1 = Dot4(static_cast<int>(u[j][9 + i]), xb[i], s1);
          r0 = Dot4(qa[j][i], ya[i], r0);
          r1 = Dot4(static_cast<int>(u[j][9 + i]), yb[i], r1);
        }
        constexpr float kRes = 1.0F / 127.0F;
        acc[j][t] =
            fmaf(d0[j] * sx0,
                 fmaf(static_cast<float>(r0), kRes, static_cast<float>(s0)),
                 acc[j][t]);
        acc[j][t] =
            fmaf(d1[j] * sx1,
                 fmaf(static_cast<float>(r1), kRes, static_cast<float>(s1)),
                 acc[j][t]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < RPW; ++j) {
    const std::uint32_t r = row0 + j;
    WaveReduceWrite<NT>(acc[j], y, r, rows, r < rows);
  }
}

// ---------------------------------------------------------------------------
// F16 / F32 GEMV with x slabs in LDS. y[t * rows + r]; k % 8 == 0.
// ---------------------------------------------------------------------------
template<int NT, int RPW, bool kHalf>
__global__ __launch_bounds__(256) void GemvDenseKernel(
    const void* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, std::uint32_t k, std::uint32_t rows) {
  constexpr int kE = kHalf ? 8 : 4;  // weights per 16-byte load
  constexpr std::size_t kElemBytes = kHalf ? 2 : 4;
  __shared__ __attribute__((aligned(16))) float xs[NT][kSlab];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const std::uint32_t row0 = (blockIdx.x * 8 + wave) * RPW;
  const std::uint8_t* wrow[RPW];
#pragma unroll
  for (int j = 0; j < RPW; ++j) {
    const std::uint32_t r = min(row0 + j, rows - 1);
    wrow[j] =
        static_cast<const std::uint8_t*>(w) + std::size_t{r} * k * kElemBytes;
  }
  float acc[RPW][NT];
#pragma unroll
  for (int j = 0; j < RPW; ++j)
#pragma unroll
    for (int t = 0; t < NT; ++t)
      acc[j][t] = 0.0F;

  for (std::uint32_t s0 = 0; s0 < k; s0 += kSlab) {
    const std::uint32_t slab = min(static_cast<std::uint32_t>(kSlab), k - s0);
    __syncthreads();
    const std::uint32_t q4 = slab / 4;
    for (std::uint32_t i = tid; i < NT * q4; i += 256) {
      const std::uint32_t t = i / q4;
      const std::uint32_t c = i % q4;
      reinterpret_cast<float4*>(xs[t])[c] =
          reinterpret_cast<const float4*>(x + std::size_t{t} * k + s0)[c];
    }
    __syncthreads();
    const std::uint32_t nchunks = slab / kE;
    const auto step = [&](std::uint32_t c) {
      float wf[RPW][kE];
#pragma unroll
      for (int j = 0; j < RPW; ++j) {
        const uint4 v = *reinterpret_cast<const uint4*>(
            wrow[j] + (std::size_t{s0} + std::size_t{c} * kE) * kElemBytes);
        if constexpr (kHalf) {
          const std::uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            wf[j][2 * i] = HalfBitsToFloat(words[i] & 0xFFFFU);
            wf[j][2 * i + 1] = HalfBitsToFloat(words[i] >> 16);
          }
        } else {
          wf[j][0] = __uint_as_float(v.x);
          wf[j][1] = __uint_as_float(v.y);
          wf[j][2] = __uint_as_float(v.z);
          wf[j][3] = __uint_as_float(v.w);
        }
      }
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        float xv[kE];
#pragma unroll
        for (int i = 0; i < kE / 4; ++i) {
          const float4 f =
              reinterpret_cast<const float4*>(xs[t])[c * (kE / 4) + i];
          xv[4 * i] = f.x;
          xv[4 * i + 1] = f.y;
          xv[4 * i + 2] = f.z;
          xv[4 * i + 3] = f.w;
        }
#pragma unroll
        for (int j = 0; j < RPW; ++j)
#pragma unroll
          for (int i = 0; i < kE; ++i)
            acc[j][t] = fmaf(wf[j][i], xv[i], acc[j][t]);
      }
    };
    if constexpr (kHalf) {
      for (std::uint32_t c = lane; c < nchunks; c += 32) {
        step(c);
      }
    } else {
      // F32 (router): several 16-byte weight loads in flight per lane.
#pragma unroll 4
      for (std::uint32_t c = lane; c < nchunks; c += 32) {
        step(c);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < RPW; ++j) {
    const std::uint32_t r = row0 + j;
    WaveReduceWrite<NT>(acc[j], y, r, rows, r < rows);
  }
}

// ---------------------------------------------------------------------------
// WMMA GEMM over Q8_0 or F16 weights, F32 activations -> F16 -> F32 accum.
// Activations are the WMMA A operand and the weights the B operand so each
// lane writes consecutive output rows. A block of WT x WR waves computes a
// (16*TF*WT tokens) x (16*RF*WR weight rows) tile; each wave owns TF x RF
// 16x16 accumulators. K advances in steps of 32 through a double-buffered LDS
// stage: the global loads of step kb + 1 are issued before the WMMAs of step
// kb, converted into the other buffer afterwards, and one barrier per step
// separates the buffers. Every output element is one accumulator chain over
// consecutive 16-wide K slices whatever the tile shape, so results depend on
// neither n nor the tile configuration (the dispatch below picks the shape
// from n, rows alone).
// ---------------------------------------------------------------------------
constexpr int kLds = 40;  // halves per LDS row (32 + 8 pad)

/// xh = F16(x), four values per thread.
__global__ __launch_bounds__(256) void ConvertActKernel(
    const float* __restrict__ x, __half* __restrict__ xh, std::size_t units) {
  const std::size_t i = std::size_t{blockIdx.x} * 256 + threadIdx.x;
  if (i >= units)
    return;
  const float4 v = reinterpret_cast<const float4*>(x)[i];
  __half h[4] = {__float2half(v.x), __float2half(v.y), __float2half(v.z),
                 __float2half(v.w)};
  __builtin_memcpy(reinterpret_cast<std::uint8_t*>(xh) + i * 8, h, 8);
}

template<int TF, int RF, int WT, int WR>
struct GemmShape {
  static constexpr int kBN = 16 * TF * WT;  // tokens per block
  static constexpr int kBM = 16 * RF * WR;  // weight rows per block
  static constexpr int kThreads = 32 * WT * WR;
};

template<bool kQ8, bool kXH, int TF, int RF, int WT, int WR>
__global__ __launch_bounds__(32 * WT * WR) void GemmKernel(
    const std::uint8_t* __restrict__ w, const void* __restrict__ x,
    float* __restrict__ y, std::uint32_t n_tokens, std::uint32_t k,
    std::uint32_t rows, std::uint32_t ldy) {
  using S = GemmShape<TF, RF, WT, WR>;
  constexpr int kBN = S::kBN;
  constexpr int kBM = S::kBM;
  constexpr int kNT = S::kThreads;
  constexpr int kWPT = kBM * 4 / kNT;  // 8-weight units per thread
  static_assert(kBN * 8 % kNT == 0 && kBM * 4 % kNT == 0);
  __shared__ __attribute__((aligned(16))) __half s_x[2][kBN][kLds];
  __shared__ __attribute__((aligned(16))) __half s_w[2][kBM][kLds];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sub = lane & 15;
  const int half_id = lane >> 4;
  const int wt = wave % WT;  // token slice of the tile
  const int wr = wave / WT;  // row slice of the tile
  const std::uint32_t t0 = blockIdx.x * kBN;
  const std::uint32_t r0 = blockIdx.y * kBM;
  const std::size_t w_row_bytes =
      kQ8 ? std::size_t{k / 32} * 34 : std::size_t{k} * 2;

  // Staging roles: x unit = (token, 4 consecutive k); w unit = (row, 8 k).
  // F32 activations (kXH false): float4 units; F16 activations (kXH true,
  // converted once by ConvertActKernel): 8-half units, 4 per row of a stage.
  constexpr int kXU = kXH ? kBN * 4 : kBN * 8;
  constexpr int kXPT = (kXU + kNT - 1) / kNT;
  const std::uint8_t* xp[kXPT];
  bool x_live[kXPT];
  int x_row[kXPT], x_c4[kXPT];
#pragma unroll
  for (int j = 0; j < kXPT; ++j) {
    const int idx = tid + j * kNT;
    x_row[j] = kXH ? idx >> 2 : idx >> 3;
    x_c4[j] = kXH ? idx & 3 : idx & 7;  // unit within the row's 32 k values
    const std::uint32_t tok = t0 + x_row[j];
    x_live[j] = tok < n_tokens && idx < kXU;
    xp[j] = static_cast<const std::uint8_t*>(x) +
            (std::size_t{x_live[j] ? tok : 0} * k + x_c4[j] * (kXH ? 8 : 4)) *
                (kXH ? 2 : 4);
  }
  const std::uint8_t* wp[kWPT];
  bool w_live[kWPT];
  int w_row[kWPT], w_q[kWPT];
#pragma unroll
  for (int j = 0; j < kWPT; ++j) {
    const int idx = tid + j * kNT;
    w_row[j] = idx >> 2;
    w_q[j] = idx & 3;
    const std::uint32_t row = r0 + w_row[j];
    w_live[j] = row < rows;
    wp[j] = w + std::size_t{w_live[j] ? row : 0} * w_row_bytes;
  }

  v8f acc[TF][RF];
#pragma unroll
  for (int i = 0; i < TF; ++i)
#pragma unroll
    for (int j = 0; j < RF; ++j)
      acc[i][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

  std::conditional_t<kXH, uint4, float4> xr[kXPT];
  uint4 wr_regs[kWPT];  // F16: 8 halves; Q8: 8 int8 in .x/.y
  std::uint32_t scale_bits[kWPT];
  const std::uint32_t nkb = k / 32;

  const auto fetch = [&](std::uint32_t kb) {
#pragma unroll
    for (int j = 0; j < kXPT; ++j) {
      using XT = std::conditional_t<kXH, uint4, float4>;
      xr[j] = x_live[j] ? *reinterpret_cast<const XT*>(
                              xp[j] + std::size_t{kb} * 32 * (kXH ? 2 : 4))
                        : XT{};
    }
#pragma unroll
    for (int j = 0; j < kWPT; ++j) {
      if constexpr (kQ8) {
        const auto* blk = reinterpret_cast<const std::uint16_t*>(
            wp[j] + std::size_t{kb} * 34);
        std::uint16_t h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          h[i] = w_live[j] ? blk[1 + w_q[j] * 4 + i] : 0;
        scale_bits[j] = w_live[j] ? blk[0] : 0;
        wr_regs[j] = make_uint4(h[0] | (std::uint32_t{h[1]} << 16),
                                h[2] | (std::uint32_t{h[3]} << 16), 0, 0);
      } else {
        const auto* src = reinterpret_cast<const uint4*>(
            wp[j] + std::size_t{kb} * 64 + w_q[j] * 16);
        wr_regs[j] = w_live[j] ? *src : make_uint4(0, 0, 0, 0);
      }
    }
  };
  const auto commit = [&](int buf) {
#pragma unroll
    for (int j = 0; j < kWPT; ++j) {
      auto* dw = reinterpret_cast<uint4*>(&s_w[buf][w_row[j]][w_q[j] * 8]);
      if constexpr (kQ8) {
        // d * q for int8 q, as one convert and one fused multiply-add per
        // value on the offset-binary byte (q ^ 0x80 = q + 128): the product
        // of the 11-bit scale and an 8-bit value is exact in F32, so this is
        // exactly d * q.
        const float d = HalfBitsToFloat(scale_bits[j]);
        const float bias = -128.0F * d;
        const std::uint32_t u[2] = {wr_regs[j].x ^ 0x80808080U,
                                    wr_regs[j].y ^ 0x80808080U};
        __half h[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
          h[i] = __float2half(__builtin_fmaf(
              static_cast<float>(
                  static_cast<std::uint8_t>(u[i >> 2] >> (8 * (i & 3)))),
              d, bias));
        __builtin_memcpy(dw, h, 16);
      } else {
        *dw = wr_regs[j];
      }
    }
#pragma unroll
    for (int j = 0; j < kXPT; ++j) {
      if (!(kXPT * kNT == kXU || tid + j * kNT < kXU)) {
        // idle staging slot (fewer units than threads)
      } else if constexpr (kXH) {
        __builtin_memcpy(&s_x[buf][x_row[j]][x_c4[j] * 8], &xr[j], 16);
      } else {
        __half hx[4] = {__float2half(xr[j].x), __float2half(xr[j].y),
                        __float2half(xr[j].z), __float2half(xr[j].w)};
        __builtin_memcpy(&s_x[buf][x_row[j]][x_c4[j] * 4], hx, 8);
      }
    }
  };

  fetch(0);
  commit(0);
  __syncthreads();
  for (std::uint32_t kb = 0; kb < nkb; ++kb) {
    const int buf = static_cast<int>(kb & 1);
    const bool more = kb + 1 < nkb;
    if (more)
      fetch(kb + 1);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      v16h a[TF], b[RF];
#pragma unroll
      for (int i = 0; i < TF; ++i)
        __builtin_memcpy(&a[i], &s_x[buf][(wt * TF + i) * 16 + sub][ks * 16],
                         32);
#pragma unroll
      for (int j = 0; j < RF; ++j)
        __builtin_memcpy(&b[j], &s_w[buf][(wr * RF + j) * 16 + sub][ks * 16],
                         32);
#pragma unroll
      for (int i = 0; i < TF; ++i)
#pragma unroll
        for (int j = 0; j < RF; ++j)
          acc[i][j] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
    }
    if (more)
      commit(buf ^ 1);
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TF; ++i)
#pragma unroll
    for (int j = 0; j < RF; ++j) {
      const std::uint32_t row = r0 + (wr * RF + j) * 16 + sub;
      if (row >= rows)
        continue;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const std::uint32_t tok = t0 + (wt * TF + i) * 16 + 2 * e + half_id;
        if (tok < n_tokens)
          y[std::size_t{tok} * ldy + row] = acc[i][j][e];
      }
    }
}

// ---------------------------------------------------------------------------
// Q4_K embedding gather.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void EmbedQ4KKernel(
    const std::uint8_t* __restrict__ table,
    const std::int32_t* __restrict__ tokens, float* __restrict__ out,
    std::uint32_t hidden) {
  const std::uint32_t t = blockIdx.x;
  const std::int32_t id = tokens[t];
  const std::uint32_t nsb = hidden / 256;
  float* dst = out + std::size_t{t} * hidden;
  if (id < 0) {
    for (std::uint32_t i = threadIdx.x; i < hidden; i += 256)
      dst[i] = 0.0F;
    return;
  }
  const std::uint8_t* row =
      table + std::size_t{static_cast<std::uint32_t>(id)} * nsb * 144;
  const int e = threadIdx.x;
  const int j = e >> 5;  // sub-block 0..7
  const int l = e & 31;
  for (std::uint32_t sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* blk = row + std::size_t{sb} * 144;
    const float d = HalfBitsToFloat(blk[0] | (blk[1] << 8));
    const float dmin = HalfBitsToFloat(blk[2] | (blk[3] << 8));
    const std::uint8_t* sc = blk + 4;
    int s, m;
    if (j < 4) {
      s = sc[j] & 63;
      m = sc[j + 4] & 63;
    } else {
      s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
      m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    const int byte = blk[16 + 32 * (j >> 1) + l];
    const int nib = (j & 1) ? (byte >> 4) : (byte & 0xF);
    dst[sb * 256 + e] = d * static_cast<float>(s) * static_cast<float>(nib) -
                        dmin * static_cast<float>(m);
  }
}

// ---------------------------------------------------------------------------
// RMSNorm over groups, residual add, clamped SwiGLU.
// ---------------------------------------------------------------------------
__global__ void RmsNormRowsKernel(const float* x, const float* gamma,
                                  float* out, std::uint32_t dim,
                                  std::uint32_t groups, std::uint32_t d,
                                  float eps) {
  __shared__ float partial[8];
  const std::uint32_t b = blockIdx.x;
  const std::size_t base =
      std::size_t{b / groups} * dim + std::size_t{b % groups} * d;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = x[base + i];
    ss = fmaf(v, v, ss);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    ss += __shfl_xor(ss, off, 32);
  const int wave = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0)
    partial[wave] = ss;
  __syncthreads();
  float total = 0.0F;
  for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i)
    total += partial[i];
  const float inv = rsqrtf(total / static_cast<float>(d) + eps);
  for (std::uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = x[base + i] * inv;
    out[base + i] = gamma != nullptr ? v * gamma[i] : v;
  }
}

__global__ void AddResidualKernel(float* res, const float* x,
                                  std::size_t count) {
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < count; i += stride)
    res[i] += x[i];
}

__global__ void SwiGluClampedKernel(const float* gate, const float* up,
                                    float* out, std::size_t count,
                                    float limit) {
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const float g = gate[i];
    float a = g / (1.0F + expf(-g));
    float u = up[i];
    if (limit > 0.0F) {
      a = fminf(a, limit);
      u = fminf(fmaxf(u, -limit), limit);
    }
    out[i] = a * u;
  }
}

// ---------------------------------------------------------------------------
// Decode-path fusions (n <= 8 rows of kHidden): residual update + RMSNorm +
// activation quantisation in one launch. Every value goes through the same
// expressions, in the same order, as the separate kernels (AddResidual,
// moe Combine, RmsNormRows, QuantizeAct), so the results are bit-identical.
// ---------------------------------------------------------------------------
constexpr std::uint32_t kNormThreads = 256;
constexpr std::uint32_t kNormPer = kHidden / kNormThreads;  // 10
static_assert(kHidden % kNormThreads == 0);

// kMode 0: x = res; 1: res += addend; 2: res = (res + shared) + sum_k w[k] *
// expert_out[k] (moe Combine); out = rmsnorm(res) * gamma; planes of out.
template<int kMode>
__global__ __launch_bounds__(kNormThreads) void NormQuantKernel(
    float* res, const float* __restrict__ addend,
    const float* __restrict__ expert_out, const float* __restrict__ weights,
    const float* __restrict__ shared, const float* __restrict__ gamma,
    float* __restrict__ out, std::int8_t* __restrict__ q,
    std::int8_t* __restrict__ q2, float* __restrict__ dq, float eps) {
  __shared__ float partial[kNormThreads / 32];
  const std::size_t row = blockIdx.x;
  const std::size_t base = row * kHidden;
  float vals[kNormPer];
  float ss = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kNormPer; ++j) {
    const std::uint32_t i = threadIdx.x + j * kNormThreads;
    float v;
    if constexpr (kMode == 0) {
      v = res[base + i];
    } else if constexpr (kMode == 1) {
      v = res[base + i];
      v += addend[base + i];
      res[base + i] = v;
    } else {
      float acc = res[base + i];
      if (shared != nullptr)
        acc += shared[base + i];
      float sum = 0.0F;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float w = weights[row * 8 + k];
        const float e = expert_out[(row * 8 + k) * kHidden + i];
        sum = fmaf(w, e, sum);
      }
      acc += sum;
      res[base + i] = acc;
      v = acc;
    }
    vals[j] = v;
    ss = fmaf(v, v, ss);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    ss += __shfl_xor(ss, off, 32);
  const int wave = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0)
    partial[wave] = ss;
  __syncthreads();
  float total = 0.0F;
  for (int i = 0; i < static_cast<int>(kNormThreads >> 5); ++i)
    total += partial[i];
  const float inv = rsqrtf(total / static_cast<float>(kHidden) + eps);
#pragma unroll
  for (std::uint32_t j = 0; j < kNormPer; ++j) {
    const std::uint32_t i = threadIdx.x + j * kNormThreads;
    const float v = vals[j] * inv;
    const float y = v * gamma[i];
    out[base + i] = y;
    QuantizeLane(y, base + i, q, q2, dq);
  }
}

// out = clamped swiglu(gate, up); planes of out (thread per element; count is
// a multiple of 32).
__global__ __launch_bounds__(256) void SwiGluQuantKernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ out, std::int8_t* __restrict__ q,
    std::int8_t* __restrict__ q2, float* __restrict__ dq, std::size_t count,
    float limit) {
  const std::size_t i = std::size_t{blockIdx.x} * 256 + threadIdx.x;
  float r = 0.0F;
  if (i < count) {
    const float g = gate[i];
    float a = g / (1.0F + expf(-g));
    float u = up[i];
    if (limit > 0.0F) {
      a = fminf(a, limit);
      u = fminf(fmaxf(u, -limit), limit);
    }
    r = a * u;
    out[i] = r;
  }
  // Quantisation waves are whole (count % 32 == 0), so every lane of a wave
  // that holds live elements is live.
  if (i < count)
    QuantizeLane(r, i, q, q2, dq);
}

// ---------------------------------------------------------------------------
// Host helpers.
// ---------------------------------------------------------------------------

#define LING_DISPATCH_NT(n, CALL) \
  switch (n) {                    \
    case 1: {                     \
      constexpr int NT = 1;       \
      CALL;                       \
      break;                      \
    }                             \
    case 2: {                     \
      constexpr int NT = 2;       \
      CALL;                       \
      break;                      \
    }                             \
    case 3: {                     \
      constexpr int NT = 3;       \
      CALL;                       \
      break;                      \
    }                             \
    case 4: {                     \
      constexpr int NT = 4;       \
      CALL;                       \
      break;                      \
    }                             \
    case 5: {                     \
      constexpr int NT = 5;       \
      CALL;                       \
      break;                      \
    }                             \
    case 6: {                     \
      constexpr int NT = 6;       \
      CALL;                       \
      break;                      \
    }                             \
    case 7: {                     \
      constexpr int NT = 7;       \
      CALL;                       \
      break;                      \
    }                             \
    default: {                    \
      constexpr int NT = 8;       \
      CALL;                       \
      break;                      \
    }                             \
  }

template<bool kQ8, bool kXH, int TF, int RF, int WT, int WR>
void LaunchGemmShape(const std::uint8_t* w, const void* x, float* y,
                     std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t rows, std::uint32_t ldy,
                     hipStream_t stream) {
  using S = GemmShape<TF, RF, WT, WR>;
  const dim3 grid((n_tokens + S::kBN - 1) / S::kBN,
                  (rows + S::kBM - 1) / S::kBM);
  GemmKernel<kQ8, kXH, TF, RF, WT, WR>
      <<<grid, S::kThreads, 0, stream>>>(w, x, y, n_tokens, k, rows, ldy);
}

/// Work-group processors (two CUs each; HIP counts WGPs as multiprocessors on
/// gfx11) of the current device.
int DeviceWgps() {
  static const int wgps = [] {
    int dev = 0, count = 0;
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetAttribute(&count, hipDeviceAttributeMultiprocessorCount,
                              dev) != hipSuccess ||
        count < 1)
      return 20;  // gfx1151
    return count;
  }();
  return wgps;
}

/// Tile shape from (n, rows) and the device only; the arithmetic per output
/// element is the same for every shape (see GemmKernel). The 128 x 128 tile
/// amortises the staging work best but only has ~3 resident blocks per WGP;
/// the 64 x 64 tile has ~6 and runs at ~0.7 of its rate per FLOP (fitted to
/// the gfx1151 per-shape table of WS13 round 1). The cost is the padded
/// tile area of all blocks, with at least one full round of resident blocks
/// (a smaller grid leaves the GPU partly idle).
int PickGemmShape(std::uint32_t n_tokens, std::uint32_t rows) {
  if (n_tokens <= 16)
    return 3;
  if (n_tokens <= 32)
    return 2;
  if (n_tokens <= 64)
    return 1;
  const double wgps = DeviceWgps();
  const auto cost = [&](int bn, int bm, double per_wgp, double eff) {
    const double blocks = std::ceil(n_tokens / static_cast<double>(bn)) *
                          std::ceil(rows / static_cast<double>(bm));
    return std::max(blocks, wgps * per_wgp) * bn * bm / eff;
  };
  return cost(128, 128, 3, 1.0) <= cost(64, 64, 6, 0.7) ? 0 : 1;
}

template<bool kQ8, bool kXH>
void LaunchGemmAs(int shape, const std::uint8_t* w, const void* x, float* y,
                  std::uint32_t n, std::uint32_t k, std::uint32_t rows,
                  std::uint32_t ldy, hipStream_t stream) {
  static_assert(kGemmShapes == 4, "update the shape switch");
  if (shape < 0 || shape >= kGemmShapes)
    throw std::invalid_argument("ling dense: invalid GEMM tile shape");
  switch (shape) {
    case 0:  // 128 tokens x 128 rows, 8 waves of 64 x 32
      LaunchGemmShape<kQ8, kXH, 4, 2, 2, 4>(w, x, y, n, k, rows, ldy, stream);
      break;
    case 1:  // 64 x 64, 4 waves of 32 x 32
      LaunchGemmShape<kQ8, kXH, 2, 2, 2, 2>(w, x, y, n, k, rows, ldy, stream);
      break;
    case 2:  // 32 x 128, 4 waves of 32 x 32
      LaunchGemmShape<kQ8, kXH, 2, 2, 1, 4>(w, x, y, n, k, rows, ldy, stream);
      break;
    default:  // 3: 16 x 128, 4 waves of 16 x 32
      LaunchGemmShape<kQ8, kXH, 1, 2, 1, 4>(w, x, y, n, k, rows, ldy, stream);
      break;
  }
}

/// x_half: scratch for the activations rounded to F16 (n_tokens * k halves,
/// converted once instead of by every row block); null = convert in the
/// kernel.
void LaunchGemm(bool q8, const void* w, const float* x, float* y,
                std::uint32_t n_tokens, std::uint32_t k, std::uint32_t rows,
                std::uint32_t ldy, int shape, void* x_half,
                hipStream_t stream) {
  if (n_tokens == 0 || rows == 0)
    return;
  if (shape < 0)
    shape = PickGemmShape(n_tokens, rows);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  if (x_half != nullptr) {
    const std::size_t total = std::size_t{n_tokens} * k;
    const std::size_t units = total / 4;  // k % 32 == 0
    ConvertActKernel<<<static_cast<unsigned>((units + 255) / 256), 256, 0,
                       stream>>>(x, static_cast<__half*>(x_half), units);
    if (q8)
      LaunchGemmAs<true, true>(shape, wb, x_half, y, n_tokens, k, rows, ldy,
                               stream);
    else
      LaunchGemmAs<false, true>(shape, wb, x_half, y, n_tokens, k, rows, ldy,
                                stream);
  } else if (q8) {
    LaunchGemmAs<true, false>(shape, wb, x, y, n_tokens, k, rows, ldy, stream);
  } else {
    LaunchGemmAs<false, false>(shape, wb, x, y, n_tokens, k, rows, ldy, stream);
  }
}

void LaunchGemvDense(bool half_w, const void* w, const float* x, float* y,
                     std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t rows, hipStream_t stream) {
  // n_tokens in 1..8.
  if (half_w) {
    constexpr int RPW = 2;
    const dim3 grid((rows + 8 * RPW - 1) / (8 * RPW));
    LING_DISPATCH_NT(n_tokens, (GemvDenseKernel<NT, RPW, true>
                                <<<grid, 256, 0, stream>>>(w, x, y, k, rows)));
  } else {
    constexpr int RPW = 1;
    const dim3 grid((rows + 8 * RPW - 1) / (8 * RPW));
    LING_DISPATCH_NT(n_tokens, (GemvDenseKernel<NT, RPW, false>
                                <<<grid, 256, 0, stream>>>(w, x, y, k, rows)));
  }
}

}  // namespace

std::size_t ScratchBytes(WeightType type, std::uint32_t n_tokens,
                         std::uint32_t k) {
  if (type == WeightType::kF16 || type == WeightType::kF32)
    return 0;
  return QuantActBytes(n_tokens, k);
}

namespace {

// One GEMV launch (<= 3 groups) over the quantised activations of `n` tokens
// that start at q/q2/d (n <= 8, whole tokens fit the LDS budget).
void LaunchGemvQ8(const Q8Group* groups, int n_groups, const std::int8_t* q,
                  const std::int8_t* q2, const float* d, std::uint32_t n,
                  std::uint32_t k, hipStream_t stream) {
  const std::size_t per_token = 2 * std::size_t{k} + k / 32 * 4;
  const std::size_t lds = std::size_t{n} * per_token;
  const int rpw = n >= 4 ? 4 : 2;
  Q8Groups g{};
  std::uint32_t blocks = 0;
  for (int i = 0; i < 3; ++i) {
    g.first_block[i] = UINT32_MAX;
  }
  for (int i = 0; i < n_groups; ++i) {
    g.w[i] = static_cast<const std::uint8_t*>(groups[i].w);
    g.y[i] = groups[i].y;
    g.rows[i] = groups[i].rows;
    g.first_block[i] = blocks;
    blocks += (groups[i].rows + 8 * rpw - 1) / (8 * rpw);
  }
  const dim3 grid(blocks);
  if (n >= 4) {
    LING_DISPATCH_NT(
        n, (GemvQ8Kernel<NT, 4><<<grid, 256, lds, stream>>>(g, q, q2, d, k)));
  } else {
    LING_DISPATCH_NT(
        n, (GemvQ8Kernel<NT, 2><<<grid, 256, lds, stream>>>(g, q, q2, d, k)));
  }
}

}  // namespace

QuantView QuantLayout(void* base, std::uint32_t n_tokens, std::uint32_t k) {
  auto* p = static_cast<std::uint8_t*>(base);
  const std::size_t plane = Align256(std::size_t{n_tokens} * k);
  return {reinterpret_cast<std::int8_t*>(p),
          reinterpret_cast<std::int8_t*>(p + plane),
          reinterpret_cast<float*>(p + 2 * plane)};
}

std::size_t QuantActBytes(std::uint32_t n_tokens, std::uint32_t k) {
  return 2 * Align256(std::size_t{n_tokens} * k) +
         Align256(std::size_t{n_tokens} * (k / 32) * sizeof(float));
}

bool GemvQ8_0Fits(std::uint32_t n_tokens, std::uint32_t k) {
  return n_tokens >= 1 && n_tokens <= kMaxGemvTokens && k % 64 == 0 &&
         kGemvLdsBytes / (2 * std::size_t{k} + k / 32 * 4) >= 1;
}

void QuantizeF16ToQ8_0(const __half* w, void* out, std::uint64_t rows,
                       std::uint32_t k, hipStream_t stream) {
  const std::uint64_t blocks = rows * (k / 32);
  QuantizeF16ToQ8_0Kernel<<<static_cast<unsigned>((blocks + 255) / 256), 256, 0,
                            stream>>>(w, static_cast<std::uint16_t*>(out),
                                      blocks);
}

void QuantizeActivations(const float* x, std::uint32_t n_tokens,
                         std::uint32_t k, void* quant, hipStream_t stream) {
  const QuantView v = QuantLayout(quant, n_tokens, k);
  const std::size_t total = std::size_t{n_tokens} * k;
  QuantizeActKernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0,
                      stream>>>(x, v.q, v.q2, v.d, total);
}

void GemvQ8_0Quantized(const Q8Group* groups, int n_groups, const void* quant,
                       std::uint32_t n_tokens, std::uint32_t k,
                       hipStream_t stream) {
  if (n_tokens == 0 || n_groups == 0)
    return;
  // Sub-batches as in GemvQ8_0: as many tokens per launch as fit the LDS
  // budget; per-token arithmetic does not depend on the sub-batch size.
  const std::size_t per_token = 2 * std::size_t{k} + k / 32 * 4;
  const auto sub_max = static_cast<std::uint32_t>(
      std::min<std::size_t>(kGemvLdsBytes / per_token, kMaxGemvTokens));
  const QuantView v = QuantLayout(const_cast<void*>(quant), n_tokens, k);
  for (std::uint32_t t0 = 0; t0 < n_tokens; t0 += sub_max) {
    const std::uint32_t n = std::min(sub_max, n_tokens - t0);
    Q8Group shifted[3];
    for (int i = 0; i < n_groups; ++i) {
      shifted[i] = {groups[i].w, groups[i].y + std::size_t{t0} * groups[i].rows,
                    groups[i].rows};
    }
    LaunchGemvQ8(shifted, n_groups, v.q + std::size_t{t0} * k,
                 v.q2 + std::size_t{t0} * k, v.d + std::size_t{t0} * (k / 32),
                 n, k, stream);
  }
}

void GemvQ8_0(const void* w, const float* x, float* y, std::uint32_t n_tokens,
              std::uint32_t k, std::uint32_t rows, void* scratch,
              hipStream_t stream) {
  if (n_tokens == 0 || rows == 0)
    return;
  // Tokens per launch: as many as fit the LDS budget (at most 8). Every token
  // goes through the same per-token arithmetic whatever the sub-batch size, so
  // results for n <= 8 never depend on n; the weights are read once per
  // sub-batch.
  const std::size_t per_token = 2 * std::size_t{k} + k / 32 * 4;
  const std::size_t fit = kGemvLdsBytes / per_token;
  if (n_tokens > kMaxGemvTokens || k % 64 != 0 || fit == 0) {
    GemmQ8_0(w, x, y, n_tokens, k, rows, scratch, stream);
    return;
  }
  const auto sub_max =
      static_cast<std::uint32_t>(std::min<std::size_t>(fit, kMaxGemvTokens));
  for (std::uint32_t t0 = 0; t0 < n_tokens; t0 += sub_max) {
    const std::uint32_t n = std::min(sub_max, n_tokens - t0);
    const QuantView v = QuantLayout(scratch, n, k);
    const std::size_t total = std::size_t{n} * k;
    QuantizeActKernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0,
                        stream>>>(x + std::size_t{t0} * k, v.q, v.q2, v.d,
                                  total);
    const Q8Group group{w, y + std::size_t{t0} * rows, rows};
    LaunchGemvQ8(&group, 1, v.q, v.q2, v.d, n, k, stream);
  }
}

void GemmQ8_0(const void* w, const float* x, float* y, std::uint32_t n_tokens,
              std::uint32_t k, std::uint32_t rows, void* scratch,
              hipStream_t stream) {
  LaunchGemm(true, w, x, y, n_tokens, k, rows, rows, -1, scratch, stream);
}

void GemmQ8_0Strided(const void* w, const float* x, float* y,
                     std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t rows, std::uint32_t ldy, void* scratch,
                     hipStream_t stream) {
  LaunchGemm(true, w, x, y, n_tokens, k, rows, ldy, -1, scratch, stream);
}

int GemmShapeFor(std::uint32_t n_tokens, std::uint32_t rows) {
  return PickGemmShape(n_tokens, rows);
}

void GemmQ8_0Shape(const void* w, const float* x, float* y,
                   std::uint32_t n_tokens, std::uint32_t k, std::uint32_t rows,
                   std::uint32_t ldy, int shape, void* scratch,
                   hipStream_t stream) {
  LaunchGemm(true, w, x, y, n_tokens, k, rows, ldy, shape, scratch, stream);
}

void GemmF16Shape(const __half* w, const float* x, float* y,
                  std::uint32_t n_tokens, std::uint32_t k, std::uint32_t rows,
                  int shape, void* scratch, hipStream_t stream) {
  LaunchGemm(false, w, x, y, n_tokens, k, rows, rows, shape, scratch, stream);
}

void GemvF32(const float* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, hipStream_t stream) {
  for (std::uint32_t t = 0; t < n_tokens; t += kMaxGemvTokens) {
    const std::uint32_t n = std::min(kMaxGemvTokens, n_tokens - t);
    LaunchGemvDense(false, w, x + std::size_t{t} * k, y + std::size_t{t} * rows,
                    n, k, rows, stream);
  }
}

void GemvF16(const __half* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, void* scratch,
             hipStream_t stream) {
  if (n_tokens == 0 || rows == 0)
    return;
  if (n_tokens > kMaxGemvTokens) {
    GemmF16(w, x, y, n_tokens, k, rows, scratch, stream);
    return;
  }
  LaunchGemvDense(true, w, x, y, n_tokens, k, rows, stream);
}

void GemmF16(const __half* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, void* scratch,
             hipStream_t stream) {
  LaunchGemm(false, w, x, y, n_tokens, k, rows, rows, -1, scratch, stream);
}

void EmbedQ4K(const void* table, const std::int32_t* tokens, float* out,
              std::uint32_t n_tokens, std::uint32_t hidden,
              hipStream_t stream) {
  if (n_tokens == 0)
    return;
  EmbedQ4KKernel<<<n_tokens, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(table), tokens, out, hidden);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream) {
  if (n_rows == 0)
    return;
  const std::uint32_t d = dim / groups;
  const std::uint32_t threads = 32 * std::min(8U, (d + 127) / 128);
  RmsNormRowsKernel<<<n_rows * groups, threads, 0, stream>>>(x, gamma, out, dim,
                                                             groups, d, eps);
}

void RmsNormQuantRows(const float* x, const float* gamma, float* out,
                      std::uint32_t n_rows, float eps, void* quant,
                      hipStream_t stream) {
  if (n_rows == 0)
    return;
  const QuantView v = QuantLayout(quant, n_rows, kHidden);
  NormQuantKernel<0><<<n_rows, kNormThreads, 0, stream>>>(
      const_cast<float*>(x), nullptr, nullptr, nullptr, nullptr, gamma, out,
      v.q, v.q2, v.d, eps);
}

void AddResidualRmsNormQuant(float* res, const float* addend,
                             const float* gamma, float* out,
                             std::uint32_t n_rows, float eps, void* quant,
                             hipStream_t stream) {
  if (n_rows == 0)
    return;
  const QuantView v = QuantLayout(quant, n_rows, kHidden);
  NormQuantKernel<1><<<n_rows, kNormThreads, 0, stream>>>(
      res, addend, nullptr, nullptr, nullptr, gamma, out, v.q, v.q2, v.d, eps);
}

void CombineRmsNormQuant(float* res, const float* expert_out,
                         const float* weights, const float* shared_out,
                         const float* gamma, float* out, std::uint32_t n_rows,
                         float eps, void* quant, hipStream_t stream) {
  if (n_rows == 0)
    return;
  const QuantView v = QuantLayout(quant, n_rows, kHidden);
  NormQuantKernel<2><<<n_rows, kNormThreads, 0, stream>>>(
      res, nullptr, expert_out, weights, shared_out, gamma, out, v.q, v.q2, v.d,
      eps);
}

void SwiGluClampedQuant(const float* gate, const float* up, float* out,
                        std::size_t count, std::uint32_t k, float limit,
                        void* quant, hipStream_t stream) {
  if (count == 0)
    return;
  const QuantView v =
      QuantLayout(quant, static_cast<std::uint32_t>(count / k), k);
  SwiGluQuantKernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                      stream>>>(gate, up, out, v.q, v.q2, v.d, count, limit);
}

void AddResidual(float* res, const float* x, std::size_t count,
                 hipStream_t stream) {
  if (count == 0)
    return;
  const std::size_t blocks = std::min<std::size_t>((count + 255) / 256, 65535);
  AddResidualKernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(res, x,
                                                                       count);
}

void SwiGluClamped(const float* gate, const float* up, float* out,
                   std::size_t count, float limit, hipStream_t stream) {
  if (count == 0)
    return;
  const std::size_t blocks = std::min<std::size_t>((count + 255) / 256, 65535);
  SwiGluClampedKernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
      gate, up, out, count, limit);
}

}  // namespace gufo::models::ling_3_flash::rocm::dense
