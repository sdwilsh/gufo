// Ling-3.0-flash dense/linear kernels: Q8_0 / F16 / F32 projections, Q4_K
// embedding gather, RMSNorm, residual add and clamped SwiGLU. Wave32 (gfx11).
//
// Design summary
//  * GemvQ8_0 (decode, 1..8 tokens): activations are first quantised to int8
//    with one F32 scale per 32 values (the Q8_1 scheme llama.cpp uses for Q8_0
//    matmuls, without the block sum). Each wave streams whole rows of the
//    weights exactly once with dword loads (a pair of 34-byte Q8_0 blocks is 17
//    aligned dwords), multiplies with v_dot4_i32_iu8 against the int8
//    activations held in LDS, and applies the two scales per block.
//  * GemmQ8_0 / GemmF16 (prefill): 64x64 tile, K step 32, F16 WMMA with F32
//    accumulation. Q8_0 codes are dequantised to F16 on the fly while staging
//    to LDS; activations are rounded to F16. No activation scale quantisation.
//  * GemvF16 / GemvF32: one wave per row (RPW rows interleaved), x slabs of
//  1024
//    floats staged in LDS, F32 accumulation, 16-byte weight loads.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

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
__global__ __launch_bounds__(256) void QuantizeActKernel(
    const float* __restrict__ x, std::int8_t* __restrict__ q,
    float* __restrict__ d, std::size_t total) {
  const std::size_t i = std::size_t{blockIdx.x} * 256 + threadIdx.x;
  const float v = i < total ? x[i] : 0.0F;
  float m = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor(m, off, 32));
  const float inv = m > 0.0F ? 127.0F / m : 0.0F;
  if (i < total) {
    q[i] = static_cast<std::int8_t>(__float2int_rn(v * inv));
    if ((threadIdx.x & 31) == 0)
      d[i >> 5] = m * (1.0F / 127.0F);
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

template<int NT, int RPW>
__global__ __launch_bounds__(256) void GemvQ8Kernel(
    const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
    const float* __restrict__ xd, float* __restrict__ y, std::uint32_t k,
    std::uint32_t rows) {
  extern __shared__ __attribute__((aligned(16))) std::uint8_t smem[];
  std::int8_t* s_q = reinterpret_cast<std::int8_t*>(smem);
  float* s_d = reinterpret_cast<float*>(smem + std::size_t{NT} * k);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  {
    const auto* src = reinterpret_cast<const uint4*>(xq);
    auto* dst = reinterpret_cast<uint4*>(s_q);
    for (std::uint32_t i = tid; i < NT * k / 16; i += 256)
      dst[i] = src[i];
    const std::uint32_t nd = NT * (k / 32);
    for (std::uint32_t i = tid; i < nd; i += 256)
      s_d[i] = xd[i];
  }
  __syncthreads();

  const std::uint32_t npairs = k / 64;
  const std::size_t row_bytes = std::size_t{npairs} * 68;
  const std::uint32_t row0 = (blockIdx.x * 8 + wave) * RPW;
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
      const float sx0 = s_d[t * (k / 32) + 2 * p];
      const float sx1 = s_d[t * (k / 32) + 2 * p + 1];
#pragma unroll
      for (int j = 0; j < RPW; ++j) {
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s0 = Dot4(qa[j][i], xa[i], s0);
          s1 = Dot4(static_cast<int>(u[j][9 + i]), xb[i], s1);
        }
        acc[j][t] = fmaf(d0[j] * sx0, static_cast<float>(s0), acc[j][t]);
        acc[j][t] = fmaf(d1[j] * sx1, static_cast<float>(s1), acc[j][t]);
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
    for (std::uint32_t c = lane; c < nchunks; c += 32) {
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
// Block: 128 threads = 4 waves, tile 64 tokens x 64 weight rows, K step 32.
// The activations are the WMMA A operand and the weights the B operand so
// each lane writes consecutive output rows.
// ---------------------------------------------------------------------------
constexpr int kBM = 64;   // weight rows per block
constexpr int kBN = 64;   // tokens per block
constexpr int kLds = 40;  // halves per LDS row (32 + 8 pad)

template<bool kQ8>
__global__ __launch_bounds__(128) void GemmKernel(
    const std::uint8_t* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, std::uint32_t n_tokens, std::uint32_t k,
    std::uint32_t rows, std::uint32_t ldy) {
  __shared__ __attribute__((aligned(16))) __half s_x[kBN][kLds];
  __shared__ __attribute__((aligned(16))) __half s_w[kBM][kLds];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int sub = lane & 15;
  const int half_id = lane >> 4;
  const int wt = wave >> 1;  // token half of the tile
  const int wr = wave & 1;   // row half of the tile
  const std::uint32_t t0 = blockIdx.x * kBN;
  const std::uint32_t r0 = blockIdx.y * kBM;
  const int ld_row = tid >> 1;  // staging row (token or weight row)
  const int ld_half = tid & 1;  // which 16 k values of the stage
  const std::uint32_t wrow = r0 + ld_row;
  const std::uint32_t xtok = t0 + ld_row;
  const bool w_live = wrow < rows;
  const bool x_live = xtok < n_tokens;
  const std::size_t w_row_bytes =
      kQ8 ? std::size_t{k / 32} * 34 : std::size_t{k} * 2;
  const std::uint8_t* wp = w + std::size_t{w_live ? wrow : 0} * w_row_bytes;
  const float* xp = x + std::size_t{x_live ? xtok : 0} * k;

  v8f acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      acc[i][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

  uint4 wr_regs[2];
  std::uint32_t scale_bits = 0;
  float4 xr[4];
  const std::uint32_t nkb = k / 32;

  const auto fetch = [&](std::uint32_t kb) {
    if constexpr (kQ8) {
      const auto* blk =
          reinterpret_cast<const std::uint16_t*>(wp + std::size_t{kb} * 34);
      std::uint16_t h[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        h[i] = w_live ? blk[1 + ld_half * 8 + i] : 0;
      scale_bits = w_live ? blk[0] : 0;
      __builtin_memcpy(&wr_regs[0], h, 16);
    } else {
      const auto* src = reinterpret_cast<const uint4*>(
          wp + std::size_t{kb} * 64 + ld_half * 32);
      wr_regs[0] = w_live ? src[0] : make_uint4(0, 0, 0, 0);
      wr_regs[1] = w_live ? src[1] : make_uint4(0, 0, 0, 0);
    }
    const auto* xs = reinterpret_cast<const float4*>(xp + std::size_t{kb} * 32 +
                                                     ld_half * 16);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      xr[i] = x_live ? xs[i] : make_float4(0, 0, 0, 0);
  };
  const auto commit = [&]() {
    auto* dw = reinterpret_cast<uint4*>(&s_w[ld_row][ld_half * 16]);
    if constexpr (kQ8) {
      const float d = HalfBitsToFloat(scale_bits);
      std::int8_t q[16];
      __builtin_memcpy(q, &wr_regs[0], 16);
      __half h[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        h[i] = __float2half(static_cast<float>(q[i]) * d);
      __builtin_memcpy(&dw[0], &h[0], 16);
      __builtin_memcpy(&dw[1], &h[8], 16);
    } else {
      dw[0] = wr_regs[0];
      dw[1] = wr_regs[1];
    }
    __half hx[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      hx[4 * i] = __float2half(xr[i].x);
      hx[4 * i + 1] = __float2half(xr[i].y);
      hx[4 * i + 2] = __float2half(xr[i].z);
      hx[4 * i + 3] = __float2half(xr[i].w);
    }
    auto* dx = reinterpret_cast<uint4*>(&s_x[ld_row][ld_half * 16]);
    __builtin_memcpy(&dx[0], &hx[0], 16);
    __builtin_memcpy(&dx[1], &hx[8], 16);
  };

  fetch(0);
  for (std::uint32_t kb = 0; kb < nkb; ++kb) {
    commit();
    __syncthreads();
    if (kb + 1 < nkb)
      fetch(kb + 1);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      v16h a[2], b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const __half* pa = &s_x[wt * 32 + i * 16 + sub][ks * 16];
        const __half* pb = &s_w[wr * 32 + i * 16 + sub][ks * 16];
        __builtin_memcpy(&a[i], pa, 32);
        __builtin_memcpy(&b[i], pb, 32);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          acc[i][j] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const std::uint32_t row = r0 + wr * 32 + j * 16 + sub;
      if (row >= rows)
        continue;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const std::uint32_t tok = t0 + wt * 32 + i * 16 + 2 * e + half_id;
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
// Host helpers.
// ---------------------------------------------------------------------------
std::size_t QuantBytes(std::uint32_t n_tokens, std::uint32_t k) {
  return Align256(std::size_t{n_tokens} * k) +
         Align256(std::size_t{n_tokens} * (k / 32) * sizeof(float));
}

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

void LaunchGemm(bool q8, const void* w, const float* x, float* y,
                std::uint32_t n_tokens, std::uint32_t k, std::uint32_t rows,
                std::uint32_t ldy, hipStream_t stream) {
  if (n_tokens == 0 || rows == 0)
    return;
  const dim3 grid((n_tokens + kBN - 1) / kBN, (rows + kBM - 1) / kBM);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  if (q8)
    GemmKernel<true>
        <<<grid, 128, 0, stream>>>(wb, x, y, n_tokens, k, rows, ldy);
  else
    GemmKernel<false>
        <<<grid, 128, 0, stream>>>(wb, x, y, n_tokens, k, rows, ldy);
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
  return QuantBytes(n_tokens, k);
}

void GemvQ8_0(const void* w, const float* x, float* y, std::uint32_t n_tokens,
              std::uint32_t k, std::uint32_t rows, void* scratch,
              hipStream_t stream) {
  if (n_tokens == 0 || rows == 0)
    return;
  const std::size_t lds = std::size_t{n_tokens} * (k + k / 32 * 4);
  if (n_tokens > kMaxGemvTokens || k % 64 != 0 || lds > kGemvLdsBytes) {
    GemmQ8_0(w, x, y, n_tokens, k, rows, scratch, stream);
    return;
  }
  auto* q = static_cast<std::int8_t*>(scratch);
  auto* d = reinterpret_cast<float*>(static_cast<std::uint8_t*>(scratch) +
                                     Align256(std::size_t{n_tokens} * k));
  const std::size_t total = std::size_t{n_tokens} * k;
  QuantizeActKernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0,
                      stream>>>(x, q, d, total);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  if (n_tokens >= 4) {
    constexpr int RPW = 4;
    const dim3 grid((rows + 8 * RPW - 1) / (8 * RPW));
    LING_DISPATCH_NT(n_tokens,
                     (GemvQ8Kernel<NT, RPW>
                      <<<grid, 256, lds, stream>>>(wb, q, d, y, k, rows)));
  } else {
    constexpr int RPW = 2;
    const dim3 grid((rows + 8 * RPW - 1) / (8 * RPW));
    LING_DISPATCH_NT(n_tokens,
                     (GemvQ8Kernel<NT, RPW>
                      <<<grid, 256, lds, stream>>>(wb, q, d, y, k, rows)));
  }
}

void GemmQ8_0(const void* w, const float* x, float* y, std::uint32_t n_tokens,
              std::uint32_t k, std::uint32_t rows, void*, hipStream_t stream) {
  LaunchGemm(true, w, x, y, n_tokens, k, rows, rows, stream);
}

void GemmQ8_0Strided(const void* w, const float* x, float* y,
                     std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t rows, std::uint32_t ldy, void*,
                     hipStream_t stream) {
  LaunchGemm(true, w, x, y, n_tokens, k, rows, ldy, stream);
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
             std::uint32_t k, std::uint32_t rows, hipStream_t stream) {
  if (n_tokens == 0 || rows == 0)
    return;
  if (n_tokens > kMaxGemvTokens) {
    GemmF16(w, x, y, n_tokens, k, rows, stream);
    return;
  }
  LaunchGemvDense(true, w, x, y, n_tokens, k, rows, stream);
}

void GemmF16(const __half* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, hipStream_t stream) {
  LaunchGemm(false, w, x, y, n_tokens, k, rows, rows, stream);
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
