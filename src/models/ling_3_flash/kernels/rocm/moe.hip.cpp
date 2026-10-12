// SPDX-License-Identifier: MIT
// Ling-3.0-flash (bailingmoe3) MoE: sigmoid group-limited router, routed
// expert gate/up/down kernels (decode GEMV and grouped WMMA prefill) and the
// combine. Math: artifacts/ling-3-flash/PLAN.md section 1.
//
// Decode (1..8 tokens): a one-block prep kernel lists the distinct experts the
// batch selects, then a GEMV block reads one expert's rows once and dots them
// against every token that picked it; activations stay F32, weights are
// dequantised in registers.
// Prefill: (token, slot) pairs are sorted by expert into <=TN-token tiles; a
// 4-wave block stages a 64-row dequantised F16 weight tile and a TN-token F16
// activation tile in LDS and runs v_wmma_f32_16x16x16_f16 with F32
// accumulators.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"
#include "src/models/ling_3_flash/kernels/rocm/mmq/quant_units.hip.hpp"

namespace gufo::models::ling_3_flash::rocm::moe {
namespace {

using detail::RowBytes;
using detail::Unit;

constexpr std::uint32_t kExperts = 512;
constexpr std::uint32_t kUsed = 8;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kFf = 768;

[[noreturn]] void Unsupported() {
  throw std::invalid_argument("ling moe: unsupported expert weight type");
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool Better(float v, int i, float bv, int bi) {
  return v > bv || (v == bv && i < bi);
}

__global__ __launch_bounds__(256) void RouteKernel(const float* logits,
                                                   const float* bias,
                                                   std::int32_t* ids,
                                                   float* weights) {
  __shared__ float p[512];
  __shared__ float sel[512];
  __shared__ float gscore[8];
  __shared__ int gmask;
  __shared__ float wv[8];
  __shared__ int wi[8];
  __shared__ int chosen[8];
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const float* lg = logits + static_cast<std::size_t>(t) * 512;
  for (int i = tid; i < 512; i += 256) {
    const float pi = 1.0F / (1.0F + expf(-lg[i]));
    p[i] = pi;
    sel[i] = pi + bias[i];
  }
  __syncthreads();
  if (tid < 8) {
    float t1 = -INFINITY, t2 = -INFINITY;
    for (int i = 0; i < 64; ++i) {
      const float v = sel[tid * 64 + i];
      if (v > t1) {
        t2 = t1;
        t1 = v;
      } else if (v > t2) {
        t2 = v;
      }
    }
    gscore[tid] = t1 + t2;
  }
  __syncthreads();
  if (tid == 0) {
    int mask = 0;
    for (int r = 0; r < 4; ++r) {
      int best = -1;
      for (int g = 0; g < 8; ++g) {
        if ((mask >> g) & 1) {
          continue;
        }
        if (best < 0 || gscore[g] > gscore[best]) {
          best = g;
        }
      }
      mask |= 1 << best;
    }
    gmask = mask;
  }
  __syncthreads();
  for (int i = tid; i < 512; i += 256) {
    if (!((gmask >> (i >> 6)) & 1)) {
      sel[i] = -INFINITY;
    }
  }
  __syncthreads();
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int r = 0; r < 8; ++r) {
    float bv = sel[tid];
    int bi = tid;
    const float v2 = sel[tid + 256];
    if (Better(v2, tid + 256, bv, bi)) {
      bv = v2;
      bi = tid + 256;
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off, 32);
      const int oi = __shfl_xor(bi, off, 32);
      if (Better(ov, oi, bv, bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      wv[wave] = bv;
      wi[wave] = bi;
    }
    __syncthreads();
    if (tid == 0) {
      float cv = wv[0];
      int ci = wi[0];
      for (int w = 1; w < 8; ++w) {
        if (Better(wv[w], wi[w], cv, ci)) {
          cv = wv[w];
          ci = wi[w];
        }
      }
      chosen[r] = ci;
      sel[ci] = -INFINITY;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float sum = 0.0F;
    for (int r = 0; r < 8; ++r) {
      sum += p[chosen[r]];
    }
    const float denom = fmaxf(sum, 6.1035e-5F);
    for (int r = 0; r < 8; ++r) {
      ids[t * 8 + r] = chosen[r];
      weights[t * 8 + r] = p[chosen[r]] / denom * 2.5F;
    }
  }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float SwiGlu(float g, float u, float limit) {
  float a = g / (1.0F + expf(-g));
  if (limit != 0.0F) {
    a = fminf(a, limit);
    u = fminf(fmaxf(u, -limit), limit);
  }
  return a * u;
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------
struct DecodeEntry {
  int expert;
  int count;
  int pair[8];  // token * 8 + slot, ascending
};
constexpr int kDecodeMaxEntries = 64;
constexpr std::size_t kDecodeHeader = 16;

__global__ __launch_bounds__(64) void DecodePrepKernel(const std::int32_t* ids,
                                                       std::uint32_t n_tokens,
                                                       std::uint8_t* scratch) {
  __shared__ int e[64];
  __shared__ int rank[64];
  const int p = threadIdx.x;
  const int n = static_cast<int>(n_tokens) * 8;
  e[p] = p < n ? ids[p] : -1;
  __syncthreads();
  const bool valid = p < n && e[p] >= 0 && e[p] < 512;
  bool leader = valid;
  for (int q = 0; q < p && leader; ++q) {
    leader = e[q] != e[p];
  }
  rank[p] = leader ? 1 : 0;
  __syncthreads();
  if (p == 0) {
    int run = 0;
    for (int q = 0; q < 64; ++q) {
      const int f = rank[q];
      rank[q] = run;
      run += f;
    }
    *reinterpret_cast<int*>(scratch) = run;
  }
  __syncthreads();
  if (leader) {
    auto* entries = reinterpret_cast<DecodeEntry*>(scratch + kDecodeHeader);
    DecodeEntry& en = entries[rank[p]];
    en.expert = e[p];
    int c = 0;
    for (int q = p; q < n; ++q) {
      if (e[q] == e[p]) {
        en.pair[c++] = q;
      }
    }
    en.count = c;
  }
}

__device__ __forceinline__ float Dot32(const float* w, const float* x) {
  float a = 0.0F;
#pragma unroll
  for (int i = 0; i < 32; i += 4) {
    const float4 v = *reinterpret_cast<const float4*>(x + i);
    a = fmaf(w[i], v.x, a);
    a = fmaf(w[i + 1], v.y, a);
    a = fmaf(w[i + 2], v.z, a);
    a = fmaf(w[i + 3], v.w, a);
  }
  return a;
}

/// One block = 256/LR rows of one distinct expert; LR lanes share a row.
template<WeightType T, bool kGateUp, int LR>
__global__ __launch_bounds__(256) void DecodeKernel(
    const std::uint8_t* w0, const std::uint8_t* w1, const float* x, float limit,
    float* out, const std::uint8_t* scratch) {
  constexpr std::uint32_t kK = kGateUp ? kHidden : kFf;
  constexpr std::uint32_t kRows = kGateUp ? kFf : kHidden;
  constexpr std::size_t kRb = RowBytes<T>(kK);
  using Block = typename Unit<T>::Block;
  const int slot = blockIdx.y;
  if (slot >= *reinterpret_cast<const int*>(scratch)) {
    return;
  }
  const DecodeEntry en =
      reinterpret_cast<const DecodeEntry*>(scratch + kDecodeHeader)[slot];
  const int r = blockIdx.x * (256 / LR) + threadIdx.x / LR;
  const int lane = threadIdx.x % LR;
  const std::size_t roff =
      (static_cast<std::size_t>(en.expert) * kRows + r) * kRb;
  const std::uint8_t* rg = w0 + roff;
  const std::uint8_t* ru = w1 + roff;
  float ag[8], au[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    ag[i] = 0.0F;
    au[i] = 0.0F;
  }
  float wv[32];
  for (std::uint32_t u = lane; u < kK / 32; u += LR) {
    const std::uint32_t sb = u >> 3, s = u & 7;
    Unit<T>::Dequant(reinterpret_cast<const Block*>(rg)[sb],
                     static_cast<int>(s), wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i < en.count) {
        const int pr = en.pair[i];
        const float* xs = x +
                          (kGateUp ? static_cast<std::size_t>(pr >> 3) * kHidden
                                   : static_cast<std::size_t>(pr) * kFf) +
                          u * 32;
        ag[i] += Dot32(wv, xs);
      }
    }
    if constexpr (kGateUp) {
      Unit<T>::Dequant(reinterpret_cast<const Block*>(ru)[sb],
                       static_cast<int>(s), wv);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < en.count) {
          const float* xs =
              x + static_cast<std::size_t>(en.pair[i] >> 3) * kHidden + u * 32;
          au[i] += Dot32(wv, xs);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    for (int off = LR / 2; off > 0; off >>= 1) {
      ag[i] += __shfl_xor(ag[i], off, LR);
      if constexpr (kGateUp) {
        au[i] += __shfl_xor(au[i], off, LR);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i < en.count) {
        const std::size_t pr = static_cast<std::size_t>(en.pair[i]);
        if constexpr (kGateUp) {
          out[pr * kFf + r] = SwiGlu(ag[i], au[i], limit);
        } else {
          out[pr * kHidden + r] = ag[i];
        }
      }
    }
  }
}

template<WeightType T>
void LaunchDecode(const void* w0, const void* w1, bool gate_up, const float* x,
                  float limit, float* out, std::uint32_t n_tokens,
                  void* scratch, hipStream_t stream) {
  const dim3 block(256);
  const unsigned slots = n_tokens * kUsed;
  if (gate_up) {
    hipLaunchKernelGGL((DecodeKernel<T, true, 16>), dim3(kFf / 16, slots),
                       block, 0, stream, static_cast<const std::uint8_t*>(w0),
                       static_cast<const std::uint8_t*>(w1), x, limit, out,
                       static_cast<const std::uint8_t*>(scratch));
  } else {
    hipLaunchKernelGGL((DecodeKernel<T, false, 8>), dim3(kHidden / 32, slots),
                       block, 0, stream, static_cast<const std::uint8_t*>(w0),
                       static_cast<const std::uint8_t*>(w1), x, limit, out,
                       static_cast<const std::uint8_t*>(scratch));
  }
}

void DecodeDispatch(const void* w0, const void* w1, bool gate_up,
                    WeightType type, const float* x, const std::int32_t* ids,
                    std::uint32_t n_tokens, float limit, float* out,
                    void* scratch, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  if (n_tokens > 8) {
    throw std::invalid_argument("ling moe decode: n_tokens must be 1..8");
  }
  hipLaunchKernelGGL(DecodePrepKernel, dim3(1), dim3(64), 0, stream, ids,
                     n_tokens, static_cast<std::uint8_t*>(scratch));
  switch (type) {
    case WeightType::kIQ4_XS:
      LaunchDecode<WeightType::kIQ4_XS>(w0, w1, gate_up, x, limit, out,
                                        n_tokens, scratch, stream);
      break;
    case WeightType::kQ4_K:
      LaunchDecode<WeightType::kQ4_K>(w0, w1, gate_up, x, limit, out, n_tokens,
                                      scratch, stream);
      break;
    case WeightType::kQ5_K:
      LaunchDecode<WeightType::kQ5_K>(w0, w1, gate_up, x, limit, out, n_tokens,
                                      scratch, stream);
      break;
    case WeightType::kQ6_K:
      LaunchDecode<WeightType::kQ6_K>(w0, w1, gate_up, x, limit, out, n_tokens,
                                      scratch, stream);
      break;
    default:
      Unsupported();
  }
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------
// Scratch: [0] tile_count, then sorted pairs, then tiles {expert,start,count}.
constexpr std::size_t kPrefillHeader = 16;

std::uint32_t TileBound(std::uint32_t n_tokens, std::uint32_t tn) {
  const std::uint32_t pairs = n_tokens * kUsed;
  return std::min(kExperts, pairs) + pairs / tn + 1;
}

__global__ __launch_bounds__(1024) void PrefillPrepKernel(
    const std::int32_t* ids, std::uint32_t pairs, std::uint32_t tn,
    std::uint8_t* scratch) {
  __shared__ int cnt[512];
  __shared__ int off[512];
  __shared__ int cur[512];
  const int tid = threadIdx.x;
  if (tid < 512) {
    cnt[tid] = 0;
    cur[tid] = 0;
  }
  __syncthreads();
  for (std::uint32_t p = tid; p < pairs; p += 1024) {
    const int e = ids[p];
    if (e >= 0 && e < 512) {
      atomicAdd(&cnt[e], 1);
    }
  }
  __syncthreads();
  int* header = reinterpret_cast<int*>(scratch);
  int* sorted = header + kPrefillHeader / 4;
  int* tiles = sorted + pairs;
  if (tid == 0) {
    int run = 0, nt = 0;
    for (int e = 0; e < 512; ++e) {
      off[e] = run;
      for (int s = 0; s < cnt[e]; s += static_cast<int>(tn)) {
        tiles[nt * 3 + 0] = e;
        tiles[nt * 3 + 1] = run + s;
        tiles[nt * 3 + 2] = min(static_cast<int>(tn), cnt[e] - s);
        ++nt;
      }
      run += cnt[e];
    }
    header[0] = nt;
  }
  __syncthreads();
  for (std::uint32_t p = tid; p < pairs; p += 1024) {
    const int e = ids[p];
    if (e >= 0 && e < 512) {
      sorted[off[e] + atomicAdd(&cur[e], 1)] = static_cast<int>(p);
    }
  }
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v16h LoadFrag(const _Float16* p) {
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
  struct {
    uint4 lo, hi;
  } u{a, b};
  return __builtin_bit_cast(v16h, u);
}

constexpr int kLdsStride = 136;  // 128 halves + 8 pad (row stride 272 bytes)

template<WeightType T, bool kGateUp, int NF>
__global__ __launch_bounds__(128) void PrefillKernel(
    const std::uint8_t* w0, const std::uint8_t* w1, const float* x, float limit,
    float* out, const std::uint8_t* scratch, std::uint32_t pairs) {
  constexpr int kTn = 16 * NF;
  constexpr std::uint32_t kK = kGateUp ? kHidden : kFf;
  constexpr std::uint32_t kRows = kGateUp ? kFf : kHidden;
  constexpr std::size_t kRb = RowBytes<T>(kK);
  using Block = typename Unit<T>::Block;
  __shared__ __attribute__((aligned(16))) _Float16 ws[64 * kLdsStride];
  __shared__ __attribute__((aligned(16))) _Float16 xs[kTn * kLdsStride];
  __shared__ int pair_s[kTn];

  const int* header = reinterpret_cast<const int*>(scratch);
  const int tile = blockIdx.y;
  if (tile >= header[0]) {
    return;
  }
  const int* sorted = header + kPrefillHeader / 4;
  const int* tiles = sorted + pairs;
  const int expert = tiles[tile * 3];
  const int start = tiles[tile * 3 + 1];
  const int count = tiles[tile * 3 + 2];
  const int rt = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;

  if (tid < kTn) {
    pair_s[tid] = tid < count ? sorted[start + tid] : -1;
  }
  // Dequant mapping: thread = (weight row, half of the 128-wide chunk).
  const int wrow = tid >> 1;
  const int h = tid & 1;
  const std::uint8_t* mat = w0;
  int row;
  if constexpr (kGateUp) {
    mat = wrow < 32 ? w0 : w1;
    row = rt * 32 + (wrow & 31);
  } else {
    row = rt * 64 + wrow;
  }
  const std::uint8_t* rowp =
      mat + (static_cast<std::size_t>(expert) * kRows + row) * kRb;
  __syncthreads();

  v8f acc[NF];
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    acc[f] = v8f{};
  }

  for (std::uint32_t chunk = 0; chunk < kK / 128; ++chunk) {
    {
      const Block& blk = reinterpret_cast<const Block*>(rowp)[chunk >> 1];
      float tmp[32];
#pragma unroll
      for (int uu = 0; uu < 2; ++uu) {
        Unit<T>::Dequant(blk, static_cast<int>((chunk & 1) * 4 + h * 2 + uu),
                         tmp);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          struct {
            _Float16 v[8];
          } hv;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            hv.v[j] = static_cast<_Float16>(tmp[m * 8 + j]);
          }
          *reinterpret_cast<uint4*>(
              &ws[wrow * kLdsStride + h * 64 + uu * 32 + m * 8]) =
              __builtin_bit_cast(uint4, hv);
        }
      }
    }
    for (int idx = tid; idx < kTn * 32; idx += 128) {
      const int j = idx >> 5, c4 = idx & 31;
      float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const int pr = pair_s[j];
      if (pr >= 0) {
        const float* src =
            x +
            (kGateUp ? static_cast<std::size_t>(pr >> 3) * kHidden
                     : static_cast<std::size_t>(pr) * kFf) +
            chunk * 128 + c4 * 4;
        v = *reinterpret_cast<const float4*>(src);
      }
      _Float16* d = &xs[j * kLdsStride + c4 * 4];
      d[0] = static_cast<_Float16>(v.x);
      d[1] = static_cast<_Float16>(v.y);
      d[2] = static_cast<_Float16>(v.z);
      d[3] = static_cast<_Float16>(v.w);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const v16h a =
          LoadFrag(&ws[(wave * 16 + (lane & 15)) * kLdsStride + kk * 16]);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        const v16h b =
            LoadFrag(&xs[(f * 16 + (lane & 15)) * kLdsStride + kk * 16]);
        acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[f]);
      }
    }
    __syncthreads();
  }

  // Epilogue through LDS (aliases the weight tile): C[row][token].
  float* cs = reinterpret_cast<float*>(ws);
  constexpr int kCs = kTn + 1;
#pragma unroll
  for (int f = 0; f < NF; ++f) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      cs[(wave * 16 + 2 * i + (lane >> 4)) * kCs + f * 16 + (lane & 15)] =
          acc[f][i];
    }
  }
  __syncthreads();
  if constexpr (kGateUp) {
    for (int idx = tid; idx < 32 * kTn; idx += 128) {
      const int j = idx >> 5, r = idx & 31;
      if (j < count) {
        out[static_cast<std::size_t>(pair_s[j]) * kFf + rt * 32 + r] =
            SwiGlu(cs[r * kCs + j], cs[(32 + r) * kCs + j], limit);
      }
    }
  } else {
    for (int idx = tid; idx < 64 * kTn; idx += 128) {
      const int j = idx >> 6, r = idx & 63;
      if (j < count) {
        out[static_cast<std::size_t>(pair_s[j]) * kHidden + rt * 64 + r] =
            cs[r * kCs + j];
      }
    }
  }
}

template<WeightType T, int NF>
void LaunchPrefillTile(const void* w0, const void* w1, bool gate_up,
                       const float* x, float limit, float* out,
                       std::uint32_t n_tokens, void* scratch,
                       hipStream_t stream) {
  const std::uint32_t pairs = n_tokens * kUsed;
  const unsigned tiles = TileBound(n_tokens, 16 * NF);
  const auto* w0p = static_cast<const std::uint8_t*>(w0);
  const auto* w1p = static_cast<const std::uint8_t*>(w1);
  const auto* sp = static_cast<const std::uint8_t*>(scratch);
  if (gate_up) {
    hipLaunchKernelGGL((PrefillKernel<T, true, NF>), dim3(kFf / 32, tiles),
                       dim3(128), 0, stream, w0p, w1p, x, limit, out, sp,
                       pairs);
  } else {
    hipLaunchKernelGGL((PrefillKernel<T, false, NF>), dim3(kHidden / 64, tiles),
                       dim3(128), 0, stream, w0p, w1p, x, limit, out, sp,
                       pairs);
  }
}

template<WeightType T>
void LaunchPrefill(const void* w0, const void* w1, bool gate_up, const float* x,
                   float limit, float* out, std::uint32_t n_tokens,
                   std::uint32_t nf, void* scratch, hipStream_t stream) {
  switch (nf) {
    case 1:
      LaunchPrefillTile<T, 1>(w0, w1, gate_up, x, limit, out, n_tokens, scratch,
                              stream);
      break;
    case 2:
      LaunchPrefillTile<T, 2>(w0, w1, gate_up, x, limit, out, n_tokens, scratch,
                              stream);
      break;
    default:
      LaunchPrefillTile<T, 4>(w0, w1, gate_up, x, limit, out, n_tokens, scratch,
                              stream);
      break;
  }
}

/// Token-tile fragments (16 tokens each): empty tiles waste matrix-core issue,
/// so pick by the mean tokens per expert (n_tokens / 64).
std::uint32_t PickFrags(std::uint32_t n_tokens) {
  return n_tokens <= 512 ? 1 : (n_tokens <= 2048 ? 2 : 4);
}

void PrefillDispatch(const void* w0, const void* w1, bool gate_up,
                     WeightType type, const float* x, const std::int32_t* ids,
                     std::uint32_t n_tokens, float limit, float* out,
                     void* scratch, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const std::uint32_t nf = PickFrags(n_tokens);
  hipLaunchKernelGGL(PrefillPrepKernel, dim3(1), dim3(1024), 0, stream, ids,
                     n_tokens * kUsed, 16 * nf,
                     static_cast<std::uint8_t*>(scratch));
  switch (type) {
    case WeightType::kIQ4_XS:
      LaunchPrefill<WeightType::kIQ4_XS>(w0, w1, gate_up, x, limit, out,
                                         n_tokens, nf, scratch, stream);
      break;
    case WeightType::kQ4_K:
      LaunchPrefill<WeightType::kQ4_K>(w0, w1, gate_up, x, limit, out, n_tokens,
                                       nf, scratch, stream);
      break;
    case WeightType::kQ5_K:
      LaunchPrefill<WeightType::kQ5_K>(w0, w1, gate_up, x, limit, out, n_tokens,
                                       nf, scratch, stream);
      break;
    case WeightType::kQ6_K:
      LaunchPrefill<WeightType::kQ6_K>(w0, w1, gate_up, x, limit, out, n_tokens,
                                       nf, scratch, stream);
      break;
    default:
      Unsupported();
  }
}

// ---------------------------------------------------------------------------
// Combine
// ---------------------------------------------------------------------------
__global__ void CombineKernel(float* res, const float* expert_out,
                              const float* weights, const float* shared,
                              std::uint32_t n4) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) {
    return;
  }
  const std::uint32_t t = i / (kHidden / 4);
  const std::uint32_t c = i % (kHidden / 4);
  float4 acc = reinterpret_cast<const float4*>(res)[i];
  if (shared != nullptr) {
    const float4 s = reinterpret_cast<const float4*>(shared)[i];
    acc.x += s.x;
    acc.y += s.y;
    acc.z += s.z;
    acc.w += s.w;
  }
  float4 sum = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float w = weights[t * 8 + k];
    const float4 e = reinterpret_cast<const float4*>(
        expert_out + (static_cast<std::size_t>(t) * 8 + k) * kHidden)[c];
    sum.x = fmaf(w, e.x, sum.x);
    sum.y = fmaf(w, e.y, sum.y);
    sum.z = fmaf(w, e.z, sum.z);
    sum.w = fmaf(w, e.w, sum.w);
  }
  acc.x += sum.x;
  acc.y += sum.y;
  acc.z += sum.z;
  acc.w += sum.w;
  reinterpret_cast<float4*>(res)[i] = acc;
}

}  // namespace

void Route(const float* logits, const float* bias, std::int32_t* ids,
           float* weights, std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  hipLaunchKernelGGL(RouteKernel, dim3(n_tokens), dim3(256), 0, stream, logits,
                     bias, ids, weights);
}

std::size_t PrefillScratchBytes(std::uint32_t n_tokens) {
  const std::size_t pairs = std::size_t{n_tokens} * kUsed;
  // Sized for the narrowest tile (16 tokens), the largest tile count.
  return kPrefillHeader + 4 * pairs + 12 * std::size_t{TileBound(n_tokens, 16)};
}

std::size_t DecodeScratchBytes() {
  return kDecodeHeader + sizeof(DecodeEntry) * kDecodeMaxEntries;
}

void ExpertGateUpDecode(const void* gate, const void* up, WeightType type,
                        const float* x, const std::int32_t* ids,
                        std::uint32_t n_tokens, float limit, float* act,
                        void* scratch, hipStream_t stream) {
  DecodeDispatch(gate, up, true, type, x, ids, n_tokens, limit, act, scratch,
                 stream);
}

void ExpertDownDecode(const void* down, WeightType type, const float* act,
                      const std::int32_t* ids, std::uint32_t n_tokens,
                      float* out, void* scratch, hipStream_t stream) {
  DecodeDispatch(down, down, false, type, act, ids, n_tokens, 0.0F, out,
                 scratch, stream);
}

void ExpertGateUpPrefill(const void* gate, const void* up, WeightType type,
                         const float* x, const std::int32_t* ids,
                         std::uint32_t n_tokens, float limit, float* act,
                         void* scratch, hipStream_t stream) {
  PrefillDispatch(gate, up, true, type, x, ids, n_tokens, limit, act, scratch,
                  stream);
}

void ExpertDownPrefill(const void* down, WeightType type, const float* act,
                       const std::int32_t* ids, std::uint32_t n_tokens,
                       float* out, void* scratch, hipStream_t stream) {
  PrefillDispatch(down, down, false, type, act, ids, n_tokens, 0.0F, out,
                  scratch, stream);
}

void Combine(float* res, const float* expert_out, const float* weights,
             const float* shared_out, std::uint32_t n_tokens,
             hipStream_t stream) {
  const std::uint32_t n4 = n_tokens * (kHidden / 4);
  if (n4 == 0) {
    return;
  }
  hipLaunchKernelGGL(CombineKernel, dim3((n4 + 255) / 256), dim3(256), 0,
                     stream, res, expert_out, weights, shared_out, n4);
}

}  // namespace gufo::models::ling_3_flash::rocm::moe
