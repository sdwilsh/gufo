// Ling-3.0-flash multi-head latent attention, PLAN section 1. Absorbed form for
// decode and small chunks; non-absorbed prefill (see below) for chunks of more
// than 8 tokens.
//
// All 32 heads share one 576-wide F16 cache row [512 latent | 64 k_pe]; the
// latent half doubles as the value. Prefill and decode use one WMMA kernel:
// a block owns 32 query rows (two 16-row WMMA tiles) and sweeps the cache in
// 16-key tiles, so every cache row is read once per block for all its rows.
//   prefill: 16 consecutive tokens x 2 heads, causal.
//   decode : 1 token x 32 heads over one key chunk (split-K, merged after).
// The softmax statistics and the P.V accumulators stay in F32; Q enters the
// matrix cores as F16 (K and P are F16 too), like llama.cpp's flash path.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"
#include "src/models/ling_3_flash/layout.hpp"

namespace gufo::models::ling_3_flash::rocm::mla {
namespace {

constexpr std::uint32_t kHeadsN = kHeads;
constexpr std::uint32_t kNope = kMlaNope;
constexpr std::uint32_t kRope = kMlaRope;
constexpr std::uint32_t kLatent = kMlaLatent;
constexpr std::uint32_t kWidth = kMlaCacheWidth;
constexpr std::uint32_t kQHead = kMlaQHeadDim;
constexpr std::uint32_t kPairs = kRope / 2;
constexpr float kRmsEps = 1.0e-6F;

// ---------------------------------------------------------------------------
// RoPE (NORM, adjacent pairs). cos/sin come from a double-precision angle so
// long positions do not lose phase to float rounding.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void RopeTable(float* cos_t, float* sin_t,
                                          std::int32_t position,
                                          std::uint32_t i) {
  const double freq =
      pow(static_cast<double>(kMlaRopeBase), -2.0 * i / double{kRope});
  double s = 0.0;
  double c = 0.0;
  sincos(static_cast<double>(position) * freq, &s, &c);
  cos_t[i] = static_cast<float>(c);
  sin_t[i] = static_cast<float>(s);
}

__global__ void LatentNormRopeKernel(const float* __restrict__ kv_a,
                                     const float* __restrict__ gamma,
                                     const std::int32_t* __restrict__ positions,
                                     float* __restrict__ out) {
  __shared__ float red[256];
  __shared__ float cos_t[kPairs];
  __shared__ float sin_t[kPairs];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const float* in = kv_a + std::size_t{t} * kWidth;
  float* o = out + std::size_t{t} * kWidth;
  if (tid < kPairs) {
    RopeTable(cos_t, sin_t, positions[t], tid);
  }
  const float a = in[tid];
  const float b = in[tid + 256];
  red[tid] = a * a + b * b;
  __syncthreads();
  for (std::uint32_t s = 128; s > 0; s >>= 1) {
    if (tid < s) {
      red[tid] += red[tid + s];
    }
    __syncthreads();
  }
  const float scale = rsqrtf(red[0] / static_cast<float>(kLatent) + kRmsEps);
  o[tid] = a * scale * gamma[tid];
  o[tid + 256] = b * scale * gamma[tid + 256];
  if (tid < kPairs) {
    const float x0 = in[kLatent + (2 * tid)];
    const float x1 = in[kLatent + (2 * tid) + 1];
    o[kLatent + (2 * tid)] = (x0 * cos_t[tid]) - (x1 * sin_t[tid]);
    o[kLatent + (2 * tid) + 1] = (x0 * sin_t[tid]) + (x1 * cos_t[tid]);
  }
}

// Row t of `kv` lands at cache row positions[t]: the position is read on the
// device so a captured graph replays for any context length.
__global__ void CacheAppendKernel(__half* __restrict__ cache,
                                  const float* __restrict__ kv,
                                  const std::int32_t* __restrict__ positions,
                                  std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count) {
    const std::size_t t = i / kWidth;
    cache[(std::size_t{static_cast<std::uint32_t>(positions[t])} * kWidth) +
          (i % kWidth)] = __float2half_rn(kv[i]);
  }
}

__global__ void QRopeKernel(const float* __restrict__ q,
                            const std::int32_t* __restrict__ positions,
                            float* __restrict__ q_abs) {
  __shared__ float cos_t[kPairs];
  __shared__ float sin_t[kPairs];
  const std::uint32_t t = blockIdx.x;
  if (threadIdx.x < kPairs) {
    RopeTable(cos_t, sin_t, positions[t], threadIdx.x);
  }
  __syncthreads();
  for (std::uint32_t p = threadIdx.x; p < kHeadsN * kPairs; p += blockDim.x) {
    const std::uint32_t h = p / kPairs;
    const std::uint32_t i = p % kPairs;
    const float* in =
        q + ((std::size_t{t} * kHeadsN + h) * kQHead) + kNope + (2 * i);
    float* o =
        q_abs + ((std::size_t{t} * kHeadsN + h) * kWidth) + kLatent + (2 * i);
    o[0] = (in[0] * cos_t[i]) - (in[1] * sin_t[i]);
    o[1] = (in[0] * sin_t[i]) + (in[1] * cos_t[i]);
  }
}

// ---------------------------------------------------------------------------
// Per-head Q8_0 matrix products shared by q absorption and the v_b up
// projection: y[t][h][row] = sum_k x[t][h][k] * w[h][row][k] (* sigmoid gate).
// A block owns 16 tokens x 64 rows of one head; the 64 x 128 weight chunk is
// dequantised once into LDS and reused by all 16 tokens.
// ---------------------------------------------------------------------------
constexpr std::uint32_t kQ8Tokens = 16;
constexpr std::uint32_t kQ8Rows = 64;
constexpr std::uint32_t kQ8Chunk = 128;
constexpr std::uint32_t kQ8BlockBytes = 34;
constexpr std::uint32_t kWsStride = kQ8Chunk + 1;

template<std::uint32_t kK>
__launch_bounds__(256) __global__
    void Q8HeadGemmKernel(const float* __restrict__ x,
                          const std::uint8_t* __restrict__ w,
                          float* __restrict__ y, std::uint32_t n_tokens,
                          std::uint32_t rows, std::size_t x_t, std::size_t x_h,
                          std::size_t y_t, std::size_t y_h,
                          const float* __restrict__ gate) {
  static_assert(kK % kQ8Chunk == 0, "whole chunks");
  __shared__ float ws[kQ8Rows * kWsStride];
  __shared__ float xs[kQ8Tokens * kQ8Chunk];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t t0 = blockIdx.x * kQ8Tokens;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t r0 = blockIdx.z * kQ8Rows;
  constexpr std::size_t kRowBytes = (kK / 32) * kQ8BlockBytes;
  const std::uint8_t* wh = w + (std::size_t{h} * rows * kRowBytes);

  const std::uint32_t col = tid % kQ8Rows;  // output row of this thread
  const std::uint32_t tg = tid / kQ8Rows;   // tokens tg*4 .. tg*4+3
  float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};

  for (std::uint32_t kc = 0; kc < kK / kQ8Chunk; ++kc) {
    {
      const std::uint32_t row_l = tid / 4;
      const std::uint32_t bb = tid % 4;
      const std::uint8_t* blk = wh + (std::size_t{r0 + row_l} * kRowBytes) +
                                (std::size_t{(kc * 4) + bb} * kQ8BlockBytes);
      const float d = __half2float(*reinterpret_cast<const __half*>(blk));
      const auto* qs = reinterpret_cast<const std::uint16_t*>(blk + 2);
      float* dst = &ws[(row_l * kWsStride) + (bb * 32)];
#pragma unroll
      for (std::uint32_t j = 0; j < 16; ++j) {
        const std::uint32_t pair = qs[j];
        dst[2 * j] = d * static_cast<float>(static_cast<std::int8_t>(pair));
        dst[(2 * j) + 1] =
            d * static_cast<float>(static_cast<std::int8_t>(pair >> 8));
      }
    }
    for (std::uint32_t i = tid; i < kQ8Tokens * kQ8Chunk; i += 256) {
      const std::uint32_t t = i / kQ8Chunk;
      const std::uint32_t k = i % kQ8Chunk;
      xs[i] = (t0 + t < n_tokens)
                  ? x[((t0 + t) * x_t) + (h * x_h) + (kc * kQ8Chunk) + k]
                  : 0.0F;
    }
    __syncthreads();
    for (std::uint32_t k = 0; k < kQ8Chunk; ++k) {
      const float wv = ws[(col * kWsStride) + k];
#pragma unroll
      for (std::uint32_t j = 0; j < 4; ++j) {
        acc[j] = fmaf(wv, xs[(((tg * 4) + j) * kQ8Chunk) + k], acc[j]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (std::uint32_t j = 0; j < 4; ++j) {
    const std::uint32_t t = t0 + (tg * 4) + j;
    if (t < n_tokens) {
      float v = acc[j];
      if (gate != nullptr) {
        v *= 1.0F / (1.0F + expf(-gate[(t * kHeadsN) + h]));
      }
      y[(t * y_t) + (h * y_h) + r0 + col] = v;
    }
  }
}

// ---------------------------------------------------------------------------
// WMMA attention over the shared latent cache.
// ---------------------------------------------------------------------------
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    uint4 u[2];
    v16h f;
  } x;
  const auto* s = reinterpret_cast<const uint4*>(p);
  x.u[0] = s[0];
  x.u[1] = s[1];
  return x.f;
}

constexpr std::uint32_t kKeys = 16;                    // keys per tile
constexpr std::uint32_t kKStride = kWidth + 8;         // K tile row (halves)
constexpr std::uint32_t kVtStride = kKeys + 8;         // V^T / P row (halves)
constexpr std::uint32_t kAttnRows = 32;                // two 16-row WMMA tiles
constexpr std::uint32_t kKSteps = kWidth / 16;         // 36
constexpr std::uint32_t kKStepsPerWave = kKSteps / 4;  // 9
constexpr std::uint32_t kTokensPerBlock = 16;
constexpr std::uint32_t kMaxSplits = 96;
constexpr std::uint32_t kSplitKeys = 256;  // target keys per split

struct SplitPlan {
  std::uint32_t splits;
  std::uint32_t chunk;
};

// Split-K plan of one decode token. Depends on the context only, and runs on
// the device (the launch grid is the maximum plan) so one captured graph
// serves every context length.
__host__ __device__ __forceinline__ SplitPlan
PlanSplits(std::uint32_t context) {
  std::uint32_t splits = (context + kSplitKeys - 1) / kSplitKeys;
  splits = splits < 1U ? 1U : (splits > kMaxSplits ? kMaxSplits : splits);
  std::uint32_t chunk = (context + splits - 1) / splits;
  chunk = (chunk + kKeys - 1) / kKeys * kKeys;
  splits = (context + chunk - 1) / chunk;
  return {splits, chunk};
}

// Batch invariance. The matrix core adds the two products of each key pair
// (2j, 2j+1) with a shared exponent alignment, and a masked key's product
// 0 * V still takes its exponent from V. A row whose last valid key is the
// even member of a pair therefore saw the V row of its masked partner, so its
// result depended on what the cache or the tile fill held there (real rows in
// one chunk layout, zeros in another). Such rows (odd valid count below 16 in
// the tile) get the P.V product again with every pair holding one real key
// and one hard zero (P and V), which does not depend on the partner. Rows with
// an even count, or a full tile, are unaffected by masked content and keep
// the single product; the choice depends only on the row's own position.
__device__ __forceinline__ bool OddTail(std::uint32_t kb, std::uint32_t key_end,
                                        std::uint32_t qpos, bool live) {
  const std::uint32_t last = min(qpos + 1U, key_end);  // exclusive
  if (!live || last <= kb)
    return false;
  const std::uint32_t valid = last - kb;
  return valid < kKeys && (valid & 1U) != 0U;
}

// Decode attention: one token, all 32 heads, one key chunk per block (split-K).
__launch_bounds__(256) __global__
    void AttentionKernel(const float* __restrict__ q,
                         const __half* __restrict__ cache,
                         const std::int32_t* __restrict__ positions,
                         float* __restrict__ out, float* __restrict__ part_m,
                         float* __restrict__ part_l) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  constexpr float kScale = 0.07216878364870323F;  // 1 / sqrt(192)

  // The token's position is device data (see PlanSplits).
  const std::uint32_t context = static_cast<std::uint32_t>(positions[0]) + 1U;
  const SplitPlan plan = PlanSplits(context);
  if (blockIdx.x >= plan.splits) {
    return;  // grid covers the largest plan
  }
  // Row mapping: tile rb, row r is head rb * 16 + r of the one query token.
  const std::uint32_t split = blockIdx.x;
  const std::uint32_t qpos = context - 1;
  const std::uint32_t key_begin = split * plan.chunk;
  const std::uint32_t key_end = min(context, key_begin + plan.chunk);

  __shared__ __half k_lds[kKeys * kKStride];
  __shared__ __half vt_lds[kLatent * kVtStride];
  __shared__ float s_part[4][2][16][17];
  __shared__ __half p_lds[kAttnRows * kVtStride];
  __shared__ float row_scale[kAttnRows];
  __shared__ float row_max_s[kAttnRows];
  __shared__ float row_sum_s[kAttnRows];

  // Q fragments: wave (rb = wave & 1, kq = wave >> 1) owns 9 of the 36 k steps.
  const std::uint32_t s_rb = wave & 1U;
  const std::uint32_t s_kq = wave >> 1U;
  v16h q_frag[kKStepsPerWave];
  {
    const float* q_row = q + (std::size_t{(s_rb * 16) + sub} * kWidth);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const auto* qp =
          reinterpret_cast<const float4*>(q_row + (((s_kq * 9) + ks) * 16));
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = qp[v];
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * kScale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * kScale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * kScale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * kScale);
      }
    }
  }

  // O accumulators: wave owns latent dim tiles wave*4 .. wave*4+3, both rb.
  v8f o_acc[2][4] = {};
  // Softmax row owned by 8 consecutive threads (2 keys each).
  const std::uint32_t rr = tid / 8;
  const std::uint32_t seg = tid % 8;
  const std::uint32_t sm_rb = rr / 16;
  const std::uint32_t sm_r = rr % 16;
  float run_max = -INFINITY;
  float run_sum = 0.0F;

  // One 16-key tile step. `kTail` tiles may hold rows with an odd valid-key
  // count (see OddTail); the sweep below keeps them out of the hot loop.
  const auto tile = [&](std::uint32_t kb, auto tail) {
    constexpr bool kTail = decltype(tail)::value;
    // 1. stage the K tile (contiguous in the cache), zero past the range.
    {
      // All of this thread's loads are issued before the first LDS store, so
      // the tile costs one memory latency instead of one per 16-byte chunk.
      constexpr std::uint32_t kChunks = kKeys * (kWidth / 8);  // 1152
      constexpr std::uint32_t kPer = (kChunks + 255) / 256;    // 5
      uint4 stage[kPer];
#pragma unroll
      for (std::uint32_t i = 0; i < kPer; ++i) {
        const std::uint32_t idx = tid + (i * 256);
        const std::uint32_t key = idx / (kWidth / 8);
        const std::uint32_t c8 = idx % (kWidth / 8);
        stage[i] = make_uint4(0U, 0U, 0U, 0U);
        if (idx < kChunks && kb + key < key_end) {
          stage[i] = *reinterpret_cast<const uint4*>(
              cache + (std::size_t{kb + key} * kWidth) + (c8 * 8));
        }
      }
#pragma unroll
      for (std::uint32_t i = 0; i < kPer; ++i) {
        const std::uint32_t idx = tid + (i * 256);
        if (idx < kChunks) {
          const std::uint32_t key = idx / (kWidth / 8);
          const std::uint32_t c8 = idx % (kWidth / 8);
          *reinterpret_cast<uint4*>(&k_lds[(key * kKStride) + (c8 * 8)]) =
              stage[i];
        }
      }
    }
    __syncthreads();

    // 2. S partials over this wave's k steps.
    {
      v8f s = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const v16h kf =
            LoadFrag(&k_lds[(sub * kKStride) + (((s_kq * 9) + ks) * 16)]);
        s = Wmma(q_frag[ks], kf, s);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_part[s_kq][s_rb][(2 * i) + half_id][sub] = s[i];
      }
    }
    __syncthreads();

    // 3a. V^T for the P.V products (values are the latent half of K).
#pragma unroll
    for (std::uint32_t pass = 0; pass < 2; ++pass) {
      const std::uint32_t d = tid + (pass * 256);
      alignas(16) __half col[kKeys];
#pragma unroll
      for (std::uint32_t key = 0; key < kKeys; ++key) {
        col[key] = k_lds[(key * kKStride) + d];
      }
      auto* dst = reinterpret_cast<uint4*>(&vt_lds[d * kVtStride]);
      const auto* src = reinterpret_cast<const uint4*>(col);
      dst[0] = src[0];
      dst[1] = src[1];
    }
    // 3b. online softmax of this tile.
    {
      float vals[2];
      float part_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < 2; ++m) {
        const std::uint32_t j = (seg * 2) + m;
        const std::uint32_t key = kb + j;
        const bool valid = key < key_end && key <= qpos;
        vals[m] = valid
                      ? (s_part[0][sm_rb][sm_r][j] + s_part[1][sm_rb][sm_r][j] +
                         s_part[2][sm_rb][sm_r][j] + s_part[3][sm_rb][sm_r][j])
                      : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < 8; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float next_max = fmaxf(run_max, part_max);
      const float prior =
          std::isinf(run_max) ? 0.0F : __expf(run_max - next_max);
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < 2; ++m) {
        const float w = std::isinf(vals[m]) ? 0.0F : __expf(vals[m] - next_max);
        const __half ph = __float2half_rn(w);
        part_sum += __half2float(ph);
        p_lds[(rr * kVtStride) + (seg * 2) + m] = ph;
      }
#pragma unroll
      for (std::uint32_t off = 1; off < 8; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      run_max = next_max;
      run_sum = (run_sum * prior) + part_sum;
      if (seg == 0) {
        row_scale[rr] = prior;
      }
    }
    __syncthreads();

    // 4. rescale O and accumulate P.V.
#pragma unroll
    for (std::uint32_t rb = 0; rb < 2; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 4; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }
    if constexpr (!kTail) {
#pragma unroll
      for (std::uint32_t t = 0; t < 4; ++t) {
        const v16h vf =
            LoadFrag(&vt_lds[(((wave * 4) + t) * 16 + sub) * kVtStride]);
#pragma unroll
        for (std::uint32_t rb = 0; rb < 2; ++rb) {
          const v16h pf = LoadFrag(&p_lds[((rb * 16) + sub) * kVtStride]);
          o_acc[rb][t] = Wmma(pf, vf, o_acc[rb][t]);
        }
      }
    } else {
      // Rows with an odd valid count take the pair-isolated product. Some live
      // row has qpos in [kb, kb + 14] with qpos - kb even (block-uniform).
      const bool ragged = OddTail(kb, key_end, qpos, true);
#pragma unroll
      for (std::uint32_t t = 0; t < 4; ++t) {
        const v16h vf =
            LoadFrag(&vt_lds[(((wave * 4) + t) * 16 + sub) * kVtStride]);
#pragma unroll
        for (std::uint32_t rb = 0; rb < 2; ++rb) {
          const v16h pf = LoadFrag(&p_lds[((rb * 16) + sub) * kVtStride]);
          v8f iso = {};
          if (ragged) {
            iso = o_acc[rb][t];
#pragma unroll
            for (std::uint32_t parity = 0; parity < 2; ++parity) {
              v16h pp = pf;
              v16h vv = vf;
#pragma unroll
              for (std::uint32_t i = 0; i < 16; ++i) {
                if ((i & 1U) != parity) {
                  pp[i] = 0;
                  vv[i] = 0;
                }
              }
              iso = Wmma(pp, vv, iso);
            }
          }
          o_acc[rb][t] = Wmma(pf, vf, o_acc[rb][t]);
          if (ragged) {
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i) {
              const std::uint32_t row = (2 * i) + half_id;
              if (OddTail(kb, key_end, qpos, true))
                o_acc[rb][t][i] = iso[i];
            }
          }
        }
      }
    }
    // The next iteration's K staging only touches k_lds, which nobody reads
    // after the barrier above; its first barrier orders p_lds/vt_lds reuse.
  };
  {
    // Tiles every row sees in full (or not at all) come first; the last tiles,
    // where the causal / cache-end boundary falls, run the tail variant.
    const std::uint32_t full_end = key_end;
    std::uint32_t kb = key_begin;
    for (; kb < key_end && kb + kKeys <= full_end; kb += kKeys)
      tile(kb, std::false_type{});
    for (; kb < key_end; kb += kKeys)
      tile(kb, std::true_type{});
  }

  if (seg == 0) {
    row_max_s[rr] = run_max;
    row_sum_s[rr] = run_sum;
  }
  __syncthreads();

  // Unnormalised partials relative to this split's own maximum.
  if (seg == 0) {
    const std::uint32_t head = (sm_rb * 16) + sm_r;
    part_m[(split * kHeadsN) + head] = run_max;
    part_l[(split * kHeadsN) + head] = run_sum;
  }
#pragma unroll
  for (std::uint32_t rb = 0; rb < 2; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < 4; ++t) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t head = (rb * 16) + (2 * i) + half_id;
        out[(((std::size_t{split} * kHeadsN) + head) * kLatent) +
            (((wave * 4) + t) * 16) + sub] = o_acc[rb][t][i];
      }
    }
  }
}

__global__ void DecodeMergeKernel(const float* __restrict__ part_o,
                                  const float* __restrict__ part_m,
                                  const float* __restrict__ part_l,
                                  const std::int32_t* __restrict__ positions,
                                  float* __restrict__ out) {
  const std::uint32_t splits =
      PlanSplits(static_cast<std::uint32_t>(positions[0]) + 1U).splits;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t d = threadIdx.x;
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    if (part_l[(s * kHeadsN) + h] > 0.0F) {
      m = fmaxf(m, part_m[(s * kHeadsN) + h]);
    }
  }
  float total = 0.0F;
  float acc = 0.0F;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float l = part_l[(s * kHeadsN) + h];
    if (l > 0.0F) {
      const float w = __expf(part_m[(s * kHeadsN) + h] - m);
      total += w * l;
      acc += w * part_o[(((std::size_t{s} * kHeadsN) + h) * kLatent) + d];
    }
  }
  out[(std::size_t{h} * kLatent) + d] = total > 0.0F ? acc / total : 0.0F;
}

std::size_t PartialFloats(std::uint32_t splits) {
  return std::size_t{splits} * kHeadsN * (kLatent + 2);
}

// ---------------------------------------------------------------------------
// Non-absorbed prefill (PLAN section 1, R5). For a prefill chunk of n > 8
// tokens the cached latent rows are expanded per head,
//   k_nope[p][h][d] = sum_c latent[p][c] * k_b[h][c][d]
//   v[p][h][e]      = sum_c latent[p][c] * v_b[h][e][c]
// (a pure function of the cached row) and the 32 heads run flash attention with
// q.k over 192 dims (128 nope + the shared 64-wide rope part read from the
// cache) and p.v over 128, 320 MACs per pair against 1088 absorbed. The
// expansion is one WMMA GEMM per key block (F16 operands, F32 accumulation,
// k_b/v_b dequantised once to F16 at load), so a cache row expands to the same
// bits whichever chunk asks for it, whether it was expanded once into the
// session's persistent store (rows below the budget) or into a transient
// two-block buffer per call (rows above it). Keys are folded in absolute
// 2048-key blocks: each block yields (m, l, O) relative to its own maximum,
// computed in parallel over a group of blocks, and a merge kernel folds the
// blocks in absolute order, so the bits do not depend on the group size or on
// the chunking.
// ---------------------------------------------------------------------------
constexpr std::uint32_t kExpKeys = 2048;  // keys expanded per block
constexpr std::uint32_t kNaTile = 16;     // keys per attention tile
constexpr std::uint32_t kNaChains = 4;
constexpr std::uint32_t kExpGroup = 2;  // transient blocks expanded together
constexpr std::size_t kPartBudgetBytes = std::size_t{64} << 20;
constexpr std::uint32_t kNaRows = 64;  // queries per block (4 waves x 16)
constexpr std::uint32_t kNaDim = kNope + kRope;    // 192
constexpr std::uint32_t kNaDSteps = kNaDim / 16;   // 12
constexpr std::uint32_t kNaKStride = kNaDim + 8;   // K tile row (halves)
constexpr std::uint32_t kNaVStride = kNaTile + 8;  // V^T tile row (halves)
constexpr std::uint32_t kNaWRows = kNope + kMlaValueDim;  // 256 rows per head

// W16[h][n][c]: n < 128 -> k_b[h][c][n] (transposed), n >= 128 ->
// v_b[h][n-128][c].
__global__ void BuildW16Kernel(const std::uint8_t* __restrict__ k_b,
                               const std::uint8_t* __restrict__ v_b,
                               __half* __restrict__ w16) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= std::size_t{kHeadsN} * kNaWRows * kLatent) {
    return;
  }
  const std::uint32_t c = i % kLatent;
  const std::uint32_t n = (i / kLatent) % kNaWRows;
  const std::uint32_t h = i / (std::size_t{kNaWRows} * kLatent);
  constexpr std::size_t kHeadElems = std::size_t{kLatent} * kNope;  // 65536
  const std::uint8_t* base;
  std::size_t j;
  if (n < kNope) {
    base = k_b;
    j = std::size_t{c} * kNope + n;
  } else {
    base = v_b;
    j = std::size_t{n - kNope} * kLatent + c;
  }
  j += std::size_t{h} * kHeadElems;
  const std::uint8_t* blk = base + (j / 32) * kQ8BlockBytes;
  const float d = __half2float(*reinterpret_cast<const __half*>(blk));
  const auto q = static_cast<std::int8_t>(blk[2 + (j % 32)]);
  w16[i] = __float2half_rn(d * static_cast<float>(q));
}

__global__ void QPrepKernel(const float* __restrict__ q,
                            const std::int32_t* __restrict__ positions,
                            float* __restrict__ q_out) {
  __shared__ float cos_t[kPairs];
  __shared__ float sin_t[kPairs];
  const std::uint32_t t = blockIdx.x;
  if (threadIdx.x < kPairs) {
    RopeTable(cos_t, sin_t, positions[t], threadIdx.x);
  }
  __syncthreads();
  for (std::uint32_t p = threadIdx.x; p < kHeadsN * kQHead; p += blockDim.x) {
    const std::uint32_t h = p / kQHead;
    const std::uint32_t d = p % kQHead;
    const float* in = q + ((std::size_t{t} * kHeadsN + h) * kQHead);
    float* o = q_out + ((std::size_t{t} * kHeadsN + h) * kQHead);
    if (d < kNope) {
      o[d] = in[d];
    } else if (((d - kNope) & 1U) == 0U) {
      const std::uint32_t i = (d - kNope) / 2;
      const float x0 = in[d];
      const float x1 = in[d + 1];
      o[d] = (x0 * cos_t[i]) - (x1 * sin_t[i]);
      o[d + 1] = (x0 * sin_t[i]) + (x1 * cos_t[i]);
    }
  }
}

// Expansion GEMM, out[m][n] = sum_c A[m][c] B[n][c] with c contiguous in both
// operands. A block owns 128 keys x one head x (K mode: 128 d | V mode: 128 e)
// and sweeps c in 64-wide chunks through LDS, the next chunk's global loads
// in flight while the current one multiplies. Eight waves in a 2 x 4 grid,
// each 64 x 32 outputs (4 x 2 WMMA tiles). Keys at or past `end` read as zero
// rows, so everything the attention reads beyond the cache end is finite
// zeros. Output index of key k is idx = k - obase (obase a multiple of 16),
// written only when idx < out_rows:
//   K mode: out[(h * out_rows + idx) * 128 + d]            (F16)
//   V mode: out[((h * out_rows / 16 + idx / 16) * 128 + e) * 16 + slot(idx)]:
//           V^T in 16-key tiles, each tile's 128 x 16 values contiguous (one
//           4 KiB read per attention tile instead of 128 strided 32-byte
//           pieces), keys permuted inside the tile so slot j holds key 2j and
//           slot 8 + j key 2j + 1 (the attention builds its P fragments in
//           that slot order).
// blockIdx.z selects a transient slot: its keys start kExpKeys later and its
// output moves by `slot_stride` halves.
constexpr std::uint32_t kExpChunk = 64;
constexpr std::uint32_t kExpStride = kExpChunk + 8;
constexpr std::uint32_t kExpTile = 128;  // keys per block

template<bool kVMode>
__launch_bounds__(256) __global__
    void ExpandKernel(const __half* __restrict__ cache,
                      const __half* __restrict__ w16, __half* __restrict__ out,
                      std::uint32_t start, std::uint32_t end,
                      std::uint32_t obase, std::uint32_t out_rows,
                      std::size_t slot_stride) {
  __shared__ __half lat_s[kExpTile * kExpStride];
  __shared__ __half w_s[128 * kExpStride];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t slot_keys = blockIdx.z * kExpKeys;
  const std::uint32_t key0 = start + slot_keys + (blockIdx.x * kExpTile);
  const std::uint32_t idx0 = key0 - (obase + slot_keys);
  out += blockIdx.z * slot_stride;
  const std::uint32_t wm = (wave & 1U) * 64U;   // wave offset along M
  const std::uint32_t wn = (wave >> 1U) * 32U;  // wave offset along N
  const __half* wh =
      w16 + ((std::size_t{h} * kNaWRows + (kVMode ? kNope : 0U)) * kLatent);

  // Each thread moves 4 uint4 of each tile per chunk: rows (tid >> 3) + 32 j.
  const std::uint32_t c8 = tid & 7U;
  const std::uint32_t r0 = tid >> 3U;
  uint4 lat_r[4];
  uint4 w_r[4];
  const auto fetch = [&](std::uint32_t kc) {
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      const std::uint32_t r = r0 + (j * 32U);
      lat_r[j] = make_uint4(0U, 0U, 0U, 0U);
      if (key0 + r < end) {
        lat_r[j] = *reinterpret_cast<const uint4*>(
            cache + (std::size_t{key0 + r} * kWidth) + (kc * kExpChunk) +
            (c8 * 8));
      }
      w_r[j] = *reinterpret_cast<const uint4*>(wh + (std::size_t{r} * kLatent) +
                                               (kc * kExpChunk) + (c8 * 8));
    }
  };

  v8f acc[4][2] = {};
  fetch(0);
  for (std::uint32_t kc = 0; kc < kLatent / kExpChunk; ++kc) {
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      const std::uint32_t r = r0 + (j * 32U);
      *reinterpret_cast<uint4*>(&lat_s[(r * kExpStride) + (c8 * 8)]) = lat_r[j];
      *reinterpret_cast<uint4*>(&w_s[(r * kExpStride) + (c8 * 8)]) = w_r[j];
    }
    __syncthreads();
    if (kc + 1 < kLatent / kExpChunk) {
      fetch(kc + 1);
    }
#pragma unroll
    for (std::uint32_t ks = 0; ks < kExpChunk / 16; ++ks) {
      v16h af[4];
      v16h bf[2];
      // M operand: 4 tiles of 16 rows; N operand: 2 tiles.
#pragma unroll
      for (std::uint32_t i = 0; i < 4; ++i) {
        af[i] = LoadFrag(
            kVMode ? &w_s[((wm + (i * 16) + sub) * kExpStride) + (ks * 16)]
                   : &lat_s[((wm + (i * 16) + sub) * kExpStride) + (ks * 16)]);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 2; ++i) {
        bf[i] = LoadFrag(
            kVMode ? &lat_s[((wn + (i * 16) + sub) * kExpStride) + (ks * 16)]
                   : &w_s[((wn + (i * 16) + sub) * kExpStride) + (ks * 16)]);
      }
#pragma unroll
      for (std::uint32_t mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (std::uint32_t ni = 0; ni < 2; ++ni) {
          acc[mi][ni] = Wmma(af[mi], bf[ni], acc[mi][ni]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (std::uint32_t mi = 0; mi < 4; ++mi) {
#pragma unroll
    for (std::uint32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const __half value = __float2half_rn(acc[mi][ni][i]);
        if constexpr (kVMode) {
          const std::uint32_t e = wm + (mi * 16) + row;
          const std::uint32_t idx = idx0 + wn + (ni * 16) + sub;
          const std::uint32_t slot = ((idx & 1U) << 3) | ((idx & 15U) >> 1U);
          if (idx < out_rows) {
            out[((((std::size_t{h} * (out_rows / kNaTile)) + (idx / kNaTile)) *
                  kMlaValueDim) +
                 e) *
                    kNaTile +
                slot] = value;
          }
        } else {
          const std::uint32_t idx = idx0 + wm + (mi * 16) + row;
          const std::uint32_t d = wn + (ni * 16) + sub;
          if (idx < out_rows) {
            out[((std::size_t{h} * out_rows + idx) * kNope) + d] = value;
          }
        }
      }
    }
  }
}

// Value of `x` in the lane 16 away (rows of 16 lanes swap): one VALU op, no
// LDS round trip.
__device__ __forceinline__ std::uint32_t SwapRows(std::uint32_t x) {
  return __builtin_amdgcn_permlanex16(x, x, 0x76543210U, 0xfedcba98U, false,
                                      false);
}

__device__ __forceinline__ std::uint32_t PackHalves(const __half a,
                                                    const __half b) {
  return std::uint32_t{__half_as_ushort(a)} |
         (std::uint32_t{__half_as_ushort(b)} << 16U);
}

// Where the expanded keys of a group of blocks live: absolute rows of the
// session's persistent store (blocks below persistent_blocks) or slots of the
// per-call transient buffer.
struct NaBlocks {
  const __half* kx_persistent;
  const __half* vt_persistent;
  std::uint32_t persistent_rows;
  std::uint32_t persistent_blocks;
  const __half* kx_transient;
  const __half* vt_transient;
  std::size_t slot_elems;  // halves per transient slot of K (and of V)
};

// Flash attention over one expanded 2048-key block (blockIdx.z of the group). A
// block owns 64 consecutive queries (four waves, 16 each) of one head. Scores
// are computed transposed (S^T = K . Q^T, keys on the accumulator rows) so a
// lane holds 16 scores of one query: the softmax statistics are per lane (plus
// one exchange with the lane 16 away) and the P fragments for P.V need no trip
// through LDS. O is accumulated transposed as well (O^T[e][query]).
//
// Batch invariance. Every query sees the same absolute sequence of 16-key
// tiles (and P.V products) from key 0 whatever chunk it arrived in.
// The WMMA adds the two products of each key slot pair with one exponent
// alignment, so a masked key (P = 0) takes its exponent from V and perturbs a
// real partner. A 16-key product in which a query has some but not all keys
// valid is therefore recomputed with every slot pair holding one real key and
// one hard zero (P and V), once per slot parity, and a product with no valid
// key leaves the accumulator untouched. Both depend only on the query's own
// position; fully valid products are unaffected by masked content.
__launch_bounds__(128) __global__ void NaAttentionKernel(
    const float* __restrict__ q, const __half* __restrict__ cache,
    const NaBlocks blocks, std::uint32_t first_pos, std::uint32_t n_tokens,
    std::uint32_t b0, float* __restrict__ part_o, float* __restrict__ part_m,
    float* __restrict__ part_l) {
  constexpr float kScale = 0.07216878364870323F;  // 1 / sqrt(192)
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t q0 = blockIdx.x * kNaRows;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t total = first_pos + n_tokens;
  const std::uint32_t b = b0 + blockIdx.z;
  const std::uint32_t kb0 = b * kExpKeys;
  const std::uint32_t blk_end = first_pos + min(q0 + kNaRows, n_tokens);
  if (kb0 >= blk_end) {
    return;  // block-uniform: no query here reaches this key block
  }
  const std::uint32_t block_end = min(kb0 + kExpKeys, blk_end);
  // Persistent rows live at their absolute index; transient ones in the slot
  // of this block, relative to its first key.
  const bool persistent = b < blocks.persistent_blocks;
  const __half* kx =
      persistent
          ? blocks.kx_persistent
          : blocks.kx_transient + (std::size_t{blockIdx.z} * blocks.slot_elems);
  const __half* vt =
      persistent
          ? blocks.vt_persistent
          : blocks.vt_transient + (std::size_t{blockIdx.z} * blocks.slot_elems);
  const std::uint32_t rows = persistent ? blocks.persistent_rows : kExpKeys;
  const std::uint32_t origin = persistent ? 0U : kb0;

  const std::uint32_t row = q0 + (wave * 16U) + sub;
  const bool live = row < n_tokens;
  const std::uint32_t qpos = first_pos + row;
  const std::uint32_t last_key = live ? qpos + 1U : 0U;  // exclusive
  const std::uint32_t wave_end =
      q0 + (wave * 16U) < n_tokens
          ? first_pos + min(q0 + (wave * 16U) + 16U, n_tokens)
          : 0U;

  __shared__ __half k_lds[kNaTile * kNaKStride];
  __shared__ __half v_lds[kMlaValueDim * kNaVStride];

  // Q enters the matrix cores as F16 scaled by 1/sqrt(192). The 8 nope k steps
  // stay in registers; the 4 rope steps sit in LDS (one 16-row slab per wave)
  // and are read per use, which keeps registers for the accumulators.
  constexpr std::uint32_t kNopeSteps = kNope / 16;
  constexpr std::uint32_t kQStride = kRope + 8;
  __shared__ __half q_lds[kNaRows * kQStride];
  v16h q_frag[kNopeSteps];
  {
    const float* q_row =
        q + ((std::size_t{live ? row : 0U} * kHeadsN + h) * kNaDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kNaDSteps; ++ks) {
      const auto* qp = reinterpret_cast<const float4*>(q_row + (ks * 16));
      alignas(16) __half packed[16];
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.F, 0.F, 0.F, 0.F);
        packed[(v * 4) + 0] = __float2half_rn(f.x * kScale);
        packed[(v * 4) + 1] = __float2half_rn(f.y * kScale);
        packed[(v * 4) + 2] = __float2half_rn(f.z * kScale);
        packed[(v * 4) + 3] = __float2half_rn(f.w * kScale);
      }
      if (ks < kNopeSteps) {
#pragma unroll
        for (std::uint32_t i = 0; i < 16; ++i) {
          q_frag[ks][i] = static_cast<_Float16>(packed[i]);
        }
      } else if (half_id == 0U) {
        auto* d4 =
            reinterpret_cast<uint4*>(&q_lds[(((wave * 16U) + sub) * kQStride) +
                                            ((ks - kNopeSteps) * 16)]);
        const auto* s4 = reinterpret_cast<const uint4*>(packed);
        d4[0] = s4[0];
        d4[1] = s4[1];
      }
    }
  }
  __syncthreads();

  v8f o_acc[8] = {};
  float run_max = -INFINITY;
  float run_sum = 0.0F;  // this lane's share (lane and lane ^ 16 add up)
  // The next tile's K/V rows are fetched into registers while the current
  // tile multiplies (the serial walk over a block's keys is latency bound);
  // the data and the arithmetic are the same as staging in the loop body.
  struct Staged {
    uint4 k[2];
    uint4 pe;
    uint4 v[2];
  };
  const auto fetch = [&](std::uint32_t kb) {
    Staged r;
#pragma unroll
    for (std::uint32_t j = 0; j < 2; ++j) {
      const std::uint32_t idx = tid + (j * 128U);
      const std::uint32_t key = idx / (kNope / 8);
      const std::uint32_t c8 = idx % (kNope / 8);
      r.k[j] = *reinterpret_cast<const uint4*>(
          kx + ((std::size_t{h} * rows + (kb - origin) + key) * kNope) +
          (c8 * 8));
      // The tile's V^T (128 x 16) is contiguous: unit idx = (e, half).
      r.v[j] = *reinterpret_cast<const uint4*>(
          vt +
          (((std::size_t{h} * (rows / kNaTile)) + ((kb - origin) / kNaTile)) *
           (kMlaValueDim * kNaTile)) +
          (idx * 8));
    }
    const std::uint32_t key = tid / (kRope / 8);
    const std::uint32_t c8 = tid % (kRope / 8);
    r.pe = make_uint4(0U, 0U, 0U, 0U);
    if (kb + key < total) {
      r.pe = *reinterpret_cast<const uint4*>(
          cache + (std::size_t{kb + key} * kWidth) + kLatent + (c8 * 8));
    }
    return r;
  };
  static_assert(kNaTile * (kNope / 8) == 256 && kNaTile * (kRope / 8) == 128 &&
                    kMlaValueDim * (kNaTile / 8) == 256,
                "register staging assumes 128 threads and 16-key tiles");
  Staged cur = fetch(kb0);
  for (std::uint32_t kb = kb0; kb < block_end; kb += kNaTile) {
    // Stage the K tile (nope from the expansion, rope from the cache) and the
    // V^T tile. Keys at or past the cache end are zero.
#pragma unroll
    for (std::uint32_t j = 0; j < 2; ++j) {
      const std::uint32_t idx = tid + (j * 128U);
      *reinterpret_cast<uint4*>(&k_lds[((idx / (kNope / 8)) * kNaKStride) +
                                       ((idx % (kNope / 8)) * 8)]) = cur.k[j];
      *reinterpret_cast<uint4*>(&v_lds[((idx / (kNaTile / 8)) * kNaVStride) +
                                       ((idx % (kNaTile / 8)) * 8)]) = cur.v[j];
    }
    *reinterpret_cast<uint4*>(&k_lds[((tid / (kRope / 8)) * kNaKStride) +
                                     kNope + ((tid % (kRope / 8)) * 8)]) =
        cur.pe;
    __syncthreads();
    if (kb + kNaTile < block_end) {
      cur = fetch(kb + kNaTile);
    }

    if (kb < wave_end) {
      // S^T for the 16-key tile: lane holds keys kb + 2i + half_id of query
      // `sub`. Four independent accumulation chains (3 k steps each) hide the
      // WMMA latency. The source fixes the sum order. Fast-math may still
      // reassociate it, but identically for every query of one binary, so
      // layout invariance does not depend on it.
      v8f part[kNaChains] = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kNaDSteps; ++ks) {
        const v16h kf = LoadFrag(&k_lds[(sub * kNaKStride) + (ks * 16)]);
        const v16h qf =
            ks < kNopeSteps
                ? q_frag[ks]
                : LoadFrag(&q_lds[(((wave * 16U) + sub) * kQStride) +
                                  ((ks - kNopeSteps) * 16)]);
        part[ks % kNaChains] = Wmma(kf, qf, part[ks % kNaChains]);
      }
      const v8f s = (part[0] + part[1]) + (part[2] + part[3]);
      float vals[8];
      float tile_max = -INFINITY;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t key = kb + (2 * i) + half_id;
        vals[i] = key < last_key ? s[i] : -INFINITY;
        tile_max = fmaxf(tile_max, vals[i]);
      }
      tile_max =
          fmaxf(tile_max, __uint_as_float(SwapRows(__float_as_uint(tile_max))));
      const float next_max = fmaxf(run_max, tile_max);
      const float prior =
          std::isinf(run_max) ? 0.0F : __expf(run_max - next_max);
      float part_sum = 0.0F;
      std::uint32_t own[4];
      {
        __half ph[8];
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const float w =
              std::isinf(vals[i]) ? 0.0F : __expf(vals[i] - next_max);
          ph[i] = __float2half_rn(w);
          part_sum += __half2float(ph[i]);
        }
#pragma unroll
        for (std::uint32_t j = 0; j < 4; ++j) {
          own[j] = PackHalves(ph[2 * j], ph[(2 * j) + 1]);
        }
      }
      run_max = next_max;
      run_sum = (run_sum * prior) + part_sum;
#pragma unroll
      for (std::uint32_t t = 0; t < 8; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[t][i] *= prior;
        }
      }

      // P fragment (slot j = key 2j, slot 8 + j = key 2j + 1).
      union {
        std::uint32_t u[8];
        v16h f;
      } pf;
#pragma unroll
      for (std::uint32_t j = 0; j < 4; ++j) {
        const std::uint32_t other = SwapRows(own[j]);
        pf.u[j] = half_id == 0U ? own[j] : other;
        pf.u[4 + j] = half_id == 0U ? other : own[j];
      }
      const std::uint32_t valid = last_key > kb ? min(last_key - kb, 16U) : 0U;
      // Empty or partial for some query that is written out; the lanes of
      // rows past the chunk end (valid 0) are never stored, and a WMMA output
      // column depends only on its own P column, so they do not take part.
      const bool any_special = __any(live && valid != 16U);
#pragma unroll
      for (std::uint32_t t = 0; t < 8; ++t) {
        const v16h vf = LoadFrag(&v_lds[(((t * 16) + sub) * kNaVStride)]);
        if (!any_special) {
          o_acc[t] = Wmma(vf, pf.f, o_acc[t]);
        } else {
          const v8f old = o_acc[t];
          const v8f full = Wmma(vf, pf.f, old);
          v8f iso = old;
#pragma unroll
          for (std::uint32_t parity = 0; parity < 2; ++parity) {
            v16h pp = pf.f;
            v16h vv = vf;
#pragma unroll
            for (std::uint32_t i = 0; i < 16; ++i) {
              if ((i & 1U) != parity) {
                pp[i] = 0;
                vv[i] = 0;
              }
            }
            iso = Wmma(vv, pp, iso);
          }
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            o_acc[t][i] = valid == 16U  ? full[i]
                          : valid == 0U ? old[i]
                                        : iso[i];
          }
        }
      }
    }
    __syncthreads();
  }

  // Unnormalised partial of this block, relative to the row's block maximum.
  // (Cross-half exchange before any divergent branch; ds_bpermute, not
  // permlanex16, which the compiler may sink into the divergent region.)
  const float block_sum = run_sum + __shfl_xor(run_sum, 16);
  if (!live) {
    return;
  }
  const std::size_t pi =
      ((std::size_t{blockIdx.z} * n_tokens + row) * kHeadsN) + h;
  if (half_id == 0U) {
    part_m[pi] = run_max;
    part_l[pi] = block_sum;
  }
#pragma unroll
  for (std::uint32_t t = 0; t < 8; ++t) {
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const std::uint32_t e = (t * 16) + (2 * i) + half_id;
      part_o[(pi * kMlaValueDim) + e] = o_acc[t][i];
    }
  }
}

// Folds the blocks b0 .. b0 + g - 1 of one (row, head) into the running
// (M, L, O) in absolute block order, then (last group) normalises, applies
// the gate and writes the output. Blocks the row cannot see (all keys past
// its position) are skipped.
__launch_bounds__(128) __global__
    void NaMergeKernel(const float* __restrict__ part_o,
                       const float* __restrict__ part_m,
                       const float* __restrict__ part_l,
                       float* __restrict__ fold, const float* __restrict__ gate,
                       float* __restrict__ out, std::uint32_t first_pos,
                       std::uint32_t n_tokens, std::uint32_t b0,
                       std::uint32_t g, bool last) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t e = threadIdx.x;
  const std::uint32_t qpos = first_pos + row;
  const std::size_t ri = (std::size_t{row} * kHeadsN) + h;
  float* fold_ml = fold;  // [row][h][2], then O [row][h][128]
  float* fold_o = fold + (std::size_t{n_tokens} * kHeadsN * 2);
  float m = -INFINITY;
  float l = 0.0F;
  float o = 0.0F;
  if (b0 != 0) {
    m = fold_ml[ri * 2];
    l = fold_ml[(ri * 2) + 1];
    o = fold_o[(ri * kMlaValueDim) + e];
  }
  for (std::uint32_t z = 0; z < g; ++z) {
    if ((b0 + z) * kExpKeys > qpos) {
      break;
    }
    const std::size_t pi = ((std::size_t{z} * n_tokens + row) * kHeadsN) + h;
    const float mb = part_m[pi];
    const float lb = part_l[pi];
    const float next = fmaxf(m, mb);
    const float a = std::isinf(m) ? 0.0F : __expf(m - next);
    const float c = __expf(mb - next);
    l = (l * a) + (lb * c);
    o = (o * a) + (part_o[(pi * kMlaValueDim) + e] * c);
    m = next;
  }
  if (!last) {
    if (e == 0) {
      fold_ml[ri * 2] = m;
      fold_ml[(ri * 2) + 1] = l;
    }
    fold_o[(ri * kMlaValueDim) + e] = o;
    return;
  }
  float inv = l > 0.0F ? 1.0F / l : 0.0F;
  if (gate != nullptr) {
    inv *= 1.0F / (1.0F + expf(-gate[ri]));
  }
  out[(ri * kMlaValueDim) + e] = o * inv;
}

}  // namespace

void LatentNormRope(const float* kv_a, const float* gamma,
                    const std::int32_t* positions, float* kv_out,
                    std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  LatentNormRopeKernel<<<n_tokens, 256, 0, stream>>>(kv_a, gamma, positions,
                                                     kv_out);
}

void CacheAppend(__half* layer_cache, const float* kv,
                 const std::int32_t* positions, std::uint32_t n_tokens,
                 hipStream_t stream) {
  const std::size_t count = std::size_t{n_tokens} * kWidth;
  if (count == 0) {
    return;
  }
  CacheAppendKernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                      stream>>>(layer_cache, kv, positions, count);
}

void QAbsorb(const float* q, const void* k_b, const std::int32_t* positions,
             float* q_abs, std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const dim3 grid((n_tokens + kQ8Tokens - 1) / kQ8Tokens, kHeadsN,
                  kLatent / kQ8Rows);
  Q8HeadGemmKernel<kNope><<<grid, 256, 0, stream>>>(
      q, static_cast<const std::uint8_t*>(k_b), q_abs, n_tokens, kLatent,
      std::size_t{kHeadsN} * kQHead, kQHead, std::size_t{kHeadsN} * kWidth,
      kWidth, nullptr);
  QRopeKernel<<<n_tokens, 256, 0, stream>>>(q, positions, q_abs);
}

std::size_t DecodeScratchBytes(std::uint32_t /*max_context*/) {
  // Fixed layout for the largest plan: the plan itself is chosen on the device.
  return PartialFloats(kMaxSplits) * sizeof(float);
}

void DecodeAttention(const float* q_abs, const __half* layer_cache,
                     const std::int32_t* position, float* out_latent,
                     void* scratch, hipStream_t stream) {
  auto* part_o = static_cast<float*>(scratch);
  float* part_m = part_o + (std::size_t{kMaxSplits} * kHeadsN * kLatent);
  float* part_l = part_m + (std::size_t{kMaxSplits} * kHeadsN);
  AttentionKernel<<<kMaxSplits, 256, 0, stream>>>(q_abs, layer_cache, position,
                                                  part_o, part_m, part_l);
  DecodeMergeKernel<<<kHeadsN, kLatent, 0, stream>>>(part_o, part_m, part_l,
                                                     position, out_latent);
}

std::size_t ExpansionWeightBytes() {
  return std::size_t{kHeadsN} * kNaWRows * kLatent * sizeof(__half);
}

void BuildExpansionWeights(const void* k_b, const void* v_b, __half* w16,
                           hipStream_t stream) {
  const std::size_t count = std::size_t{kHeadsN} * kNaWRows * kLatent;
  BuildW16Kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                   stream>>>(static_cast<const std::uint8_t*>(k_b),
                             static_cast<const std::uint8_t*>(v_b), w16);
}

namespace {
constexpr std::size_t kSlotElems = std::size_t{kHeadsN} * kExpKeys * kNope;

// Blocks per attention group: as many as the partial budget holds for n rows.
std::uint32_t PartialGroup(std::uint32_t n_tokens) {
  const std::size_t block =
      std::size_t{n_tokens} * kHeadsN * (kMlaValueDim + 2) * sizeof(float);
  return static_cast<std::uint32_t>(
      std::clamp<std::size_t>(kPartBudgetBytes / block, 1, 32));
}

std::size_t PartialBytes(std::uint32_t max_batch) {
  return std::max(kPartBudgetBytes, std::size_t{max_batch} * kHeadsN *
                                        (kMlaValueDim + 2) * sizeof(float));
}
}  // namespace

std::size_t NonAbsorbedScratchBytes(std::uint32_t max_batch) {
  const std::size_t expanded = 2 * kExpGroup * kSlotElems * sizeof(__half);
  const std::size_t fold = std::size_t{std::max(max_batch, 1U)} * kHeadsN *
                           (kMlaValueDim + 2) * sizeof(float);
  return expanded + fold + PartialBytes(std::max(max_batch, 1U));
}

std::size_t ExpandedLayerBytes(std::uint32_t rows) {
  return std::size_t{rows} * kHeadsN * kNaWRows * sizeof(__half);
}

void ExpandRows(const __half* layer_cache, const __half* w16,
                __half* layer_expanded, std::uint32_t rows, std::uint32_t start,
                std::uint32_t end, hipStream_t stream) {
  end = std::min(end, rows);
  if (start >= end) {
    return;
  }
  __half* vt = layer_expanded + (std::size_t{kHeadsN} * rows * kNope);
  const dim3 grid((end - start + kExpTile - 1) / kExpTile, kHeadsN, 1);
  ExpandKernel<false><<<grid, 256, 0, stream>>>(
      layer_cache, w16, layer_expanded, start, end, 0, rows, 0);
  ExpandKernel<true>
      <<<grid, 256, 0, stream>>>(layer_cache, w16, vt, start, end, 0, rows, 0);
}

void QPrepare(const float* q, const std::int32_t* positions, float* q_rope,
              std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  QPrepKernel<<<n_tokens, 256, 0, stream>>>(q, positions, q_rope);
}

void PrefillAttentionNonAbsorbed(
    const float* q_rope, const __half* layer_cache, const __half* w16,
    const __half* layer_expanded, std::uint32_t expanded_rows,
    const float* gate_logits, std::uint32_t first_pos, std::uint32_t n_tokens,
    float* out, void* scratch, hipStream_t stream, PrefillTimes* times) {
  if (n_tokens == 0) {
    return;
  }
  const std::uint32_t total = first_pos + n_tokens;
  const std::uint32_t n_blocks = (total + kExpKeys - 1) / kExpKeys;
  auto* kx_t = static_cast<__half*>(scratch);
  __half* vt_t = kx_t + (kExpGroup * kSlotElems);
  auto* part_o = reinterpret_cast<float*>(vt_t + (kExpGroup * kSlotElems));
  const std::uint32_t group_cap = PartialGroup(n_tokens);
  float* part_ml =
      part_o + (std::size_t{group_cap} * n_tokens * kHeadsN * kMlaValueDim);
  float* part_m = part_ml;
  float* part_l = part_m + (std::size_t{group_cap} * n_tokens * kHeadsN);
  float* fold = reinterpret_cast<float*>(
      reinterpret_cast<char*>(vt_t + (kExpGroup * kSlotElems)) +
      PartialBytes(std::max(n_tokens, 1U)));
  NaBlocks blocks{};
  blocks.persistent_rows = expanded_rows;
  blocks.persistent_blocks = expanded_rows / kExpKeys;
  blocks.kx_persistent = layer_expanded;
  blocks.vt_persistent =
      layer_expanded + (std::size_t{kHeadsN} * expanded_rows * kNope);
  blocks.kx_transient = kx_t;
  blocks.vt_transient = vt_t;
  blocks.slot_elems = kSlotElems;
  const dim3 attn_grid_xy((n_tokens + kNaRows - 1) / kNaRows, kHeadsN);
  hipEvent_t ev[2] = {nullptr, nullptr};
  if (times != nullptr) {
    (void)hipEventCreate(&ev[0]);
    (void)hipEventCreate(&ev[1]);
  }
  // With `times`, each launch is bracketed by events and waited for.
  const auto timed = [&](double* sink, auto&& launch) {
    if (times == nullptr) {
      launch();
      return;
    }
    (void)hipEventRecord(ev[0], stream);
    launch();
    (void)hipEventRecord(ev[1], stream);
    (void)hipEventSynchronize(ev[1]);
    float ms = 0.0F;
    (void)hipEventElapsedTime(&ms, ev[0], ev[1]);
    *sink += ms;
  };
  double unused = 0.0;
  for (std::uint32_t b = 0; b < n_blocks;) {
    const bool persistent = b < blocks.persistent_blocks;
    std::uint32_t g = std::min(group_cap, n_blocks - b);
    if (persistent) {
      g = std::min(g, blocks.persistent_blocks - b);
    } else {
      g = std::min(g, kExpGroup);
      const std::uint32_t span = std::min(g * kExpKeys, total - (b * kExpKeys));
      const dim3 grid((std::min(span, kExpKeys) + kExpTile - 1) / kExpTile,
                      kHeadsN, g);
      timed(times != nullptr ? &times->expand_ms : &unused, [&] {
        ExpandKernel<false><<<grid, 256, 0, stream>>>(
            layer_cache, w16, kx_t, b * kExpKeys, total, b * kExpKeys, kExpKeys,
            kSlotElems);
        ExpandKernel<true><<<grid, 256, 0, stream>>>(
            layer_cache, w16, vt_t, b * kExpKeys, total, b * kExpKeys, kExpKeys,
            kSlotElems);
      });
    }
    timed(times != nullptr ? &times->attn_ms : &unused, [&] {
      NaAttentionKernel<<<dim3(attn_grid_xy.x, attn_grid_xy.y, g), 128, 0,
                          stream>>>(q_rope, layer_cache, blocks, first_pos,
                                    n_tokens, b, part_o, part_m, part_l);
    });
    timed(times != nullptr ? &times->merge_ms : &unused, [&] {
      NaMergeKernel<<<dim3(n_tokens, kHeadsN), 128, 0, stream>>>(
          part_o, part_m, part_l, fold, gate_logits, out, first_pos, n_tokens,
          b, g, b + g == n_blocks);
    });
    b += g;
  }
  if (times != nullptr) {
    (void)hipEventDestroy(ev[0]);
    (void)hipEventDestroy(ev[1]);
  }
}

void VbUpProjGate(const float* lat, const void* v_b, const float* gate_logits,
                  float* out, std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const dim3 grid((n_tokens + kQ8Tokens - 1) / kQ8Tokens, kHeadsN,
                  kMlaValueDim / kQ8Rows);
  Q8HeadGemmKernel<kLatent><<<grid, 256, 0, stream>>>(
      lat, static_cast<const std::uint8_t*>(v_b), out, n_tokens, kMlaValueDim,
      std::size_t{kHeadsN} * kLatent, kLatent,
      std::size_t{kHeadsN} * kMlaValueDim, kMlaValueDim, gate_logits);
}

}  // namespace gufo::models::ling_3_flash::rocm::mla
