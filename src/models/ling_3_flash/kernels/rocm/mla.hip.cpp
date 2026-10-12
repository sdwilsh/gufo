// Ling-3.0-flash multi-head latent attention (absorbed form), PLAN section 1.
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

__global__ void CacheAppendKernel(__half* __restrict__ cache,
                                  const float* __restrict__ kv,
                                  std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count) {
    cache[i] = __float2half_rn(kv[i]);
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

template<bool kDecode>
__launch_bounds__(256) __global__
    void AttentionKernel(const float* __restrict__ q,
                         const __half* __restrict__ cache,
                         std::uint32_t first_pos, std::uint32_t n_tokens,
                         std::uint32_t context, std::uint32_t chunk,
                         float* __restrict__ out, float* __restrict__ part_m,
                         float* __restrict__ part_l) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  constexpr float kScale = 0.07216878364870323F;  // 1 / sqrt(192)

  // Row mapping: tile rb, row r.
  const std::uint32_t q0 = blockIdx.x * kTokensPerBlock;  // prefill only
  const std::uint32_t split = blockIdx.x;                 // decode only
  const auto row_token = [&](std::uint32_t r) { return kDecode ? 0U : q0 + r; };
  const auto row_head = [&](std::uint32_t rb, std::uint32_t r) {
    return kDecode ? (rb * 16) + r : (blockIdx.y * 2) + rb;
  };
  const auto row_live = [&](std::uint32_t r) {
    return kDecode || (q0 + r < n_tokens);
  };
  const auto row_qpos = [&](std::uint32_t r) {
    return kDecode ? context - 1 : first_pos + q0 + r;
  };

  std::uint32_t key_begin = 0;
  std::uint32_t key_end = 0;
  if constexpr (kDecode) {
    key_begin = split * chunk;
    key_end = min(context, key_begin + chunk);
  } else {
    key_end = first_pos + min(q0 + kTokensPerBlock, n_tokens);
  }

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
    const bool live = row_live(sub);
    const float* q_row = q +
                         (std::size_t{live ? row_token(sub) : 0U} *
                          (kDecode ? 0U : kHeadsN * kWidth)) +
                         (std::size_t{row_head(s_rb, sub)} * kWidth);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const auto* qp =
          reinterpret_cast<const float4*>(q_row + (((s_kq * 9) + ks) * 16));
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.F, 0.F, 0.F, 0.F);
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

  for (std::uint32_t kb = key_begin; kb < key_end; kb += kKeys) {
    // 1. stage the K tile (contiguous in the cache), zero past the range.
    for (std::uint32_t idx = tid; idx < kKeys * (kWidth / 8); idx += 256) {
      const std::uint32_t key = idx / (kWidth / 8);
      const std::uint32_t c8 = idx % (kWidth / 8);
      uint4 v = make_uint4(0U, 0U, 0U, 0U);
      if (kb + key < key_end) {
        v = *reinterpret_cast<const uint4*>(
            cache + (std::size_t{kb + key} * kWidth) + (c8 * 8));
      }
      *reinterpret_cast<uint4*>(&k_lds[(key * kKStride) + (c8 * 8)]) = v;
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
      const bool live = row_live(sm_r);
      const std::uint32_t qpos = row_qpos(sm_r);
      float vals[2];
      float part_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < 2; ++m) {
        const std::uint32_t j = (seg * 2) + m;
        const std::uint32_t key = kb + j;
        const bool valid = live && key < key_end && key <= qpos;
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
    // The next iteration's K staging only touches k_lds, which nobody reads
    // after the barrier above; its first barrier orders p_lds/vt_lds reuse.
  }

  if (seg == 0) {
    row_max_s[rr] = run_max;
    row_sum_s[rr] = run_sum;
  }
  __syncthreads();

  if constexpr (kDecode) {
    // Unnormalised partials relative to this split's own maximum.
    if (seg == 0) {
      const std::uint32_t head = row_head(sm_rb, sm_r);
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
  } else {
#pragma unroll
    for (std::uint32_t rb = 0; rb < 2; ++rb) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t token = q0 + row;
        if (token < n_tokens) {
          const float sum = row_sum_s[(rb * 16) + row];
          const float inv = sum > 0.0F ? 1.0F / sum : 0.0F;
          const std::uint32_t head = (blockIdx.y * 2) + rb;
#pragma unroll
          for (std::uint32_t t = 0; t < 4; ++t) {
            out[(((std::size_t{token} * kHeadsN) + head) * kLatent) +
                (((wave * 4) + t) * 16) + sub] = o_acc[rb][t][i] * inv;
          }
        }
      }
    }
  }
}

__global__ void DecodeMergeKernel(const float* __restrict__ part_o,
                                  const float* __restrict__ part_m,
                                  const float* __restrict__ part_l,
                                  std::uint32_t splits,
                                  float* __restrict__ out) {
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

struct SplitPlan {
  std::uint32_t splits;
  std::uint32_t chunk;
};

SplitPlan PlanSplits(std::uint32_t context) {
  std::uint32_t splits =
      std::clamp((context + kSplitKeys - 1) / kSplitKeys, 1U, kMaxSplits);
  std::uint32_t chunk = (context + splits - 1) / splits;
  chunk = (chunk + kKeys - 1) / kKeys * kKeys;
  splits = (context + chunk - 1) / chunk;
  return {splits, chunk};
}

std::size_t PartialFloats(std::uint32_t splits) {
  return std::size_t{splits} * kHeadsN * (kLatent + 2);
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

void CacheAppend(__half* layer_cache, const float* kv, std::uint32_t first_pos,
                 std::uint32_t n_tokens, hipStream_t stream) {
  const std::size_t count = std::size_t{n_tokens} * kWidth;
  if (count == 0) {
    return;
  }
  CacheAppendKernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                      stream>>>(layer_cache + (std::size_t{first_pos} * kWidth),
                                kv, count);
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

std::size_t DecodeScratchBytes(std::uint32_t max_context) {
  return PartialFloats(PlanSplits(std::max(max_context, 1U)).splits) *
         sizeof(float);
}

void DecodeAttention(const float* q_abs, const __half* layer_cache,
                     std::uint32_t context, float* out_latent, void* scratch,
                     hipStream_t stream) {
  if (context == 0) {
    (void)hipMemsetAsync(out_latent, 0, sizeof(float) * kHeadsN * kLatent,
                         stream);
    return;
  }
  const SplitPlan plan = PlanSplits(context);
  auto* part_o = static_cast<float*>(scratch);
  float* part_m = part_o + (std::size_t{plan.splits} * kHeadsN * kLatent);
  float* part_l = part_m + (std::size_t{plan.splits} * kHeadsN);
  AttentionKernel<true><<<plan.splits, 256, 0, stream>>>(
      q_abs, layer_cache, 0, 1, context, plan.chunk, part_o, part_m, part_l);
  DecodeMergeKernel<<<kHeadsN, kLatent, 0, stream>>>(part_o, part_m, part_l,
                                                     plan.splits, out_latent);
}

void PrefillAttention(const float* q_abs, const __half* layer_cache,
                      std::uint32_t first_pos, std::uint32_t n_tokens,
                      float* out_latent, void* /*scratch*/,
                      hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const dim3 grid((n_tokens + kTokensPerBlock - 1) / kTokensPerBlock,
                  kHeadsN / 2);
  AttentionKernel<false>
      <<<grid, 256, 0, stream>>>(q_abs, layer_cache, first_pos, n_tokens, 0, 0,
                                 out_latent, nullptr, nullptr);
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
