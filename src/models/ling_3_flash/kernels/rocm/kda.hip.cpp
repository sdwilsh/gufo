// Kimi Delta Attention (KDA) kernels of the Ling-3.0-flash runtime. The math
// is PLAN.md section 1; the contract is kernels.hpp (namespace kda) and
// layout.hpp. The recurrent state is F32 [head][i key][j value]: decay acts on
// the key axis i, so a thread that owns one value column j keeps S[:, j] in
// 128 registers and every state row access across a wave is coalesced.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>

#include "src/models/ling_3_flash/kernels/rocm/act_quant.hip.hpp"
#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"

namespace gufo::models::ling_3_flash::rocm::kda {
namespace {

constexpr std::uint32_t kD = kKdaHeadDim;
constexpr std::uint32_t kInner = kKdaInner;
constexpr std::uint32_t kConvChannels = 3 * kInner;
constexpr std::uint32_t kConvChunk = 32;  // tokens per conv block row

[[nodiscard]] __device__ inline float Sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}
[[nodiscard]] __device__ inline float Silu(float x) {
  return x * Sigmoid(x);
}

// ---------------------------------------------------------------------------
// Conv: three causal depthwise conv4 streams + SiLU. Thread per channel over a
// chunk of tokens; the window slides in registers. The main kernel only reads
// the carried state, a second small kernel advances it (no in-place race
// between token chunks).
// ---------------------------------------------------------------------------

// One conv4 output before SiLU. The order of the four products is explicit
// (it is the order the compiler chose for ConvKernel under -ffast-math, which
// reassociates the plain expression) so the one-token fusion below and the
// multi-token kernel give the same bits regardless of compiler choices.
__device__ __forceinline__ float ConvTaps(const float4& w, float x0, float x1,
                                          float x2, float x3) {
  return fmaf(x3, w.w, fmaf(x0, w.x, fmaf(x2, w.z, x1 * w.y)));
}

__global__ void ConvKernel(const float* __restrict__ proj,
                           const float* __restrict__ w_q,
                           const float* __restrict__ w_k,
                           const float* __restrict__ w_v,
                           const float* __restrict__ state,
                           float* __restrict__ out, std::uint32_t n_tokens) {
  const std::uint32_t gc = blockIdx.x * blockDim.x + threadIdx.x;
  if (gc >= kConvChannels) {
    return;
  }
  const std::uint32_t stream = gc / kInner;
  const std::uint32_t c = gc % kInner;
  const float* w = (stream == 0 ? w_q : stream == 1 ? w_k : w_v) + c * 4;
  const float4 wv = *reinterpret_cast<const float4*>(w);
  const std::uint32_t t0 = blockIdx.y * kConvChunk;
  const std::uint32_t t1 = min(n_tokens, t0 + kConvChunk);
  const float* st = state + static_cast<std::size_t>(stream) * 3 * kInner + c;
  const float* col = proj + stream * kInner + c;
  const auto raw = [&](int t) -> float {
    return t >= 0 ? col[static_cast<std::size_t>(t) * kKdaProjRows]
                  : st[static_cast<std::size_t>(t + 3) * kInner];
  };
  const int s = static_cast<int>(t0);
  float x0 = raw(s - 3);
  float x1 = raw(s - 2);
  float x2 = raw(s - 1);
  for (std::uint32_t t = t0; t < t1; ++t) {
    const float x3 = raw(static_cast<int>(t));
    const float y = ConvTaps(wv, x0, x1, x2, x3);
    out[static_cast<std::size_t>(t) * kConvChannels + gc] = Silu(y);
    x0 = x1;
    x1 = x2;
    x2 = x3;
  }
}

__global__ void ConvStateKernel(const float* __restrict__ proj,
                                float* __restrict__ state,
                                std::uint32_t n_tokens) {
  const std::uint32_t gc = blockIdx.x * blockDim.x + threadIdx.x;
  if (gc >= kConvChannels) {
    return;
  }
  const std::uint32_t stream = gc / kInner;
  const std::uint32_t c = gc % kInner;
  float* st = state + static_cast<std::size_t>(stream) * 3 * kInner + c;
  const float* col = proj + stream * kInner + c;
  float next[3];
  // New position p holds raw token n - 3 + p; before the first token that is
  // old position n + p (always >= p, so reading all first is safe).
  for (int p = 0; p < 3; ++p) {
    const int idx = static_cast<int>(n_tokens) - 3 + p;
    next[p] = idx >= 0 ? col[static_cast<std::size_t>(idx) * kKdaProjRows]
                       : st[static_cast<std::size_t>(idx + 3) * kInner];
  }
  for (int p = 0; p < 3; ++p) {
    st[static_cast<std::size_t>(p) * kInner] = next[p];
  }
}

// Rollback checkpoints of the conv window for a small verify chunk: after
// token t (t < n_tokens - 1) the window holds raw tokens t - 2..t, written to
// `ckpt + t * ckpt_stride` in the live state's layout. Reads the state before
// ConvStateKernel advances it (same stream, launched first).
__global__ void ConvCheckpointKernel(const float* __restrict__ proj,
                                     const float* __restrict__ state,
                                     float* __restrict__ ckpt,
                                     std::size_t ckpt_stride,
                                     std::uint32_t n_tokens) {
  const std::uint32_t gc = blockIdx.x * blockDim.x + threadIdx.x;
  if (gc >= kConvChannels) {
    return;
  }
  const std::uint32_t stream = gc / kInner;
  const std::uint32_t c = gc % kInner;
  const float* st = state + static_cast<std::size_t>(stream) * 3 * kInner + c;
  const float* col = proj + stream * kInner + c;
  float* out = ckpt + static_cast<std::size_t>(stream) * 3 * kInner + c;
  for (std::uint32_t t = 0; t + 1 < n_tokens; ++t) {
    for (int p = 0; p < 3; ++p) {
      const int idx = static_cast<int>(t) + 1 - 3 + p;
      out[t * ckpt_stride + static_cast<std::size_t>(p) * kInner] =
          idx >= 0 ? col[static_cast<std::size_t>(idx) * kKdaProjRows]
                   : st[static_cast<std::size_t>(idx + 3) * kInner];
    }
  }
}

// ---------------------------------------------------------------------------
// Gate / beta prep, L2 norm, output norm-gate.
// ---------------------------------------------------------------------------

__global__ void GateBetaKernel(const float* __restrict__ proj,
                               const float* __restrict__ ssm_a,
                               const float* __restrict__ dt_bias,
                               float* __restrict__ g, float* __restrict__ beta,
                               std::uint32_t n_tokens) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t n_g = static_cast<std::size_t>(n_tokens) * kInner;
  if (idx < n_g) {
    const std::size_t t = idx / kInner;
    const std::uint32_t ch = static_cast<std::uint32_t>(idx % kInner);
    const float f = proj[t * kKdaProjRows + kKdaFaRow + ch];
    g[idx] = kKdaGateLowerBound * Sigmoid(ssm_a[ch / kD] * (f + dt_bias[ch]));
  } else if (idx < n_g + static_cast<std::size_t>(n_tokens) * kHeads) {
    const std::size_t b = idx - n_g;
    const std::size_t t = b / kHeads;
    const std::uint32_t h = static_cast<std::uint32_t>(b % kHeads);
    beta[b] = Sigmoid(proj[t * kKdaProjRows + kKdaBetaRow + h]);
  }
}

// One wave32 per (row, head); 4 floats per lane.
__device__ inline float WaveSum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor(v, off, 32);
  }
  return v;
}

// 1 / sqrt(sum(v^2) + eps) over the 128 values held by one wave (4 per lane),
// reproducing the code the compiler generates for L2NormKernel under
// -ffast-math, which reassociates the plain expression into (eps + p) + s and
// turns 1 / sqrt into a bare v_rsq_f32. The decode fusion must give the same
// bits as that kernel; the empty asm keeps the order from being reassociated.
__device__ __forceinline__ float L2Scale(const float4& v) {
  float s = v.x * v.x;
  s = fmaf(v.y, v.y, s);
  s = fmaf(v.z, v.z, s);
  s = fmaf(v.w, v.w, s);
#pragma unroll
  for (int off = 16; off > 1; off >>= 1) {
    s += __shfl_xor(s, off, 32);
  }
  const float p = __shfl_xor(s, 1, 32);
  float t = kKdaL2Eps + p;
  __asm__ volatile("" : "+v"(t));
  return __builtin_amdgcn_rsqf(t + s);
}

__global__ void L2NormKernel(float* x, std::uint32_t ld) {
  const std::uint32_t head = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x & 31;
  float4* p = reinterpret_cast<float4*>(
                  x + static_cast<std::size_t>(blockIdx.y) * ld + head * kD) +
              lane;
  float4 v = *p;
  const float r = L2Scale(v);
  v.x *= r;
  v.y *= r;
  v.z *= r;
  v.w *= r;
  *p = v;
}

// Output norm and gate of one lane's 4 values of a head (shared by the plain
// and the quantising kernel so both give the same bits).
__device__ __forceinline__ float4 OutputNormGateLane(const float4& v,
                                                     const float4& gate,
                                                     const float4& gm) {
  const float ss = WaveSum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
  const float r = 1.0F / sqrtf(ss / static_cast<float>(kD) + kRmsEps);
  float4 y;
  y.x = v.x * r * gm.x * Sigmoid(gate.x);
  y.y = v.y * r * gm.y * Sigmoid(gate.y);
  y.z = v.z * r * gm.z * Sigmoid(gate.z);
  y.w = v.w * r * gm.w * Sigmoid(gate.w);
  return y;
}

__global__ void OutputNormGateKernel(const float* __restrict__ o,
                                     const float* __restrict__ proj,
                                     const float* __restrict__ gamma,
                                     float* __restrict__ out) {
  const std::uint32_t head = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x & 31;
  const std::size_t t = blockIdx.y;
  const float4 v =
      reinterpret_cast<const float4*>(o + t * kInner + head * kD)[lane];
  const float4 gate = reinterpret_cast<const float4*>(
      proj + t * kKdaProjRows + kKdaGaRow + head * kD)[lane];
  const float4 gm = reinterpret_cast<const float4*>(gamma)[lane];
  const float4 y = OutputNormGateLane(v, gate, gm);
  reinterpret_cast<float4*>(out + t * kInner + head * kD)[lane] = y;
}

// ---------------------------------------------------------------------------
// Decode-family fusion (1..8 tokens): conv + conv-state advance (+ the rollback
// windows of a verify chunk) + SiLU + q/k L2 norm + gate/beta prep in one
// launch. Same expressions as ConvKernel/ConvStateKernel/ConvCheckpointKernel/
// L2NormKernel/GateBetaKernel, so the results are identical. A wave owns one
// 128-channel head of one stream (lane: 4 consecutive channels) and slides
// the window over the tokens in registers; each channel's state is read and
// rewritten by its own lane only.
// ---------------------------------------------------------------------------
constexpr std::uint32_t kPrepWaves = 4;                             // per block
constexpr std::uint32_t kConvHeadWaves = 3 * kHeads;                // 96
constexpr std::uint32_t kConvBlocks = kConvHeadWaves / kPrepWaves;  // 24

__global__ __launch_bounds__(kPrepWaves * 32) void PrepDecodeKernel(
    const float* __restrict__ proj, const float* __restrict__ w_q,
    const float* __restrict__ w_k, const float* __restrict__ w_v,
    float* __restrict__ state, const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias, float* __restrict__ out,
    float* __restrict__ g, float* __restrict__ beta, std::uint32_t n_tokens,
    float* __restrict__ ckpt, std::size_t ckpt_stride) {
  if (blockIdx.x < kConvBlocks) {
    const std::uint32_t wave_id = blockIdx.x * kPrepWaves + (threadIdx.x >> 5);
    const std::uint32_t stream = wave_id / kHeads;
    const std::uint32_t head = wave_id % kHeads;
    const std::uint32_t lane = threadIdx.x & 31;
    const std::uint32_t c0 = head * kD + lane * 4;  // channel within stream
    const float* w = (stream == 0 ? w_q : stream == 1 ? w_k : w_v) + c0 * 4;
    const std::size_t soff = static_cast<std::size_t>(stream) * 3 * kInner + c0;
    float* st = state + soff;
    const float* col = proj + stream * kInner + c0;
    float4 wv[4];
    float x0[4], x1[4], x2[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      wv[e] = *reinterpret_cast<const float4*>(w + e * 4);
      x0[e] = st[e];
      x1[e] = st[kInner + e];
      x2[e] = st[2 * kInner + e];
    }
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      float x3[4], y[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x3[e] = col[static_cast<std::size_t>(t) * kKdaProjRows + e];
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float yy = ConvTaps(wv[e], x0[e], x1[e], x2[e], x3[e]);
        y[e] = Silu(yy);
      }
      float4 v = make_float4(y[0], y[1], y[2], y[3]);
      if (stream < 2) {
        const float r = L2Scale(v);
        v.x *= r;
        v.y *= r;
        v.z *= r;
        v.w *= r;
      }
      reinterpret_cast<float4*>(
          out + static_cast<std::size_t>(t) * kConvChannels +
          static_cast<std::size_t>(stream) * kInner + c0)[0] = v;
      // Slide: the window now holds raw tokens t - 2..t.
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x0[e] = x1[e];
        x1[e] = x2[e];
        x2[e] = x3[e];
      }
      if (ckpt != nullptr && t + 1 < n_tokens) {
        float* ck = ckpt + t * ckpt_stride + soff;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          ck[e] = x0[e];
          ck[kInner + e] = x1[e];
          ck[2 * kInner + e] = x2[e];
        }
      }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      st[e] = x0[e];
      st[kInner + e] = x1[e];
      st[2 * kInner + e] = x2[e];
    }
    return;
  }
  const std::uint32_t idx =
      (blockIdx.x - kConvBlocks) * blockDim.x + threadIdx.x;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float* pt = proj + static_cast<std::size_t>(t) * kKdaProjRows;
    if (idx < kInner) {
      const float f = pt[kKdaFaRow + idx];
      g[static_cast<std::size_t>(t) * kInner + idx] =
          kKdaGateLowerBound * Sigmoid(ssm_a[idx / kD] * (f + dt_bias[idx]));
    } else if (idx < kInner + kHeads) {
      const std::uint32_t h = idx - kInner;
      beta[t * kHeads + h] = Sigmoid(pt[kKdaBetaRow + h]);
    }
  }
}

__global__ void OutputNormGateQuantKernel(const float* __restrict__ o,
                                          const float* __restrict__ proj,
                                          const float* __restrict__ gamma,
                                          float* __restrict__ out,
                                          std::int8_t* __restrict__ q,
                                          std::int8_t* __restrict__ q2,
                                          float* __restrict__ dq) {
  const std::uint32_t head = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x & 31;
  const std::size_t t = blockIdx.y;
  const float4 v =
      reinterpret_cast<const float4*>(o + t * kInner + head * kD)[lane];
  const float4 gate = reinterpret_cast<const float4*>(
      proj + t * kKdaProjRows + kKdaGaRow + head * kD)[lane];
  const float4 gm = reinterpret_cast<const float4*>(gamma)[lane];
  const float4 y = OutputNormGateLane(v, gate, gm);
  const std::size_t flat = t * kInner + head * kD + lane * 4;
  reinterpret_cast<float4*>(out)[flat / 4] = y;
  // Quantisation blocks of 32 values span 8 lanes (the standalone kernel's
  // wave-wide max over 32 consecutive elements).
  float m = fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
#pragma unroll
  for (int off = 4; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor(m, off, 32));
  const float vals[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    QuantizeValue(vals[e], m, &q[flat + e], &q2[flat + e]);
  }
  if ((lane & 7) == 0)
    dq[flat >> 5] = QuantScale(m);
}

// ---------------------------------------------------------------------------
// Gated delta rule. One wave32 per (head, 32-wide value tile): lane j owns the
// state column S[:, j] in 128 registers, so the 64 KiB tile streams from and to
// memory exactly once per launch (once per layer per decode token) with fully
// coalesced 128-byte rows. q, k and exp(g) of the token are shared by every
// lane through double-buffered LDS, prefetched one token ahead.
// ---------------------------------------------------------------------------

constexpr std::uint32_t kTile = 32;

struct RecurrentArgs {
  const float* q;
  const float* k;
  const float* v;
  const float* g;
  const float* beta;
  float* o;
  std::uint32_t ld_q, ld_k, ld_v, ld_g, ld_beta, ld_o;
  /// Optional rollback checkpoints: when not null, the state after token t
  /// (t < n_tokens - 1) is copied to `ckpt + t * ckpt_stride` (same [h][i][j]
  /// layout as the live state). Pure extra stores: the arithmetic is the same.
  float* ckpt;
  std::size_t ckpt_stride;
};

template<bool kCheckpoint>
__launch_bounds__(kTile) __global__
    void RecurrentKernel(float* __restrict__ state, RecurrentArgs a,
                         std::uint32_t n_tokens) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t j = blockIdx.x * kTile + lane;
  __shared__ float4 s_q[2][kD / 4];
  __shared__ float4 s_k[2][kD / 4];
  __shared__ float4 s_e[2][kD / 4];  // exp(g)

  float* S = state + static_cast<std::size_t>(h) * kD * kD + j;
  float col[kD];
#pragma unroll
  for (std::uint32_t i = 0; i < kD; ++i) {
    col[i] = S[static_cast<std::size_t>(i) * kD];
  }

  const auto row_ptr = [&](const float* base, std::uint32_t ld,
                           std::uint32_t t) {
    return reinterpret_cast<const float4*>(
               base + static_cast<std::size_t>(t) * ld + h * kD) +
           lane;
  };
  float4 rq = *row_ptr(a.q, a.ld_q, 0);
  float4 rk = *row_ptr(a.k, a.ld_k, 0);
  float4 rg = *row_ptr(a.g, a.ld_g, 0);
  float rv = a.v[h * kD + j];
  float rb = a.beta[h];
  s_q[0][lane] = rq;
  s_k[0][lane] = rk;
  s_e[0][lane] = make_float4(expf(rg.x), expf(rg.y), expf(rg.z), expf(rg.w));
  float cur_v = rv;
  float cur_b = rb;
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(kD));
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t buf = t & 1;
    const bool more = t + 1 < n_tokens;
    if (more) {
      rq = *row_ptr(a.q, a.ld_q, t + 1);
      rk = *row_ptr(a.k, a.ld_k, t + 1);
      rg = *row_ptr(a.g, a.ld_g, t + 1);
      rv = a.v[static_cast<std::size_t>(t + 1) * a.ld_v + h * kD + j];
      rb = a.beta[static_cast<std::size_t>(t + 1) * a.ld_beta + h];
    }
    // S[i][:] *= exp(g[i]); u[j] = sum_i S[i][j] * k[i].
    float u0 = 0.0F;
    float u1 = 0.0F;
    float u2 = 0.0F;
    float u3 = 0.0F;
#pragma unroll
    for (std::uint32_t i4 = 0; i4 < kD / 4; ++i4) {
      const float4 kk = s_k[buf][i4];
      const float4 ee = s_e[buf][i4];
      col[4 * i4 + 0] *= ee.x;
      col[4 * i4 + 1] *= ee.y;
      col[4 * i4 + 2] *= ee.z;
      col[4 * i4 + 3] *= ee.w;
      u0 += col[4 * i4 + 0] * kk.x;
      u1 += col[4 * i4 + 1] * kk.y;
      u2 += col[4 * i4 + 2] * kk.z;
      u3 += col[4 * i4 + 3] * kk.w;
    }
    const float delta = (cur_v - ((u0 + u1) + (u2 + u3))) * cur_b;
    // S[i][j] += k[i] * delta[j]; o[j] = sum_i S[i][j] * q[i].
    float o0 = 0.0F;
    float o1 = 0.0F;
    float o2 = 0.0F;
    float o3 = 0.0F;
#pragma unroll
    for (std::uint32_t i4 = 0; i4 < kD / 4; ++i4) {
      const float4 kk = s_k[buf][i4];
      const float4 qq = s_q[buf][i4];
      col[4 * i4 + 0] = fmaf(kk.x, delta, col[4 * i4 + 0]);
      col[4 * i4 + 1] = fmaf(kk.y, delta, col[4 * i4 + 1]);
      col[4 * i4 + 2] = fmaf(kk.z, delta, col[4 * i4 + 2]);
      col[4 * i4 + 3] = fmaf(kk.w, delta, col[4 * i4 + 3]);
      o0 += col[4 * i4 + 0] * qq.x;
      o1 += col[4 * i4 + 1] * qq.y;
      o2 += col[4 * i4 + 2] * qq.z;
      o3 += col[4 * i4 + 3] * qq.w;
    }
    a.o[static_cast<std::size_t>(t) * a.ld_o + h * kD + j] =
        ((o0 + o1) + (o2 + o3)) * scale;
    if constexpr (kCheckpoint) {
      if (more) {
        float* C = a.ckpt + static_cast<std::size_t>(t) * a.ckpt_stride +
                   static_cast<std::size_t>(h) * kD * kD + j;
#pragma unroll
        for (std::uint32_t i = 0; i < kD; ++i) {
          C[static_cast<std::size_t>(i) * kD] = col[i];
        }
      }
    }
    if (more) {
      s_q[buf ^ 1][lane] = rq;
      s_k[buf ^ 1][lane] = rk;
      s_e[buf ^ 1][lane] =
          make_float4(expf(rg.x), expf(rg.y), expf(rg.z), expf(rg.w));
      cur_v = rv;
      cur_b = rb;
      __syncthreads();
    }
  }
#pragma unroll
  for (std::uint32_t i = 0; i < kD; ++i) {
    S[static_cast<std::size_t>(i) * kD] = col[i];
  }
}

// ---------------------------------------------------------------------------
// Prefill recurrence (n > 8 tokens). The same token-sequential gated delta
// rule as RecurrentKernel, restructured for throughput. RecurrentKernel keeps
// a whole 128-row state column in one lane (256 VGPRs, spilled) and stages one
// token at a time, so a wave is latency bound (estimated ~10 us per token from
// 860M timings) with only 128 waves per layer. Here a quad of lanes shares each
// pair of state columns (lane = group * 4 + part, `part` owning float4 key rows
// f = 4 * m + part), which leaves 64 state registers and no spills, makes 256
// waves per layer, halves the LDS reads per FMA (two columns share every
// k/q/exp(g) read) and finishes the two per-token dot products with quad DPP
// adds. A block of two waves covers (head, 32 value columns). q, k, exp(g), v
// and beta of the next kBatch tokens are loaded into registers while the
// current ones compute, and the outputs leave through LDS as coalesced float4
// rows.
//
// The arithmetic is a fixed function of the token sequence: tokens are
// processed in order from the carried state with a reduction tree that does
// not depend on n, on where a call starts or on the batching, so any split of
// a prefill run into calls gives bit-identical outputs and state (the prefill
// numeric family of engine.hpp). It differs from RecurrentKernel (small-batch
// family) only in the summation order of the two per-token dot products.
// ---------------------------------------------------------------------------

constexpr std::uint32_t kRowLanes = 4;  // lanes per state column group
constexpr std::uint32_t kCpl = 2;       // adjacent state columns per lane
constexpr std::uint32_t kPfCols = 32;   // value columns per block
constexpr std::uint32_t kPfThreads = kPfCols / kCpl * kRowLanes;  // 64
constexpr std::uint32_t kBatch = 2;                       // staged tokens
constexpr std::uint32_t kMyRows4 = (kD / 4) / kRowLanes;  // float4 rows / lane
constexpr std::uint32_t kPerTokenVec = 3 * (kD / 4) + kPfCols / 4;  // q,k,g,v

// Sum over the kRowLanes adjacent lanes of a column group (lane = group * 4 +
// part): two quad DPP adds, the same tree for every lane.
__device__ inline float QuadSum(float x) {
  static_assert(kRowLanes == 4);
  const float a =
      x + __builtin_bit_cast(
              float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                                 0xB1, 0xF, 0xF, false));
  return a + __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(
                                           0, __builtin_bit_cast(int, a), 0x4E,
                                           0xF, 0xF, false));
}

struct PrefillStage {
  float4 vec[(kBatch * kPerTokenVec + kPfThreads - 1) / kPfThreads];
  float beta;
};

__launch_bounds__(kPfThreads) __global__
    void PrefillKernel(float* __restrict__ state, RecurrentArgs a,
                       std::uint32_t n_tokens) {
  // Fast-math reassociation otherwise lets each compiler pick a different
  // summation order, so the bits differ across toolchains.
#pragma clang fp reassociate(off)
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31;
  const std::uint32_t wave = tid >> 5;
  const std::uint32_t part = lane % kRowLanes;
  const std::uint32_t col = wave * (32 / kRowLanes * kCpl) +
                            (lane / kRowLanes) * kCpl;  // first of kCpl columns
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t j0 = blockIdx.x * kPfCols;

  __shared__ float4 s_q[2][kBatch][kD / 4];
  __shared__ float4 s_k[2][kBatch][kD / 4];
  __shared__ float4 s_e[2][kBatch][kD / 4];  // exp(g)
  __shared__ float4 s_v[2][kBatch][kPfCols / 4];
  __shared__ float s_b[2][kBatch];
  __shared__ float4 s_o[2][kBatch][kPfCols / 4];

  // State registers: float4 row group f = kRowLanes * m + part (4 rows each)
  // of kCpl adjacent columns.
  float st[kMyRows4][4][kCpl];
  {
    const float* S = state + static_cast<std::size_t>(h) * kD * kD + j0 + col;
#pragma unroll
    for (std::uint32_t m = 0; m < kMyRows4; ++m) {
#pragma unroll
      for (std::uint32_t c = 0; c < 4; ++c) {
        const float2 w = *reinterpret_cast<const float2*>(
            S + static_cast<std::size_t>(4 * (kRowLanes * m + part) + c) * kD);
        st[m][c][0] = w.x;
        st[m][c][1] = w.y;
      }
    }
  }

  const auto load_batch = [&](std::uint32_t base, PrefillStage& r) {
#pragma unroll
    for (std::uint32_t it = 0; it < std::size(r.vec); ++it) {
      const std::uint32_t idx = tid + it * kPfThreads;
      float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (idx < kBatch * kPerTokenVec) {
        const std::uint32_t tt = idx / kPerTokenVec;
        const std::uint32_t p = idx % kPerTokenVec;
        const std::size_t t = base + tt;
        if (t < n_tokens) {
          const float* src;
          if (p < kD / 4) {
            src = a.q + t * a.ld_q + h * kD + 4 * p;
          } else if (p < 2 * (kD / 4)) {
            src = a.k + t * a.ld_k + h * kD + 4 * (p - kD / 4);
          } else if (p < 3 * (kD / 4)) {
            src = a.g + t * a.ld_g + h * kD + 4 * (p - 2 * (kD / 4));
          } else {
            src = a.v + t * a.ld_v + h * kD + j0 + 4 * (p - 3 * (kD / 4));
          }
          v = *reinterpret_cast<const float4*>(src);
        }
      }
      r.vec[it] = v;
    }
    r.beta = 0.0F;
    if (tid < kBatch && base + tid < n_tokens) {
      r.beta = a.beta[static_cast<std::size_t>(base + tid) * a.ld_beta + h];
    }
  };
  const auto store_batch = [&](std::uint32_t buf, const PrefillStage& r) {
#pragma unroll
    for (std::uint32_t it = 0; it < std::size(r.vec); ++it) {
      const std::uint32_t idx = tid + it * kPfThreads;
      if (idx < kBatch * kPerTokenVec) {
        const std::uint32_t tt = idx / kPerTokenVec;
        const std::uint32_t p = idx % kPerTokenVec;
        const float4 v = r.vec[it];
        if (p < kD / 4) {
          s_q[buf][tt][p] = v;
        } else if (p < 2 * (kD / 4)) {
          s_k[buf][tt][p - kD / 4] = v;
        } else if (p < 3 * (kD / 4)) {
          s_e[buf][tt][p - 2 * (kD / 4)] =
              make_float4(expf(v.x), expf(v.y), expf(v.z), expf(v.w));
        } else {
          s_v[buf][tt][p - 3 * (kD / 4)] = v;
        }
      }
    }
    if (tid < kBatch) {
      s_b[buf][tid] = r.beta;
    }
  };

  PrefillStage stage;
  load_batch(0, stage);
  store_batch(0, stage);
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(kD));
  const std::uint32_t n_batches = (n_tokens + kBatch - 1) / kBatch;
  for (std::uint32_t b = 0; b < n_batches; ++b) {
    const std::uint32_t buf = b & 1;
    const std::uint32_t base = b * kBatch;
    const bool more = b + 1 < n_batches;
    if (more) {
      load_batch(base + kBatch, stage);
    }
    const std::uint32_t nt = min(kBatch, n_tokens - base);
    for (std::uint32_t tt = 0; tt < nt; ++tt) {
      float4 kk[kMyRows4];
      float u[kCpl][4] = {};
#pragma unroll
      for (std::uint32_t m = 0; m < kMyRows4; ++m) {
        const std::uint32_t f = kRowLanes * m + part;
        kk[m] = s_k[buf][tt][f];
        const float4 ee = s_e[buf][tt][f];
#pragma unroll
        for (std::uint32_t x = 0; x < kCpl; ++x) {
          st[m][0][x] *= ee.x;
          st[m][1][x] *= ee.y;
          st[m][2][x] *= ee.z;
          st[m][3][x] *= ee.w;
          u[x][0] = fmaf(st[m][0][x], kk[m].x, u[x][0]);
          u[x][1] = fmaf(st[m][1][x], kk[m].y, u[x][1]);
          u[x][2] = fmaf(st[m][2][x], kk[m].z, u[x][2]);
          u[x][3] = fmaf(st[m][3][x], kk[m].w, u[x][3]);
        }
      }
      const float2 vv = *reinterpret_cast<const float2*>(
          &reinterpret_cast<const float*>(&s_v[buf][tt][0])[col]);
      const float bt = s_b[buf][tt];
      float delta[kCpl];
      delta[0] =
          (vv.x - QuadSum((u[0][0] + u[0][1]) + (u[0][2] + u[0][3]))) * bt;
      delta[1] =
          (vv.y - QuadSum((u[1][0] + u[1][1]) + (u[1][2] + u[1][3]))) * bt;
      float o[kCpl][4] = {};
#pragma unroll
      for (std::uint32_t m = 0; m < kMyRows4; ++m) {
        const std::uint32_t f = kRowLanes * m + part;
        const float4 qq = s_q[buf][tt][f];
#pragma unroll
        for (std::uint32_t x = 0; x < kCpl; ++x) {
          st[m][0][x] = fmaf(kk[m].x, delta[x], st[m][0][x]);
          st[m][1][x] = fmaf(kk[m].y, delta[x], st[m][1][x]);
          st[m][2][x] = fmaf(kk[m].z, delta[x], st[m][2][x]);
          st[m][3][x] = fmaf(kk[m].w, delta[x], st[m][3][x]);
          o[x][0] = fmaf(st[m][0][x], qq.x, o[x][0]);
          o[x][1] = fmaf(st[m][1][x], qq.y, o[x][1]);
          o[x][2] = fmaf(st[m][2][x], qq.z, o[x][2]);
          o[x][3] = fmaf(st[m][3][x], qq.w, o[x][3]);
        }
      }
      const float o0 = QuadSum((o[0][0] + o[0][1]) + (o[0][2] + o[0][3]));
      const float o1 = QuadSum((o[1][0] + o[1][1]) + (o[1][2] + o[1][3]));
      if (part == 0) {
        *reinterpret_cast<float2*>(&reinterpret_cast<float*>(
            &s_o[buf][tt][0])[col]) = make_float2(o0 * scale, o1 * scale);
      }
    }
    if (more) {
      store_batch(buf ^ 1, stage);
    }
    __syncthreads();
    for (std::uint32_t idx = tid; idx < kBatch * (kPfCols / 4);
         idx += kPfThreads) {
      const std::uint32_t tt = idx / (kPfCols / 4);
      if (base + tt < n_tokens) {
        reinterpret_cast<float4*>(
            a.o + static_cast<std::size_t>(base + tt) * a.ld_o + h * kD +
            j0)[idx % (kPfCols / 4)] = s_o[buf][tt][idx % (kPfCols / 4)];
      }
    }
  }
  float* S = state + static_cast<std::size_t>(h) * kD * kD + j0 + col;
#pragma unroll
  for (std::uint32_t m = 0; m < kMyRows4; ++m) {
#pragma unroll
    for (std::uint32_t c = 0; c < 4; ++c) {
      *reinterpret_cast<float2*>(
          S + static_cast<std::size_t>(4 * (kRowLanes * m + part) + c) * kD) =
          make_float2(st[m][c][0], st[m][c][1]);
    }
  }
}

// ---------------------------------------------------------------------------
// State snapshot / restore: fixed-size float4 grid-stride copy.
// ---------------------------------------------------------------------------

__global__ void CopyKernel(const float4* __restrict__ src,
                           float4* __restrict__ dst, std::size_t n4) {
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    dst[i] = src[i];
  }
}

void Copy(const float* src, float* dst, std::size_t floats,
          hipStream_t stream) {
  constexpr std::uint32_t kBlock = 256;
  constexpr std::uint32_t kBlocks = 2048;
  CopyKernel<<<kBlocks, kBlock, 0, stream>>>(
      reinterpret_cast<const float4*>(src), reinterpret_cast<float4*>(dst),
      floats / 4);
}

}  // namespace

void Conv3Prefill(const float* proj, const float* w_q, const float* w_k,
                  const float* w_v, float* conv_state, float* out,
                  std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const dim3 grid(kConvChannels / 256,
                  (n_tokens + kConvChunk - 1) / kConvChunk);
  ConvKernel<<<grid, 256, 0, stream>>>(proj, w_q, w_k, w_v, conv_state, out,
                                       n_tokens);
  ConvStateKernel<<<kConvChannels / 256, 256, 0, stream>>>(proj, conv_state,
                                                           n_tokens);
}

void Conv3Checkpoint(const float* proj, const float* conv_state, float* ckpt,
                     std::size_t ckpt_stride, std::uint32_t n_tokens,
                     hipStream_t stream) {
  if (n_tokens < 2) {
    return;
  }
  ConvCheckpointKernel<<<kConvChannels / 256, 256, 0, stream>>>(
      proj, conv_state, ckpt, ckpt_stride, n_tokens);
}

void Conv3Decode(const float* proj, const float* w_q, const float* w_k,
                 const float* w_v, float* conv_state, float* out,
                 hipStream_t stream) {
  Conv3Prefill(proj, w_q, w_k, w_v, conv_state, out, 1, stream);
}

void GateBetaPrep(const float* proj, const float* ssm_a, const float* dt_bias,
                  float* g, float* beta, std::uint32_t n_tokens,
                  hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const std::size_t total =
      static_cast<std::size_t>(n_tokens) * (kInner + kHeads);
  GateBetaKernel<<<static_cast<std::uint32_t>((total + 255) / 256), 256, 0,
                   stream>>>(proj, ssm_a, dt_bias, g, beta, n_tokens);
}

void L2NormHeads(float* x, std::uint32_t ld, std::uint32_t n_rows,
                 hipStream_t stream) {
  if (n_rows == 0) {
    return;
  }
  L2NormKernel<<<dim3(kHeads / 4, n_rows), 128, 0, stream>>>(x, ld);
}

void RecurrentDecode(float* state, const float* q, const float* k,
                     const float* v, const float* g, const float* beta,
                     float* o, hipStream_t stream) {
  const RecurrentArgs a{q,      k,      v,      g,      beta,   o,       kInner,
                        kInner, kInner, kInner, kHeads, kInner, nullptr, 0};
  RecurrentKernel<false>
      <<<dim3(kD / kTile, kHeads), kTile, 0, stream>>>(state, a, 1);
}

void RecurrentSequence(float* state, const float* q, std::uint32_t ld_q,
                       const float* k, std::uint32_t ld_k, const float* v,
                       std::uint32_t ld_v, const float* g, const float* beta,
                       float* o, std::uint32_t n_tokens, hipStream_t stream,
                       float* ckpt, std::size_t ckpt_stride) {
  if (n_tokens == 0) {
    return;
  }
  const RecurrentArgs a{q,    k,    v,      g,      beta,   o,    ld_q,
                        ld_k, ld_v, kInner, kHeads, kInner, ckpt, ckpt_stride};
  if (ckpt != nullptr) {
    RecurrentKernel<true>
        <<<dim3(kD / kTile, kHeads), kTile, 0, stream>>>(state, a, n_tokens);
  } else {
    RecurrentKernel<false>
        <<<dim3(kD / kTile, kHeads), kTile, 0, stream>>>(state, a, n_tokens);
  }
}

void PrefillSequence(float* state, const float* q, std::uint32_t ld_q,
                     const float* k, std::uint32_t ld_k, const float* v,
                     std::uint32_t ld_v, const float* g, const float* beta,
                     float* o, std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const RecurrentArgs a{q,    k,    v,      g,      beta,   o,       ld_q,
                        ld_k, ld_v, kInner, kHeads, kInner, nullptr, 0};
  PrefillKernel<<<dim3(kD / kPfCols, kHeads), kPfThreads, 0, stream>>>(
      state, a, n_tokens);
}

void OutputNormGate(const float* o, const float* proj, const float* ssm_norm,
                    float* out, std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  OutputNormGateKernel<<<dim3(kHeads / 4, n_tokens), 128, 0, stream>>>(
      o, proj, ssm_norm, out);
}

void PrepDecode(const float* proj, const float* w_q, const float* w_k,
                const float* w_v, float* conv_state, const float* ssm_a,
                const float* dt_bias, float* conv_out, float* g, float* beta,
                std::uint32_t n_tokens, float* conv_ckpt,
                std::size_t ckpt_stride, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  constexpr std::uint32_t kGateBlocks =
      (kInner + kHeads + kPrepWaves * 32 - 1) / (kPrepWaves * 32);
  PrepDecodeKernel<<<kConvBlocks + kGateBlocks, kPrepWaves * 32, 0, stream>>>(
      proj, w_q, w_k, w_v, conv_state, ssm_a, dt_bias, conv_out, g, beta,
      n_tokens, conv_ckpt, ckpt_stride);
}

void OutputNormGateQuant(const float* o, const float* proj,
                         const float* ssm_norm, float* out,
                         std::uint32_t n_tokens, void* quant,
                         hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const dense::QuantView v = dense::QuantLayout(quant, n_tokens, kInner);
  OutputNormGateQuantKernel<<<dim3(kHeads / 4, n_tokens), 128, 0, stream>>>(
      o, proj, ssm_norm, out, v.q, v.q2, v.d);
}

void CopyFloats(const float* src, float* dst, std::size_t floats,
                hipStream_t stream) {
  Copy(src, dst, floats, stream);
}

void SnapshotState(const float* kda_state, const float* kda_conv,
                   float* snap_state, float* snap_conv, hipStream_t stream) {
  Copy(kda_state, snap_state, kKdaLayers * kKdaStateFloats, stream);
  Copy(kda_conv, snap_conv, kKdaLayers * kKdaConvStateFloats, stream);
}

void RestoreState(float* kda_state, float* kda_conv, const float* snap_state,
                  const float* snap_conv, hipStream_t stream) {
  Copy(snap_state, kda_state, kKdaLayers * kKdaStateFloats, stream);
  Copy(snap_conv, kda_conv, kKdaLayers * kKdaConvStateFloats, stream);
}

}  // namespace gufo::models::ling_3_flash::rocm::kda
