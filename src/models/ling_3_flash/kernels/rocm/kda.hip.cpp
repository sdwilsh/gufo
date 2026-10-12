// Kimi Delta Attention (KDA) kernels of the Ling-3.0-flash runtime. The math
// is PLAN.md section 1; the contract is kernels.hpp (namespace kda) and
// layout.hpp. The recurrent state is F32 [head][i key][j value]: decay acts on
// the key axis i, so a thread that owns one value column j keeps S[:, j] in
// 128 registers and every state row access across a wave is coalesced.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

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
    const float y = wv.x * x0 + wv.y * x1 + wv.z * x2 + wv.w * x3;
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

__global__ void L2NormKernel(float* x, std::uint32_t ld) {
  const std::uint32_t head = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const std::uint32_t lane = threadIdx.x & 31;
  float4* p = reinterpret_cast<float4*>(
                  x + static_cast<std::size_t>(blockIdx.y) * ld + head * kD) +
              lane;
  float4 v = *p;
  const float ss = WaveSum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
  const float r = 1.0F / sqrtf(ss + kKdaL2Eps);
  v.x *= r;
  v.y *= r;
  v.z *= r;
  v.w *= r;
  *p = v;
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
  const float ss = WaveSum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
  const float r = 1.0F / sqrtf(ss / static_cast<float>(kD) + kRmsEps);
  float4 y;
  y.x = v.x * r * gm.x * Sigmoid(gate.x);
  y.y = v.y * r * gm.y * Sigmoid(gate.y);
  y.z = v.z * r * gm.z * Sigmoid(gate.z);
  y.w = v.w * r * gm.w * Sigmoid(gate.w);
  reinterpret_cast<float4*>(out + t * kInner + head * kD)[lane] = y;
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
};

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
  const RecurrentArgs a{q,      k,      v,      g,      beta,   o,
                        kInner, kInner, kInner, kInner, kHeads, kInner};
  RecurrentKernel<<<dim3(kD / kTile, kHeads), kTile, 0, stream>>>(state, a, 1);
}

void PrefillRowSplit(float* state, const float* q, std::uint32_t ld_q,
                     const float* k, std::uint32_t ld_k, const float* v,
                     std::uint32_t ld_v, const float* g, const float* beta,
                     float* o, std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const RecurrentArgs a{q,    k,    v,    g,      beta,   o,
                        ld_q, ld_k, ld_v, kInner, kHeads, kInner};
  RecurrentKernel<<<dim3(kD / kTile, kHeads), kTile, 0, stream>>>(state, a,
                                                                  n_tokens);
}

void OutputNormGate(const float* o, const float* proj, const float* ssm_norm,
                    float* out, std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens == 0) {
    return;
  }
  OutputNormGateKernel<<<dim3(kHeads / 4, n_tokens), 128, 0, stream>>>(
      o, proj, ssm_norm, out);
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
