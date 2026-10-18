#ifndef GUFO_MODELS_LING_3_FLASH_CPU_OPS_HPP_
#define GUFO_MODELS_LING_3_FLASH_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "src/models/ling_3_flash/weights.hpp"

/// Scalar float32 operators of the Ling-3.0-flash reference model. The math is
/// artifacts/ling-3-flash/PLAN.md section 1; every GPU kernel is checked
/// against these. Correctness oracle only: nothing here is tuned, and weights
/// are decoded on the fly row by row (no full-tensor float copies except for
/// the small MLA k_b / v_b expansion tensors).
namespace gufo::models::ling_3_flash::cpu {

// ---- Threading and memory -------------------------------------------------

/// Number of worker threads used by ParallelFor and the parallel matvecs.
void SetThreads(unsigned threads);
[[nodiscard]] unsigned Threads() noexcept;
/// Splits [0, n) into contiguous chunks run on the pool; fn(begin, end).
/// Calls from inside a worker run serially.
void ParallelFor(std::size_t n,
                 const std::function<void(std::size_t, std::size_t)>& fn);

/// madvise(MADV_DONTNEED) over the whole pages inside [p, p + bytes), so the
/// mapped model never accumulates in this process' RSS.
void ReleasePages(const void* p, std::size_t bytes) noexcept;

// ---- Test-only F64 accumulation mode ----------------------------------------

/// When on (process-wide, reference library only; production kernels never
/// see it), the operators below accumulate in double: dot products, norms,
/// softmax, sigmoid/silu, rope angles, the router and the MLA attention.
/// Weights are the same dequantised floats; only the arithmetic is wider.
/// Default off, which leaves every F32 path bit-identical.
void SetF64(bool on) noexcept;
[[nodiscard]] bool F64() noexcept;

// ---- Quantised matrix access ----------------------------------------------

/// y[r] = dot(W[slice][r], x) for r in [0, t.rows). `slice` selects the
/// stacked matrix (expert or head). Parallel across rows when `parallel`.
void MatVec(const TensorRef& t, std::uint64_t slice, std::span<const float> x,
            std::span<float> y, bool parallel = true);
/// MatVec with double input and output, always accumulated in double (the
/// weights are still the dequantised floats). Used by the F64 mode.
void MatVecD(const TensorRef& t, std::uint64_t slice, std::span<const double> x,
             std::span<double> y, bool parallel = true);
/// Decodes one row of one slice (t.cols floats).
void DequantizeRow(const TensorRef& t, std::uint64_t slice, std::uint64_t row,
                   float* out);
/// Decodes a whole tensor (all slices) to float, depth-major then row-major.
[[nodiscard]] std::vector<float> DequantizeAll(const TensorRef& t);

// ---- Elementwise -----------------------------------------------------------

[[nodiscard]] float Sigmoid(float x) noexcept;
[[nodiscard]] float Silu(float x) noexcept;
/// out = x * rsqrt(mean(x^2) + eps) * w (w may be nullptr). out may alias x.
void RmsNorm(std::span<const float> x, const float* w, float eps,
             std::span<float> out);
/// x / sqrt(sum(x^2) + eps) in place. The epsilon is on the un-averaged sum
/// (llama.cpp build_gdn_l2_norm = rms_norm(x, eps/n) / sqrt(n)).
void L2Norm(std::span<float> x, float eps);
/// SwiGLU: up = clamp(up, -L, L); act = min(silu(gate), L); out = act * up.
/// L <= 1e-6 turns the clamp off.
void SwiGlu(std::span<const float> gate, std::span<const float> up, float limit,
            std::span<float> out);

// ---- KDA --------------------------------------------------------------------

/// Causal depthwise conv (kernel 4) + SiLU for one stream and one token.
/// `weight` is [chan][4] (tap 3 meets the current token), `history` is the
/// stream's [3][chan] raw previous inputs (pos 0 oldest) and is advanced by one
/// token; `raw` is the current raw projection, `out` receives silu(conv).
void CausalConv4Silu(const float* weight, float* history, const float* raw,
                     float* out, std::size_t channels);
/// g[h*128+i] = -5 * sigmoid(a[h] * (fa[h*128+i] + dt_bias[h*128+i])).
void KdaGate(const float* fa, const float* dt_bias, const float* a,
             float lower_bound, float* g);
/// One token of the gated delta recurrence for all heads. `state` is
/// [h][i key][j value]. Per head: S[i][:] *= exp(g[i]);
/// delta[j] = (v[j] - sum_i S[i][j] k[i]) * beta; S[i][j] += k[i] delta[j];
/// o[j] = (sum_i S[i][j] q[i]) / sqrt(128). q and k are already L2-normalised.
void KdaStep(float* state, const float* q, const float* k, const float* v,
             const float* g, const float* beta, float* o);
/// KdaStep with a double state ([h][i][j]) and double sums (F64 mode).
void KdaStepF64(double* state, const float* q, const float* k, const float* v,
                const float* g, const float* beta, float* o);
/// o[h] = rmsnorm(o[h], norm_w) * sigmoid(gate[h]) in place for all heads.
void KdaOutputGate(float* o, const float* norm_w, const float* gate, float eps);

// ---- MLA --------------------------------------------------------------------

/// NORM (adjacent pair) RoPE on the first `n_rot` elements of each of `heads`
/// vectors laid `stride` floats apart: (x[2i], x[2i+1]) rotates by
/// pos * base^(-2i/n_rot).
void NormRope(float* x, std::uint32_t heads, std::uint32_t stride,
              std::uint32_t n_rot, std::uint32_t pos, float base);

/// Absorbed form. `q` is [32][192] (nope 128, roped pe 64); `cache` is
/// [n][576] float (normalised latent, roped k_pe). Returns per-head 128 values
/// `out` [32][128] through v_b; `latent_out` (optional, [32][512]) receives the
/// softmax-weighted latent before v_b.
void MlaAttendAbsorbed(const TensorRef& k_b, const TensorRef& v_b,
                       const float* q, const float* cache, std::size_t n,
                       float* out, float* latent_out = nullptr);

/// MlaAttendAbsorbed with every intermediate (q_lat, scores, softmax, latent,
/// v_b) in double (F64 mode). The cache is read as stored.
void MlaAttendAbsorbedF64(const TensorRef& k_b, const TensorRef& v_b,
                          const float* q, const float* cache, std::size_t n,
                          float* out, float* latent_out = nullptr);

/// Per-token K/V expansion for the non-absorbed form. `wk`/`wv` are the whole
/// k_b ([32][512][128]) and v_b ([32][128][512]) tensors as floats.
/// k_nope[h][d] = sum_r latent[r] wk[h][r][d]; v[h][j] = sum_r wv[h][j][r]
/// latent[r]; both [32][128].
void MlaExpand(const float* wk, const float* wv, const float* latent,
               float* k_nope, float* v);
/// Non-absorbed form: scores = q_nope.k_nope + q_pe.k_pe over n cached
/// tokens. `k_nope` and `v` are [n][32][128], `k_pe` is [n][64].
void MlaAttendExpanded(const float* q, const float* k_nope, const float* k_pe,
                       const float* v, std::size_t n, float* out);

// ---- Router -----------------------------------------------------------------

struct Routing {
  std::int32_t ids[8];
  float weights[8];
  /// Weights before the float cast (F64 mode accumulates with these).
  double weights_d[8];
  /// The ten best experts in selection order with their biased selection
  /// scores (sigmoid + bias); entries 8 and 9 are the first rejected
  /// experts, so top_sel[7] - top_sel[8] is the router margin.
  std::int32_t top_ids[10];
  float top_sel[10];
};
/// Sigmoid, biased group-limited top-k. Groups are scored by the sum of their
/// two best biased scores; the best `groups_used` groups are kept; the top
/// `experts_used` biased scores inside them are chosen (descending); weights
/// are the UNBIASED sigmoid of the chosen experts, divided by
/// max(sum, floor), times `scale`. `probs` (optional, experts floats)
/// receives the unbiased sigmoid.
void Route(std::span<const float> logits, const float* bias,
           std::uint32_t groups, std::uint32_t groups_used,
           std::uint32_t experts_used, float scale, float floor, Routing& out,
           float* probs = nullptr);

/// Route with double logits and double selection arithmetic (F64 mode).
void RouteF64(std::span<const double> logits, const float* bias,
              std::uint32_t groups, std::uint32_t groups_used,
              std::uint32_t experts_used, float scale, float floor,
              Routing& out, float* probs = nullptr);

}  // namespace gufo::models::ling_3_flash::cpu

#endif  // GUFO_MODELS_LING_3_FLASH_CPU_OPS_HPP_
