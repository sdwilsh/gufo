#ifndef GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/ling_3_flash/layout.hpp"

/// HIP launchers of the Ling-3.0-flash runtime, declarations only. Activations
/// are row-major float32 `[tokens][dim]` unless noted; every launch is
/// asynchronous on `stream`; weights are device pointers in their GGUF
/// encoding. Each sub-namespace is implemented by one source file
/// (`kda.hip.cpp`, `mla.hip.cpp`, `moe.hip.cpp`, `dense.hip.cpp`). Math
/// references are PLAN section 1.
/// `scratch` arguments are device memory the caller owns; the matching
/// `...ScratchBytes` function gives the size.
namespace gufo::models::ling_3_flash::rocm {

/// GGUF type ids the kernels dispatch on.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kIQ4_XS = 23,
};

// ===========================================================================
// dense: linear layers, embedding, norms, SwiGLU (dense.hip.cpp)
// ===========================================================================
namespace dense {

/// Device bytes of activation-quantisation scratch the Q8_0/k-quant paths
/// need for `n_tokens` rows of `k` columns (0 for F16/F32 weights).
std::size_t ScratchBytes(WeightType type, std::uint32_t n_tokens,
                         std::uint32_t k);

/// y[t][r] = sum_c w[r][c] * x[t][c]; w is `rows` x `k` Q8_0 rows (k % 32 ==
/// 0). Decode path, `n_tokens` small (1..8): one pass over the weights.
void GemvQ8_0(const void* w, const float* x, float* y, std::uint32_t n_tokens,
              std::uint32_t k, std::uint32_t rows, void* scratch,
              hipStream_t stream);
/// Same math as GemvQ8_0, tiled for prefill (any `n_tokens`).
void GemmQ8_0(const void* w, const float* x, float* y, std::uint32_t n_tokens,
              std::uint32_t k, std::uint32_t rows, void* scratch,
              hipStream_t stream);
/// Strided output form: y[t * ldy + r]; used to write into the stacked KDA
/// projection `[token][kKdaProjRows]` and similar wide rows.
void GemmQ8_0Strided(const void* w, const float* x, float* y,
                     std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t rows, std::uint32_t ldy, void* scratch,
                     hipStream_t stream);

/// y[t][r] = dot(w[r], x[t]) with F32 weights (routers, 512 rows).
void GemvF32(const float* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, hipStream_t stream);
/// F16 weights `[rows][k]` with F32 accumulation (the lm_head, rows = 157184).
void GemvF16(const __half* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, hipStream_t stream);
void GemmF16(const __half* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, hipStream_t stream);

/// out[t][:] = dequant(table[tokens[t]]); table is Q4_K rows of `hidden`.
void EmbedQ4K(const void* table, const std::int32_t* tokens, float* out,
              std::uint32_t n_tokens, std::uint32_t hidden, hipStream_t stream);

/// out[t][g*d..] = x[t][g*d..] * rsqrt(mean(x[t][g*d..]^2) + eps) * gamma[..]
/// for `groups` independent norms of d = dim / groups elements each; gamma is
/// `d` floats shared by every group (null = no scale).
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// res[i] += x[i] for `count` floats (the pre-norm residual stream).
void AddResidual(float* res, const float* x, std::size_t count,
                 hipStream_t stream);

/// out[t][i] = min(silu(gate[t][i]), L) * clamp(up[t][i], -L, L) with
/// `limit` = L; L == 0 disables both clamps (plain silu(gate) * up).
void SwiGluClamped(const float* gate, const float* up, float* out,
                   std::size_t count, float limit, hipStream_t stream);

}  // namespace dense

// ===========================================================================
// kda: Kimi delta attention (kda.hip.cpp)
// ===========================================================================
namespace kda {

/// Causal depthwise conv4 + SiLU over q, k and v. `proj` is the stacked
/// projection `[t][kKdaProjRows]` (q, k, v at kKdaQRow/KRow/VRow); conv
/// weights are F32 `[chan][tap]` per stream; out `[t][3 * 4096]` (q|k|v):
/// out = silu(sum_tap w[c][tap] * x[t - 3 + tap]). `conv_state` is the layer's
/// kKdaConvStateFloats window; it is read, then advanced by one token.
void Conv3Decode(const float* proj, const float* w_q, const float* w_k,
                 const float* w_v, float* conv_state, float* out,
                 hipStream_t stream);
/// Same math over `n_tokens` consecutive tokens; `conv_state` is the state
/// before the first token and holds the state after the last on return.
void Conv3Prefill(const float* proj, const float* w_q, const float* w_k,
                  const float* w_v, float* conv_state, float* out,
                  std::uint32_t n_tokens, hipStream_t stream);

/// Gate and beta prep: g[t][h*128 + i] = -5 * sigmoid(ssm_a[h] *
/// (proj[t][kKdaFaRow + h*128 + i] + dt_bias[h*128 + i])) (per channel, key
/// axis); beta[t][h] = sigmoid(proj[t][kKdaBetaRow + h]).
void GateBetaPrep(const float* proj, const float* ssm_a, const float* dt_bias,
                  float* g, float* beta, std::uint32_t n_tokens,
                  hipStream_t stream);

/// In-place per-head L2 norm of `n_rows` rows of 32 heads x 128:
/// x / sqrt(sum(x^2) + 1e-6) (eps on the un-averaged sum). `ld` is the row
/// stride in floats (the conv output has ld = 3 * 4096; call for q and k).
void L2NormHeads(float* x, std::uint32_t ld, std::uint32_t n_rows,
                 hipStream_t stream);

/// One-token gated delta rule for one layer. With S[h][i][j] (F32, key i,
/// value j): S[i][:] *= exp(g[i]); delta[j] = (v[j] - sum_i S[i][j]*k[i]) *
/// beta; S[i][j] += k[i]*delta[j]; o[j] = (sum_i S[i][j]*q[i]) / sqrt(128).
/// q,k (normalised) and v are `[4096]` rows, g `[4096]`, beta `[32]`, o
/// `[4096]`.
void RecurrentDecode(float* state, const float* q, const float* k,
                     const float* v, const float* g, const float* beta,
                     float* o, hipStream_t stream);
/// The same recurrence applied sequentially over `n_tokens` (row-split over
/// the value axis across blocks); inputs are `[t][4096]` rows (`ld_*` strides
/// in floats), beta `[t][32]`; `state` is updated to the last token.
void PrefillRowSplit(float* state, const float* q, std::uint32_t ld_q,
                     const float* k, std::uint32_t ld_k, const float* v,
                     std::uint32_t ld_v, const float* g, const float* beta,
                     float* o, std::uint32_t n_tokens, hipStream_t stream);

/// out[t][h*128 + j] = rmsnorm_128(o[t][h*128..]; ssm_norm)[j] *
/// sigmoid(proj[t][kKdaGaRow + h*128 + j]); eps 1e-6.
void OutputNormGate(const float* o, const float* proj, const float* ssm_norm,
                    float* out, std::uint32_t n_tokens, hipStream_t stream);

/// Rollback copies of the recurrent state of all 35 KDA layers (state then
/// conv, in layout.hpp order): Snapshot copies live -> snap, Restore the
/// reverse. Launch-compatible with HIP graphs (no host sync).
void SnapshotState(const float* kda_state, const float* kda_conv,
                   float* snap_state, float* snap_conv, hipStream_t stream);
void RestoreState(float* kda_state, float* kda_conv, const float* snap_state,
                  const float* snap_conv, hipStream_t stream);

}  // namespace kda

// ===========================================================================
// mla: multi-head latent attention, absorbed form (mla.hip.cpp)
// ===========================================================================
namespace mla {

/// For each token: latent = rmsnorm_512(kv_a[t][0..512]; gamma, eps 1e-6);
/// k_pe = NORM rope(kv_a[t][512..576], positions[t], base 6e6) with adjacent
/// pairs (x[2i], x[2i+1]) rotated by pos * base^(-2i/64). Writes one cache row
/// `[t][576]` (latent then k_pe) in F32.
void LatentNormRope(const float* kv_a, const float* gamma,
                    const std::int32_t* positions, float* kv_out,
                    std::uint32_t n_tokens, hipStream_t stream);

/// cache[(first_pos + t) * 576 + c] = half(kv[t][c]) for the layer's cache base
/// pointer (already offset by MlaCacheLayout::LayerStride() * mla index).
void CacheAppend(__half* layer_cache, const float* kv, std::uint32_t first_pos,
                 std::uint32_t n_tokens, hipStream_t stream);

/// q_abs[t][h][0..512] = sum_d q_nope[t][h][d] * k_b[h][c][d] (k_b Q8_0
/// `[head][512][128]`); q_abs[t][h][512..576] = NORM rope(q_pe[t][h]) at
/// positions[t]. q is `[t][32][192]` (128 nope then 64 rope per head).
void QAbsorb(const float* q, const void* k_b, const std::int32_t* positions,
             float* q_abs, std::uint32_t n_tokens, hipStream_t stream);

/// Device bytes of split-K partial scratch for one decode attention call.
std::size_t DecodeScratchBytes(std::uint32_t max_context);
/// One query token over `context` cached rows: s[h][p] = q_abs[h] . cache[p]
/// / sqrt(192) over all 576 channels; out[h][0..512] = sum_p softmax(s)[h][p] *
/// cache[p][0..512] (latent rows act as values). Split-K across `context`.
void DecodeAttention(const float* q_abs, const __half* layer_cache,
                     std::uint32_t context, float* out_latent, void* scratch,
                     hipStream_t stream);
/// Prefill: query t sits at position `first_pos + t` and attends causally to
/// cache rows 0..first_pos+t (the chunk's own rows are already appended).
/// q_abs `[t][32][576]` -> out_latent `[t][32][512]`; same math as decode.
void PrefillAttention(const float* q_abs, const __half* layer_cache,
                      std::uint32_t first_pos, std::uint32_t n_tokens,
                      float* out_latent, void* scratch, hipStream_t stream);

/// out[t][h][d] = sum_c lat[t][h][c] * v_b[h][d][c] (v_b Q8_0
/// `[head][128][512]`) * sigmoid(gate_logits[t][h]); `out` is `[t][4096]`
/// ready for attn_output. gate_logits `[t][32]` = attn_gate . rmsnorm(x).
void VbUpProjGate(const float* lat, const void* v_b, const float* gate_logits,
                  float* out, std::uint32_t n_tokens, hipStream_t stream);

}  // namespace mla

// ===========================================================================
// moe: routing and routed experts (moe.hip.cpp)
// ===========================================================================
namespace moe {

/// Sigmoid group-limited router. p = sigmoid(logits[t][0..512]); sel = p +
/// bias; group score = sum of the top 2 sel of each of 8 groups of 64; keep the
/// top 4 groups; ids = top 8 experts by sel within kept groups; weights =
/// p[ids] / max(sum p[ids], 6.1035e-5) * 2.5 (unbiased p). Writes ids int32
/// `[t][8]` and weights F32 `[t][8]`. Ties break toward the lower index.
void Route(const float* logits, const float* bias, std::int32_t* ids,
           float* weights, std::uint32_t n_tokens, hipStream_t stream);

/// Device bytes of scratch (sorted token lists) the prefill kernels need.
std::size_t PrefillScratchBytes(std::uint32_t n_tokens);
/// Device bytes of scratch (unique-expert list) the decode kernels need.
std::size_t DecodeScratchBytes();

/// Decode / small verify batch, `n_tokens` in 1..8: act[t][k][f] =
/// swiglu_clamped(gate_e . x[t], up_e . x[t]) for the 8 experts e = ids[t][k]
/// (gate/up `[512][768][2560]`, type IQ4_XS or Q5_K (Q4_K/Q6_K also work),
/// clamp L = swiglu_clamp_exp[layer], 0 = off); x `[t][2560]`, ids `[t][8]`,
/// act `[t][8][768]`. Each distinct selected expert is read once per call
/// even when several tokens pick it. `scratch` is DecodeScratchBytes().
void ExpertGateUpDecode(const void* gate, const void* up, WeightType type,
                        const float* x, const std::int32_t* ids,
                        std::uint32_t n_tokens, float limit, float* act,
                        void* scratch, hipStream_t stream);
/// Decode, `n_tokens` in 1..8: out[t][k][:] = down_e . act[t][k] for e =
/// ids[t][k] (down `[512][2560][768]`, type Q4_K or Q6_K); act `[t][8][768]`,
/// out `[t][8][2560]`, unweighted. Same single-read-per-expert guarantee.
void ExpertDownDecode(const void* down, WeightType type, const float* act,
                      const std::int32_t* ids, std::uint32_t n_tokens,
                      float* out, void* scratch, hipStream_t stream);
/// Prefill grouped form of ExpertGateUpDecode for `n_tokens` rows; x
/// `[t][2560]`, ids `[t][8]`, act `[t][8][768]`. Tokens are grouped per expert
/// and run through F16 WMMA tiles (F32 accumulation). `scratch` is
/// PrefillScratchBytes(n_tokens).
void ExpertGateUpPrefill(const void* gate, const void* up, WeightType type,
                         const float* x, const std::int32_t* ids,
                         std::uint32_t n_tokens, float limit, float* act,
                         void* scratch, hipStream_t stream);
/// Prefill grouped form of ExpertDownDecode; act `[t][8][768]` -> out
/// `[t][8][2560]`, unweighted.
void ExpertDownPrefill(const void* down, WeightType type, const float* act,
                       const std::int32_t* ids, std::uint32_t n_tokens,
                       float* out, void* scratch, hipStream_t stream);

/// res[t][i] += sum_k weights[t][k] * expert_out[t][k][i] + shared_out[t][i]
/// (the shared expert is ungated); in-place on the residual stream.
void Combine(float* res, const float* expert_out, const float* weights,
             const float* shared_out, std::uint32_t n_tokens,
             hipStream_t stream);

}  // namespace moe

}  // namespace gufo::models::ling_3_flash::rocm

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_KERNELS_HPP_
