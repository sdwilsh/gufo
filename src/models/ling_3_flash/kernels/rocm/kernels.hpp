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

/// Device bytes of scratch the Q8_0 paths need for `n_tokens` rows of `k`
/// columns: the int8 activation planes of GemvQ8_0 and, for the GEMMs, the
/// activations rounded to F16 (2 * n_tokens * k bytes). 0 for F16/F32 weights.
/// Every Q8_0 entry point may overwrite the whole area; keep it free of live
/// data. The F16 head GEMM needs `2 * n_tokens * k` bytes of the same buffer.
std::size_t ScratchBytes(WeightType type, std::uint32_t n_tokens,
                         std::uint32_t k);

/// y[t][r] = sum_c w[r][c] * x[t][c]; w is `rows` x `k` Q8_0 rows (k % 32 ==
/// 0). Decode path, `n_tokens` small (1..8): one pass over the weights.
void GemvQ8_0(const void* w, const float* x, float* y, std::uint32_t n_tokens,
              std::uint32_t k, std::uint32_t rows, void* scratch,
              hipStream_t stream);
/// Same math as GemvQ8_0, tiled for prefill (any `n_tokens`). `scratch`
/// (>= ScratchBytes) is written: it receives the activations rounded to F16 so
/// the row blocks do not each convert them. Null converts inside the kernel
/// (same bits).
void GemmQ8_0(const void* w, const float* x, float* y, std::uint32_t n_tokens,
              std::uint32_t k, std::uint32_t rows, void* scratch,
              hipStream_t stream);
/// Strided output form: y[t * ldy + r]; used to write into the stacked KDA
/// projection `[token][kKdaProjRows]` and similar wide rows. `scratch` is
/// written as for GemmQ8_0.
void GemmQ8_0Strided(const void* w, const float* x, float* y,
                     std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t rows, std::uint32_t ldy, void* scratch,
                     hipStream_t stream);

/// Test/benchmark hook: GemmQ8_0Strided with an explicit tile shape
/// (0..kGemmShapes - 1: 128x128, 64x64, 32x128, 16x128 tokens x rows). Every
/// shape returns the same bits as the default dispatch (GemmQ8_0Strided).
inline constexpr int kGemmShapes = 4;
/// The shape the default dispatch picks for (n_tokens, rows) on a device with
/// `wgps` work-group processors (0 = the current device).
int GemmShapeFor(std::uint32_t n_tokens, std::uint32_t rows, int wgps = 0);
void GemmQ8_0Shape(const void* w, const float* x, float* y,
                   std::uint32_t n_tokens, std::uint32_t k, std::uint32_t rows,
                   std::uint32_t ldy, int shape, void* scratch,
                   hipStream_t stream);
void GemmF16Shape(const __half* w, const float* x, float* y,
                  std::uint32_t n_tokens, std::uint32_t k, std::uint32_t rows,
                  int shape, void* scratch, hipStream_t stream);

/// y[t][r] = dot(w[r], x[t]) with F32 weights (routers, 512 rows).
void GemvF32(const float* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, hipStream_t stream);
/// F16 weights `[rows][k]` with F32 accumulation (the lm_head, rows = 157184).
/// `scratch` (>= 2 * n_tokens * k bytes, written) holds the F16 activations
/// of the GEMM path (n_tokens > 8); null converts inside the kernel.
void GemvF16(const __half* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, void* scratch,
             hipStream_t stream);
void GemmF16(const __half* w, const float* x, float* y, std::uint32_t n_tokens,
             std::uint32_t k, std::uint32_t rows, void* scratch,
             hipStream_t stream);

/// Re-encodes `rows` x `k` F16 weights as Q8_0 (k % 32 == 0) into `out`
/// (rows * k / 32 * 34 bytes; ggml's quantize_row_q8_0 reference: d = absmax /
/// 127 stored as F16, q = round(x / d)). One-time load work (the MTP draft
/// head); asynchronous on `stream`.
void QuantizeF16ToQ8_0(const __half* w, void* out, std::uint64_t rows,
                       std::uint32_t k, hipStream_t stream);

/// Quantised activations as the Q8_0 GEMV consumes them: two int8 planes
/// (value, residual) of `n_tokens` x k and one F32 scale per 32 values, in one
/// buffer of QuantActBytes(n_tokens, k) bytes (layout depends on n_tokens, k).
struct QuantView {
  std::int8_t* q;
  std::int8_t* q2;
  float* d;
};
QuantView QuantLayout(void* base, std::uint32_t n_tokens, std::uint32_t k);
std::size_t QuantActBytes(std::uint32_t n_tokens, std::uint32_t k);
/// True when GemvQ8_0Quantized accepts (n_tokens in 1..8, k % 64 == 0).
bool GemvQ8_0Fits(std::uint32_t n_tokens, std::uint32_t k);
/// Quantises x[n][k] into `quant` (what GemvQ8_0 does internally).
void QuantizeActivations(const float* x, std::uint32_t n_tokens,
                         std::uint32_t k, void* quant, hipStream_t stream);
struct Q8Group {
  const void* w;  ///< `rows` Q8_0 rows of k columns
  float* y;       ///< y[t * rows + r]
  std::uint32_t rows;
};
/// GemvQ8_0 over already quantised activations, for up to three weight
/// matrices in one launch (rows computed exactly as by separate launches).
void GemvQ8_0Quantized(const Q8Group* groups, int n_groups, const void* quant,
                       std::uint32_t n_tokens, std::uint32_t k,
                       hipStream_t stream);

/// Decode fusions (n_rows <= 8 rows of 2560): norm + quantisation in one
/// launch, bit-identical to RmsNormRows (+ AddResidual / moe::Combine) followed
/// by QuantizeActivations. `out` is the normalised F32 row, `quant` the planes.
/// out = norm(x) * gamma
void RmsNormQuantRows(const float* x, const float* gamma, float* out,
                      std::uint32_t n_rows, float eps, void* quant,
                      hipStream_t stream);
/// res += addend; out = norm(res) * gamma
void AddResidualRmsNormQuant(float* res, const float* addend,
                             const float* gamma, float* out,
                             std::uint32_t n_rows, float eps, void* quant,
                             hipStream_t stream);
/// moe::Combine into res, then out = norm(res) * gamma
void CombineRmsNormQuant(float* res, const float* expert_out,
                         const float* weights, const float* shared_out,
                         const float* gamma, float* out, std::uint32_t n_rows,
                         float eps, void* quant, hipStream_t stream);
/// SwiGluClamped over count = rows * k values, and the planes of the result
/// (count % 32 == 0).
void SwiGluClampedQuant(const float* gate, const float* up, float* out,
                        std::size_t count, std::uint32_t k, float limit,
                        void* quant, hipStream_t stream);

/// out[t][:] = dequant(table[tokens[t]]); table is Q4_K rows of `hidden`.
void EmbedQ4K(const void* table, const std::int32_t* tokens, float* out,
              std::uint32_t n_tokens, std::uint32_t hidden, hipStream_t stream);
/// Same for Q8_0 rows (34-byte blocks of 32).
void EmbedQ8_0(const void* table, const std::int32_t* tokens, float* out,
               std::uint32_t n_tokens, std::uint32_t hidden,
               hipStream_t stream);
/// Embedding gather for the table's own format (Q4_K or Q8_0).
void Embed(WeightType type, const void* table, const std::int32_t* tokens,
           float* out, std::uint32_t n_tokens, std::uint32_t hidden,
           hipStream_t stream);

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
/// The same recurrence applied sequentially over `n_tokens` with the decode
/// kernel's arithmetic (the small-batch numeric family, n <= 8: bit-identical
/// to `n_tokens` RecurrentDecode calls). Inputs are `[t][4096]` rows (`ld_*`
/// strides in floats), beta `[t][32]`; `state` is updated to the last token.
///
/// With `ckpt` not null the recurrent state after every token but the last is
/// also stored, token t at `ckpt + t * ckpt_stride` (the live layout,
/// kKdaStateFloats): rollback points for speculative verification. The
/// outputs and the final state are unchanged.
void RecurrentSequence(float* state, const float* q, std::uint32_t ld_q,
                       const float* k, std::uint32_t ld_k, const float* v,
                       std::uint32_t ld_v, const float* g, const float* beta,
                       float* o, std::uint32_t n_tokens, hipStream_t stream,
                       float* ckpt = nullptr, std::size_t ckpt_stride = 0);
/// Conv window after every token but the last of an `n_tokens` chunk (token t
/// at `ckpt + t * ckpt_stride`, layout of the live window,
/// kKdaConvStateFloats). Launch before Conv3Prefill, which advances
/// `conv_state`.
void Conv3Checkpoint(const float* proj, const float* conv_state, float* ckpt,
                     std::size_t ckpt_stride, std::uint32_t n_tokens,
                     hipStream_t stream);
/// Copies `floats` floats (multiple of 4) on the stream.
void CopyFloats(const float* src, float* dst, std::size_t floats,
                hipStream_t stream);
/// The same recurrence, token-sequential, restructured for throughput (the
/// prefill numeric family, n > 8; valid for any n >= 1). The result is a fixed
/// function of the token sequence: any split into calls, at any token, is
/// bit-identical to one call. It differs from RecurrentSequence only in
/// rounding order (summation order of the two per-token dot products, possibly
/// fast-math contraction; about 1e-7 relative).
void PrefillSequence(float* state, const float* q, std::uint32_t ld_q,
                     const float* k, std::uint32_t ld_k, const float* v,
                     std::uint32_t ld_v, const float* g, const float* beta,
                     float* o, std::uint32_t n_tokens, hipStream_t stream);

/// out[t][h*128 + j] = rmsnorm_128(o[t][h*128..]; ssm_norm)[j] *
/// sigmoid(proj[t][kKdaGaRow + h*128 + j]); eps 1e-6.
void OutputNormGate(const float* o, const float* proj, const float* ssm_norm,
                    float* out, std::uint32_t n_tokens, hipStream_t stream);

/// Decode-family (1..8 tokens) fusion of Conv3Prefill, GateBetaPrep and
/// L2NormHeads on q and k (and, with `conv_ckpt`, Conv3Checkpoint):
/// `conv_out` `[t][3 * 4096]` (q|k normalised, v), g `[t][4096]`, beta
/// `[t][32]`. Bit-identical to calling those separately.
void PrepDecode(const float* proj, const float* w_q, const float* w_k,
                const float* w_v, float* conv_state, const float* ssm_a,
                const float* dt_bias, float* conv_out, float* g, float* beta,
                std::uint32_t n_tokens, float* conv_ckpt,
                std::size_t ckpt_stride, hipStream_t stream);
/// OutputNormGate that also writes the activation planes of `out` for the
/// following Q8_0 GEMV (dense::QuantLayout(quant, n_tokens, 4096)).
void OutputNormGateQuant(const float* o, const float* proj,
                         const float* ssm_norm, float* out,
                         std::uint32_t n_tokens, void* quant,
                         hipStream_t stream);

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

/// cache[positions[t] * 576 + c] = half(kv[t][c]) for the layer's cache base
/// pointer (already offset by MlaCacheLayout::LayerStride() * mla index).
/// `positions` is device memory (so a captured graph replays at any position).
void CacheAppend(__half* layer_cache, const float* kv,
                 const std::int32_t* positions, std::uint32_t n_tokens,
                 hipStream_t stream);

/// q_abs[t][h][0..512] = sum_d q_nope[t][h][d] * k_b[h][c][d] (k_b Q8_0
/// `[head][512][128]`); q_abs[t][h][512..576] = NORM rope(q_pe[t][h]) at
/// positions[t]. q is `[t][32][192]` (128 nope then 64 rope per head).
void QAbsorb(const float* q, const void* k_b, const std::int32_t* positions,
             float* q_abs, std::uint32_t n_tokens, hipStream_t stream);

/// Device bytes of split-K partial scratch for one decode attention call.
std::size_t DecodeScratchBytes(std::uint32_t max_context);
/// One query token at `*position` (device int32) over `*position + 1` cached
/// rows: s[h][p] = q_abs[h] . cache[p] / sqrt(192) over all 576 channels;
/// out[h][0..512] = sum_p softmax(s)[h][p] * cache[p][0..512] (latent rows act
/// as values). Split-K across the context; the split plan is a function of the
/// context alone and is evaluated on the device (the grid is the maximum plan,
/// surplus blocks exit), so the launch is identical for every position and can
/// be captured in a HIP graph.
void DecodeAttention(const float* q_abs, const __half* layer_cache,
                     const std::int32_t* position, float* out_latent,
                     void* scratch, hipStream_t stream);
/// Device milliseconds of the parts of one PrefillAttentionNonAbsorbed call
/// (profiling aid: passing it brackets each launch with events and waits).
struct PrefillTimes {
  double expand_ms{0.0};  ///< Transient expansion of rows past the store.
  double attn_ms{0.0};    ///< Per-block flash attention.
  double merge_ms{0.0};   ///< Ordered fold, normalise, gate.
};

/// Non-absorbed prefill (n > 8 token chunks). `w16` is the per-layer F16
/// expansion weight (ExpansionWeightBytes()): per head 256 rows of 512, rows
/// 0..127 = k_b[h][:, d] (transposed), 128..255 = v_b[h][e][:]; both Q8_0
/// tensors dequantised then rounded to F16 once at load.
std::size_t ExpansionWeightBytes();
void BuildExpansionWeights(const void* k_b, const void* v_b, __half* w16,
                           hipStream_t stream);
/// Transient device bytes for chunks of up to `max_batch` tokens: expanded K/V
/// of two 2048-key blocks, a bounded group of per-block attention partials
/// (>= 64 MiB) and the fold state.
std::size_t NonAbsorbedScratchBytes(std::uint32_t max_batch);
/// Persistent expanded K/V of one layer for rows [0, rows) (a multiple of
/// 2048): per head k_nope `[row][128]` then V^T in 16-key tiles
/// `[row / 16][128][16]` (keys permuted in each tile), F16,
/// ExpandedLayerBytes(rows) = rows * 16 KiB.
std::size_t ExpandedLayerBytes(std::uint32_t rows);
/// Expands cache rows [start, min(end, rows)) into the persistent store. The
/// result is a pure function of the cache row (same bits as the transient
/// expansion). It also rewrites [start - start % 128, start) with identical
/// bits.
void ExpandRows(const __half* layer_cache, const __half* w16,
                __half* layer_expanded, std::uint32_t rows, std::uint32_t start,
                std::uint32_t end, hipStream_t stream);
/// q_rope[t][h][0..128] = q_nope, q_rope[t][h][128..192] = NORM rope(q_pe) at
/// positions[t]; q is `[t][32][192]`.
void QPrepare(const float* q, const std::int32_t* positions, float* q_rope,
              std::uint32_t n_tokens, hipStream_t stream);
/// Prefill: query t at position `first_pos + t` attends causally to cache rows
/// 0..first_pos+t (already appended): per head p = softmax(q . [k_nope | k_pe]
/// / sqrt(192)) over expanded K/V, out[t][h][0..128] = p.v (times
/// sigmoid(gate_logits[t][h]) when `gate_logits` is not null), ready for
/// attn_output. Keys below `expanded_rows` (a multiple of 2048, may be 0) are
/// read from `layer_expanded` (rows must have been expanded with ExpandRows);
/// the rest are expanded per call into `scratch`. Keys fold in absolute
/// 2048-key blocks (per-block partials merged in block order), so any split of
/// a token run into chunks gives identical bits, and so does moving rows
/// between the persistent store and the transient path.
void PrefillAttentionNonAbsorbed(
    const float* q_rope, const __half* layer_cache, const __half* w16,
    const __half* layer_expanded, std::uint32_t expanded_rows,
    const float* gate_logits, std::uint32_t first_pos, std::uint32_t n_tokens,
    float* out, void* scratch, hipStream_t stream,
    PrefillTimes* times = nullptr);

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

/// Device bytes of scratch (sorted token lists and their F16 activations) the
/// prefill kernels need.
std::size_t PrefillScratchBytes(std::uint32_t n_tokens);
/// Device bytes of scratch (unique-expert list) the decode kernels need.
std::size_t DecodeScratchBytes();

/// Decode / small verify batch, `n_tokens` in 1..8: act[t][k][f] =
/// swiglu_clamped(gate_e . x[t], up_e . x[t]) for the 8 experts e = ids[t][k]
/// (gate/up `[512][768][2560]`, type IQ4_XS, Q4_K, Q5_K or Q6_K,
/// clamp L = swiglu_clamp_exp[layer], 0 = off); x `[t][2560]`, ids `[t][8]`,
/// act `[t][8][768]`. Each distinct selected expert is read once per call
/// even when several tokens pick it. `scratch` is DecodeScratchBytes().
void ExpertGateUpDecode(const void* gate, const void* up, WeightType type,
                        const float* x, const std::int32_t* ids,
                        std::uint32_t n_tokens, float limit, float* act,
                        void* scratch, hipStream_t stream);
/// Decode, `n_tokens` in 1..8: out[t][k][:] = down_e . act[t][k] for e =
/// ids[t][k] (down `[512][2560][768]`, type Q4_K, Q5_K or Q6_K); act
/// `[t][8][768]`, out `[t][8][2560]`, unweighted. Same single-read-per-expert
/// guarantee.
void ExpertDownDecode(const void* down, WeightType type, const float* act,
                      const std::int32_t* ids, std::uint32_t n_tokens,
                      float* out, void* scratch, hipStream_t stream);
/// Route for 1..8 decode tokens that also builds the expert plan the *Planned
/// launches use (the last block to finish does it). `counter` is a device word
/// that must be zero before the first call and is left zero.
void RouteDecode(const float* logits, const float* bias, std::int32_t* ids,
                 float* weights, std::uint32_t n_tokens, void* scratch,
                 std::uint32_t* counter, hipStream_t stream);
/// ExpertGateUpDecode using the plan RouteDecode left in `scratch`.
void ExpertGateUpDecodePlanned(const void* gate, const void* up,
                               WeightType type, const float* x,
                               const std::int32_t* ids, std::uint32_t n_tokens,
                               float limit, float* act, void* scratch,
                               hipStream_t stream);
/// ExpertDownDecode right after ExpertGateUpDecode on the same `ids` and
/// `scratch`: reuses that call's expert plan (one launch fewer).
void ExpertDownDecodePlanned(const void* down, WeightType type,
                             const float* act, const std::int32_t* ids,
                             std::uint32_t n_tokens, float* out, void* scratch,
                             hipStream_t stream);
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
/// Test/benchmark hooks: the prefill kernels with an explicit token-tile
/// width (nf * 16 tokens, 1 or 2; 0 = the dispatch default). Every combination
/// returns the same bits.
void ExpertGateUpPrefillTile(const void* gate, const void* up, WeightType type,
                             const float* x, const std::int32_t* ids,
                             std::uint32_t n_tokens, float limit, float* act,
                             void* scratch, std::uint32_t nf,
                             hipStream_t stream);
void ExpertDownPrefillTile(const void* down, WeightType type, const float* act,
                           const std::int32_t* ids, std::uint32_t n_tokens,
                           float* out, void* scratch, std::uint32_t nf,
                           hipStream_t stream);

/// res[t][i] += sum_k weights[t][k] * expert_out[t][k][i] + shared_out[t][i]
/// (the shared expert is ungated); in-place on the residual stream.
void Combine(float* res, const float* expert_out, const float* weights,
             const float* shared_out, std::uint32_t n_tokens,
             hipStream_t stream);

}  // namespace moe

// ===========================================================================
// mtp: draft-block input stage and helpers (mtp.hip.cpp)
// ===========================================================================
namespace mtp {

/// out[t] = [ rmsnorm(e[t]; enorm) | rmsnorm(h[t]; hnorm) ] (eps `eps`), the
/// concatenation eh_proj consumes: e, h `[t][2560]`, out `[t][5120]`.
void ConcatNormRows(const float* e, const float* h, const float* enorm,
                    const float* hnorm, float* out, std::uint32_t n_rows,
                    float eps, hipStream_t stream);
/// positions[t] = base + t (device memory).
void FillPositions(std::int32_t* positions, std::int32_t base, std::uint32_t n,
                   hipStream_t stream);
/// Per row, the index (lowest on ties) and value of the largest logit.
void ArgmaxRows(const float* logits, std::uint32_t n_rows, std::uint32_t vocab,
                std::int32_t* index, float* value, hipStream_t stream);

}  // namespace mtp

}  // namespace gufo::models::ling_3_flash::rocm

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_KERNELS_HPP_
