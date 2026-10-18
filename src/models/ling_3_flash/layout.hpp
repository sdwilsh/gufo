#ifndef GUFO_MODELS_LING_3_FLASH_LAYOUT_HPP_
#define GUFO_MODELS_LING_3_FLASH_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

/// Shared contracts of the Ling-3.0-flash (`bailingmoe3`) runtime: fixed
/// dimensions, tensor row orders and device buffer layouts. Everything here is
/// constants and POD so kernels, the engine and tests can code against one
/// definition. Math is specified in artifacts/ling-3-flash/PLAN.md section 1.
namespace gufo::models::ling_3_flash {

/// Device time of one kind of decode operation (summed over layers and steps),
/// from stream events between consecutive launches of an eager step; the
/// interval includes the launch gap before the operation. `bytes` is the
/// weight/state/cache traffic the operation must read (a lower bound), so
/// bytes / ms is the effective bandwidth.
struct OpTiming {
  std::string name;
  std::uint32_t count{0};
  double ms{0.0};
  double bytes{0.0};
};

/// Host cost of the last forward step or MTP draft step beside its device
/// span: how much of a step is launch issue, device work and host waiting
/// (launch-overhead profiling; measured on every step, steady clock).
struct LaunchTiming {
  /// Step entry until the last launch of the trunk (forward: inputs, every
  /// block, before the output head) or of the whole draft step was issued; a
  /// replayed graph is one launch.
  double issue_ms{0.0};
  double wall_ms{0.0};    ///< Step entry until the host saw it complete.
  double device_ms{0.0};  ///< First to last launch on the device (events).
  bool graph{false};      ///< The forward replayed a captured graph.
};

// ---- Trunk geometry -------------------------------------------------------
inline constexpr std::uint32_t kBlockCount = 43;   ///< Trunk plus MTP block.
inline constexpr std::uint32_t kTrunkLayers = 42;  ///< Blocks 0..41.
inline constexpr std::uint32_t kMtpBlock = 42;
inline constexpr std::uint32_t kHidden = 2560;
inline constexpr std::uint32_t kVocab = 157184;
inline constexpr std::uint32_t kHeads = 32;  ///< KDA and MLA head count.
inline constexpr float kRmsEps = 1e-6F;
inline constexpr std::uint32_t kLeadingDenseLayers = 2;
inline constexpr std::uint32_t kDenseFf = 6144;
/// Blocks 5, 11, ..., 41 (and the MTP block 42) are MLA; the rest are KDA.
inline constexpr std::uint32_t kMlaInterval = 6;
inline constexpr std::uint32_t kMlaLayers = 7;  ///< In the trunk.
inline constexpr std::uint32_t kKdaLayers = kTrunkLayers - kMlaLayers;

[[nodiscard]] constexpr bool IsMlaLayer(std::uint32_t il) noexcept {
  return (il + 1) % kMlaInterval == 0 || il == kMtpBlock;
}
[[nodiscard]] constexpr bool IsDenseLayer(std::uint32_t il) noexcept {
  return il < kLeadingDenseLayers;
}
/// Dense index of a trunk KDA layer among the 35 KDA layers (state arrays).
[[nodiscard]] constexpr std::uint32_t KdaIndex(std::uint32_t il) noexcept {
  return il - il / kMlaInterval;
}
/// Dense index of a trunk MLA layer among the 7 MLA layers (cache arrays).
[[nodiscard]] constexpr std::uint32_t MlaIndex(std::uint32_t il) noexcept {
  return il / kMlaInterval;
}

// ---- KDA (kimi delta attention) -------------------------------------------
inline constexpr std::uint32_t kKdaHeadDim = 128;
inline constexpr std::uint32_t kKdaInner = kHeads * kKdaHeadDim;  ///< 4096
inline constexpr std::uint32_t kKdaConvKernel = 4;
inline constexpr std::uint32_t kKdaConvHistory = kKdaConvKernel - 1;
inline constexpr float kKdaGateLowerBound = -5.0F;
inline constexpr float kKdaL2Eps = 1e-6F;  ///< x / sqrt(sum(x^2) + eps)

/// Row order of the stacked KDA input projection (Q8_0, 2560 columns). The
/// separate GGUF tensors `attn_q`, `attn_k`, `attn_v`, `ssm_f_a`, `ssm_g_a`
/// and `ssm_beta` are row-concatenated at upload time (same type, so Q8_0
/// rows copy verbatim: 80 blocks x 34 bytes = 2720 bytes per row). Within each
/// 4096-row group the order is head-major: row = head * 128 + d.
/// Output row r of token t is `proj[t * kKdaProjRows + r]`.
inline constexpr std::uint32_t kKdaQRow = 0;
inline constexpr std::uint32_t kKdaKRow = 1 * kKdaInner;
inline constexpr std::uint32_t kKdaVRow = 2 * kKdaInner;
inline constexpr std::uint32_t kKdaFaRow = 3 * kKdaInner;  ///< gate input
inline constexpr std::uint32_t kKdaGaRow = 4 * kKdaInner;  ///< output gate
inline constexpr std::uint32_t kKdaBetaRow = 5 * kKdaInner;
inline constexpr std::uint32_t kKdaProjRows = kKdaBetaRow + kHeads;  ///< 20512
inline constexpr std::size_t kQ8_0RowBytes = (kHidden / 32) * 34;

/// Recurrent state of one KDA layer and sequence: F32 `[head][i key][j value]`
/// = 32 * 128 * 128 floats (2 MiB). Element (h, i, j) is at
/// `h * 128 * 128 + i * 128 + j`; decay applies along i, reads along j.
inline constexpr std::size_t kKdaStateFloats =
    std::size_t{kHeads} * kKdaHeadDim * kKdaHeadDim;
/// Conv state of one KDA layer and sequence: F32 `[stream q,k,v][pos][chan]`
/// = 3 * 3 * 4096 floats. `pos` 0 is the oldest of the previous three raw
/// projection outputs (before conv and SiLU); pos 2 the newest. The conv
/// weights are `[chan][tap]` (GGUF ne = {4, 1, 4096}); tap 3 meets the
/// current token: y[t] = silu(sum_tap w[c][tap] * x[t - 3 + tap]).
inline constexpr std::size_t kKdaConvStateFloats =
    std::size_t{3} * kKdaConvHistory * kKdaInner;

// ---- MLA (multi-head latent attention, absorbed form) ---------------------
inline constexpr std::uint32_t kMlaNope = 128;
inline constexpr std::uint32_t kMlaRope = 64;
inline constexpr std::uint32_t kMlaQHeadDim = kMlaNope + kMlaRope;  ///< 192
inline constexpr std::uint32_t kMlaQRows = kHeads * kMlaQHeadDim;   ///< 6144
inline constexpr std::uint32_t kMlaLatent = 512;                    ///< kv_lora
inline constexpr std::uint32_t kMlaValueDim = 128;
/// Cache row: 512 normalized latent then 64 roped k_pe, F16.
inline constexpr std::uint32_t kMlaCacheWidth = kMlaLatent + kMlaRope;  ///< 576
inline constexpr float kMlaRopeBase = 6.0e6F;  ///< NORM (adjacent-pair) RoPE.

/// F16 latent cache of all trunk MLA layers for one sequence.
/// Element (mla layer m, token p, c) is at
/// `m * layer_stride + p * kMlaCacheWidth + c`, `layer_stride` =
/// `capacity * kMlaCacheWidth` elements; `capacity` is the max context.
struct MlaCacheLayout {
  std::uint32_t capacity{0};  ///< Max tokens per layer.
  [[nodiscard]] constexpr std::size_t LayerStride() const noexcept {
    return std::size_t{capacity} * kMlaCacheWidth;
  }
  [[nodiscard]] constexpr std::size_t Elements() const noexcept {
    return LayerStride() * kMlaLayers;
  }
  [[nodiscard]] constexpr std::size_t Bytes() const noexcept {
    return Elements() * 2;
  }
};

// ---- MoE ------------------------------------------------------------------
inline constexpr std::uint32_t kExperts = 512;
inline constexpr std::uint32_t kExpertsUsed = 8;
inline constexpr std::uint32_t kExpertGroups = 8;
inline constexpr std::uint32_t kExpertGroupsUsed = 4;
inline constexpr std::uint32_t kExpertFf = 768;
inline constexpr std::uint32_t kSharedFf = 768;
inline constexpr float kExpertWeightsScale = 2.5F;
inline constexpr float kExpertWeightsNormFloor = 6.1035e-5F;

/// Routing result per token, row-major: ids int32 `[token][8]` (expert index
/// 0..511, in descending selection score) and weights F32 `[token][8]`
/// (final, normalized and scaled; the combine applies them as is).
struct RoutingBuffers {
  std::int32_t* ids{nullptr};
  float* weights{nullptr};
};

// ---- Session --------------------------------------------------------------
/// Device buffers owned by one sequence. Pointers are device addresses; the
/// struct is plain data so it can be passed to kernels or captured by graphs.
struct SessionState {
  float* kda_state{nullptr};  ///< [35 kda layers][kKdaStateFloats]
  float* kda_conv{nullptr};   ///< [35 kda layers][kKdaConvStateFloats]
  void* mla_cache{nullptr};   ///< F16, MlaCacheLayout
  /// MTP block's own latent cache: one more MLA layer, F16 `[capacity][576]`
  /// (null without MTP). Row p is computed from (trunk hidden of position p,
  /// token p + 1); see mtp.hpp.
  void* mtp_cache{nullptr};
  /// Output-normed trunk hidden state of the last committed position
  /// (kHidden floats, null without MTP): the MTP input of the next draft.
  float* mtp_h{nullptr};
  MlaCacheLayout cache;
  std::uint32_t position{0};  ///< Tokens already consumed (next position).
  /// Persistent expanded K/V of the MLA layers (mla::ExpandedLayerBytes(
  /// expanded_rows) per layer), rows [0, expanded_rows) with expanded_rows a
  /// multiple of 2048 (0 = none). Rows are expanded lazily by the prefill
  /// attention: `mla_expanded_to[m]` rows of layer m are current; an append at
  /// first_pos lowers it to first_pos, so truncation, restore and rewrites
  /// need no other bookkeeping.
  void* mla_expanded{nullptr};  ///< F16
  std::uint32_t expanded_rows{0};
  mutable std::uint32_t mla_expanded_to[kMlaLayers]{};
};

/// MTP (multi-token prediction) draft block, a full MLA + MoE block (block 42)
/// that predicts the token after next from (trunk hidden, next token). Verify
/// chunks are at most kMaxVerifyTokens tokens (the small-batch family), so at
/// most kMaxDrafts drafts per cycle.
inline constexpr std::uint32_t kMaxVerifyTokens = 8;
inline constexpr std::uint32_t kMaxDrafts = kMaxVerifyTokens - 1;

/// Byte sizes of the per-sequence state as functions of the max context.
struct SessionSizes {
  std::size_t kda_state;
  std::size_t kda_conv;
  std::size_t mla_cache;
  /// MTP latent cache and trunk-hidden carry (0 without MTP).
  std::size_t mtp_cache;
  /// Rollback snapshot of the recurrent state (KDA state + conv); the MLA
  /// cache needs none, rollback truncates `position`.
  std::size_t snapshot;
  [[nodiscard]] constexpr std::size_t Total() const noexcept {
    return kda_state + kda_conv + mla_cache + mtp_cache;
  }
};
[[nodiscard]] constexpr SessionSizes SessionBytes(std::uint32_t max_context,
                                                  bool mtp = false) noexcept {
  const std::size_t state = kKdaLayers * kKdaStateFloats * sizeof(float);
  const std::size_t conv = kKdaLayers * kKdaConvStateFloats * sizeof(float);
  const std::size_t mtp_bytes =
      mtp ? std::size_t{max_context} * kMlaCacheWidth * 2 +
                kHidden * sizeof(float)
          : 0;
  return {state, conv, MlaCacheLayout{max_context}.Bytes(), mtp_bytes,
          state + conv};
}

/// Per-token F32 activation rows of the scratch arena a forward pass needs,
/// in floats per token. Callers size scratch as `max_tokens * floats`.
inline constexpr std::size_t kHiddenRowFloats = kHidden;
inline constexpr std::size_t kKdaProjRowFloats = kKdaProjRows;
inline constexpr std::size_t kMlaQAbsorbedRowFloats =
    std::size_t{kHeads} * kMlaCacheWidth;

/// Largest chunk of tokens of the small-batch numeric family (engine.hpp).
inline constexpr std::size_t kSmallBatchMaxTokens = 8;

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_LAYOUT_HPP_
