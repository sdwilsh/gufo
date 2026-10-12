#ifndef GUFO_MODELS_LING_3_FLASH_WEIGHTS_HPP_
#define GUFO_MODELS_LING_3_FLASH_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling_3_flash/config.hpp"

namespace gufo::models::ling_3_flash {

/// Non-owning view of one GGUF tensor. `cols` x `rows` x `depth` follows the
/// GGUF convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `depth` (ne[2]) stacks whole matrices
/// (experts, or heads for `attn_k_b` / `attn_v_b`, or channels for the conv
/// weights, which are [4, 1, 4096]).
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t depth{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * depth;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * depth;
  }
  /// Base address of one stacked matrix (expert or head).
  [[nodiscard]] const std::uint8_t* Slice(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// Formats this runtime accepts for the routed expert matrices. Adding a
/// k-quant means adding it here (and to FormatOf in weights.cpp); the gate and
/// up tensors of one layer must share a type.
[[nodiscard]] bool IsAllowedExpertGateUpType(core::GgmlType type) noexcept;
[[nodiscard]] bool IsAllowedExpertDownType(core::GgmlType type) noexcept;

/// One trunk block. Fields not used by the block's kind stay empty.
struct LayerWeights {
  LayerKind kind{LayerKind::kKda};
  bool moe{false};

  TensorRef attn_norm;  ///< [2560] F32
  TensorRef ffn_norm;   ///< [2560] F32

  // KDA. attn_q/attn_k/attn_v/ssm_f_a/ssm_g_a are Q8_0 [2560 -> 4096] and
  // ssm_beta Q8_0 [2560 -> 32]; the engine stacks them (layout.hpp).
  TensorRef attn_q;   ///< KDA [2560 -> 4096]; MLA [2560 -> 6144]
  TensorRef attn_k;   ///< KDA only
  TensorRef attn_v;   ///< KDA only
  TensorRef ssm_f_a;  ///< gate input
  TensorRef ssm_g_a;  ///< output gate input
  TensorRef ssm_beta;
  TensorRef ssm_conv1d_q;  ///< F32 [4, 1, 4096]
  TensorRef ssm_conv1d_k;
  TensorRef ssm_conv1d_v;
  TensorRef ssm_dt;       ///< F32 [4096] bias (`ssm_dt.bias`)
  TensorRef ssm_a;        ///< F32 [32] = exp(A_log) (named `ssm_a`, no suffix)
  TensorRef ssm_norm;     ///< F32 [128]
  TensorRef attn_output;  ///< Q8_0 [4096 -> 2560], both kinds

  // MLA.
  TensorRef attn_kv_a_mqa;   ///< Q8_0 [2560 -> 576]
  TensorRef attn_kv_a_norm;  ///< F32 [512]
  TensorRef attn_k_b;        ///< Q8_0 [128, 512, 32]
  TensorRef attn_v_b;        ///< Q8_0 [512, 128, 32]
  TensorRef attn_gate;       ///< Q8_0 [2560 -> 32]

  // Dense SwiGLU (blocks 0-1).
  TensorRef ffn_gate;  ///< Q8_0 [2560 -> 6144]
  TensorRef ffn_up;
  TensorRef ffn_down;  ///< Q8_0 [6144 -> 2560]

  // MoE (blocks 2..).
  TensorRef router;         ///< F32 [2560 -> 512]
  TensorRef exp_probs_b;    ///< F32 [512] selection bias
  TensorRef ffn_gate_exps;  ///< [2560 -> 768] x 512, IQ4_XS, Q4_K, Q5_K or Q6_K
  TensorRef ffn_up_exps;
  TensorRef ffn_down_exps;   ///< [768 -> 2560] x 512, Q4_K, Q5_K or Q6_K
  TensorRef ffn_gate_shexp;  ///< Q8_0 [2560 -> 768]
  TensorRef ffn_up_shexp;
  TensorRef ffn_down_shexp;  ///< Q8_0 [768 -> 2560]
};

/// MTP block (block 42): a full MLA + MoE block plus the nextn projections.
struct MtpWeights {
  LayerWeights block;
  TensorRef eh_proj;            ///< Q8_0 [5120 -> 2560]
  TensorRef enorm;              ///< F32 [2560]
  TensorRef hnorm;              ///< F32 [2560]
  TensorRef layer_output_norm;  ///< F32 [2560]
};

/// True when the GGUF carries the block-42 MTP weights (without binding them).
[[nodiscard]] bool HasMtpBlock(const core::GgufReader& reader);

struct ModelWeights {
  Config config;
  TensorRef token_embd;              ///< Q4_K or Q8_0 [2560 -> 157184]
  TensorRef output_norm;             ///< F32 [2560]
  TensorRef output;                  ///< F16 [2560 -> 157184]
  std::vector<LayerWeights> layers;  ///< Trunk blocks 0..41.
  /// Present when the file carries block 42; absent files are accepted and the
  /// block is skipped. A partially present block is an error.
  std::optional<MtpWeights> mtp;

  /// Binds and validates the artifact. Tensor payloads stay mapped and
  /// untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_WEIGHTS_HPP_
