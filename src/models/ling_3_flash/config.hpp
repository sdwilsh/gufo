#ifndef GUFO_MODELS_LING_3_FLASH_CONFIG_HPP_
#define GUFO_MODELS_LING_3_FLASH_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling_3_flash/layout.hpp"

namespace gufo::models::ling_3_flash {

enum class LayerKind : std::uint8_t {
  kKda,  ///< Kimi delta attention (head_count_kv == 0).
  kMla,  ///< Multi-head latent attention (head_count_kv == 1).
};

/// Architecture parameters of a `bailingmoe3` GGUF artifact. Every value is
/// read from the file and then checked against the one shape this runtime
/// implements (Ling-3.0-flash: 42 trunk layers + 1 MTP block); the constants
/// the kernels compile against live in layout.hpp.
struct Config {
  std::uint32_t num_layers{0};      ///< Trunk layers (MTP block excluded).
  std::uint32_t num_layers_all{0};  ///< block_count, includes the MTP block.
  std::uint32_t nextn_layers{0};    ///< nextn_predict_layers.
  std::uint32_t hidden_size{0};
  std::uint32_t vocab_size{0};
  std::uint32_t context_length{0};
  float rms_eps{0.0F};
  float rope_theta{0.0F};
  std::uint32_t rope_dim{0};  ///< 64, NORM style, MLA only.
  std::uint32_t num_heads{0};
  std::uint32_t dense_ff{0};  ///< feed_forward_length of the dense blocks.
  std::uint32_t leading_dense_layers{0};

  // KDA.
  std::uint32_t kda_head_dim{0};
  std::uint32_t kda_conv_kernel{0};
  float kda_gate_lower_bound{0.0F};

  // MLA.
  std::uint32_t kv_lora_rank{0};
  std::uint32_t key_length{0};        ///< Cache row width, 576.
  std::uint32_t value_length{0};      ///< Absorbed value width, 512 latent.
  std::uint32_t key_length_mla{0};    ///< 192 = nope + rope.
  std::uint32_t value_length_mla{0};  ///< 128.

  // Mixture of experts.
  std::uint32_t num_experts{0};
  std::uint32_t num_experts_used{0};
  std::uint32_t expert_groups{0};
  std::uint32_t expert_groups_used{0};
  std::uint32_t expert_ff{0};
  std::uint32_t shared_expert_ff{0};
  float expert_weights_scale{0.0F};

  /// Per block, `num_layers_all` entries (the MTP block included).
  std::vector<LayerKind> kinds;
  std::vector<float> swiglu_clamp_exp;    ///< Routed experts; 0 = off.
  std::vector<float> swiglu_clamp_shexp;  ///< Dense FFN and shared expert.

  [[nodiscard]] bool IsMla(std::uint32_t il) const noexcept {
    return kinds[il] == LayerKind::kMla;
  }
  [[nodiscard]] bool IsDense(std::uint32_t il) const noexcept {
    return il < leading_dense_layers;
  }

  /// Reads and validates the `bailingmoe3.*` metadata. On failure returns
  /// nullopt and, when given, describes the first problem in `error_msg`.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

[[nodiscard]] inline bool IsLing3Flash(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "bailingmoe3";
}

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_CONFIG_HPP_
