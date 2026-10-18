#include "src/models/ling_3_flash/config.hpp"

#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::ling_3_flash {
namespace {

constexpr std::string_view kArchitecture = "bailingmoe3";
constexpr std::string_view kPrefix = "bailingmoe3.";
constexpr std::uint32_t kGatingSigmoid = 2;

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  static std::string Key(std::string_view suffix) {
    return std::string(kPrefix) + std::string(suffix);
  }

  std::uint32_t U32(std::string_view suffix) {
    const auto value = gguf.GetMetadataUint32(Key(suffix));
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + Key(suffix));
    }
    return value.value_or(0);
  }

  float F32(std::string_view suffix) {
    const auto value = gguf.GetMetadataFloat32(Key(suffix));
    if (!value || !std::isfinite(*value)) {
      Fail("missing or invalid GGUF float " + Key(suffix));
      return 0.0F;
    }
    return *value;
  }

  bool Bool(std::string_view suffix) {
    const auto value = gguf.GetMetadataBool(Key(suffix));
    if (!value.has_value()) {
      Fail("missing or invalid GGUF bool " + Key(suffix));
    }
    return value.value_or(false);
  }

  /// Per-block integer array with exactly `count` entries.
  std::vector<std::uint64_t> U64Array(std::string_view suffix,
                                      std::size_t count) {
    std::vector<std::uint64_t> out;
    const auto* meta = gguf.FindMetadata(Key(suffix));
    if (meta == nullptr) {
      Fail("missing GGUF array " + Key(suffix));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto v : *s) {
        if (v < 0) {
          Fail("negative value in GGUF array " + Key(suffix));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + Key(suffix));
      return {};
    }
    if (out.size() != count) {
      Fail(Key(suffix) + " has " + std::to_string(out.size()) +
           " entries, expected " + std::to_string(count));
      return {};
    }
    return out;
  }

  /// Per-block float array with exactly `count` finite, non-negative entries.
  std::vector<float> ClampArray(std::string_view suffix, std::size_t count) {
    std::vector<float> out;
    const auto* meta = gguf.FindMetadata(Key(suffix));
    const auto* d = meta != nullptr
                        ? std::get_if<std::vector<double>>(&meta->value)
                        : nullptr;
    if (d == nullptr) {
      Fail("missing or invalid GGUF float array " + Key(suffix));
      return out;
    }
    if (d->size() != count) {
      Fail(Key(suffix) + " has " + std::to_string(d->size()) +
           " entries, expected " + std::to_string(count));
      return out;
    }
    for (const double v : *d) {
      if (!std::isfinite(v) || v < 0.0) {
        Fail(Key(suffix) + " must hold finite non-negative limits");
        return {};
      }
      out.push_back(static_cast<float>(v));
    }
    return out;
  }
};

bool Near(float a, float b, float rel = 1e-5F) {
  return std::fabs(a - b) <= rel * std::fabs(b);
}

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not bailingmoe3");
    return std::nullopt;
  }

  Config c;
  c.num_layers_all = r.U32("block_count");
  c.nextn_layers = r.U32("nextn_predict_layers");
  if (!r.ok) {
    return std::nullopt;
  }
  if (c.nextn_layers >= c.num_layers_all) {
    r.Fail("nextn_predict_layers must be below block_count");
    return std::nullopt;
  }
  c.num_layers = c.num_layers_all - c.nextn_layers;
  c.context_length = r.U32("context_length");
  c.hidden_size = r.U32("embedding_length");
  c.vocab_size = r.U32("vocab_size");
  c.dense_ff = r.U32("feed_forward_length");
  c.leading_dense_layers = r.U32("leading_dense_block_count");
  c.num_heads = r.U32("attention.head_count");
  c.rms_eps = r.F32("attention.layer_norm_rms_epsilon");
  c.rope_theta = r.F32("rope.freq_base");
  c.rope_dim = r.U32("rope.dimension_count");

  c.kda_head_dim = r.U32("kda.head_dim");
  c.kda_conv_kernel = r.U32("ssm.conv_kernel");
  c.kda_gate_lower_bound = r.F32("kda.gate_lower_bound");
  const bool safe_gate = r.Bool("kda.safe_gate");

  c.kv_lora_rank = r.U32("attention.kv_lora_rank");
  c.key_length = r.U32("attention.key_length");
  c.value_length = r.U32("attention.value_length");
  c.key_length_mla = r.U32("attention.key_length_mla");
  c.value_length_mla = r.U32("attention.value_length_mla");

  c.num_experts = r.U32("expert_count");
  c.num_experts_used = r.U32("expert_used_count");
  c.expert_groups = r.U32("expert_group_count");
  c.expert_groups_used = r.U32("expert_group_used_count");
  c.expert_ff = r.U32("expert_feed_forward_length");
  c.shared_expert_ff = r.U32("expert_shared_feed_forward_length");
  const std::uint32_t shared_count = r.U32("expert_shared_count");
  const std::uint32_t gating = r.U32("expert_gating_func");
  const bool weights_norm = r.Bool("expert_weights_norm");
  c.expert_weights_scale = r.F32("expert_weights_scale");

  const auto kv_heads = r.U64Array("attention.head_count_kv", c.num_layers_all);
  c.swiglu_clamp_exp = r.ClampArray("swiglu_clamp_exp", c.num_layers_all);
  c.swiglu_clamp_shexp = r.ClampArray("swiglu_clamp_shexp", c.num_layers_all);
  if (!r.ok) {
    return std::nullopt;
  }

  // head_count_kv is the layer-kind switch: 0 is KDA, 1 is MLA (one shared
  // latent head).
  for (const auto kv : kv_heads) {
    if (kv > 1) {
      r.Fail("attention.head_count_kv entries must be 0 (KDA) or 1 (MLA)");
      return std::nullopt;
    }
    c.kinds.push_back(kv == 1 ? LayerKind::kMla : LayerKind::kKda);
  }

  // The kernels are specialised for exactly this artifact; reject any other
  // profile before allocating or launching.
  if (c.num_layers != kTrunkLayers || c.num_layers_all != kBlockCount ||
      c.nextn_layers != 1) {
    r.Fail(
        "unsupported Ling layer count: need 42 trunk layers and 1 MTP block");
    return std::nullopt;
  }
  for (std::uint32_t il = 0; il < c.num_layers_all; ++il) {
    if (c.IsMla(il) != IsMlaLayer(il)) {
      r.Fail(
          "unsupported layer pattern: MLA must be at blocks 5, 11, ..., 41 "
          "(and the MTP block); block " +
          std::to_string(il) + " disagrees");
      return std::nullopt;
    }
  }
  if (c.hidden_size != kHidden || c.vocab_size != kVocab ||
      c.num_heads != kHeads || c.dense_ff != kDenseFf ||
      c.leading_dense_layers != kLeadingDenseLayers || c.context_length == 0 ||
      c.context_length > INT32_MAX) {
    r.Fail("unsupported Ling dimensions (hidden, vocab, heads or dense FFN)");
    return std::nullopt;
  }
  if (!Near(c.rms_eps, kRmsEps, 1e-3F) || !Near(c.rope_theta, kMlaRopeBase) ||
      c.rope_dim != kMlaRope) {
    r.Fail("unsupported Ling norm epsilon or RoPE parameters");
    return std::nullopt;
  }
  if (c.kda_head_dim != kKdaHeadDim || c.kda_conv_kernel != kKdaConvKernel ||
      !safe_gate || c.kda_gate_lower_bound != kKdaGateLowerBound) {
    r.Fail(
        "unsupported KDA geometry: need head_dim 128, conv kernel 4 and a "
        "safe gate with lower bound -5");
    return std::nullopt;
  }
  if (c.kv_lora_rank != kMlaLatent || c.key_length != kMlaCacheWidth ||
      c.value_length != kMlaValueDim || c.key_length_mla != kMlaQHeadDim ||
      c.value_length_mla != kMlaValueDim) {
    r.Fail(
        "unsupported MLA geometry: need kv_lora_rank 512, key/value length "
        "576/128 and MLA key/value length 192/128");
    return std::nullopt;
  }
  if (c.num_experts != kExperts || c.num_experts_used != kExpertsUsed ||
      c.expert_groups != kExpertGroups ||
      c.expert_groups_used != kExpertGroupsUsed || c.expert_ff != kExpertFf ||
      c.shared_expert_ff != kSharedFf || shared_count != 1) {
    r.Fail(
        "unsupported MoE geometry: need 512 experts, 8 used in 4 of 8 "
        "groups, expert width 768 and one shared expert of width 768");
    return std::nullopt;
  }
  if (gating != kGatingSigmoid || !weights_norm ||
      c.expert_weights_scale != kExpertWeightsScale) {
    r.Fail(
        "unsupported expert routing: need sigmoid gating (2), normalised "
        "weights and scale 2.5");
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::ling_3_flash
