#include "src/models/ling_3_flash/config.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "src/models/ling_3_flash/layout.hpp"
#include "src/models/ling_3_flash/weights.hpp"

namespace ling = gufo::models::ling_3_flash;

namespace {

using Value = std::variant<std::uint32_t, float, bool, std::string,
                           std::vector<std::uint32_t>, std::vector<float>>;
using Metadata = std::map<std::string, Value>;

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type;
  std::size_t bytes;
};

// Minimal GGUF: metadata plus an optional tensor table with zeroed payloads.
std::unique_ptr<gufo::core::GgufReader> Open(
    const Metadata& fields, const std::vector<Tensor>& tensors = {}) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{tensors.size()});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, bool>) {
            pod(std::uint32_t{7});
            pod(std::uint8_t{item});
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            pod(std::uint32_t{std::is_same_v<T, std::vector<float>> ? 6U : 4U});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          }
        },
        value);
  }
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    text(t.name);
    pod(std::uint32_t{static_cast<std::uint32_t>(t.dims.size())});
    for (const auto d : t.dims)
      pod(d);
    pod(t.type);
    pod(offset);
    offset += (t.bytes + 31) / 32 * 32;
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  bytes.resize(bytes.size() + offset);
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  assert(reader && error.empty());
  // The reader borrows the buffer; keep it alive for the process lifetime.
  static std::vector<std::vector<std::uint8_t>> keep;
  keep.push_back(std::move(bytes));
  return gufo::core::GgufReader::OpenMemory(keep.back().data(),
                                            keep.back().size(), &error);
}

std::optional<ling::Config> Parse(const Metadata& fields,
                                  std::string* error = nullptr) {
  const auto reader = Open(fields);
  std::string message;
  const auto result = ling::Config::FromGguf(*reader, &message);
  assert(result.has_value() == message.empty());
  if (error != nullptr)
    *error = message;
  return result;
}

Metadata ValidMetadata() {
  Metadata f{{"general.architecture", std::string{"bailingmoe3"}}};
  const auto u32 = [&](const char* key, std::uint32_t v) {
    f[std::string{"bailingmoe3."} + key] = v;
  };
  u32("block_count", 43);
  u32("context_length", 262144);
  u32("embedding_length", 2560);
  u32("feed_forward_length", 6144);
  u32("attention.head_count", 32);
  u32("expert_count", 512);
  u32("expert_used_count", 8);
  u32("expert_group_count", 8);
  u32("expert_group_used_count", 4);
  u32("expert_gating_func", 2);
  u32("attention.key_length", 576);
  u32("attention.value_length", 128);
  u32("vocab_size", 157184);
  u32("ssm.conv_kernel", 4);
  u32("kda.head_dim", 128);
  u32("attention.kv_lora_rank", 512);
  u32("rope.dimension_count", 64);
  u32("attention.key_length_mla", 192);
  u32("attention.value_length_mla", 128);
  u32("expert_feed_forward_length", 768);
  u32("expert_shared_feed_forward_length", 768);
  u32("expert_shared_count", 1);
  u32("leading_dense_block_count", 2);
  u32("nextn_predict_layers", 1);
  f["bailingmoe3.rope.freq_base"] = 6000000.0F;
  f["bailingmoe3.attention.layer_norm_rms_epsilon"] = 1e-6F;
  f["bailingmoe3.kda.safe_gate"] = true;
  f["bailingmoe3.kda.gate_lower_bound"] = -5.0F;
  f["bailingmoe3.expert_weights_scale"] = 2.5F;
  f["bailingmoe3.expert_weights_norm"] = true;
  std::vector<std::uint32_t> kv(43);
  std::vector<float> exp(43), shexp(43);
  for (std::uint32_t il = 0; il < 43; ++il) {
    kv[il] = (il % 6 == 5 || il == 42) ? 1 : 0;
    exp[il] = (il >= 35 && il <= 41) ? 4.0F : 0.0F;
    shexp[il] = il >= 34 && il <= 39   ? 5.0F
                : il == 40 || il == 41 ? 7.0F
                                       : 0.0F;
  }
  f["bailingmoe3.attention.head_count_kv"] = kv;
  f["bailingmoe3.swiglu_clamp_exp"] = exp;
  f["bailingmoe3.swiglu_clamp_shexp"] = shexp;
  return f;
}

void CheckValid() {
  const auto c = Parse(ValidMetadata());
  assert(c);
  assert(c->num_layers == 42 && c->num_layers_all == 43 &&
         c->nextn_layers == 1);
  for (std::uint32_t il = 0; il < 43; ++il) {
    const bool mla = il == 5 || il == 11 || il == 17 || il == 23 || il == 29 ||
                     il == 35 || il == 41 || il == 42;
    assert(c->IsMla(il) == mla);
    assert(ling::IsMlaLayer(il) == mla);
    assert(c->IsDense(il) == (il < 2));
    assert(c->swiglu_clamp_exp[il] == (il >= 35 && il <= 41 ? 4.0F : 0.0F));
  }
  assert(c->swiglu_clamp_shexp[33] == 0.0F &&
         c->swiglu_clamp_shexp[34] == 5.0F);
  assert(c->swiglu_clamp_shexp[39] == 5.0F &&
         c->swiglu_clamp_shexp[40] == 7.0F);
  assert(c->swiglu_clamp_shexp[41] == 7.0F &&
         c->swiglu_clamp_shexp[42] == 0.0F);
  assert(c->vocab_size == 157184 && c->expert_weights_scale == 2.5F);
  // Dense indices cover the 35 KDA and 7 MLA layers without gaps.
  std::uint32_t kda = 0, mla = 0;
  for (std::uint32_t il = 0; il < 42; ++il) {
    if (c->IsMla(il))
      assert(ling::MlaIndex(il) == mla++);
    else
      assert(ling::KdaIndex(il) == kda++);
  }
  assert(kda == ling::kKdaLayers && mla == ling::kMlaLayers);
  static_assert(ling::kKdaProjRows == 20512);
  static_assert(ling::kMlaCacheWidth == 576);
}

void CheckRejections() {
  const auto valid = ValidMetadata();
  std::string error;

  auto wrong = valid;
  wrong["general.architecture"] = std::string{"qwen4exp"};
  assert(!Parse(wrong, &error) &&
         error.find("bailingmoe3") != std::string::npos);

  wrong = valid;
  wrong["bailingmoe3.expert_count"] = std::uint32_t{256};
  assert(!Parse(wrong, &error) &&
         error.find("MoE geometry") != std::string::npos);

  for (const char* key :
       {"bailingmoe3.expert_gating_func", "bailingmoe3.swiglu_clamp_exp",
        "bailingmoe3.attention.head_count_kv", "bailingmoe3.kda.safe_gate",
        "bailingmoe3.attention.kv_lora_rank"}) {
    wrong = valid;
    wrong.erase(key);
    assert(!Parse(wrong, &error) && error.find(key) != std::string::npos);
  }

  wrong = valid;
  wrong["bailingmoe3.expert_gating_func"] = std::uint32_t{1};  // softmax
  assert(!Parse(wrong, &error) && error.find("sigmoid") != std::string::npos);

  wrong = valid;
  wrong["bailingmoe3.expert_weights_norm"] = false;
  assert(!Parse(wrong));

  // A KDA layer where MLA belongs.
  wrong = valid;
  std::get<std::vector<std::uint32_t>>(
      wrong["bailingmoe3.attention.head_count_kv"])[5] = 0;
  assert(!Parse(wrong, &error) && error.find("block 5") != std::string::npos);

  wrong = valid;
  std::get<std::vector<std::uint32_t>>(
      wrong["bailingmoe3.attention.head_count_kv"])[3] = 2;
  assert(!Parse(wrong));

  // Per-layer arrays must describe every block.
  wrong = valid;
  std::get<std::vector<float>>(wrong["bailingmoe3.swiglu_clamp_shexp"])
      .pop_back();
  assert(!Parse(wrong, &error) &&
         error.find("42 entries") != std::string::npos);

  wrong = valid;
  std::get<std::vector<float>>(wrong["bailingmoe3.swiglu_clamp_exp"])[0] =
      -1.0F;
  assert(!Parse(wrong));

  wrong = valid;
  wrong["bailingmoe3.block_count"] = std::uint32_t{44};
  assert(!Parse(wrong));

  wrong = valid;
  wrong["bailingmoe3.nextn_predict_layers"] = std::uint32_t{0};
  assert(!Parse(wrong));

  wrong = valid;
  wrong["bailingmoe3.rope.freq_base"] = 10000.0F;
  assert(!Parse(wrong));

  wrong = valid;
  wrong["bailingmoe3.kda.head_dim"] = std::uint32_t{64};
  assert(!Parse(wrong));
}

void CheckWeightBinding() {
  // Full tensors are hundreds of megabytes, so this test exercises only the
  // early failure paths. Shape and format checks run per tensor.
  std::string error;
  auto reader = Open(ValidMetadata());
  assert(!ling::ModelWeights::Bind(*reader, &error) &&
         error.find("missing tensor token_embd.weight") != std::string::npos);

  reader =
      Open(ValidMetadata(), {{"token_embd.weight", {256, 4}, 12, 144 * 4}});
  assert(!ling::ModelWeights::Bind(*reader, &error) &&
         error.find("token_embd.weight has shape") != std::string::npos);

  assert(ling::IsAllowedExpertGateUpType(gufo::core::GgmlType::kIQ4_XS));
  assert(ling::IsAllowedExpertGateUpType(gufo::core::GgmlType::kQ5_K));
  assert(!ling::IsAllowedExpertGateUpType(gufo::core::GgmlType::kQ4_K));
  assert(ling::IsAllowedExpertDownType(gufo::core::GgmlType::kQ4_K));
  assert(ling::IsAllowedExpertDownType(gufo::core::GgmlType::kQ6_K));
  assert(!ling::IsAllowedExpertDownType(gufo::core::GgmlType::kIQ4_XS));
}

}  // namespace

int main() {
  CheckValid();
  CheckRejections();
  CheckWeightBinding();
  std::cout << "Ling-3.0-flash metadata checks passed.\n";
}
