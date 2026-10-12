#include "src/models/ling_3_flash/weights.hpp"

#include <array>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>

namespace gufo::models::ling_3_flash {
namespace {

using core::GgmlType;

/// The accepted routed expert formats, in one place (see weights.hpp).
constexpr std::array<GgmlType, 4> kExpertGateUpTypes{
    GgmlType::kIQ4_XS, GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K};
constexpr std::array<GgmlType, 3> kExpertDownTypes{
    GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K};

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this runtime decodes. Unknown types get a
/// zero block, which every validation below rejects.
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kF16:
      return {1, 2};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    case GgmlType::kIQ4_XS:
      return {256, 136};
    default:
      return {0, 0};
  }
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t depth, std::span<const GgmlType> types) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      Fail("missing tensor " + name);
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_depth = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_depth != depth ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_depth) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(depth) + "]");
      return t;
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    const Format format = FormatOf(info->type);
    if (!type_ok || format.block == 0 || cols % format.block != 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.depth = depth;
    t.name = info->name;
    // Locate the shard so disk readers can address the payload directly, and
    // so a payload running past its shard is caught here rather than as a
    // fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.SizeBytes() <= regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t depth, std::initializer_list<GgmlType> types) {
    return Get(name, cols, rows, depth,
               std::span<const GgmlType>(types.begin(), types.size()));
  }

  /// Gate/up share one type per layer; the down type is checked on its own.
  void CheckExpertTypes(const std::string& prefix, const LayerWeights& l) {
    if (ok && l.ffn_gate_exps.type != l.ffn_up_exps.type) {
      Fail(prefix + "ffn_gate_exps and ffn_up_exps use different formats");
    }
  }

  LayerWeights Layer(const std::string& p, LayerKind kind, bool moe) {
    LayerWeights l;
    l.kind = kind;
    l.moe = moe;
    constexpr std::uint64_t kHiddenDim = kHidden;
    const auto f32 = {GgmlType::kF32};
    const auto q8 = {GgmlType::kQ8_0};

    l.attn_norm = Get(p + "attn_norm.weight", kHiddenDim, 1, 1, f32);
    l.ffn_norm = Get(p + "ffn_norm.weight", kHiddenDim, 1, 1, f32);
    l.attn_output = Get(p + "attn_output.weight", kKdaInner, kHiddenDim, 1, q8);

    if (kind == LayerKind::kKda) {
      l.attn_q = Get(p + "attn_q.weight", kHiddenDim, kKdaInner, 1, q8);
      l.attn_k = Get(p + "attn_k.weight", kHiddenDim, kKdaInner, 1, q8);
      l.attn_v = Get(p + "attn_v.weight", kHiddenDim, kKdaInner, 1, q8);
      l.ssm_f_a = Get(p + "ssm_f_a.weight", kHiddenDim, kKdaInner, 1, q8);
      l.ssm_g_a = Get(p + "ssm_g_a.weight", kHiddenDim, kKdaInner, 1, q8);
      l.ssm_beta = Get(p + "ssm_beta.weight", kHiddenDim, kHeads, 1, q8);
      l.ssm_conv1d_q =
          Get(p + "ssm_conv1d_q.weight", kKdaConvKernel, 1, kKdaInner, f32);
      l.ssm_conv1d_k =
          Get(p + "ssm_conv1d_k.weight", kKdaConvKernel, 1, kKdaInner, f32);
      l.ssm_conv1d_v =
          Get(p + "ssm_conv1d_v.weight", kKdaConvKernel, 1, kKdaInner, f32);
      l.ssm_dt = Get(p + "ssm_dt.bias", kKdaInner, 1, 1, f32);
      // Stored as [1, 32]: one scalar per head.
      l.ssm_a = Get(p + "ssm_a", 1, kHeads, 1, f32);
      l.ssm_norm = Get(p + "ssm_norm.weight", kKdaHeadDim, 1, 1, f32);
    } else {
      l.attn_q = Get(p + "attn_q.weight", kHiddenDim, kMlaQRows, 1, q8);
      l.attn_kv_a_mqa =
          Get(p + "attn_kv_a_mqa.weight", kHiddenDim, kMlaCacheWidth, 1, q8);
      l.attn_kv_a_norm =
          Get(p + "attn_kv_a_norm.weight", kMlaLatent, 1, 1, f32);
      l.attn_k_b = Get(p + "attn_k_b.weight", kMlaNope, kMlaLatent, kHeads, q8);
      l.attn_v_b =
          Get(p + "attn_v_b.weight", kMlaLatent, kMlaValueDim, kHeads, q8);
      l.attn_gate = Get(p + "attn_gate.weight", kHiddenDim, kHeads, 1, q8);
    }

    if (!moe) {
      l.ffn_gate = Get(p + "ffn_gate.weight", kHiddenDim, kDenseFf, 1, q8);
      l.ffn_up = Get(p + "ffn_up.weight", kHiddenDim, kDenseFf, 1, q8);
      l.ffn_down = Get(p + "ffn_down.weight", kDenseFf, kHiddenDim, 1, q8);
      return l;
    }
    l.router = Get(p + "ffn_gate_inp.weight", kHiddenDim, kExperts, 1, f32);
    l.exp_probs_b = Get(p + "exp_probs_b.bias", kExperts, 1, 1, f32);
    l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", kHiddenDim, kExpertFf,
                          kExperts, kExpertGateUpTypes);
    l.ffn_up_exps = Get(p + "ffn_up_exps.weight", kHiddenDim, kExpertFf,
                        kExperts, kExpertGateUpTypes);
    l.ffn_down_exps = Get(p + "ffn_down_exps.weight", kExpertFf, kHiddenDim,
                          kExperts, kExpertDownTypes);
    CheckExpertTypes(p, l);
    l.ffn_gate_shexp =
        Get(p + "ffn_gate_shexp.weight", kHiddenDim, kSharedFf, 1, q8);
    l.ffn_up_shexp =
        Get(p + "ffn_up_shexp.weight", kHiddenDim, kSharedFf, 1, q8);
    l.ffn_down_shexp =
        Get(p + "ffn_down_shexp.weight", kSharedFf, kHiddenDim, 1, q8);
    return l;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format f = FormatOf(type);
  if (f.block == 0 || cols % f.block != 0) {
    return 0;
  }
  return static_cast<std::size_t>(cols / f.block) * f.bytes;
}

bool IsAllowedExpertGateUpType(GgmlType type) noexcept {
  for (const auto t : kExpertGateUpTypes) {
    if (t == type)
      return true;
  }
  return false;
}

bool IsAllowedExpertDownType(GgmlType type) noexcept {
  for (const auto t : kExpertDownTypes) {
    if (t == type)
      return true;
  }
  return false;
}

bool HasMtpBlock(const core::GgufReader& reader) {
  return reader.FindTensor("blk." + std::to_string(kMtpBlock) +
                           ".nextn.eh_proj.weight") != nullptr;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;

  w.token_embd = b.Get("token_embd.weight", kHidden, kVocab, 1,
                       {GgmlType::kQ4_K, GgmlType::kQ8_0});
  w.output_norm = b.Get("output_norm.weight", kHidden, 1, 1, {GgmlType::kF32});
  w.output = b.Get("output.weight", kHidden, kVocab, 1, {GgmlType::kF16});
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    w.layers.push_back(b.Layer("blk." + std::to_string(il) + ".", c.kinds[il],
                               !c.IsDense(il)));
  }

  // The MTP block is optional: trunk-only files simply lack it.
  const std::string mtp = "blk." + std::to_string(kMtpBlock) + ".";
  if (b.ok && HasMtpBlock(reader)) {
    MtpWeights m;
    m.block = b.Layer(mtp, c.kinds[kMtpBlock], true);
    m.eh_proj = b.Get(mtp + "nextn.eh_proj.weight", 2 * kHidden, kHidden, 1,
                      {GgmlType::kQ8_0});
    m.enorm =
        b.Get(mtp + "nextn.enorm.weight", kHidden, 1, 1, {GgmlType::kF32});
    m.hnorm =
        b.Get(mtp + "nextn.hnorm.weight", kHidden, 1, 1, {GgmlType::kF32});
    m.layer_output_norm = b.Get(mtp + "layer_output_norm.weight", kHidden, 1, 1,
                                {GgmlType::kF32});
    w.mtp = std::move(m);
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::ling_3_flash
