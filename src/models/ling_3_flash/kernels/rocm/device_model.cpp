#include "src/models/ling_3_flash/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <initializer_list>

#include "src/core/hip/weight_upload.hpp"

namespace gufo::models::ling_3_flash::rocm {
namespace {

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<void*>& allocations;
  UploadStats& stats;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  void* Allocate(std::size_t size, const std::string& what) {
    void* ptr = nullptr;
    const auto status = hipMalloc(&ptr, size);
    if (status != hipSuccess) {
      Fail("hipMalloc of " + std::to_string(size) + " bytes failed for " +
           what + ": " + hipGetErrorString(status));
      return nullptr;
    }
    allocations.push_back(ptr);
    stats.bytes += size;
    ++stats.tensors;
    return ptr;
  }

  static DeviceTensor Describe(const TensorRef& t, void* ptr) {
    DeviceTensor d;
    d.data = ptr;
    d.type = static_cast<WeightType>(static_cast<std::uint32_t>(t.type));
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.depth = static_cast<std::uint32_t>(t.depth);
    return d;
  }

  bool Queue(const TensorRef& t, void* destination) {
    if (!stager.Copy(t.shard, t.file_offset, t.SizeBytes(), destination,
                     error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr && !error->empty() ? ": " + *error
                                                : std::string()));
      return false;
    }
    return true;
  }

  DeviceTensor Copy(const TensorRef& t) {
    if (t.empty() || !ok) {
      return {};
    }
    void* ptr = Allocate(t.SizeBytes(), std::string(t.name));
    if (ptr == nullptr || !Queue(t, ptr)) {
      return {};
    }
    return Describe(t, ptr);
  }

  /// Row-concatenates Q8_0 matrices of one width. Same type, so the encoded
  /// rows copy verbatim and the parts simply sit back to back.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    if (!ok) {
      return {};
    }
    const TensorRef& first = **parts.begin();
    std::size_t size = 0;
    std::uint64_t rows = 0;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != first.type || t->cols != first.cols ||
          t->depth != 1) {
        Fail("stacked upload needs tensors of one type and width");
        return {};
      }
      size += t->SizeBytes();
      rows += t->rows;
    }
    void* ptr = Allocate(size, "stacked " + std::string(first.name));
    if (ptr == nullptr) {
      return {};
    }
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!Queue(*t, static_cast<std::uint8_t*>(ptr) + offset)) {
        return {};
      }
      offset += t->SizeBytes();
    }
    TensorRef stacked = first;
    stacked.rows = rows;
    return Describe(stacked, ptr);
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.kind = l.kind;
    d.moe = l.moe;
    d.attn_norm = Copy(l.attn_norm);
    d.ffn_norm = Copy(l.ffn_norm);
    d.attn_output = Copy(l.attn_output);
    if (l.kind == LayerKind::kKda) {
      d.kda_proj = Stack({&l.attn_q, &l.attn_k, &l.attn_v, &l.ssm_f_a,
                          &l.ssm_g_a, &l.ssm_beta});
      if (ok && d.kda_proj.rows != kKdaProjRows) {
        Fail("stacked KDA projection has " + std::to_string(d.kda_proj.rows) +
             " rows");
      }
      d.conv_q = Copy(l.ssm_conv1d_q);
      d.conv_k = Copy(l.ssm_conv1d_k);
      d.conv_v = Copy(l.ssm_conv1d_v);
      d.ssm_dt = Copy(l.ssm_dt);
      d.ssm_a = Copy(l.ssm_a);
      d.ssm_norm = Copy(l.ssm_norm);
    } else {
      d.attn_q = Copy(l.attn_q);
      d.kv_a = Copy(l.attn_kv_a_mqa);
      d.kv_a_norm = Copy(l.attn_kv_a_norm);
      d.k_b = Copy(l.attn_k_b);
      d.v_b = Copy(l.attn_v_b);
      d.attn_gate = Copy(l.attn_gate);
    }
    if (!l.moe) {
      d.ffn_gate = Copy(l.ffn_gate);
      d.ffn_up = Copy(l.ffn_up);
      d.ffn_down = Copy(l.ffn_down);
      return d;
    }
    d.router = Copy(l.router);
    d.exp_probs_b = Copy(l.exp_probs_b);
    d.gate_exps = Copy(l.ffn_gate_exps);
    d.up_exps = Copy(l.ffn_up_exps);
    d.down_exps = Copy(l.ffn_down_exps);
    d.gate_shexp = Copy(l.ffn_gate_shexp);
    d.up_shexp = Copy(l.ffn_up_shexp);
    d.down_shexp = Copy(l.ffn_down_shexp);
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& w,
                                                 const core::GgufReader& reader,
                                                 std::uint32_t layer_first,
                                                 std::uint32_t layer_limit,
                                                 std::string* error_msg) {
  const auto begin = std::chrono::steady_clock::now();
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  auto stager = hip::WeightUpload::Create(regions, error_msg);
  if (!stager) {
    return nullptr;
  }
  Uploader up{*stager, m->allocations_, m->stats_, error_msg};
  if (layer_first >= w.layers.size()) {
    if (error_msg != nullptr) {
      *error_msg = "layer_first is past the trunk";
    }
    return nullptr;
  }
  m->first_ = layer_first;
  const std::size_t available = w.layers.size() - layer_first;
  const std::size_t count = layer_limit == 0
                                ? available
                                : std::min<std::size_t>(layer_limit, available);
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_norm_ = up.Copy(w.output_norm);
  m->output_ = up.Copy(w.output);
  m->layers_.reserve(count);
  for (std::size_t il = 0; il < count && up.ok; ++il) {
    m->layers_.push_back(up.Layer(w.layers[layer_first + il]));
    if ((il + 1) % 6 == 0 || il + 1 == count) {
      std::fprintf(
          stderr, "[ling] queued %zu/%zu layers, %.2f GiB\n", il + 1, count,
          static_cast<double>(m->stats_.bytes) / (1024.0 * 1024.0 * 1024.0));
    }
  }
  if (!up.ok) {
    (void)stager->Finish(nullptr);
    return nullptr;
  }
  if (!stager->Finish(error_msg)) {
    return nullptr;
  }
  m->stats_.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  std::fprintf(
      stderr, "[ling] uploaded %.2f GiB in %.1f s (%zu allocations)\n",
      static_cast<double>(m->stats_.bytes) / (1024.0 * 1024.0 * 1024.0),
      m->stats_.seconds, m->stats_.tensors);
  return m;
}

}  // namespace gufo::models::ling_3_flash::rocm
