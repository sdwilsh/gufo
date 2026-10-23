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

  /// Row-concatenates Q8_0 matrices of one width. The parts share a type, so
  /// the encoded rows copy verbatim and sit back to back.
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

bool DeviceModel::BuildDraftHead(DeviceModel& m, std::string* error_msg) {
  const DeviceTensor& head = m.output_;
  if (head.type != WeightType::kF16 || head.cols % 64 != 0 || head.depth != 1) {
    if (error_msg != nullptr) {
      *error_msg =
          "the output head is not an F16 matrix the draft head can "
          "re-encode";
    }
    return false;
  }
  const std::size_t bytes = std::size_t{head.rows} * (head.cols / 32) * 34;
  void* q8 = nullptr;
  if (const auto status = hipMalloc(&q8, bytes); status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "hipMalloc of the MTP draft head failed: " +
                   std::string(hipGetErrorString(status));
    }
    return false;
  }
  m.allocations_.push_back(q8);
  m.stats_.bytes += bytes;
  ++m.stats_.tensors;
  dense::QuantizeF16ToQ8_0(head.f16(), q8, head.rows, head.cols, nullptr);
  if (const auto status = hipDeviceSynchronize(); status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "MTP draft head build failed: " +
                   std::string(hipGetErrorString(status));
    }
    return false;
  }
  m.mtp_.draft_head = {.data = q8,
                       .type = WeightType::kQ8_0,
                       .cols = head.cols,
                       .rows = head.rows,
                       .depth = 1};
  return true;
}

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& w,
                                                 const core::GgufReader& reader,
                                                 std::uint32_t layer_first,
                                                 std::uint32_t layer_limit,
                                                 std::string* error_msg,
                                                 bool with_mtp) {
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
  if (with_mtp) {
    if (!w.mtp) {
      if (error_msg != nullptr) {
        *error_msg = "the artifact has no MTP block (blk.42)";
      }
      return nullptr;
    }
    m->mtp_.block = up.Layer(w.mtp->block);
    m->mtp_.eh_proj = up.Copy(w.mtp->eh_proj);
    m->mtp_.enorm = up.Copy(w.mtp->enorm);
    m->mtp_.hnorm = up.Copy(w.mtp->hnorm);
    m->mtp_.layer_output_norm = up.Copy(w.mtp->layer_output_norm);
    m->has_mtp_ = true;
  }
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
  if (with_mtp && !BuildDraftHead(*m, error_msg)) {
    return nullptr;
  }
  // F16 expansion weights for non-absorbed MLA prefill (dequantised once).
  for (DeviceLayer& layer : m->layers_) {
    if (layer.kind == LayerKind::kKda) {
      continue;
    }
    void* w16 = nullptr;
    const std::size_t bytes = mla::ExpansionWeightBytes();
    const auto status = hipMalloc(&w16, bytes);
    if (status != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg =
            std::string("hipMalloc of MLA expansion weights failed: ") +
            hipGetErrorString(status);
      }
      return nullptr;
    }
    m->allocations_.push_back(w16);
    m->stats_.bytes += bytes;
    ++m->stats_.tensors;
    mla::BuildExpansionWeights(layer.k_b.data, layer.v_b.data,
                               static_cast<__half*>(w16), nullptr);
    if (const auto launch = hipGetLastError(); launch != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg = std::string("MLA expansion weight launch failed: ") +
                     hipGetErrorString(launch);
      }
      return nullptr;
    }
    layer.mla_w16.data = w16;
    layer.mla_w16.type = WeightType::kF16;
  }
  if (const auto sync = hipDeviceSynchronize(); sync != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = std::string("MLA expansion weight build failed: ") +
                   hipGetErrorString(sync);
    }
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
