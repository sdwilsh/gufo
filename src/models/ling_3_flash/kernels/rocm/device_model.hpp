#ifndef GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"
#include "src/models/ling_3_flash/weights.hpp"

namespace gufo::models::ling_3_flash::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  WeightType type{WeightType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t depth{1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
  [[nodiscard]] const __half* f16() const noexcept {
    return static_cast<const __half*>(data);
  }
};

/// One trunk block. Fields not used by the block's kind stay empty.
struct DeviceLayer {
  LayerKind kind{LayerKind::kKda};
  bool moe{false};
  DeviceTensor attn_norm, ffn_norm, attn_output;

  /// q | k | v | f_a | g_a | beta rows stacked in layout.hpp order
  /// (Q8_0, kKdaProjRows x 2560).
  DeviceTensor kda_proj;
  DeviceTensor conv_q, conv_k, conv_v, ssm_dt, ssm_a, ssm_norm;

  DeviceTensor attn_q, kv_a, kv_a_norm, k_b, v_b, attn_gate;

  DeviceTensor ffn_gate, ffn_up, ffn_down;  ///< dense FFN (blocks 0-1)

  DeviceTensor router, exp_probs_b;
  DeviceTensor gate_exps, up_exps, down_exps;
  DeviceTensor gate_shexp, up_shexp, down_shexp;
};

struct UploadStats {
  std::size_t bytes{0};
  std::size_t tensors{0};
  double seconds{0.0};
};

/// The trunk (blocks 0..41, without the MTP block) in device memory.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams tensors disk to device with core::hip::WeightUpload; the mapped
  /// GGUF payloads are never read. `layer_first` and `layer_limit` (0 = to the
  /// end) upload only blocks [first, first + limit): a debug aid for GPUs too
  /// small for the full artifact. Progress goes to stderr.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::uint32_t layer_first, std::uint32_t layer_limit,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const DeviceTensor& token_embd() const noexcept {
    return token_embd_;
  }
  [[nodiscard]] const DeviceTensor& output_norm() const noexcept {
    return output_norm_;
  }
  [[nodiscard]] const DeviceTensor& output() const noexcept { return output_; }
  /// Resident blocks; element k is block first_layer() + k.
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] std::uint32_t first_layer() const noexcept { return first_; }
  [[nodiscard]] const UploadStats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept {
    return stats_.bytes;
  }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_, output_norm_, output_;
  std::vector<DeviceLayer> layers_;
  std::uint32_t first_{0};
  std::vector<void*> allocations_;
  UploadStats stats_;
};

}  // namespace gufo::models::ling_3_flash::rocm

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_DEVICE_MODEL_HPP_
