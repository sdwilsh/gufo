#ifndef GUFO_MODELS_LING_3_FLASH_GPU_DUMP_HPP_
#define GUFO_MODELS_LING_3_FLASH_GPU_DUMP_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::models::ling_3_flash {

/// Receives intermediate tensors while the GPU runs. `layer` is the block
/// index (-1 for model-level tensors). Rows arrive in token order and are
/// appended; `width` is the row length.
///
/// Names match the CPU reference dumps (reference.hpp): `embd` (layer 0
/// input), `mixer_out` (attention output after wo), `h_mid` (after the mixer
/// residual), `h_out` (block output), `kda_o` (recurrence output, before norm
/// and gate), `router_ids` (int32 [8]) and `router_w` ([8]).
class GpuTensorSink {
public:
  virtual ~GpuTensorSink() = default;
  [[nodiscard]] virtual bool Wants(std::string_view name, int layer) const = 0;
  virtual void Float(std::string_view name, int layer, std::uint32_t width,
                     std::span<const float> rows) = 0;
  virtual void Int(std::string_view name, int layer, std::uint32_t width,
                   std::span<const std::int32_t> rows) = 0;
};

/// Collects tensors in memory and writes them in the CPU reference format:
/// `<name>.l<NN>.{f32,i32}` (raw little endian, [tokens][width]) plus an
/// `index.json` ("gufo-ling-dump-v1") so tools/models/ling_3_flash/
/// compare_layers.py reads both directories alike.
class GpuDumpRecorder final : public GpuTensorSink {
public:
  /// Empty `layers` keeps every layer; otherwise only the listed blocks.
  /// `names` empty keeps every name.
  explicit GpuDumpRecorder(std::set<int> layers = {},
                           std::set<std::string, std::less<>> names = {})
      : layers_(std::move(layers)), names_(std::move(names)) {}

  [[nodiscard]] bool Wants(std::string_view name, int layer) const override;
  void Float(std::string_view name, int layer, std::uint32_t width,
             std::span<const float> rows) override;
  void Int(std::string_view name, int layer, std::uint32_t width,
           std::span<const std::int32_t> rows) override;
  void Clear() { series_.clear(); }

  /// Rows collected so far for (name, layer); empty when absent.
  [[nodiscard]] std::span<const float> Get(
      const std::string& name, int layer, std::uint32_t* width = nullptr) const;

  [[nodiscard]] std::span<const std::int32_t> GetInt(const std::string& name,
                                                     int layer) const;

  /// `extra_fields` is spliced into the top-level object (`"k": v,` pairs),
  /// `extra_entries` into "entries" (comma separated objects).
  [[nodiscard]] bool Write(const std::string& dir,
                           const std::string& extra_entries,
                           const std::string& extra_fields,
                           std::string* error_msg) const;

private:
  struct Series {
    bool is_int{false};
    std::uint32_t width{0};
    std::vector<float> f;
    std::vector<std::int32_t> i;
  };
  std::set<int> layers_;
  std::set<std::string, std::less<>> names_;
  std::map<std::pair<std::string, int>, Series> series_;
};

/// Reads a raw float32 file into `out`; false when missing or misaligned.
[[nodiscard]] bool ReadF32Dump(const std::string& path,
                               std::vector<float>* out);

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_GPU_DUMP_HPP_
