#ifndef GUFO_MODELS_LING_3_FLASH_REFERENCE_HPP_
#define GUFO_MODELS_LING_3_FLASH_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/ling_3_flash/weights.hpp"

namespace gufo::models::ling_3_flash {

/// Receives named intermediates while the reference runs. `layer` is the block
/// index (-1 for model-level tensors). `token` is the position of the token in
/// the stream the model was fed (per block in teacher mode).
///
/// Names: embd (layer 0 input), mixer_out (attention output after wo),
/// h_mid (after the mixer residual), ffn_out, h_out (block output), kda_o
/// (recurrence output [4096], before norm and gate), mla_ctx ([4096] after the
/// v_b step, before the per-head gate), router_ids (int32 [8]), router_w ([8]),
/// router_top_ids (int32 [10], the eight selected experts then the two best
/// rejected ones) and router_top_sel (float [10], their biased selection
/// scores sigmoid + bias; router_top_sel[7] - [8] is the selection margin).
class DumpSink {
public:
  virtual ~DumpSink() = default;
  [[nodiscard]] virtual bool Wants(std::string_view name, int layer) const = 0;
  virtual void Float(std::string_view name, int layer, std::uint32_t token,
                     std::span<const float> data) = 0;
  virtual void Int(std::string_view name, int layer, std::uint32_t token,
                   std::span<const std::int32_t> data) = 0;
};

/// In-memory DumpSink that writes `<name>.l<NN>.{f32,i32}` files (raw
/// little-endian, shape [tokens, n]) and an `index.json` into a directory.
class DumpRecorder final : public DumpSink {
public:
  /// Empty `layers` keeps everything; otherwise only the listed block indices
  /// (-1 selects the model-level tensors such as result_norm).
  explicit DumpRecorder(std::vector<int> layers = {})
      : layers_(std::move(layers)) {}

  /// Restricts the recorder to the named intermediates (empty keeps all).
  void SetNames(std::vector<std::string> names) { names_ = std::move(names); }

  [[nodiscard]] bool Wants(std::string_view name, int layer) const override;
  void Float(std::string_view name, int layer, std::uint32_t token,
             std::span<const float> data) override;
  void Int(std::string_view name, int layer, std::uint32_t token,
           std::span<const std::int32_t> data) override;

  /// Writes the files plus index.json; `extra_entries` is spliced into the
  /// "entries" array (already comma separated JSON objects) and `extra_fields`
  /// into the top-level object (`"key": value,` pairs).
  [[nodiscard]] bool Write(const std::string& dir,
                           const std::string& extra_entries,
                           const std::string& extra_fields,
                           std::string* error_msg) const;

private:
  struct Series {
    bool is_int{false};
    std::uint32_t width{0};
    std::uint32_t tokens{0};
    std::vector<float> f;
    std::vector<std::int32_t> i;
  };
  std::vector<int> layers_;
  std::vector<std::string> names_;
  std::map<std::pair<std::string, int>, Series> series_;
};

/// Single-token, float32, scalar reference of the Ling-3.0-flash trunk
/// (PLAN.md section 1). Weights are decoded row by row from the mapped file
/// and released after use, so resident memory stays small. The trunk Step()
/// does not run the MTP block; MtpStep() runs it on request (mtp.hpp). Speed is
/// irrelevant here.
class ReferenceModel {
public:
  struct Options {
    std::uint32_t max_context{4096};
    unsigned threads{4};
    /// Round cached latent/k_pe values to F16 (what the oracle's f16 KV does).
    bool f16_cache{false};
    /// Cross-check absorbed vs non-absorbed MLA on the first MLA block.
    bool check_mla{true};
    /// Emulate the F16 roundings of the GPU PREFILL kernels (default off):
    /// every Q8_0/K-quant matmul rounds the activation row and the
    /// dequantised weights to F16 (RNE) before a dot accumulated like the WMMA
    /// unit (F32, truncated toward zero after every 8 products)
    /// (GemmQ8_0 and the MoE prefill WMMA kernels), the latent cache is F16,
    /// and MLA attention follows the WMMA kernel: Q*scale in F16, 16-key tiles
    /// with running max, P = F16(exp(s - max)) with the sum over the rounded P,
    /// F32 accumulation. F32 routers, KDA, k_b/v_b products, norms and
    /// residuals stay F32 exactly like the GPU.
    bool emulate_gpu_f16{false};
    /// With the above: round the lm_head activation to F16 as GemmF16 does
    /// (the GPU only does so when more than 8 logit rows are computed).
    bool emulate_f16_head{false};
    /// Test-only F64 mode (cpu::SetF64, process-wide): dot products, norms,
    /// softmax, sigmoid/silu, rope angles, router, MoE/FFN combine, MLA
    /// attention and the KDA recurrence (with a double state) accumulate in
    /// double. Activations between operators are still stored as float.
    /// Ignored together with emulate_gpu_f16.
    bool f64{false};
    /// madvise(MADV_DONTNEED) large weight ranges after use.
    bool release_pages{true};
  };

  ReferenceModel(const ModelWeights& weights, const Options& options);

  /// Runs one token through the whole trunk at the next position. `logits`
  /// (vocab floats) is filled when non-empty; empty skips the output matrix.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          DumpSink* sink = nullptr,
                          std::string* error_msg = nullptr);

  /// Runs block `il` alone on a supplied input (teacher forcing). Each block
  /// keeps its own position and recurrent state, so a block must see its
  /// tokens in order.
  [[nodiscard]] bool RunLayer(std::uint32_t il, std::span<const float> in,
                              std::span<float> out, DumpSink* sink = nullptr,
                              std::string* error_msg = nullptr);

  /// One MTP draft step (mtp.hpp): the block-42 draft for `token` (the token
  /// after the one whose hidden state is `h`) at the next MTP position,
  /// appending its latent cache row. `h` is the output-normed trunk hidden of
  /// the previous position (LastHidden() after Step()) or the previous draft's
  /// `h_next`. `logits` (vocab floats) and `h_next` (hidden floats, the
  /// shared-head-normed block output) are filled when non-empty. Needs the
  /// artifact's MTP block. The MTP position advances by one.
  [[nodiscard]] bool MtpStep(std::int32_t token, std::span<const float> h,
                             std::span<float> logits, std::span<float> h_next,
                             DumpSink* sink = nullptr,
                             std::string* error_msg = nullptr);
  /// MTP cache rows written so far (the next MtpStep's position).
  [[nodiscard]] std::uint32_t MtpPosition() const noexcept {
    return layer_pos_.size() > kMtpBlock ? layer_pos_[kMtpBlock] : 0;
  }
  /// Cuts the MTP cache back to `position` rows (rolling a draft chain back).
  void MtpRewind(std::uint32_t position);
  /// Output-normed hidden state (result_norm) of the last Step().
  [[nodiscard]] std::span<const float> LastHidden() const noexcept {
    return last_hidden_;
  }

  /// Embedding of one token (hidden floats).
  [[nodiscard]] bool Embed(std::int32_t token, std::span<float> out,
                           std::string* error_msg = nullptr) const;

  /// Clears all recurrent state, caches and positions.
  void Reset();
  [[nodiscard]] std::uint32_t Position() const noexcept {
    return layer_pos_.empty() ? 0 : layer_pos_[0];
  }

  /// Largest relative L2 difference between absorbed and non-absorbed MLA
  /// output seen so far, and how many (token) comparisons it covers.
  struct MlaCheck {
    int layer{-1};
    std::uint32_t comparisons{0};
    double max_rel_l2{0.0};
  };
  [[nodiscard]] const MlaCheck& mla_check() const noexcept { return check_; }

private:
  void Mv(const TensorRef& t, std::uint64_t slice, std::span<const float> x,
          std::span<float> y, bool parallel = true) const;
  void MvGpuF16(const TensorRef& t, std::uint64_t slice,
                std::span<const float> x, std::span<float> y,
                bool parallel) const;
  void MlaAttendGpuF16(const LayerWeights& l, const float* q,
                       const float* cache, std::size_t n, float* out) const;
  void Release(const TensorRef& t, std::uint64_t slice) const;
  void ReleaseAll(const TensorRef& t) const;
  void Block(std::uint32_t il, std::span<float> h, DumpSink* sink);
  void Kda(std::uint32_t il, std::span<const float> x, std::span<float> out,
           std::uint32_t token, DumpSink* sink);
  void Mla(std::uint32_t il, std::span<const float> x, std::span<float> out,
           std::uint32_t token, DumpSink* sink);
  void Ffn(const TensorRef& gate, const TensorRef& up, const TensorRef& down,
           float clamp, std::span<const float> x, std::span<float> out,
           DumpSink* sink = nullptr, int layer = -1,
           std::uint32_t token = 0) const;
  void Moe(std::uint32_t il, std::span<const float> x, std::span<float> out,
           std::uint32_t token, DumpSink* sink);

  [[nodiscard]] const LayerWeights& Layer(std::uint32_t il) const {
    return il == kMtpBlock ? w_.mtp->block : w_.layers[il];
  }
  [[nodiscard]] std::uint32_t StateLayers() const noexcept {
    return c_.num_layers + (w_.mtp ? 1U : 0U);
  }

  const ModelWeights& w_;
  const Config& c_;
  std::vector<float> last_hidden_;
  Options opt_;
  std::vector<std::uint32_t> layer_pos_;
  std::vector<std::vector<float>> kda_state_;     ///< per block, [h][i][j]
  std::vector<std::vector<double>> kda_state64_;  ///< same, F64 mode
  std::vector<std::vector<float>>
      kda_conv_;  ///< per block, [3 stream][3][4096]
  std::vector<std::vector<float>> mla_cache_;  ///< per block, [pos][576]

  // Non-absorbed cross-check (first MLA block only).
  int check_layer_{-1};
  std::vector<float> check_wk_;
  std::vector<float> check_wv_;
  std::vector<float> check_k_nope_;
  std::vector<float> check_v_;
  std::vector<float> check_k_pe_;
  MlaCheck check_;
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_REFERENCE_HPP_
