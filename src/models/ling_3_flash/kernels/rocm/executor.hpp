#ifndef GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/ling_3_flash/gpu_dump.hpp"
#include "src/models/ling_3_flash/kernels/rocm/device_model.hpp"
#include "src/models/ling_3_flash/layout.hpp"

namespace gufo::models::ling_3_flash::rocm {

class Executor;

/// Device state of one sequence: KDA recurrent state and conv window of the
/// 35 KDA layers plus the F16 latent cache of the 7 MLA layers (layout.hpp).
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] std::uint32_t capacity() const noexcept {
    return state_.cache.capacity;
  }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return SessionBytes(state_.cache.capacity).Total();
  }
  /// Zeroes the recurrent state. The latent cache needs no clearing: rows at
  /// or past the position are always rewritten before they are read.
  [[nodiscard]] bool Reset(std::string* error_msg = nullptr);
  /// Tokens consumed so far (the next position). Rolling back is assigning a
  /// smaller value after restoring the recurrent state.
  [[nodiscard]] std::uint32_t position() const noexcept {
    return state_.position;
  }
  /// Sets the next position. Persistent expanded K/V rows are valid only for
  /// cache rows below the position (a rollback rewrites rows from here on), so
  /// the per-layer watermarks never stay above it.
  void set_position(std::uint32_t position) noexcept {
    state_.position = position;
  }

private:
  friend class Executor;
  Session() = default;
  SessionState state_;
};

/// Rollback copy of the recurrent state of a Session (device memory). The
/// MLA cache is not copied; rollback truncates the position instead.
class RecurrentSnapshot {
public:
  ~RecurrentSnapshot();
  RecurrentSnapshot(const RecurrentSnapshot&) = delete;
  RecurrentSnapshot& operator=(const RecurrentSnapshot&) = delete;
  [[nodiscard]] std::size_t bytes() const noexcept;

private:
  friend class Executor;
  RecurrentSnapshot() = default;
  float* state_{nullptr};
  float* conv_{nullptr};
};

/// Time spent by the last Forward, from device events.
struct ForwardTiming {
  double total_ms{0.0};
  double head_ms{0.0};  ///< Output norm and lm_head, copy to host excluded.
  std::vector<float> layer_ms;  ///< Per trunk block.
};

struct ForwardRequest {
  std::span<const std::int32_t> tokens;
  /// Host destination of the logits, `rows * vocab` floats, where rows =
  /// tokens.size() - logits_from. Empty skips the output head.
  std::span<float> logits;
  /// First token row whose logits are wanted (UINT32_MAX = last row only).
  std::uint32_t logits_from{UINT32_MAX};
  GpuTensorSink* sink{nullptr};
};

/// Single-stream layer loop of the Ling-3.0-flash trunk (PLAN section 1) over
/// a resident DeviceModel. Owns the activation scratch; sessions own state.
class Executor {
public:
  struct Options {
    std::uint32_t max_batch{2048};  ///< Tokens per Forward.
    std::uint32_t max_context{4096};
    std::uint32_t logit_group{64};  ///< Output-head rows per launch.
  };
  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, const Options& options,
      std::string* error_msg = nullptr);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  /// Runs `tokens` (at most max_batch) at positions position()..position()+n-1
  /// of `session` through the whole trunk and advances its position.
  [[nodiscard]] bool Forward(Session& session, const ForwardRequest& request,
                             std::string* error_msg = nullptr);

  /// Teacher forcing: runs block `il` alone on `in` (host `[n][2560]`
  /// hidden rows) at positions first_pos.. of `session` and writes the block
  /// output to `out` (same shape). Only block `il`'s state in `session` is
  /// touched; the caller feeds the tokens in order and resets between passes.
  [[nodiscard]] bool RunBlock(Session& session, std::uint32_t il,
                              std::uint32_t first_pos,
                              std::span<const float> in, std::span<float> out,
                              GpuTensorSink* sink,
                              std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<RecurrentSnapshot> SaveRecurrent(
      const Session& session, std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreRecurrent(Session& session,
                                      const RecurrentSnapshot& snapshot,
                                      std::string* error_msg = nullptr);

  /// Copies one KDA layer's state / one MLA layer's cache rows to the host
  /// (tests compare them across chunkings).
  [[nodiscard]] bool ReadKdaState(const Session& session, std::uint32_t il,
                                  std::vector<float>* state,
                                  std::vector<float>* conv,
                                  std::string* error_msg = nullptr);
  [[nodiscard]] bool ReadMlaCache(const Session& session, std::uint32_t il,
                                  std::uint32_t rows,
                                  std::vector<std::uint16_t>* out,
                                  std::string* error_msg = nullptr);

  [[nodiscard]] const ForwardTiming& last_timing() const noexcept {
    return timing_;
  }
  [[nodiscard]] const Options& options() const noexcept { return options_; }
  [[nodiscard]] std::uint32_t layer_count() const noexcept {
    return static_cast<std::uint32_t>(model_.layers().size());
  }
  [[nodiscard]] std::uint32_t first_layer() const noexcept {
    return model_.first_layer();
  }
  /// Device bytes of activation scratch (excludes weights and sessions).
  [[nodiscard]] std::size_t scratch_bytes() const noexcept {
    return scratch_total_;
  }

private:
  struct Buffers;
  Executor(const DeviceModel& model, const Options& options);
  bool Init(std::string* error_msg);

  bool Block(const Session& session, std::uint32_t il, std::uint32_t first_pos,
             std::uint32_t n, GpuTensorSink* sink, std::string* error_msg);
  void Kda(const DeviceLayer& l, const Session& session, std::uint32_t il,
           std::uint32_t n, GpuTensorSink* sink);
  void Mla(const DeviceLayer& l, const Session& session, std::uint32_t il,
           std::uint32_t first_pos, std::uint32_t n, GpuTensorSink* sink);
  void Ffn(const DeviceLayer& l, std::uint32_t il, std::uint32_t n,
           GpuTensorSink* sink);
  void Q8(const DeviceTensor& w, const float* x, float* y, std::uint32_t n);
  void DumpFloat(GpuTensorSink* sink, const char* name, int layer,
                 const float* device, std::uint32_t rows, std::uint32_t width);
  void DumpInt(GpuTensorSink* sink, const char* name, int layer,
               const std::int32_t* device, std::uint32_t rows,
               std::uint32_t width);
  bool UploadInputs(std::span<const std::int32_t> tokens, std::uint32_t n,
                    std::uint32_t first_pos, std::string* error_msg);
  bool Head(std::uint32_t n, std::uint32_t from, std::span<float> logits,
            std::string* error_msg);

  const DeviceModel& model_;
  Options options_;
  hipStream_t stream_{nullptr};
  std::unique_ptr<Buffers> b_;
  std::size_t scratch_total_{0};
  std::vector<hipEvent_t> events_;
  ForwardTiming timing_;
};

}  // namespace gufo::models::ling_3_flash::rocm

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
