#ifndef GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "src/models/ling_3_flash/gpu_dump.hpp"
#include "src/models/ling_3_flash/kernels/rocm/device_model.hpp"
#include "src/models/ling_3_flash/layout.hpp"

namespace gufo::models::ling_3_flash::rocm {

class Executor;

/// Pinned staging buffer of the snapshot transfer stream.
inline constexpr std::size_t kTransferStagingBytes = std::size_t{64} << 20;

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
    for (std::uint32_t& rows : state_.mla_expanded_to) {
      rows = std::min(rows, position);
    }
  }
  /// Hook called after the whole session state (recurrent state, conv
  /// windows, cache rows [0, position)) was replaced outside a forward pass,
  /// by a snapshot restore. Contract for state that is derived from, or kept
  /// beside, the session state (it must be rebuilt or updated here, never
  /// stored in snapshots):
  ///   * it runs with the session idle, after the copies completed and the
  ///     position was set to `position`;
  ///   * device-side positions (WS10) must take `position` as the new next
  ///     position;
  ///   * derived caches (WS12 expanded K/V) must lower their valid-rows
  ///     watermark to 0 and be rebuilt lazily with identical bits; a pool
  ///     shared between sessions must be updated under its own lock, because
  ///     a restore may run while another session executes.
  /// Nothing is derived yet, so it is empty.
  void OnStateReplaced(std::uint32_t position) noexcept { (void)position; }

private:
  friend class Executor;
  Session() = default;
  void DestroyGraphs() noexcept;
  SessionState state_;
  /// Decode steps (1..8 tokens) captured as HIP graphs, keyed by (n << 4 |
  /// first logits row): at most 44 keys per session (n tokens x n + 1 head
  /// layouts), a few MB each. Graphs stay valid across Reset() (the state
  /// pointers do not change) and are freed with the session. The nodes hold
  /// this session's state pointers, so the graphs live and die with it; the
  /// position-dependent values are read from device memory and need no graph
  /// update.
  std::map<std::uint32_t, hipGraphExec_t> graphs_;
  std::set<std::uint32_t> warmed_;
  /// Returns the persistent-store bytes to the executor's budget on release.
  std::shared_ptr<std::atomic<std::uint64_t>> expanded_pool_;
  std::uint64_t expanded_bytes_{0};
  /// Recorded at the end of every forward pass on this session; snapshot
  /// copies wait on it instead of trusting that the pass has been synchronized.
  hipEvent_t forward_done_{nullptr};
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
  /// Wants the per-block split in last_timing(): the step then runs eagerly
  /// (a replayed graph is measured as a whole, with layer_ms and head_ms 0).
  bool block_timing{false};
  /// Merges per-operation device times of this step into the vector (eager).
  std::vector<OpTiming>* op_profile{nullptr};
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
    /// Replay small-batch (n <= 8) steps as per-session HIP graphs. Tests turn
    /// it off to compare against the eager launch sequence; the results are
    /// bit-identical either way.
    bool graphs{true};
    /// Device bytes, shared by all sessions of the executor, for persistent
    /// expanded K/V of the MLA layers (112 KiB per cached token). A session
    /// takes whole 2048-token blocks first come first served up to its context;
    /// rows beyond it are expanded per call with identical bits. 12 GiB covers
    /// 109,000 tokens of one session; 0 disables the store.
    std::uint64_t expanded_kv_bytes{std::uint64_t{12} << 30};
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

  /// Self-contained state copy (state_io.cpp). `rows` is the position; the
  /// destinations are the section pointers of a snapshot payload
  /// (snapshot_format.hpp). Uses a private stream and only the session's own
  /// buffers, so it may run on a worker while another session executes; the
  /// session must be idle (its last forward completed).
  struct SnapshotBuffers {
    std::uint8_t* kda_state{nullptr};
    std::uint8_t* kda_conv{nullptr};
    std::uint8_t* mla{nullptr};
  };
  struct ConstSnapshotBuffers {
    const std::uint8_t* kda_state{nullptr};
    const std::uint8_t* kda_conv{nullptr};
    const std::uint8_t* mla{nullptr};
  };
  [[nodiscard]] bool SaveSnapshot(const Session& session, std::uint32_t rows,
                                  const SnapshotBuffers& out,
                                  std::string* error_msg = nullptr) const;
  /// Overwrites the session's KDA state, conv window and MLA rows
  /// [0, rows) bit for bit and invalidates derived caches. The position is
  /// the caller's to set.
  [[nodiscard]] bool RestoreSnapshot(Session& session, std::uint32_t rows,
                                     const ConstSnapshotBuffers& in,
                                     std::string* error_msg = nullptr) const;

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

  /// Graph replays so far (tests and the bench check the graph path ran).
  [[nodiscard]] std::uint64_t graph_replays() const noexcept {
    return graph_replays_;
  }
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

  /// One block. `fuse` (decode family only) selects the fused launches; with
  /// `x_ready` the attention norm of this block (x and its quantised planes)
  /// was already produced by the previous block's last launch, and `next` (the
  /// next resident block, if any) gets its attention norm fused into this
  /// block's last launch.
  bool Block(const Session& session, std::uint32_t il, std::uint32_t first_pos,
             std::uint32_t n, GpuTensorSink* sink, bool fuse, bool x_ready,
             const DeviceLayer* next, std::string* error_msg);
  void Kda(const DeviceLayer& l, const Session& session, std::uint32_t il,
           std::uint32_t n, bool fuse, GpuTensorSink* sink);
  void Mla(const DeviceLayer& l, const Session& session, std::uint32_t il,
           std::uint32_t first_pos, std::uint32_t n, bool fuse,
           GpuTensorSink* sink);
  void Ffn(const DeviceLayer& l, std::uint32_t il, std::uint32_t n, bool fuse,
           const DeviceLayer* next, GpuTensorSink* sink);
  void Q8(const DeviceTensor& w, const float* x, float* y, std::uint32_t n);
  /// Attributes the stream time since the previous mark to `label` (profile
  /// steps only; a no-op otherwise). `bytes` is the traffic the op must read.
  void Mark(const char* label, std::size_t bytes);
  void FinishProfile(std::vector<OpTiming>* out);
  /// Q8_0 GEMVs over the planes of the fused norm in `xq` (up to 3 weights).
  void Q8Quantized(
      std::initializer_list<std::pair<const DeviceTensor*, float*>> groups,
      const void* quant, std::uint32_t n, std::uint32_t k);
  void DumpFloat(GpuTensorSink* sink, const char* name, int layer,
                 const float* device, std::uint32_t rows, std::uint32_t width);
  void DumpInt(GpuTensorSink* sink, const char* name, int layer,
               const std::int32_t* device, std::uint32_t rows,
               std::uint32_t width);
  bool UploadInputs(std::span<const std::int32_t> tokens, std::uint32_t n,
                    std::uint32_t first_pos, std::string* error_msg);
  bool Head(std::uint32_t n, std::uint32_t from, std::span<float> logits,
            std::string* error_msg);
  /// Embedding, all blocks and (head_from < n) the output norm, logits GEMV
  /// and argmax of rows [head_from, n) (at most logit_group of them), as
  /// launches only: no host synchronisation, so it can be graph-captured.
  bool Trunk(const Session& session, std::uint32_t n, std::uint32_t first_pos,
             std::uint32_t head_from, bool events, GpuTensorSink* sink,
             std::string* error_msg);
  void HeadLaunch(std::uint32_t n, std::uint32_t from);
  /// Small-batch (n <= 8, no sink) forward through a per-session graph.
  bool ForwardGraph(Session& session, const ForwardRequest& request,
                    std::uint32_t n, std::uint32_t first_pos,
                    std::uint32_t from, bool* launched, std::string* error_msg);
  bool FinishForward(Session& session, const ForwardRequest& request,
                     std::uint32_t n, std::uint32_t from, std::uint32_t layers,
                     bool graph, std::string* error_msg);

  const DeviceModel& model_;
  Options options_;
  hipStream_t stream_{nullptr};
  /// Snapshot transfers: one private stream and one pinned staging buffer per
  /// executor (state_io.cpp), serialized by `transfer_mutex_`. No streams or
  /// pageable copies are created per snapshot.
  hipStream_t transfer_stream_{nullptr};
  void* transfer_staging_{nullptr};
  mutable std::mutex transfer_mutex_;
  std::unique_ptr<Buffers> b_;
  std::shared_ptr<std::atomic<std::uint64_t>> expanded_pool_;
  std::size_t scratch_total_{0};
  std::vector<hipEvent_t> events_;
  ForwardTiming timing_;
  bool graphs_enabled_{true};
  bool capture_invalidated_{false};
  bool capturing_{false};  ///< Trunk() is being recorded into a graph
  hipStream_t branch_stream_{nullptr};  ///< parallel branch inside a capture
  hipEvent_t fork_event_{nullptr};
  hipEvent_t join_event_{nullptr};
  struct OpMark {
    const char* label;
    std::size_t bytes;
  };
  bool profiling_{false};
  std::vector<OpMark> marks_;
  std::vector<hipEvent_t>
      mark_events_;  ///< [0] = start, [i + 1] after marks_[i]
  std::uint64_t graph_replays_{0};
  /// Pinned staging for the per-step host inputs of graph replays (tokens,
  /// positions).
  std::int32_t* staging_{nullptr};

  /// Parts of the prefill MLA attention call (name, ms) recorded while an op
  /// profile is requested; merged into the profile by FinishProfile().
  std::vector<std::pair<const char*, double>> detail_ops_;
};

}  // namespace gufo::models::ling_3_flash::rocm

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
