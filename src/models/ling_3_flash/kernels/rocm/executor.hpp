#ifndef GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
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
#include "src/models/ling_3_flash/kernels/rocm/host_copy_pool.hpp"
#include "src/models/ling_3_flash/layout.hpp"

namespace gufo::models::ling_3_flash {
class ContinuationAdapter;
class ContinuationHooks;
}  // namespace gufo::models::ling_3_flash

namespace gufo::models::ling_3_flash::rocm {

class Executor;
class Session;

/// Keeps the append-only rows [0, rows) of one session's MLA cache (and rows
/// [0, rows - 1) of its MTP cache) from being rewritten while they are copied
/// to the host in the background (Executor::CaptureSnapshotState /
/// CopySnapshotRows). Decode only appends past `rows`, so it never waits; a
/// reset, a restore, a forward at an earlier position or the session's
/// destruction waits for the release. Movable; releases on destruction.
class RowHold {
public:
  RowHold() = default;
  ~RowHold() { Release(); }
  RowHold(RowHold&& other) noexcept
      : session_(std::exchange(other.session_, nullptr)),
        rows_(std::exchange(other.rows_, 0)) {}
  RowHold& operator=(RowHold&& other) noexcept {
    if (this != &other) {
      Release();
      session_ = std::exchange(other.session_, nullptr);
      rows_ = std::exchange(other.rows_, 0);
    }
    return *this;
  }
  RowHold(const RowHold&) = delete;
  RowHold& operator=(const RowHold&) = delete;
  [[nodiscard]] explicit operator bool() const noexcept {
    return session_ != nullptr;
  }
  /// Idempotent. The session may be destroyed as soon as this returns.
  void Release() noexcept;

private:
  friend class Executor;
  RowHold(const Session* session, std::uint32_t rows) noexcept
      : session_(session), rows_(rows) {}
  const Session* session_{nullptr};
  std::uint32_t rows_{0};
};

/// Pinned staging buffer of the snapshot transfer stream.
inline constexpr std::size_t kTransferStagingBytes = std::size_t{64} << 20;
/// Threads (calling thread included) that copy between snapshot host buffers
/// and the staging buffer. Do not raise it without measuring concurrent decode
/// during a restore: the host copy, SDMA and decode share one LPDDR5X pool.
inline constexpr unsigned kHostCopyThreads = 2;

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
    return SessionBytes(state_.cache.capacity, state_.mtp_cache != nullptr)
        .Total();
  }
  /// True when the session carries the MTP block's latent cache.
  [[nodiscard]] bool mtp_enabled() const noexcept {
    return state_.mtp_cache != nullptr;
  }
  /// True when the MTP cache rows [0, position - 1) and the trunk-hidden carry
  /// are consistent with the session's tokens (always after an MTP forward
  /// from position 0; false after Reset and after a restore from a snapshot
  /// without MTP state, until the session is reset).
  [[nodiscard]] bool mtp_valid() const noexcept { return mtp_valid_; }
  void set_mtp_valid(bool valid) noexcept { mtp_valid_ = valid; }
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
  /// by a snapshot restore. It runs with the session idle, after the copies
  /// completed, and makes everything that is derived from, or kept beside,
  /// the session state consistent with the new contents (derived state is
  /// never stored in snapshots):
  ///   * the position becomes `position` (the next step uploads it to the
  ///     device-side token/position inputs that the eager launches and the
  ///     captured decode graphs read; the graphs hold no position of their
  ///     own, so a replay after a restore starts at the restored position; the
  ///     shared-expert branch stream and its fork/join events exist only
  ///     inside a graph capture, whose branches join before the capture ends,
  ///     so no stream or event stays pending across a restore.
  ///   * The persistent expanded K/V rows (the watermark of every MLA layer)
  ///     are invalidated to 0, because the restore rewrote cache rows below
  ///     the old watermark. They are re-expanded lazily with identical bits.
  ///     The watermark is session-private. The only cross-session state, the
  ///     executor's byte budget, is an atomic counter that a restore does not
  ///     touch (the store keeps its size), so no lock is needed.
  void OnStateReplaced(std::uint32_t position) noexcept {
    state_.position = position;
    for (std::uint32_t& rows : state_.mla_expanded_to) {
      rows = 0;
    }
  }
  /// As above, for a restore that verified `same[m]` leading latent rows of
  /// MLA layer m to be bit-identical to the rows they replace: those rows keep
  /// their expansion (it is a pure function of the row), the rest is
  /// invalidated.
  void OnStateReplaced(std::uint32_t position,
                       const std::uint32_t (&same)[kMlaLayers]) noexcept {
    state_.position = position;
    for (std::uint32_t m = 0; m < kMlaLayers; ++m) {
      state_.mla_expanded_to[m] =
          std::min({state_.mla_expanded_to[m], same[m], position});
    }
  }
  /// Rows of the persistent expanded K/V that are current in the layer with the
  /// most. All resident MLA layers agree. Tests use it to see how much a
  /// restore had to re-expand.
  [[nodiscard]] std::uint32_t expanded_rows_valid() const noexcept {
    return *std::max_element(std::begin(state_.mla_expanded_to),
                             std::end(state_.mla_expanded_to));
  }
  /// Rows of this session's persistent expanded K/V store (whole 2048-row
  /// blocks granted at creation; 0 = every row is expanded per call).
  [[nodiscard]] std::uint32_t expanded_store_rows() const noexcept {
    return state_.expanded_rows;
  }
  /// Component-cache callbacks of this session (null outside that cache).
  [[nodiscard]] const ContinuationHooks* continuation_hooks() const noexcept {
    return continuation_hooks_;
  }

private:
  friend class Executor;
  friend class ::gufo::models::ling_3_flash::ContinuationAdapter;
  Session() = default;
  void DestroyGraphs() noexcept;
  SessionState state_;
  /// Component-cache adapter callbacks (continuation_hooks.hpp); null for
  /// sessions outside the component cache.
  ContinuationHooks* continuation_hooks_{nullptr};
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
  /// Snapshot saves/restores of this session in progress (on a worker
  /// thread). A batched forward refuses a session with a transfer in flight.
  mutable std::atomic<std::uint32_t> transfers_{0};
  bool mtp_valid_{false};
  /// Background row copies of this session (RowHold): the largest `rows` of
  /// any live hold, 0 when none. Read lock-free on every forward.
  mutable std::atomic<std::uint32_t> held_rows_{0};
  mutable std::mutex holds_mutex_;
  mutable std::condition_variable holds_released_;
  mutable std::multiset<std::uint32_t> holds_;
  friend class RowHold;
  /// Registers a hold on rows [0, rows).
  void AcquireRowHold(std::uint32_t rows) const;
  void ReleaseRowHold(std::uint32_t rows) const noexcept;
  /// Blocks while a hold covers a row a write starting at `first_write_row`
  /// would reach, i.e. while some hold has more than `first_write_row` rows
  /// (an MTP row q is reached by first_write_row = q + 1: its snapshot rows
  /// are [0, rows - 1)). Call before any device write that may reach such a
  /// row and before taking executor locks (a copy worker needs them).
  void AwaitRowHolds(std::uint32_t first_write_row) const;
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
  /// Records rollback points (the KDA state and conv window after every row
  /// but the last) so the step can be cut back with MtpRollback(). Needs a
  /// small-batch chunk (n <= 8) and Options::mtp; such a step runs eagerly.
  bool rollback{false};
  /// When not empty, the index of the largest logit of each row from
  /// `logits_from` on (rows = n - from entries), computed on the device; the
  /// output head runs even if `logits` is empty. The rows' logits stay in the
  /// executor (ReadLogitRow) until the next forward or draft.
  std::span<std::int32_t> argmax;
};

/// One MTP draft step: the draft block at `position` for `token` (the token
/// after the position's trunk token), with the trunk hidden of the session's
/// last committed position (`chain` false) or the previous draft's own hidden
/// output (`chain` true, positions advance by one per step).
struct DraftRequest {
  std::int32_t token{0};
  std::uint32_t position{0};
  bool chain{false};
  /// Host destination of the draft logits (vocab floats); empty = argmax only.
  std::span<float> logits;
  /// Test hook: when not empty, the hidden input (kHidden floats) comes from
  /// the host instead of the session/chain and the step needs no valid MTP
  /// state (the caller owns the positions: row `position` is written and rows
  /// 0..position are attended). `h_out`, when not empty, receives the draft's
  /// shared-head-normed hidden output.
  std::span<const float> h_host;
  std::span<float> h_out;
  /// Merges per-operation device times of this step into the vector.
  std::vector<OpTiming>* op_profile{nullptr};
};

/// One session's part of a batched decode step (Executor::ForwardBatch): its
/// tokens are appended at the session's position, as Forward would.
struct BatchSegment {
  Session* session{nullptr};
  /// 1..8 tokens (the small-batch family); all segments of a step together
  /// hold at most 8 rows.
  std::span<const std::int32_t> tokens;
  /// Host destination of the logits of rows [logits_from, n), `vocab` floats
  /// each. Empty skips the output head for this segment.
  std::span<float> logits;
  /// First token row whose logits are wanted (UINT32_MAX = last row only).
  std::uint32_t logits_from{UINT32_MAX};
  /// When not empty, the device argmax of each row from `logits_from` on (the
  /// head runs even if `logits` is empty), as ForwardRequest::argmax.
  std::span<std::int32_t> argmax;
  /// Records rollback points of this segment (2..8 tokens; an MTP executor),
  /// so it can be cut back with MtpRollback(..., rollback_at), as a
  /// Forward(rollback = true) of these tokens alone.
  bool rollback{false};
  /// Out (status kDone): where this segment's rollback points, MTP hidden
  /// rows and head rows live until the next forward or draft. Pass it to
  /// MtpRollback and ReadLogitRow (logit_row is the row of logits_from).
  struct Placement {
    std::uint32_t ckpt0{0};
    std::uint32_t hn0{0};
    std::uint32_t logit_row{0};
  };
  Placement placement{};
  enum class Status : std::uint8_t {
    kPending,
    kDone,      ///< Ran; the session advanced by its tokens.
    kRejected,  ///< Refused before any launch; the session is untouched.
    kFailed,    ///< The step failed after launch; the session state is lost.
  };
  Status status{Status::kPending};  ///< Out.
  std::string error;                ///< Out: why it was rejected or failed.
};

/// One session's draft step inside Executor::MtpDraftBatch (DraftRequest
/// without the test hooks).
struct DraftLane {
  Session* session{nullptr};
  std::int32_t token{0};
  std::uint32_t position{0};
  /// The hidden input is the previous draft's own output (chain) or the
  /// session's trunk-hidden carry. A chained lane reads the output row it
  /// wrote at the same index of the previous MtpDraftBatch call, so callers
  /// keep each chain at a fixed index and let shorter chains drop off the end.
  bool chain{false};
  /// Host destination of the draft logits (vocab floats); empty = argmax only.
  std::span<float> logits;
  std::int32_t argmax{-1};  ///< Out.
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
    /// Run the MTP block: sessions get an MTP latent cache, every forward
    /// keeps it filled, and verify steps can record rollback points (about
    /// 0.5 GiB of executor scratch). Needs a DeviceModel with the MTP block.
    bool mtp{false};
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

  /// One eager forward pass over the segments of several sessions (one
  /// segment per session, at most 8 rows in all). The rows share the weight
  /// passes; each segment's KDA conv window and recurrence, MLA cache append
  /// and attention (over its own context), MTP cache fill and logits use only
  /// its own session. Every segment's logits, recurrent state, conv windows,
  /// cache rows and MTP rows are bit-identical to Forward of the same tokens
  /// on that session alone (the decode-family kernels compute every row
  /// independently of the others; engine.hpp, batch invariance).
  /// Each segment is validated before anything is launched (distinct
  /// session, 1..8 tokens, row budget, context capacity, token range, no
  /// snapshot transfer in flight); a refused segment is kRejected and the
  /// others still run. Returns false when the pass failed after launch: every
  /// launched segment is then kFailed and its session state undefined (the
  /// caller must drop it without device calls; the device may be lost).
  /// Records every member's forward event, so snapshot copies wait for it.
  [[nodiscard]] bool ForwardBatch(std::span<BatchSegment> segments,
                                  std::string* error_msg = nullptr);
  /// Most rows (and so sessions of one token) one ForwardBatch takes: the
  /// decode family's 8, within max_batch and one output-head group, and with
  /// MTP within the hidden rows (rows + segments <= max_batch + 1).
  [[nodiscard]] std::uint32_t batch_row_limit() const noexcept {
    return std::min(
        {std::uint32_t{8}, options_.max_batch, options_.logit_group,
         options_.mtp ? (options_.max_batch + 1) / 2 : std::uint32_t{8}});
  }
  /// Test hook: the MtpDraftBatch call after the next `skip` ones fails after
  /// its launches (host bookkeeping only: no device fault).
  void FailDraftBatchForTest(std::uint32_t skip) noexcept {
    draft_failure_in_ = skip + 1;
  }
  /// Where a FailBatchesForTest failure is reported.
  enum class BatchFailurePoint : std::uint8_t {
    kAfterSync,   ///< as a device error surfacing at the final wait
    kBeforeSync,  ///< right after the trunk launches, before the final wait
  };
  /// Test hook: the next `count` ForwardBatch passes fail after their
  /// launches at `point` (host-side bookkeeping only: no device fault).
  void FailBatchesForTest(
      std::uint32_t count,
      BatchFailurePoint point = BatchFailurePoint::kAfterSync) noexcept {
    batch_failures_for_test_ = count;
    batch_failure_point_ = point;
  }

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
  /// `mtp_h` (kHidden floats) and `mtp` (F16 rows [0, rows - 1) of the MTP
  /// latent cache, packed) are copied only when not null (MTP sessions).
  struct SnapshotBuffers {
    std::uint8_t* kda_state{nullptr};
    std::uint8_t* kda_conv{nullptr};
    std::uint8_t* mla{nullptr};
    std::uint8_t* mtp_h{nullptr};
    std::uint8_t* mtp{nullptr};
  };
  struct ConstSnapshotBuffers {
    const std::uint8_t* kda_state{nullptr};
    const std::uint8_t* kda_conv{nullptr};
    const std::uint8_t* mla{nullptr};
    const std::uint8_t* mtp_h{nullptr};
    const std::uint8_t* mtp{nullptr};
  };
  /// The whole copy, synchronously: CaptureSnapshotState + CopySnapshotRows.
  [[nodiscard]] bool SaveSnapshot(const Session& session, std::uint32_t rows,
                                  const SnapshotBuffers& out,
                                  std::string* error_msg = nullptr) const;
  /// Phase 1 of a split copy: awaits the session's last forward, then copies
  /// the state the next forward mutates (KDA state, conv windows, MTP carry:
  /// about 76 MiB) into `out`. On success `*hold` keeps rows [0, rows) of the
  /// session unwritten until CopySnapshotRows finished (or the hold is
  /// released); the caller may then let the session decode. `out.mla` and
  /// `out.mtp` are untouched. A failure leaves no hold.
  [[nodiscard]] bool CaptureSnapshotState(
      const Session& session, std::uint32_t rows, const SnapshotBuffers& out,
      RowHold* hold, std::string* error_msg = nullptr) const;
  /// Phase 2: copies the append-only MLA rows [0, rows) and MTP rows
  /// [0, rows - 1) into `out.mla` / `out.mtp`. Needs the hold of
  /// CaptureSnapshotState (decode only appends past `rows`, so it may run
  /// beside forwards of the session), runs on any thread, takes the transfer
  /// mutex one pinned-staging chunk at a time so other transfers interleave,
  /// and stops early (false) when `*abort` becomes true.
  [[nodiscard]] bool CopySnapshotRows(const Session& session,
                                      std::uint32_t rows,
                                      const SnapshotBuffers& out,
                                      const std::atomic<bool>* abort,
                                      std::string* error_msg = nullptr) const;
  /// Test hooks: while held, a CopySnapshotRows pauses before each chunk;
  /// the next `count` CopySnapshotRows calls fail after their first chunk.
  void HoldRowCopiesForTest(bool hold) noexcept;
  void FailRowCopiesForTest(std::uint32_t count) noexcept {
    row_copy_failures_for_test_ = count;
  }
  /// Overwrites the session's KDA state, conv window and MLA rows
  /// [0, rows) bit for bit and invalidates derived caches; with MTP buffers
  /// also the MTP rows and carry (the session becomes mtp_valid), otherwise an
  /// MTP session becomes not mtp_valid. The position is the caller's to set.
  [[nodiscard]] bool RestoreSnapshot(Session& session, std::uint32_t rows,
                                     const ConstSnapshotBuffers& in,
                                     std::string* error_msg = nullptr) const;

  /// Runs `submit` on the host once the session's last forward completed and
  /// while no decode graph is being captured on the executor stream (the
  /// component cache submits its transfers on its own streams from other
  /// threads). False, without running `submit`, when the wait fails.
  [[nodiscard]] bool WithSessionIdle(const Session& session,
                                     const std::function<void()>& submit,
                                     std::string* error_msg = nullptr) const;
  /// Process-wide: a HIP call of a forward, draft, rollback or cache transfer
  /// failed since the last successful forward. The component cache then
  /// submits no further device work for borrowed rows (they are retired),
  /// so release paths cannot wait on a lost device. Cleared by the next
  /// successful forward or by a usable device probe.
  [[nodiscard]] static bool DeviceSuspect() noexcept;
  static void NoteDeviceError() noexcept;
  static void ClearDeviceSuspect() noexcept;

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

  /// Selects the head of the MTP draft step: the Q8_0 copy built at load
  /// (default) or the trunk's F16 head (the A/B reference of the acceptance
  /// measurements; the verify path is unaffected). Not thread-safe against a
  /// running draft.
  void SetMtpDraftHeadQ8(bool q8) noexcept { draft_head_q8_ = q8; }
  [[nodiscard]] bool MtpDraftHeadQ8() const noexcept { return draft_head_q8_; }
  /// Runs the MTP draft block for one position (see DraftRequest) and returns
  /// the index of the largest draft logit. The session must be mtp_valid().
  /// Writes MTP cache row `position` (rows past the committed frontier are
  /// rewritten by the next forward, so a draft chain needs no cleanup).
  [[nodiscard]] bool MtpDraft(Session& session, const DraftRequest& request,
                              std::int32_t* argmax,
                              std::string* error_msg = nullptr);
  /// MtpDraft for up to 8 sessions in one pass: the draft block's weights,
  /// the eh_proj and the draft head are read once; each lane uses its own MTP
  /// cache, position and hidden input. Every lane's logits, argmax and MTP
  /// cache row are bit-identical to MtpDraft of that lane alone. All lanes
  /// must be mtp_valid sessions (checked; false and nothing launched
  /// otherwise). A failure after launch leaves the lanes' MTP rows at their
  /// draft positions undefined (past the committed frontier: the next forward
  /// rewrites them); the trunk state and the MTP carry are not touched.
  [[nodiscard]] bool MtpDraftBatch(std::span<DraftLane> lanes,
                                   std::string* error_msg = nullptr);
  /// Cuts the last Forward(rollback = true) of `n` tokens from `first_pos` back
  /// to its first `keep` rows (1 <= keep <= n): restores the KDA state and conv
  /// window after row keep - 1, sets the position and the MTP carry. MLA
  /// caches need no restore (the position truncates them). For a segment of a
  /// ForwardBatch, `at` is its BatchSegment::placement.
  [[nodiscard]] bool MtpRollback(Session& session, std::uint32_t first_pos,
                                 std::uint32_t n, std::uint32_t keep,
                                 std::string* error_msg = nullptr,
                                 BatchSegment::Placement at = {});
  /// Copies row `row` of the logits left by the last head launch (a Forward
  /// with logits or argmax, or MtpDraft) to the host.
  [[nodiscard]] bool ReadLogitRow(std::uint32_t row, std::span<float> out,
                                  std::string* error_msg = nullptr);
  [[nodiscard]] bool mtp_enabled() const noexcept { return options_.mtp; }

  /// Copies the MTP latent cache rows [0, rows) / the trunk-hidden carry to the
  /// host (tests).
  [[nodiscard]] bool ReadMtpCache(const Session& session, std::uint32_t rows,
                                  std::vector<std::uint16_t>* out,
                                  std::string* error_msg = nullptr);
  [[nodiscard]] bool ReadMtpHidden(const Session& session,
                                   std::vector<float>* out,
                                   std::string* error_msg = nullptr);

  /// Graph replays so far (tests and the bench check the graph path ran).
  [[nodiscard]] std::uint64_t graph_replays() const noexcept {
    return graph_replays_;
  }
  /// Decode-graph captures invalidated by another thread's runtime call; the
  /// step ran eagerly and the next one captures again (logged at 1, 2, 4, ...).
  [[nodiscard]] std::uint64_t capture_invalidations() const noexcept {
    return capture_invalidations_;
  }
  /// Test hook: the next `count` decode-graph captures are invalidated the
  /// way an unserialized snapshot wait used to break them (another thread
  /// waits on the session's forward event mid-capture).
  void BreakCapturesForTest(std::uint32_t count) noexcept {
    capture_breaks_for_test_ = count;
  }
  [[nodiscard]] const ForwardTiming& last_timing() const noexcept {
    return timing_;
  }
  /// Host issue / wall time and device span of the last Forward or MtpDraft.
  [[nodiscard]] const LaunchTiming& last_launch() const noexcept {
    return launch_;
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

  /// Rows [row0, row0 + n) of a forward pass: `n` tokens of `session` at
  /// positions first_pos... Forward runs one lane (row0 0); ForwardBatch one
  /// per segment, in ascending row order. Row-wise and weight-shared launches
  /// cover all rows; the per-session ones (KDA conv window and recurrence, MLA
  /// cache append and attention) run once per lane on its own state.
  struct Lane {
    const Session* session{nullptr};
    std::uint32_t first_pos{0};
    std::uint32_t n{0};
    std::uint32_t row0{0};
    /// Rollback points after each row but the last, from checkpoint `ckpt0`.
    bool rollback{false};
    std::uint32_t ckpt0{0};
  };
  /// One block over the lanes' `n` rows in all. `fuse` (decode family only)
  /// selects the fused launches; with `x_ready` the attention norm of this
  /// block (x and its quantised planes) was already produced by the previous
  /// block's last launch, and `next` (the next resident block, if any) gets
  /// its attention norm fused into this block's last launch.
  bool Block(std::span<const Lane> lanes, std::uint32_t il, std::uint32_t n,
             GpuTensorSink* sink, bool fuse, bool x_ready,
             const DeviceLayer* next, std::string* error_msg);
  bool BlockOf(const DeviceLayer& l, std::span<const Lane> lanes,
               std::uint32_t il, std::uint32_t n, GpuTensorSink* sink,
               bool fuse, bool x_ready, const DeviceLayer* next,
               std::string* error_msg);
  /// After a forward of n tokens from first_pos, rows row0.. of the pass:
  /// advances the MTP cache and the trunk-hidden carry (no-op without MTP).
  /// Uses hidden rows [hn0, hn0 + n + 1) (the carry, then the normed rows;
  /// MtpRollback reads them). Overwrites activation rows [0, n) of `h`, so
  /// lanes of one pass are filled in ascending row order (a lane's rows lie
  /// at or past row0 >= the rows the earlier lanes overwrote).
  bool MtpAfterForward(Session& session, std::uint32_t n,
                       std::uint32_t first_pos, std::uint32_t row0,
                       std::uint32_t hn0, std::string* error_msg);
  void Kda(const DeviceLayer& l, std::span<const Lane> lanes, std::uint32_t il,
           std::uint32_t n, bool fuse, GpuTensorSink* sink);
  void Mla(const DeviceLayer& l, std::span<const Lane> lanes, std::uint32_t il,
           std::uint32_t n, bool fuse, GpuTensorSink* sink);
  void Ffn(const DeviceLayer& l, std::uint32_t il, std::uint32_t n, bool fuse,
           const DeviceLayer* next, GpuTensorSink* sink);
  void Q8(const DeviceTensor& w, const float* x, float* y, std::uint32_t n);
  /// Attributes the stream time since the previous mark to `label` (profile
  /// steps only; a no-op otherwise). `bytes` is the traffic the op must read.
  void Mark(const char* label, std::size_t bytes);
  /// Starts an op profile at the current stream position (Mark / Finish).
  void BeginProfile();
  void FinishProfile(std::vector<OpTiming>* out);
  /// Milliseconds since the current step entered Forward / MtpDraft.
  [[nodiscard]] double StepMs() const;
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
  /// Uploads per-row tokens and positions (same length) and waits.
  bool UploadRows(std::span<const std::int32_t> tokens,
                  std::span<const std::int32_t> positions,
                  std::string* error_msg);
  bool Head(std::uint32_t n, std::uint32_t from, std::span<float> logits,
            std::span<std::int32_t> argmax, std::string* error_msg);
  /// Embedding, all blocks and (head_from < n) the output norm, logits GEMV
  /// and argmax of rows [head_from, n) (at most logit_group of them), as
  /// launches only: no host synchronisation, so it can be graph-captured.
  bool Trunk(std::span<const Lane> lanes, std::uint32_t n,
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
  bool draft_head_q8_{true};
  hipStream_t stream_{nullptr};
  /// Snapshot transfers: one private stream and one pinned staging buffer per
  /// executor (state_io.cpp), serialized by `transfer_mutex_`. No streams or
  /// pageable copies are created per snapshot.
  hipStream_t transfer_stream_{nullptr};
  void* transfer_staging_{nullptr};
  std::unique_ptr<HostCopyPool> copy_pool_;
  /// Order the reuse of the two staging halves (state_io.cpp).
  hipEvent_t transfer_events_[2]{nullptr, nullptr};
  mutable std::mutex transfer_mutex_;
  /// Transfers waiting for `transfer_mutex_` (captures and restores). The
  /// background row copy of a snapshot yields to them between its staging
  /// units: std::mutex is not fair, and a copy that relocks at once starves a
  /// restore for the whole copy.
  mutable std::atomic<std::uint32_t> transfers_waiting_{0};
  mutable std::mutex row_copy_gate_mutex_;
  mutable std::condition_variable row_copy_gate_changed_;
  bool row_copies_held_for_test_{false};
  mutable std::atomic<std::uint32_t> row_copy_failures_for_test_{0};
  /// Held while a decode graph is captured on `stream_` and by any other
  /// thread's host wait on an event recorded on `stream_` (a snapshot worker
  /// waiting for its session's last forward). HIP fails such a wait during a
  /// thread-local capture of that stream and breaks the capture. The broken
  /// capture's shared-expert branch launches then run outside the graph, race
  /// the eager fallback, and can leave the executor stream in the failed
  /// capture. Lock order: transfer_mutex_ before capture_mutex_.
  mutable std::mutex capture_mutex_;
  std::unique_ptr<Buffers> b_;
  std::shared_ptr<std::atomic<std::uint64_t>> expanded_pool_;
  std::size_t scratch_total_{0};
  std::vector<hipEvent_t> events_;
  ForwardTiming timing_;
  LaunchTiming launch_;
  std::chrono::steady_clock::time_point step_start_{};
  hipEvent_t draft_events_[2]{nullptr, nullptr};  ///< draft device span
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
  std::uint64_t capture_invalidations_{0};
  std::uint32_t capture_breaks_for_test_{0};
  std::uint32_t batch_failures_for_test_{0};
  std::uint32_t draft_failure_in_{0};
  BatchFailurePoint batch_failure_point_{BatchFailurePoint::kAfterSync};
  /// Pinned staging for the per-step host inputs of graph replays (tokens,
  /// positions).
  std::int32_t* staging_{nullptr};

  /// Parts of the prefill MLA attention call (name, ms) recorded while an op
  /// profile is requested; merged into the profile by FinishProfile().
  std::vector<std::pair<const char*, double>> detail_ops_;
};

}  // namespace gufo::models::ling_3_flash::rocm

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
