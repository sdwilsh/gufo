// Session state copies for snapshots: plain device<->host transfers of the KDA
// state, KDA conv window and the F16 MLA latent rows. No kernels. A save has
// two phases (executor.hpp): the state a decode step mutates is copied while
// the caller waits; the append-only latent rows (the bulk of the payload) may
// be copied afterwards, beside decode, under a RowHold. All copies go
// through one per-executor non-blocking stream and one pinned staging buffer
// (chunked), so no stream is created and no pageable memory is handed to the
// HIP runtime per snapshot (a pageable copy issued from a worker thread can
// invalidate a global-mode graph capture running elsewhere). The session's last
// forward pass is awaited through its event, never assumed finished; that wait
// holds the executor's capture mutex, because it may run on a worker thread
// while the scheduler thread captures a decode graph on the executor stream
// (executor.hpp). Host copies to and from the staging halves use a small
// persistent thread pool (host_copy_pool.hpp), overlapped with the DMA of the
// neighbouring half.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"
#include "src/models/ling_3_flash/kernels/rocm/host_copy_pool.hpp"

namespace gufo::models::ling_3_flash::rocm {
namespace {

constexpr std::size_t kRowBytes = kMlaCacheWidth * sizeof(std::uint16_t);

bool Check(hipError_t status, const char* what, std::string* error) {
  if (status == hipSuccess) {
    return true;
  }
  if (error != nullptr) {
    *error = std::string(what) + ": " + hipGetErrorString(status);
  }
  return false;
}

/// Waits for an event recorded on the executor stream, never during a decode
/// graph capture on that stream.
bool AwaitForward(std::mutex& capture_mutex, hipEvent_t forward_done,
                  std::string* error) {
  const std::lock_guard<std::mutex> lock(capture_mutex);
  return Check(hipEventSynchronize(forward_done),
               "wait for the session's last forward", error);
}

/// Marks a snapshot transfer of one session as in flight for its duration
/// (Executor::ForwardBatch refuses sessions with one).
class TransferScope {
public:
  explicit TransferScope(std::atomic<std::uint32_t>& count) : count_(count) {
    count_.fetch_add(1, std::memory_order_acq_rel);
  }
  ~TransferScope() { count_.fetch_sub(1, std::memory_order_acq_rel); }
  TransferScope(const TransferScope&) = delete;
  TransferScope& operator=(const TransferScope&) = delete;

private:
  std::atomic<std::uint32_t>& count_;
};

/// The staging buffer as two halves, so that the DMA of one chunk overlaps the
/// host copy of its neighbour. `events` order reuse of a half.
struct Staging {
  hipStream_t stream;
  HostCopyPool& pool;
  std::uint8_t* halves[2];
  hipEvent_t events[2];
  static constexpr std::size_t kChunkBytes = kTransferStagingBytes / 2;
};

Staging MakeStaging(hipStream_t stream, HostCopyPool& pool, void* buffer,
                    const hipEvent_t (&events)[2]) {
  auto* base = static_cast<std::uint8_t*>(buffer);
  return Staging{stream,
                 pool,
                 {base, base + Staging::kChunkBytes},
                 {events[0], events[1]}};
}

/// Waits for queued copies after a failure: they still use the staging halves.
void Drain(const Staging& st) {
  (void)hipStreamSynchronize(st.stream);
}

/// Device -> host through the pinned staging buffer.
bool CopyOut(const Staging& st, std::uint8_t* host, const void* device,
             std::size_t bytes, std::string* error) {
  const auto* src = static_cast<const std::uint8_t*>(device);
  const auto chunk = [&](std::size_t k) {
    return std::min(Staging::kChunkBytes, bytes - k * Staging::kChunkBytes);
  };
  const auto issue = [&](std::size_t k) {
    const std::size_t half = k & 1;
    return Check(hipMemcpyAsync(st.halves[half], src + k * Staging::kChunkBytes,
                                chunk(k), hipMemcpyDeviceToHost, st.stream),
                 "snapshot copy out", error) &&
           Check(hipEventRecord(st.events[half], st.stream),
                 "snapshot copy out event", error);
  };
  const std::size_t chunks =
      (bytes + Staging::kChunkBytes - 1) / Staging::kChunkBytes;
  if (chunks == 0) {
    return true;
  }
  if (!issue(0)) {
    Drain(st);
    return false;
  }
  for (std::size_t k = 0; k < chunks; ++k) {
    // The other half was drained by the host copy of chunk k - 1.
    if ((k + 1 < chunks && !issue(k + 1)) ||
        !Check(hipEventSynchronize(st.events[k & 1]), "snapshot copy out sync",
               error)) {
      Drain(st);
      return false;
    }
    st.pool.Copy(host + k * Staging::kChunkBytes, st.halves[k & 1], chunk(k));
  }
  return true;
}

/// Host -> device through the pinned staging buffer.
bool CopyIn(const Staging& st, void* device, const std::uint8_t* host,
            std::size_t bytes, std::string* error) {
  auto* dst = static_cast<std::uint8_t*>(device);
  for (std::size_t at = 0, k = 0; at < bytes; at += Staging::kChunkBytes, ++k) {
    const std::size_t half = k & 1;
    const std::size_t n = std::min(Staging::kChunkBytes, bytes - at);
    // The DMA that last read this half (chunk k - 2) must be done.
    if (k >= 2 && !Check(hipEventSynchronize(st.events[half]),
                         "snapshot copy in sync", error)) {
      Drain(st);
      return false;
    }
    st.pool.Copy(st.halves[half], host + at, n);
    if (!Check(hipMemcpyAsync(dst + at, st.halves[half], n,
                              hipMemcpyHostToDevice, st.stream),
               "snapshot copy in", error) ||
        !Check(hipEventRecord(st.events[half], st.stream),
               "snapshot copy in event", error)) {
      Drain(st);
      return false;
    }
  }
  if (!Check(hipStreamSynchronize(st.stream), "snapshot copy in sync", error)) {
    return false;
  }
  return true;
}

/// Takes the transfer mutex and counts as waiting until it has it.
class TransferLock {
public:
  TransferLock(std::mutex& mutex, std::atomic<std::uint32_t>& waiting)
      : mutex_(mutex) {
    waiting.fetch_add(1);
    try {
      mutex_.lock();
    } catch (...) {
      waiting.fetch_sub(1);
      throw;
    }
    waiting.fetch_sub(1);
  }
  ~TransferLock() { mutex_.unlock(); }
  TransferLock(const TransferLock&) = delete;
  TransferLock& operator=(const TransferLock&) = delete;

private:
  std::mutex& mutex_;
};

/// Staging unit of the background row copy: a restore waits for at most one.
constexpr std::size_t kRowCopyUnitBytes = std::size_t{8} << 20;

}  // namespace

// ---- Row holds -------------------------------------------------------------

void Session::AcquireRowHold(std::uint32_t rows) const {
  const std::lock_guard<std::mutex> lock(holds_mutex_);
  holds_.insert(rows);
  held_rows_.store(*holds_.rbegin(), std::memory_order_release);
}

void Session::ReleaseRowHold(std::uint32_t rows) const noexcept {
  const std::lock_guard<std::mutex> lock(holds_mutex_);
  holds_.erase(holds_.find(rows));
  held_rows_.store(holds_.empty() ? 0 : *holds_.rbegin(),
                   std::memory_order_release);
  // Notified under the lock: a destructor waiting here may free the condition
  // variable as soon as it wakes and re-takes the mutex.
  holds_released_.notify_all();
}

void Session::AwaitRowHolds(std::uint32_t first_write_row) const {
  if (held_rows_.load(std::memory_order_acquire) <= first_write_row) {
    return;
  }
  std::unique_lock<std::mutex> lock(holds_mutex_);
  holds_released_.wait(lock, [&] {
    return holds_.empty() || *holds_.rbegin() <= first_write_row;
  });
}

void RowHold::Release() noexcept {
  if (session_ != nullptr) {
    const Session* session = std::exchange(session_, nullptr);
    session->ReleaseRowHold(std::exchange(rows_, 0));
  }
}

// ---- Save ------------------------------------------------------------------

bool Executor::CaptureSnapshotState(const Session& session, std::uint32_t rows,
                                    const SnapshotBuffers& out, RowHold* hold,
                                    std::string* error_msg) const {
  if (rows > session.capacity()) {
    if (error_msg != nullptr) {
      *error_msg = "snapshot rows exceed the session capacity";
    }
    return false;
  }
  const auto& st = session.state_;
  if (out.mtp_h != nullptr && st.mtp_cache != nullptr && !session.mtp_valid_) {
    if (error_msg != nullptr) {
      *error_msg = "the session's MTP state is not valid";
    }
    return false;
  }
  // Registered before the first copy: from here on the rows stay as they are
  // for the later row copy, whatever the caller does with the session.
  RowHold local;
  if (rows != 0) {
    session.AcquireRowHold(rows);
    local = RowHold(&session, rows);
  }
  const TransferScope in_flight(session.transfers_);
  const TransferLock lock(transfer_mutex_, transfers_waiting_);
  const Staging staging = MakeStaging(transfer_stream_, *copy_pool_,
                                      transfer_staging_, transfer_events_);
  if (!AwaitForward(capture_mutex_, session.forward_done_, error_msg) ||
      !CopyOut(staging, out.kda_state, st.kda_state,
               kKdaLayers * kKdaStateFloats * sizeof(float), error_msg) ||
      !CopyOut(staging, out.kda_conv, st.kda_conv,
               kKdaLayers * kKdaConvStateFloats * sizeof(float), error_msg)) {
    return false;
  }
  if (out.mtp_h != nullptr && st.mtp_cache != nullptr &&
      !CopyOut(staging, out.mtp_h, st.mtp_h, kHidden * sizeof(float),
               error_msg)) {
    return false;
  }
  *hold = std::move(local);
  return true;
}

void Executor::HoldRowCopiesForTest(bool hold) noexcept {
  const std::lock_guard<std::mutex> lock(row_copy_gate_mutex_);
  row_copies_held_for_test_ = hold;
  row_copy_gate_changed_.notify_all();
}

bool Executor::CopySnapshotRows(const Session& session, std::uint32_t rows,
                                const SnapshotBuffers& out,
                                const std::atomic<bool>* abort,
                                std::string* error_msg) const {
  if (rows > session.capacity()) {
    if (error_msg != nullptr) {
      *error_msg = "snapshot rows exceed the session capacity";
    }
    return false;
  }
  const auto& st = session.state_;
  bool inject = false;
  for (auto failures = row_copy_failures_for_test_.load(); failures != 0;) {
    if (row_copy_failures_for_test_.compare_exchange_weak(failures,
                                                          failures - 1)) {
      inject = true;
      break;
    }
  }
  // One locked unit per staging chunk: device -> pinned staging -> host
  // buffer. The gate and abort flag are checked between chunks, outside the
  // lock, so another transfer waits for at most one chunk.
  const auto copy = [&](std::uint8_t* host, const void* device,
                        std::size_t bytes) {
    for (std::size_t at = 0; at < bytes;) {
      {
        std::unique_lock<std::mutex> gate(row_copy_gate_mutex_);
        while (row_copies_held_for_test_ &&
               !(abort != nullptr && abort->load())) {
          row_copy_gate_changed_.wait_for(gate, std::chrono::milliseconds(5));
        }
      }
      if (abort != nullptr && abort->load()) {
        if (error_msg != nullptr) {
          *error_msg = "snapshot row copy aborted";
        }
        return false;
      }
      // Let a waiting capture or restore go first, once per unit: with
      // back-to-back transfers the counter may never reach zero, and the copy
      // still makes one unit of progress each time round.
      if (transfers_waiting_.load() != 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      const std::size_t n = std::min(kRowCopyUnitBytes, bytes - at);
      const std::lock_guard<std::mutex> lock(transfer_mutex_);
      if (!Check(hipMemcpyAsync(transfer_staging_,
                                static_cast<const std::uint8_t*>(device) + at,
                                n, hipMemcpyDeviceToHost, transfer_stream_),
                 "snapshot row copy out", error_msg) ||
          !Check(hipStreamSynchronize(transfer_stream_),
                 "snapshot row copy out sync", error_msg)) {
        return false;
      }
      std::memcpy(host + at, transfer_staging_, n);
      at += n;
      if (inject) {
        if (error_msg != nullptr) {
          *error_msg = "injected snapshot row copy failure";
        }
        return false;
      }
    }
    return true;
  };
  const std::size_t width = std::size_t{rows} * kRowBytes;
  for (std::uint32_t m = 0; m < kMlaLayers && rows != 0; ++m) {
    if (!copy(out.mla + m * width,
              static_cast<const std::uint8_t*>(st.mla_cache) +
                  m * st.cache.LayerStride() * sizeof(std::uint16_t),
              width)) {
      return false;
    }
  }
  if (out.mtp != nullptr && st.mtp_cache != nullptr) {
    // Rows [0, rows - 1): the last row is written by the next forward.
    const std::size_t mtp_width = std::size_t{rows - (rows != 0)} * kRowBytes;
    if (!copy(out.mtp, st.mtp_cache, mtp_width)) {
      return false;
    }
  }
  return true;
}

bool Executor::SaveSnapshot(const Session& session, std::uint32_t rows,
                            const SnapshotBuffers& out,
                            std::string* error_msg) const {
  RowHold hold;
  return CaptureSnapshotState(session, rows, out, &hold, error_msg) &&
         CopySnapshotRows(session, rows, out, nullptr, error_msg);
}

bool Executor::RestoreSnapshot(Session& session, std::uint32_t rows,
                               const ConstSnapshotBuffers& in,
                               std::string* error_msg) const {
  if (rows > session.capacity()) {
    if (error_msg != nullptr) {
      *error_msg = "snapshot rows exceed the session capacity";
    }
    return false;
  }
  auto& st = session.state_;
  // The restore rewrites rows a background copy may still be reading. Wait
  // before taking the transfer mutex, which that copy needs.
  session.AwaitRowHolds(0);
  const TransferScope in_flight(session.transfers_);
  const TransferLock lock(transfer_mutex_, transfers_waiting_);
  const Staging staging = MakeStaging(transfer_stream_, *copy_pool_,
                                      transfer_staging_, transfer_events_);
  if (!AwaitForward(capture_mutex_, session.forward_done_, error_msg) ||
      !CopyIn(staging, st.kda_state, in.kda_state,
              kKdaLayers * kKdaStateFloats * sizeof(float), error_msg) ||
      !CopyIn(staging, st.kda_conv, in.kda_conv,
              kKdaLayers * kKdaConvStateFloats * sizeof(float), error_msg)) {
    return false;
  }
  const std::size_t width = std::size_t{rows} * kRowBytes;
  // The persistent expanded K/V rows of a layer are a pure function of the
  // latent rows below them. Rows that the restore leaves bit-identical (the
  // shared prefix of the old and the restored history, *in the same numeric
  // family*, which only the bytes can tell) keep their expansion; compare the
  // watermarked rows with the snapshot and rewrite only from the first
  // difference on.
  std::uint32_t same[kMlaLayers]{};
  constexpr std::size_t kCompareRows = 4096;
  std::vector<std::uint8_t> device_rows(kCompareRows * kRowBytes);
  for (std::uint32_t m = 0; m < kMlaLayers && rows != 0; ++m) {
    auto* layer = static_cast<std::uint8_t*>(st.mla_cache) +
                  m * st.cache.LayerStride() * sizeof(std::uint16_t);
    const std::uint32_t limit = std::min(st.mla_expanded_to[m], rows);
    std::uint32_t row = 0;
    while (row < limit) {
      const std::uint32_t n =
          std::min<std::uint32_t>(kCompareRows, limit - row);
      if (!CopyOut(staging, device_rows.data(),
                   layer + std::size_t{row} * kRowBytes,
                   std::size_t{n} * kRowBytes, error_msg)) {
        return false;
      }
      const auto* snap = in.mla + m * width + std::size_t{row} * kRowBytes;
      const auto diff = std::mismatch(device_rows.begin(),
                                      device_rows.begin() + n * kRowBytes, snap)
                            .first -
                        device_rows.begin();
      if (static_cast<std::size_t>(diff) < std::size_t{n} * kRowBytes) {
        row += static_cast<std::uint32_t>(diff / kRowBytes);
        break;
      }
      row += n;
    }
    same[m] = std::min(row, limit);
    if (!CopyIn(staging, layer + std::size_t{same[m]} * kRowBytes,
                in.mla + m * width + std::size_t{same[m]} * kRowBytes,
                width - std::size_t{same[m]} * kRowBytes, error_msg)) {
      return false;
    }
  }
  if (st.mtp_cache != nullptr) {
    session.mtp_valid_ = false;
    if (in.mtp_h != nullptr) {
      const std::size_t mtp_width = std::size_t{rows - (rows != 0)} * kRowBytes;
      if (!CopyIn(staging, st.mtp_h, in.mtp_h, kHidden * sizeof(float),
                  error_msg) ||
          !CopyIn(staging, st.mtp_cache, in.mtp, mtp_width, error_msg)) {
        return false;
      }
      session.mtp_valid_ = true;
    }
  }
  session.set_position(rows);
  session.OnStateReplaced(rows, same);
  return true;
}

}  // namespace gufo::models::ling_3_flash::rocm
