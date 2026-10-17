// Synchronous session state copies for snapshots: plain device-to-host and
// host-to-device transfers of the KDA state, the KDA conv window, and the F16
// MLA latent rows. No kernels. These are test and tool utilities. The serving
// path saves and restores through the component cache (snapshot_copy.hip).
// All copies go through one non-blocking stream per executor and one pinned
// staging buffer, copied in chunks and created on first use. The host side is
// a single-threaded memcpy overlapped with the DMA of the neighbouring half.
// The copy awaits the session's last forward pass through its event instead of
// assuming it finished. That wait holds the executor's capture mutex, because
// it may run on a worker thread while the scheduler thread captures a decode
// graph on the executor stream (executor.hpp).
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "src/models/ling_3_flash/continuation_hooks.hpp"
#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"

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
  std::uint8_t* halves[2];
  hipEvent_t events[2];
  static constexpr std::size_t kChunkBytes = kTransferStagingBytes / 2;
};

Staging MakeStaging(hipStream_t stream, void* buffer,
                    const hipEvent_t (&events)[2]) {
  auto* base = static_cast<std::uint8_t*>(buffer);
  return Staging{
      stream, {base, base + Staging::kChunkBytes}, {events[0], events[1]}};
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
    std::memcpy(host + k * Staging::kChunkBytes, st.halves[k & 1], chunk(k));
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
    std::memcpy(st.halves[half], host + at, n);
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

}  // namespace

bool Executor::EnsureTransferStaging(std::string* error_msg) const {
  // Each member is created once, on its own null check: after a failure the
  // next call reuses what already exists instead of leaking it, and ~Executor
  // frees whatever was created.
  return (transfer_stream_ != nullptr ||
          Check(
              hipStreamCreateWithFlags(&transfer_stream_, hipStreamNonBlocking),
              "transfer stream", error_msg)) &&
         (transfer_events_[0] != nullptr ||
          Check(hipEventCreateWithFlags(&transfer_events_[0],
                                        hipEventDisableTiming),
                "transfer event", error_msg)) &&
         (transfer_events_[1] != nullptr ||
          Check(hipEventCreateWithFlags(&transfer_events_[1],
                                        hipEventDisableTiming),
                "transfer event", error_msg)) &&
         (transfer_staging_ != nullptr ||
          Check(hipHostMalloc(&transfer_staging_, kTransferStagingBytes),
                "pinned snapshot staging", error_msg));
}

bool Executor::SaveSnapshot(const Session& session, std::uint32_t rows,
                            const SnapshotBuffers& out,
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
  const TransferScope in_flight(session.transfers_);
  const std::lock_guard<std::mutex> lock(transfer_mutex_);
  if (!EnsureTransferStaging(error_msg)) {
    return false;
  }
  const Staging staging =
      MakeStaging(transfer_stream_, transfer_staging_, transfer_events_);
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
  const std::size_t width = std::size_t{rows} * kRowBytes;
  for (std::uint32_t m = 0; m < kMlaLayers && rows != 0; ++m) {
    if (!CopyOut(staging, out.mla + m * width,
                 static_cast<const std::uint8_t*>(st.mla_cache) +
                     m * st.cache.LayerStride() * sizeof(std::uint16_t),
                 width, error_msg)) {
      return false;
    }
  }
  if (out.mtp != nullptr && st.mtp_cache != nullptr) {
    // Rows [0, rows - 1): the last row is written by the next forward.
    const std::size_t mtp_width = std::size_t{rows - (rows != 0)} * kRowBytes;
    if (!CopyOut(staging, out.mtp, st.mtp_cache, mtp_width, error_msg)) {
      return false;
    }
  }
  return true;
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
  if (session.continuation_hooks_ != nullptr) {
    // A component-cache slot preserves its borrowed rows before they change.
    try {
      session.continuation_hooks_->BeforeWrite(0, session.capacity(), 0,
                                               session.capacity());
    } catch (const std::exception& e) {
      if (error_msg != nullptr)
        *error_msg = e.what();
      return false;
    }
  }
  const TransferScope in_flight(session.transfers_);
  const std::lock_guard<std::mutex> lock(transfer_mutex_);
  if (!EnsureTransferStaging(error_msg)) {
    return false;
  }
  const Staging staging =
      MakeStaging(transfer_stream_, transfer_staging_, transfer_events_);
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
