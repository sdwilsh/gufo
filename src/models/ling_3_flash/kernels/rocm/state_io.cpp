// Session state copies for snapshots: plain device<->host transfers of the KDA
// state, KDA conv window and the F16 MLA latent rows. No kernels. All copies go
// through one per-executor non-blocking stream and one pinned staging buffer
// (chunked), so no stream is created and no pageable memory is handed to the
// HIP runtime per snapshot (a pageable copy issued from a worker thread can
// invalidate a global-mode graph capture running elsewhere). The session's last
// forward pass is awaited through its event, never assumed finished.
#include <algorithm>
#include <cstring>

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

/// Device -> host through the pinned staging buffer.
bool CopyOut(hipStream_t stream, void* staging, std::uint8_t* host,
             const void* device, std::size_t bytes, std::string* error) {
  for (std::size_t at = 0; at < bytes;) {
    const std::size_t n = std::min(kTransferStagingBytes, bytes - at);
    if (!Check(hipMemcpyAsync(staging,
                              static_cast<const std::uint8_t*>(device) + at, n,
                              hipMemcpyDeviceToHost, stream),
               "snapshot copy out", error) ||
        !Check(hipStreamSynchronize(stream), "snapshot copy out sync", error)) {
      return false;
    }
    std::memcpy(host + at, staging, n);
    at += n;
  }
  return true;
}

/// Host -> device through the pinned staging buffer.
bool CopyIn(hipStream_t stream, void* staging, void* device,
            const std::uint8_t* host, std::size_t bytes, std::string* error) {
  for (std::size_t at = 0; at < bytes;) {
    const std::size_t n = std::min(kTransferStagingBytes, bytes - at);
    std::memcpy(staging, host + at, n);
    if (!Check(hipMemcpyAsync(static_cast<std::uint8_t*>(device) + at, staging,
                              n, hipMemcpyHostToDevice, stream),
               "snapshot copy in", error) ||
        !Check(hipStreamSynchronize(stream), "snapshot copy in sync", error)) {
      return false;
    }
    at += n;
  }
  return true;
}

}  // namespace

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
  const std::lock_guard<std::mutex> lock(transfer_mutex_);
  if (!Check(hipEventSynchronize(session.forward_done_),
             "wait for the session's last forward", error_msg) ||
      !CopyOut(transfer_stream_, transfer_staging_, out.kda_state, st.kda_state,
               kKdaLayers * kKdaStateFloats * sizeof(float), error_msg) ||
      !CopyOut(transfer_stream_, transfer_staging_, out.kda_conv, st.kda_conv,
               kKdaLayers * kKdaConvStateFloats * sizeof(float), error_msg)) {
    return false;
  }
  const std::size_t width = std::size_t{rows} * kRowBytes;
  for (std::uint32_t m = 0; m < kMlaLayers && rows != 0; ++m) {
    if (!CopyOut(transfer_stream_, transfer_staging_, out.mla + m * width,
                 static_cast<const std::uint8_t*>(st.mla_cache) +
                     m * st.cache.LayerStride() * sizeof(std::uint16_t),
                 width, error_msg)) {
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
  const std::lock_guard<std::mutex> lock(transfer_mutex_);
  if (!Check(hipEventSynchronize(session.forward_done_),
             "wait for the session's last forward", error_msg) ||
      !CopyIn(transfer_stream_, transfer_staging_, st.kda_state, in.kda_state,
              kKdaLayers * kKdaStateFloats * sizeof(float), error_msg) ||
      !CopyIn(transfer_stream_, transfer_staging_, st.kda_conv, in.kda_conv,
              kKdaLayers * kKdaConvStateFloats * sizeof(float), error_msg)) {
    return false;
  }
  const std::size_t width = std::size_t{rows} * kRowBytes;
  for (std::uint32_t m = 0; m < kMlaLayers && rows != 0; ++m) {
    if (!CopyIn(transfer_stream_, transfer_staging_,
                static_cast<std::uint8_t*>(st.mla_cache) +
                    m * st.cache.LayerStride() * sizeof(std::uint16_t),
                in.mla + m * width, width, error_msg)) {
      return false;
    }
  }
  session.set_position(rows);
  session.OnStateReplaced(rows);
  return true;
}

}  // namespace gufo::models::ling_3_flash::rocm
