#ifndef GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_SNAPSHOT_COPY_HPP_
#define GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_SNAPSHOT_COPY_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::ling_3_flash::rocm {

/// Byte-exact asynchronous download for the component cache. Aligned
/// registered host destinations are written by a blit kernel through their
/// device alias (no SDMA engine); other destinations keep HIP's copy path.
/// The caller keeps both buffers alive until the stream finishes the copy.
[[nodiscard]] hipError_t CopySnapshotToHost(void* destination,
                                            const void* source,
                                            std::size_t bytes,
                                            hipStream_t stream);

/// Device-readable address of `pointer` (device memory, or registered/pinned
/// host memory through its alias); null for pageable host memory.
[[nodiscard]] const void* DeviceReadable(const void* pointer) noexcept;

/// Restores `rows` latent rows of `row_bytes` into `destination` (device).
/// The first `compare_rows` rows are compared with what the destination
/// already holds: a differing row is rewritten and its index (`base_row` +
/// row) lowers `*first_mismatch` (device, atomic minimum); identical rows are
/// left untouched. The other rows are copied. `source` must be device
/// readable (DeviceReadable). Byte-exact for any alignment.
[[nodiscard]] hipError_t CompareCopyRows(
    void* destination, const void* source, std::size_t rows,
    std::size_t compare_rows, std::size_t row_bytes, std::uint32_t base_row,
    std::uint32_t* first_mismatch, hipStream_t stream);

}  // namespace gufo::models::ling_3_flash::rocm

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_SNAPSHOT_COPY_HPP_
