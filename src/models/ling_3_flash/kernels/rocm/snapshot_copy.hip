// Component-cache transfer kernels (not part of any model numerics).
#include <algorithm>

#include "src/models/ling_3_flash/kernels/rocm/snapshot_copy.hpp"

namespace gufo::models::ling_3_flash::rocm {
namespace {

__global__ void CopyAligned(unsigned char* destination,
                            const unsigned char* source, std::size_t bytes) {
  const auto thread = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const auto stride = std::size_t{gridDim.x} * blockDim.x;
  for (auto i = thread; i < bytes / sizeof(uint4); i += stride)
    reinterpret_cast<uint4*>(destination)[i] =
        reinterpret_cast<const uint4*>(source)[i];
  for (auto i = bytes / sizeof(uint4) * sizeof(uint4) + thread; i < bytes;
       i += stride)
    destination[i] = source[i];
}

// One block per row (grid-stride over rows). Word type W is uint4 when every
// row starts 16-byte aligned on both sides, unsigned char otherwise.
template<class W>
__global__ void CompareCopy(unsigned char* destination,
                            const unsigned char* source, std::size_t rows,
                            std::size_t compare_rows, std::size_t row_bytes,
                            std::uint32_t base_row,
                            std::uint32_t* first_mismatch) {
  const std::size_t words = row_bytes / sizeof(W);
  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    auto* d = reinterpret_cast<W*>(destination + row * row_bytes);
    const auto* s = reinterpret_cast<const W*>(source + row * row_bytes);
    int differs = 1;
    if (row < compare_rows) {
      int local = 0;
      for (std::size_t w = threadIdx.x; w < words; w += blockDim.x) {
        const W a = d[w], b = s[w];
        if constexpr (sizeof(W) == sizeof(uint4))
          local |= a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
        else
          local |= a != b;
      }
      differs = __syncthreads_or(local);
      if (differs && threadIdx.x == 0)
        atomicMin(first_mismatch, base_row + static_cast<std::uint32_t>(row));
    }
    if (differs)
      for (std::size_t w = threadIdx.x; w < words; w += blockDim.x)
        d[w] = s[w];
    __syncthreads();
  }
}

bool Aligned(const void* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & 15) == 0;
}

}  // namespace

hipError_t CopySnapshotToHost(void* destination, const void* source,
                              std::size_t bytes, hipStream_t stream) {
  if (bytes >= 4096 && destination && source && Aligned(destination) &&
      Aligned(source)) {
    void* mapped{};
    const auto status = hipHostGetDevicePointer(&mapped, destination, 0);
    if (status == hipSuccess && mapped && Aligned(mapped)) {
      const auto blocks = static_cast<unsigned>(
          std::min<std::size_t>(160, (bytes - 1) / 4096 + 1));
      hipLaunchKernelGGL(CopyAligned, dim3(blocks), dim3(256), 0, stream,
                         static_cast<unsigned char*>(mapped),
                         static_cast<const unsigned char*>(source), bytes);
      return hipGetLastError();
    }
    if (status != hipSuccess) {
      if (status != hipErrorInvalidValue)
        return status;
      // Pageable host memory: keep HIP's download, clear the sticky error.
      (void)hipGetLastError();
    }
  }
  return hipMemcpyAsync(destination, source, bytes, hipMemcpyDeviceToHost,
                        stream);
}

const void* DeviceReadable(const void* pointer) noexcept {
  if (pointer == nullptr)
    return nullptr;
  hipPointerAttribute_t attributes{};
  if (hipPointerGetAttributes(&attributes, pointer) != hipSuccess) {
    (void)hipGetLastError();
    return nullptr;
  }
  if (attributes.type == hipMemoryTypeDevice ||
      attributes.type == hipMemoryTypeManaged)
    return pointer;
  if (attributes.type == hipMemoryTypeHost && attributes.devicePointer)
    return static_cast<const unsigned char*>(attributes.devicePointer) +
           (static_cast<const unsigned char*>(pointer) -
            static_cast<const unsigned char*>(attributes.hostPointer));
  return nullptr;
}

hipError_t CompareCopyRows(void* destination, const void* source,
                           std::size_t rows, std::size_t compare_rows,
                           std::size_t row_bytes, std::uint32_t base_row,
                           std::uint32_t* first_mismatch, hipStream_t stream) {
  if (rows == 0)
    return hipSuccess;
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(rows, 1024));
  auto* d = static_cast<unsigned char*>(destination);
  const auto* s = static_cast<const unsigned char*>(source);
  if (Aligned(d) && Aligned(s) && row_bytes % sizeof(uint4) == 0)
    hipLaunchKernelGGL(CompareCopy<uint4>, dim3(blocks), dim3(128), 0, stream,
                       d, s, rows, compare_rows, row_bytes, base_row,
                       first_mismatch);
  else
    hipLaunchKernelGGL(CompareCopy<unsigned char>, dim3(blocks), dim3(256), 0,
                       stream, d, s, rows, compare_rows, row_bytes, base_row,
                       first_mismatch);
  return hipGetLastError();
}

}  // namespace gufo::models::ling_3_flash::rocm
