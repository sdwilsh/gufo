// Ling-3.0-flash MTP (multi-token prediction) helper kernels: the input stage
// of the draft block (see src/models/ling_3_flash/mtp.hpp for the math).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "src/models/ling_3_flash/kernels/rocm/kernels.hpp"

namespace gufo::models::ling_3_flash::rocm::mtp {
namespace {

// out[t] = [ rmsnorm(e[t]; enorm) | rmsnorm(h[t]; hnorm) ], 2 * kHidden floats.
// One block per (row, half); the norm is the expression of dense::RmsNormRows
// (fused multiply-add sum of squares, wave shuffle tree, rsqrt(mean + eps)).
__global__ void ConcatNormKernel(const float* e, const float* h,
                                 const float* enorm, const float* hnorm,
                                 float* out, float eps) {
  __shared__ float partial[8];
  const std::uint32_t row = blockIdx.x >> 1;
  const bool second = (blockIdx.x & 1) != 0;
  const float* x = (second ? h : e) + std::size_t{row} * kHidden;
  const float* gamma = second ? hnorm : enorm;
  float* dst = out + std::size_t{row} * 2 * kHidden + (second ? kHidden : 0);
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < kHidden; i += blockDim.x) {
    const float v = x[i];
    ss = fmaf(v, v, ss);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    ss += __shfl_xor(ss, off, 32);
  }
  const int wave = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    partial[wave] = ss;
  }
  __syncthreads();
  float total = 0.0F;
  for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i) {
    total += partial[i];
  }
  const float inv = rsqrtf(total / static_cast<float>(kHidden) + eps);
  for (std::uint32_t i = threadIdx.x; i < kHidden; i += blockDim.x) {
    dst[i] = x[i] * inv * gamma[i];
  }
}

__global__ void PositionsKernel(std::int32_t* positions, std::int32_t base,
                                std::uint32_t n) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) {
    positions[t] = base + static_cast<std::int32_t>(t);
  }
}

constexpr std::uint32_t kArgmaxThreads = 256;

// Lowest index of the largest finite logit of each row (the rule of
// sampling::SamplerState::SampleGreedy: non-finite values are skipped, ties go
// to the lower index); -1 when no value is finite. One block per row.
__global__ void ArgmaxKernel(const float* logits, std::uint32_t vocab,
                             std::int32_t* index, float* value) {
  __shared__ float best_v[kArgmaxThreads];
  __shared__ std::int32_t best_i[kArgmaxThreads];
  const float* row = logits + std::size_t{blockIdx.x} * vocab;
  float bv = -INFINITY;
  std::int32_t bi = -1;
  for (std::uint32_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = row[i];
    if (isfinite(v) && (bi < 0 || v > bv)) {
      bv = v;
      bi = static_cast<std::int32_t>(i);
    }
  }
  best_v[threadIdx.x] = bv;
  best_i[threadIdx.x] = bi;
  __syncthreads();
  for (std::uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = best_v[threadIdx.x + s];
      const std::int32_t oi = best_i[threadIdx.x + s];
      const float mv = best_v[threadIdx.x];
      const std::int32_t mi = best_i[threadIdx.x];
      if (oi >= 0 && (mi < 0 || ov > mv || (ov == mv && oi < mi))) {
        best_v[threadIdx.x] = ov;
        best_i[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    index[blockIdx.x] = best_i[0];
    value[blockIdx.x] = best_v[0];
  }
}

}  // namespace

void ConcatNormRows(const float* e, const float* h, const float* enorm,
                    const float* hnorm, float* out, std::uint32_t n_rows,
                    float eps, hipStream_t stream) {
  if (n_rows == 0) {
    return;
  }
  ConcatNormKernel<<<n_rows * 2, 256, 0, stream>>>(e, h, enorm, hnorm, out,
                                                   eps);
}

void FillPositions(std::int32_t* positions, std::int32_t base, std::uint32_t n,
                   hipStream_t stream) {
  if (n == 0) {
    return;
  }
  PositionsKernel<<<(n + 63) / 64, 64, 0, stream>>>(positions, base, n);
}

void ArgmaxRows(const float* logits, std::uint32_t n_rows, std::uint32_t vocab,
                std::int32_t* index, float* value, hipStream_t stream) {
  if (n_rows == 0) {
    return;
  }
  ArgmaxKernel<<<n_rows, kArgmaxThreads, 0, stream>>>(logits, vocab, index,
                                                      value);
}

}  // namespace gufo::models::ling_3_flash::rocm::mtp
