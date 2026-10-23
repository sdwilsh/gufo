#ifndef GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_ACT_QUANT_HIP_HPP_
#define GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_ACT_QUANT_HIP_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

// Two-plane int8 activation quantisation shared by every producer of GEMV
// activations: the standalone QuantizeActKernel and the fused norm, SwiGLU,
// and output-gate kernels. The arithmetic keeps the order and the
// reciprocal-multiply form that the standalone kernel compiled to under
// -ffast-math, which rewrites x / y as x * rcp(y) and reassociates. Every
// producer therefore yields the same planes bit for bit. The empty asm pins
// the one product the compiler would otherwise reassociate.
//   m   = max |x| over the 32-value block
//   q1  = rn(x * (127 * rcp(m))),  s = m / 127
//   q2  = rn(((x - q1 * s) * 127) * rcp(s))
namespace gufo::models::ling_3_flash::rocm {

__device__ __forceinline__ void QuantizeValue(float v, float m,
                                              std::int8_t* __restrict__ q,
                                              std::int8_t* __restrict__ q2) {
  float inv = m > 0.0F ? __builtin_amdgcn_rcpf(m) * 127.0F : 0.0F;
  __asm__ volatile("" : "+v"(inv));
  const int hi = __float2int_rn(v * inv);
  const float s = m * (1.0F / 127.0F);
  float t = fmaf(-s, static_cast<float>(hi), v);
  t *= 127.0F;
  __asm__ volatile("" : "+v"(t));
  t *= __builtin_amdgcn_rcpf(s);
  *q = static_cast<std::int8_t>(hi);
  *q2 = static_cast<std::int8_t>(m > 0.0F ? __float2int_rn(t) : 0);
}

/// Scale of a block: m / 127.
__device__ __forceinline__ float QuantScale(float m) {
  return m * (1.0F / 127.0F);
}

}  // namespace gufo::models::ling_3_flash::rocm

#endif  // GUFO_MODELS_LING_3_FLASH_KERNELS_ROCM_ACT_QUANT_HIP_HPP_
