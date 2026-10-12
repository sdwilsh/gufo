// Ling-3.0-Flash non-absorbed MLA prefill microbenchmark. It times one layer's
// PrefillAttentionNonAbsorbed for a chunk of n tokens at a given depth, with
// the persistent expanded-K/V store laid out as the engine does (rows rounded
// up to whole 2048-row blocks for the deepest case, expanded once up front).
// It prints one JSON line per case: device ms per call (min and median of the
// timed calls), the per-launch breakdown (expand, flash, and merge, from the
// event-bracketed profiling mode), and an FNV-1a digest of the F32 output. The
// digest lets two builds of the kernel source be compared bit for bit:
//
//   tools/bench/build.sh tools/models/ling_3_flash/mla_prefill_bench.hip
//   /tmp/mla_prefill_bench [--depths 0,16384,65536] [--ns 9,16,64,128,512]
//                          [--reps 10] [--store-rows auto|0|N]
//
// The kernel source is included directly (GUFO_LING_MLA_SOURCE overrides the
// path, e.g. a saved copy of the parent); compile with the production flags
// (-O3 -ffast-math -fno-finite-math-only) when comparing digests.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef GUFO_LING_MLA_SOURCE
#define GUFO_LING_MLA_SOURCE "src/models/ling_3_flash/kernels/rocm/mla.hip.cpp"
#endif
#include GUFO_LING_MLA_SOURCE

namespace {

namespace mla = gufo::models::ling_3_flash::rocm::mla;
namespace lc = gufo::models::ling_3_flash;

void Ok(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

__device__ float Hash01(std::uint64_t i, std::uint32_t seed) {
  std::uint64_t x = i * 0x9E3779B97F4A7C15ULL + seed * 0xD1B54A32D192ED03ULL;
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 29;
  return static_cast<float>(x >> 40) * (1.0F / 16777216.0F);
}

// Roughly normal values (sum of four uniforms), sigma `s`.
__device__ float Noise(std::uint64_t i, std::uint32_t seed, float s) {
  const float u = Hash01(4 * i, seed) + Hash01(4 * i + 1, seed) +
                  Hash01(4 * i + 2, seed) + Hash01(4 * i + 3, seed);
  return (u - 2.0F) * 1.7320508F * s;
}

__global__ void FillHalf(__half* p, std::size_t n, std::uint32_t seed, float s,
                         std::uint32_t width, std::uint32_t split, float s_hi) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < n) {
    const bool hi = width != 0 && (i % width) >= split;
    p[i] = __float2half_rn(Noise(i, seed, hi ? s_hi : s));
  }
}

__global__ void FillFloat(float* p, std::size_t n, std::uint32_t seed,
                          float s) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = Noise(i, seed, s);
}

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + 255) / 256);
}

std::uint64_t Fnv(const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

std::vector<std::uint32_t> ParseList(const char* s) {
  std::vector<std::uint32_t> out;
  for (const char* p = s; *p != '\0';) {
    char* end = nullptr;
    out.push_back(static_cast<std::uint32_t>(std::strtoul(p, &end, 10)));
    p = *end == ',' ? end + 1 : end;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::uint32_t> depths{0, 16384, 65536};
  std::vector<std::uint32_t> ns{9, 16, 64, 128, 512};
  int reps = 10;
  long store_arg = -1;  // auto
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--depths") == 0)
      depths = ParseList(argv[i + 1]);
    else if (std::strcmp(argv[i], "--ns") == 0)
      ns = ParseList(argv[i + 1]);
    else if (std::strcmp(argv[i], "--reps") == 0)
      reps = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--store-rows") == 0)
      store_arg = std::strcmp(argv[i + 1], "auto") == 0
                      ? -1
                      : std::strtol(argv[i + 1], nullptr, 10);
  }
  const std::uint32_t max_total =
      *std::max_element(depths.begin(), depths.end()) +
      *std::max_element(ns.begin(), ns.end());
  const std::uint32_t store_rows = store_arg < 0
                                       ? (max_total + 1 + 2047) / 2048 * 2048
                                       : static_cast<std::uint32_t>(store_arg);
  const std::uint32_t max_n = *std::max_element(ns.begin(), ns.end());
  constexpr std::uint32_t kW = lc::kMlaCacheWidth;
  constexpr std::uint32_t kH = lc::kHeads;
  constexpr std::uint32_t kQ = lc::kMlaNope + lc::kMlaRope;
  constexpr std::uint32_t kV = lc::kMlaValueDim;

  hipDeviceProp_t prop{};
  Ok(hipGetDeviceProperties(&prop, 0), "props");
  const std::size_t cache_rows = std::max(max_total, store_rows) + 16;
  __half* cache = nullptr;
  __half* w16 = nullptr;
  __half* store = nullptr;
  float* q = nullptr;
  float* gate = nullptr;
  float* out = nullptr;
  void* scratch = nullptr;
  Ok(hipMalloc(&cache, cache_rows * kW * sizeof(__half)), "cache");
  Ok(hipMalloc(&w16, mla::ExpansionWeightBytes()), "w16");
  if (store_rows > 0)
    Ok(hipMalloc(&store, mla::ExpandedLayerBytes(store_rows)), "store");
  Ok(hipMalloc(&q, std::size_t{max_n} * kH * kQ * sizeof(float)), "q");
  Ok(hipMalloc(&gate, std::size_t{max_n} * kH * sizeof(float)), "gate");
  Ok(hipMalloc(&out, std::size_t{max_n} * kH * kV * sizeof(float)), "out");
  Ok(hipMalloc(&scratch, mla::NonAbsorbedScratchBytes(max_n)), "scratch");
  {
    const std::size_t nc = cache_rows * kW;
    FillHalf<<<Blocks(nc), 256>>>(cache, nc, 11, 1.0F, kW, lc::kMlaLatent,
                                  1.5F);
    const std::size_t nw = mla::ExpansionWeightBytes() / sizeof(__half);
    FillHalf<<<Blocks(nw), 256>>>(w16, nw, 12, 0.05F, 0, 0, 0.0F);
    const std::size_t nq = std::size_t{max_n} * kH * kQ;
    FillFloat<<<Blocks(nq), 256>>>(q, nq, 13, 1.7F);
    FillFloat<<<Blocks(std::size_t{max_n} * kH), 256>>>(
        gate, std::size_t{max_n} * kH, 14, 1.5F);
    if (store_rows > 0) {
      Ok(hipMemset(store, 0, mla::ExpandedLayerBytes(store_rows)), "clear");
      mla::ExpandRows(cache, w16, store, store_rows, 0,
                      std::min<std::uint32_t>(max_total, store_rows), nullptr);
    }
    Ok(hipDeviceSynchronize(), "init");
  }
  std::printf("{\"device\":\"%s\",\"cus\":%d,\"store_rows\":%u}\n", prop.name,
              prop.multiProcessorCount, store_rows);
  hipEvent_t e0, e1;
  Ok(hipEventCreate(&e0), "event");
  Ok(hipEventCreate(&e1), "event");
  std::vector<float> host;
  for (const std::uint32_t depth : depths) {
    for (const std::uint32_t n : ns) {
      const auto call = [&](mla::PrefillTimes* t) {
        mla::PrefillAttentionNonAbsorbed(q, cache, w16, store, store_rows, gate,
                                         depth, n, out, scratch, nullptr, t);
      };
      call(nullptr);
      call(nullptr);
      Ok(hipDeviceSynchronize(), "warm");
      host.resize(std::size_t{n} * kH * kV);
      Ok(hipMemcpy(host.data(), out, host.size() * sizeof(float),
                   hipMemcpyDeviceToHost),
         "download");
      const std::uint64_t digest =
          Fnv(host.data(), host.size() * sizeof(float));
      std::vector<float> ms;
      for (int r = 0; r < reps; ++r) {
        Ok(hipEventRecord(e0, nullptr), "record");
        call(nullptr);
        Ok(hipEventRecord(e1, nullptr), "record");
        Ok(hipEventSynchronize(e1), "sync");
        float m = 0.0F;
        Ok(hipEventElapsedTime(&m, e0, e1), "elapsed");
        ms.push_back(m);
      }
      std::sort(ms.begin(), ms.end());
      mla::PrefillTimes parts;
      const int prof_reps = 3;
      for (int r = 0; r < prof_reps; ++r)
        call(&parts);
      std::printf(
          "{\"depth\":%u,\"n\":%u,\"ms_min\":%.4f,\"ms_median\":%.4f,"
          "\"expand_ms\":%.4f,\"attn_ms\":%.4f,\"merge_ms\":%.4f,"
          "\"digest\":\"%016llx\"}\n",
          depth, n, ms.front(), ms[ms.size() / 2], parts.expand_ms / prof_reps,
          parts.attn_ms / prof_reps, parts.merge_ms / prof_reps,
          static_cast<unsigned long long>(digest));
      std::fflush(stdout);
    }
  }
  return 0;
}
