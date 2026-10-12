// HostCopyPool: the split copy equals memcpy for every size, alignment and
// thread count, repeated calls and concurrent callers are safe, and the pool
// joins its workers on destruction.
#include "src/models/ling_3_flash/kernels/rocm/host_copy_pool.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {
using gufo::models::ling_3_flash::rocm::HostCopyPool;

int failures = 0;
void Expect(bool ok, const char* what, std::size_t a = 0, std::size_t b = 0) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s (%zu, %zu)\n", what, a, b);
    ++failures;
  }
}

std::vector<std::uint8_t> Pattern(std::size_t n, std::uint32_t seed) {
  std::vector<std::uint8_t> v(n);
  std::uint32_t x = seed * 2654435761u + 1;
  for (auto& b : v) {
    x = x * 1664525u + 1013904223u;
    b = static_cast<std::uint8_t>(x >> 24);
  }
  return v;
}

void CheckCopy(HostCopyPool& pool, std::size_t n, std::size_t src_off,
               std::size_t dst_off, std::uint32_t seed) {
  const auto src = Pattern(n + src_off + 8, seed);
  std::vector<std::uint8_t> dst(n + dst_off + 8, 0xA5);
  pool.Copy(dst.data() + dst_off, src.data() + src_off, n);
  Expect(std::memcmp(dst.data() + dst_off, src.data() + src_off, n) == 0,
         "copy equals source", n, dst_off);
  for (std::size_t i = 0; i < dst_off; ++i) {
    Expect(dst[i] == 0xA5, "byte before destination untouched", i);
  }
  for (std::size_t i = dst_off + n; i < dst.size(); ++i) {
    Expect(dst[i] == 0xA5, "byte after destination untouched", i);
  }
}
}  // namespace

int main() {
  constexpr std::size_t kMin = HostCopyPool::kMinParallelBytes;
  const std::size_t sizes[] = {
      0,        1,           4095,         kMin - 1, kMin,
      kMin + 1, kMin + 4097, 5 * kMin + 3, 8 * kMin, 13 * kMin + 12345};
  for (const unsigned threads : {1u, 2u, 3u, 4u, 8u}) {
    HostCopyPool pool(threads);
    std::uint32_t seed = 1;
    for (const std::size_t n : sizes) {
      CheckCopy(pool, n, 0, 0, seed++);
      CheckCopy(pool, n, 3, 5, seed++);
    }
    for (int i = 0; i < 50; ++i) {
      CheckCopy(pool, 3 * kMin + static_cast<std::size_t>(i), 1, 2, seed++);
    }
  }
  {
    // Concurrent callers are serialized, never interleaved.
    HostCopyPool pool(3);
    std::vector<std::thread> callers;
    for (unsigned t = 0; t < 4; ++t) {
      callers.emplace_back([&pool, t] {
        for (int i = 0; i < 20; ++i) {
          CheckCopy(pool, 2 * kMin + 17 * t, 0, 0, 100 + t * 31 + i);
        }
      });
    }
    for (auto& c : callers) {
      c.join();
    }
  }
  {
    // Construction and destruction without any copy.
    for (int i = 0; i < 20; ++i) {
      HostCopyPool pool(4);
    }
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::puts("host_copy_pool_test: ok");
  return 0;
}
