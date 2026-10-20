// Host random inputs for the Ling-3.0-Flash op tests that compare against
// stored digests (parent_digests.hpp). std::mt19937 is fixed by the standard,
// but std::normal_distribution and std::uniform_real_distribution are not.
// libstdc++ 16 redraws in std::generate_canonical when a draw rounds to 1.0
// (P0952), so the same seed gave different inputs under GCC 15 and GCC 16.
// This generator uses only integer operations, float arithmetic, and
// std::sqrt, std::log, and std::cos, so a seed gives the same inputs on every
// toolchain. The host test files are built without fast-math.
#ifndef GUFO_TESTS_MODELS_LING_3_FLASH_TEST_RNG_HPP_
#define GUFO_TESTS_MODELS_LING_3_FLASH_TEST_RNG_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace gufo::tests::ling {

// xorshift64 with the seed spread by the 2^64 / phi constant.
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) { Seed(seed); }
  void Seed(std::uint64_t seed) { s = seed * 0x9E3779B97F4A7C15ULL + 1; }
  std::uint64_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  // Uniform in [0, 1) from the top 24 bits, so every value is exact in float.
  float Uniform() {
    return static_cast<float>((Next() >> 40) * (1.0 / 16777216.0));
  }
  float Uniform(float lo, float hi) { return lo + (hi - lo) * Uniform(); }
  // Integer in [lo, hi] from the top 32 bits. The modulo bias is below 2^-24.
  int Int(int lo, int hi) {
    const auto range = static_cast<std::uint64_t>(hi - lo) + 1;
    return lo + static_cast<int>((Next() >> 32) % range);
  }
  // Box-Muller, one value per call.
  float Normal() {
    const float u1 = std::max(Uniform(), 1e-7F), u2 = Uniform();
    return std::sqrt(-2.0F * std::log(u1)) * std::cos(6.2831853F * u2);
  }
  float Normal(float mean, float sigma) { return mean + sigma * Normal(); }
};

// True when GUFO_PRINT_GOLDENS=1. The digest tests then print the
// parent_digests.hpp initializer for each case and do not compare.
inline bool PrintGoldens() {
  const char* v = std::getenv("GUFO_PRINT_GOLDENS");
  return v != nullptr && std::strcmp(v, "1") == 0;
}

}  // namespace gufo::tests::ling

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_TEST_RNG_HPP_
