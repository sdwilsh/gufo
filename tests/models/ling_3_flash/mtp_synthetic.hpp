#ifndef GUFO_TESTS_MODELS_LING_3_FLASH_MTP_SYNTHETIC_HPP_
#define GUFO_TESTS_MODELS_LING_3_FLASH_MTP_SYNTHETIC_HPP_

// Deterministic synthetic inputs of the MTP draft-block checks, shared by the
// CPU reference test and the GPU op test: a draft step needs only a (token,
// trunk hidden) pair, so both sides can run block 42 alone on the same inputs
// without the 42-block trunk.

#include <cmath>
#include <cstdint>
#include <vector>

namespace gufo::tests::ling {

inline constexpr std::uint32_t kMtpSyntheticSteps = 4;

/// Token fed at draft step `step` (a mix of common and rare ids).
[[nodiscard]] inline std::int32_t MtpSyntheticToken(std::uint32_t step) {
  constexpr std::int32_t tokens[] = {1021, 35, 90210, 147531,
                                     4417, 12, 70000, 3};
  return tokens[step % 8];
}

/// Output-normed-trunk-like hidden row: roughly unit-variance normals (Box
/// Muller on an LCG) with a few large channels, as real normed hiddens have.
[[nodiscard]] inline std::vector<float> MtpSyntheticHidden(
    std::uint32_t step, std::uint32_t width) {
  std::uint64_t state = 0x9E3779B97F4A7C15ULL * (step + 1);
  const auto next = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((state >> 11) + 1) / 9007199254740994.0;
  };
  std::vector<float> h(width);
  for (std::uint32_t i = 0; i < width; ++i) {
    const double r = std::sqrt(-2.0 * std::log(next()));
    const double a = 6.283185307179586 * next();
    h[i] = static_cast<float>(r * std::cos(a));
  }
  for (std::uint32_t i = 7; i < width; i += 331) {
    h[i] *= 6.0F;
  }
  return h;
}

}  // namespace gufo::tests::ling

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_MTP_SYNTHETIC_HPP_
