// FNV-1a over host bytes, for the Ling-3.0-Flash op tests that compare
// outputs bit for bit across launch shapes and against stored digests.
#ifndef GUFO_TESTS_MODELS_LING_3_FLASH_TEST_FNV_HPP_
#define GUFO_TESTS_MODELS_LING_3_FLASH_TEST_FNV_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::tests::ling {

inline std::uint64_t Fnv(const void* p, std::size_t bytes) {
  std::uint64_t h = 1469598103934665603ULL;
  const auto* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < bytes; ++i)
    h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}

}  // namespace gufo::tests::ling

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_TEST_FNV_HPP_
