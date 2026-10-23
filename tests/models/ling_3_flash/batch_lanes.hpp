// Small helpers of the batched-decode Ling-3.0-Flash model binaries
// (batch_test, batch_mtp_test, batch_segments_test, decode_digest).
#ifndef GUFO_TESTS_MODELS_LING_3_FLASH_BATCH_LANES_HPP_
#define GUFO_TESTS_MODELS_LING_3_FLASH_BATCH_LANES_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "src/models/ling_3_flash/engine.hpp"

namespace gufo::tests::ling {

namespace l3 = gufo::models::ling_3_flash;

using Bytes = std::vector<std::uint8_t>;

/// The whole state of a session: its snapshot payload.
inline bool State(const l3::Session& s, Bytes* out, std::string* error) {
  auto snap = s.SaveSnapshot(error);
  if (!snap) {
    return false;
  }
  out->assign(snap->bytes().begin(), snap->bytes().end());
  return true;
}

inline l3::Session::BatchLane Item(l3::Session* session, std::int32_t token) {
  l3::Session::BatchLane lane;
  lane.session = session;
  lane.token = token;
  return lane;
}

inline std::int32_t Tok(std::uint32_t i) {
  return static_cast<std::int32_t>((i * 2654435761U + 12345U) % 150000U);
}

}  // namespace gufo::tests::ling

#endif  // GUFO_TESTS_MODELS_LING_3_FLASH_BATCH_LANES_HPP_
