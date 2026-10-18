// CPU checks of the Ling session snapshot payload layout: the sizes the server
// budgets RAM against, section order and the header size.
#include "src/models/ling_3_flash/snapshot_format.hpp"

#include <cstdio>
#include <cstdlib>

namespace l3 = gufo::models::ling_3_flash;

#define EXPECT(cond)                                                         \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

int main() {
  // Fixed part: 35 KDA layers of 2 MiB state + 144 KiB conv, logits, header.
  constexpr std::size_t kState = 35ULL * 32 * 128 * 128 * 4;
  constexpr std::size_t kConv = 35ULL * 3 * 3 * 4096 * 4;
  constexpr std::size_t kLogits = 157184ULL * 4;
  EXPECT(l3::SnapshotPayloadBytes(0) == 64 + kLogits + kState + kConv);
  // Per token: 4 bytes of token id + 7 MLA layers x 576 F16.
  EXPECT(l3::SnapshotPayloadBytes(1) - l3::SnapshotPayloadBytes(0) ==
         4 + 7 * 576 * 2);
  EXPECT(l3::SnapshotPayloadBytes(60000) ==
         l3::SnapshotPayloadBytes(0) + 60000ULL * 8068);
  const auto at = l3::SnapshotLayout(1000);
  EXPECT(at.tokens == l3::kSnapshotHeaderBytes);
  EXPECT(at.logits == at.tokens + 4000);
  EXPECT(at.kda_state == at.logits + kLogits);
  EXPECT(at.kda_conv == at.kda_state + kState);
  EXPECT(at.mla == at.kda_conv + kConv);
  EXPECT(at.total == at.mla + 7ULL * 1000 * 576 * 2);
  EXPECT(l3::kSnapshotPayloadVersion >= 1);
  EXPECT(l3::kNumericsId.size() == 64);
  // Prefill chunking keeps checkpoints out of the small-batch family.
  using l3::PrefillChunkTokens;
  EXPECT(PrefillChunkTokens(517, 1000, 512) == 508);  // 512 would leave 5
  EXPECT(PrefillChunkTokens(520, 1000, 512) == 511);  // tail 8 -> 9
  EXPECT(PrefillChunkTokens(512, 1000, 512) == 512);  // no tail
  EXPECT(PrefillChunkTokens(300, 1000, 512) == 300);  // whole remainder
  EXPECT(PrefillChunkTokens(260, 256, 512) == 251);   // grant leaves 4
  EXPECT(PrefillChunkTokens(5, 512, 512) == 5);       // below 9: small family
  EXPECT(PrefillChunkTokens(20, 12, 512) == 11);      // tail 8 -> 9
  EXPECT(PrefillChunkTokens(12, 9, 512) == 9);        // cannot rebalance
  EXPECT(PrefillChunkTokens(30, 12, 512) == 12);      // tail 18, untouched
  for (std::size_t left = 1; left < 2200; ++left) {
    for (const std::size_t grant : {16, 100, 512, 1 << 20}) {
      std::size_t at = 0;
      std::size_t guard = 0;
      while (at < left && guard++ < 4000) {
        const std::size_t n = PrefillChunkTokens(left - at, grant, 512);
        EXPECT(n >= 1 && n <= 512 && n <= grant && n <= left - at);
        at += n;
        // Never a 1..8 tail unless that is all that was left to begin with
        // (or the grant itself was below 9).
        if (left - at != 0 && left - at <= 8 && n > 9) {
          std::fprintf(stderr, "left=%zu grant=%zu n=%zu tail=%zu\n", left,
                       grant, n, left - at);
          EXPECT(false);
        }
      }
      EXPECT(at == left);
    }
  }
  std::puts("ling snapshot format ok");
  return 0;
}
