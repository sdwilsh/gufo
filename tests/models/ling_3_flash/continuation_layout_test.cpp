// CPU checks of the Ling-3.0-Flash component-cache layout: descriptors,
// frontiers, byte totals, the backing plan, and the prefill pass plan. The test
// also checks how the cache package handles the layout's two chunk sizes (16384
// token rows and 512 latent rows) through capture, chunk sharing, and the disk
// manifest.
#include "src/models/ling_3_flash/continuation_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "src/cache/checkpoint.hpp"
#include "src/cache/disk.hpp"
#include "src/models/ling_3_flash/layout.hpp"
#include "src/models/ling_3_flash/snapshot_format.hpp"

namespace l3 = gufo::models::ling_3_flash;
namespace cache = gufo::cache;

#define EXPECT(cond)                                                         \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr std::size_t kState = 32ULL * 128 * 128 * 4;
constexpr std::size_t kConv = 35ULL * 3 * 3 * 4096 * 4;
constexpr std::size_t kLogits = 157184ULL * 4;

void Geometry() {
  const l3::ContinuationLayout ar(false, 32768), mtp(true, 32768);
  EXPECT(ar.Components().size() == 1 + 7 + 35 + 3);
  EXPECT(mtp.Components().size() == ar.Components().size() + 2);
  // Dense IDs from 1, in order; kinds match their bytes.
  for (const auto* layout : {&ar, &mtp}) {
    std::uint32_t id = 1;
    for (const auto& c : layout->Components()) {
      EXPECT(c.descriptor.id.value == id++);
      EXPECT(c.descriptor.layout_version == l3::kContinuationLayoutVersion);
      if (c.descriptor.kind == cache::ComponentKind::kAppendRows) {
        EXPECT(c.descriptor.row_bytes && c.descriptor.rows_per_chunk &&
               !c.descriptor.state_bytes);
      } else {
        EXPECT(!c.descriptor.row_bytes && !c.descriptor.rows_per_chunk &&
               c.descriptor.state_bytes);
      }
    }
  }
  EXPECT(ar.PrivateBytes() ==
         35 * kState + kConv + kLogits + l3::kContinuationMetadataBytes);
  EXPECT(ar.PrivateBytes() == 79190272);
  EXPECT(mtp.PrivateBytes() == ar.PrivateBytes() + 2560 * 4);
  // Rows per token: 4 (token) + 7 x 1152, plus 1152 for MTP rows.
  EXPECT(ar.RowBytes(1000) == 1000ULL * 8068);
  EXPECT(mtp.RowBytes(1000) == 1000ULL * 8068 + 999ULL * 1152);
  EXPECT(mtp.RowBytes(0) == 0);
  // Latent chunks are exactly nine 64 KiB pages; a token chunk is one.
  EXPECT(l3::kLatentRowsPerChunk * 1152 == 9 * 65536);
  EXPECT(l3::kTokenRowsPerChunk * 4 == 65536);
  EXPECT(ar.Hash() != mtp.Hash());
  EXPECT(ar.Hash() == l3::ContinuationLayout(false, 1024).Hash());
}

void Frontiers() {
  const l3::ContinuationLayout mtp(true, 4096);
  for (std::uint32_t p : {0U, 1U, 2U, 513U, 4096U}) {
    const auto positions = mtp.Positions(p);
    EXPECT(positions.size() == mtp.Components().size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
      const auto& c = mtp.Components()[i];
      EXPECT(positions[i].id == c.descriptor.id);
      const auto want =
          c.part == l3::ContinuationPart::kMtpRows ? (p ? p - 1 : 0) : p;
      EXPECT(positions[i].valid_rows == want);
    }
  }
  bool threw = false;
  try {
    (void)mtp.Positions(4097);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

void Backing() {
  const l3::ContinuationLayout ar(false, 196608);
  EXPECT(ar.BackingPlan(1 << 20, 1).empty());
  for (std::size_t gib : {2ULL, 8ULL, 15ULL}) {
    const auto plan = ar.BackingPlan(gib << 30, 1);
    EXPECT(plan.size() == 1);
    EXPECT(plan[0].block_bytes == 65536);
    // Never more than the budget, metadata included.
    EXPECT(plan[0].blocks * plan[0].block_bytes + plan[0].metadata_bytes <=
           (gib << 30));
    // Holds at least one full private checkpoint.
    EXPECT(plan[0].blocks * plan[0].block_bytes > ar.PrivateBytes());
  }
  bool threw = false;
  try {
    (void)ar.BackingPlan(8ULL << 30, 9);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

// The serving loop: one Prefill call per step, sized by PrefillChunkTokens.
std::vector<cache::Rows> ServedPasses(cache::Rows first, cache::Rows end,
                                      std::size_t capacity) {
  std::vector<cache::Rows> passes;
  while (first < end) {
    const auto n = l3::PrefillChunkTokens(end - first, end - first, capacity);
    EXPECT(n > l3::kSmallBatchMaxTokens || n == end - first);
    first += n;
    passes.push_back(first);
  }
  return passes;
}

void PrefillPlan() {
  EXPECT((l3::PlannedPrefillPasses(0, 1030, 512) ==
          std::vector<cache::Rows>{512, 1021, 1030}));
  EXPECT(l3::PlannedPrefillPasses(7, 7, 512).empty());
  EXPECT((l3::PlannedPrefillPasses(100, 105, 512) ==
          std::vector<cache::Rows>{105}));
  for (cache::Rows first : {0ULL, 3ULL, 2047ULL, 4090ULL})
    for (cache::Rows length = 1; length < 3000; length += 37)
      for (std::size_t capacity : {9ULL, 64ULL, 512ULL, 2048ULL})
        EXPECT(l3::PlannedPrefillPasses(first, first + length, capacity) ==
               ServedPasses(first, first + length, capacity));
}

cache::Payload Store(cache::ResourceLedger& ledger,
                     const cache::PayloadRequest& request) {
  auto reservation =
      ledger.Reserve(cache::ResourceCategory::kBackingFree, request.bytes);
  auto pool = reservation.Convert();
  auto assigned = pool.ReserveBacking(request.category);
  return cache::Payload::Committed(assigned.Convert(),
                                   std::make_shared<int>(0));
}

// Two checkpoints of one execution share full chunks of each chunk size, and
// a published manifest with both sizes encodes and decodes.
void PackageChunks() {
  const l3::ContinuationLayout layout(true, 65536);
  const std::vector<cache::ComponentDescriptor> descriptors = [&] {
    std::vector<cache::ComponentDescriptor> d;
    for (const auto& c : layout.Components())
      d.push_back(c.descriptor);
    return d;
  }();
  cache::ResourceLedger ledger(cache::ResourceLimits{
      std::size_t{8} << 30, std::size_t{8} << 30, 0, std::size_t{8} << 30});
  auto history = cache::ExecutionHistory::Cold(ledger, descriptors, {1, 2, 3});
  std::vector<cache::Token> tokens(40000, 5);
  const auto capture = [&](std::uint32_t p) {
    return history.Capture(
        {std::span<const cache::Token>(tokens).first(p),
         cache::InputIdentity(p), layout.Positions(p),
         cache::CheckpointPurpose::kPrompt, 1},
        [&](const cache::PayloadRequest& r) { return Store(ledger, r); });
  };
  const auto a = capture(20000);
  const auto b = capture(40000);
  EXPECT(a && b && a->IsValid() && b->IsValid());
  for (std::size_t i = 0; i < layout.Components().size(); ++i) {
    const auto& ca = a->Components()[i];
    const auto& cb = b->Components()[i];
    const auto& d = layout.Components()[i].descriptor;
    if (d.kind != cache::ComponentKind::kAppendRows) {
      EXPECT(ca.private_state && cb.private_state && ca.chunks.empty());
      continue;
    }
    EXPECT(ca.chunks.size() == ca.position.valid_rows / d.rows_per_chunk);
    EXPECT(bool(ca.tail) == bool(ca.position.valid_rows % d.rows_per_chunk));
    // Every full chunk of the shorter checkpoint is shared.
    for (std::size_t n = 0; n < ca.chunks.size(); ++n)
      EXPECT(ca.chunks[n].SameChunk(cb.chunks[n]));
    if (layout.Components()[i].part == l3::ContinuationPart::kTokens)
      EXPECT(ca.chunks.size() == 1 && cb.chunks.size() == 2);
    else
      EXPECT(cb.chunks.size() == cb.position.valid_rows / 512);
  }
  cache::DiskManifest manifest;
  manifest.checkpoint = b->Id();
  manifest.lineage = b->Lineage();
  manifest.input = {};
  manifest.tokens.assign(b->Tokens().begin(), b->Tokens().end());
  std::uint32_t file = 0;
  const auto next = [&](std::size_t bytes) {
    cache::DiskPayload p;
    ++file;
    for (unsigned i = 0; i < 4; ++i)
      p.file[i] = static_cast<std::uint8_t>(file >> (8 * i));
    p.bytes = bytes;
    return p;
  };
  for (const auto& c : b->Components()) {
    cache::DiskComponent dc{c.descriptor, c.position, {}, {}, {}};
    for (const auto& chunk : c.chunks)
      dc.chunks.push_back(next(chunk.Storage().Bytes()));
    if (c.tail)
      dc.tail = next(c.tail->Bytes());
    if (c.private_state)
      dc.private_state = next(c.private_state->Bytes());
    manifest.components.push_back(std::move(dc));
  }
  const auto decoded = cache::DecodeManifest(cache::EncodeManifest(manifest));
  EXPECT(decoded.components.size() == manifest.components.size());
  EXPECT(decoded.components[0].descriptor.rows_per_chunk ==
         l3::kTokenRowsPerChunk);
  EXPECT(decoded.components[1].descriptor.rows_per_chunk ==
         l3::kLatentRowsPerChunk);
}

}  // namespace

int main() {
  Geometry();
  Frontiers();
  Backing();
  PrefillPlan();
  PackageChunks();
  std::puts("ling continuation layout: ok");
  return 0;
}
