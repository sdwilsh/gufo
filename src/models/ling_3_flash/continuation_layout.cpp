#include "src/models/ling_3_flash/continuation_layout.hpp"

#include <algorithm>
#include <stdexcept>

#include "src/models/ling_3_flash/layout.hpp"
#include "src/models/ling_3_flash/snapshot_format.hpp"

namespace gufo::models::ling_3_flash {

ContinuationLayout::ContinuationLayout(bool mtp, std::uint32_t context)
    : mtp_(mtp), context_(context) {
  if (context == 0)
    throw std::invalid_argument("invalid Ling continuation context");
  const auto add = [&](ContinuationPart part, std::size_t row,
                       cache::Rows rows_per_chunk, std::size_t state,
                       std::uint32_t layer = 0) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    components_.push_back({{{id},
                            kContinuationLayoutVersion,
                            row ? cache::ComponentKind::kAppendRows
                                : cache::ComponentKind::kPrivateState,
                            row,
                            row ? rows_per_chunk : 0,
                            state},
                           part,
                           layer});
  };
  constexpr std::size_t kLatentRowBytes =
      kMlaCacheWidth * sizeof(std::uint16_t);
  add(ContinuationPart::kTokens, sizeof(std::int32_t), kTokenRowsPerChunk, 0);
  for (std::uint32_t m = 0; m < kMlaLayers; ++m)
    add(ContinuationPart::kMla, kLatentRowBytes, kLatentRowsPerChunk, 0, m);
  if (mtp)
    add(ContinuationPart::kMtpRows, kLatentRowBytes, kLatentRowsPerChunk, 0);
  for (std::uint32_t k = 0; k < kKdaLayers; ++k)
    add(ContinuationPart::kKdaState, 0, 0, kKdaStateFloats * sizeof(float), k);
  add(ContinuationPart::kKdaConv, 0, 0,
      std::size_t{kKdaLayers} * kKdaConvStateFloats * sizeof(float));
  add(ContinuationPart::kLogits, 0, 0, std::size_t{kVocab} * sizeof(float));
  add(ContinuationPart::kMetadata, 0, 0, kContinuationMetadataBytes);
  if (mtp)
    add(ContinuationPart::kMtpHidden, 0, 0,
        std::size_t{kHidden} * sizeof(float));
}

std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t position) const {
  if (position > context_)
    throw std::invalid_argument("Ling continuation frontier exceeds context");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& c : components_)
    positions.push_back(
        {c.descriptor.id, c.part == ContinuationPart::kMtpRows
                              ? std::max<std::uint32_t>(position, 1) - 1
                              : position});
  return positions;
}

std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& c : components_)
    bytes += c.descriptor.state_bytes;
  return bytes;
}

std::size_t ContinuationLayout::RowBytes(std::uint32_t position) const {
  const auto positions = Positions(position);
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < components_.size(); ++i)
    bytes += components_[i].descriptor.row_bytes * positions[i].valid_rows;
  return bytes;
}

std::uint64_t ContinuationLayout::Hash() const noexcept {
  // FNV-1a over the descriptor fields and parts, in component order.
  std::uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&](std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
      hash ^= (value >> (8 * i)) & 0xff;
      hash *= 1099511628211ULL;
    }
  };
  for (const auto& c : components_) {
    mix(c.descriptor.id.value);
    mix(c.descriptor.layout_version);
    mix(static_cast<std::uint64_t>(c.descriptor.kind));
    mix(c.descriptor.row_bytes);
    mix(c.descriptor.rows_per_chunk);
    mix(c.descriptor.state_bytes);
    mix(static_cast<std::uint64_t>(c.part));
    mix(c.layer);
  }
  return hash;
}

std::vector<ContinuationBackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram, std::uint32_t concurrency) const {
  if (!concurrency || concurrency > 8)
    throw std::invalid_argument("invalid continuation backing concurrency");
  // Private frontiers, rows and tails share the same committed pages (as for
  // Flash-Next): fixed per-component classes strand capacity. Contiguous page
  // groups keep one payload owner and charge all slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}

std::vector<cache::Rows> PlannedPrefillPasses(cache::Rows first,
                                              cache::Rows end,
                                              std::size_t capacity) {
  if (first > end || capacity == 0)
    throw std::invalid_argument("invalid Ling prefill plan");
  std::vector<cache::Rows> passes;
  while (first < end) {
    first += PrefillChunkTokens(end - first, end - first, capacity);
    passes.push_back(first);
  }
  return passes;
}

}  // namespace gufo::models::ling_3_flash
