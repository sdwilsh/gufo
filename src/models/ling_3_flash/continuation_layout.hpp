#ifndef GUFO_MODELS_LING_3_FLASH_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_LING_3_FLASH_CONTINUATION_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/cache/types.hpp"

namespace gufo::models::ling_3_flash {

/// State inventory of one Ling-3.0-flash session for the component cache
/// (src/cache). IDs refer to this representation, never to an allocation.
/// Not components: the expanded K/V store (derived from the latent rows),
/// HIP graphs and executor scratch, the MTP length/copy/cohort controllers
/// (reset per request) and the prompt-lookup index (rebuilt from the tokens).
enum class ContinuationPart : std::uint8_t {
  kTokens,     ///< host Session::tokens_, append rows (copied, not borrowed)
  kMla,        ///< F16 latent rows of one trunk MLA layer, append rows
  kMtpRows,    ///< F16 latent rows of the MTP block, frontier P - 1
  kKdaState,   ///< F32 recurrent state of one KDA layer, private
  kKdaConv,    ///< F32 conv windows of all 35 KDA layers, private
  kLogits,     ///< host logits of the last token, private
  kMetadata,   ///< fixed host image (ContinuationMetadata), private
  kMtpHidden,  ///< F32 trunk-hidden carry of position P - 1, private
};

struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};  ///< MLA index 0..6 or KDA index 0..34
};

/// Bytes of the private metadata component; ContinuationMetadata must fit.
inline constexpr std::size_t kContinuationMetadataBytes = 256;
/// One 64 KiB backing page of tokens per chunk.
inline constexpr cache::Rows kTokenRowsPerChunk = 16384;
/// 512 latent rows of 1152 bytes are exactly nine 64 KiB backing pages.
inline constexpr cache::Rows kLatentRowsPerChunk = 512;
/// Bumped when the component list, a descriptor or the metadata changes.
inline constexpr std::uint32_t kContinuationLayoutVersion = 1;

struct ContinuationBackingClass {
  std::size_t block_bytes{}, blocks{}, metadata_bytes{};
};

class ContinuationLayout {
public:
  ContinuationLayout(bool mtp, std::uint32_t context);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] bool mtp() const noexcept { return mtp_; }
  [[nodiscard]] std::uint32_t context() const noexcept { return context_; }
  /// Exact component frontiers of a coherent state after `position` tokens:
  /// private state and tokens/latent rows at P, the MTP rows at max(P, 1) - 1
  /// (row P - 1 needs token P and is rewritten by the next forward).
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      std::uint32_t position) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  /// Append-row bytes of a checkpoint at `position`.
  [[nodiscard]] std::size_t RowBytes(std::uint32_t position) const;
  /// Stable hash of the descriptor list (metadata and identity check).
  [[nodiscard]] std::uint64_t Hash() const noexcept;
  /// Shared 64 KiB pages for private state, rows and tails (the Flash-Next
  /// allocator): one class, sized from the RAM budget.
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes, std::uint32_t concurrency) const;

private:
  bool mtp_;
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
};

/// Pass ends serving actually runs for prompt tokens [first, end): one Prefill
/// call per step of at most `capacity` tokens, each sized by
/// PrefillChunkTokens (snapshot_format.hpp), which never leaves a 1..8-token
/// tail when it can rebalance. For 1030 tokens at capacity 512: 512, 1021,
/// 1030 (passes of 512, 509 and 9). Used only for optional grid snapping.
[[nodiscard]] std::vector<cache::Rows> PlannedPrefillPasses(
    cache::Rows first, cache::Rows end, std::size_t capacity);

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_CONTINUATION_LAYOUT_HPP_
