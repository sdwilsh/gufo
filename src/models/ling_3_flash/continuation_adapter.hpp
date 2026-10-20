#ifndef GUFO_MODELS_LING_3_FLASH_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_LING_3_FLASH_CONTINUATION_ADAPTER_HPP_

#include <functional>
#include <memory>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/models/ling_3_flash/continuation_layout.hpp"
#include "src/models/ling_3_flash/engine.hpp"

namespace gufo::hip {
class TransferStream;
}

namespace gufo::models::ling_3_flash {

/// Component-cache adapter (src/cache/adapter.hpp) for Ling-3.0-flash, AR or
/// MTP. Each slot owns one public Session; serving binds its runner state to
/// it and executes through the ordinary Session API. Contract details:
/// artifacts/ling-3-flash/CACHE-ADAPTER-DESIGN.md (component inventory,
/// guard sites, numeric families, device loss).
///
/// The caller supplies the artifact/tokenizer/template identity; the adapter
/// appends its layout, mode, context, prefill capacity, verify width and
/// resident trunk range. Restores keep the persistent expanded K/V of latent
/// rows that the restore leaves bit-identical (compare-then-copy), as the
/// legacy snapshot restore did.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(std::shared_ptr<Model>, std::uint32_t context,
                      std::uint32_t max_drafts, cache::Identity);
  ~ContinuationAdapter() override;
  /// The slot's session, for execution; throws when the slot is not ready.
  [[nodiscard]] Session& GetSession(cache::Slot&);
  [[nodiscard]] const ContinuationLayout& layout() const noexcept {
    return layout_;
  }
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes, std::uint32_t concurrency) const {
    return layout_.BackingPlan(ram_bytes, concurrency);
  }
  /// Latent rows whose persistent expansion the last successful restore of
  /// this slot kept, in the MLA layer that kept the most (all resident layers
  /// agree; tests; 0 when none).
  [[nodiscard]] std::uint32_t KeptExpansionRows(const cache::Slot&) const;
  /// Test hook: the next `count` pre-launch HIP calls of the adapter (the slot
  /// constructor and BeginRestore preparation) fail as if out of memory.
  static void FailPreLaunchForTest(std::uint32_t count) noexcept;

  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot&, cache::ComponentId, cache::Rows,
      cache::Rows) const override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  struct Piece;
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  [[nodiscard]] std::vector<Piece> PrivatePieces(
      State&, const ContinuationComponent&) const;
  [[nodiscard]] std::byte* RowData(State&, const ContinuationComponent&) const;
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load, cache::Stream&,
      const std::function<void(hip::TransferStream&)>& submit);
  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  std::uint32_t max_drafts_;
  ContinuationLayout layout_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_CONTINUATION_ADAPTER_HPP_
