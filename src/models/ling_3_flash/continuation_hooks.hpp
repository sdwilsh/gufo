#ifndef GUFO_MODELS_LING_3_FLASH_CONTINUATION_HOOKS_HPP_
#define GUFO_MODELS_LING_3_FLASH_CONTINUATION_HOOKS_HPP_

#include <cstdint>

namespace gufo::models::ling_3_flash {

/// Installed only in sessions owned by the component-cache adapter
/// (continuation_adapter.hpp); every other session leaves the pointer null, so
/// the inactive cost is one pointer test. Both callbacks run on the host before
/// any device work of the step is submitted (including a graph replay).
class ContinuationHooks {
public:
  virtual ~ContinuationHooks() = default;
  /// Whether the slot may execute now: no restore in progress, no latched
  /// failure and no cache transfer in flight. Batched decode checks it per
  /// lane at admission, so one lane that is not ready never fails its peers.
  [[nodiscard]] virtual bool Ready() const noexcept = 0;
  /// Before a step writes MLA latent rows [first, end) of every trunk MLA
  /// layer and MTP latent rows [mtp_first, mtp_end) (empty ranges: none).
  /// Preserves cache borrowers of rows that are still intact below the write.
  /// Appends at the frontier (prefill, decode, verify, MTP drafts) never reach
  /// such rows and never call the cache guard. Throws to refuse the step,
  /// before anything is launched: the session is then untouched.
  virtual void BeforeWrite(std::uint32_t first, std::uint32_t end,
                           std::uint32_t mtp_first, std::uint32_t mtp_end) = 0;
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_CONTINUATION_HOOKS_HPP_
