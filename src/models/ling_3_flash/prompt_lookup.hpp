#ifndef GUFO_MODELS_LING_3_FLASH_PROMPT_LOOKUP_HPP_
#define GUFO_MODELS_LING_3_FLASH_PROMPT_LOOKUP_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace gufo::models::ling_3_flash {

/// Prompt lookup: draft tokens copied from the session's own committed tokens
/// (the prompt and what was generated). The context a cycle continues is the
/// committed tokens plus the cycle's pending tokens; when its last
/// kKeyTokens tokens occurred earlier, the tokens that followed the earlier
/// occurrence are the proposals. Among the most recent occurrences of the key
/// (at most kMaxCandidates), the one whose preceding tokens agree with the
/// context furthest back (at most kMaxMatch tokens) wins, the most recent on
/// ties. A match shorter than the minimum proposes nothing: short matches
/// ("of the") are poor predictions and would only cost verify rows. The
/// default of 12 follows the gufo#426 study of this rule (matches of 12 or
/// more tokens were accepted 82-94 percent of the time, an ungated 3-token
/// rule cost 6-8 percent on every task).
///
/// The index is incremental and bounded: Extend() indexes only the tokens
/// appended since its last call (an open-addressing table from the 3-token key
/// to the newest occurrence, and one chain link per token to the previous
/// occurrence); Find() is one probe plus at most kMaxCandidates x kMaxMatch
/// token comparisons, whatever the context length; a bounded Extend() lets a
/// long prefix be indexed over several calls (Find() searches what is indexed).
/// Host memory: 4 bytes per
/// token for the chain plus 16 bytes per table slot at a load of at most 1/2
/// plus 256 KB of up-front headroom (HostBytes(); at most 68 bytes per token
/// just after a table doubling, about 46 B/token or 9 MB for 200k random ids,
/// far less for repeated text).
///
/// A proposal is a function of the committed tokens alone, which is what makes
/// it a deterministic draft: greedy verification is exact for any draft, and a
/// sampled cycle verifies it as a point-mass proposal (mtp_sampling.hpp).
class PromptLookup {
public:
  static constexpr std::size_t kKeyTokens = 3;
  static constexpr std::size_t kMinMatch = 12;
  static constexpr std::size_t kMaxMatch = 64;
  static constexpr std::size_t kMaxCandidates = 16;

  struct Match {
    std::size_t start{0};   ///< committed index of the first proposal
    std::size_t length{0};  ///< matched context tokens (>= the minimum)
  };

  explicit PromptLookup(std::size_t min_match = kMinMatch)
      : min_match_(std::max(min_match, kKeyTokens)) {}

  void Clear() noexcept {
    std::vector<Slot>().swap(slots_);
    std::vector<std::uint32_t>().swap(chain_);
    used_ = 0;
    last_ = 0;
  }

  /// Tokens already indexed (the length of the committed run Extend() saw).
  [[nodiscard]] std::size_t Size() const noexcept { return chain_.size(); }

  /// Host bytes held by the index.
  [[nodiscard]] std::size_t HostBytes() const noexcept {
    return slots_.capacity() * sizeof(Slot) +
           chain_.capacity() * sizeof(std::uint32_t);
  }

  [[nodiscard]] std::size_t min_match() const noexcept { return min_match_; }

  /// Indexes the committed tokens appended since the last call, at most
  /// `max_new` of them (a long prefix is then indexed over several calls and
  /// Find() proposes nothing until it has caught up; a partial index is safe).
  /// `committed` must extend what was indexed before; anything else (a shorter
  /// run, a changed last indexed token) starts the index over, and callers
  /// Clear() when they replace the tokens.
  void Extend(std::span<const std::int32_t> committed,
              std::size_t max_new = std::numeric_limits<std::size_t>::max()) {
    if (committed.size() < chain_.size() ||
        (!chain_.empty() && committed[chain_.size() - 1] != last_)) {
      Clear();
    }
    if (committed.size() == chain_.size()) {
      return;
    }
    if (committed.size() > std::numeric_limits<std::uint32_t>::max() - 1) {
      return;  // positions are 32-bit; far beyond any context
    }
    const std::size_t stop = std::min(
        committed.size(), chain_.size() + std::min(max_new, committed.size()));
    if (stop - chain_.size() > 64) {
      chain_.reserve(stop);  // a prompt; steps grow geometrically
    }
    // Size the table for every key of the whole run (and some decode headroom)
    // up front, so no later step pays a rehash of a long prefix.
    if (2 * (committed.size() + kHeadroom) > slots_.size()) {
      Presize(committed.size() + kHeadroom);
    }
    for (std::size_t end = chain_.size(); end < stop; ++end) {
      // chain_[end] links the occurrences of the key that ends just before
      // `end` (the continuation committed[end] exists for all of them).
      std::uint32_t previous = kNone;
      if (end >= kKeyTokens) {
        if (2 * (used_ + 1) > slots_.size()) {
          Grow();
        }
        Slot& slot = SlotFor(Key(committed.subspan(end - kKeyTokens)));
        previous = slot.head;
        slot.head = static_cast<std::uint32_t>(end);
      }
      chain_.push_back(previous);
    }
    last_ = committed[chain_.size() - 1];
    // The newest key (ending at committed.size()) has no continuation yet; it
    // is indexed once the next token commits.
  }

  /// Best match for the context `committed` + `tail` among the indexed
  /// committed tokens, or a zero-length match. `tail` holds the uncommitted
  /// part (the pending token and kept drafts); `committed` must be what
  /// Extend() saw.
  [[nodiscard]] Match Find(std::span<const std::int32_t> committed,
                           std::span<const std::int32_t> tail) const {
    const std::size_t context = committed.size() + tail.size();
    // During catch-up (chain_ shorter than committed) only the indexed prefix
    // is searched; every candidate lies below chain_.size().
    if (context < kKeyTokens || chain_.size() > committed.size() ||
        slots_.empty()) {
      return {};
    }
    const auto at = [&](std::size_t i) {
      return i < committed.size() ? committed[i] : tail[i - committed.size()];
    };
    std::array<std::int32_t, kKeyTokens> key_tokens;
    for (std::size_t i = 0; i < kKeyTokens; ++i) {
      key_tokens[i] = at(context - kKeyTokens + i);
    }
    const Slot* slot = Lookup(Key(key_tokens));
    if (slot == nullptr) {
      return {};
    }
    Match best{};
    std::uint32_t end = slot->head;
    for (std::size_t n = 0; n < kMaxCandidates && end != kNone; ++n) {
      // An occurrence ending at `end` proposes committed[end...]. Compare the
      // tokens before it with the context, nearest first (the key included, so
      // a hash collision cannot produce a match).
      std::size_t length = 0;
      while (length < kMaxMatch && length < end && length < context &&
             committed[end - length - 1] == at(context - length - 1)) {
        ++length;
      }
      if (length > best.length) {
        best = {end, length};
      }
      end = chain_[end];
    }
    return best.length >= min_match_ ? best : Match{};
  }

private:
  static constexpr std::uint32_t kNone =
      std::numeric_limits<std::uint32_t>::max();
  static constexpr std::size_t kHeadroom = 8192;
  static constexpr std::uint64_t kEmpty =
      std::numeric_limits<std::uint64_t>::max();

  struct Slot {
    std::uint64_t key{kEmpty};
    std::uint32_t head{kNone};
  };

  /// Token ids are below 2^21 for any vocabulary here (Ling: 157,184), so a key
  /// of three packs exactly; larger ids only alias keys, which Find() rejects
  /// by comparing the tokens themselves.
  static std::uint64_t Key(std::span<const std::int32_t> tokens) {
    std::uint64_t key = 0;
    for (std::size_t i = 0; i < kKeyTokens; ++i) {
      key = (key << 21) |
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[i])) &
             0x1FFFFF);
    }
    return key;
  }

  static std::size_t Hash(std::uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return static_cast<std::size_t>(key);
  }

  [[nodiscard]] const Slot* Lookup(std::uint64_t key) const {
    const std::size_t mask = slots_.size() - 1;
    for (std::size_t i = Hash(key) & mask;; i = (i + 1) & mask) {
      if (slots_[i].key == key) {
        return &slots_[i];
      }
      if (slots_[i].key == kEmpty) {
        return nullptr;
      }
    }
  }

  /// The slot of `key`, inserted when new.
  Slot& SlotFor(std::uint64_t key) {
    const std::size_t mask = slots_.size() - 1;
    std::size_t i = Hash(key) & mask;
    while (slots_[i].key != key && slots_[i].key != kEmpty) {
      i = (i + 1) & mask;
    }
    if (slots_[i].key == kEmpty) {
      slots_[i].key = key;
      ++used_;
    }
    return slots_[i];
  }

  /// A table of at least 2 * `keys` slots (power of two).
  void Presize(std::size_t keys) {
    std::size_t want = 1024;
    while (want < 2 * keys) {
      want *= 2;
    }
    if (want > slots_.size()) {
      Rehash(want);
    }
  }

  void Grow() { Rehash(std::max<std::size_t>(slots_.size() * 2, 1024)); }

  void Rehash(std::size_t size) {
    std::vector<Slot> old(size);
    old.swap(slots_);
    for (const Slot& slot : old) {
      if (slot.key != kEmpty) {
        const std::size_t mask = slots_.size() - 1;
        std::size_t i = Hash(slot.key) & mask;
        while (slots_[i].key != kEmpty) {
          i = (i + 1) & mask;
        }
        slots_[i] = slot;
      }
    }
  }

  std::size_t min_match_;
  std::vector<Slot> slots_;  ///< power-of-two open-addressing table
  std::size_t used_{0};
  std::vector<std::uint32_t> chain_;  ///< previous occurrence per position
  std::int32_t last_{0};              ///< the last token indexed
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_PROMPT_LOOKUP_HPP_
