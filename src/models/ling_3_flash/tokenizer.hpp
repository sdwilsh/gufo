#ifndef GUFO_MODELS_LING_3_FLASH_TOKENIZER_HPP_
#define GUFO_MODELS_LING_3_FLASH_TOKENIZER_HPP_

#include <unicode/uversion.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gufo::core {
class GgufReader;
}

U_NAMESPACE_BEGIN
class RegexPattern;
U_NAMESPACE_END

namespace gufo::models::ling_3_flash {

using TokenId = std::uint32_t;
inline constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;

/// llama.cpp / GGUF `tokenizer.ggml.token_type` values.
enum class TokenType : std::int32_t {
  kUndefined = 0,
  kNormal = 1,
  kUnknown = 2,
  kControl = 3,
  kUserDefined = 4,
  kUnused = 5,
  kByte = 6,
};

/// Byte-level BPE tokenizer for the GGUF `gpt2` model with the `bailingmoe2`
/// pre-tokenizer (llama.cpp's greedy regex, evaluated with ICU). Mirrors
/// llama.cpp b11146: every control, user-defined and unknown token found in
/// the text becomes one token (control/unknown only with parse_special)
/// (longest text first, as llama.cpp partitions it); the rest is pre-split,
/// then merged by rank.
class LingTokenizer {
public:
  ~LingTokenizer();
  LingTokenizer(const LingTokenizer&) = delete;
  LingTokenizer& operator=(const LingTokenizer&) = delete;

  /// Rejects anything but `tokenizer.ggml.model == gpt2` with pre
  /// `bailingmoe2` (the pre-tokenizer is not configurable here).
  [[nodiscard]] static std::unique_ptr<LingTokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Direct construction (tests, synthetic vocabularies). Tokens use the GPT-2
  /// byte-to-unicode spelling. `token_types` may be empty (all normal).
  [[nodiscard]] static std::unique_ptr<LingTokenizer> CreateFromVocabulary(
      std::span<const std::string> tokens, std::span<const std::string> merges,
      std::span<const std::int32_t> token_types, std::string* error_msg);

  /// Encodes UTF-8 text. No BOS/EOS is added (the GGUF has add_bos=false and
  /// add_eos=false); the chat template carries every delimiter.
  ///
  /// `literal_ranges` (byte offsets into `text`) are conversation data: no
  /// special or user-defined token may match any byte inside them, so message
  /// bodies cannot forge turn markers. Elsewhere, and across the range edges,
  /// tokenization is that of the whole text.
  using LiteralRange = std::pair<std::size_t, std::size_t>;
  [[nodiscard]] std::vector<TokenId> Encode(
      std::string_view text, bool parse_special = true,
      std::span<const LiteralRange> literal_ranges = {}) const;

  /// Concatenated bytes of the tokens. Control tokens render as their text
  /// when `render_special`, otherwise as nothing. The result may end in a
  /// partial UTF-8 sequence; see Utf8CompletePrefix for streaming.
  [[nodiscard]] std::string Decode(std::span<const TokenId> ids,
                                   bool render_special = true) const;
  [[nodiscard]] std::string DecodeToken(TokenId id,
                                        bool render_special = true) const;

  /// Length of the longest prefix of `bytes` that does not end in an
  /// incomplete UTF-8 sequence. A streaming consumer emits that prefix and
  /// keeps the remainder until the next token arrives.
  [[nodiscard]] static std::size_t Utf8CompletePrefix(
      std::string_view bytes) noexcept;

  [[nodiscard]] std::size_t VocabSize() const noexcept {
    return id_to_token_.size();
  }
  [[nodiscard]] TokenId BosTokenId() const noexcept { return bos_; }
  [[nodiscard]] TokenId EosTokenId() const noexcept { return eos_; }
  [[nodiscard]] TokenId PadTokenId() const noexcept { return pad_; }
  [[nodiscard]] bool AddBos() const noexcept { return add_bos_; }
  [[nodiscard]] TokenId FindToken(std::string_view text) const noexcept;
  [[nodiscard]] bool IsSpecialToken(TokenId id) const noexcept;

private:
  struct PairHash {
    std::size_t operator()(
        const std::pair<TokenId, TokenId>& p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32) ^ p.second;
    }
  };
  struct Merge {
    std::uint32_t rank;
    TokenId token;
  };

  LingTokenizer();

  void MergePiece(std::string_view piece, std::vector<TokenId>& out) const;
  void EncodeText(std::string_view text, std::vector<TokenId>& out) const;

  std::vector<std::string> id_to_token_;  // vocabulary spelling
  std::vector<std::string> id_to_bytes_;  // byte-decoded
  std::vector<std::int32_t> token_type_;
  /// Intermediate merge results that are not tokens (ids >= vocab size).
  std::vector<std::string> virtual_bytes_;
  std::unordered_map<std::string, TokenId> virtual_ids_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::pair<TokenId, TokenId>, Merge, PairHash> merges_;
  /// Specials, longest text first (llama.cpp's cache_special_tokens order).
  std::vector<std::pair<std::string, TokenId>> specials_;
  TokenId byte_token_[256];
  TokenId bos_{kInvalidTokenId};
  TokenId eos_{kInvalidTokenId};
  TokenId pad_{kInvalidTokenId};
  bool add_bos_{false};
  std::unique_ptr<icu::RegexPattern> pattern_;
};

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_TOKENIZER_HPP_
