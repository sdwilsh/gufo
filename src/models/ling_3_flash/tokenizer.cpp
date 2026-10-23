#include "src/models/ling_3_flash/tokenizer.hpp"

#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::ling_3_flash {
namespace {

// llama.cpp's LLAMA_VOCAB_PRE_TYPE_BAILINGMOE regex (possessive quantifiers
// of the original tokenizer.json replaced by greedy ones, as upstream does).
constexpr const char16_t* kBailingPattern =
    u"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
    u"\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

/// GPT-2 byte <-> unicode table: printable bytes map to themselves, the rest
/// to U+0100 upwards in byte order.
const std::array<char32_t, 256>& ByteToCodepoint() {
  static const auto table = [] {
    std::array<char32_t, 256> t{};
    char32_t next = 256;
    for (int b = 0; b < 256; ++b) {
      const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                          (b >= 0xAE && b <= 0xFF);
      t[b] = direct ? static_cast<char32_t>(b) : next++;
    }
    return t;
  }();
  return table;
}

bool IsSpecialType(std::int32_t type) noexcept {
  return type == static_cast<int>(TokenType::kControl) ||
         type == static_cast<int>(TokenType::kUserDefined) ||
         type == static_cast<int>(TokenType::kUnknown);
}

std::string ByteSpelling(std::uint8_t byte) {
  std::string out;
  AppendUtf8(out, ByteToCodepoint()[byte]);
  return out;
}

/// Inverse of the byte spelling. Characters outside the table (literal text in
/// a control token) are kept as UTF-8.
std::string UnspellBytes(std::string_view text) {
  static const auto inverse = [] {
    std::unordered_map<char32_t, std::uint8_t> m;
    for (int b = 0; b < 256; ++b) {
      m[ByteToCodepoint()[b]] = static_cast<std::uint8_t>(b);
    }
    return m;
  }();
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    const auto b0 = static_cast<std::uint8_t>(text[i]);
    std::size_t len = b0 < 0x80         ? 1
                      : (b0 >> 5) == 6  ? 2
                      : (b0 >> 4) == 14 ? 3
                      : (b0 >> 3) == 30 ? 4
                                        : 1;
    len = std::min(len, text.size() - i);
    char32_t cp = len == 1 ? b0 : (b0 & (0xFF >> (len + 1)));
    for (std::size_t k = 1; k < len; ++k) {
      cp = (cp << 6) | (static_cast<std::uint8_t>(text[i + k]) & 0x3F);
    }
    if (const auto it = inverse.find(cp); it != inverse.end()) {
      out.push_back(static_cast<char>(it->second));
    } else {
      out.append(text.substr(i, len));
    }
    i += len;
  }
  return out;
}

struct Utf16Text {
  icu::UnicodeString text;
  /// Byte offset in the source of each UTF-16 unit (size + 1 entries).
  std::vector<std::uint32_t> byte_offset;
};

/// UTF-8 -> UTF-16 with a unit->byte map. Invalid bytes become U+FFFD so the
/// regex still sees (and splits at) them; slicing stays on source bytes.
Utf16Text ToUtf16(std::string_view s) {
  Utf16Text out;
  out.byte_offset.reserve(s.size() + 1);
  std::size_t i = 0;
  while (i < s.size()) {
    const auto b0 = static_cast<std::uint8_t>(s[i]);
    std::size_t len = 1;
    char32_t cp = 0xFFFD;
    if (b0 < 0x80) {
      cp = b0;
    } else {
      const std::size_t want = (b0 >> 5) == 6    ? 2
                               : (b0 >> 4) == 14 ? 3
                               : (b0 >> 3) == 30 ? 4
                                                 : 0;
      if (want != 0 && i + want <= s.size()) {
        char32_t v = b0 & (0xFF >> (want + 1));
        bool ok = true;
        for (std::size_t k = 1; k < want; ++k) {
          const auto b = static_cast<std::uint8_t>(s[i + k]);
          if ((b & 0xC0) != 0x80) {
            ok = false;
            break;
          }
          v = (v << 6) | (b & 0x3F);
        }
        const char32_t min = want == 2 ? 0x80 : want == 3 ? 0x800 : 0x10000;
        if (ok && v >= min && v <= 0x10FFFF && !(v >= 0xD800 && v <= 0xDFFF)) {
          cp = v;
          len = want;
        }
      }
    }
    const auto before = out.text.length();
    out.text.append(static_cast<UChar32>(cp));
    for (auto u = before; u < out.text.length(); ++u) {
      out.byte_offset.push_back(static_cast<std::uint32_t>(i));
    }
    i += len;
  }
  out.byte_offset.push_back(static_cast<std::uint32_t>(s.size()));
  return out;
}

}  // namespace

/// Byte-exact memo of EncodeText results, least recently used first out. The
/// tokens of a fragment depend on its bytes alone (the pre-tokenizer and the
/// merges are immutable), so a hit returns precisely what EncodeText would
/// compute; keys are compared in full, never by hash alone.
class LingTokenizer::FragmentCache {
public:
  /// Shorter fragments cost less to tokenize than to look up.
  static constexpr std::size_t kMinBytes = 256;
  /// Keys plus tokens kept (about 2.5 MB per 100k-token conversation).
  static constexpr std::size_t kDefaultBudgetBytes = std::size_t{64} << 20;

  using Tokens = std::shared_ptr<const std::vector<TokenId>>;

  [[nodiscard]] static std::size_t Hash(std::string_view text) noexcept {
    return std::hash<std::string_view>{}(text);
  }

  [[nodiscard]] Tokens Find(std::string_view text, std::size_t hash) {
    const std::lock_guard lock(mutex_);
    for (auto [it, end] = index_.equal_range(hash); it != end; ++it) {
      if (it->second->key == text) {
        entries_.splice(entries_.begin(), entries_, it->second);
        return it->second->tokens;
      }
    }
    return nullptr;
  }

  void SetBudget(std::size_t bytes) {
    const std::lock_guard lock(mutex_);
    budget_ = bytes;
    Trim();
  }

  void Insert(std::string_view text, std::size_t hash, Tokens tokens) {
    const std::size_t cost = Cost(text, *tokens);
    const std::lock_guard lock(mutex_);
    if (cost > budget_ / 4) {
      return;
    }
    for (auto [it, end] = index_.equal_range(hash); it != end; ++it) {
      if (it->second->key == text) {
        return;  // another request computed it first
      }
    }
    entries_.push_front({std::string(text), hash, std::move(tokens), cost});
    try {
      index_.emplace(hash, entries_.begin());
    } catch (...) {
      entries_.pop_front();  // never keep an entry the index cannot find
      throw;
    }
    bytes_ += cost;
    Trim();
  }

private:
  /// Drops least recently used entries until the budget holds. Locked.
  void Trim() {
    while (bytes_ > budget_ && !entries_.empty()) {
      const auto last = std::prev(entries_.end());
      for (auto [it, end] = index_.equal_range(last->hash); it != end; ++it) {
        if (it->second == last) {
          index_.erase(it);
          bytes_ -= std::min(bytes_, last->cost);
          break;
        }
      }
      entries_.erase(last);
    }
  }

  struct Entry {
    std::string key;
    std::size_t hash;
    Tokens tokens;
    std::size_t cost;
  };

  static std::size_t Cost(std::string_view text,
                          const std::vector<TokenId>& tokens) {
    return text.size() + tokens.size() * sizeof(TokenId) + 128;
  }

  std::mutex mutex_;
  std::list<Entry> entries_;
  std::unordered_multimap<std::size_t, std::list<Entry>::iterator> index_;
  std::size_t budget_{kDefaultBudgetBytes};
  std::size_t bytes_{0};
};

LingTokenizer::LingTokenizer() : fragments_(std::make_unique<FragmentCache>()) {
  std::fill(std::begin(byte_token_), std::end(byte_token_), kInvalidTokenId);
}
LingTokenizer::~LingTokenizer() = default;

std::unique_ptr<LingTokenizer> LingTokenizer::CreateFromVocabulary(
    std::span<const std::string> tokens, std::span<const std::string> merges,
    std::span<const std::int32_t> token_types, std::string* error_msg) {
  if (tokens.empty()) {
    SetError(error_msg, "Ling tokenizer: empty vocabulary");
    return nullptr;
  }
  if (!token_types.empty() && token_types.size() != tokens.size()) {
    SetError(error_msg,
             "Ling tokenizer: token_type length differs from tokens");
    return nullptr;
  }
  std::unique_ptr<LingTokenizer> t(new LingTokenizer());
  t->id_to_token_.assign(tokens.begin(), tokens.end());
  t->token_type_.assign(tokens.size(),
                        static_cast<std::int32_t>(TokenType::kNormal));
  if (!token_types.empty()) {
    t->token_type_.assign(token_types.begin(), token_types.end());
  }
  t->token_to_id_.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    t->token_to_id_.emplace(tokens[i],
                            static_cast<TokenId>(i));  // first id wins
  }
  for (int b = 0; b < 256; ++b) {
    if (const auto it =
            t->token_to_id_.find(ByteSpelling(static_cast<std::uint8_t>(b)));
        it != t->token_to_id_.end()) {
      t->byte_token_[b] = it->second;
    }
  }

  // Merge halves and results are matched by spelling, as llama.cpp does. A
  // result that is not a vocabulary token (an intermediate step such as
  // "<think" on the way to "<think>") gets a virtual id past the vocabulary;
  // a piece that ends on one is emitted as its byte tokens.
  const auto intern = [&](const std::string& spelling) {
    if (const auto it = t->token_to_id_.find(spelling);
        it != t->token_to_id_.end()) {
      return it->second;
    }
    const auto id =
        static_cast<TokenId>(tokens.size() + t->virtual_bytes_.size());
    t->virtual_bytes_.push_back(UnspellBytes(spelling));
    t->virtual_ids_.emplace(spelling, id);
    return id;
  };
  const auto lookup =
      [&](const std::string& spelling) -> std::optional<TokenId> {
    if (const auto it = t->token_to_id_.find(spelling);
        it != t->token_to_id_.end()) {
      return it->second;
    }
    if (const auto it = t->virtual_ids_.find(spelling);
        it != t->virtual_ids_.end()) {
      return it->second;
    }
    return std::nullopt;
  };
  struct Parsed {
    std::string left, right;
  };
  std::vector<Parsed> parsed(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const std::string_view m = merges[rank];
    const auto space = m.find(' ', 1);
    if (space == std::string_view::npos) {
      continue;
    }
    parsed[rank] = {std::string(m.substr(0, space)),
                    std::string(m.substr(space + 1))};
    intern(parsed[rank].left + parsed[rank].right);
  }
  t->merges_.reserve(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    if (parsed[rank].left.empty()) {
      continue;
    }
    const auto a = lookup(parsed[rank].left);
    const auto b = lookup(parsed[rank].right);
    if (!a || !b) {
      continue;  // a half that no merge can produce: the pair cannot occur
    }
    t->merges_.emplace(
        std::pair{*a, *b},
        Merge{rank, *lookup(parsed[rank].left + parsed[rank].right)});
  }

  t->id_to_bytes_.resize(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const bool special = IsSpecialType(t->token_type_[i]);
    t->id_to_bytes_[i] = special ? tokens[i] : UnspellBytes(tokens[i]);
    if (special && !tokens[i].empty()) {
      t->specials_.emplace_back(tokens[i], static_cast<TokenId>(i));
    }
  }
  std::stable_sort(t->specials_.begin(), t->specials_.end(),
                   [](const auto& a, const auto& b) {
                     return a.first.size() > b.first.size();
                   });
  t->special_tree_.emplace_back();
  for (std::size_t i = 0; i < t->specials_.size(); ++i) {
    std::uint32_t node = 0;
    for (const unsigned char byte : t->specials_[i].first) {
      auto& next = t->special_tree_[node].next;
      auto it = std::find_if(next.begin(), next.end(),
                             [&](const auto& e) { return e.first == byte; });
      if (it == next.end()) {
        next.emplace_back(byte,
                          static_cast<std::uint32_t>(t->special_tree_.size()));
        t->special_tree_.emplace_back();
        node = static_cast<std::uint32_t>(t->special_tree_.size() - 1);
      } else {
        node = it->second;
      }
    }
    // Spellings can repeat (one per token id); every one stays a candidate.
    t->special_tree_[node].specials.push_back(static_cast<std::int32_t>(i));
    t->special_first_[static_cast<unsigned char>(t->specials_[i].first[0])] =
        true;
  }

  UErrorCode status = U_ZERO_ERROR;
  UParseError parse_error;
  t->pattern_.reset(icu::RegexPattern::compile(
      icu::UnicodeString(kBailingPattern), 0, parse_error, status));
  if (U_FAILURE(status)) {
    SetError(
        error_msg,
        std::string("Ling tokenizer: pre-tokenizer regex failed to compile: ") +
            u_errorName(status));
    return nullptr;
  }
  return t;
}

std::unique_ptr<LingTokenizer> LingTokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  if (reader.GetMetadataString("tokenizer.ggml.model") !=
      std::optional<std::string_view>{"gpt2"}) {
    SetError(error_msg, "Ling tokenizer requires tokenizer.ggml.model == gpt2");
    return nullptr;
  }
  if (reader.GetMetadataString("tokenizer.ggml.pre") !=
      std::optional<std::string_view>{"bailingmoe2"}) {
    SetError(error_msg,
             "Ling tokenizer requires tokenizer.ggml.pre == bailingmoe2");
    return nullptr;
  }
  const auto token_views =
      reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (token_views.empty()) {
    SetError(error_msg, "GGUF metadata missing tokenizer.ggml.tokens");
    return nullptr;
  }
  std::vector<std::string> tokens(token_views.begin(), token_views.end());
  const auto merge_views =
      reader.GetMetadataStringArray("tokenizer.ggml.merges");
  std::vector<std::string> merges(merge_views.begin(), merge_views.end());

  std::vector<std::int32_t> types;
  if (const auto* meta = reader.FindMetadata("tokenizer.ggml.token_type");
      meta != nullptr) {
    if (const auto* i = std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      types.assign(i->begin(), i->end());
    } else if (const auto* u =
                   std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      types.assign(u->begin(), u->end());
    }
  }
  if (types.empty()) {
    SetError(error_msg, "GGUF metadata missing tokenizer.ggml.token_type");
    return nullptr;
  }
  auto t = CreateFromVocabulary(tokens, merges, types, error_msg);
  if (t == nullptr) {
    return nullptr;
  }
  const auto id = [&](const char* key) {
    const auto v = reader.GetMetadataUint32(key);
    return v && *v < t->id_to_token_.size() ? *v : kInvalidTokenId;
  };
  t->bos_ = id("tokenizer.ggml.bos_token_id");
  t->eos_ = id("tokenizer.ggml.eos_token_id");
  t->pad_ = id("tokenizer.ggml.padding_token_id");
  t->add_bos_ =
      reader.GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(false);
  return t;
}

void LingTokenizer::MergePiece(std::string_view piece,
                               std::vector<TokenId>& out) const {
  if (piece.empty()) {
    return;
  }
  // One symbol per byte, merged lowest rank first, leftmost on ties (llama.cpp
  // bigram queue order), over a linked list so long runs stay O(n log n).
  std::vector<TokenId> sym;
  sym.reserve(piece.size());
  for (const unsigned char c : piece) {
    if (byte_token_[c] != kInvalidTokenId) {
      sym.push_back(byte_token_[c]);
    }
  }
  const std::size_t n = sym.size();
  if (n <= 1) {
    out.insert(out.end(), sym.begin(), sym.end());
    return;
  }
  constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> prev(n), next(n);
  for (std::size_t i = 0; i < n; ++i) {
    prev[i] = i == 0 ? kNone : i - 1;
    next[i] = i + 1 < n ? i + 1 : kNone;
  }
  struct Candidate {
    std::uint32_t rank;
    std::size_t index;
    TokenId left, right;
  };
  const auto later = [](const Candidate& a, const Candidate& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.index > b.index;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)> queue(
      later);
  const auto offer = [&](std::size_t l, std::size_t r) {
    if (const auto it = merges_.find({sym[l], sym[r]}); it != merges_.end()) {
      queue.push({it->second.rank, l, sym[l], sym[r]});
    }
  };
  for (std::size_t i = 0; i + 1 < n; ++i) {
    offer(i, i + 1);
  }
  while (!queue.empty()) {
    const Candidate c = queue.top();
    queue.pop();
    const std::size_t r = next[c.index];
    if (sym[c.index] != c.left || r == kNone || sym[r] != c.right) {
      continue;  // stale
    }
    sym[c.index] = merges_.find({c.left, c.right})->second.token;
    const std::size_t after = next[r];
    next[c.index] = after;
    if (after != kNone) {
      prev[after] = c.index;
    }
    sym[r] = kInvalidTokenId;
    if (prev[c.index] != kNone) {
      offer(prev[c.index], c.index);
    }
    if (after != kNone) {
      offer(c.index, after);
    }
  }
  for (std::size_t i = 0; i != kNone; i = next[i]) {
    if (sym[i] < id_to_token_.size()) {
      out.push_back(sym[i]);
      continue;
    }
    for (const unsigned char b : virtual_bytes_[sym[i] - id_to_token_.size()]) {
      if (byte_token_[b] != kInvalidTokenId) {
        out.push_back(byte_token_[b]);
      }
    }
  }
}

void LingTokenizer::EncodeText(std::string_view text,
                               std::vector<TokenId>& out) const {
  if (text.empty()) {
    return;
  }
  const Utf16Text utf16 = ToUtf16(text);
  UErrorCode status = U_ZERO_ERROR;
  std::unique_ptr<icu::RegexMatcher> matcher(
      pattern_->matcher(utf16.text, status));
  if (U_FAILURE(status)) {
    throw std::runtime_error("Ling tokenizer: regex matcher failed");
  }
  matcher->setStackLimit(0, status);  // 0 = grow the heap stack as needed
  std::size_t covered = 0;            // byte offset already emitted
  while (matcher->find(status) && U_SUCCESS(status)) {
    const auto begin =
        utf16.byte_offset[static_cast<std::size_t>(matcher->start(status))];
    const auto end =
        utf16.byte_offset[static_cast<std::size_t>(matcher->end(status))];
    if (begin > covered) {  // text no alternative matched: still BPE it
      MergePiece(text.substr(covered, begin - covered), out);
    }
    MergePiece(text.substr(begin, end - begin), out);
    covered = end;
  }
  if (U_FAILURE(status)) {
    throw std::runtime_error("Ling tokenizer: regex evaluation failed");
  }
  if (covered < text.size()) {
    MergePiece(text.substr(covered), out);
  }
}

std::vector<bool> LingTokenizer::SpecialsPresent(std::string_view text) const {
  std::vector<bool> present(specials_.size(), false);
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (!special_first_[static_cast<unsigned char>(text[i])]) {
      continue;
    }
    std::uint32_t node = 0;
    for (std::size_t k = i; k < text.size(); ++k) {
      const auto byte = static_cast<std::uint8_t>(text[k]);
      const auto& next = special_tree_[node].next;
      const auto it =
          std::find_if(next.begin(), next.end(),
                       [&](const auto& e) { return e.first == byte; });
      if (it == next.end()) {
        break;
      }
      node = it->second;
      for (const auto special : special_tree_[node].specials) {
        present[static_cast<std::size_t>(special)] = true;
      }
    }
  }
  return present;
}

void LingTokenizer::SetFragmentCacheBudget(std::size_t bytes) {
  fragments_->SetBudget(bytes);
}

void LingTokenizer::EncodeFragment(std::string_view text,
                                   std::vector<TokenId>& out) const {
  if (text.size() < FragmentCache::kMinBytes) {
    EncodeText(text, out);
    return;
  }
  const auto hash = FragmentCache::Hash(text);
  auto tokens = fragments_->Find(text, hash);
  if (tokens == nullptr) {
    std::vector<TokenId> computed;
    EncodeText(text, computed);
    tokens = std::make_shared<const std::vector<TokenId>>(std::move(computed));
    fragments_->Insert(text, hash, tokens);
  }
  out.insert(out.end(), tokens->begin(), tokens->end());
}

std::vector<TokenId> LingTokenizer::Encode(
    std::string_view text, bool parse_special,
    std::span<const LiteralRange> literal_ranges) const {
  const auto in_literal = [&](std::size_t begin, std::size_t length) {
    return std::ranges::any_of(literal_ranges, [&](const LiteralRange& r) {
      return r.first < begin + length && begin < r.second;
    });
  };
  std::vector<TokenId> out;
  struct Fragment {
    bool is_special;
    TokenId id;
    std::size_t begin, length;
  };
  const auto type_of = [&](TokenId id) {
    return static_cast<TokenType>(token_type_[id]);
  };
  std::vector<Fragment> fragments{{false, 0, 0, text.size()}};
  const auto present = SpecialsPresent(text);
  for (std::size_t index = 0; index < specials_.size(); ++index) {
    const auto& [spelling, id] = specials_[index];
    // Like llama.cpp, control and unknown tokens are literal text unless
    // parse_special; user-defined tokens (<think>, <tool_call>, ...) always
    // match.
    if (!parse_special && type_of(id) != TokenType::kUserDefined) {
      continue;
    }
    if (!present[index]) {
      continue;
    }
    std::vector<Fragment> next;
    next.reserve(fragments.size() + 2);
    for (const Fragment& f : fragments) {
      if (f.is_special) {
        next.push_back(f);
        continue;
      }
      const std::string_view body = text.substr(f.begin, f.length);
      std::size_t pos = 0;
      while (true) {
        auto hit = body.find(spelling, pos);
        while (hit != std::string_view::npos &&
               in_literal(f.begin + hit, spelling.size())) {
          hit = body.find(spelling, hit + 1);
        }
        if (hit == std::string_view::npos) {
          break;
        }
        if (hit > pos) {
          next.push_back({false, 0, f.begin + pos, hit - pos});
        }
        next.push_back({true, id, 0, 0});
        pos = hit + spelling.size();
      }
      if (pos < body.size()) {
        next.push_back({false, 0, f.begin + pos, body.size() - pos});
      }
    }
    fragments = std::move(next);
  }
  for (const Fragment& f : fragments) {
    if (f.is_special) {
      out.push_back(f.id);
    } else {
      EncodeFragment(text.substr(f.begin, f.length), out);
    }
  }
  return out;
}

std::string LingTokenizer::DecodeToken(TokenId id, bool render_special) const {
  if (id >= id_to_bytes_.size()) {
    return {};
  }
  if (!render_special && IsSpecialToken(id)) {
    return {};
  }
  return id_to_bytes_[id];
}

std::string LingTokenizer::Decode(std::span<const TokenId> ids,
                                  bool render_special) const {
  std::string out;
  for (const TokenId id : ids) {
    out += DecodeToken(id, render_special);
  }
  return out;
}

std::size_t LingTokenizer::Utf8CompletePrefix(std::string_view bytes) noexcept {
  // Look back at most 3 bytes for a lead byte whose sequence is unfinished.
  const std::size_t n = bytes.size();
  for (std::size_t back = 1; back <= std::min<std::size_t>(3, n); ++back) {
    const auto b = static_cast<std::uint8_t>(bytes[n - back]);
    if ((b & 0xC0) == 0x80) {
      continue;  // continuation byte; keep looking for its lead
    }
    const std::size_t need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
    return need > back ? n - back : n;
  }
  return n;
}

TokenId LingTokenizer::FindToken(std::string_view text) const noexcept {
  const auto it = token_to_id_.find(std::string(text));
  return it == token_to_id_.end() ? kInvalidTokenId : it->second;
}

bool LingTokenizer::IsSpecialToken(TokenId id) const noexcept {
  if (id >= token_type_.size()) {
    return false;
  }
  return IsSpecialType(token_type_[id]);
}

}  // namespace gufo::models::ling_3_flash
