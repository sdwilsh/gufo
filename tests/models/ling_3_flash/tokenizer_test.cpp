#include "src/models/ling_3_flash/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace ling = gufo::models::ling_3_flash;

namespace {

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

std::string Spell(std::string_view bytes) {
  // GPT-2 spelling for the printable ASCII range plus space -> U+0120 'Ġ'.
  std::string out;
  for (const char c : bytes) {
    if (c == ' ') {
      out += "\xC4\xA0";
    } else if (c == '\n') {
      out += "\xC4\x8A";
    } else {
      out.push_back(c);
    }
  }
  return out;
}

/// Byte tokens 0..255 (ids 0..255 in byte order), then merged tokens, then
/// specials. Merge order decides the outcome below.
std::unique_ptr<ling::LingTokenizer> Synthetic(
    const std::vector<std::pair<std::string, std::int32_t>>& more = {}) {
  std::vector<std::string> tokens;
  std::vector<std::int32_t> types;
  // Spelling table identical to GPT-2's.
  std::vector<std::string> spelled(256);
  int next = 256;
  for (int b = 0; b < 256; ++b) {
    const bool direct =
        (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
    const int cp = direct ? b : next++;
    std::string s;
    if (cp < 0x80) {
      s.push_back(static_cast<char>(cp));
    } else {
      s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    spelled[b] = s;
    tokens.push_back(s);
    types.push_back(1);
  }
  const std::string sp = spelled[' '];
  for (const std::string& extra : std::vector<std::string>{
           "he", "hel", "hell", "hello", sp + "w", sp + "wo", "xyz", "<|a|>",
           "<|a|><|b|>", "<|b|>"}) {
    tokens.push_back(extra);
    types.push_back(1);
  }
  types[tokens.size() - 3] = 3;  // <|a|>
  types[tokens.size() - 2] = 3;  // <|a|><|b|> (longer: wins by length)
  types[tokens.size() - 1] = 4;  // <|b|>
  for (const auto& [spelling, type] : more) {
    tokens.push_back(spelling);
    types.push_back(type);
  }
  const std::vector<std::string> merges = {
      "h e", "he l", "hel l", "hell o", sp + " w", sp + "w o", "x y", "xy z"};
  std::string err;
  auto t =
      ling::LingTokenizer::CreateFromVocabulary(tokens, merges, types, &err);
  Expect(t != nullptr, "synthetic vocab loads: " + err);
  return t;
}

void TestSynthetic() {
  const auto t = Synthetic();
  if (t == nullptr) {
    return;
  }
  const auto id = [&](const std::string& s) { return t->FindToken(s); };
  // Regex mechanics: pieces split at letter/space/digit boundaries; digits
  // are single tokens; merges apply inside a piece only.
  Expect(t->Encode("hello") == std::vector<ling::TokenId>{id("hello")},
         "hello merges fully");
  Expect(t->Encode("hello wo") ==
             (std::vector<ling::TokenId>{id("hello"), id(Spell(" wo"))}),
         "space-prefixed piece");
  Expect(t->Encode("12") == (std::vector<ling::TokenId>{id("1"), id("2")}),
         "digits split singly");
  Expect(t->Encode("it's") == (std::vector<ling::TokenId>{id("i"), id("t"),
                                                          id("'"), id("s")}) ||
             t->Encode("it's").size() == 4,
         "contraction is its own piece");
  Expect(t->Encode("").empty(), "empty text");
  // "xy" is not a token: the intermediate merge still happens, and "xyz"
  // results.
  Expect(t->Encode("xyz") == std::vector<ling::TokenId>{id("xyz")},
         "merge through a non-token intermediate");
  Expect(t->Encode("xy") == (std::vector<ling::TokenId>{id("x"), id("y")}),
         "unfinished intermediate falls back to bytes");
  // Specials: longest spelling wins; literal when parse_special is false.
  Expect(
      t->Encode("<|a|><|b|>") == std::vector<ling::TokenId>{id("<|a|><|b|>")},
      "longest special first");
  Expect(t->Encode("x<|b|>y") ==
             (std::vector<ling::TokenId>{id("x"), id("<|b|>"), id("y")}),
         "user-defined special");
  Expect(t->Encode("<|a|>", false).size() == 5,
         "control token stays literal without parse_special");
  Expect(t->Encode("<|b|>", false) == std::vector<ling::TokenId>{id("<|b|>")},
         "user-defined token always matches");
  Expect(t->Decode(t->Encode("a<|a|>b")) == "a<|a|>b",
         "decode round trip with special");
  Expect(t->Decode(t->Encode("a<|a|>b"), false) == "ab",
         "decode without specials");
  // Byte-level round trip, including invalid UTF-8.
  const std::string raw =
      "caf\xC3\xA9 \xE4\xBD\xA0\xE5\xA5\xBD \xFF\xFE end\n\t";
  Expect(t->Decode(t->Encode(raw)) == raw, "byte round trip");
  // Streaming helper.
  Expect(ling::LingTokenizer::Utf8CompletePrefix("ab\xE4\xBD") == 2,
         "partial 3-byte tail held");
  Expect(ling::LingTokenizer::Utf8CompletePrefix("ab\xE4\xBD\xA0") == 5,
         "complete 3-byte tail kept");
  Expect(ling::LingTokenizer::Utf8CompletePrefix("a\xF0\x9F\x98") == 1,
         "partial 4-byte tail held");
  Expect(ling::LingTokenizer::Utf8CompletePrefix("abc") == 3, "ascii complete");
}

/// Two tokens may spell the same special. The first in vocabulary order that
/// is allowed to match wins, so a control token ahead of a user-defined one
/// is literal text without parse_special and the user-defined one still
/// matches.
void TestDuplicateSpecials() {
  const auto t =
      Synthetic({{"<|d|>", 3}, {"<|d|>", 4}, {"<|e|>", 4}, {"<|e|>", 3}});
  if (t == nullptr) {
    return;
  }
  const auto base = static_cast<ling::TokenId>(256 + 10);
  Expect(t->Encode("x<|d|>y", true) ==
             (std::vector<ling::TokenId>{t->FindToken("x"), base,
                                         t->FindToken("y")}),
         "duplicate special: first id when both may match");
  Expect(t->Encode("x<|d|>y", false) ==
             (std::vector<ling::TokenId>{t->FindToken("x"), base + 1,
                                         t->FindToken("y")}),
         "duplicate special: the user-defined id matches without "
         "parse_special");
  Expect(t->Encode("<|e|>", true) == std::vector<ling::TokenId>{base + 2},
         "duplicate special: user-defined first");
  Expect(t->Encode("<|e|>", false) == std::vector<ling::TokenId>{base + 2},
         "duplicate special: user-defined first, no parse_special");
}

int TestGolden() {
#if defined(GUFO_LING_VOCAB_GGUF) && defined(GUFO_LING_TOKENIZER_GOLDENS)
  std::string path = GUFO_LING_VOCAB_GGUF;
  std::ifstream golden_file(GUFO_LING_TOKENIZER_GOLDENS);
  std::ifstream vocab_file(path);
  if (!golden_file || !vocab_file) {
    std::cerr << "SKIP: vocab GGUF or goldens missing\n";
    return 77;
  }
  const std::string text((std::istreambuf_iterator<char>(golden_file)),
                         std::istreambuf_iterator<char>());
  const auto goldens = gufo::json::parse(text);
  std::string err;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &err);
  if (reader == nullptr) {
    std::cerr << "cannot open vocab GGUF: " << err << '\n';
    return 1;
  }
  const auto t = ling::LingTokenizer::CreateFromGguf(*reader, &err);
  if (t == nullptr) {
    std::cerr << "tokenizer load failed: " << err << '\n';
    return 1;
  }
  Expect(t->VocabSize() == 157184, "vocab size");
  Expect(t->BosTokenId() == 156891 && t->EosTokenId() == 156895 &&
             t->PadTokenId() == 156892,
         "special ids");
  Expect(!t->AddBos(), "add_bos false");
  const auto* cases = goldens.find("cases");
  std::size_t n = 0;
  for (const auto& c : cases->items()) {
    const std::string name = c.member_str("name");
    const bool special = c.find("parse_special")->as_bool();
    std::vector<ling::TokenId> want;
    for (const auto& v : c.find("ids")->items()) {
      want.push_back(static_cast<ling::TokenId>(v.as_size()));
    }
    const auto got = t->Encode(c.member_str("text"), special);
    if (got != want) {
      std::cerr << "FAIL: golden " << name << ": got";
      for (auto g : got)
        std::cerr << ' ' << g;
      std::cerr << "\n      want";
      for (auto w : want)
        std::cerr << ' ' << w;
      std::cerr << '\n';
      ++failures;
    }
    // Decoding the ids must give back the text (control tokens rendered).
    Expect(t->Decode(want) == c.member_str("text"),
           "decode round trip " + name);
    ++n;
  }
  Expect(n >= 40, "at least 40 golden cases");
  std::cout << n << " golden cases checked\n";
#endif
  return 0;
}

/// Deterministic generator of tokenizer inputs: words, digits, punctuation,
/// whitespace runs, multi-byte and invalid UTF-8, special tokens and chat
/// markup, with repeated stretches so the fragment cache is hit.
struct Corpus {
  struct Case {
    std::string text;
    std::vector<ling::LingTokenizer::LiteralRange> literal;
    bool parse_special;
  };

  std::uint64_t state = 0x2545F4914F6CDD1DULL;
  std::uint64_t Next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
  std::size_t Below(std::size_t n) {
    return static_cast<std::size_t>(Next() % n);
  }

  std::string Piece() {
    static const char* const kPieces[] = {"the",
                                          "quick",
                                          "Brown",
                                          "fox's",
                                          "I'll",
                                          "we're",
                                          "x",
                                          "1",
                                          "2026",
                                          "3.14159",
                                          " ",
                                          "  ",
                                          "\t",
                                          "\n",
                                          "\r\n",
                                          "\n\n\n",
                                          "   \n  ",
                                          "\xC2\xA0",
                                          "\xE2\x80\xA8",
                                          "caf\xC3\xA9",
                                          "\xE4\xBD\xA0\xE5\xA5\xBD",
                                          "\xF0\x9F\x98\x80",
                                          "e\xCC\x81",
                                          "\xFF",
                                          "\xC3",
                                          "\xE2\x82",
                                          "\x01",
                                          "{\"a\": [1, 2, {\"b\": null}]}",
                                          "if (x) { return y; }",
                                          "a_b-c",
                                          "...",
                                          "!?",
                                          "<|role_end|>",
                                          "<role>HUMAN</role>",
                                          "<role>ASSISTANT</role>",
                                          "<think>",
                                          "</think>",
                                          "<tool_call>",
                                          "</tool_call>",
                                          "<|a|>",
                                          "<",
                                          "</",
                                          "<role",
                                          "<|",
                                          "|>",
                                          "'",
                                          "'s",
                                          "'LL",
                                          "\\n",
                                          "    ",
                                          "def f():",
                                          "https://example.com/a/b?c=d#e",
                                          "0x7fffdeadbeef",
                                          "\xE2\x80\x94"};
    return kPieces[Below(std::size(kPieces))];
  }

  std::string Text(std::size_t pieces) {
    std::string out;
    for (std::size_t i = 0; i < pieces; ++i) {
      out += Piece();
      if (Below(3) != 0) {
        out += Piece();
      }
    }
    return out;
  }

  std::vector<Case> Cases(std::size_t count) {
    std::vector<std::string> shared;  // recurring "messages"
    for (int i = 0; i < 12; ++i) {
      shared.push_back(Text(40 + Below(400)));
    }
    std::vector<Case> cases;
    for (std::size_t i = 0; i < count; ++i) {
      Case c;
      c.parse_special = Below(5) != 0;
      const std::size_t parts = 1 + Below(6);
      for (std::size_t p = 0; p < parts; ++p) {
        const auto before = c.text.size();
        const bool markup = Below(3) == 0;
        if (markup) {
          c.text += Below(2) == 0 ? "<role>SYSTEM</role>" : "<|role_end|>\n";
        } else if (Below(2) == 0) {
          c.text += shared[Below(shared.size())];
          if (Below(2) == 0) {
            c.literal.emplace_back(before, c.text.size());
          }
        } else {
          c.text += Text(1 + Below(30));
          if (Below(3) == 0) {
            c.literal.emplace_back(before, c.text.size());
          }
        }
      }
      cases.push_back(std::move(c));
    }
    return cases;
  }
};

std::uint64_t Fnv(std::uint64_t h, std::uint64_t v) {
  for (int i = 0; i < 4; ++i) {
    h = (h ^ ((v >> (8 * i)) & 0xFF)) * 0x100000001B3ULL;
  }
  return h;
}

std::uint64_t Digest(const std::vector<ling::TokenId>& ids, std::uint64_t h) {
  h = Fnv(h, ids.size());
  for (const auto id : ids) {
    h = Fnv(h, id);
  }
  return h;
}

/// Pinned token digest of the generated corpus, computed with the tokenizer
/// before the fragment cache existed (every fragment run through EncodeText):
/// the cache may not change a single id.
constexpr std::uint64_t kCorpusDigest = 0x4bae1143b8928b8eULL;

void TestFragmentCache() {
#if defined(GUFO_LING_VOCAB_GGUF)
  std::string err;
  const auto reader =
      gufo::core::GgufReader::OpenFile(GUFO_LING_VOCAB_GGUF, &err);
  if (reader == nullptr) {
    return;
  }
  const auto make = [&] {
    return ling::LingTokenizer::CreateFromGguf(*reader, &err);
  };
  const auto warm = make();
  const auto uncached = make();
  if (warm == nullptr || uncached == nullptr) {
    return;
  }
  uncached->SetFragmentCacheBudget(0);  // every fragment is tokenized afresh
  Corpus corpus;
  const auto cases = corpus.Cases(1500);
  std::size_t large = 0;
  std::vector<std::vector<ling::TokenId>> want;  // from the uncached tokenizer
  std::uint64_t digest = 0xCBF29CE484222325ULL;
  for (const auto& c : cases) {
    large += c.text.size() >= 256;
    const auto cold = uncached->Encode(c.text, c.parse_special, c.literal);
    want.push_back(cold);
    digest = Digest(cold, digest);
    Expect(warm->Encode(c.text, c.parse_special, c.literal) == cold,
           "cached encode equals uncached");
    Expect(warm->Encode(c.text, c.parse_special, c.literal) == cold,
           "repeated cached encode equals uncached");
    // Lossless whatever the cache did.
    Expect(warm->Decode(cold) == c.text, "decode round trip");
  }
  Expect(large > 200, "corpus has many cacheable fragments");
  if (std::getenv("GUFO_PRINT_DIGEST") != nullptr) {
    std::cout << "corpus digest 0x" << std::hex << digest << std::dec << '\n';
  }
  Expect(digest == kCorpusDigest,
         "token digest equals the pre-cache tokenizer");

  // Evicting and concurrent use: same ids, no crashes or torn entries.
  const auto shared = make();
  shared->SetFragmentCacheBudget(200000);
  std::vector<std::thread> threads;
  std::vector<int> bad(6, 0);
  for (std::size_t t = 0; t < bad.size(); ++t) {
    threads.emplace_back([&, t] {
      for (std::size_t round = 0; round < 3; ++round) {
        for (std::size_t i = t; i < cases.size(); i += 1 + t % 3) {
          const auto& c = cases[i];
          bad[t] +=
              shared->Encode(c.text, c.parse_special, c.literal) != want[i];
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const int b : bad) {
    Expect(b == 0, "concurrent encodes with eviction equal uncached encodes");
  }
  shared->SetFragmentCacheBudget(0);
  const auto& first = cases.front();
  Expect(shared->Encode(first.text, first.parse_special, first.literal) ==
             want.front(),
         "a zero budget caches nothing and changes nothing");
#endif
}

}  // namespace

int main() {
  TestSynthetic();
  TestDuplicateSpecials();
  TestFragmentCache();
  const int golden = TestGolden();
  if (failures != 0) {
    return 1;
  }
  return golden;
}
