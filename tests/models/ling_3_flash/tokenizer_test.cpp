#include "src/models/ling_3_flash/tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
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
std::unique_ptr<ling::LingTokenizer> Synthetic() {
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

}  // namespace

int main() {
  TestSynthetic();
  const int golden = TestGolden();
  if (failures != 0) {
    return 1;
  }
  return golden;
}
