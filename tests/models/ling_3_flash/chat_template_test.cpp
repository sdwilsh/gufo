#include "src/models/ling_3_flash/chat_template.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
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

std::string ReadFile(const char* path, bool* ok) {
  std::ifstream f(path, std::ios::binary);
  *ok = static_cast<bool>(f);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

template<class F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

// Memory GGUF with string metadata only.
std::vector<std::uint8_t> MiniGguf(const std::string& arch,
                                   const std::string& tmpl) {
  std::vector<std::uint8_t> b;
  const auto bytes = [&](const void* p, std::size_t n) {
    b.insert(b.end(), static_cast<const std::uint8_t*>(p),
             static_cast<const std::uint8_t*>(p) + n);
  };
  const auto u32 = [&](std::uint32_t v) { bytes(&v, 4); };
  const auto u64 = [&](std::uint64_t v) { bytes(&v, 8); };
  const auto str = [&](const std::string& s) {
    u64(s.size());
    bytes(s.data(), s.size());
  };
  bytes("GGUF", 4);
  u32(3);
  u64(0);
  u64(2);
  for (const auto& [k, v] :
       {std::pair<std::string, std::string>{"general.architecture", arch},
        {"tokenizer.chat_template", tmpl}}) {
    str(k);
    u32(8);
    str(v);
  }
  while (b.size() % 32 != 0)
    b.push_back(0);
  return b;
}

void TestGoldens() {
  bool ok = false;
  const std::string tmpl = ReadFile(GUFO_LING_CHAT_TEMPLATE, &ok);
  Expect(ok, "template fixture readable");
  const auto* p = reinterpret_cast<const std::uint8_t*>(tmpl.data());
  Expect(gufo::crypto::Sha256Hex({p, tmpl.size()}) ==
             ling::ArtifactTemplateSha256(),
         "fixture template has the pinned SHA-256");

  const std::string text = ReadFile(GUFO_LING_CHAT_GOLDENS, &ok);
  Expect(ok, "goldens readable");
  const auto goldens = gufo::json::parse(text);
  std::size_t n = 0;
  for (const auto& c : goldens.find("cases")->items()) {
    std::vector<ling::ChatMessage> messages;
    for (const auto& m : c.find("messages")->items()) {
      ling::ChatMessage msg;
      msg.role = m.member_str("role");
      msg.content = m.member_str("content");
      msg.reasoning_content = m.member_str("reasoning_content");
      for (const auto& tc : m.find("tool_calls")->items()) {
        ling::ChatMessage::ToolCall call;
        call.name = tc.member_str("name");
        for (const auto& a : tc.find("arguments")->items()) {
          call.arguments.push_back({a.member_str("name"), a.member_str("value"),
                                    a.find("is_string")->as_bool()});
        }
        msg.tool_calls.push_back(std::move(call));
      }
      messages.push_back(std::move(msg));
    }
    std::vector<ling::ChatTool> tools;
    for (const auto& t : c.find("tools")->items()) {
      tools.push_back({t.str()});
    }
    ling::ChatTemplateOptions options;
    if (const auto* v = c.find("enable_thinking");
        v != nullptr && v->is_bool()) {
      options.enable_thinking = v->as_bool();
    }
    if (const auto* v = c.find("thinking_option");
        v != nullptr && v->is_string()) {
      options.thinking_option = v->str();
    }
    options.add_generation_prompt = c.find("add_generation_prompt")->as_bool();
    const std::string got = ling::RenderChat(messages, tools, options);
    const std::string want = c.member_str("expected");
    if (got != want) {
      std::cerr << "FAIL: chat golden " << c.member_str("name")
                << "\n  got:  " << got << "\n  want: " << want << '\n';
      ++failures;
    }
    ++n;
  }
  Expect(n >= 8, "at least 8 chat goldens");
  std::cout << n << " chat goldens checked\n";
}

void TestFailures() {
  const auto msg = [](const char* role) {
    ling::ChatMessage m;
    m.role = role;
    m.content = "x";
    return m;
  };
  const std::vector<ling::ChatMessage> user = {msg("user")};
  Expect(Throws([&] { (void)ling::RenderChat({}); }),
         "empty messages rejected");
  const std::vector<ling::ChatMessage> dev = {msg("developer")};
  Expect(Throws([&] { (void)ling::RenderChat(dev); }), "unknown role rejected");
  const std::vector<ling::ChatTool> bad_json = {{"{\"a\": "}};
  Expect(Throws([&] { (void)ling::RenderChat(user, bad_json); }),
         "malformed tool JSON rejected");
  const std::vector<ling::ChatTool> exponent = {{"{\"a\": 1e5}"}};
  Expect(Throws([&] { (void)ling::RenderChat(user, exponent); }),
         "exponent number rejected");
  Expect(Throws([&] { (void)ling::PythonJsonDumps("[1.50]"); }),
         "trailing-zero float rejected");
  Expect(ling::PythonJsonDumps(" {\"a\":[1,2.5,\"\\u00fc\\n\"],\"b\":{}} ") ==
             "{\"a\": [1, 2.5, \"\xC3\xBC\\n\"], \"b\": {}}",
         "Python-style respelling");
}

void TestGate() {
  bool ok = false;
  const std::string tmpl = ReadFile(GUFO_LING_CHAT_TEMPLATE, &ok);
  std::string err;
  auto good = MiniGguf("bailingmoe3", tmpl);
  auto reader =
      gufo::core::GgufReader::OpenMemory(good.data(), good.size(), &err);
  Expect(reader != nullptr && ling::ValidateGgufTemplate(*reader, &err),
         "pinned template accepted: " + err);

  auto changed = MiniGguf("bailingmoe3", tmpl + " ");
  reader =
      gufo::core::GgufReader::OpenMemory(changed.data(), changed.size(), &err);
  Expect(reader != nullptr && !ling::ValidateGgufTemplate(*reader, &err),
         "modified template rejected");
  auto other = MiniGguf("qwen35", tmpl);
  reader = gufo::core::GgufReader::OpenMemory(other.data(), other.size(), &err);
  Expect(reader != nullptr && !ling::ValidateGgufTemplate(*reader, &err),
         "wrong architecture rejected");

  // The committed vocab-only GGUF carries the real template.
  if (std::ifstream(GUFO_LING_VOCAB_GGUF).good()) {
    auto vocab = gufo::core::GgufReader::OpenFile(GUFO_LING_VOCAB_GGUF, &err);
    Expect(vocab != nullptr && ling::ValidateGgufTemplate(*vocab, &err),
           "vocab GGUF template accepted: " + err);
  }
}

}  // namespace

int main() {
  TestGoldens();
  TestFailures();
  TestGate();
  return failures == 0 ? 0 : 1;
}
