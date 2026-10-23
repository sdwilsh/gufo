#include "src/models/ling_3_flash/chat_template.hpp"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/crypto/sha256.hpp"
#include "src/models/ling_3_flash/tokenizer.hpp"
#include "src/models/ling_3_flash/tool_calls.hpp"

namespace gufo::models::ling_3_flash {
namespace {

constexpr std::string_view kRoleEnd = "<|role_end|>";

constexpr std::string_view kToolsPreamble =
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within "
    "<tools></tools> "
    "XML tags:\n<tools>";
constexpr std::string_view kToolsEpilogue =
    "\n</tools>\n\nIf none of the functions can be used, point it out. If the "
    "given question lacks the parameters required by the function, also point "
    "it out.\nIf you need to use a function, for each function call, output "
    "the function name and arguments within the following XML "
    "format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n"
    "<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n"
    "<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>\n";

[[noreturn]] void Fail(const std::string& message) {
  throw std::invalid_argument("Ling chat template: " + message);
}

bool Contains(std::string_view text, std::string_view needle) {
  return text.find(needle) != std::string_view::npos;
}

// Python str.lstrip/rstrip/strip with a '\n' strip set.
std::string_view LStripNl(std::string_view s) {
  while (!s.empty() && s.front() == '\n')
    s.remove_prefix(1);
  return s;
}
std::string_view RStripNl(std::string_view s) {
  while (!s.empty() && s.back() == '\n')
    s.remove_suffix(1);
  return s;
}

// ---- Python json.dumps(ensure_ascii=False) re-spelling ----------------------

void AppendJsonString(std::string& out, std::string_view s) {
  out.push_back('"');
  for (const unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

class JsonRespeller {
public:
  explicit JsonRespeller(std::string_view text) : text_(text) {}

  std::string Run() {
    SkipSpace();
    Value(0);
    SkipSpace();
    if (pos_ != text_.size())
      Bad("trailing characters");
    return std::move(out_);
  }

private:
  [[noreturn]] void Bad(const char* why) const {
    Fail(std::string("invalid JSON (") + why + ") at byte " +
         std::to_string(pos_));
  }
  void SkipSpace() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
            text_[pos_] == '\r')) {
      ++pos_;
    }
  }
  bool Eat(char c) {
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }
  void Literal(std::string_view word) {
    if (text_.substr(pos_, word.size()) != word)
      Bad("bad literal");
    pos_ += word.size();
    out_ += word;
  }
  std::uint32_t Hex4() {
    if (pos_ + 4 > text_.size())
      Bad("short \\u escape");
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= c - '0';
      else if (c >= 'a' && c <= 'f')
        v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        v |= c - 'A' + 10;
      else
        Bad("bad \\u escape");
    }
    return v;
  }
  std::string ParseString() {
    std::string s;
    if (!Eat('"'))
      Bad("expected string");
    while (true) {
      if (pos_ >= text_.size())
        Bad("unterminated string");
      const char c = text_[pos_++];
      if (c == '"')
        return s;
      if (static_cast<unsigned char>(c) < 0x20)
        Bad("control character in string");
      if (c != '\\') {
        s.push_back(c);
        continue;
      }
      if (pos_ >= text_.size())
        Bad("unterminated escape");
      const char e = text_[pos_++];
      switch (e) {
        case '"':
          s.push_back('"');
          break;
        case '\\':
          s.push_back('\\');
          break;
        case '/':
          s.push_back('/');
          break;
        case 'b':
          s.push_back('\b');
          break;
        case 'f':
          s.push_back('\f');
          break;
        case 'n':
          s.push_back('\n');
          break;
        case 'r':
          s.push_back('\r');
          break;
        case 't':
          s.push_back('\t');
          break;
        case 'u': {
          std::uint32_t cp = Hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && text_.substr(pos_, 2) == "\\u") {
            const std::size_t save = pos_;
            pos_ += 2;
            const std::uint32_t lo = Hex4();
            if (lo >= 0xDC00 && lo < 0xE000) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else {
              pos_ = save;  // lone high surrogate: Python keeps it as is
            }
          }
          if (cp >= 0xD800 && cp < 0xE000)
            Bad("lone surrogate escape");
          AppendUtf8(s, cp);
          break;
        }
        default:
          Bad("bad escape");
      }
    }
  }
  void Number() {
    const std::size_t begin = pos_;
    Eat('-');
    if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9')
      Bad("bad number");
    if (text_[pos_] == '0') {
      ++pos_;
    } else {
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9')
        ++pos_;
    }
    bool fraction = false;
    if (Eat('.')) {
      fraction = true;
      const std::size_t digits = pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9')
        ++pos_;
      if (pos_ == digits)
        Bad("bad fraction");
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      Fail("number with an exponent has no verbatim Python spelling: " +
           std::string(text_.substr(begin, pos_ - begin)) + "e...");
    }
    const std::string_view lexeme = text_.substr(begin, pos_ - begin);
    // Python prints a float with its shortest repr: x.0 is kept, but 1.50
    // becomes 1.5 and -0 stays -0 only as an int.
    if (fraction && lexeme.back() == '0' && lexeme[lexeme.size() - 2] != '.') {
      Fail("number with trailing fractional zeros: " + std::string(lexeme));
    }
    out_ += lexeme;
  }
  void Value(int depth) {
    if (depth > 128)
      Bad("nesting too deep");
    if (pos_ >= text_.size())
      Bad("unexpected end");
    const char c = text_[pos_];
    if (c == '{') {
      ++pos_;
      out_.push_back('{');
      SkipSpace();
      if (!Eat('}')) {
        while (true) {
          SkipSpace();
          AppendJsonString(out_, ParseString());
          SkipSpace();
          if (!Eat(':'))
            Bad("expected ':'");
          out_ += ": ";
          SkipSpace();
          Value(depth + 1);
          SkipSpace();
          if (Eat('}'))
            break;
          if (!Eat(','))
            Bad("expected ',' or '}'");
          out_ += ", ";
        }
      }
      out_.push_back('}');
    } else if (c == '[') {
      ++pos_;
      out_.push_back('[');
      SkipSpace();
      if (!Eat(']')) {
        while (true) {
          SkipSpace();
          Value(depth + 1);
          SkipSpace();
          if (Eat(']'))
            break;
          if (!Eat(','))
            Bad("expected ',' or ']'");
          out_ += ", ";
        }
      }
      out_.push_back(']');
    } else if (c == '"') {
      AppendJsonString(out_, ParseString());
    } else if (c == 't') {
      Literal("true");
    } else if (c == 'f') {
      Literal("false");
    } else if (c == 'n') {
      Literal("null");
    } else {
      Number();
    }
  }

  std::string_view text_;
  std::size_t pos_{0};
  std::string out_;
};

}  // namespace

std::string PythonJsonDumps(std::string_view json) {
  return JsonRespeller(json).Run();
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       const ChatTemplateOptions& options) {
  if (messages.empty()) {
    Fail("messages is empty (the template reads messages[0])");
  }
  for (const auto& m : messages) {
    if (m.role != "system" && m.role != "user" && m.role != "assistant" &&
        m.role != "tool") {
      Fail("unsupported role '" + m.role + "'");
    }
  }

  // enable_thinking wins over thinking_option.
  std::string thinking = "on";
  if (options.enable_thinking.has_value()) {
    thinking = *options.enable_thinking ? "on" : "off";
  } else if (options.thinking_option.has_value()) {
    thinking = *options.thinking_option;
  }

  std::string out;
  // Conversation data is recorded so the tokenizer never turns text inside it
  // into control tokens (`<|role_end|>` in a tool result must stay text).
  const auto data = [&](std::string_view text) {
    if (options.literal_spans != nullptr && !text.empty()) {
      options.literal_spans->emplace_back(out.size(), out.size() + text.size());
    }
    out += text;
  };
  out += "<role>SYSTEM</role>";
  const bool first_is_system = messages[0].role == "system";
  const std::string_view first_content = messages[0].content;
  const bool marker =
      first_is_system && (Contains(first_content, "detailed thinking on") ||
                          Contains(first_content, "detailed thinking off"));
  if (!tools.empty()) {
    if (first_is_system) {
      data(first_content);
      out += '\n';
    }
    out += kToolsPreamble;
    for (const auto& tool : tools) {
      out += '\n';
      data(PythonJsonDumps(tool.definition_json));
    }
    out += kToolsEpilogue;
    if (!marker) {
      out += "detailed thinking " + thinking;
    }
  } else if (first_is_system) {
    data(first_content);
    if (!marker) {
      out += '\n';
      out += "detailed thinking " + thinking;
    }
  } else {
    out += "detailed thinking " + thinking;
  }
  out += kRoleEnd;

  for (std::size_t i = 0; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    if (i + 1 == messages.size() && options.stable_prefix_bytes != nullptr) {
      *options.stable_prefix_bytes = out.size();
    }
    if (m.role == "user") {
      out += "<role>HUMAN</role>";
      data(m.content);
      out += kRoleEnd;
    } else if (m.role == "system") {
      if (i != 0) {  // the leading system message is folded into the header
        out += "<role>SYSTEM</role>";
        data(m.content);
        out += kRoleEnd;
      }
    } else if (m.role == "assistant") {
      std::string content = m.content;
      std::string reasoning;
      if (!m.reasoning_content.empty()) {
        reasoning = m.reasoning_content;
      } else if (const auto end = content.find("</think>");
                 end != std::string::npos) {
        // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
        std::string_view head =
            RStripNl(std::string_view(content).substr(0, end));
        if (const auto open = head.rfind("<think>");
            open != std::string_view::npos) {
          head.remove_prefix(open + 7);
        }
        reasoning = std::string(LStripNl(head));
        // content.split('</think>')[-1].lstrip('\n')
        const auto last = content.rfind("</think>");
        content =
            std::string(LStripNl(std::string_view(content).substr(last + 8)));
      }
      // preserved_thinking is hard-wired true in the template.
      if (!reasoning.empty()) {
        out += "<role>ASSISTANT</role>\n<think>";
        data(LStripNl(RStripNl(reasoning)));
        out += "</think>";
        data(LStripNl(content));
      } else {
        out += "<role>ASSISTANT</role>\n<think></think>";
        data(content);
      }
      for (std::size_t k = 0; k < m.tool_calls.size(); ++k) {
        const auto& call = m.tool_calls[k];
        if (k != 0 || !content.empty()) {
          out += '\n';
        }
        out += BailingToolSyntax::kCallOpen;
        data(call.name);
        out += '\n';
        for (const auto& arg : call.arguments) {
          out += BailingToolSyntax::kKeyOpen;
          data(arg.name);
          out += BailingToolSyntax::kKeyClose;
          out += '\n';
          out += BailingToolSyntax::kValueOpen;
          data(arg.is_string ? arg.value : PythonJsonDumps(arg.value));
          out += BailingToolSyntax::kValueClose;
        }
        out += '\n';
        out += BailingToolSyntax::kCallClose;
      }
      out += kRoleEnd;
    } else {  // tool
      if (i == 0 || messages[i - 1].role != "tool") {
        out += "<role>OBSERVATION</role>";
      }
      out += "\n<tool_response>\n";
      data(m.content);
      out += "\n</tool_response>";
      if (i + 1 == messages.size() || messages[i + 1].role != "tool") {
        out += kRoleEnd;
      }
    }
  }

  if (options.add_generation_prompt) {
    out += "<role>ASSISTANT</role>";
    if (thinking == "on") {
      out += "\n<think>";
    } else if (thinking == "off") {
      out += "\n<think></think>";
    }
  }
  return out;
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  const auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return false;
  };
  if (reader.GetMetadataString("general.architecture") !=
      std::optional<std::string_view>{"bailingmoe3"}) {
    return fail("Ling artifact has unexpected architecture");
  }
  const auto raw = reader.GetMetadataString("tokenizer.chat_template");
  if (!raw.has_value() || raw->empty()) {
    return fail("Ling GGUF is missing tokenizer.chat_template metadata");
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(raw->data());
  const std::string hash = crypto::Sha256Hex({bytes, raw->size()});
  if (hash != ArtifactTemplateSha256()) {
    return fail(
        "Ling GGUF chat template SHA-256 is not the pinned Bailing V3 "
        "template: " +
        hash);
  }
  return true;
}

}  // namespace gufo::models::ling_3_flash
