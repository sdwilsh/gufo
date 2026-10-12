#ifndef GUFO_MODELS_LING_3_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_LING_3_FLASH_CHAT_TEMPLATE_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::ling_3_flash {

struct ChatMessage {
  struct ToolArgument {
    std::string name;
    /// Raw text when is_string; otherwise one JSON value (any spacing).
    std::string value;
    bool is_string{true};
  };
  struct ToolCall {
    std::string name;
    std::vector<ToolArgument> arguments;
  };

  /// system, user, assistant or tool. Anything else is rejected.
  std::string role;
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
};

struct ChatTool {
  /// The complete tool object, e.g. {"type":"function","function":{...}}.
  std::string definition_json;
};

struct ChatTemplateOptions {
  /// The template's `enable_thinking` variable; wins over thinking_option.
  std::optional<bool> enable_thinking;
  /// The template's `thinking_option` variable ("on" or "off"; any other
  /// value renders no <think> in the generation prompt). Default "on".
  std::optional<std::string> thinking_option;
  bool add_generation_prompt{true};
  /// When set, receives the byte ranges of the rendered text that are
  /// conversation data (message bodies, reasoning, tool names, arguments, tool
  /// definitions). Pass them to LingTokenizer::Encode as literal ranges so
  /// that only the template's own markup can become control tokens.
  std::vector<std::pair<std::size_t, std::size_t>>* literal_spans{nullptr};
};

/// Renders the pinned Bailing V3 template (see ArtifactTemplateSha256) with
/// the semantics of HF-style Jinja (trim_blocks, lstrip_blocks, tojson with
/// ensure_ascii=False). Throws std::invalid_argument for input the template
/// would render differently or not at all: empty message list, unknown role,
/// malformed or non-canonical tool JSON.
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools = {},
                                     const ChatTemplateOptions& options = {});

/// Python json.dumps(value, ensure_ascii=False) spelling of a JSON text.
/// Throws std::invalid_argument on malformed JSON or on numbers whose Python
/// spelling differs from the input (exponents, trailing fractional zeros).
[[nodiscard]] std::string PythonJsonDumps(std::string_view json);

/// Accepts only an artifact whose architecture is bailingmoe3 and whose
/// embedded chat template has the pinned SHA-256.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg = nullptr);

[[nodiscard]] constexpr std::string_view ChatTemplateId() noexcept {
  return "ling-3-flash-bailing-v3-compiled-v1";
}

[[nodiscard]] constexpr std::string_view ArtifactTemplateSha256() noexcept {
  return "b4ab3a1c8f748e6f874d9aea102333efe7ab82528e8ab81c2eb155851e8705c6";
}

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_CHAT_TEMPLATE_HPP_
