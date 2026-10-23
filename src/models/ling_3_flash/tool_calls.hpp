#ifndef GUFO_MODELS_LING_3_FLASH_TOOL_CALLS_HPP_
#define GUFO_MODELS_LING_3_FLASH_TOOL_CALLS_HPP_

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/ling_3_flash/chat_template.hpp"

namespace gufo::models::ling_3_flash {

/// The literals that frame a Bailing V3 tool call. RenderChat writes historical
/// calls with these and ScanBailingToolCalls reads the same bytes back, so the
/// wire format cannot change in one direction only (#290).
struct BailingToolSyntax {
  static constexpr std::string_view kCallOpen = "<tool_call>";
  static constexpr std::string_view kCallClose = "</tool_call>";
  static constexpr std::string_view kKeyOpen = "<arg_key>";
  static constexpr std::string_view kKeyClose = "</arg_key>";
  static constexpr std::string_view kValueOpen = "<arg_value>";
  static constexpr std::string_view kValueClose = "</arg_value>";
};

/// A declared tool as the scanner reads it: its function name and the JSON
/// schema of its parameters. The views must outlive the scan.
struct BailingTool {
  std::string_view name;
  std::string_view parameters_json;
};

/// Views of `tools`, any sized range of objects with `name` and
/// `parameters_json` (such as the API's tool declarations), for one scan.
template<typename Tools>
[[nodiscard]] std::vector<BailingTool> BailingTools(const Tools& tools) {
  std::vector<BailingTool> declared;
  declared.reserve(tools.size());
  for (const auto& tool : tools)
    declared.push_back(
        {.name = tool.name, .parameters_json = tool.parameters_json});
  return declared;
}

/// Parser for the Bailing V3 (Ling-3.0-flash) tool-call syntax:
///
///   <tool_call>NAME[\n]<arg_key>K</arg_key>[\n]<arg_value>V</arg_value>...
///   [\n]</tool_call>
///
/// Ported from vLLM's `ling3` parser (vllm/parser/ling3.py, which reuses the
/// GLM-4.7 XML engine in vllm/parser/glm47_moe.py and the type coercion in
/// vllm/tool_parsers/utils.py at 84bcbc62), cross-checked with llama.cpp
/// common/parsers/ling3.cpp at 7fe450e19.
struct BailingScan {
  struct Call {
    std::string name;
    std::vector<ChatMessage::ToolArgument> arguments;
    /// Byte range of the call, `<tool_call>` through `</tool_call>`.
    std::size_t begin{0};
    std::size_t end{0};
    /// Partial scans only: the call is still being written. `arguments` holds
    /// the finished ones; `pending` is the key and the settled start of a
    /// text argument whose value is still open.
    bool partial{false};
    std::optional<std::pair<std::string, std::string>> pending;
  };
  /// Valid calls to declared tools, in order of appearance.
  std::vector<Call> calls;
  /// Ranges that opened a call but did not yield one (undeclared name, empty
  /// name, truncated). The caller decides whether they surface as content.
  std::vector<std::pair<std::size_t, std::size_t>> rejected;
};

// Schema typing stays in the scanner, unlike DeepSeek's "schema interpretation
// stays with the API surface": Bailing's string-vs-JSON value framing is a
// property of the dialect. The scanner ports vLLM's coerce_to_schema_type and
// is golden-tested in tests/cli/bailing_tool_calls_test.cpp.
/// Scans `text` for calls. Each `<tool_call>` ends at the first following
/// `</tool_call>` outside an argument value (a value may hold the closer as
/// text; see FindCallEnd). Values are typed from the declared tool's JSON
/// schema as vLLM does: string-only (or untyped) properties keep their raw
/// text; other types are coerced (null, integer, number, boolean, object,
/// array in that priority) and fall back to JSON, then to text. Unlike vLLM,
/// typing follows local `$ref`s (at the root and in properties) and `const`,
/// as the Bailing grammar does, and a recursive or unresolvable reference
/// reads JSON, then text. Keys are stripped unless spelled exactly as a
/// declared property. `accept_unterminated` says the model finished the
/// output (EOS) rather than a stop string or the token limit cutting it. Only
/// then is a call with no closing tag accepted, when it ends right after a
/// complete argument (its unfinished argument is dropped), and only then does
/// a value that never closes end at the next `</tool_call>` (vLLM); in cut
/// output that closer is still inside the value, so the call is unfinished.
/// A cut call with only a name (`<tool_call>name` and nothing else) is
/// accepted the same way, as `name {}`. `emit_undeclared` keeps calls whose
/// name is not among `tools` (a deviation from vLLM); with it unset they land
/// in `rejected`. An unterminated call (cut after its name or an argument) is
/// recovered only for a declared tool: prose that merely mentions the opener
/// must not become a call. An empty `tools` list declares everything.
/// `quoted(position)`, when set, says whether a byte of `text` lies in a
/// quoted example (code span or fence); an opener there is ordinary text.
/// With `partial` (the output is still growing) the last call may be unfinished
/// and is then returned flagged `partial`: it needs a declared name that the
/// first `<arg_key>` has ended, and is never a call the final scan would not
/// accept. Everything it publishes is final: a finished argument, or the part
/// of a text argument that no later byte can change (see StableValuePrefix).
/// `accepted(end)` is called after each finished call with the end of its
/// bytes, so the caller can stop quote state from running across them. A key
/// repeated with another value, or a value cut off by an early `</tool_call>`,
/// rejects the call.
/// Never throws.
[[nodiscard]] BailingScan ScanBailingToolCalls(
    std::string_view text, std::span<const BailingTool> tools,
    bool accept_unterminated, bool emit_undeclared,
    const std::function<bool(std::size_t)>& quoted = {}, bool partial = false,
    const std::function<void(std::size_t)>& accepted = {});

}  // namespace gufo::models::ling_3_flash

#endif  // GUFO_MODELS_LING_3_FLASH_TOOL_CALLS_HPP_
