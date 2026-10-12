#ifndef GUFO_SERVER_BAILING_TOOL_CALLS_HPP_
#define GUFO_SERVER_BAILING_TOOL_CALLS_HPP_

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {

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
    std::vector<tokenization::ChatMessage::ToolArgument> arguments;
    /// Byte range of the call, `<tool_call>` through `</tool_call>`.
    std::size_t begin{0};
    std::size_t end{0};
  };
  /// Valid calls to declared tools, in order of appearance.
  std::vector<Call> calls;
  /// Ranges that opened a call but did not yield one (undeclared name, empty
  /// name, truncated). The caller decides whether they surface as content.
  std::vector<std::pair<std::size_t, std::size_t>> rejected;
};

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
/// Never throws.
[[nodiscard]] BailingScan ScanBailingToolCalls(
    std::string_view text, std::span<const tokenization::ChatTool> tools,
    bool accept_unterminated, bool emit_undeclared,
    const std::function<bool(std::size_t)>& quoted = {});

}  // namespace gufo::server

#endif  // GUFO_SERVER_BAILING_TOOL_CALLS_HPP_
