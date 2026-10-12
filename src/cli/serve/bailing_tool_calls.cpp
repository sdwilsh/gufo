#include "src/cli/serve/bailing_tool_calls.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <set>

#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"

namespace gufo::server {
namespace {

constexpr std::string_view kCallStart = "<tool_call>";
constexpr std::string_view kCallEnd = "</tool_call>";
constexpr std::string_view kKeyStart = "<arg_key>";
constexpr std::string_view kKeyEnd = "</arg_key>";
constexpr std::string_view kValueStart = "<arg_value>";
constexpr std::string_view kValueEnd = "</arg_value>";

using Argument = tokenization::ChatMessage::ToolArgument;

bool IsSpace(char c) {
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

std::string_view Strip(std::string_view value) {
  while (!value.empty() && IsSpace(value.front()))
    value.remove_prefix(1);
  while (!value.empty() && IsSpace(value.back()))
    value.remove_suffix(1);
  return value;
}

std::string Lower(std::string_view value) {
  std::string out(value);
  for (auto& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::optional<json::Value> TryParse(std::string_view text) noexcept {
  try {
    return json::parse(text);
  } catch (...) {
    return std::nullopt;
  }
}

bool Finite(const json::Value& value) {
  if (value.is_number())
    return std::isfinite(value.as_double());
  if (value.is_array())
    return std::ranges::all_of(value.items(),
                               [](const auto& item) { return Finite(item); });
  if (value.is_object())
    return std::ranges::all_of(value.members(), [](const auto& member) {
      return Finite(member.second);
    });
  return true;
}

// vllm.tool_parsers.utils.extract_types_from_schema, plus `const` (a
// finite value has its JSON kind, as enum members do) and local `$ref`s, which
// the Bailing grammar follows too (JsonConstraint::ResolveReference): a
// referenced parameter must not lose its type to the text fallback. A
// reference that points back at the tool's own parameters (a recursive
// schema) sets `unknown`: the value has no definite kind. An unresolvable
// reference adds no type, so the value reads as a string, as the grammar's raw
// text and vLLM do.
constexpr int kMaxSchemaDepth = 16;

void AddValueType(const json::Value& value, std::set<std::string>* types) {
  if (value.is_null())
    types->insert("null");
  else if (value.is_bool())
    types->insert("boolean");
  else if (value.is_number())
    types->insert(std::floor(value.as_double()) == value.as_double()
                      ? "integer"
                      : "number");
  else if (value.is_string())
    types->insert("string");
  else if (value.is_array())
    types->insert("array");
  else if (value.is_object())
    types->insert("object");
}

void CollectTypes(const json::Value& root, const json::Value& schema,
                  std::set<std::string>* types, bool* unknown, int depth = 0) {
  if (!schema.is_object())
    return;
  if (depth > kMaxSchemaDepth) {
    *unknown = true;
    return;
  }
  if (const auto* type = schema.find("type")) {
    if (type->is_string())
      types->insert(type->get_str());
    else if (type->is_array())
      for (const auto& item : type->items())
        if (item.is_string())
          types->insert(item.get_str());
  }
  if (const auto* values = schema.find("enum"); values && values->is_array())
    for (const auto& value : values->items())
      AddValueType(value, types);
  if (const auto* value = schema.find("const"))
    AddValueType(*value, types);
  if (const auto* reference = schema.find("$ref")) {
    const json::Value* target = nullptr;
    try {
      target = sampling::JsonConstraint::ResolveReference(root, *reference);
    } catch (const std::invalid_argument&) {
    }
    // An unresolvable target adds no type: the grammar sends it to raw text.
    if (target == &root)
      *unknown = true;
    else if (target)
      CollectTypes(root, *target, types, unknown, depth + 1);
  }
  for (const char* choice : {"anyOf", "oneOf", "allOf"})
    if (const auto* list = schema.find(choice); list && list->is_array())
      for (const auto& item : list->items())
        CollectTypes(root, item, types, unknown, depth + 1);
}

// The types a value is coerced to. Untyped parameters are text, as in vLLM;
// a parameter of unknown kind gets none, so Coerce reads JSON, then text.
std::set<std::string> SchemaTypes(const json::Value& root,
                                  const json::Value& schema) {
  std::set<std::string> types;
  bool unknown = false;
  CollectTypes(root, schema, &types, &unknown);
  if (types.empty() && !unknown)
    types.insert("string");
  return types;
}

// The finite values (`enum` members, `const`) a parameter admits, through
// `$ref`s and anyOf/oneOf branches. A branch without one makes the set open
// (`open`), and an empty result means the parameter is not finite.
void CollectMembers(const json::Value& root, const json::Value& schema,
                    std::vector<json::Value>* members, bool* open,
                    int depth = 0) {
  if (!schema.is_object() || depth > kMaxSchemaDepth) {
    *open = true;
    return;
  }
  bool finite = false;
  if (const auto* values = schema.find("enum"); values && values->is_array()) {
    finite = true;
    members->insert(members->end(), values->items().begin(),
                    values->items().end());
  }
  if (const auto* value = schema.find("const")) {
    finite = true;
    members->push_back(*value);
  }
  if (const auto* reference = schema.find("$ref")) {
    const json::Value* target = nullptr;
    try {
      target = sampling::JsonConstraint::ResolveReference(root, *reference);
    } catch (const std::invalid_argument&) {
    }
    if (!target || target == &root)
      *open = true;
    else
      CollectMembers(root, *target, members, open, depth + 1);
    return;
  }
  bool branched = false;
  for (const char* choice : {"anyOf", "oneOf"})
    if (const auto* list = schema.find(choice); list && list->is_array()) {
      branched = true;
      for (const auto& item : list->items())
        CollectMembers(root, item, members, open, depth + 1);
    }
  if (!finite && !branched)
    *open = true;
}

// A coerced value that is no member of a finite parameter, where the raw text
// is a string member, is that string: the grammar admits the member as raw
// text, so {"enum": ["42", 17]} reads `42` as "42", not the integer 42.
Argument PreferMember(Argument argument, const json::Value& root,
                      const json::Value& schema, std::string_view raw) {
  if (argument.is_string)
    return argument;
  std::vector<json::Value> members;
  bool open = false;
  CollectMembers(root, schema, &members, &open);
  if (open || members.empty())
    return argument;
  const auto member = [&](const auto& matches) {
    return std::ranges::any_of(members, matches);
  };
  const auto coerced = TryParse(argument.value);
  if (coerced && member([&](const json::Value& value) {
        return value.dump() == coerced->dump();
      }))
    return argument;
  if (member([&](const json::Value& value) {
        return value.is_string() && value.get_str() == raw;
      }))
    return {.name = std::move(argument.name),
            .value = std::string(raw),
            .is_string = true};
  return argument;
}

std::string NormalizeType(std::string_view name) {
  static constexpr std::array<std::pair<std::string_view, std::string_view>, 24>
      kAliases{{{"str", "string"},     {"text", "string"},
                {"varchar", "string"}, {"char", "string"},
                {"enum", "string"},    {"int", "integer"},
                {"int32", "integer"},  {"int64", "integer"},
                {"uint", "integer"},   {"uint32", "integer"},
                {"uint64", "integer"}, {"long", "integer"},
                {"short", "integer"},  {"unsigned", "integer"},
                {"float", "number"},   {"float32", "number"},
                {"float64", "number"}, {"double", "number"},
                {"bool", "boolean"},   {"dict", "object"},
                {"arr", "array"},      {"list", "array"},
                {"sequence", "array"}, {"", ""}}};
  const auto lowered = Lower(Strip(name));
  for (const auto& [alias, canonical] : kAliases)
    if (!alias.empty() && lowered == alias)
      return std::string(canonical);
  return lowered;
}

// Python int(): optional sign and digits, surrounding whitespace allowed.
std::optional<std::string> ParseInteger(std::string_view raw) {
  const auto text = Strip(raw);
  std::size_t i = 0;
  bool negative = false;
  if (i < text.size() && (text[i] == '+' || text[i] == '-'))
    negative = text[i++] == '-';
  const auto digits = text.substr(i);
  if (digits.empty() ||
      !std::ranges::all_of(digits, [](char c) { return c >= '0' && c <= '9'; }))
    return std::nullopt;
  const auto first = digits.find_first_not_of('0');
  if (first == std::string_view::npos)
    return std::string("0");
  return (negative ? "-" : "") + std::string(digits.substr(first));
}

// Python float() restricted to finite decimal spellings. Integral values
// become integers, as `val if val != int(val) else int(val)` does.
std::optional<std::string> ParseNumber(std::string_view raw) {
  const auto text = Strip(raw);
  std::size_t i = 0;
  if (i < text.size() && (text[i] == '+' || text[i] == '-'))
    ++i;
  std::size_t digits = 0;
  while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
    ++i;
    ++digits;
  }
  if (i < text.size() && text[i] == '.') {
    ++i;
    while (i < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[i]))) {
      ++i;
      ++digits;
    }
  }
  if (digits == 0)
    return std::nullopt;
  if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
    ++i;
    if (i < text.size() && (text[i] == '+' || text[i] == '-'))
      ++i;
    std::size_t exponent = 0;
    while (i < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[i]))) {
      ++i;
      ++exponent;
    }
    if (exponent == 0)
      return std::nullopt;
  }
  if (i != text.size())
    return std::nullopt;
  const std::string owned(text);
  const double value = std::strtod(owned.c_str(), nullptr);
  if (!std::isfinite(value))
    return std::nullopt;
  if (std::floor(value) == value) {
    char buffer[400];
    std::snprintf(buffer, sizeof buffer, "%.0f", value);
    return std::string(
        buffer[0] == '-' && buffer[1] == '0' && buffer[2] == 0 ? "0" : buffer);
  }
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof buffer, value);
  return std::string(buffer, result.ptr);
}

Argument FromValue(std::string name, const json::Value& value) {
  if (value.is_string())
    return {
        .name = std::move(name), .value = value.get_str(), .is_string = true};
  return {.name = std::move(name), .value = value.dump(), .is_string = false};
}

// vllm.tool_parsers.utils.coerce_to_schema_type
Argument Coerce(std::string name, std::string_view value,
                const std::set<std::string>& declared) {
  std::set<std::string> types;
  for (const auto& type : declared)
    types.insert(NormalizeType(type));
  const auto as_text = [&] {
    return Argument{
        .name = name, .value = std::string(value), .is_string = true};
  };
  // Numbers travel through json::Value (double), so integers beyond 2^53 lose
  // precision, and ones beyond ~309 digits do not parse at all: those keep
  // their original text as a string instead of losing the key.
  const auto from_json = [&](const std::string& text) {
    auto parsed = TryParse(text);
    return parsed ? FromValue(name, *parsed) : as_text();
  };
  if (types.contains("null") && Lower(value) == "null")
    return {.name = std::move(name), .value = "null", .is_string = false};
  if (types.contains("integer"))
    if (const auto number = ParseInteger(value))
      return from_json(*number);
  if (types.contains("number"))
    if (const auto number = ParseNumber(value))
      return from_json(*number);
  if (types.contains("boolean")) {
    const auto lowered = Lower(Strip(value));
    if (lowered == "true" || lowered == "1")
      return {.name = std::move(name), .value = "true", .is_string = false};
    if (lowered == "false" || lowered == "0")
      return {.name = std::move(name), .value = "false", .is_string = false};
  }
  if (types.contains("object") || types.contains("array")) {
    if (const auto parsed = TryParse(value); parsed && Finite(*parsed))
      return FromValue(name, *parsed);
  }
  if (types.contains("string"))
    return as_text();
  const auto parsed = TryParse(value);
  if (!parsed || !Finite(*parsed))
    return as_text();
  return FromValue(std::move(name), *parsed);
}

// The `</arg_value>` that ends a value starting at `from`. Argument text may
// itself contain closing tags, so a closer counts only when what follows is
// what the format allows next: another `<arg_key>`, the call's `</tool_call>`
// or the end of `text`. Otherwise the first closer wins, as in vLLM, where a
// stray closer ends the value and the rest of the body is ignored.
//
// With `partial` the text may still grow. Then a closer whose continuation is
// not yet known (nothing, or the start of what could become `<arg_key>` or
// `</tool_call>`) does not close the value, and neither does the first-closer
// fallback: a later closer may still qualify.
std::size_t FindValueClose(std::string_view text, std::size_t from,
                           bool partial = false) {
  const auto first = text.find(kValueEnd, from);
  for (auto at = first; at != std::string_view::npos;
       at = text.find(kValueEnd, at + 1)) {
    auto after = at + kValueEnd.size();
    while (after < text.size() && IsSpace(text[after]))
      ++after;
    const auto rest = text.substr(after);
    if (rest.empty()
            ? !partial
            : rest.starts_with(kKeyStart) || rest.starts_with(kCallEnd))
      return at;
    if (partial && (kKeyStart.starts_with(rest) || kCallEnd.starts_with(rest)))
      return std::string_view::npos;
  }
  return partial ? std::string_view::npos : first;
}

// The part of an unfinished value `tail` that cannot change: nothing from the
// first `</arg_value>` on (it may or may not close the value), and no tail
// that could still grow into one or into half a UTF-8 character.
std::string_view StableValuePrefix(std::string_view tail) {
  if (const auto first = tail.find(kValueEnd); first != std::string_view::npos)
    return tail.substr(0, first);
  for (auto size = std::min(tail.size(), kValueEnd.size() - 1); size > 0;
       --size)
    if (kValueEnd.starts_with(tail.substr(tail.size() - size))) {
      tail.remove_suffix(size);
      break;
    }
  std::size_t back = 0;
  while (back < tail.size() && back < 4 &&
         (static_cast<unsigned char>(tail[tail.size() - 1 - back]) & 0xC0) ==
             0x80)
    ++back;
  if (back < tail.size()) {
    const auto lead = static_cast<unsigned char>(tail[tail.size() - 1 - back]);
    const std::size_t need = lead >= 0xF0   ? 4
                             : lead >= 0xE0 ? 3
                             : lead >= 0xC0 ? 2
                                            : 1;
    if (need > back + 1)
      tail.remove_suffix(back + 1);
  }
  return tail;
}

// The `</tool_call>` ending the call whose body starts at `body_begin`: the
// first one outside an argument value. vLLM cuts at the first `</tool_call>`
// anywhere, which loses an argument that holds the closer as text; llama.cpp
// parses the pairs in sequence and keeps it. Walks the same pairs as
// ParseArguments. A value that never closes falls back to the first closer
// only when the model `finished` the output: in output a stop string or the
// token limit cut, that closer is still inside the unfinished value.
std::size_t FindCallEnd(std::string_view text, std::size_t body_begin,
                        bool finished, bool partial = false) {
  std::size_t cursor = body_begin;
  while (true) {
    const auto call_end = text.find(kCallEnd, cursor);
    const auto key_open = text.find(kKeyStart, cursor);
    if (call_end == std::string_view::npos ||
        (key_open == std::string_view::npos || call_end < key_open))
      return call_end;
    const auto key_begin = key_open + kKeyStart.size();
    cursor = key_begin;
    const auto key_close = text.find(kKeyEnd, key_begin);
    if (key_close == std::string_view::npos)
      return call_end;
    auto value_open = key_close + kKeyEnd.size();
    while (value_open < text.size() && IsSpace(text[value_open]))
      ++value_open;
    if (!text.substr(value_open).starts_with(kValueStart))
      continue;
    const auto value_begin = value_open + kValueStart.size();
    const auto value_close = FindValueClose(text, value_begin, partial);
    if (value_close == std::string_view::npos)
      return finished ? text.find(kCallEnd, value_begin)
                      : std::string_view::npos;
    cursor = value_close + kValueEnd.size();
  }
}

struct Parsed {
  std::vector<std::pair<std::string, std::string>> arguments;
  // A key repeated with a different value, as for Qwen calls: the call is
  // invalid rather than letting either value win.
  bool conflict{false};
  // A value opened but never closed.
  bool dangling{false};
  // Partial parsing only: the argument whose value is still being written.
  std::optional<std::pair<std::string, std::string>> pending;
  // True when the body ends after a complete argument and nothing else.
  bool tidy_end{false};
};

// <arg_key>K</arg_key>\s*<arg_value>V</arg_value> pairs in `body`; text that
// does not belong to a pair is ignored, a repeated key with the same value is
// dropped and one with another value makes the call invalid (as for Qwen). Keys
// are stripped as in vLLM, except a key spelled exactly as one of the declared
// `properties` (which may begin or end with spaces) keeps that spelling.
// Cost: every `find` resumes at the cursor and a pair that fails to close
// breaks out, so the scan is linear except that an `<arg_key>` without its
// partner rescans the tail once per such key: O(n * k) for k unmatched keys in
// one call body. Generation length bounds both.
Parsed ParseArguments(std::string_view body, const json::Value* properties,
                      bool partial = false) {
  Parsed parsed;
  std::size_t cursor = 0;
  std::size_t last_end = 0;
  while (true) {
    const auto key_open = body.find(kKeyStart, cursor);
    if (key_open == std::string_view::npos)
      break;
    const auto key_begin = key_open + kKeyStart.size();
    cursor = key_begin;
    const auto key_close = body.find(kKeyEnd, key_begin);
    if (key_close == std::string_view::npos)
      break;
    auto value_open = key_close + kKeyEnd.size();
    while (value_open < body.size() && IsSpace(body[value_open]))
      ++value_open;
    if (!body.substr(value_open).starts_with(kValueStart))
      continue;
    const auto value_begin = value_open + kValueStart.size();
    const auto value_close = FindValueClose(body, value_begin, partial);
    const auto spelled = body.substr(key_begin, key_close - key_begin);
    const std::string key(properties &&
                                  properties->contains(std::string(spelled))
                              ? spelled
                              : Strip(spelled));
    if (value_close == std::string_view::npos) {
      parsed.dangling = true;
      if (partial)
        parsed.pending.emplace(
            key, std::string(StableValuePrefix(body.substr(value_begin))));
      break;
    }
    std::string value(body.substr(value_begin, value_close - value_begin));
    const auto existing = std::ranges::find(
        parsed.arguments, key, [](const auto& item) { return item.first; });
    if (existing != parsed.arguments.end())
      parsed.conflict |= existing->second != value;  // identical: dropped
    else
      parsed.arguments.emplace_back(key, std::move(value));
    cursor = last_end = value_close + kValueEnd.size();
  }
  parsed.tidy_end = last_end > 0 && Strip(body.substr(last_end)).empty();
  return parsed;
}

// The parameters object of a tool schema, through a root `$ref` as the
// grammar reads it.
const json::Value* FindProperties(const json::Value& root) {
  const auto* schema = &root;
  for (int depth = 0; schema->is_object() && schema->contains("$ref");
       ++depth) {
    const json::Value* target = nullptr;
    try {
      target = sampling::JsonConstraint::ResolveReference(
          root, *schema->find("$ref"));
    } catch (const std::invalid_argument&) {
    }
    if (!target || target == schema || depth > kMaxSchemaDepth)
      return nullptr;
    schema = target;
  }
  const auto* properties = schema->find("properties");
  return properties && properties->is_object() && !properties->members().empty()
             ? properties
             : nullptr;
}

}  // namespace

BailingScan ScanBailingToolCalls(
    std::string_view text, std::span<const tokenization::ChatTool> tools,
    bool accept_unterminated, bool emit_undeclared,
    const std::function<bool(std::size_t)>& quoted, bool partial,
    const std::function<void(std::size_t)>& accepted) {
  BailingScan scan;
  std::size_t cursor = 0;
  while (true) {
    const auto begin = text.find(kCallStart, cursor);
    if (begin == std::string_view::npos)
      break;
    const auto body_begin = begin + kCallStart.size();
    if (quoted && quoted(begin)) {
      cursor = body_begin;
      continue;
    }
    const auto close =
        FindCallEnd(text, body_begin, accept_unterminated, partial);
    const bool terminated = close != std::string_view::npos;
    const auto body_end = terminated ? close : text.size();
    const auto end = terminated ? close + kCallEnd.size() : text.size();
    cursor = end;
    const auto body = text.substr(body_begin, body_end - body_begin);

    const auto name_end = body.find(kKeyStart);
    const std::string name(Strip(body.substr(
        0, name_end == std::string_view::npos ? body.size() : name_end)));
    const tokenization::ChatTool* tool = nullptr;
    for (const auto& candidate : tools)
      if (candidate.name == name) {
        tool = &candidate;
        break;
      }
    // Deviation from vLLM, which drops undeclared names: when the caller asks,
    // the call is emitted as generated (arguments stay text) so the client's
    // unknown-tool error path can answer it. Names with markup or newlines
    // are never emitted.
    const bool plausible =
        !name.empty() && name.find_first_of("<>\n\r") == std::string::npos;
    const bool declared =
        plausible && (tools.empty() || tool != nullptr || emit_undeclared);
    std::optional<json::Value> schema;
    if (tool)
      schema = TryParse(tool->parameters_json);
    const auto* properties = schema ? FindProperties(*schema) : nullptr;
    // A call still being written: its name is complete once the first argument
    // opens, and only then is it safe to publish.
    bool streaming = partial && !terminated && declared &&
                     (tool != nullptr || tools.empty()) &&
                     name_end != std::string_view::npos;
    auto parsed = ParseArguments(body, properties, streaming);
    // A call is published only once an argument value has opened or one has
    // finished, not on its name and first key alone. The grammar forces the
    // header tokens, so a length cut within them would otherwise leave an
    // empty, incomplete call in the final output.
    streaming = streaming && (parsed.dangling || !parsed.arguments.empty());
    // A call with no tag at all after its name is `name {}`, as in vLLM.
    const bool bare = name_end == std::string_view::npos;
    const bool complete =
        terminated ||
        (accept_unterminated && (tool != nullptr || tools.empty()) &&
         ((!parsed.arguments.empty() && parsed.tidy_end) || bare));
    // A conflicting repeat, or a value cut off by an early `</tool_call>`,
    // leaves no safe call: a stream may already have published either.
    const bool invalid = parsed.conflict || (terminated && parsed.dangling);
    if (!declared || invalid || !(complete || streaming)) {
      scan.rejected.emplace_back(begin, end);
      continue;
    }
    BailingScan::Call call;
    call.name = name;
    call.begin = begin;
    call.end = end;
    for (auto& [key, value] : parsed.arguments) {
      const auto* property = properties ? properties->find(key) : nullptr;
      if (property && property->is_object())
        call.arguments.push_back(
            PreferMember(Coerce(key, value, SchemaTypes(*schema, *property)),
                         *schema, *property, value));
      else
        call.arguments.push_back(
            {.name = key, .value = std::move(value), .is_string = true});
    }
    if (streaming && !complete) {
      call.partial = true;
      // Only a value that is certainly text can be published before it
      // closes; the others are typed from their final spelling.
      if (parsed.pending) {
        auto& [key, value] = *parsed.pending;
        const auto* property = properties ? properties->find(key) : nullptr;
        const bool text =
            !property || !property->is_object() ||
            SchemaTypes(*schema, *property) == std::set<std::string>{"string"};
        const bool repeated = std::ranges::any_of(
            call.arguments, [&](const auto& item) { return item.name == key; });
        if (text && !repeated)
          call.pending = std::move(parsed.pending);
      }
    }
    scan.calls.push_back(std::move(call));
    if (accepted && !scan.calls.back().partial)
      accepted(end);
  }
  return scan;
}

}  // namespace gufo::server
