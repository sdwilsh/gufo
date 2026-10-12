// Golden tests for the Bailing V3 (Ling-3.0-flash) tool-call scanner. Cases
// 1-3 mirror vLLM tests/parser/engine/test_ling3.py; the rest follow its
// glm47 coercion rules (vllm/tool_parsers/utils.py coerce_to_schema_type).
#include "src/cli/serve/bailing_tool_calls.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace {

using gufo::server::BailingScan;
using gufo::server::ScanBailingToolCalls;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

gufo::tokenization::ChatTool Tool(std::string name, std::string parameters) {
  gufo::tokenization::ChatTool tool;
  tool.name = std::move(name);
  tool.parameters_json = std::move(parameters);
  return tool;
}

std::vector<gufo::tokenization::ChatTool> Tools() {
  return {
      Tool("get_weather",
           R"({"type":"object","properties":{"city":{"type":"string"}}})"),
      Tool("ping", R"({"type":"object","properties":{}})"),
      Tool("typed", R"({"type":"object","properties":{
        "text":{"type":"string"},
        "n":{"type":"integer"},
        "x":{"type":"number"},
        "flag":{"type":"boolean"},
        "none":{"type":"null"},
        "maybe":{"type":["string","null"]},
        "obj":{"type":"object"},
        "arr":{"type":"array"},
        "any":{"anyOf":[{"type":"integer"},{"type":"string"}]},
        "choice":{"enum":[1,2,3]},
        "untyped":{"description":"no type"},
        "alias":{"type":"int"}}})"),
  };
}

std::string Arguments(const BailingScan::Call& call) {
  auto object = gufo::json::Value::object();
  for (const auto& argument : call.arguments)
    object[argument.name] = argument.is_string
                                ? gufo::json::Value(argument.value)
                                : gufo::json::parse(argument.value);
  return object.dump();
}

BailingScan Scan(const std::string& text, bool unterminated = false,
                 bool undeclared = false) {
  const auto tools = Tools();
  return ScanBailingToolCalls(text, tools, unterminated, undeclared);
}

std::string One(const std::string& name, const std::string& pairs) {
  return "<tool_call>" + name + pairs + "\n</tool_call>";
}

std::string Pair(const std::string& key, const std::string& value) {
  return "<arg_key>" + key + "</arg_key>\n<arg_value>" + value + "</arg_value>";
}

void TestVllmGoldens() {
  auto scan = Scan(
      "<tool_call>get_weather<arg_key>city</arg_key>"
      "<arg_value>Beijing</arg_value></tool_call>");
  Expect(scan.calls.size() == 1 && scan.calls[0].name == "get_weather" &&
             Arguments(scan.calls[0]) == R"({"city":"Beijing"})",
         "no newline between tags");
  scan = Scan("<tool_call>ping</tool_call>");
  Expect(scan.calls.size() == 1 && Arguments(scan.calls[0]) == "{}",
         "call without arguments");
  // Template rendering: name, newline, key, value on the next line.
  scan = Scan(
      "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>"
      "Rome</arg_value>\n</tool_call>");
  Expect(scan.calls.size() == 1 &&
             Arguments(scan.calls[0]) == R"({"city":"Rome"})",
         "template spelling");
}

void TestTyping() {
  const auto scan = Scan(
      One("typed", Pair("text", "007") + Pair("n", " 42 ") + Pair("x", "1.50") +
                       Pair("flag", "True") + Pair("none", "NULL") +
                       Pair("maybe", "null") + Pair("obj", R"({"a": [1, 2]})") +
                       Pair("arr", "[1,\"b\"]") + Pair("any", "12") +
                       Pair("choice", "2") + Pair("untyped", "5") +
                       Pair("alias", "9") + Pair("unknown", "7")));
  Expect(scan.calls.size() == 1, "typed call parsed");
  const auto args = gufo::json::parse(Arguments(scan.calls[0]));
  Expect(args.find("text")->is_string() && args.member_str("text") == "007",
         "string schema keeps raw text");
  Expect(args.find("n")->is_number() && args.member_double("n") == 42,
         "integer");
  Expect(args.find("x")->is_number() && args.member_double("x") == 1.5,
         "number");
  Expect(args.find("flag")->is_bool() && args.find("flag")->as_bool(),
         "boolean accepts True");
  Expect(args.find("none")->is_null(), "null accepts any case");
  Expect(args.find("maybe")->is_null(), "string|null prefers null");
  Expect(args.find("obj")->is_object() && args.find("arr")->is_array(),
         "object and array parse as JSON");
  Expect(args.find("any")->is_number(), "anyOf unions use integer first");
  Expect(args.find("choice")->is_number(), "enum infers integer");
  Expect(args.find("untyped")->is_string() && args.member_str("untyped") == "5",
         "typeless property is a string");
  Expect(args.find("alias")->is_number(), "vLLM type aliases are honoured");
  Expect(args.find("unknown")->is_string(), "unknown key stays a string");
  // Failed coercion falls back to JSON, then text.
  const auto bad = Scan(One(
      "typed", Pair("n", "many") + Pair("flag", "maybe") + Pair("x", "inf")));
  const auto fallback = gufo::json::parse(Arguments(bad.calls.at(0)));
  Expect(fallback.find("n")->is_string() &&
             fallback.find("flag")->is_string() &&
             fallback.find("x")->is_string(),
         "uncoercible values stay text");
}

void TestStructure() {
  // Values are raw: newlines and angle brackets survive; keys are stripped.
  auto scan = Scan(One("get_weather",
                       "<arg_key> city </arg_key>\n<arg_value>\n a < b \n"
                       "</arg_value>"));
  Expect(scan.calls.size() == 1 &&
             Arguments(scan.calls[0]) == "{\"city\":\"\\n a < b \\n\"}",
         "values keep whitespace");
  // Repeated key: first position, last value.
  scan =
      Scan(One("typed", Pair("n", "1") + Pair("text", "a") + Pair("n", "2")));
  Expect(Arguments(scan.calls.at(0)) == R"({"n":2,"text":"a"})",
         "duplicate key semantics");
  // Parallel calls, spans and surrounding prose.
  const std::string first = One("ping", "");
  const std::string second = One("get_weather", Pair("city", "Oslo"));
  const std::string text = "A" + first + "B" + second + "C";
  scan = Scan(text);
  Expect(scan.calls.size() == 2 && scan.calls[0].begin == 1 &&
             scan.calls[0].end == 1 + first.size() &&
             scan.calls[1].begin == 2 + first.size() &&
             scan.calls[1].end == text.size() - 1,
         "call byte spans");
  // Undeclared names are rejected, not executed.
  scan = Scan(One("rm_rf", Pair("path", "/")));
  Expect(scan.calls.empty() && scan.rejected.size() == 1, "undeclared tool");
  // A <tool_call> inside an argument value is data only up to the first
  // closing tag, exactly like the reference lexer.
  scan = Scan(One("get_weather", Pair("city", "x<tool_call>y")));
  Expect(scan.calls.size() == 1 &&
             Arguments(scan.calls[0]) == R"({"city":"x<tool_call>y"})",
         "opening tag inside value");
}

void TestTruncation() {
  const std::string partial =
      "<tool_call>get_weather\n" + Pair("city", "Rome") + "\n";
  Expect(Scan(partial).calls.empty() && Scan(partial).rejected.size() == 1,
         "unterminated call is rejected by default");
  auto scan = Scan(partial, true);
  Expect(scan.calls.size() == 1 &&
             Arguments(scan.calls[0]) == R"({"city":"Rome"})",
         "clean stop after a complete argument keeps the call");
  scan = Scan("<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Ro",
              true);
  Expect(scan.calls.empty(), "cut mid-value is never executed");
  scan = Scan("<tool_call>get_wea", true);
  Expect(scan.calls.empty(), "cut mid-name of an undeclared tool is dropped");
  scan = Scan("<tool_call>ping\n", true);
  Expect(scan.calls.size() == 1 && scan.calls[0].name == "ping" &&
             Arguments(scan.calls[0]) == "{}",
         "a bare name after a clean stop is `name {}`");
  Expect(Scan("<tool_call>ping").calls.empty(),
         "a bare name after a length stop is not a call");
  // Prose that merely names the opener is not a call, even at a clean stop
  // with undeclared calls allowed.
  scan = Scan("<tool_call>ping</tool_call>\nI used <tool_call> above.", true,
              true);
  Expect(scan.calls.size() == 1 && scan.calls[0].name == "ping" &&
             scan.rejected.size() == 1,
         "an unterminated bare undeclared name is not recovered");
  scan = Scan("<tool_call>nope\n" + Pair("a", "b") + "\n", true, true);
  Expect(scan.calls.empty() && scan.rejected.size() == 1,
         "an unterminated undeclared call with arguments is not recovered");
  // Openers inside quoted examples are skipped when the caller says so.
  {
    const std::string text =
        "<tool_call>ping</tool_call>\nUse `<tool_call>` next time.";
    const auto tools = Tools();
    const auto quoted = [&](std::size_t at) {
      const auto open = text.find("`<tool_call>`");
      return at > open && at < open + 13;
    };
    const auto scan_quoted =
        ScanBailingToolCalls(text, tools, true, true, quoted);
    Expect(scan_quoted.calls.size() == 1 && scan_quoted.rejected.empty(),
           "a quoted opener is neither a call nor a rejected range");
  }
  scan = Scan("<tool_call></tool_call>");
  Expect(scan.calls.empty(), "empty name is rejected");
  scan = Scan("no calls here <arg_key>x</arg_key>");
  Expect(scan.calls.empty() && scan.rejected.empty(), "plain prose");
}

// A closer spelled inside an argument value is argument text (llama.cpp parses
// the pairs in order; vLLM's first-closer cut would lose the argument).
void TestCloserInsideValue() {
  const std::string literal = "<think>literal</think></tool_call>";
  auto scan = Scan("<tool_call>get_weather<arg_key>city</arg_key><arg_value>" +
                   literal + "</arg_value></tool_call>");
  Expect(scan.calls.size() == 1 && scan.rejected.empty() &&
             Arguments(scan.calls[0]) ==
                 R"({"city":"<think>literal</think></tool_call>"})",
         "a closer inside a value is argument text");
  // Two calls: the first call's value keeps the closer, the second is intact.
  const std::string first = One("get_weather", Pair("city", literal));
  scan = Scan(first + "\nthen\n" + One("ping", ""));
  Expect(scan.calls.size() == 2 && scan.calls[0].end == first.size() &&
             Arguments(scan.calls[0]) ==
                 R"({"city":"<think>literal</think></tool_call>"})" &&
             scan.calls[1].name == "ping",
         "calls after a closer-in-value call are kept");
  // Several arguments, closers in a middle value.
  scan = Scan(One("typed", Pair("text", "a</tool_call>b") + Pair("n", "3")));
  Expect(scan.calls.size() == 1 &&
             Arguments(scan.calls[0]) == R"({"text":"a</tool_call>b","n":3})",
         "closer in a middle value");
  // A value that never closes still ends at the first closer (vLLM) when
  // the model finished the output.
  const std::string unclosed =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>x"
      "</tool_call>";
  scan = Scan(unclosed, true);
  Expect(scan.calls.size() == 1 && scan.calls[0].arguments.empty() &&
             scan.rejected.empty(),
         "unterminated value falls back to the first closer");
  // Cut by a stop string or the token limit, that closer is value text: the
  // call is unfinished, even when the value quotes a whole call.
  scan = Scan(unclosed);
  Expect(scan.calls.empty() && scan.rejected.size() == 1,
         "a cut value does not end at a closer inside it");
  scan = Scan(
      "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>"
      "<tool_call><function=record><parameter=content>AAAA</parameter>"
      "</function></tool_call> ");
  Expect(scan.calls.empty() && scan.rejected.size() == 1,
         "a call quoted in a cut value is neither a call nor ends one");
  // A literal closer for the value is text too when more follows it.
  scan = Scan(
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>a"
      "</arg_value>b</arg_value></tool_call>");
  Expect(scan.calls.size() == 1 &&
             Arguments(scan.calls[0]) == R"({"city":"a</arg_value>b"})",
         "value closer followed by text is argument text");
}

// Typing follows the schema the grammar enforced: root and property
// references (with JSON pointer escapes), `const`, and recursive references.
void TestReferencesAndConst() {
  const std::vector<gufo::tokenization::ChatTool> tools{
      Tool("record", R"({"$ref":"#/$defs/Arguments","$defs":{
        "number/type~":{"type":"integer","enum":[42]},
        "b":{"type":"boolean"},
        "s":{"type":"string","enum":["42"]},
        "z":{"type":"null"},
        "a":{"type":"array","items":{"type":"integer"}},
        "Arguments":{"type":"object","properties":{
          "n":{"$ref":"#/$defs/number~1type~0"},
          "b":{"$ref":"#/$defs/b"},
          "s":{"$ref":"#/$defs/s"},
          "z":{"$ref":"#/$defs/z"},
          "a":{"$ref":"#/$defs/a"},
          "missing":{"$ref":"#/$defs/nowhere"}}}}})"),
      Tool("finite", R"({"type":"object","properties":{
        "text":{"const":"42"},
        "union":{"anyOf":[{"const":"42"},{"type":"null"}]},
        "mixed":{"enum":["42",17]},
        "node":{"$ref":"#"}}})"),
  };
  auto scan = ScanBailingToolCalls(
      One("record", Pair("n", "42") + Pair("b", "true") + Pair("s", "42") +
                        Pair("z", "null") + Pair("a", "[1]") +
                        Pair("missing", "[2]")),
      tools, false, false);
  Expect(scan.calls.size() == 1 &&
             Arguments(scan.calls[0]) ==
                 R"({"n":42,"b":true,"s":"42","z":null,"a":[1],"missing":[2]})",
         "referenced parameters keep their JSON types");
  scan = ScanBailingToolCalls(
      One("finite", Pair("text", "42") + Pair("union", "42") +
                        Pair("mixed", "17") + Pair("node", "5")),
      tools, false, false);
  Expect(scan.calls.size() == 1 &&
             Arguments(scan.calls[0]) ==
                 R"({"text":"42","union":"42","mixed":17,"node":5})",
         "const has its value's kind; a recursive reference reads JSON");
  // A raw string member of a mixed enum stays that string; a coerced value
  // that is a member keeps its kind.
  scan = ScanBailingToolCalls(
      One("finite", Pair("mixed", "42") + Pair("union", "null")), tools, false,
      false);
  Expect(Arguments(scan.calls.at(0)) == R"({"mixed":"42","union":null})",
         "a string enum member is not coerced to another kind");
  scan = ScanBailingToolCalls(One("finite", Pair("node", "leaf")), tools, false,
                              false);
  Expect(Arguments(scan.calls.at(0)) == R"({"node":"leaf"})",
         "a recursive reference falls back to text");
}

void TestUndeclaredAndHuge() {
  const std::string rogue = One("rm_rf", Pair("path", "/"));
  auto scan = Scan(rogue, false, true);
  Expect(scan.calls.size() == 1 && scan.calls[0].name == "rm_rf" &&
             Arguments(scan.calls[0]) == R"({"path":"/"})" &&
             scan.rejected.empty(),
         "undeclared call is emitted as generated when asked");
  Expect(Scan("<tool_call>a<b</tool_call>", false, true).calls.empty(),
         "names with markup are never emitted");
  // Integers beyond double range keep their text instead of losing the key.
  const std::string huge(400, '9');
  scan = Scan(
      One("typed", Pair("n", huge) + Pair("x", "1" + std::string(400, '0'))));
  Expect(scan.calls.size() == 1 && scan.calls[0].arguments.size() == 2 &&
             scan.calls[0].arguments[0].name == "n" &&
             scan.calls[0].arguments[0].is_string &&
             scan.calls[0].arguments[0].value == huge &&
             scan.calls[0].arguments[1].name == "x",
         "unparseable numbers fall back to text with their key");
  scan = Scan(One("typed", Pair("n", "9007199254740993")));
  Expect(scan.calls.at(0).arguments.at(0).is_string == false,
         "integers beyond 2^53 are numbers (precision is double's)");
}

}  // namespace

int main() {
  TestUndeclaredAndHuge();
  TestCloserInsideValue();
  TestVllmGoldens();
  TestTyping();
  TestStructure();
  TestTruncation();
  TestReferencesAndConst();
  std::cout << "bailing_tool_calls_test passed\n";
  return 0;
}
