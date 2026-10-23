#!/usr/bin/env python3
"""Generate tests/models/ling_3_flash/fixtures/chat_template_goldens.json.

Renders the GGUF's "Bailing V3" template with jinja2 the way Hugging Face
transformers does (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks,
tojson = json.dumps(ensure_ascii=False)) and stores both the HF-shaped
conversation and the flat form the C++ renderer takes.

usage: make_chat_goldens.py TEMPLATE.jinja OUT.json
"""
import json
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. Zürich"},
        "days": {"type": "integer", "minimum": 1, "default": 3},
        "units": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "description": "Search \"the web\"\nsecond line\t<tab> & more",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}

U = lambda c: {"role": "user", "content": c}
S = lambda c: {"role": "system", "content": c}
A = lambda c, **kw: {"role": "assistant", "content": c, **kw}
T = lambda c: {"role": "tool", "content": c}


def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


CASES = [
    ("plain-gen", [U("Hello!")], [], {}, True),
    ("plain-nogen", [U("Hello!")], [], {}, False),
    ("system-gen", [S("You are a helpful assistant."), U("Hi")], [], {}, True),
    ("system-with-thinking-marker", [S("Be brief. detailed thinking off"), U("Hi")], [], {}, True),
    ("thinking-off", [S("Be brief."), U("Hi")], [], {"enable_thinking": False}, True),
    ("thinking-on-explicit", [U("Hi")], [], {"enable_thinking": True}, True),
    ("thinking-option-off", [U("Hi")], [], {"thinking_option": "off"}, True),
    ("thinking-option-other", [U("Hi")], [], {"thinking_option": "auto"}, True),
    ("enable-thinking-wins", [U("Hi")], [], {"enable_thinking": False, "thinking_option": "on"}, True),
    ("multi-turn", [S("Sys"), U("What is 2+2?"), A("4"), U("And 3+3?"), A("6"), U("Thanks")], [], {}, True),
    ("assistant-reasoning-field",
     [U("Q1"), A("A1", reasoning_content="\n\nthinking hard\n"), U("Q2")], [], {}, True),
    ("assistant-think-in-content",
     [U("Q1"), A("<think>\nlet me see\n</think>\n\nanswer one"), U("Q2")], [], {}, True),
    ("assistant-think-multiple",
     [U("Q"), A("pre<think>a</think>b</think>\n\nc")], [], {}, False),
    ("assistant-empty-content", [U("Q"), A(""), U("Q2")], [], {}, True),
    ("system-mid-conversation", [U("a"), S("later system"), A("b"), U("c")], [], {}, True),
    ("tools-system", [S("You are an agent."), U("Weather in Paris?")], [WEATHER, SEARCH], {}, True),
    ("tools-no-system", [U("Weather in Paris?")], [WEATHER], {}, True),
    ("tools-system-thinking-marker", [S("detailed thinking on"), U("x")], [WEATHER], {}, True),
    ("tools-thinking-off", [U("x")], [WEATHER], {"enable_thinking": False}, True),
    ("tool-roundtrip",
     [S("Agent."), U("Weather in Paris?"),
      A("", tool_calls=[call("get_weather", city="Paris", days=2, units="c")]),
      T("{\"temp\": 12, \"sky\": \"rain\"}"),
      A("It is 12 degrees and raining.")],
     [WEATHER], {}, True),
    ("tool-roundtrip-reasoning",
     [U("Weather in Paris?"),
      A("Checking.", reasoning_content="need the tool",
        tool_calls=[call("get_weather", city="Paris")]),
      T("sunny")],
     [WEATHER], {}, True),
    ("tool-parallel-calls",
     [U("Compare."),
      A("Let me look.", tool_calls=[call("get_weather", city="Paris"), call("get_weather", city="Rome", days=1)]),
      T("sunny"), T("cloudy"),
      A("Paris sunny, Rome cloudy."), U("ok")],
     [WEATHER], {}, False),
    ("tool-args-types",
     [U("go"),
      A("", tool_calls=[call("search", q="line1\nline2 \"quoted\" ü ✓", opts={"deep": True, "n": [1, 2.5, None], "name": "Zoë   \x01"},
                             flag=False, count=7, ratio=0.25, nothing=None, empty="")])],
     [SEARCH], {}, True),
    ("tool-call-no-args", [U("go"), A("", tool_calls=[call("noop")])], [SEARCH], {}, False),
    ("tool-call-content-and-calls",
     [U("go"), A("Sure.\n\n", tool_calls=[call("a", x="1"), call("b", y="2")])], [SEARCH], {}, True),
    ("tool-result-multiline", [U("go"), A("", tool_calls=[call("a")]), T("line1\nline2\n")], [], {}, True),
    ("unicode-content", [S("Système ✓"), U("日本語のテキスト 🚀 <role>HUMAN</role> {{ x }} {% y %}")], [], {}, True),
    ("last-query-after-tool",
     [U("one"), A("", reasoning_content="r1", tool_calls=[call("a")]), T("res"), A("two", reasoning_content="r2"), U("three")],
     [SEARCH], {}, True),
]


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def to_flat(msg):
    out = {"role": msg["role"], "content": msg.get("content", ""),
           "reasoning_content": msg.get("reasoning_content", ""), "tool_calls": []}
    for tc in msg.get("tool_calls", []):
        fn = tc["function"]
        args = []
        for k, v in fn["arguments"].items():
            if isinstance(v, str):
                args.append({"name": k, "value": v, "is_string": True})
            else:
                args.append({"name": k, "value": json.dumps(v, ensure_ascii=False, separators=(",", ":")),
                             "is_string": False})
        out["tool_calls"].append({"name": fn["name"], "arguments": args})
    return out


def main():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    template = env.from_string(open(sys.argv[1], encoding="utf-8").read())
    cases = []
    for name, messages, tools, opts, gen in CASES:
        kwargs = dict(opts)
        expected = template.render(messages=messages, tools=tools or None,
                                   add_generation_prompt=gen, **kwargs)
        cases.append({
            "name": name,
            "messages": [to_flat(m) for m in messages],
            "tools": [json.dumps(t, ensure_ascii=False, separators=(",", ":")) for t in tools],
            "enable_thinking": opts.get("enable_thinking"),
            "thinking_option": opts.get("thinking_option"),
            "add_generation_prompt": gen,
            "expected": expected,
        })
    json.dump({"reference": f"jinja2 {jinja2.__version__} HF-style sandbox, trim_blocks, lstrip_blocks",
               "cases": cases}, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
