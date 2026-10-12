#!/usr/bin/env python3
"""Show where a re-rendered assistant turn diverges from the model's own tokens.

For sample generations in the layout the Bailing V3 template documents, this
parses them the way `gufo serve` does (reasoning split, content, typed calls),
re-renders the turn through the pinned template (jinja2, HF semantics) and
prints the first differing byte for three client behaviours: OpenAI-faithful
(reasoning echoed, arguments in model order), Hermes-style (no reasoning echo,
content stripped, arguments sorted compact JSON) and Hermes with echo.

usage: roundtrip_report.py [TEMPLATE.jinja]
"""
import json
import sys
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = Path(__file__).resolve().parents[3]
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE / "tests/models/ling_3_flash/fixtures/chat_template.jinja"

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.policies["json.dumps_function"] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
env.filters["tojson"] = lambda value, ensure_ascii=True, **kw: json.dumps(value, ensure_ascii=ensure_ascii)
template = env.from_string(TEMPLATE.read_text())

PROMPT_TAIL = "<role>ASSISTANT</role>\n<think>"
ARGS = ("<arg_key>path</arg_key>{n}<arg_value>a.py</arg_value>{n}<arg_key>line</arg_key>{n}"
        "<arg_value>12</arg_value>")


def samples(think_gap, arg_gap):
    """think_gap: text around </think>; arg_gap: separator between argument tags."""
    before, after = think_gap
    return {
        "prose then one call": (f"Check the file first.{before}</think>{after}Looking.\n"
                                f"<tool_call>read_file\n<arg_key>path</arg_key>{arg_gap}"
                                f"<arg_value>a.py</arg_value>\n</tool_call>"),
        "two args": (f"Edit it.{before}</think>{after}<tool_call>edit\n"
                     + ARGS.format(n=arg_gap) + "\n</tool_call>"),
        "answer only": f"Think.{before}</think>{after}The answer is 4.",
        "call inside reasoning": "Need a tool.\n<tool_call>ping\n</tool_call>",
    }


def parse(generated):
    reasoning, _, rest = generated.partition("</think>") if "</think>" in generated else ("", "", generated)
    if "</think>" not in generated and "<tool_call>" in generated:
        reasoning, _, rest = generated.partition("<tool_call>")
        rest = "<tool_call>" + rest
    content, _, calls = rest.partition("<tool_call>")
    out = []
    for body in ([calls.split("</tool_call>")[0]] if calls else []):
        name, _, args = body.partition("\n") if "<arg_key>" not in body.split("\n")[0] else (body.split("<arg_key>")[0], "", "<arg_key>" + body.split("<arg_key>", 1)[1])
        pairs = {}
        while "<arg_key>" in args:
            key, _, args = args.partition("<arg_key>")[2].partition("</arg_key>")
            value, _, args = args.partition("<arg_value>")[2].partition("</arg_value>")
            pairs[key.strip()] = int(value) if value.isdigit() else value
        out.append({"name": name.strip(), "arguments": pairs})
    return reasoning.strip("\n"), content.strip("\n"), out


def render(reasoning, content, calls, echo, hermes):
    message = {"role": "assistant", "content": content.strip() if hermes else content}
    if echo and reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = [{"type": "function", "function": {
            "name": c["name"],
            "arguments": dict(sorted(c["arguments"].items())) if hermes else c["arguments"]}}
            for c in calls]
    return template.render(messages=[{"role": "user", "content": "go"}, message],
                           add_generation_prompt=False)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


for layout, think_gap, arg_gap in (
        ("model writes newlines (\\n</think>\\n\\n, \\n between args)", ("\n", "\n\n"), "\n"),
        ("model writes no separators (</think>content, tags back to back)", ("", ""), "")):
    print(f"##### {layout}")
    for name, generated in samples(think_gap, arg_gap).items():
        reasoning, content, calls = parse(generated)
        actual = ("<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>go"
                  "<|role_end|>" + PROMPT_TAIL + generated + "<|role_end|>")
        print(f"== {name}")
        for label, echo, hermes in (("openai (echo)", True, False),
                                    ("hermes (no echo)", False, True),
                                    ("hermes + echo", True, True)):
            rendered = render(reasoning, content, calls, echo, hermes)
            at = first_diff(rendered, actual)
            if at is None:
                print(f"  {label}: identical")
            else:
                print(f"  {label}: diverges at byte {at}\n    model   : "
                      f"{actual[max(at-12, 0):at+30]!r}\n    rendered: "
                      f"{rendered[max(at-12, 0):at+30]!r}")
