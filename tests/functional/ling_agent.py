"""Ling-3.0-flash agentic checks against a running `gufo serve` (stdlib only).

Requests mimic the production client (Hermes Agent through a gateway): streaming
Chat Completions with include_usage, top-level reasoning_effort, no sampler
values, tool arguments re-sent as sorted compact ASCII JSON, assistant content
stripped and reasoning NOT echoed. A variant echoes reasoning_content. Every
tool result the model's calls produce is executed in a disposable fixture and
printed, so a reviewer can read what the parser delivered.

    python3 ling_agent.py --base-url http://127.0.0.1:8080 --model Ling-3.0-flash \
        --output /work/ling-agent
"""

import argparse
import http.client
import json
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a UTF-8 text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "First line, 1-based"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Replace a file with new contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "terminal", "description": "Run a shell command in the project directory.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
            "background": {"type": "boolean"}},
            "required": ["command"]}}},
]

SYSTEM = ("You are a coding agent working in a project directory. Use the tools to "
          "inspect, edit and verify. When the task is verified, answer DONE.")


class Client:
    def __init__(self, base_url, model):
        url = urllib.parse.urlsplit(base_url)
        self.host, self.port, self.model = url.hostname, url.port or 80, model
        self.requests = []

    def chat(self, body, stream=True):
        body = {"model": self.model, "stream": stream, **body}
        if stream:
            body["stream_options"] = {"include_usage": True}
        conn = http.client.HTTPConnection(self.host, self.port, timeout=600)
        started = time.monotonic()
        conn.request("POST", "/v1/chat/completions", json.dumps(body),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        record = {"request": body, "status": response.status,
                  "http_request_id": response.getheader("X-Request-ID")}
        self.requests.append(record)
        if response.status != 200:
            record["error"] = response.read().decode("utf-8", "replace")
            raise AssertionError(f"HTTP {response.status}: {record['error'][:500]}")
        result = {"content": "", "reasoning": "", "calls": [], "finish": None,
                  "usage": None, "chunks": 0}
        if not stream:
            message = json.loads(response.read())
            choice = message["choices"][0]
            result.update(content=choice["message"].get("content") or "",
                          reasoning=choice["message"].get("reasoning_content") or "",
                          finish=choice["finish_reason"], usage=message.get("usage"))
            result["calls"] = [{"id": c["id"], "index": i, "name": c["function"]["name"],
                                "arguments": c["function"]["arguments"]}
                               for i, c in enumerate(choice["message"].get("tool_calls") or [])]
        else:
            slots = {}
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                result["chunks"] += 1
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    result["content"] += delta.get("content") or ""
                    result["reasoning"] += delta.get("reasoning_content") or ""
                    for call in delta.get("tool_calls") or []:
                        slot = slots.setdefault(call["index"], {
                            "id": call.get("id"), "index": call["index"], "name": "",
                            "arguments": ""})
                        assert not slot["id"] or not call.get("id") or slot["id"] == call["id"]
                        slot["id"] = slot["id"] or call.get("id")
                        function = call.get("function") or {}
                        slot["name"] += function.get("name") or ""
                        slot["arguments"] += function.get("arguments") or ""
                    result["finish"] = choice.get("finish_reason") or result["finish"]
            result["calls"] = [slots[k] for k in sorted(slots)]
        result["seconds"] = round(time.monotonic() - started, 3)
        record["response"] = result
        return result


def canonical(arguments):
    """Hermes re-serialises tool-call arguments before every request."""
    return json.dumps(json.loads(arguments), separators=(",", ":"), sort_keys=True)


def assistant_message(result, echo):
    message = {"role": "assistant", "content": result["content"].strip(),
               "tool_calls": [{"id": c["id"], "type": "function", "function": {
                   "name": c["name"], "arguments": canonical(c["arguments"])}}
                              for c in result["calls"]]}
    if not message["tool_calls"]:
        del message["tool_calls"]
    if echo and result["reasoning"].strip():
        message["reasoning_content"] = result["reasoning"]
    return message


def run_tool(root, call):
    """Execute one call in the fixture; return the text the model sees."""
    try:
        args = json.loads(call["arguments"])
        if call["name"] == "read_file":
            lines = (root / args["path"]).read_text().splitlines(True)
            return "".join(lines[max(int(args.get("offset", 1)) - 1, 0):])
        if call["name"] == "write_file":
            (root / args["path"]).write_text(args["content"])
            return f"wrote {len(args['content'])} bytes to {args['path']}"
        if call["name"] == "terminal":
            done = subprocess.run(args["command"], shell=True, cwd=root, text=True,
                                  capture_output=True, timeout=min(int(args.get("timeout", 30)), 60))
            return f"exit {done.returncode}\n{done.stdout}{done.stderr}"
        return f"unknown tool {call['name']}"
    except Exception as error:  # the model must see failures like a real agent would
        return f"error: {error!r}"


FIXTURE = {
    "calc.py": "def add(a, b):\n    return a - b\n\n\ndef scale(values, factor):\n    return [v * factor for v in values]\n",
    "check.py": "from calc import add, scale\nassert add(2, 3) == 5, add(2, 3)\nassert scale([1, 2], 3) == [3, 6]\nprint('ok')\n",
}
TASK = ("calc.py has a bug. Run `python3 check.py` to see it, read calc.py, fix it with "
        "write_file, then re-run check.py to verify, and answer DONE.")


def agent_loop(client, name, echo, effort, report, max_turns=10, **extra):
    root = Path(tempfile.mkdtemp(prefix="ling-agent-"))
    for path, text in FIXTURE.items():
        (root / path).write_text(text)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    turns, calls_seen = [], 0
    for turn in range(max_turns):
        result = client.chat({"messages": messages, "tools": TOOLS, "reasoning_effort": effort,
                              **extra})
        turn_info = {"turn": turn, "finish": result["finish"], "content": result["content"],
                     "reasoning_chars": len(result["reasoning"]), "usage": result["usage"],
                     "calls": []}
        assert result["usage"], "streaming usage chunk missing"
        assert "<tool_call>" not in result["content"] + result["reasoning"], \
            f"tool markup leaked into text: {result['content'][:300]!r}"
        assert result["finish"] in ("tool_calls", "stop"), result["finish"]
        assert (result["finish"] == "tool_calls") == bool(result["calls"]), result
        assert len({c["id"] for c in result["calls"]}) == len(result["calls"])
        messages.append(assistant_message(result, echo))
        for call in result["calls"]:
            json.loads(call["arguments"])  # must be valid JSON
            output = run_tool(root, call)
            calls_seen += 1
            turn_info["calls"].append({"name": call["name"], "arguments": call["arguments"],
                                       "result": output})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        turns.append(turn_info)
        if not result["calls"]:
            break
    check = subprocess.run(["python3", "check.py"], cwd=root, capture_output=True, text=True)
    report.update(turns=turns, tool_calls=calls_seen, final_check=check.stdout + check.stderr)
    assert check.returncode == 0, f"fixture still broken: {check.stderr[-300:]}"
    assert calls_seen >= 2 and "DONE" in turns[-1]["content"].upper(), turns[-1]


def case_hermes_loop(client, report):
    agent_loop(client, "hermes", echo=False, effort="high", report=report)


def case_hermes_loop_reasoning_echo(client, report):
    agent_loop(client, "echo", echo=True, effort="high", report=report)


def case_thinking_off_loop(client, report):
    agent_loop(client, "off", echo=False, effort="none", report=report, temperature=0)


def case_typed_parallel(client, report):
    tools = [{"type": "function", "function": {
        "name": "set_options", "description": "Record options.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string"}, "count": {"type": "integer"},
            "ratio": {"type": "number"}, "enabled": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "extra": {"type": "object"}},
            "required": ["label", "count", "enabled"]}}}]
    result = client.chat({
        "messages": [{"role": "user", "content": (
            "Call set_options twice in one reply. First: label \"007\" (a string), count 3, "
            "ratio 0.5, enabled true, tags [\"a\",\"b\"], extra {\"k\": 1}. Second: label \"x\", "
            "count 10, enabled false.")}],
        "tools": tools, "reasoning_effort": "low", "temperature": 0}, stream=True)
    report["calls"] = result["calls"]
    assert len(result["calls"]) == 2, result
    first, second = (json.loads(c["arguments"]) for c in result["calls"])
    assert first["label"] == "007" and first["count"] == 3 and first["enabled"] is True, first
    assert isinstance(first["count"], int) and isinstance(first.get("ratio", 0.5), float), first
    assert first.get("tags") == ["a", "b"] and first.get("extra") == {"k": 1}, first
    assert second["count"] == 10 and second["enabled"] is False, second
    assert [c["index"] for c in result["calls"]] == [0, 1]


def case_tool_choice(client, report):
    base = {"messages": [{"role": "user", "content": "Read README.md"}], "tools": TOOLS,
            "temperature": 0, "reasoning_effort": "none"}
    required = client.chat({**base, "tool_choice": "required"}, stream=False)
    named = client.chat({**base, "tool_choice": {"type": "function",
                                                 "function": {"name": "terminal"}}}, stream=True)
    none = client.chat({**base, "tool_choice": "none"}, stream=False)
    report.update(required=required["calls"], named=named["calls"], none_content=none["content"][:200])
    assert required["finish"] == "tool_calls" and required["calls"], required
    assert named["calls"] and all(c["name"] == "terminal" for c in named["calls"]), named
    assert not none["calls"] and "<tool_call>" not in none["content"], none


def case_greedy_stream_matches_buffered(client, report):
    body = {"messages": [{"role": "user", "content": "Run `ls` using the terminal tool."}],
            "tools": TOOLS, "temperature": 0, "reasoning_effort": "none", "max_tokens": 256}
    a, b = client.chat(body, stream=True), client.chat(body, stream=False)
    key = lambda r: ([(c["name"], json.loads(c["arguments"])) for c in r["calls"]],
                     r["content"].strip(), r["finish"])
    report.update(streamed=key(a), buffered=key(b))
    assert key(a) == key(b), (key(a), key(b))


def case_reasoning_split(client, report):
    on = client.chat({"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
                      "reasoning_effort": "high", "temperature": 0, "max_tokens": 1500})
    off = client.chat({"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
                       "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0,
                       "max_tokens": 200}, stream=False)
    report.update(on={"reasoning_chars": len(on["reasoning"]), "content": on["content"]},
                  off={"reasoning_chars": len(off["reasoning"]), "content": off["content"]})
    assert on["reasoning"].strip() and "391" in on["content"], on
    assert "</think>" not in on["content"] + on["reasoning"]
    assert not off["reasoning"] and "391" in off["content"], off


TOOL_PROMPT = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tools}"
    "\n</tools>\n\nIf none of the functions can be used, point it out. If the given question "
    "lacks the parameters required by the function, also point it out.\nIf you need to use a "
    "function, for each function call, output the function name and arguments within the "
    "following XML format:\n<tool_call>{{function-name}}\n<arg_key>{{arg-key-1}}</arg_key>\n"
    "<arg_value>{{arg-value-1}}</arg_value>\n<arg_key>{{arg-key-2}}</arg_key>\n"
    "<arg_value>{{arg-value-2}}</arg_value>\n...\n</tool_call>\n")


def case_raw_layout(client, report):
    """Record the model's own byte layout (informational; feeds the template round trip)."""
    tools = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in TOOLS)
    layouts = {}
    for name, question, think in (
            ("answer_thinking", "What is 17 * 23?", True),
            ("call_thinking", "Write the text hello to out.txt, then read it back.", True),
            ("call_no_thinking", "Write the text hello to out.txt, then read it back.", False)):
        prompt = ("<role>SYSTEM</role>" + TOOL_PROMPT.format(tools=tools)
                  + "detailed thinking " + ("on" if think else "off") + "<|role_end|>"
                  + f"<role>HUMAN</role>{question}<|role_end|><role>ASSISTANT</role>\n<think>"
                  + ("" if think else "</think>"))
        conn = http.client.HTTPConnection(client.host, client.port, timeout=600)
        conn.request("POST", "/v1/completions", json.dumps({
            "model": client.model, "prompt": prompt, "max_tokens": 700, "temperature": 0}),
            {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read().decode("utf-8", "replace")
        layouts[name] = {"status": response.status,
                         "text": (json.loads(body)["choices"][0]["text"] if response.status == 200
                                  else body[:300])}
    report["layouts"] = layouts


CASES = [case_raw_layout, case_reasoning_split, case_tool_choice, case_typed_parallel,
         case_greedy_stream_matches_buffered, case_hermes_loop,
         case_hermes_loop_reasoning_echo, case_thinking_off_loop]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Run only named cases")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = Client(args.base_url, args.model)
    rows = []
    for case in CASES:
        if args.case and case.__name__ not in args.case:
            continue
        row = {"name": case.__name__, "status": "pass"}
        start, before = time.monotonic(), len(client.requests)
        try:
            case(client, row)
        except Exception as error:  # report every case, then fail the run
            row.update(status="fail", error=repr(error))
        row["seconds"] = round(time.monotonic() - start, 1)
        row["requests"] = client.requests[before:]
        rows.append(row)
        print(f"{row['name']}: {row['status']} {row['seconds']}s", file=sys.stderr, flush=True)
    (args.output / "report.json").write_text(json.dumps(rows, indent=1))
    summary = [{k: v for k, v in r.items() if k != "requests"} for r in rows]
    print(json.dumps({"suite": "ling_agent", "cases": summary}, indent=1))
    return 0 if all(r["status"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
