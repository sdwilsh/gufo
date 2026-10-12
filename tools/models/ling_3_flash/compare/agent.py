"""Agentic-coding task suite: sandbox, OpenAI tool-calling loop, and grader.

Every task is a small Python repo under tasks/<id>/{task.json, repo/, hidden/,
solution/}.  The model sees only repo/, through the tools, and edits a copy in
a per-episode sandbox.  The grader runs the original visible tests plus the
hidden tests against the edited sandbox, so editing the tests cannot pass.
The sandbox lives under /work and the tools cannot leave it.  No network is
needed.
"""
import hashlib
import http.client
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

from common import log

TASKS_DIR = Path(__file__).resolve().parent / "tasks"

SYSTEM_PROMPT = (
    "You are a coding agent working in a small Python project. Use the provided tools "
    "to inspect the code, make the change, and verify it with run_tests. Paths are "
    "relative to the project root. When the work is finished and verified, reply with a "
    "short summary and make no further tool calls.")

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory (one entry per line; directories end with '/').",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory relative to the project root; '.' for the root"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a UTF-8 text file with 1-based line numbers.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or completely replace a file with the given contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the project's unittest suite (python3 -m unittest). "
                                              "Optionally pass a test module such as tests.test_foo.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "Optional dotted test module, e.g. tests.test_foo"}},
            "required": []}}},
]
TOOL_SPECS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

MAX_READ = 24000
MAX_WRITE = 200000
MAX_TEST_OUTPUT = 5000


class ToolError(Exception):
    pass


def load_tasks(only=None):
    tasks = []
    for path in sorted(TASKS_DIR.iterdir()):
        if not (path / "task.json").exists():
            continue
        meta = json.loads((path / "task.json").read_text())
        meta["id"] = path.name
        meta["dir"] = path
        if only and path.name not in only and not any(path.name.startswith(o) for o in only):
            continue
        tasks.append(meta)
    return tasks


def copy_tree(src, dst):
    for path in sorted(Path(src).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            target = Path(dst) / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))
    os.setsid()


def run_unittest(root, args, timeout):
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": str(root), "PYTHONHASHSEED": "0", "NO_PROXY": "*", "LANG": "C.UTF-8"}
    command = [sys.executable, "-m", "unittest", *args]
    try:
        done = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                              timeout=timeout, preexec_fn=_limits, stdin=subprocess.DEVNULL)
        return done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired as error:
        out = (error.stdout or b"") + (error.stderr or b"")
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        return 124, out + f"\n[timeout after {timeout}s]"


class Sandbox:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def resolve(self, rel):
        if not isinstance(rel, str) or not rel or "\0" in rel:
            raise ToolError("path must be a non-empty string")
        target = Path(os.path.realpath(self.root / rel))
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"path {rel!r} is outside the project")
        return target

    def list_dir(self, path):
        target = self.resolve(path)
        if not target.is_dir():
            raise ToolError(f"{path}: not a directory")
        names = []
        for child in sorted(target.iterdir()):
            if child.name in ("__pycache__", ".git"):
                continue
            names.append(child.name + ("/" if child.is_dir() else ""))
        return "\n".join(names) or "(empty)"

    def read_file(self, path):
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"{path}: no such file")
        try:
            text = target.read_text()
        except UnicodeDecodeError:
            raise ToolError(f"{path}: not a UTF-8 text file")
        numbered = "".join(f"{i:4d}  {line}" for i, line in enumerate(text.splitlines(True), 1))
        if len(numbered) > MAX_READ:
            numbered = numbered[:MAX_READ] + "\n[truncated]"
        return numbered or "(empty file)"

    def write_file(self, path, content):
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if len(content) > MAX_WRITE:
            raise ToolError("content too large")
        target = self.resolve(path)
        if target.is_dir():
            raise ToolError(f"{path}: is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def run_tests(self, target=None):
        args = ["discover", "-s", "tests", "-t", "."]
        if target:
            if not isinstance(target, str) or not all(p.isidentifier() for p in target.split(".")):
                raise ToolError("target must be a dotted module name such as tests.test_foo")
            args = [target]
        code, output = run_unittest(self.root, args, 30)
        if len(output) > MAX_TEST_OUTPUT:
            output = "[earlier output truncated]\n" + output[-MAX_TEST_OUTPUT:]
        return f"exit code {code}\n{output}"


def validate_call(call):
    """Return (error or None, parsed_args).  Checks name, JSON, required keys, and types."""
    name = call.get("name")
    if name not in TOOL_SPECS:
        return f"unknown tool {name!r}", None
    raw = call.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) and raw.strip() else ({} if raw in ("", None) else raw)
    except ValueError as error:
        return f"arguments are not valid JSON ({error})", None
    if not isinstance(args, dict):
        return "arguments are not a JSON object", None
    spec = TOOL_SPECS[name]
    for key in spec.get("required", []):
        if key not in args:
            return f"missing required argument {key!r}", args
    for key, value in args.items():
        prop = spec["properties"].get(key)
        if prop is None:
            return f"unexpected argument {key!r}", args
        if prop["type"] == "string" and not isinstance(value, str):
            return f"argument {key!r} must be a string", args
    return None, args


def execute_call(sandbox, name, args):
    try:
        return getattr(sandbox, name)(**args), None
    except ToolError as error:
        return f"error: {error}", str(error)
    except Exception as error:  # a real agent shows the failure to the model
        return f"error: {error!r}", repr(error)


class ChatClient:
    """Non-streaming Chat Completions request with explicit tools."""

    def __init__(self, base_url, model):
        url = urllib.parse.urlsplit(base_url)
        self.host, self.port, self.model = url.hostname, url.port or 80, model

    def complete(self, messages, tools, max_tokens, seed, timeout, extra=None):
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "max_tokens": max_tokens, "stream": False, **(extra or {})}
        if seed is not None:
            body["seed"] = seed
        conn = http.client.HTTPConnection(self.host, self.port, timeout=max(5.0, timeout))
        started = time.monotonic()
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(body),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read()
        finally:
            conn.close()
        seconds = time.monotonic() - started
        if response.status != 200:
            return {"http_status": response.status, "error": data.decode("utf-8", "replace")[:500],
                    "seconds": seconds}
        payload = json.loads(data)
        choice = payload["choices"][0]
        message = choice["message"]
        return {"http_status": 200, "seconds": seconds, "finish_reason": choice.get("finish_reason"),
                "content": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                "tool_calls": [{"id": c.get("id"), "name": (c.get("function") or {}).get("name"),
                                "arguments": (c.get("function") or {}).get("arguments")}
                               for c in (message.get("tool_calls") or [])],
                "usage": payload.get("usage") or {}, "timings": payload.get("timings") or {},
                # gufo serve reports its per-request metrics under usage.gufo
                "metrics": payload.get("metrics") or (payload.get("usage") or {}).get("gufo") or {}}


def grade(task, root):
    """Restore the original visible tests, add the hidden ones, run everything."""
    shutil.rmtree(Path(root) / "tests", ignore_errors=True)  # drop edited/added tests
    copy_tree(task["dir"] / "repo" / "tests", Path(root) / "tests")
    copy_tree(task["dir"] / "hidden", root)
    code, output = run_unittest(root, ["discover", "-s", "tests", "-t", ".", "-v"], 60)
    ran = None
    for line in output.splitlines():
        if line.startswith("Ran "):
            ran = int(line.split()[1])
    return {"passed": code == 0 and ran is not None and ran >= task["expected_tests"],
            "exit_code": code, "tests_run": ran, "expected_tests": task["expected_tests"],
            "output_tail": output[-1500:]}


def run_episode(task, client, work, seed, max_steps, time_limit, max_tokens, context,
                echo_reasoning=False, extra_request=None, complete=None, repeat=None):
    """Run one task attempt and return the per-episode row (JSON-serializable)."""
    root = Path(work) / f"{task['id']}-s{seed}"
    if root.exists():
        shutil.rmtree(root)
    copy_tree(task["dir"] / "repo", root)
    sandbox = Sandbox(root)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]}]
    steps = []
    # Full per-turn transcript.  compare.py writes it to a side file, not the row.
    transcript = []
    counters = {"calls": 0, "invalid_json": 0, "unknown_tool": 0, "bad_args": 0, "tool_errors": 0,
                "empty_turns": 0}
    started = time.monotonic()
    stop = "max_steps"
    complete = complete or client.complete
    for step in range(1, max_steps + 1):
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            stop = "time_limit"
            break
        row = {"step": step,
               "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True,
                                                          separators=(",", ":")).encode()).hexdigest()}
        turn = {"step": step, "prompt_sha256": row["prompt_sha256"]}
        transcript.append(turn)
        try:
            reply = complete(messages, TOOLS, max_tokens, seed, remaining + 5, extra_request)
        except (OSError, http.client.HTTPException, ValueError, KeyError) as error:
            row["transport_error"] = turn["transport_error"] = repr(error)
            steps.append(row)
            stop = "transport_error"
            break
        row.update({k: reply.get(k) for k in ("http_status", "seconds", "finish_reason", "usage",
                                              "timings", "metrics")})
        usage_now = reply.get("usage") or {}
        metrics_now = reply.get("metrics") or usage_now.get("gufo") or {}
        row["cache_hit"] = metrics_now.get("cache_hit")
        row["cached_tokens"] = usage_now.get("cached_tokens",
                                             (usage_now.get("prompt_tokens_details") or {}).get("cached_tokens"))
        row["prefill_tokens"] = metrics_now.get("prefill_tokens")
        row["completion_tokens"] = usage_now.get("completion_tokens")
        row["draft_tokens"] = usage_now.get("draft_tokens")
        row["draft_tokens_accepted"] = usage_now.get("draft_tokens_accepted")
        row["tool_result_sha256"] = []
        row["content_chars"] = len(reply.get("content") or "")
        row["reasoning_chars"] = len(reply.get("reasoning") or "")
        turn.update({k: reply.get(k) for k in ("http_status", "seconds", "finish_reason", "usage",
                                               "content", "reasoning")})
        turn["tool_calls"] = []
        if reply.get("http_status") != 200:
            row["error"] = turn["error"] = reply.get("error")
            steps.append(row)
            stop = "http_error"
            break
        calls = reply["tool_calls"]
        message = {"role": "assistant", "content": (reply["content"] or "").strip()}
        if echo_reasoning and reply["reasoning"].strip():
            message["reasoning_content"] = reply["reasoning"]
        row["calls"] = []
        if calls:
            normalised = []
            for i, call in enumerate(calls):
                call_id = call.get("id") or f"call_{step}_{i}"
                args_text = call["arguments"] if isinstance(call["arguments"], str) else json.dumps(call["arguments"] or {})
                normalised.append({"id": call_id, "type": "function",
                                   "function": {"name": call["name"] or "", "arguments": args_text}})
            message["tool_calls"] = normalised
        messages.append(message)
        if not calls:
            if reply["finish_reason"] == "length":
                row["note"] = "truncated at max_tokens"
            if not message["content"]:
                counters["empty_turns"] += 1
            steps.append(row)
            stop = "final" if reply["finish_reason"] != "length" else "length"
            break
        for call, wire in zip(calls, message["tool_calls"]):
            counters["calls"] += 1
            error, args = validate_call({"name": call["name"], "arguments": call["arguments"]})
            record = {"name": call["name"], "valid": error is None}
            # the arguments exactly as the server sent them (normally a JSON string)
            said = {"id": call.get("id"), "wire_id": wire["id"], "name": call["name"],
                    "arguments": call["arguments"], "valid": error is None, "error": error}
            t0 = time.monotonic()
            if error:
                key = ("unknown_tool" if error.startswith("unknown") else
                       "invalid_json" if "JSON" in error else "bad_args")
                counters[key] += 1
                record["error"] = error
                result = f"error: invalid tool call: {error}"
            else:
                result, exec_error = execute_call(sandbox, call["name"], args)
                if exec_error:
                    counters["tool_errors"] += 1
                    record["tool_error"] = exec_error
                said["tool_error"] = exec_error
            record["seconds"] = round(time.monotonic() - t0, 3)
            said["result"] = result
            turn["tool_calls"].append(said)
            row["calls"].append(record)
            messages.append({"role": "tool", "tool_call_id": wire["id"], "content": result})
            row["tool_result_sha256"].append(hashlib.sha256(result.encode()).hexdigest())
        steps.append(row)
        last = steps[-1].get("usage") or {}
        if last.get("prompt_tokens") and last["prompt_tokens"] + max_tokens > context:
            stop = "context_exhausted"
            break
    wall = time.monotonic() - started
    result = grade(task, root)
    usage = [s.get("usage") or {} for s in steps]
    row = {"task": task["id"], "category": task["category"], "seed": seed, "repeat": repeat,
           "passed": result["passed"],
           "stop": stop, "steps": len(steps), "wall_seconds": round(wall, 2),
           "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
           "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
           "step_seconds": [round(s.get("seconds") or 0, 2) for s in steps],
           "turns": [{k: s.get(k) for k in ("step", "prompt_sha256", "tool_result_sha256", "cache_hit",
                                            "cached_tokens", "prefill_tokens", "finish_reason",
                                            "completion_tokens", "draft_tokens", "draft_tokens_accepted",
                                            "content_chars", "reasoning_chars")} for s in steps],
           "calls": counters, "grade": result, "step_log": steps,
           "transcript": {"system": SYSTEM_PROMPT, "prompt": task["prompt"], "echo_reasoning": echo_reasoning,
                          "turns": transcript}}
    return row


def summarize_episodes(rows):
    from common import summarize, wilson, mean_ci
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    calls = sum(r["calls"]["calls"] for r in rows)
    bad = sum(r["calls"]["invalid_json"] + r["calls"]["unknown_tool"] + r["calls"]["bad_args"] for r in rows)
    by_repeat = {}
    for r in rows:
        key = r["repeat"] if r.get("repeat") is not None else r["seed"]
        by_repeat.setdefault(key, []).append(1.0 if r["passed"] else 0.0)
    per_task = {}
    for r in rows:
        per_task.setdefault(r["task"], []).append(r)
    step_seconds = [s for r in rows for s in r["step_seconds"]]
    return {
        "episodes": total, "passed": passed,
        # graded as passing although the loop ran out of steps (no final answer turn)
        "passed_at_max_steps": sum(1 for r in rows if r["passed"] and r["stop"] == "max_steps"),
        "pass_rate_pooled_wilson95": wilson(passed, total),
        "pass_rate_per_repeat": {str(k): sum(v) / len(v) for k, v in sorted(by_repeat.items())},
        "pass_rate_across_repeats_t95": mean_ci([sum(v) / len(v) for v in by_repeat.values()]),
        "tasks_solved_every_repeat": sum(1 for v in per_task.values() if all(r["passed"] for r in v)),
        "tasks_solved_any_repeat": sum(1 for v in per_task.values() if any(r["passed"] for r in v)),
        "steps": summarize([r["steps"] for r in rows]),
        "wall_seconds": summarize([r["wall_seconds"] for r in rows]),
        "prompt_tokens": summarize([r["prompt_tokens"] for r in rows]),
        "completion_tokens": summarize([r["completion_tokens"] for r in rows]),
        "completion_tokens_total": sum(r["completion_tokens"] for r in rows),
        "length_stopped_turns": sum(1 for r in rows for t in (r.get("turns") or [])
                                    if t.get("finish_reason") == "length"),
        "agent_step_latency_seconds": summarize(step_seconds),
        "tool_calls": calls,
        "tool_call_validity": (1 - bad / calls) if calls else None,
        "tool_call_failures": {k: sum(r["calls"][k] for r in rows)
                               for k in ("invalid_json", "unknown_tool", "bad_args", "tool_errors", "empty_turns")},
        "stops": {s: sum(1 for r in rows if r["stop"] == s) for s in sorted({r["stop"] for r in rows})},
        "per_task": {t: {"passed": sum(1 for r in v if r["passed"]), "episodes": len(v),
                         "mean_steps": sum(r["steps"] for r in v) / len(v),
                         "mean_wall_seconds": sum(r["wall_seconds"] for r in v) / len(v),
                         "mean_completion_tokens": sum(r["completion_tokens"] for r in v) / len(v)}
                     for t, v in sorted(per_task.items())},
    }
