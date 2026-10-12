#!/usr/bin/env python3
"""Replay a Hermes-Agent-style tool loop and report prompt-cache reuse per step.

Hermes Agent (NousResearch/hermes-agent) re-sends the whole history on every
step with these changes relative to what the model generated: prior
`reasoning_content` is dropped (default) or echoed verbatim (`reasoning_echo`);
assistant text is stripped; tool-call arguments are re-serialised with
json.dumps(sort_keys=True, separators=(",", ":")); the system prompt and tool
results are frozen; a background review fork replays the full history plus a
review prompt concurrently with the next step; compaction rewrites the middle.

For models whose template keeps prior-turn thinking (Ling-3.0-flash) the
re-rendered history never equals the live session's tokens, so reuse must come
from the snapshot taken at the end of each request's prompt (the generation
prompt boundary). Every step k >= 2 must therefore reuse at least the previous
step's whole prompt and prefill only the previous assistant turn (as the client
re-rendered it), the new tool result and the generation prompt.

Checks (all strict; exit 1 on failure, JSON report on --output):
  * step k reuses >= prompt_tokens(k-1) - ALLOW tokens and prefills at most
    prompt_tokens(k) - prompt_tokens(k-1) + ALLOW;
  * the answer (reasoning, text, tool calls) equals an uncached control;
  * review fork: runs concurrently with the next step; both reuse the previous
    prompt and the fork answers like its uncached control;
  * compaction: a rewritten middle keeps the head; the run reports the reuse
    (the head has no checkpoint until a shared prefix is learned; reported,
    asserted only after one repeat);
  * variants: strip (default Hermes) and echo (`model.reasoning_echo: true`).
Only the standard library is used.
"""

import argparse
import http.client
import json
import sys
import threading
import time
from urllib.parse import urlsplit

from check_report import check_line

# Tokens after the previous prompt that a restore may still miss: tokenizer
# merges at the boundary and the template framing around the generation prompt.
ALLOW = 16

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "offset": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"]}}},
]


def system_prompt(label, lines):
    body = "".join(
        f"Guideline {i}: keep edits small, run the tests after each change, and never "
        f"touch files outside the workspace. Prefer reading before writing.\n"
        for i in range(lines))
    return (f"You are a coding agent ({label}). Use the tools to inspect the repository, "
            f"then answer briefly.\n{body}")


def tool_result(step, lines):
    return "".join(
        f"src/module_{step}/file_{i}.py:{i + 1}: def handler_{step}_{i}(request, context): "
        f"return dispatch(context, request, route='r{step}_{i}')  # step {step}\n"
        for i in range(lines))


def canonical_arguments(arguments):
    """Hermes `_canonicalize_api_tool_calls`: sorted keys, compact, ASCII."""
    try:
        return json.dumps(json.loads(arguments), separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return "{}"


def wire_messages(history, echo):
    """The messages Hermes sends: stripped text, canonical arguments, no reasoning."""
    out = []
    for message in history:
        wire = {key: value for key, value in message.items()
                if key not in ("reasoning_content", "_synthesized")}
        if isinstance(wire.get("content"), str):
            wire["content"] = wire["content"].strip()
        if wire["role"] == "assistant":
            if wire.get("tool_calls"):
                wire["tool_calls"] = [
                    {**call, "function": {**call["function"], "arguments":
                     canonical_arguments(call["function"]["arguments"])}}
                    for call in wire["tool_calls"]]
            if echo:
                wire["reasoning_content"] = message.get("reasoning_content") or " "
        out.append(wire)
    return out


# Completion cap of every request that is compared with an uncached control
# (agent steps, the review fork and their controls): Ling spends 192 tokens on
# reasoning alone, and a warm answer cut off by the cap against a complete cold
# one is a truncation artefact, not a cache difference. Other requests keep 192.
COMPARED_MAX_TOKENS = 2048


class Client:
    def __init__(self, base_url, model, timeout):
        self.url = urlsplit(base_url)
        self.model = model
        self.timeout = timeout

    def body(self, messages, cold=False, max_tokens=192, tools=True, stream=True):
        body = {"model": self.model, "messages": messages, "temperature": 0, "seed": 7,
                "max_completion_tokens": max_tokens}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        if tools:
            body["tools"] = TOOLS
        if cold:
            body["cache_prompt"] = False
        return body

    def chat(self, messages, cold=False, max_tokens=192, tools=True):
        body = self.body(messages, cold, max_tokens, tools)
        connection = http.client.HTTPConnection(self.url.hostname, self.url.port,
                                                timeout=self.timeout)
        started = time.monotonic()
        connection.request("POST", self.url.path.rstrip("/") + "/chat/completions",
                           json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:400]!r}")
        text, reasoning, calls, usage, finish = "", "", {}, None, None
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(call.get("index", len(calls)),
                                            {"id": "", "type": "function",
                                             "function": {"name": "", "arguments": ""}})
                    slot["id"] = call.get("id") or slot["id"]
                    function = call.get("function") or {}
                    slot["function"]["name"] += function.get("name") or ""
                    slot["function"]["arguments"] += function.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
        request_id = response.getheader("X-Request-ID")
        connection.close()
        assert usage is not None and finish is not None, (usage, finish)
        gufo = usage["gufo"]
        return {"text": text, "reasoning": reasoning, "tool_calls": [calls[k] for k in sorted(calls)],
                "finish": finish, "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "cached_tokens": usage["cached_tokens"], "prefill_tokens": gufo["prefill_tokens"],
                "cache_hit": gufo["cache_hit"],
                "cache_miss_reason": gufo.get("cache_miss_reason"),
                "cache_common_prefix_tokens": gufo.get("cache_common_prefix_tokens"),
                "cache_restore_ms": gufo["cache_restore_ms"], "prefill_ms": gufo["prefill_ms"],
                "request_seq": gufo.get("request_seq"),
                "recent_prefix_overlaps": gufo.get("recent_prefix_overlaps") or {},
                "http_request_id": request_id,
                "wall_ms": round((time.monotonic() - started) * 1000, 1)}


def comparable_calls(calls):
    """Tool calls without their per-request random ids."""
    return [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in calls]


def same_answer(a, b, exact=True):
    """Warm answer == uncached control.

    exact: answer text, reasoning text, finish and tool calls (names and
    canonical arguments). Used where the restored state was produced by prefill
    (the strip variant restores prompt-end checkpoints).

    not exact (the echo variant): the reused prefix includes tokens the previous
    turn DECODED, and decode kernels are not bit-identical to the prefill a cold
    run performs, so free-form reasoning may fork at a near-tie. As in
    tests/functional/cache_growth.py:39-44 `answer()` (and docs/KV-CACHE.md:84-90, "live
    frontier ... including the tokens it generated"), the answer text, tool
    calls, finish reason and whether reasoning is present stay strict; the
    reasoning text is not compared. Exact equality is asserted for unchanged
    retries of the same request, as cache_growth.py does.
    """
    if exact:
        return (a["text"], a["reasoning"], a["finish"], comparable_calls(a["tool_calls"])) == \
            (b["text"], b["reasoning"], b["finish"], comparable_calls(b["tool_calls"]))
    return (a["text"], a["finish"], comparable_calls(a["tool_calls"]),
            bool(a["reasoning"].strip())) == \
        (b["text"], b["finish"], comparable_calls(b["tool_calls"]),
         bool(b["reasoning"].strip()))


def first_difference(warm, cold):
    """Where two answers first differ, for diagnosing a failed control."""
    out = {}
    for field in ("reasoning", "text", "finish"):
        x, y = warm[field], cold[field]
        if x != y:
            i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            out[field] = {"first_differing_char": i, "warm": x[max(0, i - 40):i + 80],
                          "cold": y[max(0, i - 40):i + 80], "warm_len": len(x), "cold_len": len(y)}
    if comparable_calls(warm["tool_calls"]) != comparable_calls(cold["tool_calls"]):
        out["tool_calls"] = {"warm": comparable_calls(warm["tool_calls"]),
                             "cold": comparable_calls(cold["tool_calls"])}
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "prefill_tokens"):
        out[key] = {"warm": warm[key], "cold": cold[key]}
    return out


def run_variant(client, variant, args, failures, report):
    echo = variant == "echo"
    label = f"hermes_{variant}_{int(time.time())}"
    history = [{"role": "system", "content": system_prompt(label, args.system_lines)},
               {"role": "user", "content": "Find where the request handlers are defined, "
                "read two of them and summarise what they do."}]
    steps = []
    control_requests = []

    def record(name, result, previous):
        row = {"name": name, "loop": variant, **{k: result[k] for k in (
            "prompt_tokens", "cached_tokens", "prefill_tokens", "cache_hit", "cache_miss_reason",
            "cache_common_prefix_tokens", "cache_restore_ms", "prefill_ms", "wall_ms",
            "completion_tokens", "finish", "request_seq", "http_request_id")}}
        if previous is not None:
            row["previous_prompt_tokens"] = previous["prompt_tokens"]
            row["previous_request_seq"] = previous["request_seq"]
            row["common_prefix_with_previous"] = result["recent_prefix_overlaps"].get(
                str(previous["request_seq"]))
            row["reuse_floor"] = previous["prompt_tokens"] - ALLOW
            row["prefill_ceiling"] = result["prompt_tokens"] - previous["prompt_tokens"] + ALLOW
        steps.append(row)
        print(f"[{variant}] {name}: prompt={row['prompt_tokens']} cached={row['cached_tokens']} "
              f"prefill={row['prefill_tokens']} hit={row['cache_hit']} "
              f"reason={row['cache_miss_reason']}", file=sys.stderr, flush=True)
        return row

    def assistant_turn(result, step):
        calls = result["tool_calls"]
        synthesized = not calls
        if synthesized:
            calls = [{"id": f"call_{step}", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": f"src/module_{step}/file_0.py"})}}]
        message = {"role": "assistant", "content": result["text"], "tool_calls": calls,
                   "reasoning_content": result["reasoning"], "_synthesized": synthesized}
        return message, calls

    def check_step(name, result, previous):
        row = record(name, result, previous)
        if previous is not None:
            if result["cached_tokens"] < row["reuse_floor"]:
                failures.append(check_line(
                    f"{variant}/{name}_reuse", f"cached_tokens >= {row['reuse_floor']} "
                    f"(previous prompt {previous['prompt_tokens']})",
                    f"cached_tokens={result['cached_tokens']} "
                    f"miss={result['cache_miss_reason']}"))
            if result["prefill_tokens"] > row["prefill_ceiling"]:
                failures.append(check_line(
                    f"{variant}/{name}_prefill", f"prefill_tokens <= {row['prefill_ceiling']}",
                    f"prefill_tokens={result['prefill_tokens']}"))
        assert result["cached_tokens"] + result["prefill_tokens"] == result["prompt_tokens"], result
        return row

    def control(name, messages, warm):
        cold = client.chat(messages, cold=True, max_tokens=COMPARED_MAX_TOKENS)
        assert cold["cached_tokens"] == 0, cold
        control_requests.append({"name": name + ":control", "loop": variant, **{k: cold[k] for k in (
            "prompt_tokens", "cached_tokens", "prefill_tokens", "cache_hit", "cache_miss_reason",
            "request_seq", "http_request_id")}})
        row = steps[-1]
        row["control_prefill_tokens"] = cold["prefill_tokens"]
        row["comparison"] = "answer-level-report" if echo else "exact"
        truncated = "length" in (cold["finish"], warm["finish"])
        if truncated:
            row["comparison"] = "truncated"  # reported, not gated
        row["answer_equals_control"] = same_answer(cold, warm, exact=not echo) and \
            cold["prompt_tokens"] == warm["prompt_tokens"]
        if not row["answer_equals_control"] and not truncated:
            row["difference"] = first_difference(warm, cold)
            settings = {k: v for k, v in client.body(messages, max_tokens=COMPARED_MAX_TOKENS).items()
                        if k not in ("messages", "tools")}
            row["request_settings"] = settings
            # Restored decode-family state differs numerically from cold prefill; near-tie
            # flips are allowed by the reproducibility contract, so echo only reports.
            if not echo:
                failures.append(check_line(
                    f"{variant}/{name}_control", "answer equals the uncached control",
                    json.dumps({'settings': settings, **row['difference']})))
        return cold

    previous = None
    older = None
    ran = 0
    real_calls = 0
    for step in range(1, args.steps + 1):
        if previous is not None and older is not None:
            growth = previous["prompt_tokens"] - older["prompt_tokens"]
            # Leave room for the fork, the next step and generation.
            if previous["prompt_tokens"] + 3 * growth + 256 > args.context:
                print(f"[{variant}] stopping after {ran} steps: context {args.context}",
                      file=sys.stderr, flush=True)
                break
        messages = wire_messages(history, echo)
        result = client.chat(messages, max_tokens=COMPARED_MAX_TOKENS)
        check_step(f"step{step}", result, previous)
        if args.controls:
            control(f"step{step}", messages, result)
        assistant, calls = assistant_turn(result, step)
        real_calls += 0 if assistant["_synthesized"] else len(calls)
        history.append(assistant)
        for call in calls:
            history.append({"role": "tool", "tool_call_id": call["id"],
                            "content": tool_result(step, args.tool_result_lines)})
        older, previous = previous, result
        ran += 1

    # Review fork: replays the full history plus a review prompt while the main
    # loop's next request runs. Both share the prefix up to the last main prompt.
    fork_history = history + [{"role": "user", "content":
                               "Review the conversation so far. Reply with one sentence."}]
    main_messages = wire_messages(history, echo)
    outcome = {}

    def go(key, messages, tools):
        outcome[key] = client.chat(messages, tools=tools, max_tokens=COMPARED_MAX_TOKENS)

    threads = [threading.Thread(target=go, args=("main", main_messages, True)),
               threading.Thread(target=go, args=("fork", wire_messages(fork_history, echo), True))]
    for thread in threads:
        thread.start()
        time.sleep(0.05)
    for thread in threads:
        thread.join()
    check_step("next_step_with_fork_running", outcome["main"], previous)
    # The fork's prompt extends the last prompt as well.
    check_step("review_fork", outcome["fork"], previous)
    if args.controls:
        cold = client.chat(wire_messages(fork_history, echo), cold=True,
                           max_tokens=COMPARED_MAX_TOKENS)
        control_requests.append({"name": "review_fork:control", "loop": variant, **{k: cold[k] for k in (
            "prompt_tokens", "cached_tokens", "prefill_tokens", "cache_hit", "cache_miss_reason",
            "request_seq", "http_request_id")}})
        steps[-1]["comparison"] = "answer-level-report" if echo else "exact"
        fork_truncated = "length" in (cold["finish"], outcome["fork"]["finish"])
        if fork_truncated:
            steps[-1]["comparison"] = "truncated"  # reported, not gated
        steps[-1]["answer_equals_control"] = same_answer(cold, outcome["fork"], exact=not echo)
        if not steps[-1]["answer_equals_control"] and not fork_truncated:
            steps[-1]["difference"] = first_difference(outcome["fork"], cold)
            # Echo is report-only (decode-family state differs numerically from cold prefill).
            if not echo:
                failures.append(check_line(
                    f"{variant}/review_fork_control", "answer equals the uncached control",
                    json.dumps(steps[-1]['difference'])))
    assistant, calls = assistant_turn(outcome["main"], ran + 1)
    real_calls += 0 if assistant["_synthesized"] else len(calls)
    probe = client.body(main_messages, tools=True, stream=False)
    history.append(assistant)
    for call in calls:
        history.append({"role": "tool", "tool_call_id": call["id"],
                        "content": tool_result(ran + 1, args.tool_result_lines)})
    previous = outcome["main"]

    # Compaction: the middle is replaced by a summary; head and last turn stay.
    # Reported (a head checkpoint exists only once a shared prefix was learned);
    # the repeat of the same compacted history must be reused whole.
    compacted = history[:2] + [{"role": "user", "content": "Summary of earlier work: "
                                "handlers were inspected in several modules."}] + history[-2:]
    messages = wire_messages(compacted, echo)
    first = client.chat(messages)
    record("compacted_first", first, previous)
    again = client.chat(messages)
    row = record("compacted_repeat", again, first)
    if again["cached_tokens"] < again["prompt_tokens"] - ALLOW:
        failures.append(check_line(
            f"{variant}/compacted_repeat_reuse",
            f"cached_tokens >= {again['prompt_tokens'] - ALLOW} (identical request)",
            f"cached_tokens={again['cached_tokens']}"))
    if real_calls == 0 and not args.allow_synthetic:
        failures.append(check_line(
            f"{variant}/tool_calls", "a parsed tool call in some step (or --allow-synthetic)",
            "no parsed tool call in any step"))
    report[variant] = {
        "steps_run": ran,
        "real_tool_calls": real_calls,
        "probe_request": probe,
        "steps": steps,
        "control_requests": control_requests,
        "tokens_prefilled_per_step": [s["prefill_tokens"] for s in steps if s["name"].startswith("step")],
        "tokens_reused_per_step": [s["cached_tokens"] for s in steps if s["name"].startswith("step")],
        "tokens_saved_total": sum(s["cached_tokens"] for s in steps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", required=True, help="http://host:port/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", action="append", choices=("strip", "echo"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--context", type=int, default=32768,
                        help="server --context; steps stop early to stay inside it")
    parser.add_argument("--system-lines", type=int, default=60)
    parser.add_argument("--tool-result-lines", type=int, default=30)
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="accept steps whose tool call was synthesized because the server "
                        "returned none (builds without the Ling tool-call parser)")
    parser.add_argument("--no-controls", dest="controls", action="store_false")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    client = Client(args.base_url, args.model, args.timeout)
    failures, report = [], {}
    for variant in args.variant or ["strip", "echo"]:
        try:
            run_variant(client, variant, args, failures, report)
        except Exception as error:  # keep the partial report
            failures.append(check_line(f"{variant}/run", "the replay completes",
                                       f"{type(error).__name__}: {error}"))
    probe = next((v.pop("probe_request") for v in report.values() if "probe_request" in v), None)
    for v in report.values():
        v.pop("probe_request", None)
    report = {"status": "failed" if failures else "passed", "failures": failures,
              "allow_tokens": ALLOW, "allow_synthetic": args.allow_synthetic,
              "probe_request": probe, "variants": report}
    with open(args.output, "w") as out:
        json.dump(report, out, indent=1)
    for message in failures:
        print(message, file=sys.stderr)
    print(json.dumps({v: {"prefilled_per_step": r["tokens_prefilled_per_step"],
                          "reused_per_step": r["tokens_reused_per_step"]}
                      for v, r in report["variants"].items()}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
