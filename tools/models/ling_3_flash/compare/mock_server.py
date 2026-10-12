#!/usr/bin/env python3
"""Stand-in for `gufo serve llm` used to validate the harness without a model.

Speaks just enough OpenAI: GET /ready, /v1/models, POST /v1/chat/completions.
* Requests with tools: a scripted "solver" replays each task's reference
  solution through list_dir / read_file / write_file / run_tests.  `--skill P`
  makes it succeed for a deterministic fraction P of seeds (others stop after
  reading), `--flaky` injects a malformed and an unknown tool call.
* Requests without tools (gufo eval): answers the pinned question set with
  "Answer: X", right with probability `--accuracy` (a function of the request
  seed, so the seed proxy is observable), and sleeps `--latency` seconds.
"""
import argparse
import hashlib
import http.server
import json
import random
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent  # noqa: E402

STATE = {"requests": 0, "seeds": [], "bodies": []}
LOCK = threading.Lock()


def eval_cases(path):
    data = json.loads(Path(path).read_text())
    return data["cases"]


def solver_plan(task):
    """Ordered list of (tool, args) steps for the reference solution."""
    plan = [("list_dir", {"path": "."})]
    sources = sorted(p.relative_to(task["dir"] / "repo") for p in (task["dir"] / "repo").rglob("*.py")
                     if "tests" not in p.parts and p.name != "__init__.py")
    if sources:
        plan.append(("read_file", {"path": str(sources[0])}))
    plan.append(("run_tests", {}))
    for path in sorted((task["dir"] / "solution").rglob("*.py")):
        rel = path.relative_to(task["dir"] / "solution")
        plan.append(("write_file", {"path": str(rel), "content": path.read_text()}))
    plan.append(("run_tests", {}))
    return plan


def make_handler(opts, tasks, cases):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, status, payload, ctype="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/ready":
                self._send(200, b"ok", "text/plain")
            elif self.path in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [{"id": opts.name, "object": "model",
                                                             "owned_by": "mock"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with LOCK:
                STATE["requests"] += 1
                STATE["seeds"].append(body.get("seed"))
                STATE["bodies"].append({k: body.get(k) for k in ("seed", "temperature", "max_tokens",
                                                               "max_completion_tokens", "reasoning_effort")})
            time.sleep(opts.latency)
            if body.get("tools"):
                self._send(200, self.agent_reply(body))
            else:
                self._send(200, self.eval_reply(body))

        # ---------------------------------------------------------------- eval
        def eval_reply(self, body):
            user = body["messages"][-1]["content"]
            chosen = None
            for case in cases:
                if case["question"][:80] in user:
                    chosen = case
                    break
            seed = body.get("seed", 0) or 0
            digest = hashlib.sha256(f"{seed}:{chosen['id'] if chosen else user[:20]}".encode()).digest()
            right = digest[0] / 255.0 < opts.accuracy
            answer = chosen["answer"] if (chosen and right) else "Z"
            text = f"Reasoning done.\nAnswer: {answer}"
            return {"id": "x", "object": "chat.completion", "model": opts.name,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text,
                                             "reasoning_content": "thinking " * 20}}],
                    "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 200, "total_tokens": 0}}

        # --------------------------------------------------------------- agent
        def agent_reply(self, body):
            messages = body["messages"]
            user = messages[1]["content"]
            task = next((t for t in tasks if t["prompt"] == user), None)
            seed = body.get("seed")
            rounds = sum(1 for m in messages if m["role"] == "assistant")
            prompt_tokens = len(json.dumps(messages)) // 4
            completion = 120 + rounds
            cached = prompt_tokens // 2 if rounds else 0
            # gufo serve's wire shape: per-request metrics under usage.gufo.
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                     "total_tokens": prompt_tokens + completion, "cached_tokens": cached,
                     "prompt_tokens_details": {"cached_tokens": cached},
                     "draft_tokens": 40, "draft_tokens_accepted": 30,
                     "gufo": {"cache_hit": rounds > 0, "prefill_tokens": prompt_tokens - cached,
                              "draft_rounds": 30}}
            # and its request-completion log line (the harness parses decode stats from it)
            print(f"[INFO] [http] request=r{STATE['requests']} event=completed method=POST "
                  f"path=/v1/chat/completions status=200 generated_tokens={completion} finish=stop "
                  f"decode_tps=40.0 draft_accepted=30 draft_proposed=40 draft_rounds=30",
                  file=sys.stderr, flush=True)
            plan = solver_plan(task) if task else []
            skilled = hashlib.sha256(f"{seed}:{task['id'] if task else ''}".encode()).digest()[0] / 255.0 < opts.skill
            if not skilled:
                plan = plan[:2]
            calls = []
            if opts.flaky and rounds == 0:
                calls = [{"id": "c_bad", "type": "function",
                          "function": {"name": "read_file", "arguments": "{path: oops"}},
                         {"id": "c_unk", "type": "function",
                          "function": {"name": "teleport", "arguments": "{}"}}]
            elif rounds - (1 if opts.flaky else 0) < len(plan):
                tool, args = plan[rounds - (1 if opts.flaky else 0)]
                calls = [{"id": f"c{rounds}", "type": "function",
                          "function": {"name": tool, "arguments": json.dumps(args)}}]
            message = {"role": "assistant", "content": "" if calls else "Done: summary.",
                       "reasoning_content": "plan " * 10}
            if calls:
                message["tool_calls"] = calls
            return {"id": "x", "object": "chat.completion", "model": opts.name,
                    "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                                 "message": message}], "usage": usage,
                    "timings": {"prompt_ms": 5.0, "predicted_ms": 20.0}}
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--name", default="mock")
    parser.add_argument("--accuracy", type=float, default=0.6)
    parser.add_argument("--skill", type=float, default=0.7)
    parser.add_argument("--flaky", action="store_true")
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--startup-delay", type=float, default=0.0)
    candidates = [HERE.parents[3] / "tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"
                  if len(HERE.parents) > 3 else Path("/nonexistent"),
                  Path("/usr/local/share/gufo/eval/antirez-ds4.json")]
    default_data = next((c for c in candidates if c.exists()), candidates[-1])
    parser.add_argument("--eval-data", default=str(default_data))
    opts = parser.parse_args()
    time.sleep(opts.startup_delay)
    tasks = agent.load_tasks()
    cases = eval_cases(opts.eval_data)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", opts.port), make_handler(opts, tasks, cases))
    print(f"mock server on {opts.port}", file=sys.stderr, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
