#!/usr/bin/env python3
"""Start `gufo serve` for Ling-3.0-Flash in this pod, then run the agentic suites.

Intended for a restricted gfx1151 pod (uid 1000, read-only rootfs, /work
writable). Prints one compact JSON summary on stdout (progress and the server
log go to stderr and /work/ling-serve) and exits non-zero if a required suite
fails. Required: tests/functional/ling_agent.py (tool calls, reasoning split,
Hermes-shaped loops). Informational (reported, never fatal): the shared
tests/functional/run.py suites, which target Qwen and DeepSeek grammar
guarantees.

    serve_functional.py [--model GGUF] [--bin gufo] [--functional DIR]
                        [--work DIR] [--no-sdk]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "/models/Ling-3.0-flash-AD-Q4_K_S-00001-of-00002.gguf"


def log(message):
    print(f"[ling-ws14] {message}", file=sys.stderr, flush=True)


def wait_ready(port, proc, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_json(command, log_path, timeout):
    with open(log_path, "wb") as err:
        done = subprocess.run(command, stdout=subprocess.PIPE, stderr=err, timeout=timeout)
    text = done.stdout.decode("utf-8", "replace")
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = {"raw": text[-2000:]}
    return done.returncode, parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--bin", default="gufo")
    parser.add_argument("--functional", type=Path, default=Path("/opt/gufo/functional"))
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("HOME", "/work")) / "ling-ws14")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--no-sdk", action="store_true", help="Skip the informational run.py suites")
    parser.add_argument("--serve-arg", action="append", default=[],
                        help="Extra `gufo serve llm` argument (use --serve-arg=--flag)")
    parser.add_argument("--sampling", choices=("vendor", "llamacpp"), default="vendor",
                        help="vendor: Gufo's Ling preset (0.6/0.95/top-k 20). llamacpp: "
                             "llama.cpp values, temperature-first order; top-p not "
                             "emulable (0.8, top-k 40, top-p 0.95, min-p 0.0236 = "
                             "0.05**(1/0.8), which makes Gufo's temperature-first "
                             "min-p cut equal llama.cpp's temperature-last cut)")
    args = parser.parse_args()
    if args.sampling == "llamacpp":
        args.serve_arg += ["--temperature", "0.8", "--top-k", "40", "--top-p", "0.95",
                           "--min-p", "0.0236"]
    args.work.mkdir(parents=True, exist_ok=True)
    label = ("llama.cpp values, temperature-first order; top-p not emulable"
             if args.sampling == "llamacpp" else "vendor defaults 0.6/0.95/top-k 20")
    summary = {"sampling_label": label, "sampling": args.sampling, "model": args.model, "suites": [], "serve_args": args.serve_arg}
    if not Path(args.model).exists():
        summary["error"] = f"model not found: {args.model}"
        print(json.dumps(summary, indent=1))
        return 1
    serve = [args.bin, "serve", "llm", "--host", "127.0.0.1", "--port", str(args.port),
             "--model", args.model, "--context", str(args.context), "--sessions", "2",
             "--served-model-name", "Ling-3.0-flash", *args.serve_arg]
    log("starting " + " ".join(serve))
    with open(args.work / "serve.log", "wb") as server_log:
        proc = subprocess.Popen(serve, stdout=server_log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    ok = True
    try:
        if not wait_ready(args.port, proc, args.startup_timeout):
            summary["error"] = "server did not become ready"
            ok = False
        else:
            code, report = run_json(
                [sys.executable, str(args.functional / "ling_agent.py"),
                 "--base-url", f"http://127.0.0.1:{args.port}", "--model", "Ling-3.0-flash",
                 "--output", str(args.work / "ling-agent")],
                args.work / "ling-agent.log", 3600)
            summary["suites"].append({"name": "ling_agent", "required": True,
                                      "exit_code": code, **report})
            ok = ok and code == 0
    finally:
        stop(proc)
    tail = (args.work / "serve.log").read_bytes()[-6000:].decode("utf-8", "replace")
    summary["serve_log_tail"] = tail.splitlines()[-25:]
    if not args.no_sdk and "error" not in summary:
        sdk = [sys.executable, str(args.functional / "run.py"), "--record-baseline",
               "--sampling-preset", "ling3"]
        for suite in ("sampling-defaults", "tools", "tool-reasoning"):
            sdk += ["--suite", suite]
        sdk += ["--output", str(args.work / "sdk-suites"), "--", args.bin, "serve", "llm",
                "--model", args.model, "--context", str(args.context), "--sessions", "2",
                *args.serve_arg]
        try:
            code, report = run_json(sdk, args.work / "sdk-suites.log", 5400)
        except subprocess.TimeoutExpired:
            code, report = None, {"error": "timeout"}
        summary["suites"].append({"name": "run.py(sampling-defaults,tools,tool-reasoning)",
                                  "required": False, "exit_code": code,
                                  "report": report if len(json.dumps(report)) < 4000 else "see sdk-suites/report.json"})
    summary["passed"] = ok
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
