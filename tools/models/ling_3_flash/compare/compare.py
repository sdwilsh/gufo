#!/usr/bin/env python3
"""Same-harness comparison of one model in `gufo` (one model per pod).

    compare.py --model ling|ds4|flashnext [--part bench,agent,eval]
               [--work DIR] [--gufo gufo] [--serve-url URL]

Phases, in this order (the bench runs before the server so they never share
GPU memory):
  bench  gufo bench -p 2048 -n 128 -d 0,16384,65536 -r 3 (+ speculative if asked)
  agent  14 original coding tasks x --agent-repeats through an OpenAI tool loop
         (seed per task and repeat; --agent-speculative mtp,off repeats the
         phase under each serve --speculative mode, one server each, and
         compares pass rates with Fisher's exact test)
  eval   `gufo eval` 75 cases x --eval-repeats seeded repeats, then one greedy run
A model scoring low is data, not failure: the exit status is non-zero only when
the harness could not run (server did not start, bad arguments, crash).

stdout carries exactly one JSON document (the summary with compact per-item
rows) written at the end; progress goes to stderr, where per-item rows appear as
`ROW {json}` lines so a pod log alone contains the evidence.  Full per-step
data is kept under --work (default $HOME/compare).
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
import common  # noqa: E402
from common import log, mean_ci, wilson, summarize  # noqa: E402
from models import MODELS, resolve  # noqa: E402


def row(kind, payload):
    print("ROW " + json.dumps({"kind": kind, **payload}, separators=(",", ":")), file=sys.stderr, flush=True)


def jsonl_append(path, obj):
    with open(path, "a") as handle:
        handle.write(json.dumps(obj) + "\n")


def load_rows(path):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


# ------------------------------------------------------------------- bench
def parse_md_bench(text):
    """Rows of llama-bench style markdown tables: the last cell is 'mean ± std'."""
    rows = []
    header = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        record = dict(zip(header, cells))
        for key in ("t/s",):
            if key in record and "±" in record[key]:
                mean, _, std = record[key].partition("±")
                try:
                    record["tps_mean"], record["tps_std"] = float(mean), float(std)
                except ValueError:
                    pass
        rows.append(record)
    return rows


def bench_has_samples(spec, result):
    if spec["bench_jsonl"]:
        return any(r.get("record") == "result" for r in result.get("records") or [])
    return bool(result.get("table"))


def run_bench(args, spec, work, mode):
    """One `gufo bench` process (AR, or speculative with mode/model)."""
    out_file = work / f"bench-{mode}.out"
    cmd = [args.gufo, "bench", "-m", spec["model"], "-p", "2048", "-n", "128",
           "-d", args.bench_depths, "-r", str(args.bench_repetitions)]
    if spec["bench_jsonl"]:
        cmd += ["-o", "jsonl"]
    if mode != "ar":
        cmd += ["--speculative", args.spec_mode, f"--{'mtp' if args.spec_mode == 'mtp' else 'dspark'}-model",
                args.spec_model]
    log("bench: " + " ".join(cmd))
    sampler = common.MemorySampler(watch_prefix=args.source_prefix)
    started = time.monotonic()
    with open(out_file, "wb") as out, open(work / f"bench-{mode}.err", "wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True)
        sampler.start(proc.pid)
        try:
            code = proc.wait(timeout=args.bench_timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, 9)
            proc.wait()
            code = None
    memory = sampler.stop()
    text = out_file.read_text(errors="replace")
    err_tail = (work / f"bench-{mode}.err").read_text(errors="replace")[-2000:]
    result = {"mode": mode, "command": cmd, "exit_code": code, "timed_out": timed_out,
              "wall_seconds": round(time.monotonic() - started, 1), "memory": memory,
              "stderr_tail": err_tail}
    if spec["bench_jsonl"]:
        records = []
        for line in text.splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        result["records"] = [r for r in records if r.get("record") in ("header", "result", "depth_fill")]
    else:
        result["table"] = parse_md_bench(text)
        result["raw"] = text[-6000:]
    if code != 0 or timed_out:
        result["failure"] = f"bench {mode}: " + ("timed out" if timed_out else f"exit code {code}")
    elif not bench_has_samples(spec, result):
        result["failure"] = f"bench {mode}: no sample records in the output"
    row("bench", {"mode": mode, "exit_code": code, "wall_seconds": result["wall_seconds"],
                  "table": result.get("table"), "records": result.get("records")})
    return result


def bench_phase(args, spec, work):
    out = {"ar": run_bench(args, spec, work, "ar")}
    if args.spec_model:
        out["speculative"] = run_bench(args, spec, work, args.spec_mode)
    else:
        out["speculative"] = {"measured": False, "reason": spec["speculative_note"]}
    return out


# ------------------------------------------------------------------- eval
def eval_command(args, port, output, greedy):
    cmd = [args.gufo, "eval", "--base-url", f"http://127.0.0.1:{port}/v1", "--output", str(output),
           "--questions", str(args.eval_questions)]
    if greedy:
        cmd.append("--greedy")
    return cmd


def run_eval_once(args, server_port, work, tag, seed, greedy):
    output = work / f"eval-{tag}.json"
    stdout_path = work / f"eval-{tag}.log"
    started = time.monotonic()
    proxy = common.SeedProxy(server_port, seed) if seed is not None else None
    port = server_port
    try:
        if proxy:
            proxy.__enter__()
            port = proxy.port
        cmd = eval_command(args, port, output, greedy)
        with open(stdout_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    start_new_session=True)
            for line in proc.stdout:
                log_file.write(line)
                print(f"[eval {tag}] {line.rstrip()}", file=sys.stderr, flush=True)
            code = proc.wait()
    finally:
        if proxy:
            proxy.__exit__(None, None, None)
    wall = round(time.monotonic() - started, 1)
    report = None
    try:
        report = json.loads(output.read_text())
    except (OSError, ValueError):
        pass
    return summarize_eval(tag, seed, greedy, code, wall, report)


def summarize_eval(tag, seed, greedy, code, wall, report):
    result = {"tag": tag, "seed": seed, "greedy": greedy, "exit_code": code, "wall_seconds": wall}
    if not report:
        result["error"] = "no report written"
        return result
    cases = report.get("cases", [])
    summary = report.get("summary", {})
    total = summary.get("total_cases") or len(cases)
    passed = summary.get("passed", 0)
    per_source = {}
    compact = []
    for case in cases:
        verdict = (case.get("grade") or {}).get("verdict")
        source = case.get("source")
        bucket = per_source.setdefault(source, {"cases": 0, "passed": 0})
        bucket["cases"] += 1
        bucket["passed"] += verdict == "passed"
        usage = case.get("usage") or {}
        compact.append({"i": case.get("index"), "id": case.get("case_id"), "source": source,
                        "verdict": verdict,
                        "extraction": (case.get("grade") or {}).get("extraction_status"),
                        "finish": (case.get("execution") or {}).get("status"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "seconds": round(((case.get("execution") or {}).get("transport_ms") or 0) / 1000, 1)})
    result.update({
        "total_cases": total, "completed_cases": summary.get("completed_cases"), "passed": passed,
        "failed": summary.get("failed"), "not_graded": summary.get("not_graded"),
        "length_finishes": summary.get("length_finishes"),
        "completion_tokens": summary.get("completion_tokens"),
        "accuracy": passed / total if total else None,
        "accuracy_of_graded": passed / (passed + (summary.get("failed") or 0))
        if (passed + (summary.get("failed") or 0)) else None,
        "wilson95_single_run": wilson(passed, total),
        "per_source": {k: {**v, "accuracy": v["passed"] / v["cases"]} for k, v in sorted(per_source.items())},
        "run": report.get("run"), "cases": compact})
    row("eval", {k: v for k, v in result.items() if k not in ("run",)})
    return result


def eval_phase(args, server, work):
    if args.eval_repeats < 1 and not args.eval_greedy:
        return {"skipped": True}
    deadline = time.monotonic() + args.eval_budget_hours * 3600 if args.eval_budget_hours else None
    jobs = [(f"seed{args.seed_base + k}", args.seed_base + k, False) for k in range(args.eval_repeats)]
    if args.eval_greedy:
        jobs.append(("greedy", None, True))
    resumed = {r["tag"]: r for r in load_rows(work / "eval-results.jsonl")
               if r.get("completed_cases") == r.get("total_cases") and not r.get("not_graded")}

    def one(job):
        tag, seed, greedy = job
        if tag in resumed:
            log(f"eval {tag}: resumed")
            return resumed[tag]
        if deadline and time.monotonic() > deadline:
            return {"tag": tag, "seed": seed, "skipped": "eval budget exhausted"}
        result = run_eval_once(args, server.port, work, tag, seed, greedy)
        _require_alive(server, "eval")
        jsonl_append(work / "eval-results.jsonl", result)
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.eval_parallel)) as pool:
        all_results = list(pool.map(one, jobs))
    results = [r for r in all_results if r["tag"] != "greedy"]
    greedy_result = next((r for r in all_results if r["tag"] == "greedy"), None)
    done = [r for r in results if r.get("accuracy") is not None]
    accuracies = [r["accuracy"] for r in done]
    pooled_pass = sum(r["passed"] for r in done)
    pooled_total = sum(r["total_cases"] for r in done)
    cases_by_id = {}
    for r in done:
        for c in r["cases"]:
            cases_by_id.setdefault(c["id"], []).append(c["verdict"] == "passed")
    totals = sorted({r["total_cases"] for r in done})
    n_text = "/".join(str(t) for t in totals) if totals else str(args.eval_questions)
    return {
        "repeats": results,
        "per_repeat_accuracy": {r["tag"]: r.get("accuracy") for r in results},
        "accuracy_mean_t95": mean_ci(accuracies),
        "pooled_wilson95": wilson(pooled_pass, pooled_total),
        "cases_always_passed": sum(1 for v in cases_by_id.values() if all(v)),
        "cases_never_passed": sum(1 for v in cases_by_id.values() if not any(v)),
        "per_source_mean_accuracy": _mean_by_source(done),
        "greedy": greedy_result,
        "note": "seeded repeats: each gufo eval run goes through a local proxy that adds a fixed request "
                f"`seed`; sampling and thinking are the server's model defaults; N={n_text} per repeat, so the "
                "per-repeat binomial interval is wide (see wilson95_single_run)",
    }


def _require_alive(server, phase):
    proc = getattr(server, "proc", None)
    if proc is not None and proc.poll() is not None:
        raise RuntimeError(f"gufo serve exited with {proc.returncode} during the {phase} phase")


def _mean_by_source(done):
    totals = {}
    for r in done:
        for source, v in r["per_source"].items():
            totals.setdefault(source, []).append(v["accuracy"])
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(totals.items())}


# ------------------------------------------------------------------ agent
def agent_seed(args, task_id, repeat):
    """Seed of one episode: fixed per (task, repeat), so every serve mode and
    every rerun draws the same request seeds; independent of --agent-tasks."""
    ids = [t["id"] for t in agent.load_tasks()]
    return args.seed_base + 1000 * repeat + ids.index(task_id)


def agent_phase(args, server, spec, work, label=None):
    tasks = agent.load_tasks(args.agent_tasks.split(",") if args.agent_tasks else None)
    client = agent.ChatClient(server.base_url, spec["served_name"])
    rows_path = work / "agent-episodes.jsonl"
    done = {(r["task"], r["seed"]): r for r in load_rows(rows_path)}
    deadline = time.monotonic() + args.agent_budget_hours * 3600 if args.agent_budget_hours else None
    episodes, skipped = [], 0
    sandboxes = work / "agent-sandboxes"
    sandboxes.mkdir(parents=True, exist_ok=True)

    jobs = [(task, agent_seed(args, task["id"], k), k) for k in range(args.agent_repeats) for task in tasks]
    tag = f"agent[{label}]" if label else "agent"

    def one(job):
        nonlocal skipped
        task, seed, repeat = job
        if (task["id"], seed) in done:
            log(f"{tag} {task['id']} seed {seed}: resumed")
            return done[(task["id"], seed)]
        if deadline and time.monotonic() > deadline:
            skipped += 1
            return None
        episode = agent.run_episode(task, client, sandboxes, seed, args.agent_max_steps,
                                    args.agent_time_limit, args.agent_max_tokens, args.context,
                                    echo_reasoning=args.echo_reasoning, repeat=repeat)
        if label:
            episode["mode"] = label
        jsonl_append(rows_path, episode)
        compact = {k: v for k, v in episode.items() if k not in ("step_log", "grade")}
        compact["grade"] = {k: v for k, v in episode["grade"].items() if k != "output_tail"}
        compact["steps_detail"] = [{"step": s["step"], "s": round(s.get("seconds") or 0, 1),
                                    "prompt": (s.get("usage") or {}).get("prompt_tokens"),
                                    "completion": (s.get("usage") or {}).get("completion_tokens"),
                                    "finish": s.get("finish_reason"),
                                    "calls": [(c["name"], c["valid"]) for c in s.get("calls", [])]}
                                   for s in episode["step_log"]]
        proc = getattr(server, "proc", None)
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"gufo serve exited with {proc.returncode} during the agent phase")
        if not episode["passed"]:
            compact["grade_tail"] = episode["grade"]["output_tail"][-700:]
        row("agent", compact)
        log(f"{tag} {task['id']} seed {seed}: {'PASS' if episode['passed'] else 'FAIL'} "
            f"steps={episode['steps']} stop={episode['stop']} {episode['wall_seconds']}s")
        return episode

    with ThreadPoolExecutor(max_workers=max(1, args.agent_parallel)) as pool:
        episodes = [e for e in pool.map(one, jobs) if e]
    summary = agent.summarize_episodes(episodes) if episodes else {"episodes": 0}
    summary["skipped_for_budget"] = skipped
    summary["limits"] = {"max_steps": args.agent_max_steps, "time_limit_seconds": args.agent_time_limit,
                         "max_tokens_per_request": args.agent_max_tokens, "tasks": [t["id"] for t in tasks],
                         "repeats": args.agent_repeats, "parallel": args.agent_parallel,
                         "echo_reasoning": args.echo_reasoning}
    summary["episodes_compact"] = [{k: e[k] for k in ("task", "seed", "passed", "stop", "steps",
                                                      "wall_seconds", "prompt_tokens",
                                                      "completion_tokens")}
                                   | {"repeat": e.get("repeat"), "calls": e["calls"], "turns": e.get("turns")}
                                   for e in episodes]
    return summary


def compare_modes(summaries, modes):
    """Pass rates, token totals and exact tests between the first two modes."""
    out = {"modes": {}}
    for mode in modes:
        s = summaries.get(mode) or {}
        eps = s.get("episodes_compact") or []
        out["modes"][mode] = {
            "episodes": len(eps), "passed": sum(1 for e in eps if e["passed"]),
            "pass_rate": wilson(sum(1 for e in eps if e["passed"]), len(eps)),
            "pass_rate_per_repeat": s.get("pass_rate_per_repeat"),
            "completion_tokens_total": sum(e["completion_tokens"] for e in eps),
            "completion_tokens": summarize([e["completion_tokens"] for e in eps]),
            "length_stopped_turns": s.get("length_stopped_turns"),
            "empty_turns": (s.get("tool_call_failures") or {}).get("empty_turns"),
            "stops": s.get("stops")}
    if len(modes) < 2:
        return out
    a, b = modes[0], modes[1]
    ma, mb = out["modes"][a], out["modes"][b]
    out["fisher_exact"] = {"a": a, "b": b, **common.fisher_exact(ma["passed"], ma["episodes"],
                                                                  mb["passed"], mb["episodes"])}
    key = lambda e: (e["task"], e.get("repeat"))  # noqa: E731
    ea = {key(e): e for e in summaries[a].get("episodes_compact") or []}
    eb = {key(e): e for e in summaries[b].get("episodes_compact") or []}
    pairs = sorted(set(ea) & set(eb), key=str)
    only_a = sum(1 for k in pairs if ea[k]["passed"] and not eb[k]["passed"])
    only_b = sum(1 for k in pairs if eb[k]["passed"] and not ea[k]["passed"])
    fewer = sum(1 for k in pairs if ea[k]["completion_tokens"] < eb[k]["completion_tokens"])
    more = sum(1 for k in pairs if ea[k]["completion_tokens"] > eb[k]["completion_tokens"])
    out["paired"] = {"pairs": len(pairs), "both_pass": sum(1 for k in pairs if ea[k]["passed"] and eb[k]["passed"]),
                     "both_fail": sum(1 for k in pairs if not ea[k]["passed"] and not eb[k]["passed"]),
                     f"only_{a}": only_a, f"only_{b}": only_b,
                     "mcnemar_exact_p": common.binomial_two_sided(only_a, only_a + only_b),
                     f"completion_tokens_{a}_lower": fewer, f"completion_tokens_{a}_higher": more,
                     "completion_tokens_sign_test_p": common.binomial_two_sided(fewer, fewer + more)}
    tasks = sorted({e["task"] for s in summaries.values() for e in s.get("episodes_compact") or []})
    out["per_task_passes"] = {t: {m: sum(1 for e in summaries[m].get("episodes_compact") or []
                                         if e["task"] == t and e["passed"]) for m in modes} for t in tasks}
    return out


def agent_modes_phase(args, spec, work):
    """The agent phase once per serve --speculative mode, each on its own server."""
    out = {"modes": {}, "serve": {}}
    for mode in args.agent_speculative:
        extra = [] if mode == "default" else ["--speculative", mode]
        mode_work = work / f"agent-{mode}"
        mode_work.mkdir(parents=True, exist_ok=True)
        server = common.Server(serve_command(args, spec, args.port) + extra, args.port,
                               mode_work / "serve.log", args.startup_timeout, watch_prefix=args.source_prefix)
        ok, why = server.start()
        serve = {"command": server.command, "load_seconds": server.load_seconds}
        out["serve"][mode] = serve
        if not ok:
            serve["error"] = why
            serve["log_tail"] = server.log_tail(60)
            server.stop()
            raise RuntimeError(f"gufo serve ({mode}) did not start: {why}")
        start = _log_lines(server)
        try:
            serve["log_head"] = server.log_head(80)
            out["modes"][mode] = agent_phase(args, server, spec, mode_work, label=mode)
        finally:
            serve["memory"] = server.stop()
            serve["log_tail"] = server.log_tail(30)
            serve["decode"] = decode_stats(server, [("agent", start)])
            serve["upload_fallback_lines"] = [l for l in server.log_head(100000) if "span alloc failed" in l][:5]
    out["comparison"] = compare_modes(out["modes"], args.agent_speculative)
    row("agent_comparison", out["comparison"])
    return out


# ------------------------------------------------------------------- main
def serve_command(args, spec, port):
    return [args.gufo, "serve", "--host", "127.0.0.1", "--port", str(port), "llm",
            "--model", spec["model"], "--context", str(args.context), "--sessions", str(args.sessions),
            "--served-model-name", spec["served_name"], "--cache-ram-bytes", str(args.cache_ram_bytes),
            *spec["serve_extra"], *args.serve_arg]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--part", default="bench,agent,eval", help="comma list of bench,agent,eval")
    p.add_argument("--work", type=Path, default=Path(os.environ.get("HOME", "/work")) / "compare")
    p.add_argument("--gufo", default="gufo")
    p.add_argument("--serve-url", help="use an already running server (host:port) instead of starting one")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--startup-timeout", type=float, default=3600)
    # identical serve settings across models
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--cache-ram-bytes", type=int, default=1 << 30)
    p.add_argument("--quick", action="store_true",
                   help="quick pass: agent 14 tasks x1, eval 1 seeded repeat + 1 greedy run (run "
                        "concurrently), agent limits 16 steps / 600 s; bench unchanged")
    p.add_argument("--model-path", help="override the model artifact (first shard for sharded GGUF)")
    p.add_argument("--stage-to", type=Path,
                   help="copy all shards of the model to this local directory before loading and run from there")
    p.add_argument("--serve-arg", action="append", default=[], help="extra gufo serve llm argument")
    p.add_argument("--seed-base", type=int, default=1000)
    # bench
    p.add_argument("--bench-depths", default="0,16384,65536")
    p.add_argument("--bench-repetitions", type=int, default=3)
    p.add_argument("--bench-timeout", type=float, default=4 * 3600)
    p.add_argument("--spec-mode", choices=("mtp", "dspark"))
    p.add_argument("--spec-model", help="draft/support GGUF; enables the separate speculative bench pass")
    # eval
    p.add_argument("--eval-repeats", type=int, default=3)
    p.add_argument("--eval-questions", type=int, default=75)
    p.add_argument("--eval-parallel", type=int, default=2)
    p.add_argument("--eval-greedy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-budget-hours", type=float)
    # agent
    p.add_argument("--agent-repeats", type=int, default=2,
                   help="repeats of every task; the request seed is fixed per (task, repeat)")
    p.add_argument("--agent-speculative",
                   help="comma list of serve --speculative modes (e.g. mtp,off; 'default' adds no flag): "
                        "the agent phase runs once per mode on its own server, then the modes are compared")
    p.add_argument("--agent-tasks", help="comma list of task ids/prefixes")
    p.add_argument("--agent-max-steps", type=int, default=20)
    p.add_argument("--agent-time-limit", type=float, default=900)
    p.add_argument("--agent-max-tokens", type=int, default=8192)
    p.add_argument("--agent-parallel", type=int, default=1)
    p.add_argument("--agent-budget-hours", type=float)
    p.add_argument("--echo-reasoning", action="store_true")
    args = p.parse_args(argv)
    if args.quick:
        args.eval_repeats, args.agent_repeats = 1, 1
        args.eval_greedy = True
        args.eval_parallel = max(args.eval_parallel, 2)
        args.agent_max_steps, args.agent_time_limit = min(args.agent_max_steps, 16), min(args.agent_time_limit, 600)
    args.parts = [x for x in args.part.split(",") if x]
    args.agent_speculative = [x for x in (args.agent_speculative or "").split(",") if x]
    if any(not re.fullmatch(r"[a-z]+", x) for x in args.agent_speculative):
        p.error("--agent-speculative takes a comma list of modes such as mtp,off")
    if args.agent_speculative and args.serve_url:
        p.error("--agent-speculative starts its own servers; it cannot use --serve-url")
    if any(x not in ("bench", "agent", "eval") for x in args.parts):
        p.error("--part takes bench,agent,eval")
    if args.spec_model and not args.spec_mode:
        p.error("--spec-model needs --spec-mode")
    return args


def prepare_environment(work):
    for name in ("TMPDIR", "XDG_CACHE_HOME"):
        if os.environ.get(name):
            Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    args = parse_args(argv)
    prepare_environment(args.work)
    spec = resolve(args.model)
    if args.model_path:
        spec["model"] = args.model_path
    args.source_prefix = None
    work = args.work / args.model
    work.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "gufo.ling-compare.v1", "model_key": args.model, "label": spec["label"],
               "artifact": spec["model"], "host": platform.node(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "image": os.environ.get("GUFO_IMAGE"), "sampling": spec["sampling"], "thinking": spec["thinking"],
               "sources": spec["sources"],
               "serve_settings": {"context": args.context, "sessions": args.sessions,
                                  "cache_ram_bytes": args.cache_ram_bytes, "cache_disk": None,
                                  "extra": spec["serve_extra"] + args.serve_arg},
               "harness": {"parts": args.parts, "seed_base": args.seed_base,
                           "seed_rule": "seed_base + 1000 * repeat + task index",
                           "agent_speculative": args.agent_speculative or None},
               "gpu": common.gpu_memory(), "phases": {}, "errors": []}
    harness_failed = False
    if not args.serve_url and not shutil.which(args.gufo) and not Path(args.gufo).exists():
        summary["errors"].append(f"gufo binary not found: {args.gufo}")
        harness_failed = True
    if not args.serve_url and not Path(spec["model"]).exists():
        summary["errors"].append(f"model artifact not found: {spec['model']}")
        harness_failed = True
    if harness_failed:
        print(json.dumps(summary, indent=1))
        return 2
    if args.stage_to:
        try:
            sources = common.shard_files(spec["model"])
            missing = [str(s) for s in sources if not s.exists()]
            if missing:
                raise RuntimeError(f"missing shard(s): {missing}")
            args.source_prefix = str(Path(spec["model"]).resolve().parent) + "/"
            staged, report = common.stage_files(sources, args.stage_to)
            summary["staging"] = report
            spec["model_original"] = spec["model"]
            spec["model"] = str(staged[0])
            summary["artifact_staged"] = spec["model"]
            log(f"staged {report['gib']} GiB in {report['seconds']}s to {args.stage_to}")
            row("staging", report)
        except Exception as error:
            summary["errors"].append(f"staging failed: {error}")
            print(json.dumps(summary, indent=1))
            return 2
    log(f"model {args.model}: {spec['label']} ({spec['model']}); work {work}")

    def finish():
        summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary["cgroup_memory"] = common.cgroup_memory()
        (work / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        return 2 if harness_failed else 0

    try:
        if "bench" in args.parts:
            summary["phases"]["bench"] = bench_phase(args, spec, work)
            failed = [r["failure"] for r in summary["phases"]["bench"].values() if r.get("failure")]
            if failed:
                summary["phases"]["bench"]["status"] = "failed"
                summary["errors"].extend(failed)
                harness_failed = True
            else:
                summary["phases"]["bench"]["status"] = "passed"
            (work / "summary.json").write_text(json.dumps(summary, indent=1))
        shared_parts = list(args.parts)
        if "agent" in args.parts and args.agent_speculative:
            summary["phases"]["agent"] = agent_modes_phase(args, spec, work)
            shared_parts.remove("agent")
            (work / "summary.json").write_text(json.dumps(summary, indent=1))
        if "agent" in shared_parts or "eval" in shared_parts:
            if args.serve_url:
                port = int(args.serve_url.rsplit(":", 1)[1])
                server = type("External", (), {"port": port, "base_url": f"http://127.0.0.1:{port}",
                                               "load_seconds": None})()
                ok, why = True, None
            else:
                server = common.Server(serve_command(args, spec, args.port), args.port,
                                       work / "serve.log", args.startup_timeout,
                                       watch_prefix=args.source_prefix)
                ok, why = server.start()
            summary["serve"] = {"command": getattr(server, "command", None),
                                "load_seconds": server.load_seconds}
            if not ok:
                summary["errors"].append(why)
                summary["serve"]["log_tail"] = server.log_tail(60) if hasattr(server, "log_tail") else []
                harness_failed = True
                for line in summary["serve"]["log_tail"]:
                    print(f"[serve.log] {line}", file=sys.stderr)
                if hasattr(server, "stop"):
                    server.stop()
                return finish()
            marks = {"after_load": _peak(server)}
            # Serve-log line count at each phase start: decode statistics per phase.
            phase_starts = []
            try:
                if hasattr(server, "log_head"):
                    summary["serve"]["log_head"] = server.log_head(80)
                if "agent" in shared_parts:
                    phase_starts.append(("agent", _log_lines(server)))
                    summary["phases"]["agent"] = agent_phase(args, server, spec, work)
                    marks["after_agent"] = _peak(server)
                if "eval" in shared_parts:
                    phase_starts.append(("eval", _log_lines(server)))
                    summary["phases"]["eval"] = eval_phase(args, server, work)
                    marks["after_eval"] = _peak(server)
            finally:
                if hasattr(server, "stop"):
                    summary["memory"] = {"serve": server.stop(), "peaks_after_phase": marks}
                    summary["serve"]["log_tail"] = server.log_tail(30)
                    summary["serve"]["upload_fallback_lines"] = [
                        l for l in server.log_head(100000) if "span alloc failed" in l][:5]
                summary["serve"]["decode"] = decode_stats(server, phase_starts)
    except Exception:  # harness bug or environment failure, not a score
        summary["errors"].append(traceback.format_exc())
        harness_failed = True
        log("harness failure:\n" + traceback.format_exc())
    return finish()


COMPLETED_FIELD = re.compile(r"\b(decode_tps|generated_tokens|draft_accepted|draft_proposed|draft_rounds)=([0-9.]+)")


def _log_lines(server):
    return len(server.log_head(10**9)) if hasattr(server, "log_head") else None


def decode_stats(server, phase_starts):
    """Per phase: decode_tps distribution and draft acceptance over every completed request."""
    if not hasattr(server, "log_head") or not phase_starts:
        return None
    lines = server.log_head(10**9)
    out = {}
    for i, (name, start) in enumerate(phase_starts):
        end = phase_starts[i + 1][1] if i + 1 < len(phase_starts) else len(lines)
        rows = []
        for line in lines[start or 0:end]:
            if "event=completed" not in line or "/v1/chat/completions" not in line:
                continue
            f = {k: float(v) for k, v in COMPLETED_FIELD.findall(line)}
            if f.get("generated_tokens", 0) > 0 and "decode_tps" in f:
                rows.append(f)
        tps = [r["decode_tps"] for r in rows]
        proposed = sum(r.get("draft_proposed", 0) for r in rows)
        accepted = sum(r.get("draft_accepted", 0) for r in rows)
        generated = sum(r.get("generated_tokens", 0) for r in rows)
        out[name] = {"requests": len(rows), "generated_tokens": int(generated),
                     "decode_tps": summarize(tps) if tps else None,
                     "decode_tps_p10_p50_p90": [common.percentile(tps, q) for q in (0.1, 0.5, 0.9)] if tps else None,
                     "draft_proposed": int(proposed), "draft_accepted": int(accepted),
                     "draft_acceptance": round(accepted / proposed, 4) if proposed else None}
    return out


def _peak(server):
    s = getattr(server, "sampler", None)
    if not s:
        return None
    return {"gtt_peak_gib": round(s.peak_gtt / common.GIB, 3), "process_rss_peak_gib": round(s.peak_rss / common.GIB, 3)}


if __name__ == "__main__":
    sys.exit(main())
