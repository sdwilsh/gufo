#!/usr/bin/env python3
"""Offline checks of the harness (no GPU, no model).

Covers sandbox confinement, tool-call validation, grader anti-cheating,
statistics, harness exit codes, episode transcripts (side file, log chunks,
decoder), and per-repeat seeds.
Run: python3 selftest.py"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import agent
import common
import compare
import transcripts

HERE = Path(__file__).resolve().parent


def scratch_dir():
    return Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))


class SandboxTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.task = agent.load_tasks(["t01"])[0]
        self.root = Path(self.tmp.name) / "box"
        agent.copy_tree(self.task["dir"] / "repo", self.root)
        self.box = agent.Sandbox(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_escape_rejected(self):
        for bad in ("../x", "/etc/passwd", "stats/../../x", ""):
            with self.assertRaises(agent.ToolError):
                self.box.read_file(bad)
            with self.assertRaises(agent.ToolError):
                self.box.write_file(bad, "x")

    def test_symlink_escape_rejected(self):
        (self.root / "link").symlink_to("/etc")
        with self.assertRaises(agent.ToolError):
            self.box.read_file("link/passwd")

    def test_validate_call(self):
        self.assertIsNone(agent.validate_call({"name": "list_dir", "arguments": '{"path": "."}'})[0])
        self.assertIn("JSON", agent.validate_call({"name": "list_dir", "arguments": "{oops"})[0])
        self.assertIn("unknown", agent.validate_call({"name": "rm", "arguments": "{}"})[0])
        self.assertIn("missing", agent.validate_call({"name": "write_file", "arguments": '{"path":"a"}'})[0])
        self.assertIn("string", agent.validate_call({"name": "read_file", "arguments": '{"path": 3}'})[0])
        self.assertIsNone(agent.validate_call({"name": "run_tests", "arguments": ""})[0])

    def test_run_tests_reports_failure(self):
        out = self.box.run_tests()
        self.assertIn("exit code 1", out)

    def test_run_tests_timeout(self):
        (self.root / "tests" / "test_hang.py").write_text(
            "import unittest, time\nclass T(unittest.TestCase):\n    def test(self):\n        while True: pass\n")
        code, text = agent.run_unittest(self.root, ["tests.test_hang"], 2)
        self.assertEqual(code, 124)

    def test_grader_ignores_test_edits(self):
        # Emptying the visible tests and adding a passing one must not pass an unfixed repo.
        (self.root / "tests" / "test_window.py").write_text("")
        (self.root / "tests" / "test_fake.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test(self): pass\n")
        self.assertFalse(agent.grade(self.task, self.root)["passed"])

    def test_grader_accepts_solution(self):
        agent.copy_tree(self.task["dir"] / "solution", self.root)
        self.assertTrue(agent.grade(self.task, self.root)["passed"])

    def test_bad_write_is_tool_error_not_crash(self):
        text, err = agent.execute_call(self.box, "write_file", {"path": "../escape", "content": "x"})
        self.assertTrue(err and text.startswith("error"))

    def test_episode_records_per_turn_hashes_and_cache_use(self):
        replies = [
            {"http_status": 200, "seconds": 0.1, "finish_reason": "tool_calls", "content": "", "reasoning": "",
             "tool_calls": [{"id": "c1", "name": "list_dir", "arguments": '{"path": "."}'}],
             "usage": {"prompt_tokens": 50, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}},
             "metrics": {"cache_hit": False, "prefill_tokens": 50}},
            {"http_status": 200, "seconds": 0.1, "finish_reason": "stop", "content": "done", "reasoning": "",
             "tool_calls": [],
             "usage": {"prompt_tokens": 80, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 48}},
             "metrics": {"cache_hit": True, "prefill_tokens": 32}}]
        work = scratch_dir()
        row = agent.run_episode(self.task, None, work, 1, 4, 60, 128, 4096,
                                complete=lambda *a: replies.pop(0))
        first, second = row["turns"]
        self.assertEqual(len(first["prompt_sha256"]), 64)
        self.assertNotEqual(first["prompt_sha256"], second["prompt_sha256"])
        self.assertEqual(len(first["tool_result_sha256"]), 1)
        self.assertEqual((second["cache_hit"], second["cached_tokens"], second["prefill_tokens"]), (True, 48, 32))

    def test_episode_reads_gufo_metrics_from_usage(self):
        # gufo serve reports cache_hit and prefill_tokens under usage.gufo, not a top-level "metrics".
        replies = [{"http_status": 200, "seconds": 0.1, "finish_reason": "stop", "content": "done",
                    "reasoning": "", "tool_calls": [], "metrics": {},
                    "usage": {"prompt_tokens": 80, "completion_tokens": 3, "cached_tokens": 48,
                              "draft_tokens": 2, "draft_tokens_accepted": 1,
                              "gufo": {"cache_hit": True, "prefill_tokens": 32}}}]
        work = scratch_dir()
        row = agent.run_episode(self.task, None, work, 1, 4, 60, 128, 4096,
                                complete=lambda *a: replies.pop(0), repeat=0)
        turn = row["turns"][0]
        self.assertEqual((turn["cache_hit"], turn["cached_tokens"], turn["prefill_tokens"]), (True, 48, 32))
        self.assertEqual((turn["finish_reason"], turn["completion_tokens"], turn["draft_tokens"],
                          turn["draft_tokens_accepted"]), ("stop", 3, 2, 1))
        self.assertEqual(row["repeat"], 0)

    def test_episode_transcript_keeps_raw_calls_and_results(self):
        raw_bad, raw_ok = '{path: oops', '{"path" :  "."}'
        replies = [
            {"http_status": 200, "seconds": 0.1, "finish_reason": "tool_calls", "content": "  looking\n",
             "reasoning": "I should list the root \u00e9", "usage": {"prompt_tokens": 50, "completion_tokens": 5},
             "tool_calls": [{"id": "c1", "name": "read_file", "arguments": raw_bad},
                            {"id": None, "name": "list_dir", "arguments": raw_ok}]},
            {"http_status": 200, "seconds": 0.1, "finish_reason": "stop", "content": "done", "reasoning": "ok",
             "tool_calls": [], "usage": {"prompt_tokens": 80, "completion_tokens": 3}}]
        work = scratch_dir()
        row = agent.run_episode(self.task, None, work, 1, 4, 60, 128, 4096,
                                complete=lambda *a: replies.pop(0), repeat=0)
        record = row["transcript"]
        self.assertEqual((record["system"], record["prompt"]), (agent.SYSTEM_PROMPT, self.task["prompt"]))
        first, second = record["turns"]
        self.assertEqual((first["content"], first["reasoning"], first["finish_reason"]),
                         ("  looking\n", "I should list the root \u00e9", "tool_calls"))
        bad, ok = first["tool_calls"]
        self.assertEqual((bad["name"], bad["arguments"], bad["valid"]), ("read_file", raw_bad, False))
        self.assertIn("JSON", bad["error"])
        self.assertTrue(bad["result"].startswith("error: invalid tool call"))
        self.assertEqual((ok["arguments"], ok["valid"], ok["error"], ok["wire_id"]), (raw_ok, True, None, "call_1_1"))
        self.assertIn("tests/", ok["result"])
        self.assertEqual(row["turns"][0]["tool_result_sha256"],
                         [transcripts.sha256_hex(c["result"].encode()) for c in first["tool_calls"]])
        self.assertEqual((second["content"], second["finish_reason"], second["tool_calls"]), ("done", "stop", []))
        self.assertEqual(second["prompt_sha256"], row["turns"][1]["prompt_sha256"])


class TranscriptTest(unittest.TestCase):
    def setUp(self):
        self.work = scratch_dir()
        self.records = [{"key": transcripts.episode_key("mtp", "t09_ini_comments", r, 7008 + 1000 * r),
                         "turns": [{"content": "x" * 400 + str(r), "reasoning": "\u00fc\n" * 50,
                                    "tool_calls": [{"arguments": '{"path": "a.py"}'}]}]} for r in range(2)]

    def test_file_and_log_round_trip(self):
        path = self.work / "agent-transcripts-mtp.jsonl.gz"
        refs = [transcripts.append(path, r) for r in self.records]
        self.assertEqual(refs[0]["key"], "mtp/t09_ini_comments/r0/s7008")
        read = transcripts.read_file(path)
        self.assertEqual([sha for sha, _ in read], [r["sha256"] for r in refs])
        self.assertEqual([rec for _, rec in read], self.records)
        info = transcripts.file_info(path)
        self.assertEqual(info["sha256"], transcripts.sha256_hex(path.read_bytes()))
        # Chunked lines interleave with other log output and kubectl --timestamps prefixes.
        a, b = (transcripts.log_lines(r["key"], transcripts.canonical(r), width=40) for r in self.records)
        self.assertGreater(len(a), 3)
        lines = ["[ling-compare 10:00:00] noise"]
        for i in range(max(len(a), len(b))):
            lines += [f"2026-10-05T10:00:0{i % 10}Z {x}" for x in (a[i:i + 1] + b[i:i + 1])]
        decoded = transcripts.decode_log(lines)
        self.assertEqual([(e["ok"], e["key"], e["sha256"]) for e in decoded],
                         [(True, r["key"], r["sha256"]) for r in refs])
        self.assertEqual([e["record"] for e in decoded], self.records)
        missing = transcripts.decode_log([x for x in a if " 2/" not in x])
        self.assertIn("missing chunk", missing[0]["error"])
        last = a[-1]
        tampered = a[:-1] + [last[:-4] + ("AAAA" if not last.endswith("AAAA") else "BBBB")]
        self.assertFalse(transcripts.decode_log(tampered)[0]["ok"])

    def test_cli_decode_and_verify(self):
        path = self.work / "t.jsonl.gz"
        for r in self.records:
            transcripts.append(path, r)
        log = self.work / "pod.log"
        log.write_text("\n".join(x for r in self.records
                                 for x in transcripts.log_lines(r["key"], transcripts.canonical(r))) + "\n")
        cli = [sys.executable, str(HERE / "transcripts.py")]
        out = self.work / "decoded.jsonl.gz"
        done = subprocess.run([*cli, "decode", str(log), "-o", str(out)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("2/2 transcripts verified", done.stdout)
        self.assertEqual(transcripts.read_file(out), transcripts.read_file(path))
        done = subprocess.run([*cli, "verify", str(path)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(transcripts.file_info(path)["sha256"], done.stdout)
        log.write_text("\n".join(log.read_text().splitlines()[:-1]))
        self.assertEqual(subprocess.run([*cli, "decode", str(log)], capture_output=True).returncode, 1)


class StatsTest(unittest.TestCase):
    def test_fisher_exact(self):
        # 10/14 against 14/14: P(X <= 10) = C(14,10) / C(28,24).  Two-sided doubles the symmetric tail.
        f = common.fisher_exact(10, 14, 14, 14)
        self.assertAlmostEqual(f["p_a_less"], 1001 / 20475, places=6)
        self.assertAlmostEqual(f["p_two_sided"], 2 * 1001 / 20475, places=6)
        # 10/14 against a pooled 27/28: (C(14,10) C(28,27) + C(14,9)) / C(42,37)
        self.assertAlmostEqual(common.fisher_exact(10, 14, 27, 28)["p_a_less"], 30030 / 850668, places=6)
        self.assertAlmostEqual(common.fisher_exact(7, 14, 7, 14)["p_two_sided"], 1.0)
        self.assertAlmostEqual(common.binomial_two_sided(0, 4), 0.125)

    def test_wilson(self):
        w = common.wilson(45, 75)
        self.assertAlmostEqual(w["lo"], 0.4869, places=3)
        self.assertAlmostEqual(w["hi"], 0.7034, places=3)
        self.assertEqual(common.wilson(0, 0)["n"], 0)

    def test_t_interval(self):
        m = common.mean_ci([0.6, 0.62, 0.58])
        self.assertAlmostEqual(m["mean"], 0.6)
        self.assertAlmostEqual(m["ci95"], 4.303 * 0.02 / 3 ** 0.5, places=4)
        self.assertIsNone(common.mean_ci([0.5])["ci95"])


class BenchFailureTest(unittest.TestCase):
    def run_bench(self, script, jsonl):
        work = scratch_dir()
        fake = work / "gufo"
        fake.write_text("#!/bin/sh\n" + script)
        fake.chmod(0o755)
        args = argparse.Namespace(gufo=str(fake), bench_depths="0", bench_repetitions=1, spec_model=None,
                                  spec_mode="mtp", bench_timeout=30, source_prefix=None)
        return compare.run_bench(args, {"model": "m", "bench_jsonl": jsonl}, work, "ar")

    def test_nonzero_exit_is_failure(self):
        self.assertIn("exit code 3", self.run_bench("exit 3", True)["failure"])

    def test_no_samples_is_failure(self):
        self.assertIn("no sample records", self.run_bench("echo '{\"record\": \"header\"}'", True)["failure"])
        self.assertIn("no sample records", self.run_bench("echo nothing", False)["failure"])

    def test_samples_pass(self):
        out = self.run_bench("echo '{\"record\": \"result\", \"avg_ts\": 1}'", True)
        self.assertNotIn("failure", out)


class SummaryTest(unittest.TestCase):
    def test_passed_at_max_steps(self):
        calls = {k: 0 for k in ("calls", "invalid_json", "unknown_tool", "bad_args", "tool_errors", "empty_turns")}
        rows = [{"task": "t", "seed": 1, "passed": p, "stop": s, "steps": 3, "wall_seconds": 1.0,
                 "prompt_tokens": 1, "completion_tokens": 1, "step_seconds": [1.0], "calls": calls}
                for p, s in ((True, "max_steps"), (True, "final"), (False, "max_steps"))]
        summary = agent.summarize_episodes(rows)
        self.assertEqual((summary["passed"], summary["passed_at_max_steps"]), (2, 1))


class HarnessExit(unittest.TestCase):
    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(HERE / "compare.py"), *args], capture_output=True, text=True,
                              env={**os.environ, **(env or {})})

    def test_missing_artifact_is_harness_failure(self):
        done = self.run_cli("--model", "ling", "--gufo", "/bin/true", "--work", tempfile.mkdtemp(),
                            "--part", "agent")
        self.assertEqual(done.returncode, 2)
        self.assertIn("model artifact not found", done.stdout)

    def test_agent_speculative_modes_end_to_end(self):
        # Fake gufo runs the flaky mock server (a malformed and an unknown call first).
        # Two serve modes x 3 repeats x 2 tasks, with transcripts also chunked into the log.
        work = scratch_dir()
        model = work / "model.gguf"
        model.write_bytes(b"GGUF")
        done = self.run_cli("--model", "ling", "--model-path", str(model), "--gufo", str(HERE / "fake_gufo.py"),
                            "--work", str(work), "--part", "agent", "--agent-tasks", "t01,t02",
                            "--agent-repeats", "3", "--agent-speculative", "mtp,off", "--port", "18491",
                            "--startup-timeout", "60", "--seed-base", "7000", "--agent-transcript-log",
                            env={"FAKE_MOCK_ARGS": "--flaky"})
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        summary = json.loads(done.stdout)
        agent_phase = summary["phases"]["agent"]
        self.assertEqual(set(agent_phase["modes"]), {"mtp", "off"})
        self.assertIn("--speculative", agent_phase["serve"]["off"]["command"])
        self.assertEqual(agent_phase["serve"]["off"]["command"][-1], "off")
        decode = agent_phase["serve"]["mtp"]["decode"]["agent"]
        self.assertGreater(decode["requests"], 0)
        self.assertEqual(decode["draft_acceptance"], 0.75)
        comparison = agent_phase["comparison"]
        self.assertEqual(comparison["modes"]["mtp"]["episodes"], 6)
        self.assertIsNotNone(comparison["fisher_exact"]["p_two_sided"])
        self.assertEqual(comparison["paired"]["pairs"], 6)
        seeds = {m: sorted((e["task"], e["repeat"], e["seed"]) for e in agent_phase["modes"][m]["episodes_compact"])
                 for m in ("mtp", "off")}
        self.assertEqual(seeds["mtp"], seeds["off"])
        self.assertEqual(len({s for _, _, s in seeds["mtp"]}), 6)  # one seed per (task, repeat)
        # seed = seed-base + 1000 * repeat + index in the full task list (t01 is 0, t02 is 1)
        expected = {"t01_moving_average": [7000, 8000, 9000], "t02_merge_intervals": [7001, 8001, 9001]}
        self.assertEqual(sorted(seeds["mtp"]), sorted((t, r, s) for t, v in expected.items() for r, s in enumerate(v)))
        for mode in ("mtp", "off"):
            self.assertEqual(agent_phase["modes"][mode]["seeds_per_task"], expected)
            for task, seeds_of_task in expected.items():
                self.assertEqual(comparison["per_task"][task][mode]["seeds"], seeds_of_task)
        turn = agent_phase["modes"]["mtp"]["episodes_compact"][0]["turns"][1]
        self.assertIsNotNone(turn["cache_hit"])
        self.assertIsNotNone(turn["prefill_tokens"])
        # Per-task pass counts with a Fisher p, in the summary and printed at the end.
        for task, v in comparison["per_task"].items():
            self.assertEqual(v["mtp"]["passed"], comparison["per_task_passes"][task]["mtp"])
            self.assertEqual(v["mtp"]["episodes"], 3)
            self.assertIsNotNone(v["fisher_p_two_sided"])
            self.assertRegex(done.stderr, rf"{task} mtp {v['mtp']['passed']}/3 off {v['off']['passed']}/3 fisher_p=")
        self.assertRegex(done.stderr, r"  all mtp \d+/6 off \d+/6 fisher_p=")
        # One transcript file per mode, referenced by path and sha256 from the summary and the log.
        logged = {e["sha256"]: e for e in transcripts.decode_log(done.stderr.splitlines())}
        self.assertTrue(all(e["ok"] for e in logged.values()))
        self.assertEqual(len(logged), 12)
        for mode in ("mtp", "off"):
            info = agent_phase["modes"][mode]["transcripts"]
            path = work / "ling" / f"agent-{mode}" / f"agent-transcripts-{mode}.jsonl.gz"
            self.assertEqual(info["path"], str(path))
            self.assertEqual({k: info[k] for k in ("path", "bytes", "sha256")}, transcripts.file_info(path))
            self.assertEqual(info["episodes"], 6)
            self.assertIn(f"agent transcripts {mode}: {path} bytes={info['bytes']} sha256={info['sha256']}",
                          done.stderr)
            records = {sha: record for sha, record in transcripts.read_file(path)}
            for episode in agent_phase["modes"][mode]["episodes_compact"]:
                ref = episode["transcript"]
                self.assertEqual(ref["file"], str(path))
                record = records[ref["sha256"]]
                self.assertEqual(logged[ref["sha256"]]["record"], record)
                self.assertIn(f"transcript {ref['key']} sha256={ref['sha256']}", done.stderr)
                self.assertEqual((record["mode"], record["task"], record["seed"], record["repeat"],
                                  record["passed"], record["key"]),
                                 (mode, episode["task"], episode["seed"], episode["repeat"], episode["passed"],
                                  f"{mode}/{episode['task']}/r{episode['repeat']}/s{episode['seed']}"))
                first = record["turns"][0]
                # The malformed arguments survive the HTTP round trip byte for byte.
                self.assertEqual([(c["name"], c["arguments"], c["valid"]) for c in first["tool_calls"]],
                                 [("read_file", "{path: oops", False), ("teleport", "{}", False)])
                self.assertEqual(first["reasoning"], "plan " * 10)
                self.assertEqual(len(record["turns"]), episode["steps"])
                self.assertEqual([c["valid"] for c in record["turns"][1]["tool_calls"]], [True])
                self.assertEqual(record["turns"][-1]["content"], "Done: summary.")

    def test_shared_server_records_decode_stats(self):
        work = scratch_dir()
        model = work / "model.gguf"
        model.write_bytes(b"GGUF")
        done = self.run_cli("--model", "ling", "--model-path", str(model), "--gufo", str(HERE / "fake_gufo.py"),
                            "--work", str(work), "--part", "agent", "--agent-tasks", "t01",
                            "--agent-repeats", "1", "--port", "18492", "--startup-timeout", "60")
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        serve = json.loads(done.stdout)["serve"]
        self.assertGreater(serve["decode"]["agent"]["requests"], 0)

    def test_bad_arguments(self):
        self.assertEqual(self.run_cli("--model", "nope").returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
