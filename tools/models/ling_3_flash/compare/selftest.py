#!/usr/bin/env python3
"""Offline checks of the harness itself (no GPU, no model): sandbox confinement,
tool-call validation, grader anti-cheating, statistics, harness exit codes.
Run: python3 selftest.py"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import agent
import common

HERE = Path(__file__).resolve().parent


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
        # gutting the visible tests and adding a passing one must not pass an unfixed repo
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
        work = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
        row = agent.run_episode(self.task, None, work, 1, 4, 60, 128, 4096,
                                complete=lambda *a: replies.pop(0))
        first, second = row["turns"]
        self.assertEqual(len(first["prompt_sha256"]), 64)
        self.assertNotEqual(first["prompt_sha256"], second["prompt_sha256"])
        self.assertEqual(len(first["tool_result_sha256"]), 1)
        self.assertEqual((second["cache_hit"], second["cached_tokens"], second["prefill_tokens"]), (True, 48, 32))


    def test_episode_reads_gufo_metrics_from_usage(self):
        # gufo serve puts cache_hit / prefill_tokens under usage.gufo, not a top-level "metrics"
        replies = [{"http_status": 200, "seconds": 0.1, "finish_reason": "stop", "content": "done",
                    "reasoning": "", "tool_calls": [], "metrics": {},
                    "usage": {"prompt_tokens": 80, "completion_tokens": 3, "cached_tokens": 48,
                              "draft_tokens": 2, "draft_tokens_accepted": 1,
                              "gufo": {"cache_hit": True, "prefill_tokens": 32}}}]
        work = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
        row = agent.run_episode(self.task, None, work, 1, 4, 60, 128, 4096,
                                complete=lambda *a: replies.pop(0), repeat=0)
        turn = row["turns"][0]
        self.assertEqual((turn["cache_hit"], turn["cached_tokens"], turn["prefill_tokens"]), (True, 48, 32))
        self.assertEqual((turn["finish_reason"], turn["completion_tokens"], turn["draft_tokens"],
                          turn["draft_tokens_accepted"]), ("stop", 3, 2, 1))
        self.assertEqual(row["repeat"], 0)


class StatsTest(unittest.TestCase):
    def test_fisher_exact(self):
        # 10/14 against 14/14: P(X <= 10) = C(14,10) / C(28,24); two-sided doubles the symmetric tail
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
        import argparse
        import compare
        work = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
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
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(HERE / "compare.py"), *args], capture_output=True, text=True)

    def test_missing_artifact_is_harness_failure(self):
        done = self.run_cli("--model", "ling", "--gufo", "/bin/true", "--work", tempfile.mkdtemp(),
                            "--part", "agent")
        self.assertEqual(done.returncode, 2)
        self.assertIn("model artifact not found", done.stdout)

    def test_agent_speculative_modes_end_to_end(self):
        # fake gufo -> mock server, two serve modes x 2 repeats x 2 tasks
        work = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
        model = work / "model.gguf"
        model.write_bytes(b"GGUF")
        done = self.run_cli("--model", "ling", "--model-path", str(model), "--gufo", str(HERE / "fake_gufo.py"),
                            "--work", str(work), "--part", "agent", "--agent-tasks", "t01,t02",
                            "--agent-repeats", "2", "--agent-speculative", "mtp,off", "--port", "18491",
                            "--startup-timeout", "60")
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        import json
        summary = json.loads(done.stdout)
        agent_phase = summary["phases"]["agent"]
        self.assertEqual(set(agent_phase["modes"]), {"mtp", "off"})
        self.assertIn("--speculative", agent_phase["serve"]["off"]["command"])
        self.assertEqual(agent_phase["serve"]["off"]["command"][-1], "off")
        decode = agent_phase["serve"]["mtp"]["decode"]["agent"]
        self.assertGreater(decode["requests"], 0)
        self.assertEqual(decode["draft_acceptance"], 0.75)
        comparison = agent_phase["comparison"]
        self.assertEqual(comparison["modes"]["mtp"]["episodes"], 4)
        self.assertIsNotNone(comparison["fisher_exact"]["p_two_sided"])
        self.assertEqual(comparison["paired"]["pairs"], 4)
        seeds = {m: sorted((e["task"], e["repeat"], e["seed"]) for e in agent_phase["modes"][m]["episodes_compact"])
                 for m in ("mtp", "off")}
        self.assertEqual(seeds["mtp"], seeds["off"])
        self.assertEqual(len({s for _, _, s in seeds["mtp"]}), 4)  # one seed per (task, repeat)
        turn = agent_phase["modes"]["mtp"]["episodes_compact"][0]["turns"][1]
        self.assertIsNotNone(turn["cache_hit"])
        self.assertIsNotNone(turn["prefill_tokens"])

    def test_shared_server_records_decode_stats(self):
        work = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
        model = work / "model.gguf"
        model.write_bytes(b"GGUF")
        done = self.run_cli("--model", "ling", "--model-path", str(model), "--gufo", str(HERE / "fake_gufo.py"),
                            "--work", str(work), "--part", "agent", "--agent-tasks", "t01",
                            "--agent-repeats", "1", "--port", "18492", "--startup-timeout", "60")
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        import json
        serve = json.loads(done.stdout)["serve"]
        self.assertGreater(serve["decode"]["agent"]["requests"], 0)

    def test_bad_arguments(self):
        self.assertEqual(self.run_cli("--model", "nope").returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
