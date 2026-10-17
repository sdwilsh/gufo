"""Decision logic for the Hermes replay's decode-lineage report-only controls."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_hermes import control_gate, update_decode_lineage

# Recorded strip steps from a hardware run (rc12 run-20261017T215556-8) in which the
# model emitted empty reasoning at step4, so step5 reused 15,555 = 15,443 + 112 decoded.
# "pc" is the previous request's completion_tokens. The fork's previous request is step5.
STEPS = json.loads('''[
 {"name":"step1","prompt":2262,"cached":0,"pp":null,"pc":0,"ok":true},
 {"name":"step2","prompt":4910,"cached":2262,"pp":2262,"pc":95,"ok":true},
 {"name":"step3","prompt":10167,"cached":4910,"pp":4910,"pc":139,"ok":true},
 {"name":"step4","prompt":15443,"cached":10167,"pp":10167,"pc":147,"ok":true},
 {"name":"step5","prompt":18140,"cached":15555,"pp":15443,"pc":112,"ok":false},
 {"name":"next_step_with_fork_running","prompt":22146,"cached":18276,"pp":18140,"pc":136,"ok":true},
 {"name":"review_fork","prompt":22162,"cached":22137,"pp":18140,"pc":136,"ok":false}
]''')


def healthy(steps):
    """Same run where no step reused decoded tokens: cached == previous prompt."""
    out = deepcopy(steps)
    for step in out:
        if step["pp"] is not None:
            step["cached"] = step["pp"]
    out[-1]["cached"] = 22137  # fork still reuses the sibling's prefill
    for step in out:
        step["ok"] = True
    return out


def evaluate(steps, echo=False):
    """Names of controls recorded as failures and as report-only, and the lineage."""
    lineage, failed, reported = None, [], []
    for step in steps:
        lineage = update_decode_lineage(lineage, step["name"], step["cached"],
                                        step["pp"], step["pc"])
        mismatch = not step["ok"]
        _, gated = control_gate(echo, lineage)
        if step.get("prompt_mismatch"):  # template/tokenizer bug: always strict
            failed.append(step["name"])
        elif mismatch:
            (failed if gated else reported).append(step["name"])
    return failed, reported, lineage


class DecodeLineageTest(unittest.TestCase):
    def test_recorded_run_reports_step5_and_fork(self):
        failed, reported, lineage = evaluate(STEPS)
        self.assertEqual(failed, [])
        self.assertEqual(reported, ["step5", "review_fork"])
        self.assertEqual(lineage, "step5")

    def test_cached_equal_to_previous_prompt_stays_strict(self):
        steps = deepcopy(STEPS[:4])
        steps[3]["ok"] = False
        self.assertEqual(evaluate(steps), (["step4"], [], None))

    def test_mismatch_before_decoded_step_still_fails(self):
        steps = deepcopy(STEPS)
        steps[2]["ok"] = False
        failed, reported, _ = evaluate(steps)
        self.assertEqual(failed, ["step3"])
        self.assertEqual(reported, ["step5", "review_fork"])

    def test_healthy_run_fork_reuse_does_not_set_lineage(self):
        steps = healthy(STEPS)
        steps[-1]["ok"] = False
        self.assertEqual(evaluate(steps), (["review_fork"], [], None))

    def test_fork_reuse_beyond_frontier_bound_keeps_step5_lineage(self):
        self.assertEqual(update_decode_lineage(None, "review_fork", 22137, 18140, 136), None)
        self.assertEqual(update_decode_lineage("step5", "review_fork", 22137, 18140, 136), "step5")
        self.assertEqual(evaluate(STEPS)[2], "step5")

    def test_prompt_token_mismatch_under_lineage_still_fails(self):
        steps = deepcopy(STEPS)
        steps[4]["prompt_mismatch"] = True
        failed, reported, _ = evaluate(steps)
        self.assertEqual(failed, ["step5"])
        self.assertEqual(reported, ["review_fork"])

    def test_frontier_bounds(self):
        self.assertEqual(update_decode_lineage(None, "s", 15555, 15443, 112), "s")
        self.assertIsNone(update_decode_lineage(None, "s", 15556, 15443, 112))
        self.assertIsNone(update_decode_lineage(None, "s", 15443, 15443, 112))
        self.assertIsNone(update_decode_lineage(None, "s", 10, None, 0))

    def test_lineage_is_sticky(self):
        self.assertEqual(update_decode_lineage("step5", "later", 1, 99999, 0), "step5")

    def test_gate_labels(self):
        self.assertEqual(control_gate(True, None), ("answer-level-report", False))
        self.assertEqual(control_gate(False, None), ("exact", True))
        self.assertEqual(control_gate(False, "step5"), ("exact-report", False))


if __name__ == "__main__":
    unittest.main()
