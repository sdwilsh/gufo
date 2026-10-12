#!/usr/bin/env python3
"""Check the task suite itself: the unmodified repo must FAIL grading, the
reference solution must PASS, and task.json's expected_tests must match.

    validate_tasks.py [--update]      # --update rewrites expected_tests
Exit 1 when any task is malformed.  Runs in a temp dir, needs no model.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    tmp = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    bad = 0
    tasks = agent.load_tasks()
    with tempfile.TemporaryDirectory(dir=tmp) as work:
        for task in tasks:
            task["expected_tests"] = 0
            base = Path(work) / (task["id"] + "-base")
            agent.copy_tree(task["dir"] / "repo", base)
            before = agent.grade(task, base)
            sol = Path(work) / (task["id"] + "-sol")
            agent.copy_tree(task["dir"] / "repo", sol)
            agent.copy_tree(task["dir"] / "solution", sol)
            after = agent.grade(task, sol)
            ok = (not before["passed"] and before["exit_code"] != 0 and after["exit_code"] == 0
                  and after["tests_run"])
            visible = before["tests_run"]
            if ok and args.update:
                meta_path = task["dir"] / "task.json"
                meta = json.loads(meta_path.read_text())
                meta["expected_tests"] = after["tests_run"]
                meta_path.write_text(json.dumps(meta, indent=1) + "\n")
            expected = json.loads((task["dir"] / "task.json").read_text())["expected_tests"]
            if ok and expected != after["tests_run"]:
                ok = False
            print(f"{'ok ' if ok else 'BAD'} {task['id']:24s} {task['category']:9s} baseline "
                  f"exit={before['exit_code']} ran={visible}  solution exit={after['exit_code']} "
                  f"ran={after['tests_run']} expected={expected}")
            if not ok:
                bad += 1
                print(before["output_tail"][-600:], after["output_tail"][-600:], sep="\n---\n")
    print(f"{len(tasks)} tasks, {bad} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
