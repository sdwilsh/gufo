#!/usr/bin/env python3
"""Compare gufo greedy token streams with the llama.cpp oracle (greedy.jsonl).

Both files are JSON lines with at least {"name", "tokens": [ids]}. The oracle
(run_oracle.sh) also carries "prompt_ids" and "top" (per generated token a list
of {"id", "logprob"} sorted by descending logprob).

A prompt passes when the streams are identical for >= --min-match tokens
(default 32; shorter only if both streams end together), or when the first
divergence sits at a logged near-tie of the oracle (top-1 minus top-2 logprob
< --tie-gap nats, default 0.05).

    compare_greedy.py ORACLE.jsonl GUFO.jsonl [--min-match 32] [--tie-gap 0.05]
"""

import argparse
import json
import sys


def read(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["name"]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("oracle")
    ap.add_argument("gufo")
    ap.add_argument("--min-match", type=int, default=32)
    ap.add_argument("--tie-gap", type=float, default=0.05)
    args = ap.parse_args()

    oracle, gufo = read(args.oracle), read(args.gufo)
    ok_all = True
    for name, o in oracle.items():
        g = gufo.get(name)
        if g is None:
            print(f"{name}: MISSING in gufo output")
            ok_all = False
            continue
        if "prompt_ids" in o and "prompt_ids" in g and o["prompt_ids"] != g["prompt_ids"]:
            print(f"{name}: prompt ids differ (tokenizer mismatch)")
            ok_all = False
            continue
        ot, gt = o["tokens"], g["tokens"]
        n = min(len(ot), len(gt))
        d = next((i for i in range(n) if ot[i] != gt[i]), None)
        if d is None:
            matched = n
            ok = n >= args.min_match or len(ot) == len(gt)
            note = f"identical over {n} tokens (oracle {len(ot)}, gufo {len(gt)})"
        else:
            matched = d
            gap = None
            top = o.get("top")
            if top and d < len(top) and len(top[d]) >= 2:
                gap = top[d][0]["logprob"] - top[d][1]["logprob"]
            tie = gap is not None and gap < args.tie_gap
            ok = matched >= args.min_match or tie
            note = (f"diverges at token {d}: oracle {ot[d]} vs gufo {gt[d]}"
                    + (f", oracle top-2 gap {gap:.4f} nats" if gap is not None else "")
                    + (" (near tie)" if tie else ""))
        print(f"{name}: {'PASS' if ok else 'FAIL'}  matched {matched}  {note}")
        ok_all &= ok
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
