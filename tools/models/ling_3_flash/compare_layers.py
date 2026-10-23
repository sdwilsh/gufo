#!/usr/bin/env python3
"""Per-layer relative-L2 between two dump directories.

Both directories hold `index.json` plus `<name>.lNN.{f32,i32}` files as written
by `reference_probe` (and by the GPU probe, which follows the same format).
`A` is the candidate (GPU), `B` the reference; rel-L2 = ||A - B|| / ||B||
over all tokens of one tensor, plus the worst single token.

    compare_layers.py A B [--mode teacher|free] [--names h_out,kda_o]
                          [--layers 0,5,41] [--gate 1e-2] [--json OUT]

teacher: A was produced with `--teacher-input-dir B`, every block saw the
         reference's own input, so each block's `h_out` error is local; the
         gate (default 1e-2 on `h_out`) fails the exit code.
free:    A ran the whole model from the tokens; errors compound through
         routing flips, so numbers are reported but never gate.

Integer tensors (router_ids) report the fraction of tokens whose expert set
is identical and the mean set overlap.
"""

import argparse
import json
import sys

import numpy as np


def load_index(path):
    with open(f"{path}/index.json") as f:
        idx = json.load(f)
    return {(e["name"], e["layer"]): e for e in idx["entries"]}, idx


def load(path, entry):
    dt = "<f4" if entry["dtype"] == "f32" else "<i4"
    a = np.fromfile(f"{path}/{entry['file']}", dtype=dt)
    return a.reshape(entry["shape"][0], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", help="candidate dump dir")
    ap.add_argument("b", help="reference dump dir")
    ap.add_argument("--mode", choices=["teacher", "free"], default="free")
    ap.add_argument("--names", default=None, help="comma separated; default all")
    ap.add_argument("--layers", default=None, help="comma separated; default all")
    ap.add_argument("--gate", type=float, default=1e-2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ea, _ = load_index(args.a)
    eb, _ = load_index(args.b)
    names = set(args.names.split(",")) if args.names else None
    layers = {int(x) for x in args.layers.split(",")} if args.layers else None

    rows = []
    for key in sorted(set(ea) & set(eb), key=lambda k: (k[1], k[0])):
        name, layer = key
        if name == "logits" or (names and name not in names):
            continue
        if layers is not None and layer not in layers:
            continue
        a, b = load(args.a, ea[key]), load(args.b, eb[key])
        n = min(len(a), len(b))
        if len(a) != len(b):
            print(f"warning: {name} l{layer}: {len(a)} vs {len(b)} tokens, using {n}",
                  file=sys.stderr)
        a, b = a[:n], b[:n]
        row = {"name": name, "layer": layer, "tokens": n}
        if ea[key]["dtype"] == "i32":
            same = [set(x.tolist()) == set(y.tolist()) for x, y in zip(a, b)]
            overlap = [len(set(x.tolist()) & set(y.tolist())) / len(y) for x, y in zip(a, b)]
            row.update(set_match=float(np.mean(same)), overlap=float(np.mean(overlap)))
        else:
            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            den = np.linalg.norm(b64)
            per_tok = np.linalg.norm(a64 - b64, axis=1) / np.maximum(
                np.linalg.norm(b64, axis=1), 1e-30
            )
            row.update(
                rel_l2=float(np.linalg.norm(a64 - b64) / max(den, 1e-30)),
                max_token_rel_l2=float(per_tok.max()),
                worst_token=int(per_tok.argmax()),
            )
        rows.append(row)

    if not rows:
        print("no common (name, layer) entries", file=sys.stderr)
        return 1
    print(f"{'layer':>5} {'name':<11} {'tokens':>6} {'rel-L2':>10} {'worst tok':>10}")
    failed = False
    for r in rows:
        if "rel_l2" in r:
            flag = ""
            if args.mode == "teacher" and r["name"] == "h_out" and r["rel_l2"] >= args.gate:
                flag = "  FAIL"
                failed = True
            print(f"{r['layer']:5d} {r['name']:<11} {r['tokens']:6d} {r['rel_l2']:10.3e} "
                  f"{r['max_token_rel_l2']:10.3e}{flag}")
        else:
            print(f"{r['layer']:5d} {r['name']:<11} {r['tokens']:6d} "
                  f"set match {r['set_match']:.3f}  overlap {r['overlap']:.3f}")
    h = [r for r in rows if r["name"] == "h_out"]
    if h:
        print(f"h_out: worst layer {max(h, key=lambda r: r['rel_l2'])['layer']} "
              f"rel-L2 {max(r['rel_l2'] for r in h):.3e} (gate {args.gate:g}, "
              f"{'enforced' if args.mode == 'teacher' else 'report only'})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
