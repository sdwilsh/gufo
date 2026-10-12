#!/usr/bin/env python3
"""Lists every (layer, position) where a candidate's MoE top-8 expert set
differs from a reference run, first difference first.

  compare_routing.py --ref DIR [--json OUT] NAME=DIR [NAME=DIR ...]

Every DIR is a gufo-ling-dump-v1 directory (reference_probe, or the GPU
layer_check teacher_prefill / teacher_decode / free directories) holding
`router_ids.lNN.i32`, int32 [tokens, 8]. Sets are compared, so the order
inside a row is irrelevant. The router margin comes from the reference dir's
`router_top_sel.lNN.f32` (float [tokens, 10], biased scores of the ten best
experts, selection order; margin = entry 7 minus entry 8) and
`router_top_ids.lNN.i32`. A candidate that also has these files (the CPU
reference runs do, the GPU layer_check dumps do not) gets its own margin
column. Results are ordered by (position, layer): the compute order of a
token-by-token pass, so the first row is the causal root of any later flip.
"""
import argparse
import json
import os
import struct
import sys


def read(path, fmt, width):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    vals = struct.unpack("<%d%s" % (n, fmt), data[: n * 4])
    return [vals[i : i + width] for i in range(0, n - width + 1, width)]


def layer_file(d, name, ext, layer):
    return os.path.join(d, "%s.l%02d.%s" % (name, layer, ext))


def load(d):
    """Returns {layer: dict(ids=[rows], top_ids=[rows]|None, sel=[rows]|None)}."""
    out = {}
    for layer in range(0, 64):
        p = layer_file(d, "router_ids", "i32", layer)
        if not os.path.exists(p):
            continue
        entry = {"ids": read(p, "i", 8), "top_ids": None, "sel": None}
        p = layer_file(d, "router_top_ids", "i32", layer)
        if os.path.exists(p):
            entry["top_ids"] = read(p, "i", 10)
        p = layer_file(d, "router_top_sel", "f32", layer)
        if os.path.exists(p):
            entry["sel"] = read(p, "f", 10)
        out[layer] = entry
    return out


def margin(entry, pos):
    if entry is None or entry["sel"] is None or pos >= len(entry["sel"]):
        return None
    s = entry["sel"][pos]
    return s[7] - s[8]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="reference dump dir (the F64 run)")
    ap.add_argument("--json", help="write the full result as JSON")
    ap.add_argument("cands", nargs="+", metavar="NAME=DIR")
    args = ap.parse_args()

    ref = load(args.ref)
    if not ref:
        sys.exit("no router_ids.lNN.i32 in " + args.ref)
    result = {}
    for spec in args.cands:
        name, _, d = spec.partition("=")
        cand = load(d)
        diffs = []
        compared = 0
        for layer, r in sorted(ref.items()):
            c = cand.get(layer)
            if c is None:
                continue
            for pos in range(min(len(r["ids"]), len(c["ids"]))):
                compared += 1
                rs, cs = set(r["ids"][pos]), set(c["ids"][pos])
                if rs == cs:
                    continue
                row = {
                    "layer": layer,
                    "pos": pos,
                    "ref_only": sorted(rs - cs),
                    "cand_only": sorted(cs - rs),
                    "ref_margin": margin(r, pos),
                    "cand_margin": margin(c, pos),
                }
                if r["top_ids"] is not None:
                    row["ref_8th_9th"] = [r["top_ids"][pos][7], r["top_ids"][pos][8]]
                diffs.append(row)
        diffs.sort(key=lambda x: (x["pos"], x["layer"]))
        result[name] = {"compared": compared, "differences": diffs}
        print("== %s (%s): %d of %d (layer,pos) sets differ from the reference"
              % (name, d, len(diffs), compared))
        print("   layer  pos  ref-only -> cand-only   ref margin  cand margin")
        for x in diffs:
            fm = lambda v: "      n/a" if v is None else "%9.3e" % v
            print("   %5d %4d  %s -> %s   %s  %s" % (
                x["layer"], x["pos"], x["ref_only"], x["cand_only"],
                fm(x["ref_margin"]), fm(x["cand_margin"])))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
