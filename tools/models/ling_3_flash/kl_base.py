#!/usr/bin/env python3
"""Decoder and comparator for llama.cpp `--kl-divergence-base` files.

Format (llama.cpp b11146, tools/perplexity/perplexity.cpp, `perplexity()` with
`--kl-divergence-base`), little endian:

    char[8]   "_logits_"
    uint32    n_ctx
    int32     n_vocab
    int32     n_chunk
    int32     tokens[n_chunk * n_ctx]
    per chunk, n_ctx - 1 - first rows (first = n_ctx // 2), each row of
    nv = 2 * ((n_vocab + 1) // 2) + 4 uint16 values:
        float32 scale, float32 min_log_prob   (the first 4 uint16)
        uint16  code[n_vocab] (+ 1 pad when n_vocab is odd)
    log_prob[v] = scale * code[v] + min_log_prob; code 0 stands for every
    token whose logit was below max_logit - 16 (log_prob is then the floor
    `min_log_prob`).

Row i of a chunk is the distribution after seeing chunk positions 0..first+i;
it predicts tokens[chunk][first + i + 1].

Needs numpy (`python3 -m venv --system-site-packages V && V/bin/pip install
numpy`). Commands:

    kl_base.py info BASE
    kl_base.py tokens BASE [--chunk N]        # token ids, one line
    kl_base.py selftest BASE                  # synthetic CPU-vs-CPU + BASE against itself
    kl_base.py compare BASE --logits-dir DIR [--table] [--json OUT]
    kl_base.py compare BASE --logits FILE.f32 --positions 256,257,...
    kl_base.py compare-bases BASE TEST_BASE [--table] [--json OUT]

`DIR` is a reference_probe dump (index.json + logits.f32) produced with
`--chunk <n_ctx> --logits-from-pos <n_ctx/2>` on the tokens of BASE.

Every statistic is computed exactly as llama-perplexity `--kl-divergence` does
(tools/perplexity/perplexity.cpp, `log_softmax(..., base_log_prob, ...)` and
`kl_divergence()` at b11146 / 7fe450e19): float32 arithmetic per term, the
base log-prob is `scale * code + min_log_prob` in float32, the test
log-sum-exp is `float(log(double sum expf(x - max)))`, the KL of a position is
sum over base log-prob > -16 (strictly) of `p_base * (lp_base - logit +
(max + lse))` accumulated in a double with NO renormalisation of the base,
"same top" compares the first argmax of both sides, and mean +- stderr is
`sqrt((sum2/n - mean^2) / (n - 1))` (0 for n <= 10). Run `selftest` for
validation.

`compare-bases` has no test logits (llama.cpp only writes the test side as a
base file). It uses the test base's stored normalised log-probs as the test
log-softmax (`max + lse` = 0). Residuals versus llama-perplexity on the real
logits: (1) the test side carries its own uint16 quantisation noise (zero mean,
at most 16/65535/2 per log-prob); (2) test tokens below its -16 floor are
taken at the floor value (an upper bound; they only matter where the base
probability is > e^-16); (3) ties at the test's top code (65535) resolve to the
lowest id instead of the true top-1.
"""

import argparse
import json
import math
import sys

import numpy as np

MAGIC = b"_logits_"


class KlBase:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(8) != MAGIC:
                raise ValueError(f"{path}: not a llama.cpp kl-divergence base file")
            self.n_ctx = int(np.frombuffer(f.read(4), "<u4")[0])
            self.n_vocab = int(np.frombuffer(f.read(4), "<i4")[0])
            self.n_chunk = int(np.frombuffer(f.read(4), "<i4")[0])
            n_tokens = self.n_ctx * self.n_chunk
            self.tokens = np.frombuffer(f.read(4 * n_tokens), "<i4").reshape(
                self.n_chunk, self.n_ctx
            )
        self.first = self.n_ctx // 2
        self.rows_per_chunk = self.n_ctx - 1 - self.first
        self.nv = 2 * ((self.n_vocab + 1) // 2) + 4
        self.data_offset = 8 + 12 + 4 * self.n_ctx * self.n_chunk
        expect = self.data_offset + 2 * self.nv * self.rows_per_chunk * self.n_chunk
        import os

        size = os.path.getsize(path)
        if size != expect:
            raise ValueError(f"{path}: size {size} != expected {expect}")
        self.raw = np.memmap(
            path,
            dtype="<u2",
            mode="r",
            offset=self.data_offset,
            shape=(self.n_chunk, self.rows_per_chunk, self.nv),
        )

    def positions(self):
        """Yield (chunk, in-chunk position, global position) of every row."""
        for c in range(self.n_chunk):
            for i in range(self.rows_per_chunk):
                p = self.first + i
                yield c, p, c * self.n_ctx + p

    def target(self, chunk, pos):
        return int(self.tokens[chunk, pos + 1])

    def logprobs(self, chunk, row):
        """float32 [n_vocab] log-probabilities of one row."""
        r = np.asarray(self.raw[chunk, row])
        head = r[:4].view("<f4")
        scale, min_lp = float(head[0]), float(head[1])
        codes = r[4 : 4 + self.n_vocab].astype(np.float32)
        return codes * np.float32(scale) + np.float32(min_lp)


F32 = np.float32


def mean_and_uncertainty(total, total2, count):
    """llama-perplexity's mean_and_uncertainty (doubles)."""
    if count < 1:
        return 0.0, 0.0
    f = total / count
    df = total2 / count - f * f
    df = (df / (count - 1)) ** 0.5 if df > 0 and count > 10 else 0.0
    return f, df


def covariance(suma, sumb, sumab, count):
    if count < 10:
        return 0.0
    return (sumab / count - (suma / count) * (sumb / count)) / (count - 1)


def llama_row(scale, min_lp, codes, tok, logits=None, test=None):
    """One position, exactly llama-perplexity's per-row KL computation.

    scale, min_lp: the base row's float32 head; codes: its uint16 codes;
    tok: the target token. Give `logits` (float32 [n_vocab], the tested
    model's raw logits) or `test=(scale, min_lp, codes)` of a test base file.
    Returns a dict holding the per-row values and the additive terms that
    llama accumulates (`sums`)."""
    scale, min_lp = F32(scale), F32(min_lp)
    lp_base = codes.astype(F32) * scale + min_lp
    if logits is not None:
        logits = np.asarray(logits, dtype=F32)
        imax = int(np.argmax(logits))
        max_logit = logits[imax]
        sum_exp = float(np.exp(logits - max_logit).astype(np.float64).sum())
        lse = F32(np.log(sum_exp))
        nll = F32(max_logit + lse - logits[tok])
        max_lse = F32(max_logit + lse)
    else:
        ts, tm, tc = test
        logits = tc.astype(F32) * F32(ts) + F32(tm)  # normalised log-probs
        imax = int(np.argmax(logits))
        nll = F32(-logits[tok])
        max_lse = F32(0.0)
    nll_base = F32(-(lp_base[tok]))
    mask = lp_base > F32(-16.0)
    lb = lp_base[mask]
    term = np.exp(lb) * ((lb - logits[mask]) + max_lse)  # float32 like C++
    kl = float(term.astype(np.float64).sum())
    imax_base = int(np.argmax(lp_base))
    p_diff = F32(np.exp(-nll) - np.exp(-nll_base))
    p_diff2 = float(p_diff * p_diff)
    return {
        "kl": kl,
        "top1_base": imax_base,
        "top1_test": imax,
        "nll_base": float(nll_base),
        "nll_test": float(nll),
        "p_diff": float(p_diff),
        "p_diff2": p_diff2,
        "nn": float(F32(nll * nll)),
        "nbnb": float(F32(nll_base * nll_base)),
        "nnb": float(F32(nll * nll_base)),
    }


def base_row(base, chunk, row):
    r = np.asarray(base.raw[chunk, row])
    head = r[:4].view("<f4")
    return float(head[0]), float(head[1]), r[4 : 4 + base.n_vocab]


def _decorate(out, base, chunk, pos):
    out.update(
        chunk=chunk,
        pos=pos,
        global_pos=chunk * base.n_ctx + pos,
    )
    return out


def compare(base, logits_rows, positions):
    """logits_rows: [n, n_vocab] float32 for global token `positions`."""
    index = {int(p): k for k, p in enumerate(positions)}
    out = []
    for chunk, pos, gpos in base.positions():
        k = index.get(gpos)
        if k is None:
            continue
        scale, min_lp, codes = base_row(base, chunk, pos - base.first)
        r = llama_row(scale, min_lp, codes, base.target(chunk, pos), logits=logits_rows[k])
        out.append(_decorate(r, base, chunk, pos))
    return out


def compare_bases(base, test):
    """Base-vs-base (see the module docstring for the residual)."""
    if (base.n_ctx, base.n_vocab, base.n_chunk) != (test.n_ctx, test.n_vocab, test.n_chunk):
        raise ValueError("base files have different geometry")
    if not np.array_equal(base.tokens, test.tokens):
        print("warning: the base files hold different tokens", file=sys.stderr)
    out = []
    for chunk, pos, _ in base.positions():
        row = pos - base.first
        scale, min_lp, codes = base_row(base, chunk, row)
        r = llama_row(
            scale, min_lp, codes, base.target(chunk, pos), test=base_row(test, chunk, row)
        )
        out.append(_decorate(r, base, chunk, pos))
    return out


def summarize(rows):
    """llama-perplexity's reductions over `rows` (doubles, same formulas)."""
    n = len(rows)
    col = lambda k: sum(r[k] for r in rows)  # noqa: E731
    kl = [r["kl"] for r in rows]
    nll, nll_b = col("nll_test"), col("nll_base")
    log_ppl = mean_and_uncertainty(nll, col("nn"), n)
    log_ppl_b = mean_and_uncertainty(nll_b, col("nbnb"), n)
    cov = covariance(nll, nll_b, col("nnb"), n)
    ratio = log_ppl[0] - log_ppl_b[0]
    ratio_unc = max(log_ppl[1] ** 2 + log_ppl_b[1] ** 2 - 2.0 * cov, 0.0) ** 0.5
    klm = mean_and_uncertainty(col("kl"), sum(k * k for k in kl), n)
    same = sum(1 for r in rows if r["top1_base"] == r["top1_test"])
    top = same / n
    top_unc = (top * (1 - top) / (n - 1)) ** 0.5 if n > 1 else 0.0
    pdm = mean_and_uncertainty(col("p_diff2"), sum(r["p_diff2"] ** 2 for r in rows), n)
    pd_rms = pdm[0] ** 0.5
    kls = np.sort(np.array(kl, dtype=np.float32))
    return {
        "positions": n,
        "kl_mean": klm[0],
        "kl_stderr": klm[1],
        "kl_median": float(np.median(kls)),
        "kl_p99": float(np.percentile(kls, 99)),
        "kl_max": float(kls[-1]),
        "top1_agreement": top,
        "top1_stderr": top_unc,
        "ppl_test": math.exp(log_ppl[0]),
        "ppl_base": math.exp(log_ppl_b[0]),
        "ln_ppl_ratio": ratio,
        "ln_ppl_ratio_stderr": ratio_unc,
        "dp_mean": mean_and_uncertainty(col("p_diff"), col("p_diff2"), n)[0],
        "dp_rms": pd_rms,
        "nll_base_mean": nll_b / n,
        "nll_test_mean": nll / n,
    }


def print_summary(s):
    print(
        f"positions {s['positions']}  Mean KLD {s['kl_mean']:.6f} +- {s['kl_stderr']:.6f}  "
        f"median {s['kl_median']:.6f} p99 {s['kl_p99']:.6f} max {s['kl_max']:.6f}  "
        f"Same top p {100 * s['top1_agreement']:.3f} +- {100 * s['top1_stderr']:.3f} %  "
        f"PPL(Q) {s['ppl_test']:.6f} PPL(base) {s['ppl_base']:.6f}  "
        f"ln(PPL(Q)/PPL(base)) {s['ln_ppl_ratio']:.6f} +- {s['ln_ppl_ratio_stderr']:.6f}  "
        f"RMS dp {100 * s['dp_rms']:.3f} %"
    )


def print_table(rows):
    print("chunk  pos      kl  top1_base top1_test  nll_base  nll_test")
    for r in rows:
        print(
            f"{r['chunk']:5d} {r['pos']:4d} {r['kl']:9.5f} {r['top1_base']:9d}"
            f" {r['top1_test']:9d} {r['nll_base']:9.4f} {r['nll_test']:9.4f}"
            f"{'' if r['top1_base'] == r['top1_test'] else '  *'}"
        )


def write_json(path, s, rows):
    keep = ("chunk", "pos", "global_pos", "kl", "top1_base", "top1_test", "nll_base", "nll_test")
    with open(path, "w") as f:
        json.dump({"summary": s, "rows": [{k: r[k] for k in keep} for r in rows]}, f, indent=1)


def synth_selftest(n_vocab=4096, n_rows=64, seed=1):
    """Quantise random logits with a port of llama's writer and compare the
    same logits back: the CPU-vs-CPU case, which llama prints as 0.000000."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rows):
        x = (rng.standard_normal(n_vocab) * rng.uniform(1.0, 5.0)).astype(F32)
        mx, mn = x.max(), x.min()
        mn = max(mn, F32(mx - 16))
        lse = F32(np.log(float(np.exp(x - mx).astype(np.float64).sum())))
        min_lp = F32(mn - mx - lse)
        scale = F32((mx - mn) / F32(65535.0))
        inv = F32(1) / scale
        codes = np.where(x > mn, np.rint(inv * (x - mn)), 0).astype(np.uint16)
        tok = int(rng.integers(n_vocab))
        out.append(llama_row(scale, min_lp, codes, tok, logits=x) | {"chunk": 0, "pos": 0})
    return out


def load_logits(args):
    if args.logits_dir:
        with open(f"{args.logits_dir}/index.json") as f:
            idx = json.load(f)
        entry = next(e for e in idx["entries"] if e["name"] == "logits")
        positions = entry["positions"]
        data = np.fromfile(f"{args.logits_dir}/{entry['file']}", dtype="<f4")
        return data.reshape(len(positions), entry["shape"][1]), positions, idx.get("tokens")
    positions = [int(x) for x in args.positions.split(",")]
    data = np.fromfile(args.logits, dtype="<f4")
    return data.reshape(len(positions), -1), positions, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info")
    p.add_argument("base")
    p = sub.add_parser("tokens")
    p.add_argument("base")
    p.add_argument("--chunk", type=int, default=None, help="one chunk; default all")
    p = sub.add_parser("selftest")
    p.add_argument("base")
    p = sub.add_parser("compare-bases")
    p.add_argument("base")
    p.add_argument("test_base")
    p.add_argument("--table", action="store_true", help="per-position table")
    p.add_argument("--json", help="write summary and rows as JSON")
    p = sub.add_parser("compare")
    p.add_argument("base")
    p.add_argument("--logits-dir")
    p.add_argument("--logits")
    p.add_argument("--positions")
    p.add_argument("--table", action="store_true", help="per-position table")
    p.add_argument("--json", help="write summary and rows as JSON")
    args = ap.parse_args()

    base = KlBase(args.base)
    if args.cmd == "info":
        print(
            f"n_ctx={base.n_ctx} n_vocab={base.n_vocab} n_chunk={base.n_chunk} "
            f"scored_rows_per_chunk={base.rows_per_chunk} first={base.first}"
        )
        return 0
    if args.cmd == "tokens":
        t = base.tokens if args.chunk is None else base.tokens[args.chunk : args.chunk + 1]
        print(" ".join(str(int(x)) for x in t.reshape(-1)))
        return 0

    if args.cmd == "selftest":
        syn = summarize(synth_selftest())
        print("synthetic CPU-vs-CPU: ", end="")
        print_summary(syn)
        s = summarize(compare_bases(base, base))
        print("base vs itself:       ", end="")
        print_summary(s)
        ok = abs(syn["kl_mean"]) < 5e-7 and s["kl_mean"] == 0.0 and s["top1_agreement"] == 1.0
        return 0 if ok else 1

    if args.cmd == "compare-bases":
        rows = compare_bases(base, KlBase(args.test_base))
        if args.table:
            print_table(rows)
        s = summarize(rows)
        print_summary(s)
        if args.json:
            write_json(args.json, s, rows)
        return 0

    if not args.logits_dir and not (args.logits and args.positions):
        ap.error("need --logits-dir, or --logits with --positions")
    logits, positions, tokens = load_logits(args)
    if tokens is not None:
        flat = [int(x) for x in base.tokens.reshape(-1)]
        if flat[: len(tokens)] != tokens[: len(flat)]:
            print("warning: probe tokens differ from base tokens", file=sys.stderr)
    rows = compare(base, logits, positions)
    if not rows:
        print("no overlapping positions between base file and logits", file=sys.stderr)
        return 1
    if args.table:
        print_table(rows)
    s = summarize(rows)
    print_summary(s)
    if args.json:
        write_json(args.json, s, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
