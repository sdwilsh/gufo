#!/usr/bin/env python3
"""Break-even model of Ling-3.0-flash MTP decoding on gfx1151.

Cycle cost = plain step + k draft steps + the extra verify work of k more rows
(expert-union growth, per-row MLA attention over the context, KDA rollback
points, per-row compute) + fixed per-cycle work. Speedup = E[tokens per cycle] /
cycle * AR step. Inputs are labelled [M] measured on gfx1151 (WS10 op profiles,
bench tg128) or [E] estimated from bytes/launch counts; acceptance and expert
unions come from `mtp_acceptance_probe` JSON files (CPU reference, measured on
real text).

    breakeven.py [probe.json ...] [--markdown]
"""
import json
import sys

# --- measured on gfx1151 (PLAN.md WS10 r3/r4 op profile, DSPARK-SPIKE.md) ----
AR_MS = {0: 27.9, 16384: 28.95, 65536: 31.4}   # [M] bench tg128 r4: 35.79 / 34.54 / 31.83 t/s
HEAD_MS = 3.42              # [M] F16 lm_head GEMV @235 GB/s
MOE_MS_PER_EXPERT = 0.82    # [M] ms per distinct routed expert per layer (6.53 ms / 8, all 40 MoE layers)
MLA_ATTN_MS_64K = 3.46      # [M] 7 MLA layers, one decode-attention launch per token at 64k (r4)
# --- estimated ---------------------------------------------------------------
BLOCK42_MS = 0.75           # [E] 85 MB of weights at ~170 GB/s + latency-bound kernels + eager launches
BLOCK42_ATTN_64K = MLA_ATTN_MS_64K / 7   # [E] one MLA layer of the above
DRAFT_FIXED_MS = 0.1        # [E] argmax + D2H + host sync per draft step
CKPT_MS = 0.45             # [E] 73.5 MB KDA state+conv checkpoint written per extra verified row
KDA_ROW_MS = 0.35           # [E] per extra row recurrence (latency-bound per-token loop, 35 layers)
HEAD_ROW_MS = 0.15          # [E] extra head compute per row
FILL_MS = 0.25              # [E] eh_proj + kv_a fill and hidden norm per cycle
ROLLBACK_MS = 0.75          # [E] 75 MiB read + write on a partial rejection


def attn_ms(depth):
    return MLA_ATTN_MS_64K * depth / 65536 + 0.05


def union(probes, n):
    """Mean distinct experts per MoE layer for n consecutive tokens."""
    vals = [p["expert_union_mean"][str(n)] for p in probes
            if "expert_union_mean" in p]
    if vals:
        return sum(vals) / len(vals)
    return 512 * (1 - (1 - 8 / 512) ** n)   # i.i.d. uniform routing (upper bound)


def cycle_ms(depth, k, probes, p_partial=0.5):
    ar = AR_MS[depth]
    extra = 0.0
    for n in range(2, k + 2):
        extra += MOE_MS_PER_EXPERT * (union(probes, n) - union(probes, n - 1))
        extra += attn_ms(depth) + CKPT_MS + KDA_ROW_MS + HEAD_ROW_MS
    drafts = k * (HEAD_MS + BLOCK42_MS + BLOCK42_ATTN_64K * depth / 65536 +
                  DRAFT_FIXED_MS)
    return ar + extra + drafts + FILL_MS + p_partial * ROLLBACK_MS


def expected_tokens(accept):
    e, s = 1.0, 1.0
    out = [1.0]
    for a in accept:
        s *= a
        e += s
        out.append(e)
    return out


def breakeven_uniform(depth, k, probes):
    """Smallest per-depth acceptance a (same at every depth) with speedup 1."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        speed = expected_tokens([mid] * k)[k] * AR_MS[depth] / cycle_ms(depth, k, probes)
        lo, hi = (mid, hi) if speed < 1 else (lo, mid)
    return hi


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    probes = [json.load(open(p)) for p in paths]
    print("Break-even of Ling MTP on gfx1151 ([M] measured inputs, [E] estimated)")
    print("AR step ms [M]:", AR_MS)
    print("expert union per MoE layer for n tokens:",
          {n: round(union(probes, n), 1) for n in range(1, 9)},
          "(measured, CPU reference)" if probes else "(i.i.d. uniform bound)")
    print()
    print("depth    k  cycle ms [E]  tokens/ms AR   break-even acceptance (same at every depth)")
    for depth in AR_MS:
        for k in range(1, 5):
            print(f"{depth:6d}  {k}  {cycle_ms(depth, k, probes):10.1f}  "
                  f"{1000 / AR_MS[depth]:8.1f} t/s  {breakeven_uniform(depth, k, probes):.2f}")
    for p in probes:
        print()
        print(f"{p['label']}: measured conditional acceptance by depth "
              f"{[round(a, 2) for a in p['conditional_acceptance']]} "
              f"({p['starts']} starts)")
        accept = p["conditional_acceptance"]
        for depth in AR_MS:
            row = []
            for k in range(1, len(accept) + 1):
                tokens = expected_tokens(accept[:k])[k]
                speed = tokens * AR_MS[depth] / cycle_ms(depth, k, probes)
                row.append(f"k={k}: {speed:.2f}x")
            print(f"  depth {depth:6d}: " + "  ".join(row))


if __name__ == "__main__":
    main()
