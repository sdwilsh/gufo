# Ling-3.0-Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo (the v0.17 build, gufo
349b9d95, fork tag v0.17.0.0, plus the evidence kit; see
[Benchmark method](QUALITY.md#benchmark-method)) versus
llama.cpp `server-vulkan-v0.5.0` (`0.5.0-dev`, build 11146, commit 7fe450e19) on
Vulkan with Mesa 26.0.8 RADV, at its production flags `-fa on -ctk q8_0 -ctv
q8_0`. Both engines load the same two published files, **AD-Q4_K_S** (the
production target) and **AD-Q5_K_M**. HTTP, greedy, thinking off, measured
October 18, 2026 for Gufo and October 12, 2026 for llama.cpp. Positive gain
favors Gufo.
[Quality and measurement details](QUALITY.md) · [Model identities](artifacts/model-identities.json)

**Machines.** Two Strix Halo machines of one configuration measured the files.
Machine A ran the AD-Q4_K_S tables and machine B ran the AD-Q5_K_M tables, so
the two quants share no machine. Machine B's board was replaced shortly before
the campaign, so its numbers cannot be compared with earlier measurements there.
Within a table, Gufo and llama.cpp ran back to back on the same machine, so the
gain columns are controlled. A Q4_K_S against Q5_K_M difference also carries the
machine. Machine B has the 108 GiB GTT limit that the default 12 GiB expanded
K/V store needs for Q5_K_M (see [Memory](README.md#memory-and-sessions)).

## Single user, autoregressive

Each cell runs about pp2048 / tg128, and depth is the cached prefix in tokens.
There is one sample per cell. Gufo runs `--speculative off`.

### AD-Q4_K_S (machine A)

<!-- bench:single-ar-q4ks -->
| Ling-3.0-flash AD-Q4_K_S AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1059.35 | 433.01 | +144.6% | 35.84 | 33.21 | +7.9% |
| 32,768 | 827.65 | 148.85 | +456.0% | 33.60 | 24.27 | +38.4% |
| 65,536 | 680.68 | 71.82 | +847.8% | 32.04 | 19.06 | +68.1% |
| 131,072 | 465.87 | 38.61 | +1106.6% | 29.53 | 13.35 | +121.2% |
| 196,608 | 319.45 | 25.04 | +1175.8% | 27.33 | 9.93 | +175.2% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4ks.svg)

I ran the 196,608-token llama.cpp cell of this table on machine B, because
machine A had returned to production. That row's gain therefore compares Gufo on
machine A with llama.cpp on machine B.

### AD-Q5_K_M (machine B)

<!-- bench:single-ar-q5km -->
| Ling-3.0-flash AD-Q5_K_M AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1009.71 | 402.82 | +150.7% | 34.49 | 32.04 | +7.6% |
| 32,768 | 786.16 | 142.92 | +450.1% | 32.44 | 23.63 | +37.3% |
| 65,536 | 650.63 | 69.28 | +839.1% | 30.94 | 18.58 | +66.5% |
| 131,072 | 451.02 | 36.97 | +1120.0% | 28.62 | 12.86 | +122.6% |
| 196,608 | 310.65 | 24.88 | +1148.6% | 26.53 | 9.66 | +174.6% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q5km.svg)

## Single user, MTP

Gufo drafts from block 42 and also copies repeated passages from the context
(prompt lookup). llama.cpp's own `--spec-type draft-mtp` does not reproduce its
autoregressive output on this model (the d0 qualification run differs), so I
report no speculative llama.cpp rate. The llama.cpp column here is its
autoregressive rate from the table above, and the gain is Gufo's MTP rate
against it. pp is the larger of the two workloads. Mixed is prose, and
repetitive re-quotes earlier text. There is one sample per cell, so mixed-prose
tg moves with each prompt's draft acceptance (0.01 to 1.06 accepted tokens per
step, or 20 to 75% of drafted tokens, in these runs).

### AD-Q4_K_S (machine A)

<!-- bench:single-mtp-q4ks -->
| Ling-3.0-flash AD-Q4_K_S MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | llama.cpp AR tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp AR tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1059.58 | 51.10 | 33.21 | +53.9% | 112.35 | 33.21 | +238.3% |
| 32,768 | 825.20 | 46.16 | 24.27 | +90.2% | 92.66 | 24.27 | +281.8% |
| 65,536 | 680.11 | 36.40 | 19.06 | +91.0% | 80.94 | 19.06 | +324.7% |
| 131,072 | 466.23 | 28.81 | 13.35 | +115.8% | 66.30 | 13.35 | +396.6% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:single-mtp-q5km -->
| Ling-3.0-flash AD-Q5_K_M MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | llama.cpp AR tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp AR tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1005.07 | 38.99 | 32.04 | +21.7% | 104.63 | 32.04 | +226.6% |
| 32,768 | 787.05 | 42.82 | 23.63 | +81.2% | 86.83 | 23.63 | +267.5% |
| 65,536 | 652.89 | 30.65 | 18.58 | +65.0% | 76.53 | 18.58 | +311.9% |
| 131,072 | 451.61 | 33.59 | 12.86 | +161.2% | 63.32 | 12.86 | +392.4% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q5km.svg)

## Multiple users, autoregressive

The rate is the cohort wall-clock aggregate on both sides, which is the output
tokens of all users divided by the elapsed time of the measured rounds. Gufo
batches up to eight decoding requests in one forward pass when
`--speculative off`, and llama.cpp batches its slots. Each user replays a
prepared 2,048-token session with at most 9 tokens re-prefilled.

### AD-Q4_K_S (machine A)

<!-- bench:multi-ar-q4ks -->
| Ling-3.0-flash AD-Q4_K_S AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 35.71 | 29.84 | +19.7% |
| 2 | 65.36 | 40.14 | +62.8% |
| 4 | 110.28 | 61.84 | +78.3% |
| 6 | 146.39 | 69.70 | +110.0% |
| 8 | 166.34 | 78.93 | +110.7% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:multi-ar-q5km -->
| Ling-3.0-flash AD-Q5_K_M AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 34.44 | 28.82 | +19.5% |
| 2 | 63.19 | 36.25 | +74.3% |
| 4 | 107.36 | 57.38 | +87.1% |
| 6 | 142.54 | 64.65 | +120.5% |
| 8 | 160.93 | 78.96 | +103.8% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q5km.svg)

## Multiple users, MTP

The aggregate rule is the same. MTP cohorts share their draft and verify
forwards, and each request ends exactly as if it had decoded alone. The
llama.cpp columns are N/A for the reason in the single-user MTP section. Against
the autoregressive table above, MTP leads on prose up to four users and batched
AR leads from six. Repetitive prompts gain most at few users and flatten at
113-129 tok/s.

### AD-Q4_K_S (machine A)

<!-- bench:multi-mtp-q4ks -->
| Ling-3.0-flash AD-Q4_K_S MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.10 | N/A | N/A | 112.19 | N/A | N/A |
| 2 | 91.01 | N/A | N/A | 127.88 | N/A | N/A |
| 4 | 118.53 | N/A | N/A | 133.77 | N/A | N/A |
| 6 | 144.03 | N/A | N/A | 132.87 | N/A | N/A |
| 8 | 162.03 | N/A | N/A | 133.73 | N/A | N/A |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:multi-mtp-q5km -->
| Ling-3.0-flash AD-Q5_K_M MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.97 | N/A | N/A | 103.17 | N/A | N/A |
| 2 | 80.46 | N/A | N/A | 119.30 | N/A | N/A |
| 4 | 125.78 | N/A | N/A | 122.31 | N/A | N/A |
| 6 | 139.78 | N/A | N/A | 122.52 | N/A | N/A |
| 8 | 156.70 | N/A | N/A | 125.16 | N/A | N/A |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q5km.svg)

## Memory occupation

Context capacity 133121, autoregressive, one session. Gufo's value is the peak
device-global HIP memory counter during the request, which includes the default
12 GiB expanded K/V store and matches the sysfs GTT peak (82.84 and 97.02 GiB).
The HIP counter does not see llama.cpp's Vulkan allocations, so its value is the
sysfs GTT peak over the whole table run; llama.cpp preallocates its K/V cache,
and one value covers both workloads. Negative gain means Gufo uses more.
Peaks of the other runs are in `artifacts/gtt-peaks-*.json` (Gufo, one
session: AR 83.34 and MTP 85.56 GiB for Q4_K_S; AR 97.52 and MTP 100.09 GiB for
Q5_K_M, each at context 199296).

### AD-Q4_K_S (machine A)

<!-- bench:memory-q4ks -->
| Ling-3.0-flash AD-Q4_K_S AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 82.82 | 69.95 | -15.5% |
| 16K prefix, pp4096 + tg128 | 82.82 | 69.95 | -15.5% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:memory-q5km -->
| Ling-3.0-flash AD-Q5_K_M AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 97.00 | 84.12 | -13.3% |
| 16K prefix, pp4096 + tg128 | 97.00 | 84.12 | -13.3% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q5km.svg)

## Not measured for these files

I did not repeat loading time, the sampled-MTP table, or the llama.cpp f16 K/V
control for the corrected Q4_K_S and the Q5_K_M files. Earlier f16 K/V runs on
the previous Q4_K_S build gave llama.cpp about 2.6 times the prefill and 1.23
times the decode of its q8_0 rate at 64k with no quality cost. The q8_0 tables
above are therefore the more favorable comparison for Gufo. The earlier results
are kept as history in [Experiments](EXPERIMENTS.md#history).
