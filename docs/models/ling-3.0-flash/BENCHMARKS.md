# Ling-3.0-Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo (a build of the
quant-support commit plus the evidence kit; see
[Benchmark method](QUALITY.md#benchmark-method)) versus
llama.cpp `server-vulkan-v0.5.0` (`0.5.0-dev`, build 11146, commit 7fe450e19) on
Vulkan with Mesa 26.0.8 RADV, at its production flags `-fa on -ctk q8_0 -ctv
q8_0`. Both engines load the same two published files, **AD-Q4_K_S** (the
production target) and **AD-Q5_K_M**. HTTP, greedy, thinking off, measured
October 12, 2026. Positive gain favors Gufo.
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
| 0 | 1064.36 | 433.01 | +145.8% | 35.78 | 33.21 | +7.7% |
| 32,768 | 816.20 | 148.85 | +448.3% | 33.64 | 24.27 | +38.6% |
| 65,536 | 668.80 | 71.82 | +831.2% | 32.06 | 19.06 | +68.2% |
| 131,072 | 459.72 | 38.61 | +1090.7% | 29.51 | 13.35 | +121.0% |
| 196,608 | 318.01 | 25.04 | +1170.0% | 27.30 | 9.93 | +174.9% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4ks.svg)

I ran the 196,608-token llama.cpp cell of this table on machine B, because
machine A had returned to production. That row's gain therefore compares Gufo on
machine A with llama.cpp on machine B.

### AD-Q5_K_M (machine B)

<!-- bench:single-ar-q5km -->
| Ling-3.0-flash AD-Q5_K_M AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1021.44 | 402.82 | +153.6% | 34.51 | 32.04 | +7.7% |
| 32,768 | 784.86 | 142.92 | +449.2% | 32.50 | 23.63 | +37.5% |
| 65,536 | 645.59 | 69.28 | +831.9% | 31.00 | 18.58 | +66.8% |
| 131,072 | 449.48 | 36.97 | +1115.8% | 28.61 | 12.86 | +122.5% |
| 196,608 | 312.30 | 24.88 | +1155.2% | 26.52 | 9.66 | +174.5% |
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
| 0 | 1060.48 | 50.04 | 33.21 | +50.7% | 108.39 | 33.21 | +226.4% |
| 32,768 | 810.46 | 45.24 | 24.27 | +86.4% | 90.04 | 24.27 | +271.0% |
| 65,536 | 660.80 | 35.22 | 19.06 | +84.8% | 78.71 | 19.06 | +313.0% |
| 131,072 | 458.84 | 28.81 | 13.35 | +115.8% | 64.88 | 13.35 | +386.0% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:single-mtp-q5km -->
| Ling-3.0-flash AD-Q5_K_M MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | llama.cpp AR tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp AR tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1014.61 | 38.54 | 32.04 | +20.3% | 101.53 | 32.04 | +216.9% |
| 32,768 | 780.02 | 42.21 | 23.63 | +78.6% | 85.10 | 23.63 | +260.1% |
| 65,536 | 637.67 | 30.58 | 18.58 | +64.6% | 75.08 | 18.58 | +304.1% |
| 131,072 | 447.65 | 33.20 | 12.86 | +158.2% | 62.22 | 12.86 | +383.8% |
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
| 2 | 64.27 | 40.14 | +60.1% |
| 4 | 107.34 | 61.84 | +73.6% |
| 6 | 140.73 | 69.70 | +101.9% |
| 8 | 156.35 | 78.93 | +98.1% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:multi-ar-q5km -->
| Ling-3.0-flash AD-Q5_K_M AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 34.42 | 28.82 | +19.4% |
| 2 | 62.01 | 36.25 | +71.1% |
| 4 | 104.34 | 57.38 | +81.8% |
| 6 | 137.08 | 64.65 | +112.0% |
| 8 | 152.38 | 78.96 | +93.0% |
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
| 1 | 49.80 | N/A | N/A | 107.77 | N/A | N/A |
| 2 | 87.34 | N/A | N/A | 122.23 | N/A | N/A |
| 4 | 114.01 | N/A | N/A | 127.62 | N/A | N/A |
| 6 | 137.09 | N/A | N/A | 126.90 | N/A | N/A |
| 8 | 151.75 | N/A | N/A | 128.66 | N/A | N/A |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:multi-mtp-q5km -->
| Ling-3.0-flash AD-Q5_K_M MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.39 | N/A | N/A | 100.77 | N/A | N/A |
| 2 | 74.86 | N/A | N/A | 113.10 | N/A | N/A |
| 4 | 119.72 | N/A | N/A | 119.63 | N/A | N/A |
| 6 | 133.49 | N/A | N/A | 119.51 | N/A | N/A |
| 8 | 147.08 | N/A | N/A | 119.82 | N/A | N/A |
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
