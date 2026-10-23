# Ling-3.0-Flash benchmarks

The test hardware is AMD Strix Halo `gfx1151` with 128 GB unified memory. This
page compares Gufo (the v0.21.0.0 build, gufo 0.21.0.0 at 00eea62, built with
Nix, plus the evidence kit; see [Benchmark method](QUALITY.md#benchmark-method)) with
llama.cpp `server-vulkan-v0.5.0` (`0.5.0-dev`, build 11146, commit 7fe450e19) on
Vulkan with Mesa 26.0.8 RADV, at its production flags `-fa on -ctk q8_0 -ctv
q8_0`. Both engines load the same two published files, AD-Q4_K_S (the
production target) and AD-Q5_K_M. Requests go over HTTP with greedy sampling and
thinking off. I measured Gufo on October 23, 2026 and llama.cpp on October 12,
2026. A positive gain favors Gufo.
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
| 0 | 1064.45 | 433.01 | +145.8% | 36.29 | 33.21 | +9.3% |
| 32,768 | 834.84 | 148.85 | +460.9% | 33.97 | 24.27 | +40.0% |
| 65,536 | 688.44 | 71.82 | +858.6% | 32.28 | 19.06 | +69.4% |
| 131,072 | 474.85 | 38.61 | +1129.9% | 29.58 | 13.35 | +121.6% |
| 196,608 | 328.71 | 25.04 | +1212.7% | 27.26 | 9.93 | +174.5% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4ks.svg)

I ran the 196,608-token llama.cpp cell of this table on machine B, because
machine A had returned to production. That row's gain therefore compares Gufo on
machine A with llama.cpp on machine B.

### AD-Q5_K_M (machine B)

<!-- bench:single-ar-q5km -->
| Ling-3.0-flash AD-Q5_K_M AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1011.76 | 402.82 | +151.2% | 35.02 | 32.04 | +9.3% |
| 32,768 | 792.91 | 142.92 | +454.8% | 32.77 | 23.63 | +38.7% |
| 65,536 | 658.06 | 69.28 | +849.9% | 31.22 | 18.58 | +68.0% |
| 131,072 | 457.50 | 36.97 | +1137.5% | 28.59 | 12.86 | +122.3% |
| 196,608 | 319.39 | 24.88 | +1183.7% | 26.43 | 9.66 | +173.6% |
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
tg moves with each prompt's draft acceptance (0.05 to 1.17 accepted tokens per
step, or 37.5 to 75% of drafted tokens, in these runs).

### AD-Q4_K_S (machine A)

<!-- bench:single-mtp-q4ks -->
| Ling-3.0-flash AD-Q4_K_S MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | llama.cpp AR tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp AR tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1065.46 | 52.53 | 33.21 | +58.2% | 115.12 | 33.21 | +246.6% |
| 32,768 | 831.83 | 45.03 | 24.27 | +85.5% | 93.62 | 24.27 | +285.7% |
| 65,536 | 689.30 | 35.86 | 19.06 | +88.1% | 81.49 | 19.06 | +327.5% |
| 131,072 | 475.80 | 28.96 | 13.35 | +116.9% | 65.95 | 13.35 | +394.0% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:single-mtp-q5km -->
| Ling-3.0-flash AD-Q5_K_M MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | llama.cpp AR tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp AR tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1012.15 | 45.24 | 32.04 | +41.2% | 109.08 | 32.04 | +240.4% |
| 32,768 | 801.29 | 43.22 | 23.63 | +82.9% | 88.17 | 23.63 | +273.1% |
| 65,536 | 665.07 | 35.17 | 18.58 | +89.3% | 77.12 | 18.58 | +315.1% |
| 131,072 | 459.88 | 29.28 | 12.86 | +127.7% | 63.12 | 12.86 | +390.8% |
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
| 1 | 36.26 | 29.84 | +21.5% |
| 2 | 66.43 | 40.14 | +65.5% |
| 4 | 113.58 | 61.84 | +83.7% |
| 6 | 149.43 | 69.70 | +114.4% |
| 8 | 172.57 | 78.93 | +118.6% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:multi-ar-q5km -->
| Ling-3.0-flash AD-Q5_K_M AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 34.89 | 28.82 | +21.1% |
| 2 | 63.77 | 36.25 | +75.9% |
| 4 | 110.04 | 57.38 | +91.8% |
| 6 | 147.47 | 64.65 | +128.1% |
| 8 | 165.30 | 78.96 | +109.3% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q5km.svg)

## Multiple users, MTP

The aggregate rule is the same. MTP cohorts share their draft and verify
forwards, and each request ends exactly as if it had decoded alone. The
llama.cpp columns are N/A for the reason in the single-user MTP section. Against
the autoregressive table above, MTP leads on prose up to four users and batched
AR leads from six. Repetitive prompts gain most at few users and flatten from
four users at about 137 tok/s on AD-Q4_K_S and about 128 tok/s on AD-Q5_K_M.

### AD-Q4_K_S (machine A)

<!-- bench:multi-mtp-q4ks -->
| Ling-3.0-flash AD-Q4_K_S MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52.50 | N/A | N/A | 114.82 | N/A | N/A |
| 2 | 94.44 | N/A | N/A | 131.41 | N/A | N/A |
| 4 | 137.42 | N/A | N/A | 137.85 | N/A | N/A |
| 6 | 147.03 | N/A | N/A | 136.53 | N/A | N/A |
| 8 | 168.50 | N/A | N/A | 137.94 | N/A | N/A |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:multi-mtp-q5km -->
| Ling-3.0-flash AD-Q5_K_M MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.11 | N/A | N/A | 107.81 | N/A | N/A |
| 2 | 74.13 | N/A | N/A | 122.62 | N/A | N/A |
| 4 | 125.89 | N/A | N/A | 128.05 | N/A | N/A |
| 6 | 143.29 | N/A | N/A | 127.64 | N/A | N/A |
| 8 | 164.40 | N/A | N/A | 128.83 | N/A | N/A |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q5km.svg)

## Memory occupation

The context capacity is 133121, with autoregressive decoding and one session.
Gufo's value is the peak device-global HIP memory counter during the request. It
includes the default 12 GiB expanded K/V store and matches the sysfs GTT peak
(83.47 and 97.65 GiB). The HIP counter does not see llama.cpp's Vulkan
allocations, so llama.cpp's value is the sysfs GTT peak over the whole table
run. llama.cpp preallocates its K/V cache, so one value covers both workloads. A
negative gain means Gufo uses more. The peaks of the other runs are in
`artifacts/gtt-peaks-*.json`. For Gufo with one session, they are AR 83.96 and
MTP 86.18 GiB for Q4_K_S, and AR 98.14 and MTP 100.71 GiB for Q5_K_M, each at
context 199296.

### AD-Q4_K_S (machine A)

<!-- bench:memory-q4ks -->
| Ling-3.0-flash AD-Q4_K_S AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 83.45 | 69.95 | -16.2% |
| 16K prefix, pp4096 + tg128 | 83.45 | 69.95 | -16.2% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4ks.svg)

### AD-Q5_K_M (machine B)

<!-- bench:memory-q5km -->
| Ling-3.0-flash AD-Q5_K_M AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 97.63 | 84.12 | -13.8% |
| 16K prefix, pp4096 + tg128 | 97.63 | 84.12 | -13.8% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q5km.svg)

## Not measured for these files

I did not repeat loading time, the sampled-MTP table, or the llama.cpp f16 K/V
control for the corrected Q4_K_S and the Q5_K_M files. Earlier f16 K/V runs on
the previous Q4_K_S build gave llama.cpp about 2.6 times the prefill and 1.23
times the decode of its q8_0 rate at 64k with no quality cost. The q8_0 tables
above are therefore the more favorable comparison for Gufo. The earlier results
are kept as history in [Experiments](EXPERIMENTS.md#history).
