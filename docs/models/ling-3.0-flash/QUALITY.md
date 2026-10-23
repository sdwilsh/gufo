# Ling-3.0-Flash quality

For AD-Q4_K_S and AD-Q5_K_M, Gufo's decode and prefill paths stay within mean KL 0.0016 of the F32 arbiter on both text sets, against 0.0027 to 0.0061 for llama.cpp Vulkan.

The arbiter is Gufo's scalar CPU implementation of the llama.cpp `bailingmoe3`
graph over the same GGUF (`src/models/ling_3_flash/reference.cpp`, F32
activations and an F32 latent cache; `--f64` is the tie-break mode). A score is
the KL divergence between the arbiter's next-token distribution and the engine's
for the same quantized weights. It measures how faithfully an engine runs the
file, not what quantization costs against BF16. Each quant has its own arbiter
dumps and bases.

## Engine fidelity against the F32 arbiter

Cells are mean KL (nats) / 99th-percentile KL / top-1 agreement with the
arbiter. llama.cpp's perplexity tool does not report top-1 in these runs.
`llama-perplexity --kl-divergence` and Gufo's `logit_dump` score the same base
files, so the scales are shared. The AD-Q5_K_M quality run used machine B and
the AD-Q4_K_S run used machine A. A repeat of the AD-Q4_K_S run on machine B
gave bit-identical results for every Gufo and llama.cpp step.

| Chat set, 126 positions | AD-Q4_K_S | AD-Q5_K_M |
| --- | ---: | ---: |
| Gufo decode (one token per forward) | 4.92e-06 / 0.000222 / 100.0% | 1.96e-05 / 0.00051 / 100.0% |
| Gufo prefill (bulk) | 0.00149 / 0.0187 / 99.2% | 0.000610 / 0.0141 / 99.2% |
| llama.cpp Vulkan q8_0 K/V, decode | 0.00404 / 0.0367 | 0.00329 / 0.0324 |
| llama.cpp Vulkan q8_0 K/V, prefill | 0.00506 / 0.0524 | 0.00452 / 0.0429 |
| llama.cpp Vulkan f16 K/V, decode | 0.00359 / 0.0496 | 0.00367 / 0.0437 |
| llama.cpp Vulkan f16 K/V, prefill | 0.00358 / 0.0382 | 0.00413 / 0.0505 |

| Novel text, 255 positions | AD-Q4_K_S | AD-Q5_K_M |
| --- | ---: | ---: |
| Gufo decode (one token per forward) | 0.000572 / 0.00853 / 99.2% | 0.00151 / 0.0196 / 99.2% |
| Gufo prefill (bulk) | 0.000923 / 0.0102 / 97.6% | 0.00156 / 0.0140 / 98.8% |
| llama.cpp Vulkan q8_0 K/V, decode | 0.00328 / 0.025 | 0.00287 / 0.0233 |
| llama.cpp Vulkan q8_0 K/V, prefill | 0.0061 / 0.04 | 0.00562 / 0.0372 |
| llama.cpp Vulkan f16 K/V, decode | 0.00313 / 0.0141 | 0.00273 / 0.0254 |
| llama.cpp Vulkan f16 K/V, prefill | 0.00508 / 0.0282 | 0.005 / 0.0292 |

One step fails its gate. The raw AD-Q4_K_S Gufo bulk-prefill step on the novel
text fails `logit_dump`'s top-1 gate (at least 0.98; measured 0.9765, 6 of 255
positions), although its mean KL is 0.00092. The flips are near-ties between
tokens, not distribution error. The status stays `fail` in
[quality-q4ks.json](artifacts/quality-q4ks.json). Every other step passed.

Prefill and decode KL are unchanged from the previous card, which was measured
on the v0.17.1 build. All 14 steps match that round bit for bit. The Q5_K_M
novel-text prefill row has the highest Gufo mean KL on this page, 0.00156.

Reading the numbers:

- Gufo's decode and bulk-prefill paths are 2.4 to 821 times closer to the
  arbiter than llama.cpp's Vulkan engine on the chat set (mean KL, like with
  like) and 1.8 to 6.6 times closer on the novel text. The standard error of
  Gufo's chat-set means is 29% to 75% of the mean (126 positions), so compare
  orders of magnitude, not digits.
- llama.cpp's q8_0 and f16 K/V cells differ by less than the spread between
  prefill and decode, so K/V precision does not separate the engines.
- The engine tests (the MTP and session suites) pass on both quant files:
  mtp_test 60/60 and session_test 11/11 for AD-Q4_K_S and for AD-Q5_K_M.

## What the quants themselves cost

I did not measure quantization loss against BF16, because the BF16 model does
not fit this hardware. AtomicChat published figures for the recipes both files
reproduce. They come from AtomicChat's own build and text (a fork build), and I
did not remeasure them. The mean KL is 0.0318 (top-1 96.60%) for AD-Q4_K_S and
0.0242 (top-1 97.45%) for AD-Q5_K_M, with 99th-percentile KL 0.310 and 0.201.
Both files match their recipe tensor for tensor (917 of 917 types and shapes),
so I expect the same ranking. Q5_K_M carries about a quarter less mean KL for
20% more bytes, at about 3-4% lower speed in the [benchmarks](BENCHMARKS.md)
(measured on different machines).

The weights of the new files also differ from the earlier build of AD-Q4_K_S
(same file name, IQ4_XS routed gate/up experts, Q4_K token embedding). Scoring
the new arbiters against the old CPU f16 base (built from the earlier file)
gives a mean KL of about 0.019 for each new quant, against 0.0046 when the
weights are unchanged. That is a pipeline sanity check of how far the weights
moved, not a quality verdict.

## Other checks

| Check | Result | Evidence |
| --- | --- | --- |
| MTP verification rows equal single-token decode, bitwise; greedy MTP equals AR | Passes; small batches of 3, 5 and 8 rows match token-by-token decode in 0 of 42 layers | `mtp_test`, `batch_test` |
| Sampled MTP | Seeded replay repeats; KS distribution tests pass | `mtp_distribution` |
| Session state (snapshot, restore, restart restore) | Passes, bit for bit | `session_test`, `snapshot_test` |
| Functional suites on the branch head | 43 of 44 pass. The failure is the `cache-edits` suite under a shared 4 GiB budget (best effort by design), and its separate 8 GiB server passes. Tool calls (including incomplete Responses calls) and tool streaming pass, and so do the Ling agent, coding-agent-session, and SDK replays | `tests/functional/run.py --suite all --sampling-preset ling3` |

## Long context

KL against the exact CPU arbiter at seven depths up to 205k tokens, 512 scored
tokens per window, measured on the earlier Q4_K_S file before the quant
campaign. No engine degrades with depth. The 98k spike also appears at short
context on the same text, so it belongs to the text and the reference, not the
depth.

| Depth window | Gufo | llama.cpp q8_0 K/V | llama.cpp f16 K/V |
| ---: | ---: | ---: | ---: |
| 16k | 0.0297 | 0.0333 | 0.0324 |
| 32k | 0.0078 | 0.0097 | 0.0089 |
| 64k | 0.0172 | 0.0111 | 0.0093 |
| 98k | 0.1097 | 0.1427 | 0.1186 |
| 131k | 0.0048 | 0.0080 | 0.0082 |
| 164k | 0.0052 | 0.0140 | 0.0066 |
| 205k | 0.0019 | 0.0023 | 0.0077 |

Top-1 agreement is 0.96 to 0.996 for every engine and window, and the
log-likelihood differences stay within 0.035 nats. Gufo is at least as close as
llama.cpp at every depth except 64k.

## Agent loops

The coding agent in a private session bundle sometimes repeats a refused or
failing tool call. The study traced this to the model, not the engine:

- Gufo, llama.cpp with q8_0 K/V and llama.cpp with f16 K/V loop at the same
  decision points and at statistically indistinguishable rates, with a warm or a
  cold prompt cache (n=40 per warm cell; the one cold Gufo cell has n=5).
- Across quants (earlier Q4_K_S, corrected Q4_K_S, Q5_K_M; llama.cpp q8_0 K/V,
  warm, n=40 per cell, six decision points), Q5_K_M lowers the share of exact
  repeats in four cells, leaves one unchanged and raises one. After a Bonferroni
  correction over the six cells (alpha about 0.008) only one cell changes
  significantly, and the loop rate (exact repeat or near-variant) stays at 95 to
  100% in most cells. The quant choice does not fix looping, so choose it for
  memory and quality.

The sessions, prompts and per-cell counts stay private.

## Benchmark method

I measured Gufo on October 23, 2026 on the v0.21.0.0 build (gufo 0.21.0.0 at
00eea62, built with Nix) with the evidence kit (nix-release build mode), Fedora
44 kernel 7.2.9, ROCm 7.2.3, and AMD clang 22. The Gufo binary is the ELF
`.gufo-wrapped` (SHA-256
`ae863666ee5c9e8d45722cda87cda38f6fad2985b0ae721344d3c9522e2454f8`) behind a Nix
wrapper script. Each Gufo artifact in [artifacts](artifacts/) records the
evidence-kit revision `b1c4b4129327c0c5ce2a100b09f80137d6f0ff41`, which is
00eea62 plus kit files only. The speed tables ran on the machine of each
llama.cpp reference. The KL steps for both files ran on October 23. I measured
llama.cpp on October 12, 2026 with the `server-vulkan-v0.5.0` image (build
11146, commit 7fe450e19, Mesa 26.0.8 RADV).

Gufo serves with `--sessions 1 llm --think off --max-pending-per-client 8
--cache-ram-bytes 2147483648` and `--speculative off` or `mtp`. llama.cpp serves
with `-ngl 999 -fa on -ctk q8_0 -ctv q8_0 --cache-reuse 0 --jinja --reasoning off
--cache-ram 0`. The single-user tables use context 199296 (133760 for MTP). A
cell is one sample of pp2048 / tg128 at the stated depth of synthetic cached
text. Multi-user cohorts replay a prepared session with at most 9 tokens
re-prefilled (Ling's checkpoint tail is 9 tokens). The llama.cpp MTP
qualification (draft-mtp against its own AR, d0, greedy) failed on both files,
so no llama.cpp MTP rates are reported.

Model identities, hashes, and shard sizes are in
[model-identities.json](artifacts/model-identities.json). Raw per-table results
and Gufo's memory peaks are in [artifacts](artifacts/).

## Reproduce

```sh
K=tools/models/ling_3_flash/kl_base.py
python3 $K write-base chat128.kl --n-ctx 128 --logits-dir chat1 --logits-dir chat2
python3 $K write-base window512.kl --n-ctx 512 --logits-dir window0
ling_3_flash_logit_dump --model $M --kl-base chat128.kl            # prefill
ling_3_flash_logit_dump --model $M --kl-base chat128.kl --kl-decode  # decode
llama-perplexity -m $M -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 128 -b 128 -ub 128 \
  --kl-divergence-base chat128.kl --kl-divergence     # prefill; -b 1 -ub 1 decodes
ling_3_flash_mtp_test --model $M
ling_3_flash_session_test --model $M
python3 tools/bench/model-bench.py --model ling-3.0-flash render
```

Base file SHA-256 per quant (arbiter dumps built from the same GGUF):

| Base | AD-Q4_K_S | AD-Q5_K_M |
| --- | --- | --- |
| `chat128.kl` | `087096abe9313b2df74d01d3c2c0d277c46793620d73e2b2e32534c4313a5e6e` | `4132b92c7c6d22255d498f15c7865ee90a50c6e8cec171f5907ae5afb8251f26` |
| `window512.kl` | `059db4f6b2534d2f614e17a9c61a8ae4545f755503338207620dfaf22b0d02e0` | `24d157ed78308379d8b142687752c2e76f2f56175f0f9a910ad2cf892bb4e4e1` |
