# Ling-3.0-Flash

inclusionAI's Ling-3.0-Flash (`bailingmoe3`) is a hybrid mixture-of-experts
model with 35 linear-attention (KDA) layers and 7 latent-attention (MLA) layers.
It has 124B parameters in total, 5.1B active per token, and a built-in
multi-token-prediction (MTP) block. Gufo supports two GGUF quants, published in
[`sdwilsh/Ling-3.0-flash-GGUF`](https://huggingface.co/sdwilsh/Ling-3.0-flash-GGUF):
AD-Q4_K_S (70.76 GiB, the production target) and AD-Q5_K_M (85.29 GiB, larger
and higher quality). Both rebuild AtomicChat's Atomic Dynamic recipes from the
bartowski BF16 conversion, match AtomicChat's tensor types and shapes for all
917 tensors, and add the 21 tensors of the NextN/MTP block, 938 in total.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Files

| Quant | Shards | Unsharded size | Tensor types |
| --- | --- | ---: | --- |
| `Ling-3.0-flash-AD-Q4_K_S/` | 2 | 75,981,701,568 B (70.76 GiB) | Routed experts Q4_K, edge blocks 2, 3, 39-41 Q5_K |
| `Ling-3.0-flash-AD-Q5_K_M/` | 3 | 91,576,648,128 B (85.29 GiB) | Routed experts Q5_K, edge blocks 2, 3, 39-41 Q6_K |

Both quants use a Q8_0 token embedding and an F16 output head. The attention,
KDA projections, shared experts, dense blocks, and the MTP `eh_proj` are Q8_0.
The routers, conv kernels, and norms are F32. Pass the first shard
(`Ling-3.0-flash-AD-Q4_K_S-00001-of-00002.gguf`,
`Ling-3.0-flash-AD-Q5_K_M-00001-of-00003.gguf`), and Gufo finds the others
beside it. Shard sizes and SHA-256 are in
[model-identities.json](artifacts/model-identities.json).

An earlier `AD-Q4_K_S` with the same file name used IQ4_XS routed gate/up
experts and a Q4_K token embedding. It did not match AtomicChat's placement, so
the corrected files replaced it. Gufo still loads it, and the
[history](EXPERIMENTS.md#history) keeps its measurements, but the tables here
use the corrected files.

## Architecture

- The GGUF has 43 blocks: 42 trunk blocks plus block 42, the `nextn` MTP block.
  The hidden width is 2560, the vocabulary 157184, and the native context
  262144.
- 35 KDA layers have 32 heads of 128. Each of q, k, and v passes through its own
  depthwise causal convolution (kernel 4) and SiLU, then the gated delta rule
  with a per-channel decay gate. The recurrent state is 2 MiB per layer per
  sequence.
- 7 MLA layers (5, 11, 17, 23, 29, 35, 41) have 32 heads of 128 nope + 64 rope,
  a 512-wide latent plus 64 rope dimensions per token, and a per-head output
  gate.
- Blocks 0 and 1 are dense SwiGLU (6144). Blocks 2-41 are mixtures of 512
  experts of width 768. A sigmoid router with a selection bias picks the top
  eight from four of eight groups (weights unbiased, renormalized, scaled by
  2.5), and an ungated shared expert runs alongside. SwiGLU clamps apply on
  blocks 34-41.

## Load and run

```sh
nix develop -c hf download sdwilsh/Ling-3.0-flash-GGUF \
  --include "Ling-3.0-flash-AD-Q4_K_S/*" --local-dir models/ling-3.0-flash
nix build
MODEL=models/ling-3.0-flash/Ling-3.0-flash-AD-Q4_K_S/Ling-3.0-flash-AD-Q4_K_S-00001-of-00002.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo prompt --model "$MODEL" -p "Name three rivers."
```

For the larger quant, download `Ling-3.0-flash-AD-Q5_K_M/*` and point `MODEL` at
`Ling-3.0-flash-AD-Q5_K_M-00001-of-00003.gguf`.

### Serve

```sh
# MTP (recommended up to four users)
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --sessions 4 --context 196608 --cache-ram-bytes 8589934592

# Many concurrent requests: batched autoregressive decode (up to eight)
./result/bin/gufo serve llm --model "$MODEL" --speculative off \
  --sessions 8 --context 65536 --cache-ram-bytes 8589934592
```

MTP is on by default when the GGUF carries block 42. Both files do, so no second
model is needed. `--speculative mtp` states the default, and `--speculative off`
selects autoregressive decoding. Against the
[autoregressive tables](BENCHMARKS.md#multiple-users-autoregressive),
MTP is faster for one to four users on prose, and batched autoregressive
decoding is faster from six users. The context is at most the native 262144.

AD-Q5_K_M on a machine with a 96 GiB GTT limit needs a smaller expanded K/V
store (below). Otherwise the commands are the same.

## Memory and sessions

Sessions are preallocated at `--context`, so memory is the weights, the shared
expanded K/V store, the sessions, and the working buffers.

| | AD-Q4_K_S | AD-Q5_K_M |
| --- | ---: | ---: |
| Weights | 70.76 GiB | 85.29 GiB |
| GTT peak, one session, AR (measured, default store, context 199296) | 83.34 GiB | 97.52 GiB |
| GTT peak, one session, MTP (measured) | 85.56 GiB | 100.09 GiB |

- Each session at `--context 196608` takes about 1.76 GiB with MTP and 1.55 GiB
  without. That is a 75.5 MiB recurrent state plus 8,068 bytes per context
  token, and 1,152 more with MTP for block 42. Add it for each extra
  `--sessions`.
- The expanded K/V store is one pool that all sessions share, first come first
  served (`--expanded-kv-bytes`, 12 GiB by default, 0 disables). It holds the
  expanded MLA keys and values that prefill reads. A smaller pool only slows
  deep-context speed, and outputs stay bit for bit identical.
- AD-Q5_K_M needs more than 100 GiB at the default store. I measured 97.5 GiB
  with AR and 100.1 GiB with MTP on a 108 GiB limit. On a 96 GiB limit, pass
  `--expanded-kv-bytes 4294967296`. That frees about 8 GiB. On the earlier
  Q4_K_S file, cutting the store from 12 to 4 GiB cut the peak from 81.7 to
  73.8 GiB. Q5_K_M should therefore peak near 90 GiB (AR) and 92 GiB (MTP)
  there, but this figure is an estimate, not a measurement.
- AD-Q4_K_S fits both a 96 and a 108 GiB limit at the defaults.
- A RAM cache checkpoint is 79.19 MB of private state plus 8,068 B per token
  with MTP off and 9,220 B per token with MTP. A 196,608-token conversation is
  about 1.67 GB without MTP and about 1.89 GB with it. Checkpoints of one
  conversation share their latent rows in 512-row chunks, so later checkpoints
  cost mostly their private state. Size `--cache-ram-bytes` for the contexts you
  serve. About 15 GB suits four 196k-token conversations with several
  checkpoints each.

## How it runs

- Decode quantizes activations to two int8 planes for the Q8_0 GEMVs and
  replays each decode step as a HIP graph. Forwards of up to eight rows (the
  decode family) and forwards of nine or more rows (the prefill family) each
  give a row the same result whatever the batch around it.
- Prefill uses WMMA GEMMs for the projections and experts, a token-sequential
  KDA recurrence, and non-absorbed MLA attention over the persistent expanded
  K/V store. The latent cache itself is F16. Expert GEMVs and GEMMs cover Q4_K,
  Q5_K, Q6_K, and IQ4_XS routed experts and Q8_0 or Q4_K token embeddings.
- MTP drafts from block 42, and the target verifies k+1 rows in the decode
  family, so verified rows equal single-token decode bit for bit. An adaptive
  policy falls back to plain decode when drafting does not pay.
- With MTP, prompt lookup also drafts. When the context ends in 12 or more
  tokens that occurred earlier in the prompt or reply (code edits, tool-output
  echoes, quotes), the tokens that followed are verified as the draft without
  running the drafter, up to seven per step. Greedy output is unchanged and
  sampled output keeps the target distribution. With the MTP drafts, the
  repetitive workload decodes at 2.2 to 3.0 times the autoregressive rate. On
  prose, where long copies are rare, MTP alone gives 0 to 1.4 times, depending
  on draft acceptance (AD-Q4_K_S 1.4 times at depth 0 and none at 131k;
  AD-Q5_K_M 1.12 and 1.16 times)
  ([Benchmarks](BENCHMARKS.md#single-user-mtp)).
- Up to eight decoding requests (at most `--sessions`) share one forward pass,
  with or without MTP. Each request's tokens, logits, and state stay
  bit-identical to decoding it alone
  ([Benchmarks](BENCHMARKS.md#multiple-users-autoregressive)).

## Serving

The server splits the reasoning block from the answer, and tool calls use the
Bailing syntax with a grammar. The default sampling preset (`ling3`) is
temperature 0.6, top-p 0.95, top-k 20. The RAM and disk continuation caches use
the component cache (shared latent-row chunks, copied recurrent state) with a
9-token checkpoint tail and a minimum checkpoint of 128 tokens.

## Known limits

- **Component cache.** Ling serves from the component cache (src/cache), with
  its adapter in `src/models/ling_3_flash/continuation_adapter.*`. Disk cache
  directories written by the earlier snapshot format are rebuilt cold. The
  benchmark figures on this card were measured on the component cache, with
  the v0.17.1 build.
- **Tool-call streaming.** The Bailing parser publishes a tool call only after
  its first argument value opens or an argument finishes. A header-only call (a
  function name with no argument) is never published, and an end of stream there
  yields plain text. Repeated argument keys with conflicting values make the
  call invalid. This also applies to non-streamed requests, where the last value
  used to win. The upstream functional suite that expects incremental
  tool-argument fragments passes.
- **Benchmark cost.** The 196,608-token llama.cpp cells take over an hour
  each. llama.cpp's single-user table up to 131k took 40 (Q4_K_S) and 42
  (Q5_K_M) minutes, and its 196k cell ran separately. Gufo's whole table,
  196k included, took 12 to 13 minutes. A driver left at its default compute
  timeout lost the device on the 196k llama.cpp cells, so the benchmark
  machines run a 180 s limit.
- **MTP comparison.** llama.cpp's `draft-mtp` does not reproduce its own
  autoregressive output on these files, so I compare Gufo's MTP rates with
  llama.cpp's autoregressive rate.
- **Cache edits under pressure.** `cache-edits` functional cases that rewrite
  an old turn are best effort when `--cache-ram-bytes` is only 4 GiB. They
  pass with 8 GiB.

## Tools and artifacts

Tests live in `tests/models/ling_3_flash`. Tools live in
`tools/models/ling_3_flash`: reference dumps, KL bases, comparison harnesses,
and the cold-file helper for the loading benchmark. `artifacts/` holds the
benchmark configuration, results, and charts.
