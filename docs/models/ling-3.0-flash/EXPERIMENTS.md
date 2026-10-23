# Ling-3.0-Flash experiments

Retained and rejected decisions, newest evidence first within a topic. Dates
are 2026. Speed figures come from the development runs named, not from the
[benchmark tables](BENCHMARKS.md). Log prefixes are SHA-256 prefixes of the
recorded run logs.

| Experiment | Decision / evidence |
| --- | --- |
| Corrected AD-Q4_K_S and AD-Q5_K_M quants | Retained (2026-10-11/12). The expert kernels were already templated over IQ4_XS, Q4_K, Q5_K, and Q6_K. The work was validation and glue so that gate/up may be Q4_K or Q6_K, down Q5_K, and the token embedding Q8_0 (`EmbedQ8_0`, bit-exact against the dequantized rows). Both files load and pass the engine tests on the first 12 layers, and the earlier file's decode digests are byte-identical before and after the change |
| Prompt-lookup drafts for MTP | Retained (2026-10-11). A 3-token incremental index (about 10 MB per 200k tokens of prose) proposes the tokens after an earlier match of 12 or more tokens, and the verify pass stays in the decode family. Prose is within 0.6% of plain MTP on the development GPU. With perfect copies, prompt lookup reaches 2.87x autoregressive there |
| Runner-up verification (verify the drafter's second choice) | Rejected (2026-10-11). A sibling row costs 0.11 to 0.25 of a step on this MoE, and each KDA sibling needs a 70 MiB branch state. The estimate is -3% on prose and -7% on agent text unless more than about 60% of the siblings are accepted |
| Background copy of snapshot latent rows | Superseded (2026-10-17). It was retained on 2026-10-11 for the legacy cache. Ling-3.0-Flash now serves from the component cache, which borrows device rows and spills them when idle, so the copy and its row holds were removed. Measured then: checkpoint capture 69.5 to 16.7 ms and first token 102.4 to 55.8 ms at K=1 on the development GPU. On the full model, cache restore is 13 to 16 ms faster (43 to 29 ms) and time to first token is 41 to 65 ms lower |
| Expanded K/V store budget | Retained as a shared pool, 12 GiB by default. On the earlier Q4_K_S file a store of 0 / 4 / 12 GiB peaked at 69.9 / 73.8 / 81.7 GiB. Outputs are bit-identical and only deep-context speed differs |
| Batched decode, up to eight sessions | Retained. Each row is bit-identical to solo decode. MTP cohorts share one draft width chosen from measured costs |
| Two-plane int8 decode activations | Retained. Decode relative L2 error against the reference fell from 4.46e-3 to 7.8e-5 (B2r) |
| Batch-invariance contract | Retained (WS7, B2r3). GemvQ8_0 sub-batching, an even/odd WMMA split for the MLA tail tile, and a decode family of up to eight rows. With these, 0 of 42 layers differ between 3-, 5- and 8-row batches and token-by-token decode |
| HIP-graph decode and launch fusion | Retained (WS10, 2026-10-03). d0 decode 33.65 to 35.01 tok/s, +3.6-4.0%. The GEMV tweaks of round 5 (+0.4-0.5%) were not merged |
| Token-sequential KDA prefill | Retained (WS11, 2026-10-03/04). Prefill +2-5%. A chunked WY form was rejected because the recurrence is about 1% of pp2048, and the CHAOS study (ws11r4 d8346584) and the F64 adjudication of story token 26 showed that the order of arithmetic mattered more than the form |
| Non-absorbed MLA prefill with a persistent expanded K/V store | Retained (WS12, 2026-10-03). 64k prefill 227.8 to 408.8 tok/s. A row-count threshold for the absorbed path was rejected because it breaks split invariance. Contiguous V tiles (ws31) fixed the small-chunk regression. pp16 is 2.26x / 2.55x the old rate at 16k / 64k |
| Prefill GEMM efficiency | Retained (WS13, 2026-10-04). d0 prefill 603.8 to 837.4 tok/s, bit-identical. `--max-batch 2048` reaches 996 and stays a library option. `serve` keeps 512 because a 2048-row forward blocks decode for about 3.4 s |
| MTP from block 42 | Retained as the default (WS22, 2026-10-04) after 1.10-1.50x measured and agent loops about 1.34-1.45x. llama.cpp's Vulkan MTP on Ling-3.0-Flash is a slowdown (19.66 to 10.02 tok/s at 57k) |
| HIP graphs for MTP cycles | Rejected as low value. Eager minus graph is 0.57-0.59 ms per step, about 2% (ws31-mtp-profile cd1c7c2d) |
| MoE decode GEMVs | Rejected (WS34 r1, 2026-10-06) as a net loss, 0.955-0.991x, and 0.64-0.78x for Q5_K/Q6_K gate/up. 860M timings had misled the first look (ws34-ab c1bab9b6). Round 3 keeps the router and KDA PrepDecode fusion, which is bit-identical and gives decode +0.7-0.9% and MTP +1.9-2.6% (ws34r3-ab 35a6d09d) |
| DSpark draft (inclusionAI, 1.36B BF16) | Investigation only, not built. Estimated 1.0-1.6x on chat and below AR at 64k. The licence file is missing (DSPARK-SPIKE) |
| DSpark on the XDNA2 NPU | Rejected without building (2026-10-06), based on the upstream `perf/qwen27b-dflash2-npu` measurements. The NPU streams weights at about 39.5 GB/s against the GPU's 240, and a GPU to NPU boundary costs about 2.1 ms. In those measurements, a DFlash head shard on the NPU ran 2.5% slower, an MTP `eh_proj` on the NPU ran 3.00 against 3.01 tok/s (3.73 on the GPU alone), and DS4 expert offload was 12-34x slower. The authors keep batch-1 decode on the GPU. For the Ling-3.0-Flash DSpark, 1.8-3 GB per cycle would take 45-75 ms on the NPU against 9-15 ms on the GPU. The work would be a serial dependency on shared memory bandwidth, and amdxdna firmware timeouts would occur under combined load (upstream #1605). No NPU run was made on this model |
| Sampled-MTP agent-score scare | No effect (2026-10-05). cmp-agent-ab (49953eb0) read MTP 35/42 against off 39/42 (Fisher p 0.31). Fresh seeds in cmp-t09-t13 (26ee1e46) gave 5/16 against 4/16 (p = 1). Replay with fresh seeds before concluding |
| Prompt-cache design | Retained (WS15/19). Self-contained snapshots, a 9-token tail, and prompt-end checkpoints for the coding-agent strip mode |
| llama.cpp KV q8_0 against f16 | Measurement note. On Vulkan, f16 K/V gave about 2.6x the prefill and 1.23x the decode at 64k, with no quality cost. The production flags stay q8_0, so the benchmarks use q8_0, and this control was not repeated for the new files |

## History

These are measurements of the earlier AD-Q4_K_S build (69.1 GiB, same file name,
IQ4_XS routed gate/up experts and a Q4_K token embedding), kept for context. They
were taken on machine B before its board was replaced, with an earlier Gufo
release candidate, and are not comparable cell for cell with the
[benchmarks](BENCHMARKS.md).

| Measurement | Result |
| --- | --- |
| Gufo spot run (rc4), pp2048 / tg128, AR, depth 0 and 32k | pp 1027.5 and 752.7 tok/s; tg 36.06 and 33.80 tok/s |
| Gufo spot run (rc4), MTP prose, depth 0 and 32k | tg 50.39 and 44.11 tok/s, 72.7% and 71.9% of drafts accepted |
| Gufo device memory, one session | 81.7 GiB AR and 83.9 GiB MTP (12 GiB expanded store) |

The corrected Q4_K_S file is 1.8 GB larger (Q8_0 token embedding, Q5_K edge
experts) and decodes at about the earlier rate, 35.78 against 36.06 tok/s at
depth 0, on different machines.
