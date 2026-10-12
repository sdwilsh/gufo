#!/bin/sh
# usage (from workspace root):
#   tools/models/ling_3_flash/int/build-int-image.sh [tag] [oracle-dir] [novel-oracle-dir] [peak-binary]
# Stages the build/gpu-test test binaries, build/release/gufo, the suite
# drivers, the functional harness, and the oracles into build/ling-int-image-ctx,
# then builds the image locally. Never pushes. oracle-dir holds cpu_ref/,
# greedy.jsonl, and kl_base.bin (memorised oracle). novel-oracle-dir holds
# greedy.jsonl and kl_base.bin.
# The report-only base session_test comes from the pre-merge image (BASE_IMAGE,
# default localhost/gufo-ling-int2r4). The device-loss preload helper comes from
# build/gpu-test/libdevice_loss_faults.so (target device_loss_faults).
# The small_ab, kernels_ab, decode_ab, and digest_ab stages of ling-int-bench
# compare against a parent build. They use these inputs:
#   - PARENT_GUFO (default build/base-release/gufo)
#   - the parent's decode_digest (PARENT_DIGEST, default
#     build/base-gpu-test/tests/models/ling_3_flash/ling_3_flash_decode_digest)
#   - the MLA prefill and MoE decode microbenchmarks, built from the parent's
#     and this tree's kernel source:
#       MLA_BENCH_PARENT and MLA_BENCH_CANDIDATE (default build/bench/mla_base
#       and build/bench/mla_new), from
#       tools/models/ling_3_flash/mla_prefill_bench.hip
#       MOE_BENCH_PARENT and MOE_BENCH_CANDIDATE (default build/bench/moe_base
#       and build/bench/moe_new), from
#       tools/models/ling_3_flash/moe_decode_bench.hip
# Optional experiment builds of the MoE bench, build/bench/moe_var-*, are staged
# as ling-moe-decode-bench-var-*. The MTP tests and suites
# (tools/models/ling_3_flash/mtp) are staged too.
set -eu
tag=${1:-localhost/gufo-ling-int2r7}
oracle=${2:-/tmp/b2-oracle}
novel=${3:-artifacts/ling-3-flash/oracle-novel}
peak=${4:-artifacts/ling-3-flash/container/gfx1151_peak}
here=$(cd "$(dirname "$0")" && pwd); ctx=build/ling-int-image-ctx
B=build/gpu-test/tests/models/ling_3_flash
base_image=${BASE_IMAGE:-localhost/gufo-ling-int2r4}
rm -rf "$ctx"; mkdir -p "$ctx/base" "$ctx/lib" "$ctx/tests" "$ctx/suites" "$ctx/oracle" "$ctx/oracle-novel" "$ctx/floor" "$ctx/adjudicated" "$ctx/corpus"
for t in dense_ops_test kda_ops_test mla_ops_test moe_ops_test session_test snapshot_test \
         interleave_test restore_tail_test layer_check b2 decode_digest logit_dump mtp_test \
         mtp_ops_test mtp_distribution mtp_policy_test mtp_sampled_chain_test \
         mtp_constrained_chain_test; do cp "$B/ling_3_flash_$t" "$ctx/tests/"; done
mtp="$here/../mtp"; mkdir -p "$ctx/fixtures"
cp tests/models/ling_3_flash/fixtures/mtp_reference_golden.txt "$ctx/fixtures/"
cp "$mtp/suites-mtp-tests.txt" "$ctx/suites/mtp-tests.txt"
cp "$mtp/suites-mtp-long-tests.txt" "$ctx/suites/mtp-long-tests.txt"
cp "$here/suites-decode-tests.txt" "$ctx/suites/decode-tests.txt"
cp "${PARENT_DIGEST:-build/base-gpu-test/tests/models/ling_3_flash/ling_3_flash_decode_digest}" \
  "$ctx/base/ling_3_flash_decode_digest"
cid=$(podman create "$base_image" true)
podman cp "$cid:/opt/gufo/tests/ling_3_flash_session_test" "$ctx/base/ling_3_flash_session_test"
podman rm "$cid" >/dev/null
cp build/gpu-test/libdevice_loss_faults.so "$ctx/lib/"
cp "$here/suites-base-session.txt" "$ctx/suites/base-session.txt"
cp "$here/suites-int-tests.txt" "$ctx/suites/int-tests.txt"
cp "$here/suites-int-tests.txt" "$ctx/suites/default.txt"
cp "$here/suites-token26.txt" "$ctx/suites/token26.txt"
cp tests/models/ling_3_flash/floor-memorised.json tests/models/ling_3_flash/floor-novel.json "$ctx/floor/"
cp tests/models/ling_3_flash/greedy_adjudicated.json "$ctx/adjudicated/"
cp tests/models/ling_3_flash/compare_routing.py "$ctx/compare_routing.py"
cp -L "$oracle/greedy.jsonl" "$oracle/kl_base.bin" "$ctx/oracle/"; cp -rL "$oracle/cpu_ref" "$ctx/oracle/"
cp -L "$novel/greedy.jsonl" "$novel/kl_base.bin" "$ctx/oracle-novel/"
# Real-text corpora (project docs and sources) for ling-int-bench --stages fit (MTP acceptance).
python3 "$here/../mtp/make_corpus.py" . "$ctx/corpus" "$novel/input_text.txt"
cp "$here/run-suite" "$ctx/run-suite"
cp build/release/gufo "$ctx/gufo"; cp "$peak" "$ctx/gfx1151_peak"
cp "${PARENT_GUFO:-build/base-release/gufo}" "$ctx/base/gufo-parent"
cp "${MLA_BENCH_PARENT:-build/bench/mla_base}" "$ctx/ling-mla-prefill-bench-parent"
cp "${MLA_BENCH_CANDIDATE:-build/bench/mla_new}" "$ctx/ling-mla-prefill-bench-candidate"
mkdir -p "$ctx/bin"
cp "${MOE_BENCH_PARENT:-build/bench/moe_base}" "$ctx/bin/ling-moe-decode-bench-parent"
cp "${MOE_BENCH_CANDIDATE:-build/bench/moe_new}" "$ctx/bin/ling-moe-decode-bench-candidate"
for v in build/bench/moe_var-*; do
  case "$v" in *.log | *.res) continue ;; esac
  [ -x "$v" ] && cp "$v" "$ctx/bin/ling-moe-decode-bench-var-${v##*/moe_var-}"
done
cp "$here/ling-int-suite" "$here/ling-int-bench" "$here/ling-serve-suite" "$here/ling-token26" "$here/../cache/ling-cache-suite" "$ctx/"
cp -r tests/functional "$ctx/functional"; rm -rf "$ctx/functional/__pycache__"
# tests/functional/{tool_reasoning,pi_agent}.py import gufo.* from <root>/tools.
cp -r tools/gufo "$ctx/tools-gufo"; rm -rf "$ctx/tools-gufo/__pycache__" "$ctx"/tools-gufo/*/__pycache__
cp flake.lock "$ctx/flake.lock"
podman build -t "$tag" --build-arg IMAGE="$tag" -f "$here/Containerfile.int" "$ctx"
