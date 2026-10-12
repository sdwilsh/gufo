#!/bin/sh
# usage (from workspace root, after building build/gpu-test and build/release):
#   tools/models/ling_3_flash/mtp/build-mtp-image.sh [tag] [oracle-dir] [novel-oracle-dir] [peak-binary]
# Stages the gpu-test binaries (standard + MTP tests), build/release/gufo, the suite drivers,
# the oracles and the real-text corpora into build/ling-mtp-image-ctx and builds the image
# locally. Never pushes. PARENT_GUFO (default build/base-release/gufo) becomes
# /opt/gufo/base/gufo-parent for ling-int-bench's mtp_head_ab stage.
set -eu
tag=${1:-localhost/gufo-ling-ws38}
oracle=${2:-/tmp/b2-oracle}
novel=${3:-artifacts/ling-3-flash/oracle-novel}
peak=${4:-artifacts/ling-3-flash/container/gfx1151_peak}
here=$(cd "$(dirname "$0")" && pwd); int="$here/../int"; ctx=build/ling-mtp-image-ctx
B=build/gpu-test/tests/models/ling_3_flash
rm -rf "$ctx"; mkdir -p "$ctx/base" "$ctx/tests" "$ctx/suites" "$ctx/oracle" "$ctx/oracle-novel" "$ctx/floor" \
  "$ctx/adjudicated" "$ctx/fixtures" "$ctx/corpus"
for t in dense_ops_test kda_ops_test mla_ops_test moe_ops_test session_test snapshot_test \
         interleave_test restore_tail_test layer_check b2 decode_digest logit_dump mtp_test \
         mtp_ops_test mtp_distribution mtp_head_ab \
         mtp_policy_test mtp_sampled_chain_test mtp_constrained_chain_test reload_test; do cp "$B/ling_3_flash_$t" "$ctx/tests/"; done
cp "$here/suites-mtp-tests.txt" "$ctx/suites/mtp-tests.txt"
cp "$int/suites-int-tests.txt" "$ctx/suites/int-tests.txt"
cp tests/models/ling_3_flash/floor-memorised.json tests/models/ling_3_flash/floor-novel.json "$ctx/floor/"
cp tests/models/ling_3_flash/greedy_adjudicated.json "$ctx/adjudicated/"
cp tests/models/ling_3_flash/fixtures/mtp_reference_golden.txt "$ctx/fixtures/"
cp tests/models/ling_3_flash/compare_routing.py "$ctx/compare_routing.py"
cp -L "$oracle/greedy.jsonl" "$oracle/kl_base.bin" "$ctx/oracle/"; cp -rL "$oracle/cpu_ref" "$ctx/oracle/"
cp -L "$novel/greedy.jsonl" "$novel/kl_base.bin" "$ctx/oracle-novel/"
python3 "$here/make_corpus.py" . "$ctx/corpus" "$novel/input_text.txt"
cp "$int/run-suite" "$ctx/run-suite"
cp "${PARENT_GUFO:-build/base-release/gufo}" "$ctx/base/gufo-parent"
cp build/release/gufo "$ctx/gufo"; cp "$peak" "$ctx/gfx1151_peak"; cp tools/prof/prof.py "$ctx/prof.py"
cp "$here/ling-mtp-suite" "$int/ling-int-suite" "$int/ling-int-bench" "$int/ling-serve-suite" tools/models/ling_3_flash/cache/ling-cache-suite "$ctx/"
cp -r tests/functional "$ctx/functional"; rm -rf "$ctx/functional/__pycache__"
# tests/functional/{tool_reasoning,pi_agent}.py import gufo.* from <root>/tools.
cp -r tools/gufo "$ctx/tools-gufo"; rm -rf "$ctx/tools-gufo/__pycache__" "$ctx"/tools-gufo/*/__pycache__
cp flake.lock "$ctx/flake.lock"
podman build -t "$tag" --build-arg IMAGE="$tag" -f "$here/Containerfile.mtp" "$ctx"
