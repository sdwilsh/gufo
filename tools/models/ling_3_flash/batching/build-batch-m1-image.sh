#!/bin/sh
# usage (from the workspace root):
#   tools/models/ling_3_flash/batching/build-batch-m1-image.sh [tag]
# Stages this build's gfx1151 model tests (build/gpu-test, target
# ling_3_flash_model_tests) and release gufo (build/release), the parent's
# decode_digest and release gufo (PARENT_GPU_TEST, PARENT_GUFO; default the final5
# builds of ../gufo-ling-final5), the functional harness, a real-text corpus
# (docs and sources of this tree), the device-loss preload helper
# (build/gpu-test, target device_loss_faults) and the ling-batch-m1 driver into
# build/ling-batch-m1-ctx, and builds the image locally. Never pushes.
# Pods: command ["ling-batch-m1", "--stages", "tests,serve"] or
# ["ling-batch-m1", "--stages", "c1,agg"]; amd.com/gpu: 1, memory 19Gi,
# /models read-only, /work emptyDir, no HSA_OVERRIDE.
set -eu
tag=${1:-localhost/gufo-ling-batch-m1}
here=$(cd "$(dirname "$0")" && pwd); ctx=build/ling-batch-m1-ctx
B=build/gpu-test/tests/models/ling_3_flash
parent_tests=${PARENT_GPU_TEST:-../gufo-ling-final5/build/gpu-test/tests/models/ling_3_flash}
parent_gufo=${PARENT_GUFO:-../gufo-ling-final5/build/release/gufo}
rm -rf "$ctx"; mkdir -p "$ctx/tests" "$ctx/base" "$ctx/corpus" "$ctx/lib"
for t in batch_test batch_segments_test batch_mtp_test decode_digest session_test snapshot_test interleave_test restore_tail_test mtp_test; do
  cp "$B/ling_3_flash_$t" "$ctx/tests/"
done
cp "$parent_tests/ling_3_flash_decode_digest" "$ctx/base/ling_3_flash_decode_digest-parent"
cp "$parent_gufo" "$ctx/base/gufo-parent"
cp build/release/gufo "$ctx/gufo"
# Device-loss preload helper (target device_loss_faults).
cp build/gpu-test/libdevice_loss_faults.so "$ctx/lib/"
cp "$here/ling-batch-m1" "$ctx/"
cp -r tests/functional "$ctx/functional"; rm -rf "$ctx/functional/__pycache__"
# tests/functional/{tool_reasoning,pi_agent}.py import gufo.* from <root>/tools.
cp -r tools/gufo "$ctx/tools-gufo"; rm -rf "$ctx/tools-gufo/__pycache__" "$ctx"/tools-gufo/*/__pycache__
# run.py records the flake.lock digest from <tests>/../../flake.lock.
cp flake.lock "$ctx/flake.lock"
# Real text for distinct long prompts and MTP acceptance: docs, then sources.
{ find docs -name '*.md' | sort | xargs cat; find src -name '*.cpp' -o -name '*.hpp' | sort | xargs cat; } > "$ctx/corpus/corpus.txt"
sha256sum "$ctx"/gufo "$ctx"/base/* "$ctx"/tests/* > "$ctx/SHA256SUMS"
podman build -t "$tag" --build-arg IMAGE="$tag" -f "$here/Containerfile.batch-m1" "$ctx"
