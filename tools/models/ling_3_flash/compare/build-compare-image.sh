#!/bin/sh
# usage (from workspace root): tools/models/ling_3_flash/compare/build-compare-image.sh [tag] [gufo-binary]
# Stages build/ling-compare-image-ctx and builds the comparison image locally.  Never pushes.
set -eu
tag=${1:-localhost/gufo:ling-compare}; gufo=${2:-build/release/gufo}
here=$(cd "$(dirname "$0")" && pwd); ctx=build/ling-compare-image-ctx
rm -rf "$ctx"; mkdir -p "$ctx/compare"
cp "$gufo" "$ctx/gufo"
cp tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json "$ctx/"
cp "$here/ling-compare" "$ctx/"
cp -r "$here/." "$ctx/compare/"
find "$ctx/compare" -name __pycache__ -prune -exec rm -rf {} +
rm -f "$ctx/compare/fake_gufo.py" "$ctx/compare/Containerfile.compare" "$ctx/compare/build-compare-image.sh" "$ctx/compare/ling-compare"
podman build -t "$tag" --build-arg IMAGE="$tag" -f "$here/Containerfile.compare" "$ctx"
