#!/bin/sh
# usage (from workspace root): tools/models/ling_3_flash/int/build-long-image.sh [tag]
# Stages build/release/gufo and ling-int-bench, then builds the long-context image locally.
set -eu
tag=${1:-localhost/gufo:ling-int2-long}
here=$(cd "$(dirname "$0")" && pwd); ctx=build/ling-long-image-ctx
rm -rf "$ctx"; mkdir -p "$ctx"
cp build/release/gufo "$ctx/gufo"; cp "$here/ling-int-bench" "$ctx/ling-int-bench"
podman build -t "$tag" --build-arg IMAGE="$tag" -f "$here/Containerfile.long" "$ctx"
