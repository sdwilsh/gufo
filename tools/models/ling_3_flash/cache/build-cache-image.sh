#!/bin/sh
# usage (from workspace root): tools/models/ling_3_flash/cache/build-cache-image.sh [tag] [gufo-binary]
# Stages build/ling-cache-image-ctx and builds the cache-validation image locally. Never pushes.
set -eu
tag=${1:-localhost/gufo:ling-ws15r2}; gufo=${2:-build/release/gufo}
here=$(cd "$(dirname "$0")" && pwd); ctx=build/ling-cache-image-ctx
rm -rf "$ctx"; mkdir -p "$ctx"
cp "$gufo" "$ctx/gufo"
cp "$here/ling-cache-suite" "$ctx/ling-cache-suite"
cp -r tests/functional "$ctx/functional"; rm -rf "$ctx/functional/__pycache__"
cp flake.lock "$ctx/flake.lock"
podman build -t "$tag" --build-arg IMAGE="$tag" -f "$here/Containerfile.cache" "$ctx"
