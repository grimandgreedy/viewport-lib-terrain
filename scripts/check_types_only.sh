#!/usr/bin/env bash
# Check that the terrain data types build without the renderer.
#
# With `gpu` off the crate holds only the data types, so a loader or a scene
# format can name them without linking viewport-lib or wgpu. One stray import
# of a renderer type breaks that, and the default build never notices. This
# compiles the crate with `gpu` off and fails if viewport-lib or wgpu appears
# in that build's dependency tree.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo check -q --no-default-features
leaked="$(cargo tree -q --no-default-features -e normal --prefix none | grep -E '^(viewport-lib|wgpu[a-z-]*) v' || true)"
if [ -n "$leaked" ]; then
    echo "the data-only build of viewport-lib-terrain depends on:" >&2
    printf '%s\n' "$leaked" | sort -u >&2
    exit 1
fi
echo "ok: viewport-lib-terrain data types build without viewport-lib or wgpu"
