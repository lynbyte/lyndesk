#!/usr/bin/env bash
# Build a static libx264 for phantom's optional `x264` encoder feature.
#
#   scripts/build-x264.sh            # -> target/x264/{include,lib}
#   X264_DIR=/opt/x264 scripts/build-x264.sh
#
# Then: cargo build --release -p phantom-server --features x264
# (crates/server/build.rs looks in $X264_DIR, default <workspace>/target/x264).
#
# Linux/macOS: needs a C compiler, make and nasm. Windows (x86_64-pc-windows-gnu):
# run from Git Bash with MinGW-w64 gcc, nasm and mingw32-make on PATH.
#
# LICENSE: x264 is GPL-2.0-or-later. A phantom binary built with `--features x264`
# links it and must be distributed under the GPL. The default build does not.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${X264_DIR:-$ROOT/target/x264}"
SRC="${X264_SRC:-$ROOT/target/x264-src}"
# Pinned for reproducible builds (x264 stable branch).
REV="${X264_REV:-stable}"

if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch "$REV" https://code.videolan.org/videolan/x264.git "$SRC"
fi

MAKE=make
command -v make >/dev/null 2>&1 || MAKE=mingw32-make

cd "$SRC"
./configure --prefix="$PREFIX" --enable-static --disable-cli --enable-pic \
    --disable-opencl --bit-depth=8 --chroma-format=420
"$MAKE" -j"$(nproc 2>/dev/null || echo 4)"
"$MAKE" install-lib-static
echo "x264 installed to $PREFIX"
