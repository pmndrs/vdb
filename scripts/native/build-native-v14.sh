#!/bin/sh
# Builds the native OpenVDB / NanoVDB v14.0.0 tools WITHOUT Docker (K7.7), into a
# durable cache outside the repo:
#
#   ${PMNDRS_VDB_NATIVE:-$HOME/.cache/pmndrs-vdb-native}/v14.0.0/
#     prefix/bin  nanovdb_print, nanovdb_convert, nanovdb_validate, vdb_print, and ours:
#                 normalize_vdb, vdb_probe, nvdb_probe, bake_half and the native-v14
#                 generators (gen_nvdb_blosc, gen_vdb_half)
#     prefix/lib  libopenvdb, libtbb
#
# oneTBB v2022.1.0 and OpenVDB v14.0.0 come from their GitHub tag tarballs
# (sha256-pinned below). Blosc is vendored since v14 and Boost is gone; zstd
# is off. Needs cmake, a C++17 compiler and zlib. Idempotent: a stage whose
# output exists is skipped (delete the folder to rebuild). The same steps run
# in docker/fixture-bake/Dockerfile.
#
#   sh scripts/native/build-native-v14.sh            # build (or reuse) the cache
#   sh scripts/native/build-native-v14.sh --print    # print the prefix and exit
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CACHE=${PMNDRS_VDB_NATIVE:-$HOME/.cache/pmndrs-vdb-native}
BASE="$CACHE/v14.0.0"
PREFIX="$BASE/prefix"
JOBS=${BUILD_JOBS:-4}

if [ "${1:-}" = "--print" ]; then
  echo "$PREFIX"
  exit 0
fi

OPENVDB_TAG=v14.0.0 # ecae2bda85744412b0cd47e1b8e1e4e76beb7b88
OPENVDB_SHA256=2c38592a42a9dc1f3e9749418cc9d286d88ed912232dafee954ce5156174e5b0
TBB_TAG=v2022.1.0
TBB_SHA256=ed067603ece0dc832d2881ba5c516625ac2522c665d95f767ef6304e34f961b5

DL="$CACHE/downloads"
SRC="$BASE/src"
BUILD="$BASE/build"
mkdir -p "$DL" "$SRC" "$BUILD" "$PREFIX/bin"

fetch() { # url file sha256
  if [ ! -f "$DL/$2" ]; then
    echo "• fetching $1" >&2
    curl -sSfL -o "$DL/$2.part" "$1"
    mv "$DL/$2.part" "$DL/$2"
  fi
  got=$(shasum -a 256 "$DL/$2" | cut -d' ' -f1)
  if [ "$got" != "$3" ]; then
    echo "✗ $DL/$2: sha256 $got, expected $3" >&2
    exit 1
  fi
}

fetch "https://codeload.github.com/uxlfoundation/oneTBB/tar.gz/refs/tags/$TBB_TAG" "onetbb-2022.1.0.tar.gz" "$TBB_SHA256"
fetch "https://codeload.github.com/AcademySoftwareFoundation/openvdb/tar.gz/refs/tags/$OPENVDB_TAG" "openvdb-14.0.0.tar.gz" "$OPENVDB_SHA256"
[ -d "$SRC/oneTBB-2022.1.0" ] || tar -xzf "$DL/onetbb-2022.1.0.tar.gz" -C "$SRC"
[ -d "$SRC/openvdb-14.0.0" ] || tar -xzf "$DL/openvdb-14.0.0.tar.gz" -C "$SRC"

GENERATOR=""
command -v ninja >/dev/null 2>&1 && GENERATOR="-GNinja"

#* oneTBB ======================================================================
if [ ! -f "$PREFIX/lib/cmake/TBB/TBBConfig.cmake" ]; then
  echo "• building oneTBB $TBB_TAG" >&2
  cmake $GENERATOR -S "$SRC/oneTBB-2022.1.0" -B "$BUILD/tbb" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
    -DTBB_TEST=OFF -DTBB_EXAMPLES=OFF -DTBB_STRICT=OFF -DTBBMALLOC_BUILD=ON
  cmake --build "$BUILD/tbb" -j"$JOBS"
  cmake --install "$BUILD/tbb"
fi

#* OpenVDB + NanoVDB tools =====================================================
if [ ! -x "$PREFIX/bin/nanovdb_convert" ]; then
  echo "• building OpenVDB $OPENVDB_TAG (core, vdb_print, NanoVDB tools)" >&2
  cmake $GENERATOR -S "$SRC/openvdb-14.0.0" -B "$BUILD/openvdb" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_PREFIX_PATH="$PREFIX" -DTBB_ROOT="$PREFIX" \
    -DOPENVDB_BUILD_CORE=ON -DOPENVDB_CORE_STATIC=OFF -DOPENVDB_BUILD_BINARIES=ON \
    -DOPENVDB_BUILD_UNITTESTS=OFF -DOPENVDB_BUILD_PYTHON_MODULE=OFF -DUSE_ZSTD=OFF -DUSE_CCACHE=OFF \
    -DUSE_NANOVDB=ON -DNANOVDB_USE_OPENVDB=ON -DNANOVDB_USE_TBB=ON -DNANOVDB_BUILD_TOOLS=ON \
    -DCMAKE_INSTALL_RPATH="$PREFIX/lib"
  cmake --build "$BUILD/openvdb" -j"$JOBS"
  cmake --install "$BUILD/openvdb"
fi

#* Our programs (MIT) ==========================================================
# Built against the installed headers/libs; rebuilt when a source is newer.
CXX=${CXX:-c++}
CC=${CC:-cc}
BLOSC_DIR="$SRC/openvdb-14.0.0/ext/blosc/openvdb/ext/blosc"
GEN="$ROOT/packages/vdb/test/tools/fixtures/native-v14/generators"
TOOLS="$ROOT/docker/fixture-bake"
FLAGS="-std=c++17 -O1 -ffp-contract=off"
LINK="-L$PREFIX/lib -lopenvdb -ltbb -Wl,-rpath,$PREFIX/lib"

stale() { # out src...
  out=$1; shift
  [ -x "$out" ] || return 0
  for s in "$@"; do [ "$s" -nt "$out" ] && return 0; done
  return 1
}

if stale "$BUILD/nanovdb_blosc.o" "$BLOSC_DIR/nanovdb_blosc.c"; then
  $CC -c -O1 -I"$BLOSC_DIR" "$BLOSC_DIR/nanovdb_blosc.c" -o "$BUILD/nanovdb_blosc.o"
  chmod +x "$BUILD/nanovdb_blosc.o"
fi
if stale "$PREFIX/bin/gen_nvdb_blosc" "$GEN/gen_nvdb_blosc.cpp"; then
  echo "• gen_nvdb_blosc" >&2
  $CXX $FLAGS -DNANOVDB_USE_BLOSC -I"$PREFIX/include" -I"$BLOSC_DIR" "$GEN/gen_nvdb_blosc.cpp" \
    "$BUILD/nanovdb_blosc.o" -o "$PREFIX/bin/gen_nvdb_blosc"
fi
for prog in "$GEN/gen_vdb_half.cpp" "$TOOLS/normalize_vdb.cpp" "$TOOLS/vdb_probe.cpp" "$TOOLS/nvdb_probe.cpp"; do
  name=$(basename "$prog" .cpp)
  if stale "$PREFIX/bin/$name" "$prog"; then
    echo "• $name" >&2
    # nvdb_probe reads .nvdb files (io::readGrid, ZIP and BLOSC codecs) and links OpenVDB's zlib only.
    $CXX $FLAGS -DNANOVDB_USE_OPENVDB -DNANOVDB_USE_BLOSC -DNANOVDB_USE_ZIP -I"$PREFIX/include" -I"$BLOSC_DIR" "$prog" \
      "$BUILD/nanovdb_blosc.o" $LINK -lz -o "$PREFIX/bin/$name"
  fi
done

# bake_half (K7.H): header-only NanoVDB, -O2 as scripts/bake-half-fixtures.mjs.
if stale "$PREFIX/bin/bake_half" "$TOOLS/bake_half.cpp"; then
  echo "• bake_half" >&2
  $CXX -std=c++17 -O2 -ffp-contract=off -I"$PREFIX/include" "$TOOLS/bake_half.cpp" -o "$PREFIX/bin/bake_half"
fi

echo "✓ native v14 tools in $PREFIX/bin" >&2
