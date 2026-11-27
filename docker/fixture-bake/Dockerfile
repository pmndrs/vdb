# Reproducible native-tools environment (K7.7): OpenVDB / NanoVDB v14.0.0 tools plus our native
# programs, the same set scripts/native/build-native-v14.sh builds without Docker. Per D6 this image is
# also the seed of the interim companion service.
#
# Build (the generators come in as a second build context):
#   docker build -t webvdb-fixture-bake \
#     --build-context generators=packages/vdb/test/tools/fixtures/native-v14/generators docker/fixture-bake
# Bake the primitives (default command):
#   docker run --rm -v "$PWD/fixtures:/out" webvdb-fixture-bake
# Native parity (mounts the repo at /repo):
#   pnpm build && pnpm parity:native --native=docker
# Tools:
#   docker run --rm -v "$PWD/fixtures:/out" webvdb-fixture-bake nanovdb_print -l /out/wdas/wdas_cloud_quarter_fp8.nvdb
#
# v14.0.0 vendors Blosc (ext/blosc) and dropped Boost, so the image needs only TBB and zlib. The
# primitive bake stays on the a532de55 NanoVDB headers, the vendored PNanoVDB.h pin (K7.5): its
# sidecars are identical under v14, but its .nvdb files would carry version 32.11.0 and another checksum.

FROM ubuntu:24.04 AS build
ARG OPENVDB_TAG=v14.0.0
ARG OPENVDB_SHA256=2c38592a42a9dc1f3e9749418cc9d286d88ed912232dafee954ce5156174e5b0
ARG PRIMITIVES_COMMIT=a532de5526ef791280b6483a872336a811a68542
ARG BUILD_JOBS=2

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ca-certificates curl libtbb-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN curl -sSfL -o openvdb.tar.gz "https://codeload.github.com/AcademySoftwareFoundation/openvdb/tar.gz/refs/tags/${OPENVDB_TAG}" \
    && echo "${OPENVDB_SHA256}  openvdb.tar.gz" | sha256sum -c - \
    && tar -xzf openvdb.tar.gz && mv openvdb-* openvdb && rm openvdb.tar.gz

RUN cmake -S openvdb -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENVDB_BUILD_CORE=ON \
      -DOPENVDB_CORE_STATIC=OFF \
      -DOPENVDB_BUILD_BINARIES=ON \
      -DOPENVDB_BUILD_UNITTESTS=OFF \
      -DOPENVDB_BUILD_PYTHON_MODULE=OFF \
      -DUSE_ZSTD=OFF \
      -DUSE_CCACHE=OFF \
      -DUSE_NANOVDB=ON \
      -DNANOVDB_USE_OPENVDB=ON \
      -DNANOVDB_USE_TBB=ON \
      -DNANOVDB_BUILD_TOOLS=ON \
    && cmake --build build -j"${BUILD_JOBS}" \
    && cmake --install build

# Our programs (MIT), with the flags of scripts/native/build-native-v14.sh.
ARG BLOSC=/src/openvdb/ext/blosc/openvdb/ext/blosc
COPY normalize_vdb.cpp vdb_probe.cpp nvdb_probe.cpp bake_half.cpp bake_primitives.cpp /src/ours/
COPY --from=generators gen_nvdb_blosc.cpp gen_vdb_half.cpp /src/ours/
RUN set -e; cd /src/ours \
    && gcc -c -O1 -I"${BLOSC}" "${BLOSC}/nanovdb_blosc.c" -o nanovdb_blosc.o \
    && g++ -std=c++17 -O1 -ffp-contract=off -DNANOVDB_USE_BLOSC -I/usr/local/include -I"${BLOSC}" gen_nvdb_blosc.cpp nanovdb_blosc.o \
         -o /usr/local/bin/gen_nvdb_blosc \
    && for p in gen_vdb_half normalize_vdb vdb_probe nvdb_probe; do \
         g++ -std=c++17 -O1 -ffp-contract=off -DNANOVDB_USE_OPENVDB -DNANOVDB_USE_BLOSC -DNANOVDB_USE_ZIP -I/usr/local/include \
           -I"${BLOSC}" "$p.cpp" nanovdb_blosc.o -L/usr/local/lib -lopenvdb -ltbb -lz -o "/usr/local/bin/$p"; \
       done \
    && g++ -std=c++17 -O2 -ffp-contract=off -I/usr/local/include bake_half.cpp -o /usr/local/bin/bake_half

# bake_primitives: header-only, on the PNanoVDB.h pin.
RUN curl -sSfL -o pin.tar.gz "https://codeload.github.com/AcademySoftwareFoundation/openvdb/tar.gz/${PRIMITIVES_COMMIT}" \
    && mkdir pin && tar -xzf pin.tar.gz -C pin --strip-components=1 && rm pin.tar.gz \
    && g++ -std=c++17 -O2 -ffp-contract=off -I pin/nanovdb /src/ours/bake_primitives.cpp -o /usr/local/bin/bake_primitives

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libtbb12 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local /usr/local
COPY bake-fixtures.sh /usr/local/bin/bake-fixtures.sh
RUN ldconfig && chmod +x /usr/local/bin/bake-fixtures.sh

VOLUME /out
ENTRYPOINT ["/usr/local/bin/bake-fixtures.sh"]
