// nvdb_probe: how native NanoVDB v14.0.0 (`nanovdb/io/IO.h`) reads a .nvdb file (K4.17).
// MIT, Copyright (c) 2026 Dennis Smolek and Poimandres contributors.
//
//   nvdb_probe <in.nvdb> <out.json> [i,j,k ...]
//
// Records, as JSON:
//   meta      io::readGridMetaData: every grid's name, nameKey, type, class, codec, sizes, counts, boxes,
//             plus io::stringHash(name) and io::hasGrid(file, name) (which needs nameKey == stringHash)
//   grid0     io::readGrid(file) with the default index (what a "grid 0" reader loads)
//   byIndex   io::readGrid(file, n) for every n
//   byName    io::readGrid(file, name) for every name
//   all       io::readGrid(file, -1): every grid in one buffer (its grid count)
// Each read grid lists its name, type, index bbox, active voxel count, checksum mode and the value /
// active state at the given index coordinates (Float, Fp4/8/16/N and Half grids). A read that throws is
// recorded as {"error": …} (v14's io::readGrid(file, n) fails for n >= 1 inside a multi-grid segment).
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/HalfLite.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nanovdb;

static std::vector<Coord> gCoords;

static float toFloat(float v) { return v; }
static float toFloat(Half v) { return math::HalfToFloat(v.raw); }

template<typename BuildT>
static std::string probes(const GridData* data)
{
    auto* grid = reinterpret_cast<const NanoGrid<BuildT>*>(data);
    auto acc = grid->getAccessor();
    std::string s = "[";
    char buf[160];
    for (size_t i = 0; i < gCoords.size(); ++i) {
        const Coord& c = gCoords[i];
        std::snprintf(buf, sizeof buf, "%s{\"ijk\": [%d, %d, %d], \"value\": %.9g, \"active\": %s}", i ? ", " : "", c[0], c[1],
                      c[2], double(toFloat(acc.getValue(c))), acc.isActive(c) ? "true" : "false");
        s += buf;
    }
    return s + "]";
}

static std::string describe(const GridHandle<HostBuffer>& handle, uint32_t n = 0)
{
    const GridData* data = handle.gridData(n);
    if (!data) return "null";
    const GridMetaData* meta = handle.gridMetaData(n);
    std::string values = "null";
    switch (meta->gridType()) {
    case GridType::Float: values = probes<float>(data); break;
    case GridType::Fp4: values = probes<Fp4>(data); break;
    case GridType::Fp8: values = probes<Fp8>(data); break;
    case GridType::Fp16: values = probes<Fp16>(data); break;
    case GridType::FpN: values = probes<FpN>(data); break;
    case GridType::Half: values = probes<Half>(data); break;
    default: break;
    }
    const auto b = meta->indexBBox();
    char buf[512], type[64];
    std::snprintf(buf, sizeof buf,
                  "{\"name\": \"%s\", \"type\": \"%s\", \"gridIndex\": %u, \"gridCount\": %u, \"indexBBox\": "
                  "{\"min\": [%d, %d, %d], \"max\": [%d, %d, %d]}, \"activeVoxelCount\": %llu, \"probes\": ",
                  meta->shortGridName(), toStr(type, meta->gridType()), meta->gridIndex(), meta->gridCount(), b.min()[0], b.min()[1],
                  b.min()[2], b.max()[0], b.max()[1], b.max()[2], (unsigned long long)meta->activeVoxelCount());
    return std::string(buf) + values + "}";
}

/// A read's description, or {"error": what} when native throws (recorded, not fatal).
template<typename F>
static std::string guarded(F&& read)
{
    try {
        return read();
    } catch (const std::exception& e) {
        std::string msg = e.what();
        for (auto& c : msg) if (c == '"' || c == '\\') c = '\'';
        return "{\"error\": \"" + msg + "\"}";
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: nvdb_probe <in.nvdb> <out.json> [i,j,k ...]\n");
        return 2;
    }
    const std::string file = argv[1];
    for (int a = 3; a < argc; ++a) {
        int i, j, k;
        if (std::sscanf(argv[a], "%d,%d,%d", &i, &j, &k) != 3) {
            std::fprintf(stderr, "nvdb_probe: bad coordinate %s\n", argv[a]);
            return 2;
        }
        gCoords.emplace_back(i, j, k);
    }
    FILE* f = std::fopen(argv[2], "w");
    if (!f) return 1;
    try {
        const auto meta = io::readGridMetaData(file);
        char type[64], cls[64], codec[64];
        std::fprintf(f, "{\n  \"nanovdb\": \"%d.%d.%d\",\n  \"meta\": [", NANOVDB_MAJOR_VERSION_NUMBER,
                     NANOVDB_MINOR_VERSION_NUMBER, NANOVDB_PATCH_VERSION_NUMBER);
        for (size_t n = 0; n < meta.size(); ++n) {
            const auto& m = meta[n];
            std::fprintf(f,
                         "%s\n    {\"name\": \"%s\", \"nameKey\": \"%llu\", \"stringHash\": \"%llu\", \"hasGrid\": %s, \"type\": \"%s\", "
                         "\"class\": \"%s\", \"codec\": \"%s\", \"gridSize\": %llu, \"fileSize\": %llu, \"voxelCount\": %llu, "
                         "\"indexBBox\": {\"min\": [%d, %d, %d], \"max\": [%d, %d, %d]}, \"nodeCount\": [%u, %u, %u, %u], "
                         "\"tileCount\": [%u, %u, %u], \"version\": \"%u.%u.%u\"}",
                         n ? "," : "", m.gridName.c_str(), (unsigned long long)m.nameKey,
                         (unsigned long long)io::stringHash(m.gridName), io::hasGrid(file, m.gridName) ? "true" : "false",
                         toStr(type, m.gridType),
                         toStr(cls, m.gridClass), toStr(codec, m.codec), (unsigned long long)m.gridSize, (unsigned long long)m.fileSize,
                         (unsigned long long)m.voxelCount, m.indexBBox.min()[0], m.indexBBox.min()[1], m.indexBBox.min()[2],
                         m.indexBBox.max()[0], m.indexBBox.max()[1], m.indexBBox.max()[2], m.nodeCount[0], m.nodeCount[1],
                         m.nodeCount[2], m.nodeCount[3], m.tileCount[0], m.tileCount[1], m.tileCount[2],
                         m.version.getMajor(), m.version.getMinor(), m.version.getPatch());
        }
        std::fprintf(f, "\n  ],\n  \"grid0\": %s,\n  \"byIndex\": [", describe(io::readGrid(file)).c_str());
        for (size_t n = 0; n < meta.size(); ++n)
            std::fprintf(f, "%s\n    %s", n ? "," : "", guarded([&] { return describe(io::readGrid(file, int(n))); }).c_str());
        std::fprintf(f, "\n  ],\n  \"byName\": [");
        for (size_t n = 0; n < meta.size(); ++n)
            std::fprintf(f, "%s\n    %s", n ? "," : "", guarded([&] { return describe(io::readGrid(file, meta[n].gridName)); }).c_str());
        const auto all = io::readGrid(file, -1);
        std::fprintf(f, "\n  ],\n  \"all\": {\"gridCount\": %u, \"grids\": [", all.gridCount());
        for (uint32_t n = 0; n < all.gridCount(); ++n) std::fprintf(f, "%s\n    %s", n ? "," : "", describe(all, n).c_str());
        std::fprintf(f, "\n  ]}\n}\n");
    } catch (const std::exception& e) {
        std::fclose(f);
        std::fprintf(stderr, "nvdb_probe: %s\n", e.what());
        return 1;
    }
    std::fclose(f);
    return 0;
}
