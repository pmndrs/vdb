// vdb_probe: native OpenVDB / NanoVDB v14.0.0 reference values for a .vdb file (K4.10, K4.15).
// MIT, Copyright (c) 2026 Dennis Smolek and Poimandres contributors.
//
//   vdb_probe float  <in.vdb> <grid-name> <out.json>
//       A FloatGrid (or HalfGrid, read as float) through native accessors: counts, active tiles, the
//       active bbox, background, an order-independent checksum of every active voxel (ijk + value
//       bits), f64 sums, ~200 probes (value + active state), and the NanoVDB grid
//       `tools::createNanoGrid<FloatGrid, float>` makes from it (node / tile counts, bboxes, root
//       stats, stats of sampled upper / lower / leaf nodes), checked against the OpenVDB accessor.
//   vdb_probe points <in.vdb> <out.json> [samples]
//       A PointDataGrid through `points::AttributeHandle`: per-point P (voxel offset and world
//       position), every Vec3f / int32 attribute at `samples` evenly spaced point indices (global
//       order: leaf order, then index order in the leaf), f64 sums and a bit hash of each attribute
//       over every point.
//
// Probe coordinates come from mulberry32 (the tests' PRNG) and leaf order, so a rerun writes the same
// file. Values are printed with %.9g (round-trips f32) and %.17g (f64).
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/AttributeArray.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace openvdb;

//* Shared helpers =============================================================

struct Rng { // mulberry32, as the TS tests
    uint32_t a;
    explicit Rng(uint32_t seed) : a(seed) {}
    double next()
    {
        a += 0x6d2b79f5u;
        uint32_t t = a;
        t = (t ^ (t >> 15)) * (t | 1u);
        t ^= t + (t ^ (t >> 7)) * (t | 61u);
        return double(t ^ (t >> 14)) / 4294967296.0;
    }
};

static uint32_t fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

/// Order-independent voxel checksum: two wrapping u32 sums of a mix of (i, j, k, value bits).
struct VoxelSum {
    uint32_t a = 0, b = 0;
    void add(const Coord& c, uint32_t bits)
    {
        uint32_t h = uint32_t(c[0]) * 0x9E3779B1u;
        h ^= uint32_t(c[1]) * 0x85EBCA77u;
        h ^= uint32_t(c[2]) * 0xC2B2AE3Du;
        h = fmix32(h ^ bits);
        a += h;
        b += fmix32(h ^ 0x9E3779B9u);
    }
};

/// FNV-1a over u32 words (order-dependent), as the TS tests.
struct Fnv {
    uint32_t h = 0x811c9dc5u;
    void add(uint32_t w)
    {
        for (int i = 0; i < 4; ++i) {
            h ^= (w >> (8 * i)) & 0xffu;
            h *= 0x01000193u;
        }
    }
};

static uint32_t bitsOf(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}

static GridBase::Ptr readGrid(const std::string& file, const std::string& name)
{
    io::File in(file);
    in.open();
    GridBase::Ptr grid = name.empty() ? in.getGrids()->front() : in.readGrid(name);
    in.close();
    return grid;
}

//* Value traits (float and Vec3f grids) ======================================

static uint32_t valueBits(float v) { return bitsOf(v); }
static uint32_t valueBits(const Vec3f& v) { return bitsOf(v[0]) ^ (bitsOf(v[1]) * 0x01000193u) ^ (bitsOf(v[2]) * 0x9E3779B1u); }
static uint32_t valueBits(const nanovdb::Vec3f& v) { return valueBits(Vec3f(v[0], v[1], v[2])); }
static double magnitude2(float v) { return double(v) * double(v); }
static double magnitude2(const Vec3f& v) { return double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]; }
static void addSum(double* s, float v) { s[0] += v; }
static void addSum(double* s, const Vec3f& v) { for (int c = 0; c < 3; ++c) s[c] += v[c]; }
static std::string str(float v)
{
    char b[32];
    std::snprintf(b, sizeof b, "%.9g", double(v));
    return b;
}
static std::string str(const Vec3f& v)
{
    char b[96];
    std::snprintf(b, sizeof b, "[%.9g, %.9g, %.9g]", double(v[0]), double(v[1]), double(v[2]));
    return b;
}
static std::string str(const nanovdb::Vec3f& v) { return str(Vec3f(v[0], v[1], v[2])); }

//* float / vec3 ===============================================================

template<typename GridT, typename NanoT>
static void probeGrid(const std::string& file, const std::string& name, const std::string& out)
{
    using ValueT = typename GridT::ValueType;
    GridBase::Ptr base = readGrid(file, name);
    if (!base->isType<GridT>()) throw std::runtime_error("grid " + name + " is " + base->type() + ", not " + GridT::gridType());
    typename GridT::Ptr grid = gridPtrCast<GridT>(base);
    const auto& tree = grid->tree();
    auto acc = grid->getConstAccessor();

    // Tree walk: leaves (in tree order), active tiles, checksum, sums, min/max (by magnitude for vectors).
    std::vector<Coord> leaves;
    struct Tile { Coord origin; Index level; Index dim; ValueT value; };
    std::vector<Tile> tiles;
    VoxelSum sum;
    double s1[3] = {0, 0, 0}, s2 = 0.0;
    ValueT vmin = zeroVal<ValueT>(), vmax = zeroVal<ValueT>();
    double m2min = std::numeric_limits<double>::max(), m2max = -1.0;
    uint64_t active = 0;
    auto extrema = [&](const ValueT& v) {
        const double m2 = std::is_same<ValueT, float>::value ? 0.0 : magnitude2(v);
        if constexpr (std::is_same<ValueT, float>::value) {
            if (active == 0 || v < vmin) vmin = v;
            if (active == 0 || v > vmax) vmax = v;
        } else {
            if (m2 < m2min) { m2min = m2; vmin = v; }
            if (m2 > m2max) { m2max = m2; vmax = v; }
        }
    };
    for (auto leaf = tree.cbeginLeaf(); leaf; ++leaf) leaves.push_back(leaf->origin());
    for (auto it = tree.cbeginValueOn(); it; ++it) {
        const ValueT v = *it;
        extrema(v);
        if (it.isVoxelValue()) {
            sum.add(it.getCoord(), valueBits(v));
            addSum(s1, v);
            s2 += magnitude2(v);
            ++active;
        } else {
            const CoordBBox box = it.getBoundingBox();
            tiles.push_back({box.min(), it.getLevel(), Index(box.dim()[0]), v});
            for (auto c = box.begin(); c; ++c) sum.add(*c, valueBits(v));
            const double n = double(box.volume());
            for (int c = 0; c < (std::is_same<ValueT, float>::value ? 1 : 3); ++c) {
                double one[3] = {0, 0, 0};
                addSum(one, v);
                s1[c] += n * one[c];
            }
            s2 += n * magnitude2(v);
            active += box.volume();
        }
    }
    if (active != tree.activeVoxelCount()) throw std::runtime_error("active count mismatch in the walk");
    CoordBBox bbox;
    tree.evalActiveVoxelBoundingBox(bbox);

    // Probe coordinates: tile corners and neighbours, random leaf voxels (any / active), random bbox coords, far away.
    std::vector<Coord> probes;
    for (const Tile& t : tiles) {
        probes.push_back(t.origin);
        probes.push_back(t.origin.offsetBy(int(t.dim) - 1));
        probes.push_back(t.origin.offsetBy(-1, 0, 0));
        probes.push_back(t.origin.offsetBy(int(t.dim), 0, 0));
    }
    Rng rng(0x4b410a);
    for (int k = 0; k < 64; ++k) {
        const Coord o = leaves[size_t(rng.next() * double(leaves.size()))];
        const int n = int(rng.next() * 512.0);
        probes.push_back(o.offsetBy(n >> 6, (n >> 3) & 7, n & 7));
    }
    for (int k = 0; k < 64; ++k) {
        const Coord o = leaves[size_t(rng.next() * double(leaves.size()))];
        int n = int(rng.next() * 512.0);
        const auto* leaf = tree.probeConstLeaf(o);
        for (int step = 0; step < 512 && !leaf->isValueOn(Index(n)); ++step) n = (n + 1) & 511;
        probes.push_back(o.offsetBy(n >> 6, (n >> 3) & 7, n & 7));
    }
    for (int k = 0; k < 64; ++k) {
        Coord c;
        for (int a = 0; a < 3; ++a) c[a] = bbox.min()[a] + int(rng.next() * double(bbox.dim()[a]));
        probes.push_back(c);
    }
    probes.push_back(bbox.min().offsetBy(-1));
    probes.push_back(bbox.max().offsetBy(1));
    probes.push_back(Coord(-100000, 5, 5));

    // NanoVDB: the native conversion (stats All, the default).
    auto handle = nanovdb::tools::createNanoGrid<GridT, NanoT>(*grid);
    const nanovdb::NanoGrid<NanoT>* nano = handle.template grid<NanoT>();
    const auto& ntree = nano->tree();
    auto nacc = nano->getAccessor();
    int disagree = 0;
    for (const Coord& c : probes) {
        const nanovdb::Coord nc(c[0], c[1], c[2]);
        if (valueBits(nacc.getValue(nc)) != valueBits(acc.getValue(c)) || nacc.isActive(nc) != acc.isValueOn(c)) ++disagree;
    }

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + out);
    std::fprintf(f, "{\n  \"openvdb\": \"%s\",\n  \"nanovdb\": \"%d.%d.%d\",\n", OPENVDB_LIBRARY_VERSION_STRING,
                 NANOVDB_MAJOR_VERSION_NUMBER, NANOVDB_MINOR_VERSION_NUMBER, NANOVDB_PATCH_VERSION_NUMBER);
    std::fprintf(f, "  \"grid\": \"%s\",\n  \"type\": \"%s\",\n  \"class\": \"%s\",\n  \"background\": %s,\n",
                 grid->getName().c_str(), GridT::gridType().c_str(), GridBase::gridClassToString(grid->getGridClass()).c_str(),
                 str(grid->background()).c_str());
    std::fprintf(f, "  \"activeVoxelCount\": %llu,\n  \"leafCount\": %llu,\n", (unsigned long long)tree.activeVoxelCount(),
                 (unsigned long long)tree.leafCount());
    std::fprintf(f, "  \"activeBBox\": {\"min\": [%d, %d, %d], \"max\": [%d, %d, %d]},\n", bbox.min()[0], bbox.min()[1],
                 bbox.min()[2], bbox.max()[0], bbox.max()[1], bbox.max()[2]);
    std::fprintf(f, "  \"min\": %s,\n  \"max\": %s,\n  \"sum\": [%.17g, %.17g, %.17g],\n  \"sumSquares\": %.17g,\n",
                 str(vmin).c_str(), str(vmax).c_str(), s1[0], s1[1], s1[2], s2);
    std::fprintf(f, "  \"voxelChecksum\": [%u, %u],\n  \"activeTiles\": [", sum.a, sum.b);
    for (size_t i = 0; i < tiles.size(); ++i)
        std::fprintf(f, "%s{\"origin\": [%d, %d, %d], \"level\": %u, \"dim\": %u, \"value\": %s}", i ? ", " : "",
                     tiles[i].origin[0], tiles[i].origin[1], tiles[i].origin[2], tiles[i].level, tiles[i].dim,
                     str(tiles[i].value).c_str());
    std::fprintf(f, "],\n  \"probes\": [\n");
    for (size_t i = 0; i < probes.size(); ++i) {
        const Coord& c = probes[i];
        std::fprintf(f, "    {\"ijk\": [%d, %d, %d], \"value\": %s, \"active\": %s}%s\n", c[0], c[1], c[2],
                     str(acc.getValue(c)).c_str(), acc.isValueOn(c) ? "true" : "false", i + 1 < probes.size() ? "," : "");
    }
    std::fprintf(f, "  ],\n  \"nano\": {\n    \"probesAgree\": %s,\n", disagree == 0 ? "true" : "false");
    std::fprintf(f, "    \"nodeCount\": {\"leaf\": %u, \"lower\": %u, \"upper\": %u},\n", ntree.nodeCount(0),
                 ntree.nodeCount(1), ntree.nodeCount(2));
    std::fprintf(f, "    \"activeTileCount\": {\"lower\": %u, \"upper\": %u, \"root\": %u},\n", ntree.activeTileCount(1),
                 ntree.activeTileCount(2), ntree.activeTileCount(3));
    std::fprintf(f, "    \"activeVoxelCount\": %llu,\n", (unsigned long long)nano->activeVoxelCount());
    const auto ib = nano->indexBBox();
    const auto wb = nano->worldBBox();
    std::fprintf(f, "    \"indexBBox\": {\"min\": [%d, %d, %d], \"max\": [%d, %d, %d]},\n", ib.min()[0], ib.min()[1],
                 ib.min()[2], ib.max()[0], ib.max()[1], ib.max()[2]);
    std::fprintf(f, "    \"worldBBox\": {\"min\": [%.17g, %.17g, %.17g], \"max\": [%.17g, %.17g, %.17g]},\n", wb.min()[0],
                 wb.min()[1], wb.min()[2], wb.max()[0], wb.max()[1], wb.max()[2]);
    const auto& root = ntree.root();
    std::fprintf(f, "    \"root\": {\"min\": %s, \"max\": %s, \"ave\": %.9g, \"stddev\": %.9g},\n", str(root.minimum()).c_str(),
                 str(root.maximum()).c_str(), double(root.average()), double(root.stdDeviation()));
    auto nodeStats = [&](const char* key, auto* first, uint32_t count, uint32_t want, bool last) {
        std::fprintf(f, "    \"%s\": [", key);
        const uint32_t step = std::max<uint32_t>(1, count / want);
        bool firstOut = true;
        for (uint32_t i = 0; i < count; i += step) {
            const auto* node = first + i;
            const auto o = node->origin();
            std::fprintf(f, "%s\n      {\"origin\": [%d, %d, %d], \"min\": %s, \"max\": %s, \"ave\": %.9g, \"stddev\": %.9g}",
                         firstOut ? "" : ",", o[0], o[1], o[2], str(node->minimum()).c_str(), str(node->maximum()).c_str(),
                         double(node->average()), double(node->stdDeviation()));
            firstOut = false;
        }
        std::fprintf(f, "\n    ]%s\n", last ? "" : ",");
    };
    nodeStats("upper", ntree.template getFirstNode<2>(), ntree.nodeCount(2), 16, false);
    nodeStats("lower", ntree.template getFirstNode<1>(), ntree.nodeCount(1), 16, false);
    nodeStats("leaf", ntree.template getFirstNode<0>(), ntree.nodeCount(0), 32, true);
    std::fprintf(f, "  }\n}\n");
    std::fclose(f);
    if (disagree) throw std::runtime_error(std::to_string(disagree) + " probes differ between OpenVDB and NanoVDB");
}

//* points =====================================================================

static void probePoints(const std::string& file, const std::string& out, uint64_t samples)
{
    GridBase::Ptr base = readGrid(file, "");
    if (!base->isType<points::PointDataGrid>()) throw std::runtime_error("first grid is not a PointDataGrid");
    auto grid = gridPtrCast<points::PointDataGrid>(base);
    const math::Transform& xform = grid->transform();
    const auto& tree = grid->tree();

    auto leafIt = tree.cbeginLeaf();
    if (!leafIt) throw std::runtime_error("no leaves");
    const points::AttributeSet::Descriptor& desc = leafIt->attributeSet().descriptor();
    struct Attr { std::string name; std::string type; std::string codec; size_t index; bool vec3; double sum[3] = {0, 0, 0}; Fnv fnv; };
    std::vector<Attr> attrs;
    for (const auto& [name, index] : desc.map()) {
        const NamePair& t = desc.type(index);
        if (t.first != "vec3s" && t.first != "int32") continue;
        attrs.push_back({name, t.first, t.second, index, t.first == "vec3s"});
    }
    std::sort(attrs.begin(), attrs.end(), [](const Attr& a, const Attr& b) { return a.index < b.index; });

    uint64_t total = 0;
    for (auto leaf = tree.cbeginLeaf(); leaf; ++leaf) total += leaf->pointCount();
    const uint64_t step = std::max<uint64_t>(1, total / samples);

    std::string sampleJson;
    Fnv worldFnv;
    double worldSum[3] = {0, 0, 0};
    int64_t idMin = INT64_MAX, idMax = INT64_MIN;
    uint64_t g = 0;
    char buf[512];
    for (auto leaf = tree.cbeginLeaf(); leaf; ++leaf) {
        std::vector<points::AttributeHandle<Vec3f>::Ptr> vec(attrs.size());
        std::vector<points::AttributeHandle<int32_t>::Ptr> ints(attrs.size());
        for (size_t a = 0; a < attrs.size(); ++a) {
            const points::AttributeArray& array = leaf->constAttributeArray(attrs[a].index);
            if (attrs[a].vec3) vec[a] = points::AttributeHandle<Vec3f>::create(array);
            else ints[a] = points::AttributeHandle<int32_t>::create(array);
        }
        const size_t pIndex = desc.find("P");
        auto pHandle = points::AttributeHandle<Vec3f>::create(leaf->constAttributeArray(pIndex));
        Index start = 0;
        for (Index n = 0; n < points::PointDataTree::LeafNodeType::SIZE; ++n) {
            const Index end = leaf->getValue(n);
            const Coord ijk = leaf->offsetToGlobalCoord(n);
            for (Index i = start; i < end; ++i, ++g) {
                const Vec3f p = pHandle->get(i);
                const Vec3d world = xform.indexToWorld(p + ijk.asVec3d());
                for (int c = 0; c < 3; ++c) {
                    worldSum[c] += world[c];
                    worldFnv.add(bitsOf(float(world[c])));
                }
                std::string attrJson;
                for (size_t a = 0; a < attrs.size(); ++a) {
                    if (attrs[a].vec3) {
                        const Vec3f v = vec[a]->get(i);
                        for (int c = 0; c < 3; ++c) {
                            attrs[a].sum[c] += v[c];
                            attrs[a].fnv.add(bitsOf(v[c]));
                        }
                        if (g % step == 0) {
                            std::snprintf(buf, sizeof buf, ", \"%s\": [%.9g, %.9g, %.9g]", attrs[a].name.c_str(), double(v[0]),
                                          double(v[1]), double(v[2]));
                            attrJson += buf;
                        }
                    } else {
                        const int32_t v = ints[a]->get(i);
                        attrs[a].sum[0] += double(v);
                        attrs[a].fnv.add(uint32_t(v));
                        if (attrs[a].name == "id") {
                            idMin = std::min<int64_t>(idMin, v);
                            idMax = std::max<int64_t>(idMax, v);
                        }
                        if (g % step == 0) {
                            std::snprintf(buf, sizeof buf, ", \"%s\": %d", attrs[a].name.c_str(), v);
                            attrJson += buf;
                        }
                    }
                }
                if (g % step == 0) {
                    std::snprintf(buf, sizeof buf, "%s\n    {\"index\": %llu, \"ijk\": [%d, %d, %d], \"world\": [%.9g, %.9g, %.9g], \"offset\": [%.9g, %.9g, %.9g]",
                                  sampleJson.empty() ? "" : ",", (unsigned long long)g, ijk[0], ijk[1], ijk[2], double(float(world[0])),
                                  double(float(world[1])), double(float(world[2])), double(p[0]), double(p[1]), double(p[2]));
                    sampleJson += buf;
                    sampleJson += attrJson + "}";
                }
            }
            start = end;
        }
    }

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + out);
    std::fprintf(f, "{\n  \"openvdb\": \"%s\",\n  \"grid\": \"%s\",\n  \"pointCount\": %llu,\n  \"sampleStep\": %llu,\n",
                 OPENVDB_LIBRARY_VERSION_STRING, grid->getName().c_str(), (unsigned long long)total, (unsigned long long)step);
    std::fprintf(f, "  \"attributes\": [");
    for (size_t a = 0; a < attrs.size(); ++a)
        std::fprintf(f, "%s{\"name\": \"%s\", \"type\": \"%s\", \"codec\": \"%s\", \"sum\": [%.17g, %.17g, %.17g], \"fnv1a\": %u}",
                     a ? ", " : "", attrs[a].name.c_str(), attrs[a].type.c_str(), attrs[a].codec.c_str(), attrs[a].sum[0],
                     attrs[a].sum[1], attrs[a].sum[2], attrs[a].fnv.h);
    std::fprintf(f, "],\n  \"world\": {\"sum\": [%.17g, %.17g, %.17g], \"fnv1aF32\": %u},\n", worldSum[0], worldSum[1],
                 worldSum[2], worldFnv.h);
    if (idMin <= idMax) std::fprintf(f, "  \"id\": {\"min\": %lld, \"max\": %lld},\n", (long long)idMin, (long long)idMax);
    std::fprintf(f, "  \"samples\": [%s\n  ]\n}\n", sampleJson.c_str());
    std::fclose(f);
}

int main(int argc, char** argv)
{
    try {
        initialize();
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "float" && argc == 5) {
            probeGrid<FloatGrid, float>(argv[2], argv[3], argv[4]);
        } else if (mode == "vec3" && argc == 5) {
            probeGrid<Vec3SGrid, nanovdb::Vec3f>(argv[2], argv[3], argv[4]);
        } else if (mode == "points" && (argc == 4 || argc == 5)) {
            probePoints(argv[2], argv[3], argc == 5 ? std::stoull(argv[4]) : 128);
        } else {
            std::fprintf(stderr, "usage: vdb_probe float|vec3 <in.vdb> <grid> <out.json>\n"
                                 "       vdb_probe points <in.vdb> <out.json> [samples]\n");
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vdb_probe: %s\n", e.what());
        return 1;
    }
    return 0;
}
