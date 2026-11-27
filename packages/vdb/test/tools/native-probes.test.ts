/**
 * Native OpenVDB / NanoVDB v14.0.0 reference values (K7.7: K4.10, K4.15, K4.17), checked against
 * our parser, builder, writer and CPU readers.
 *
 * The sidecars in `fixtures/native-probes/` are written by our native programs
 * (`docker/fixture-bake/vdb_probe.cpp`, `nvdb_probe.cpp`) through `pnpm parity:native --update`; they
 * hold probe values, counts, stats and hashes, never grid bytes (fixtures/ATTRIBUTION.md):
 *
 * - `bunny_cloud.density.json` (K4.10): native accessor values and active states at ~200 probes,
 *   counts, active tiles, bbox, min/max, f64 sums, a checksum of every active voxel, and the NanoVDB
 *   grid `createNanoGrid<FloatGrid, float>` makes (node / tile counts, boxes, root and sampled node
 *   stats). Needs `fixtures/vdb-samples/bunny_cloud.vdb` (not fetched; skipped without it).
 * - `explosion.v.json` (K4.15, native Vec3f): the same for the `vec3s` grid `v` of explosion.vdb.
 * - `waterfall_points.json` (K4.15): every 74,416th point's P, Cd, v and id through
 *   `points::AttributeHandle`, plus a bit hash and f64 sums of each attribute over all 9.5M points.
 * - `coarse-first.json` (K4.17): how native NanoVDB reads our multi-grid preview `.nvdb`
 *   (`coarse-first-source.mjs`, rebuilt here byte for byte): `io::readGridMetaData`, `io::hasGrid`,
 *   `io::readGrid` by name / by index / grid 0, `nanovdb_validate`, `nanovdb_convert` to `.vdb`.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  buildFromVdbDetailed,
  NvdbFile,
  parseVdb,
  vdbReadValueCpu,
  vdbReadValueVec3Cpu,
  type VDBBuiltGrid,
  type VdbGrid,
  type VdbPointDataGrid,
  type VdbPointList,
} from "../../src/tools/internal.js";
import * as tools from "../../src/tools/internal.js";
import { nodeStats, treeNodes } from "../../src/tools/cpu/march.js";
import { VEC3F_LAYOUT } from "../../src/tools/nanovdb/bytes.js";

type Coord = [number, number, number];

// Plain .mjs shared with scripts/run-native-parity.mjs, so both build the same bytes.
const { COARSE_FIRST_PROBES, coarseFirstFile } = (await import(
  /* @vite-ignore */ new URL("./fixtures/native-probes/coarse-first-source.mjs", import.meta.url).href
)) as { COARSE_FIRST_PROBES: Coord[]; coarseFirstFile: (t: typeof tools, codec: "none" | "zip") => Uint8Array };
type Value = number | Coord;

const PROBES = new URL("./fixtures/native-probes/", import.meta.url);
const SAMPLES = new URL("../../../../fixtures/vdb-samples/", import.meta.url);
const sidecar = <T>(name: string): T => JSON.parse(readFileSync(new URL(name, PROBES), "utf8")) as T;
const hasSample = (name: string): boolean => existsSync(new URL(name, SAMPLES));
function parseSample(name: string) {
  const b = readFileSync(new URL(name, SAMPLES));
  return parseVdb(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
}

//* Hashes shared with vdb_probe.cpp =========================================

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
const bitsOf = (v: number): number => {
  f32[0] = v;
  return u32[0]!;
};
function ulps(a: number, b: number): number {
  const x = bitsOf(a) | 0;
  const y = bitsOf(b) | 0;
  return Math.abs(x - y);
}
function fmix32(h: number): number {
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}
/** vdb_probe's `valueBits`: a float's bits, or the three components' bits mixed. */
function valueBits(v: Value): number {
  if (typeof v === "number") return bitsOf(v);
  return (bitsOf(v[0]) ^ Math.imul(bitsOf(v[1]), 0x01000193) ^ Math.imul(bitsOf(v[2]), 0x9e3779b1)) >>> 0;
}
/** vdb_probe's order-independent `VoxelSum` over (i, j, k, value bits). */
class VoxelSum {
  a = 0;
  b = 0;
  add(x: number, y: number, z: number, bits: number): void {
    let h = Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca77) ^ Math.imul(z, 0xc2b2ae3d);
    h = fmix32((h ^ bits) >>> 0);
    this.a = (this.a + h) >>> 0;
    this.b = (this.b + fmix32((h ^ 0x9e3779b9) >>> 0)) >>> 0;
  }
}
/** vdb_probe's `Fnv`: FNV-1a over little-endian u32 words. */
class Fnv {
  h = 0x811c9dc5;
  add(w: number): void {
    for (let i = 0; i < 4; i++) {
      this.h ^= (w >>> (8 * i)) & 0xff;
      this.h = Math.imul(this.h, 0x01000193) >>> 0;
    }
  }
}

const isOn = (mask: Uint32Array, n: number): boolean => ((mask[n >>> 5]! >>> (n & 31)) & 1) !== 0;
const f = Math.fround;
const sameValue = (got: Value, want: Value): boolean =>
  typeof want === "number" ? got === f(want) : (got as Coord).every((c, i) => c === f(want[i]!));

//* float / vec3 grids =======================================================

interface NodeStatsJson {
  origin: Coord;
  min: Value;
  max: Value;
  ave: number;
  stddev: number;
}
interface GridSidecar {
  openvdb: string;
  nanovdb: string;
  grid: string;
  type: string;
  class: string;
  background: Value;
  activeVoxelCount: number;
  leafCount: number;
  activeBBox: { min: Coord; max: Coord };
  min: Value;
  max: Value;
  sum: [number, number, number];
  sumSquares: number;
  voxelChecksum: [number, number];
  activeTiles: { origin: Coord; level: number; dim: number; value: Value }[];
  probes: { ijk: Coord; value: Value; active: boolean }[];
  nano: {
    probesAgree: boolean;
    nodeCount: { leaf: number; lower: number; upper: number };
    activeTileCount: { lower: number; upper: number; root: number };
    activeVoxelCount: number;
    indexBBox: { min: Coord; max: Coord };
    worldBBox: { min: Coord; max: Coord };
    root: Omit<NodeStatsJson, "origin">;
    upper: NodeStatsJson[];
    lower: NodeStatsJson[];
    leaf: NodeStatsJson[];
  };
}

/** Our node stats at a node address: scalar via the CPU twin, Vec3f straight from the layout. */
function ourStats(image: Uint32Array, level: 0 | 1 | 2 | 3, address: number, vec3: boolean): Omit<NodeStatsJson, "origin"> {
  if (!vec3) return nodeStats(image, level, address);
  const dv = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const L = VEC3F_LAYOUT;
  const [mn, mx, av, sd] = [
    [L.leafOffMin, L.leafOffMax, L.leafOffAve, L.leafOffStdDev],
    [L.lowerOffMin, L.lowerOffMax, L.lowerOffAve, L.lowerOffStdDev],
    [L.upperOffMin, L.upperOffMax, L.upperOffAve, L.upperOffStdDev],
    [L.rootOffMin, L.rootOffMax, L.rootOffAve, L.rootOffStdDev],
  ][level]!;
  const vec = (o: number): Coord => [0, 1, 2].map((c) => dv.getFloat32(address + o + 4 * c, true)) as Coord;
  return { min: vec(mn!), max: vec(mx!), ave: dv.getFloat32(address + av!, true), stddev: dv.getFloat32(address + sd!, true) };
}

/** Vec3f node origins (treeNodes reads scalar grids): walk the image's root, upper and lower tables. */
function vec3Nodes(image: Uint32Array): { level: 0 | 1 | 2 | 3; origin: Coord; address: number }[] {
  const v = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const L = VEC3F_LAYOUT;
  const bit = (base: number, n: number): boolean => ((v.getUint32(base + 4 * (n >>> 5), true) >>> (n & 31)) & 1) === 1;
  const root = 672 + Number(v.getBigUint64(672 + 24, true));
  const out: { level: 0 | 1 | 2 | 3; origin: Coord; address: number }[] = [{ level: 3, origin: [0, 0, 0], address: root }];
  for (let i = 0; i < v.getUint32(root + 24, true); i++) {
    const t = root + L.rootSize + i * L.rootTileSize;
    const child = Number(v.getBigInt64(t + 8, true));
    if (child === 0) continue;
    const key = v.getBigUint64(t, true);
    const uo: Coord = [Number((key >> 42n) & 0x1fffffn) << 12, Number((key >> 21n) & 0x1fffffn) << 12, Number(key & 0x1fffffn) << 12];
    const upper = root + child;
    out.push({ level: 2, origin: uo, address: upper });
    for (let un = 0; un < 32768; un++) {
      if (!bit(upper + 4128, un)) continue;
      const lo: Coord = [uo[0] + (un >> 10) * 128, uo[1] + ((un >> 5) & 31) * 128, uo[2] + (un & 31) * 128];
      const lower = upper + Number(v.getBigInt64(upper + L.upperOffTable + L.tableStride * un, true));
      out.push({ level: 1, origin: lo, address: lower });
      for (let ln = 0; ln < 4096; ln++) {
        if (!bit(lower + 544, ln)) continue;
        const leaf = lower + Number(v.getBigInt64(lower + L.lowerOffTable + L.tableStride * ln, true));
        out.push({ level: 0, origin: [lo[0] + (ln >> 8) * 8, lo[1] + ((ln >> 4) & 15) * 8, lo[2] + (ln & 15) * 8], address: leaf });
      }
    }
  }
  return out;
}

/** The shared float / Vec3f checks: parser vs native OpenVDB, our NanoVDB build vs native createNanoGrid. */
function describeGridParity(title: string, sample: string, gridName: string, sidecarName: string, vec3: boolean): void {
  describe.skipIf(!hasSample(sample))(title, () => {
    const native = sidecar<GridSidecar>(sidecarName);
    let grid: VdbGrid;
    let built: VDBBuiltGrid;
    const read = (ijk: Coord) => (vec3 ? vdbReadValueVec3Cpu(built.image, ijk) : vdbReadValueCpu(built.image, ijk));
    beforeAll(() => {
      grid = parseSample(sample).grids.find((g) => g.name === gridName)!;
      built = buildFromVdbDetailed(grid);
    }, 120_000);
    afterAll(() => {
      grid = undefined as unknown as VdbGrid;
      built = undefined as unknown as VDBBuiltGrid;
    });

    it("parser: counts, active tiles, bbox, background, min / max, sums and every active voxel (checksum) equal native OpenVDB", () => {
      expect(native.openvdb).toBe("14.0.0");
      expect(grid.gridType).toBe(native.type);
      expect(grid.activeVoxelCount).toBe(BigInt(native.activeVoxelCount));
      expect(grid.indexBBox).toEqual(native.activeBBox);
      expect(sameValue(grid.background as Value, native.background)).toBe(true);
      const tiles = [...grid.iterTiles()];
      expect(tiles.map((t) => ({ origin: t.origin, dim: t.dim }))).toEqual(native.activeTiles.map((t) => ({ origin: t.origin, dim: t.dim })));
      tiles.forEach((t, i) => expect(sameValue(t.value as Value, native.activeTiles[i]!.value)).toBe(true));

      const sum = new VoxelSum();
      const s1 = [0, 0, 0];
      let s2 = 0;
      let leaves = 0;
      let minM = Infinity;
      let maxM = -Infinity;
      let min: Value = 0;
      let max: Value = 0;
      const visit = (x: number, y: number, z: number, v: Value, weight: number): void => {
        const m2 = typeof v === "number" ? v : v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        const key = typeof v === "number" ? v : m2;
        if (key < minM) [minM, min] = [key, v];
        if (key > maxM) [maxM, max] = [key, v];
        if (typeof v === "number") s1[0] += weight * v;
        else for (let c = 0; c < 3; c++) s1[c] += weight * v[c]!;
        s2 += weight * (typeof v === "number" ? v * v : m2);
        if (weight === 1) sum.add(x, y, z, valueBits(v));
      };
      for (const leaf of grid.iterLeaves()) {
        leaves++;
        for (let n = 0; n < 512; n++) {
          if (!isOn(leaf.valueMask, n)) continue;
          const v: Value = vec3 ? [leaf.values[3 * n]!, leaf.values[3 * n + 1]!, leaf.values[3 * n + 2]!] : leaf.values[n]!;
          visit(leaf.origin[0] + (n >> 6), leaf.origin[1] + ((n >> 3) & 7), leaf.origin[2] + (n & 7), v, 1);
        }
      }
      for (const t of tiles) {
        const v = t.value as Value;
        visit(0, 0, 0, v, t.dim ** 3);
        for (let x = 0; x < t.dim; x++)
          for (let y = 0; y < t.dim; y++) for (let z = 0; z < t.dim; z++) sum.add(t.origin[0] + x, t.origin[1] + y, t.origin[2] + z, valueBits(v));
      }
      expect(leaves).toBe(native.leafCount);
      expect([sum.a, sum.b]).toEqual(native.voxelChecksum);
      expect(sameValue(min, native.min)).toBe(true);
      expect(sameValue(max, native.max)).toBe(true);
      for (let c = 0; c < 3; c++) expect(Math.abs(s1[c]! - native.sum[c]!)).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(native.sum[c]!)));
      expect(Math.abs(s2 - native.sumSquares)).toBeLessThanOrEqual(1e-9 * native.sumSquares);
    }, 120_000);

    it("parser and NanoVDB build read native's value and active state at every probe", () => {
      expect(native.nano.probesAgree).toBe(true); // native OpenVDB and NanoVDB accessors agree
      for (const p of native.probes) {
        const parsed = grid.readValue(p.ijk);
        expect(sameValue(parsed.value as Value, p.value), `parser value at ${p.ijk}`).toBe(true);
        expect(parsed.active, `parser active at ${p.ijk}`).toBe(p.active);
        const ours = read(p.ijk);
        expect(sameValue(ours.value as Value, p.value), `NanoVDB value at ${p.ijk}`).toBe(true);
        expect(ours.active, `NanoVDB active at ${p.ijk}`).toBe(p.active);
      }
    });

    it("NanoVDB build: node / tile counts, boxes, root and node stats equal native createNanoGrid", () => {
      expect(built.nodeCounts).toEqual(native.nano.nodeCount);
      expect(built.tileCounts).toEqual(native.nano.activeTileCount);
      expect(built.voxelCount).toBe(native.nano.activeVoxelCount);
      expect(built.indexBBox).toEqual(native.nano.indexBBox);
      for (const k of ["min", "max"] as const) built.worldBBox[k].forEach((w, a) => expect(w).toBeCloseTo(native.nano.worldBBox[k][a]!, 9));

      const nodes = vec3 ? vec3Nodes(built.image) : treeNodes(built.image);
      const byKey = new Map(nodes.map((n) => [`${n.level}:${n.origin.join(",")}`, n]));
      // ave / stddev: native GridStats accumulates f64 Welford, we f64 sums (DIVERGES V73): measured <= 8 f32 ulp
      // on every node of explosion `v` (pnpm parity:native), within 4 on the bunny.
      const expectStats = (label: string, ours: Omit<NodeStatsJson, "origin">, want: Omit<NodeStatsJson, "origin">): void => {
        expect(sameValue(ours.min, want.min), `${label} min`).toBe(true);
        expect(sameValue(ours.max, want.max), `${label} max`).toBe(true);
        expect(ulps(ours.ave, want.ave), `${label} ave ${ours.ave} vs ${want.ave}`).toBeLessThanOrEqual(vec3 ? 8 : 4);
        expect(ulps(ours.stddev, want.stddev), `${label} stddev ${ours.stddev} vs ${want.stddev}`).toBeLessThanOrEqual(vec3 ? 8 : 4);
      };
      expectStats("root", ourStats(built.image, 3, nodes[0]!.address, vec3), native.nano.root);
      const levels = [
        ["upper", 2],
        ["lower", 1],
        ["leaf", 0],
      ] as const;
      for (const [name, level] of levels) {
        expect(native.nano[name].length).toBeGreaterThan(0);
        for (const want of native.nano[name]) {
          const node = byKey.get(`${level}:${want.origin.join(",")}`);
          expect(node, `${name} ${want.origin}`).toBeDefined();
          expectStats(`${name} ${want.origin}`, ourStats(built.image, level, node!.address, vec3), want);
        }
      }
    });
  });
}

describeGridParity("bunny_cloud.vdb vs native OpenVDB / NanoVDB v14 (K4.10)", "bunny_cloud.vdb", "density", "bunny_cloud.density.json", false);
describeGridParity("explosion.vdb `v` vs native Vec3f (K4.15)", "explosion.vdb", "v", "explosion.v.json", true);

//* points =====================================================================

interface PointsSidecar {
  pointCount: number;
  sampleStep: number;
  attributes: { name: string; type: string; codec: string; sum: [number, number, number]; fnv1a: number }[];
  world: { sum: [number, number, number]; fnv1aF32: number };
  id: { min: number; max: number };
  samples: ({ index: number; ijk: Coord; world: Coord; offset: Coord } & Record<string, unknown>)[];
}

describe.skipIf(!hasSample("waterfall_points.vdb"))("waterfall_points.vdb vs native points::AttributeHandle (K4.15)", () => {
  const native = sidecar<PointsSidecar>("waterfall_points.json");
  let pts: VdbPointList;
  beforeAll(() => {
    pts = (parseSample("waterfall_points.vdb").grids[0] as VdbPointDataGrid).points();
  }, 120_000);
  afterAll(() => {
    pts = undefined as unknown as VdbPointList;
  });

  it("every point: Cd, v and id bit for bit (hash), sums, id range; world positions hash", () => {
    expect(pts.count).toBe(native.pointCount);
    for (const a of native.attributes.filter((x) => x.name !== "P")) {
      const arr = pts.attributes[a.name] as Float32Array | Int32Array;
      const comps = a.type === "vec3s" ? 3 : 1;
      expect(arr, a.name).toHaveLength(pts.count * comps);
      const fnv = new Fnv();
      const sum = [0, 0, 0];
      for (let i = 0; i < arr.length; i++) {
        fnv.add(a.type === "int32" ? arr[i]! >>> 0 : bitsOf(arr[i]!));
        sum[i % comps] += arr[i]!;
      }
      expect(fnv.h, `${a.name} hash`).toBe(a.fnv1a);
      for (let c = 0; c < comps; c++) expect(sum[c], `${a.name} sum ${c}`).toBe(a.sum[c]);
    }
    const ids = pts.attributes["id"] as Int32Array;
    let lo = Infinity;
    let hi = -Infinity;
    for (const id of ids) {
      if (id < lo) lo = id;
      if (id > hi) hi = id;
    }
    expect({ min: lo, max: hi }).toEqual(native.id);
    // World positions: native float(indexToWorld(ijk + P)) in f64, ours f32 of the same f64 expression.
    const fnv = new Fnv();
    const sum = [0, 0, 0];
    for (let i = 0; i < pts.positions.length; i++) {
      fnv.add(bitsOf(pts.positions[i]!));
      sum[i % 3] += pts.positions[i]!;
    }
    expect(fnv.h).toBe(native.world.fnv1aF32);
    for (let c = 0; c < 3; c++) expect(Math.abs(sum[c]! - native.world.sum[c]!)).toBeLessThanOrEqual(1e-6 * Math.abs(native.world.sum[c]!));
  }, 120_000);

  it("sampled points: world P, Cd, v and id equal native, point for point", () => {
    expect(native.samples.length).toBeGreaterThan(100);
    for (const s of native.samples) {
      const i = s.index;
      expect([0, 1, 2].map((c) => pts.positions[3 * i + c]), `P of ${i}`).toEqual(s.world.map(f));
      for (const name of ["Cd", "v"]) {
        const arr = pts.attributes[name] as Float32Array;
        expect([0, 1, 2].map((c) => arr[3 * i + c]), `${name} of ${i}`).toEqual((s[name] as Coord).map(f));
      }
      expect((pts.attributes["id"] as Int32Array)[i], `id of ${i}`).toBe(s["id"]);
    }
  });
});

//* coarse-first ===============================================================

interface NativeRead {
  name: string;
  type: string;
  gridIndex: number;
  gridCount: number;
  indexBBox: { min: Coord; max: Coord };
  activeVoxelCount: number;
  probes: { ijk: Coord; value: number; active: boolean }[] | null;
}
interface CoarseFirstSidecar {
  nanovdb: string;
  files: Record<
    "none" | "zip",
    {
      bytes: number;
      sha256: string;
      probe: {
        meta: { name: string; nameKey: string; stringHash: string; hasGrid: boolean; type: string; codec: string; gridSize: number; fileSize: number; voxelCount: number; indexBBox: { min: Coord; max: Coord } }[];
        grid0: string;
        byIndex: string[];
        byName: NativeRead[];
        all: string[];
      };
      validate: Record<string, string>;
      nanovdbConvertVdbGridOrder: string[];
    }
  >;
}

describe("coarse-first .nvdb read by native NanoVDB v14 (K4.17)", () => {
  const native = sidecar<CoarseFirstSidecar>("coarse-first.json");

  for (const codec of ["none", "zip"] as const) {
    describe(`codec ${codec}`, () => {
      const want = native.files[codec];
      const bytes = coarseFirstFile(tools, codec) as Uint8Array;
      const file = NvdbFile.fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);

      it("is the file native read (else rerun `pnpm parity:native --only=coarse-first --update`)", () => {
        expect(bytes.byteLength).toBe(want.bytes);
        expect(createHash("sha256").update(bytes).digest("hex")).toBe(want.sha256);
      });

      it("FileMetaData: native sees our names, types, sizes and boxes; nameKey is io::stringHash, so io::hasGrid finds every grid", () => {
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        let off = 16;
        want.probe.meta.forEach((m, i) => {
          expect(m.name).toBe(file.grids[i]!.name);
          expect(dv.getBigUint64(off + 16, true).toString(), `nameKey of ${m.name}`).toBe(m.stringHash);
          expect(m.hasGrid).toBe(true);
          expect(m.codec).toBe(codec.toUpperCase());
          expect(m.voxelCount).toBe(Number(dv.getBigUint64(off + 24, true)));
          off += 176 + dv.getUint32(off + 136, true);
        });
      });

      it("io::readGrid by name finds each grid, and its accessor reads our values; a grid-0 reader shows the coarsest preview", () => {
        expect(want.probe.byName.map((g) => g.name)).toEqual(file.grids.map((g) => g.name));
        expect(want.probe.all).toEqual(file.grids.map((g) => g.name));
        // Which grid "grid 0" readers (io::readGrid(file), EmberGen-style first-grid loaders) show: the preview.
        expect(want.probe.grid0).toBe(file.grids[0]!.name);
        expect(want.probe.grid0).toMatch(/\.lod8$/);
        want.probe.byName.forEach((g, i) => {
          const image = file.gridImage(i);
          expect(g.probes!.length).toBe(COARSE_FIRST_PROBES.length);
          for (const p of g.probes!) {
            const got = vdbReadValueCpu(image, p.ijk);
            // Native decodes FpN as f32 code * quantum + minimum; our oracle keeps the f64 value.
            expect(f(got.value), `${g.name} ${p.ijk}`).toBe(f(p.value));
            expect(got.active, `${g.name} ${p.ijk}`).toBe(p.active);
          }
        });
      });

      it("native tools: nanovdb_validate passes the Float grid and the single-leaf preview; nanovdb_convert keeps every grid", () => {
        const full = file.grids[file.grids.length - 1]!.name;
        expect(want.validate[full]).toBe("passed");
        expect(want.validate[file.grids[0]!.name]).toBe("passed");
        // v14's validator steps FpN leaves by the fixed header size: every multi-leaf FpN grid fails, native ones too.
        expect(want.validate[file.grids[1]!.name]).toBe("failed: Leaf node is not stored breadth-first");
        expect(want.nanovdbConvertVdbGridOrder).toEqual(file.grids.map((g) => g.name));
      });

      it("records v14's io::readGrid(file, n >= 1) defect inside a multi-grid segment", () => {
        // It never seeks past the grids before n: NONE reads grid 0 again, ZIP fails to inflate.
        expect(want.probe.byIndex[0]).toBe(file.grids[0]!.name);
        for (const r of want.probe.byIndex.slice(1)) {
          expect(r).toBe(codec === "none" ? file.grids[0]!.name : "error: UNZIP failed on byte size");
        }
      });
    });
  }
});
