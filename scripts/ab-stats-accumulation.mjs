#!/usr/bin/env node
// K7.H1 A/B: node ave/stddev from our two stats accumulations ('sums' = f64 sum / sum of squares, the
// K1–K7.H path; 'welford' = NanoVDB GridStats' f64 update) against native NanoVDB. Needs `pnpm build`.
//
//   node scripts/ab-stats-accumulation.mjs
//
// Native references (git-ignored, all optional; each row says what it read):
//   - fixtures/half/half_edge.sidecar.json (committed): native createNanoGrid<build::Grid<float>, Half> stats,
//     incl. leaves that overflowed to +-inf.
//   - fixtures/primitives/{sphere,torus,box}_fog_half.nvdb (`pnpm fixtures:bake-half`, OpenVDB v14.0.0)
//     against vdbQuantize(<name>_float.nvdb, 'half').
//   - fixtures/primitives/*_float.nvdb (`pnpm fixtures:bake`, a532de55) against vdbQuantize(float, 'float').
//   - fixtures/parity/<sample>.native.float.nvdb (`pnpm parity:native`) against buildFromVdb(<sample>.normalized.vdb).
// For every node of both trees (matched by level and origin) it counts ave/stddev f32 values that are
// bit-identical to native, the largest ulp distance, and non-finite mismatches.

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  NvdbFile,
  VDB_HALF_LEAF_CODEC,
  buildFromVdb,
  parseVdb,
  treeNodes,
  vdbBuildFromLeavesDetailed,
  vdbQuantize,
  vdbSetStatsAccumulation,
} from "../packages/vdb/dist/tools/internal.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const F = (rel) => path.join(ROOT, "fixtures", rel);
const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const load = (p) => NvdbFile.fromArrayBuffer(ab(readFileSync(p))).gridImage(0);

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
const fromBits = (b) => ((u32[0] = b >>> 0), f32[0]);
const ord = (b) => (b & 0x80000000 ? -(b & 0x7fffffff) : b);
function ulps(a, b) {
  const x = fromBits(a);
  const y = fromBits(b);
  if (Number.isNaN(x) && Number.isNaN(y)) return 0;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return x === y ? 0 : Infinity;
  return Math.abs(ord(a >>> 0) - ord(b >>> 0));
}

/** ave/stddev f32 bits of every node, keyed by level and origin. Offsets: [ave] per level for Float / Half. */
function aveStd(image) {
  const dv = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const half = dv.getUint32(636, true) === 9;
  const ave = half ? [84, 1060, 8228, 36] : [88, 1064, 8232, 40];
  const out = new Map();
  for (const n of treeNodes(image)) {
    const off = n.level === 3 ? n.address + ave[3] : n.address + ave[n.level];
    out.set(`${n.level}:${n.level === 3 ? "root" : n.origin.join(",")}`, [dv.getUint32(off, true), dv.getUint32(off + 4, true)]);
  }
  return out;
}

function compare(ours, native) {
  const a = aveStd(ours);
  const b = aveStd(native);
  let values = 0;
  let exact = 0;
  let maxUlp = 0;
  let nonFinite = 0;
  for (const [key, want] of b) {
    const got = a.get(key);
    if (!got) throw new Error(`node ${key} missing in ours`);
    for (let k = 0; k < 2; k++) {
      values++;
      const u = ulps(got[k], want[k]);
      if (u === 0) exact++;
      else if (u === Infinity) nonFinite++;
      else maxUlp = Math.max(maxUlp, u);
    }
  }
  return { values, exact, maxUlp, nonFinite };
}

/** half_edge: rebuild from the sidecar (as test/tools/half-native.test.ts) and compare with its native stats. */
function halfEdge() {
  const sc = JSON.parse(readFileSync(F("half/half_edge.sidecar.json"), "utf8"));
  const background = fromBits(sc.grid.backgroundInput);
  const leaves = new Map();
  for (const v of sc.voxels) {
    const origin = [v.ijk[0] & ~7, v.ijk[1] & ~7, v.ijk[2] & ~7];
    const key = origin.join(",");
    let leaf = leaves.get(key);
    if (!leaf) leaves.set(key, (leaf = { origin, values: new Float32Array(512).fill(background), valueMask: new Uint32Array(16) }));
    const n = ((v.ijk[0] & 7) << 6) | ((v.ijk[1] & 7) << 3) | (v.ijk[2] & 7);
    leaf.values[n] = fromBits(v.input);
    leaf.valueMask[n >>> 5] |= 1 << (n & 31);
  }
  const image = vdbBuildFromLeavesDetailed(leaves.values(), VDB_HALF_LEAF_CODEC, { background, gridName: "half_edge", gridClass: "Unknown" }).image;
  const got = aveStd(image);
  const want = new Map();
  for (const l of sc.leaves) want.set(`0:${l.origin.join(",")}`, [l.stats.ave >>> 0, l.stats.stddev >>> 0]);
  for (const l of sc.lowers) want.set(`1:${l.origin.join(",")}`, [l.stats.ave >>> 0, l.stats.stddev >>> 0]);
  for (const l of sc.uppers) want.set(`2:${l.origin.join(",")}`, [l.stats.ave >>> 0, l.stats.stddev >>> 0]);
  want.set("3:root", [sc.native.root.ave >>> 0, sc.native.root.stddev >>> 0]);
  let values = 0, exact = 0, maxUlp = 0, nonFinite = 0;
  for (const [key, w] of want) {
    const g = got.get(key);
    for (let k = 0; k < 2; k++) {
      values++;
      const u = ulps(g[k], w[k]);
      if (u === 0) exact++;
      else if (u === Infinity) nonFinite++;
      else maxUlp = Math.max(maxUlp, u);
    }
  }
  return { values, exact, maxUlp, nonFinite };
}

const cases = [["half_edge (sidecar, v14 Half)", halfEdge]];
for (const name of ["sphere_fog", "torus_fog", "box_fog"]) {
  if (existsSync(F(`primitives/${name}_float.nvdb`)) && existsSync(F(`primitives/${name}_half.nvdb`))) {
    cases.push([`${name}: quantize half vs v14 Half`, () => compare(vdbQuantize(load(F(`primitives/${name}_float.nvdb`)), "half"), load(F(`primitives/${name}_half.nvdb`)))]);
  }
  if (existsSync(F(`primitives/${name}_float.nvdb`))) {
    const native = load(F(`primitives/${name}_float.nvdb`));
    cases.push([`${name}: float re-layout vs a532 Float`, () => compare(vdbQuantize(native, "float"), native)]);
  }
}
for (const sample of ["cube", "sphere", "smoke", "utahteapot"]) {
  const src = F(`parity/${sample}.normalized.vdb`);
  const nat = F(`parity/${sample}.native.float.nvdb`);
  if (!existsSync(src) || !existsSync(nat)) continue;
  cases.push([`${sample}: buildFromVdb vs a532 Float`, () => {
    const parsed = parseVdb(ab(readFileSync(src)));
    const g = parsed.grids.find((x) => x.gridType === "Tree_float_5_4_3") ?? parsed.grids[0];
    return compare(buildFromVdb(g), load(nat));
  }]);
}

console.log("ave/stddev f32 values vs native: exact / total, max ulp (finite), non-finite mismatches");
for (const [label, run] of cases) {
  const row = [];
  for (const mode of ["sums", "welford"]) {
    vdbSetStatsAccumulation(mode);
    const r = run();
    row.push(`${mode}: ${r.exact}/${r.values} exact, max ${r.maxUlp} ulp, ${r.nonFinite} non-finite`);
  }
  vdbSetStatsAccumulation("welford");
  console.log(`${label.padEnd(44)} ${row.join(" | ")}`);
}
