#!/usr/bin/env node
// pnpm parity:native — native OpenVDB / NanoVDB v14.0.0 against our TypeScript tools (K7.7).
//
//   pnpm build && pnpm parity:native [--native=auto|local|docker] [--only=<section>,…] [--update]
//
// The native tools come from the local cache scripts/native/build-native-v14.sh builds (no Docker,
// the default when present) or from the docker/fixture-bake image (--native=docker). Sections:
//
//   samples       nanovdb_convert (Float, Fp8) of the four pinned openvdb.org samples vs
//                 parseVdb -> buildFromVdb -> vdbQuantize: topology, every active value, every byte
//   generators    the K7.C generators rerun: the committed native-v14 fixtures are reproduced
//                 (.nvdb and probes byte for byte, .vdb outside the random UUID, image SHA-256s)
//   half          bake_half (createNanoGrid<…, Half>) reproduces the committed Half sidecars; the
//                 .nvdb files land in fixtures/ for half-native.test.ts
//   bunny         K4.10: vdb_probe on bunny_cloud.vdb (if present) reproduces the committed sidecar
//   points        K4.15: vdb_probe on waterfall_points.vdb (if present) reproduces the sidecar
//   vec3          K4.15: vdb_probe on explosion.vdb `v` reproduces the sidecar, and nanovdb_convert's
//                 Vec3f grid equals our buildFromVdb voxel for voxel and node stat for node stat
//   coarse-first  K4.17: our multi-grid preview .nvdb (codec NONE and ZIP) through nanovdb_print,
//                 nanovdb_validate, nvdb_probe (io::readGrid by index and by name), nanovdb_convert to
//                 .vdb (read back by parseVdb); the procedural file reproduces the sidecar, smoke.vdb
//                 (if present) is reported
//   tests         vitest on the suites the sidecars and fixtures feed
//
// --update rewrites the committed sidecars from the native output instead of comparing.
// Report: fixtures/parity/report.json. Exit code 1 when a section fails.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, readFileSync, rmSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { ROOT, mustRun, nativeTools } from "./lib/native-tools.mjs";
import { createParityAccumulator, overallPassed } from "./lib/parity-report.mjs";
import {
  activeCoordinates,
  analyzeNanoVdbTopology,
  classifyNanoVdbByteDifferences,
  compareCoordinate,
  compareNanoVdbTopology,
} from "./lib/nanovdb-topology.mjs";
import { VDB_SAMPLES, VDB_SAMPLES_COMMIT } from "./lib/vdb-samples.mjs";

const T = await import("../packages/vdb/dist/tools/index.js");
const { readLeafQuantum } = await import("../packages/vdb/dist/tools/cpu/read-value.js");
const { vdbReadValueF32Cpu } = await import("../packages/vdb/dist/tools/cpu/march.js");
const B = await import("../packages/vdb/dist/tools/nanovdb/bytes.js");
const { VEC3F_LAYOUT } = B;
const COARSE = await import("../packages/vdb/test/tools/fixtures/native-probes/coarse-first-source.mjs");

const FIXTURES = path.join(ROOT, "fixtures");
const SAMPLES_DIR = path.join(FIXTURES, "vdb-samples");
const OUTPUT = path.join(FIXTURES, "parity");
const NATIVE_V14 = path.join(ROOT, "packages/vdb/test/tools/fixtures/native-v14");
const PROBES = path.join(ROOT, "packages/vdb/test/tools/fixtures/native-probes");

const argv = process.argv.slice(2);
const opt = (name) => argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1];
const UPDATE = argv.includes("--update");
const ALL = ["samples", "generators", "half", "bunny", "points", "vec3", "coarse-first", "tests"];
const ONLY = opt("only")?.split(",") ?? ALL;
for (const s of ONLY) if (!ALL.includes(s)) throw new Error(`unknown section ${s}; sections: ${ALL.join(", ")}`);

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const toBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const imageBytes = (img) => new Uint8Array(img.buffer, img.byteOffset, img.byteLength);
const log = (...a) => console.error(...a);

async function loadGridImages(file) {
  const f = T.NvdbFile.fromArrayBuffer(toBuffer(await readFile(file)));
  return f.grids.map((_, i) => f.gridImage(i));
}

/** Compares a fresh native JSON with the committed sidecar (or writes it with --update). */
async function sidecar(name, freshPath) {
  const committed = path.join(PROBES, name);
  const fresh = await readFile(freshPath, "utf8");
  if (UPDATE || !existsSync(committed)) {
    await writeFile(committed, fresh);
    return { sidecar: name, written: true, passed: true };
  }
  const same = JSON.stringify(JSON.parse(fresh)) === JSON.stringify(JSON.parse(await readFile(committed, "utf8")));
  return { sidecar: name, matchesCommitted: same, passed: same };
}

//* samples ====================================================================

function addComparison(accumulator, coordinate, nativeImage, typescriptImage, numericBound) {
  const native = T.vdbReadValueCpu(nativeImage, coordinate);
  const typescript = T.vdbReadValueCpu(typescriptImage, coordinate);
  accumulator.add({ coordinate, native, typescript, numericBound });
}

function fp8NumericBound(coordinate, nativeImage, typescriptImage) {
  const native = T.vdbReadValueCpu(nativeImage, coordinate);
  const typescript = T.vdbReadValueCpu(typescriptImage, coordinate);
  const float32Roundoff = Math.max(Math.abs(native.value), Math.abs(typescript.value), 1) * 2 ** -22;
  return (readLeafQuantum(nativeImage, coordinate) + readLeafQuantum(typescriptImage, coordinate)) / 2 + float32Roundoff;
}

function compareActiveUnion(accumulator, mode, nativeImage, typescriptImage, nativeTopology, typescriptTopology) {
  const nativeIterator = activeCoordinates(nativeTopology);
  const typescriptIterator = activeCoordinates(typescriptTopology);
  let n = nativeIterator.next();
  let t = typescriptIterator.next();
  let unionCoordinates = 0;
  let nativeOnly = 0;
  let typescriptOnly = 0;
  while (!n.done || !t.done) {
    let coordinate;
    if (n.done) {
      coordinate = t.value;
      typescriptOnly++;
      t = typescriptIterator.next();
    } else if (t.done) {
      coordinate = n.value;
      nativeOnly++;
      n = nativeIterator.next();
    } else {
      const ordering = compareCoordinate(n.value, t.value);
      coordinate = ordering <= 0 ? n.value : t.value;
      if (ordering === 0) {
        n = nativeIterator.next();
        t = typescriptIterator.next();
      } else if (ordering < 0) {
        nativeOnly++;
        n = nativeIterator.next();
      } else {
        typescriptOnly++;
        t = typescriptIterator.next();
      }
    }
    unionCoordinates++;
    addComparison(accumulator, coordinate, nativeImage, typescriptImage, mode === "float" ? 0 : fp8NumericBound(coordinate, nativeImage, typescriptImage));
  }
  return {
    nativeActiveCoordinates: nativeTopology.activeVoxelCount,
    typescriptActiveCoordinates: typescriptTopology.activeVoxelCount,
    unionCoordinates,
    nativeOnly,
    typescriptOnly,
  };
}

function strictReport(sample, mode, nativeImage, typescriptImage) {
  const nativeTopology = analyzeNanoVdbTopology(nativeImage);
  const typescriptTopology = analyzeNanoVdbTopology(typescriptImage);
  const topology = compareNanoVdbTopology(nativeTopology, typescriptTopology);
  const accumulator = createParityAccumulator(sample, mode, []);
  const activeUnion = compareActiveUnion(accumulator, mode, nativeImage, typescriptImage, nativeTopology, typescriptTopology);
  const report = accumulator.finish();
  const byteEvidence = classifyNanoVdbByteDifferences(nativeImage, typescriptImage, nativeTopology, typescriptTopology, {
    mode,
    semanticPassed: report.passed,
    topologyComparison: topology,
  });
  report.topology = { ...topology, nodeCounts: nativeTopology.nodeCounts, tileCounts: nativeTopology.tileCounts };
  report.activeUnion = activeUnion;
  report.byteDifferences = byteEvidence.classifications;
  report.byteDifferenceCount = byteEvidence.differingBytes;
  if (byteEvidence.differingOffsetSha256) report.differingOffsetSha256 = byteEvidence.differingOffsetSha256;
  report.unclassifiedByteDifferences = byteEvidence.unclassifiedByteDifferences;
  report.classifiedSemanticDifferences = [];
  report.passed =
    report.passed &&
    topology.passed &&
    activeUnion.nativeOnly === 0 &&
    activeUnion.typescriptOnly === 0 &&
    byteEvidence.unclassifiedByteDifferences.length === 0;
  return report;
}

async function sectionSamples(tools) {
  const reports = [];
  const sources = {};
  for (const sampleName of Object.keys(VDB_SAMPLES)) {
    const sample = path.basename(sampleName, ".vdb");
    const expected = VDB_SAMPLES[sampleName];
    const original = await readFile(path.join(SAMPLES_DIR, sampleName));
    const digest = sha256(original);
    if (original.byteLength !== expected.bytes || digest !== expected.sha256)
      throw new Error(`${sampleName} does not match the pinned fixture revision: ${original.byteLength} bytes / sha256 ${digest}`);
    sources[sample] = { file: `fixtures/vdb-samples/${sampleName}`, bytes: original.byteLength, sha256: digest, commit: VDB_SAMPLES_COMMIT };
    log(`  • ${sample}`);
    const normalized = path.join(OUTPUT, `${sample}.normalized.vdb`);
    mustRun(tools, "normalize_vdb", [path.join(SAMPLES_DIR, sampleName), normalized]);
    const parsed = T.parseVdb(toBuffer(await readFile(normalized)));
    const grid = parsed.grids.find((g) => g.gridType === "Tree_float_5_4_3") ?? parsed.grids[0];
    const ours = { float: T.buildFromVdb(grid) };
    ours.fp8 = T.vdbQuantize(ours.float, "fp8");
    for (const mode of ["float", "fp8"]) {
      const out = path.join(OUTPUT, `${sample}.native.${mode}.nvdb`);
      rmSync(out, { force: true });
      mustRun(tools, "nanovdb_convert", [...(mode === "fp8" ? ["--fp8"] : []), normalized, out]);
      await writeFile(path.join(OUTPUT, `${sample}.typescript.${mode}.nvdb`), new Uint8Array(T.writeNvdb([ours[mode]])));
      reports.push(strictReport(sample, mode, (await loadGridImages(out))[0], ours[mode]));
    }
  }
  return { sources, samples: reports, passed: overallPassed(reports) };
}

//* generators =================================================================

async function sectionGenerators(tools) {
  const out = path.join(OUTPUT, "native-v14");
  await mkdir(out, { recursive: true });
  mustRun(tools, "gen_nvdb_blosc", [out]);
  mustRun(tools, "gen_vdb_half", [out]);
  const checks = [];
  for (const f of ["sphere_fog_blosc.nvdb", "two_grids_blosc.nvdb", "nvdb-probes.json", "vdb-probes.json"]) {
    const same = Buffer.compare(await readFile(path.join(out, f)), await readFile(path.join(NATIVE_V14, f))) === 0;
    checks.push({ file: f, identical: same, passed: same });
  }
  const images = JSON.parse(await readFile(path.join(NATIVE_V14, "nvdb-images.json"), "utf8"));
  for (const [name, want] of Object.entries(images)) {
    const bytes = await readFile(path.join(out, `${name}.image`));
    const normalized = Buffer.from(bytes);
    normalized.fill(0, 8, 32);
    const ok = bytes.byteLength === want.bytes && sha256(bytes) === want.sha256 && sha256(normalized) === want.sha256HeaderNormalized;
    checks.push({ file: `${name}.image`, sha256: sha256(bytes), passed: ok });
  }
  // OpenVDB .vdb: the 36-byte ASCII UUID at bytes 21..56 is random per write, and each grid's
  // `file_mem_bytes` metadata (int64 in-memory footprint) depends on the platform (Linux/gcc: +48 B).
  const normalizeVdb = (buf) => {
    buf.fill(0, 21, 57);
    const key = Buffer.from("file_mem_bytes");
    for (let at = buf.indexOf(key); at >= 0; at = buf.indexOf(key, at + 1)) {
      const value = at + key.length + 4 + 5 + 4; // type name length, "int64", value size
      buf.fill(0, value, value + 8);
    }
    return buf;
  };
  for (const codec of ["none", "zip", "blosc"]) {
    const a = normalizeVdb(Buffer.from(await readFile(path.join(out, `half_${codec}.vdb`))));
    const b = normalizeVdb(Buffer.from(await readFile(path.join(NATIVE_V14, `half_${codec}.vdb`))));
    const same = Buffer.compare(a, b) === 0;
    checks.push({ file: `half_${codec}.vdb`, identicalOutsideUuidAndMemBytes: same, passed: same });
  }
  checks.push({ file: "warp_sphere_blosclz.nvdb", skipped: "needs NVIDIA Warp (python -I generators/warp_blosclz.py), not in the native tools", passed: true });
  return { checks, passed: checks.every((c) => c.passed) };
}

//* half =======================================================================

async function sectionHalf(tools) {
  const names = ["sphere_fog", "torus_fog", "box_fog"];
  const missing = names.filter((n) => !existsSync(path.join(FIXTURES, "primitives", `${n}_float.nvdb`)));
  if (missing.length) return { skipped: `fixtures/primitives/${missing[0]}_float.nvdb missing (pnpm fixtures:bake)`, passed: true };
  const prim = path.join(OUTPUT, "half", "primitives");
  const edge = path.join(OUTPUT, "half", "half");
  await mkdir(prim, { recursive: true });
  for (const n of names) copyFileSync(path.join(FIXTURES, "primitives", `${n}_float.nvdb`), path.join(prim, `${n}_float.nvdb`));
  mustRun(tools, "bake_half", [prim, edge]);
  const checks = [];
  const pairs = [
    ...names.map((n) => [path.join(prim, `${n}_half`), path.join(FIXTURES, "primitives", `${n}_half`)]),
    [path.join(edge, "half_edge"), path.join(FIXTURES, "half", "half_edge")],
  ];
  for (const [fresh, fixture] of pairs) {
    const same = Buffer.compare(await readFile(`${fresh}.sidecar.json`), await readFile(`${fixture}.sidecar.json`)) === 0;
    checks.push({ sidecar: path.relative(ROOT, `${fixture}.sidecar.json`), identical: same, passed: same });
    // The grids themselves are git-ignored fixtures (pnpm fixtures:bake-half): install them for the tests.
    if (same) {
      await mkdir(path.dirname(fixture), { recursive: true });
      copyFileSync(`${fresh}.nvdb`, `${fixture}.nvdb`);
    }
  }
  return { checks, passed: checks.every((c) => c.passed) };
}

//* bunny / points / vec3 sidecars ===============================================

async function sectionBunny(tools) {
  const file = path.join(SAMPLES_DIR, "bunny_cloud.vdb");
  if (!existsSync(file)) return { skipped: "fixtures/vdb-samples/bunny_cloud.vdb missing (not fetched: ATTRIBUTION.md)", passed: true };
  const out = path.join(OUTPUT, "bunny_cloud.density.json");
  mustRun(tools, "vdb_probe", ["float", file, "density", out]);
  const native = JSON.parse(await readFile(out, "utf8"));
  return { ...(await sidecar("bunny_cloud.density.json", out)), nanoProbesAgree: native.nano.probesAgree };
}

async function sectionPoints(tools) {
  const file = path.join(SAMPLES_DIR, "waterfall_points.vdb");
  if (!existsSync(file)) return { skipped: "fixtures/vdb-samples/waterfall_points.vdb missing", passed: true };
  const out = path.join(OUTPUT, "waterfall_points.json");
  mustRun(tools, "vdb_probe", ["points", file, out, "128"]);
  return sidecar("waterfall_points.json", out);
}

/** Every node of an image with the given layout (treeNodes covers scalar grids only): level, origin, address. */
function walkNodes(image, L) {
  const v = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const bit = (base, n) => ((v.getUint32(base + 4 * (n >>> 5), true) >>> (n & 31)) & 1) === 1;
  const root = B.GRID_SIZE + Number(v.getBigUint64(B.GRID_SIZE + B.TREE_OFF_NODE_OFFSET_ROOT, true));
  const out = [{ level: 3, origin: [0, 0, 0], address: root }];
  for (let i = 0; i < v.getUint32(root + B.ROOT_OFF_TABLE_SIZE, true); i++) {
    const t = root + L.rootSize + i * L.rootTileSize;
    const child = Number(v.getBigInt64(t + B.ROOT_TILE_OFF_CHILD, true));
    if (child === 0) continue;
    const key = v.getBigUint64(t + B.ROOT_TILE_OFF_KEY, true);
    const uo = [Number((key >> 42n) & 0x1fffffn) << 12, Number((key >> 21n) & 0x1fffffn) << 12, Number(key & 0x1fffffn) << 12];
    const upper = root + child;
    out.push({ level: 2, origin: uo, address: upper });
    for (let un = 0; un < 32768; un++) {
      if (!bit(upper + B.UPPER_OFF_CHILD_MASK, un)) continue;
      const lo = [uo[0] + (un >> 10) * 128, uo[1] + ((un >> 5) & 31) * 128, uo[2] + (un & 31) * 128];
      const lower = upper + Number(v.getBigInt64(upper + L.upperOffTable + L.tableStride * un, true));
      out.push({ level: 1, origin: lo, address: lower });
      for (let ln = 0; ln < 4096; ln++) {
        if (!bit(lower + B.LOWER_OFF_CHILD_MASK, ln)) continue;
        const leaf = lower + Number(v.getBigInt64(lower + L.lowerOffTable + L.tableStride * ln, true));
        out.push({ level: 0, origin: [lo[0] + (ln >> 8) * 8, lo[1] + ((ln >> 4) & 15) * 8, lo[2] + (ln & 15) * 8], address: leaf });
      }
    }
  }
  return out;
}

/** Native nanovdb_convert Vec3f vs our buildFromVdb: every voxel and node stat of every node. */
function compareVec3(native, ours) {
  const key = (n) => `${n.level}:${n.origin.join(",")}`;
  const nativeNodes = new Map(walkNodes(native, VEC3F_LAYOUT).map((n) => [key(n), n]));
  const ourNodes = new Map(walkNodes(ours, VEC3F_LAYOUT).map((n) => [key(n), n]));
  const nv = new DataView(native.buffer, native.byteOffset, native.byteLength);
  const ov = new DataView(ours.buffer, ours.byteOffset, ours.byteLength);
  const L = VEC3F_LAYOUT;
  const statOffsets = [
    [L.leafOffMin, L.leafOffMax, L.leafOffAve, L.leafOffStdDev],
    [L.lowerOffMin, L.lowerOffMax, L.lowerOffAve, L.lowerOffStdDev],
    [L.upperOffMin, L.upperOffMax, L.upperOffAve, L.upperOffStdDev],
    [L.rootOffMin, L.rootOffMax, L.rootOffAve, L.rootOffStdDev],
  ];
  let voxels = 0;
  let valueMismatches = 0;
  let minMaxMismatches = 0;
  let aveUlps = 0;
  let aveDiffering = 0;
  let worst = null;
  const statRanges = [];
  const ulps = (a, b) => {
    const f = new Float32Array([a, b]);
    const u = new Int32Array(f.buffer);
    return Math.abs(u[0] - u[1]);
  };
  const missing = [...nativeNodes.keys()].filter((k) => !ourNodes.has(k)).length + [...ourNodes.keys()].filter((k) => !nativeNodes.has(k)).length;
  for (const [k, n] of nativeNodes) {
    const o = ourNodes.get(k);
    if (!o) continue;
    const [mn, mx, av, sd] = statOffsets[n.level];
    for (let c = 0; c < 3; c++) {
      if (nv.getFloat32(n.address + mn + 4 * c, true) !== ov.getFloat32(o.address + mn + 4 * c, true)) minMaxMismatches++;
      if (nv.getFloat32(n.address + mx + 4 * c, true) !== ov.getFloat32(o.address + mx + 4 * c, true)) minMaxMismatches++;
    }
    for (const off of [av, sd]) {
      const u = ulps(nv.getFloat32(n.address + off, true), ov.getFloat32(o.address + off, true));
      if (u > 0) aveDiffering++;
      if (u > aveUlps) {
        aveUlps = u;
        worst = { level: n.level, origin: n.origin, native: nv.getFloat32(n.address + off, true), ours: ov.getFloat32(o.address + off, true) };
      }
    }
    statRanges.push([n.address + mn, n.address + sd + 4]);
    if (n.level !== 0) continue;
    for (let i = 0; i < 512; i++) {
      const ijk = [n.origin[0] + (i >> 6), n.origin[1] + ((i >> 3) & 7), n.origin[2] + (i & 7)];
      const a = T.vdbReadValueVec3Cpu(native, ijk);
      const b = T.vdbReadValueVec3Cpu(ours, ijk);
      voxels++;
      if (a.active !== b.active || a.value.some((v, c) => !Object.is(v, b.value[c]))) valueMismatches++;
    }
  }
  const nb = imageBytes(native);
  const ob = imageBytes(ours);
  // Differing bytes: GridData checksum / version (bytes 8..19), node stats (ave / stddev only, as min / max
  // compare equal above), anything else.
  statRanges.sort((a, b) => a[0] - b[0]);
  const inStats = (i) => {
    let lo = 0;
    let hi = statRanges.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (i < statRanges[mid][0]) hi = mid - 1;
      else if (i >= statRanges[mid][1]) lo = mid + 1;
      else return true;
    }
    return false;
  };
  const regions = { checksumAndVersion: 0, nodeStats: 0, other: 0 };
  if (nb.length === ob.length)
    for (let i = 0; i < nb.length; i++)
      if (nb[i] !== ob[i]) regions[i >= 8 && i < 20 ? "checksumAndVersion" : inStats(i) ? "nodeStats" : "other"]++;
  return {
    nodes: nativeNodes.size,
    nodesMissingOnEitherSide: missing,
    voxelsCompared: voxels,
    valueMismatches,
    minMaxMismatches,
    aveStddevDiffering: aveDiffering,
    maxAveStddevUlps: aveUlps,
    worstAveStddev: worst,
    sameLength: nb.length === ob.length,
    differingBytes: regions,
    // ave / stddev: native GridStats accumulates f64 Welford, we f64 sums (V174), and native itself
    // differs by build: <= 8 ulp from the macOS cache build, <= 32 from the Linux image. Bytes outside
    // the stats, the checksum and the version must be equal.
    passed: missing === 0 && valueMismatches === 0 && minMaxMismatches === 0 && aveUlps <= 64 && regions.other === 0,
  };
}

async function sectionVec3(tools) {
  const file = path.join(SAMPLES_DIR, "explosion.vdb");
  if (!existsSync(file)) return { skipped: "fixtures/vdb-samples/explosion.vdb missing", passed: true };
  const out = path.join(OUTPUT, "explosion.v.json");
  mustRun(tools, "vdb_probe", ["vec3", file, "v", out]);
  const probe = await sidecar("explosion.v.json", out);
  const nvdb = path.join(OUTPUT, "explosion.v.native.nvdb");
  rmSync(nvdb, { force: true });
  mustRun(tools, "nanovdb_convert", ["-g", "v", file, nvdb]);
  // NvdbFile loads scalar grids only: take the codec-NONE single-grid segment apart by hand
  // (FileHeader 16 B, FileMetaData 176 B with nameSize at 136, the name, then the grid).
  const raw = await readFile(nvdb);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (dv.getUint16(12, true) !== 1 || dv.getUint16(14, true) !== 0) throw new Error("expected one grid, codec NONE");
  const start = 16 + 176 + dv.getUint32(16 + 136, true);
  const native = new Uint32Array(toBuffer(raw.subarray(start, start + Number(dv.getBigUint64(16, true)))));
  const grid = T.parseVdb(toBuffer(await readFile(file))).grids.find((g) => g.name === "v");
  const convert = compareVec3(native, T.buildFromVdb(grid));
  return { probe, nanovdbConvert: convert, passed: probe.passed && convert.passed };
}

//* coarse-first ===============================================================

function validateResults(text) {
  const out = {};
  let reason = null;
  for (const line of text.split("\n")) {
    const fail = line.match(/^Validation failed: (.*)$/);
    if (fail) reason = fail[1];
    const m = line.match(/^Grid named "(.*)": (passed|failed)$/);
    if (m) {
      out[m[1]] = m[2] === "passed" ? "passed" : `failed: ${reason}`;
      reason = null;
    }
  }
  return out;
}

/** Native reading of one coarse-first file, as JSON, plus our read-back checks. */
async function nativeCoarseFirst(tools, file, probes) {
  const print = mustRun(tools, "nanovdb_print", ["-l", file]).stdout;
  // nanovdb_validate writes "Validation failed: …" to stderr and "Grid named …" to stdout: merge in order.
  const validate = validateResults(tools.run("nanovdb_validate", ["-v", file], { merged: true }).stdout);
  const probeJson = file.replace(/\.nvdb$/, ".probe.json");
  mustRun(tools, "nvdb_probe", [file, probeJson, ...probes.map((c) => c.join(","))]);
  const probe = JSON.parse(await readFile(probeJson, "utf8"));
  const vdb = file.replace(/\.nvdb$/, ".native.vdb");
  rmSync(vdb, { force: true });
  mustRun(tools, "nanovdb_convert", [file, vdb]);

  // Ours: the same file read by NvdbFile; native's .vdb read by parseVdb.
  const bytes = await readFile(file);
  const nf = T.NvdbFile.fromArrayBuffer(toBuffer(bytes));
  const converted = T.parseVdb(toBuffer(await readFile(vdb)));
  const mismatches = [];
  nf.grids.forEach((g, i) => {
    const image = nf.gridImage(i);
    const nativeGrid = probe.byName[i];
    if (nativeGrid.name !== g.name) mismatches.push(`grid ${i}: native ${nativeGrid.name}, ours ${g.name}`);
    for (const p of nativeGrid.probes ?? []) {
      const got = T.vdbReadValueCpu(image, p.ijk);
      // Quantized grids: nvdb_probe (built -ffp-contract=off) decodes code * quantum + minimum unfused in f32
      // (vdbReadValueF32Cpu, the WGSL twin); nanoToOpenVDB in the cmake build fuses it (fround of the f64 value).
      const want = Math.fround(p.value);
      const okValue = Math.fround(got.value) === want || vdbReadValueF32Cpu(image, p.ijk) === want;
      if (!okValue || got.active !== p.active) mismatches.push(`${g.name} ${p.ijk}: native ${p.value}/${p.active}, ours ${got.value}/${got.active}`);
    }
    const vg = converted.grids.find((x) => x.name === g.name);
    if (!vg) {
      mismatches.push(`${g.name} missing from nanovdb_convert's .vdb`);
      return;
    }
    // Every active voxel of the converted grid reads the same in our image (and the counts agree).
    let active = 0;
    for (const leaf of vg.iterLeaves()) {
      for (let n = 0; n < 512; n++) {
        if (((leaf.valueMask[n >>> 5] >>> (n & 31)) & 1) === 0) continue;
        active++;
        const ijk = [leaf.origin[0] + (n >> 6), leaf.origin[1] + ((n >> 3) & 7), leaf.origin[2] + (n & 7)];
        const got = T.vdbReadValueCpu(image, ijk);
        if (!got.active || Math.fround(got.value) !== leaf.values[n]) mismatches.push(`${g.name} ${ijk}: .vdb ${leaf.values[n]}, ours ${got.value}`);
      }
    }
    if (BigInt(active) !== vg.activeVoxelCount || active !== Number(nativeGrid.activeVoxelCount)) mismatches.push(`${g.name}: active counts differ`);
  });
  // io::readGrid(file, name) finds every grid, io::hasGrid sees every name (nameKey = io::stringHash).
  const nameLookupOk =
    probe.byName.every((g, i) => g.name === probe.meta[i].name) &&
    probe.meta.every((m) => m.hasGrid && m.nameKey === m.stringHash) &&
    probe.all.gridCount === nf.grids.length;
  // io::readGrid(file, n): reported, not gated. v14 reads grid n >= 1 of a multi-grid segment from grid 0's
  // data (it never seeks past the grids before it): grid 0 again (codec NONE) or a decode error (ZIP, BLOSC).
  const byIndex = probe.byIndex.map((g) => g.name ?? `error: ${g.error}`);
  return {
    summary: {
      bytes: bytes.byteLength,
      sha256: sha256(bytes),
      // Each read once: by name with the probes (the readers that work), the rest by grid name only.
      probe: {
        nanovdb: probe.nanovdb,
        meta: probe.meta,
        grid0: probe.grid0.name,
        byIndex: probe.byIndex.map((g) => g.name ?? `error: ${g.error}`),
        byName: probe.byName,
        all: probe.all.grids.map((g) => g.name),
      },
      validate,
      nanovdbConvertVdbGridOrder: converted.grids.map((g) => g.name),
    },
    print,
    checks: { mismatches: mismatches.slice(0, 20), mismatchCount: mismatches.length, nameLookupOk, grid0: probe.grid0.name, byIndex },
    passed: mismatches.length === 0 && nameLookupOk,
  };
}

async function sectionCoarseFirst(tools) {
  const dir = path.join(OUTPUT, "coarse-first");
  await mkdir(dir, { recursive: true });
  const files = {};
  const results = {};
  for (const codec of ["none", "zip"]) {
    const file = path.join(dir, `procedural-${codec}.nvdb`);
    await writeFile(file, COARSE.coarseFirstFile(T, codec));
    const r = await nativeCoarseFirst(tools, file, COARSE.COARSE_FIRST_PROBES);
    files[codec] = r.summary;
    results[`procedural-${codec}`] = { ...r.checks, validate: r.summary.validate, passed: r.passed };
    log(r.print);
  }
  const fresh = path.join(dir, "coarse-first.json");
  await writeFile(
    fresh,
    `${JSON.stringify({ nanovdb: files.none.probe.nanovdb, source: "coarse-first-source.mjs (coarseFirstFile)", files }, null, 2)}\n`,
  );
  const side = await sidecar("coarse-first.json", fresh);
  // The real use case, reported only (openvdb.org smoke, CC BY-SA, not committed).
  const smoke = path.join(SAMPLES_DIR, "smoke.vdb");
  if (existsSync(smoke)) {
    for (const codec of ["none", "zip"]) {
      const file = path.join(dir, `smoke-${codec}.nvdb`);
      await writeFile(file, T.vdbBakeCoarseFirst(T.parseVdb(toBuffer(await readFile(smoke))), { codec }));
      const r = await nativeCoarseFirst(tools, file, [[50, 100, 50], [60, 120, 60], [5, 10, 5], [10, 20, 10]]);
      results[`smoke-${codec}`] = {
        ...r.checks,
        validate: r.summary.validate,
        grid0: r.summary.probe.grid0,
        nanovdbConvertVdbGridOrder: r.summary.nanovdbConvertVdbGridOrder,
        passed: r.passed,
      };
      log(r.print);
    }
  }
  return { sidecar: side, results, passed: side.passed && Object.values(results).every((r) => r.passed) };
}

//* tests ======================================================================

function sectionTests() {
  const files = [
    "packages/vdb/test/tools/native-probes.test.ts",
    "packages/vdb/test/tools/native-v14.test.ts",
    "packages/vdb/test/tools/nvdb-blosc.test.ts",
    "packages/vdb/test/tools/half-native.test.ts",
    "packages/vdb/test/tools/breadth-first.test.ts",
  ];
  const r = spawnSync("npx", ["vitest", "run", "--testTimeout", "120000", ...files], { cwd: ROOT, stdio: "inherit" });
  return { files, passed: r.status === 0 };
}

//* Run ========================================================================

await mkdir(OUTPUT, { recursive: true });
const tools = nativeTools({ mode: opt("native") });
log(`• native tools: ${tools.mode}${tools.bin ? ` (${tools.bin})` : ""}`);
const sections = {
  samples: sectionSamples,
  generators: sectionGenerators,
  half: sectionHalf,
  bunny: sectionBunny,
  points: sectionPoints,
  vec3: sectionVec3,
  "coarse-first": sectionCoarseFirst,
  tests: sectionTests,
};
const result = { generatedAt: new Date().toISOString(), command: "pnpm parity:native", native: tools.mode, openvdb: "v14.0.0", sections: {} };
for (const name of ONLY) {
  log(`• ${name}`);
  const r = await sections[name](tools);
  result.sections[name] = r;
  log(`  ${r.passed ? "✓" : "✗"} ${name}${r.skipped ? ` (skipped: ${r.skipped})` : ""}`);
}
result.passed = Object.values(result.sections).every((s) => s.passed);
const reportPath = path.join(OUTPUT, "report.json");
await writeFile(reportPath, `${JSON.stringify(result, null, 2)}\n`);
log(`✓ report: ${path.relative(ROOT, reportPath)} (${result.passed ? "PASS" : "FAIL"})`);
if (!result.passed) process.exitCode = 1;
