// P1.4 counters (no timings): per ray, for the brick / global / nodeMax majorant walks, the segments, the accessor
// lookups the walk costs, and the majorant's optical depth against the density's, which give the expected tentative
// and null collisions of delta tracking at any sigma (tentative = sigma * ∫bound dt, real = sigma * ∫density dt). Input
// for P1.10's A/B (counters first, timing in the pathtracer's quiet pass).
//
//   pnpm build && node --experimental-strip-types scripts/p1/majorant-counters.mjs [--rays=256] [--out=<file.json>]
//
// Grids: the nine primitives (support-aligned 8³ bricks), the pathtracer's vdb-smoke rebuilt from
// fixtures/vdb-samples/smoke.vdb (its 'pathtracer' alignment), and the WDAS quarter Fp8 grid when present. Rays are
// index-space chords through the support box (from outside towards a random inner point), the same for every mode.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { NvdbFile, indexBBox, majorantWalk, parseVdb, vdbBrickMajorant, vdbBuildFromDense, vdbSampleTrilinearCpu } from "../../packages/vdb/dist/tools/internal.js";
import { rng, smokeDense, toXMajor } from "../../examples/src/harness/p1-smoke.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const arg = (name, dflt) => process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ?? dflt;
const RAYS = Number(arg("rays", 256));
const OUT = arg("out", null);
const SAMPLES = 512;

function load(rel) {
  const buf = readFileSync(path.join(ROOT, rel));
  return NvdbFile.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)).gridImage(0);
}

function rays(image, count, seed) {
  const { min, max } = indexBBox(image);
  const r = rng(seed);
  const ext = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) + 4;
  return Array.from({ length: count }, () => {
    const target = [0, 1, 2].map((a) => min[a] - 1 + r() * (max[a] - min[a] + 2));
    let d = [r() - 0.5, r() - 0.5, r() - 0.5];
    const len = Math.hypot(...d);
    d = d.map((v) => Math.fround(v / len));
    return { o: target.map((v, a) => Math.fround(v - d[a] * ext)), d };
  });
}

function counters(name, image, brick, R) {
  const out = { grid: name, rays: R.length };
  const walks = {
    brick: (ray) => majorantWalk(image, brick, ray.o, ray.d, { outside: "zero", minima: true }),
    global: (ray) => majorantWalk(image, null, ray.o, ray.d, { outside: "zero", mode: "global" }),
    nodeMax: (ray) => majorantWalk(image, null, ray.o, ray.d, { outside: "zero", mode: "nodeMax" }),
  };
  // ∫density along each ray over the support chord (the global walk's one segment), stratified midpoints.
  let density = 0;
  const chords = R.map((ray) => walks.global(ray)[0] ?? null);
  R.forEach((ray, k) => {
    const c = chords[k];
    if (!c) return;
    const dt = (c.t1 - c.t0) / SAMPLES;
    for (let s = 0; s < SAMPLES; s++) {
      const t = c.t0 + (s + 0.5) * dt;
      density += vdbSampleTrilinearCpu(image, [0, 1, 2].map((a) => ray.o[a] + ray.d[a] * t), { outside: "zero" }) * dt;
    }
  });
  out.densityDepthPerRay = density / R.length;
  for (const [mode, walk] of Object.entries(walks)) {
    let segs = 0;
    let lookups = 0;
    let depth = 0;
    let minDepth = 0;
    for (const ray of R) {
      for (const s of walk(ray)) {
        segs++;
        lookups += s.lookups;
        depth += Number.isFinite(s.max) ? s.max * (s.t1 - s.t0) : 0;
        minDepth += s.min * (s.t1 - s.t0);
      }
    }
    out[mode] = {
      segmentsPerRay: segs / R.length,
      lookupsPerRay: lookups / R.length,
      majorantDepthPerRay: depth / R.length,
      // Expected share of tentative collisions that are null (delta tracking, any sigma).
      nullShare: depth > 0 ? 1 - density / depth : 0,
      ...(mode === "brick" ? { minimaDepthPerRay: minDepth / R.length } : {}),
    };
  }
  return out;
}

const report = [];
for (const shape of ["box", "sphere", "torus"]) {
  for (const type of ["float", "fp8", "fpn"]) {
    const rel = `fixtures/primitives/${shape}_fog_${type}.nvdb`;
    if (!existsSync(path.join(ROOT, rel))) continue;
    const image = load(rel);
    report.push(counters(`${shape}_fog_${type}`, image, vdbBrickMajorant(image, { minima: true, outside: "zero" }), rays(image, RAYS, 11)));
  }
}
if (existsSync(path.join(ROOT, "fixtures/vdb-samples/smoke.vdb"))) {
  const buf = readFileSync(path.join(ROOT, "fixtures/vdb-samples/smoke.vdb"));
  const dense = smokeDense(parseVdb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
  const image = vdbBuildFromDense(toXMajor(dense.density, dense.resolution), dense.resolution);
  report.push(counters("vdb-smoke", image, vdbBrickMajorant(image, { align: "pathtracer", resolution: dense.resolution, minima: true }), rays(image, RAYS, 13)));
}
if (existsSync(path.join(ROOT, "fixtures/wdas/wdas_cloud_quarter_fp8.nvdb"))) {
  const image = load("fixtures/wdas/wdas_cloud_quarter_fp8.nvdb");
  report.push(counters("wdas_cloud_quarter_fp8", image, vdbBrickMajorant(image, { minima: true, outside: "zero" }), rays(image, Math.min(RAYS, 64), 17)));
}

const f = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
console.log("grid | mode | segments/ray | lookups/ray | ∫bound/ray | ∫density/ray | null share");
for (const r of report) {
  for (const mode of ["brick", "global", "nodeMax"]) {
    const m = r[mode];
    console.log(`${r.grid} | ${mode} | ${f(m.segmentsPerRay)} | ${f(m.lookupsPerRay)} | ${f(m.majorantDepthPerRay)} | ${f(r.densityDepthPerRay)} | ${m.nullShare.toFixed(3)}`);
  }
}
if (OUT) writeFileSync(OUT, `${JSON.stringify(report, null, 2)}\n`);
