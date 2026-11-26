import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages, diffHeatmap } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K6.1 (design §7.1): the core frame kernels on raw WebGPU (examples/src/harness/k6-core.ts), each against its
// CPU twin (tools/frame) within 2 f16 ulp; every probe's mutation must fail; every pipeline variant compiles;
// a second identical update builds nothing; two loads are deterministic.
const PAGE_URL = "/src/harness/k6-core.html";

interface Stats {
  compared: number;
  covered: number;
  over: number;
  maxUlp: number;
  maxAbs: number;
  bound: string;
  firstBad?: unknown;
}
interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  stats?: Stats;
  renderStats?: Stats;
  gridStats?: Stats;
  transmittanceStats?: Stats;
  workStats?: Stats;
  mapStats?: Stats;
  bufferStats?: Stats;
  environmentGridStats?: Stats;
  mipStats?: Stats;
  environment?: { runs: { prefilter: number; blur: number; reduce: number }[]; count: number; mode: number; valid: number; directions: number[][]; colors: number[][] };
  frame?: { cascades: number; redraw: unknown };
  occluders?: { maps: number; transmittance: number; density: number; lighting: number; gather: number };
  builds?: { first: number; second: number };
  compile?: { variants: number; errors: string[] };
  receiverStats?: Stats;
  shadow?: { light: string; layers: number; shadowed: number; receiverShadowed: number; same: { drawn: number; waiting: number }; moved: { drawn: number; waiting: number }; cascades: number; enabled: boolean };
  reduceStats?: Stats;
  probeStats?: Stats;
  irradianceStats?: Stats;
  emission?: { state: { count: number; resolution: number; emitters: number }; first: { directions: number; probes: boolean }; second: { directions: number; probes: boolean }; hookErrors: string[]; outside: number };
  motion?: { valid: number; partial: number; offCentre: number };
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string, timeout = 300_000): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K6_CORE__?: { done: boolean } }).__VDB_K6_CORE__?.done === true, undefined, { timeout });
  return (await page.evaluate(() => (window as { __VDB_K6_CORE__?: unknown }).__VDB_K6_CORE__)) as Result;
}

const detail = (r: Result) => JSON.stringify(r, null, 2);

/**
 * K6.10a: the fewest non-zero ("covered") values a K6.2b / K6.3 / K6.4 / K6.5 probe's stats must have, about half the
 * smallest count SwiftShader gives for that key (docs/archive/handoffs/K6-10a.md), so an all-zero GPU and twin pair
 * (a kernel that writes nothing, a twin that reads nothing) fails instead of matching.
 */
function minCovered(key: string, query: string): number {
  const floors: Record<string, number> = {
    transmittanceStats: 8192, // every voxel: 16384 – 49152
    environmentGridStats: 2048, // 4096
    gridStats: 140, // 294 (sparse) – 1443
    mipStats: 48, // 98 – 312
    renderStats: 192, // 384
    receiverStats: 512, // 1024
    reduceStats: 2, // 2
    probeStats: 768, // 1536 – 3072
    irradianceStats: 192, // 382 – 388
    workStats: 4096, // 8192
    mapStats: 1024, // 2048
    bufferStats: 10, // 10 – 34
  };
  if (key !== "stats") return floors[key]!;
  if (query.includes("probe=resolve")) return 1900; // 3856 – 7280
  if (query.includes("probe=integrate")) return 560; // 1120
  if (query.includes("probe=shadow")) return 95; // 191 – 597
  if (query.includes("probe=emission")) return 16; // 32
  return 130; // gather and 'frame' froxels: 265 – 585
}

test("K6.1 every frame pipeline variant compiles", async ({ page }) => {
  // K6.5 added the shadow and emission kernels (with hooks): ~43 variants, minutes on SwiftShader under load
  test.setTimeout(900_000);
  const errors = collectErrors(page);
  const r = await run(page, "?probe=compile", 900_000);
  expect(r.error, detail(r)).toBeUndefined();
  expect(r.compile!.errors, detail(r)).toEqual([]);
  expect(r.compile!.variants).toBeGreaterThanOrEqual(15);
  expect(r.uncaptured).toEqual([]);
  expect(errors).toEqual([]);
});

test("K6.4 every spectral pipeline variant compiles (both alpha prefix formats with texture-formats-tier1)", async ({ page }) => {
  test.setTimeout(1_200_000);
  const errors = collectErrors(page);
  const r = await run(page, "?probe=compile&spectral=1&tier1=1", 1_200_000);
  expect(r.error, detail(r)).toBeUndefined();
  expect(r.compile!.errors, detail(r)).toEqual([]);
  expect(r.compile!.variants).toBeGreaterThanOrEqual(26);
  expect(r.uncaptured).toEqual([]);
  expect(errors).toEqual([]);
});

const PROBES: { name: string; query: string; minCovered: number; mutations: string[] }[] = [
  { name: "view chunks", query: "?probe=view", minCovered: 512, mutations: ["coverage", "mie", "octaves", "two-sided"] },
  // K6.10a: `wgsl-*` mutations patch the kernel's WGSL (k6-core.ts WGSL_PATCHES), the others flip a twin term
  { name: "gather (one group)", query: "?probe=gather&groups=1", minCovered: 300, mutations: ["mix", "phase", "extinction", "fade", "wgsl-gather-extinction", "wgsl-gather-phase"] },
  { name: "gather (two groups)", query: "?probe=gather&groups=2", minCovered: 300, mutations: ["mix"] },
  { name: "gather (two groups, no mips)", query: "?probe=gather&groups=2&mips=0", minCovered: 300, mutations: ["extinction"] },
  { name: "resolve", query: "?probe=resolve&temporal=0", minCovered: 512, mutations: ["scan-stop", "wgsl-resolve-order"] },
  { name: "resolve (temporal)", query: "?probe=resolve&temporal=1", minCovered: 512, mutations: ["temporal-clamp", "scan-stop", "wgsl-resolve-order"] },
  { name: "integrate", query: "?probe=integrate", minCovered: 512, mutations: ["coverage", "taps"] },
  // K6.1d: the gather on the core sparse source (K6.7), volume A skipping empty space (hdda), B wrapped (no skip)
  { name: "gather sparse (lod none, one group, hdda)", query: "?probe=gather&source=sparse&lod=none&groups=1", minCovered: 200, mutations: ["skip", "extinction", "phase", "wgsl-gather-extinction"] },
  { name: "gather sparse (node-average, two groups)", query: "?probe=gather&source=sparse&lod=node-average&groups=2", minCovered: 300, mutations: ["mix"] },
  { name: "gather sparse (coarse-pyramid, one group)", query: "?probe=gather&source=sparse&lod=coarse-pyramid&groups=1", minCovered: 300, mutations: ["extinction"] },
];

for (const p of PROBES) {
  test(`K6.1 probe: ${p.name} matches its CPU twin; mutations fail`, async ({ page }) => {
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.stats!.compared, detail(r)).toBeGreaterThanOrEqual(512);
    expect(r.stats!.covered, detail(r)).toBeGreaterThanOrEqual(p.minCovered);
    expect(r.stats!.over, detail(r)).toBeLessThanOrEqual(Math.floor(r.stats!.compared * 0.001));
    if (r.renderStats) {
      expect(r.renderStats.over, detail(r)).toBe(0);
      expect(r.renderStats.covered, detail(r)).toBeGreaterThan(30);
    }
    console.log(`[k6-core] ${p.name} on ${r.adapter}: ${r.stats!.compared} texels, ${r.stats!.covered} non-zero, max ${r.stats!.maxUlp.toFixed(2)} f16 ulp, ${r.stats!.over} over`);
    for (const m of p.mutations) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad.stats!.over, `mutation ${m} must fail: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad.stats!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// K6.2 (design §7.1): 'frame' lighting, each kernel against its twin from the read-back input of the kernel before it:
// the shadow grid (vdbFrameGridDensityVoxel), the transmittance (vdbFrameTransmittanceVoxel3d on the read-back grid) and
// the fused gather (his density + lighting twins on the read-back transmittance). `mutations` maps a twin term to the
// stats it must break. A second identical update skips the grid and the transmittance (his redraw signatures).
type StatsKey = "stats" | "gridStats" | "transmittanceStats";
const FRAME_PROBES: { name: string; query: string; cascades: number; mutations: Record<string, StatsKey> }[] = [
  {
    name: "dense, one group, one cascade",
    query: "?probe=frame",
    cascades: 1,
    mutations: { "grid-absorbance": "gridStats", "grid-lod": "gridStats", diffusion: "transmittanceStats", slots: "stats", sky: "stats", extinction: "stats" },
  },
  { name: "dense, two groups (partials), two cascades", query: "?probe=frame&groups=2&cascades=2", cascades: 2, mutations: { slots: "stats" } },
  { name: "sparse, one group", query: "?probe=frame&source=sparse", cascades: 1, mutations: { "grid-absorbance": "gridStats", diffusion: "transmittanceStats" } },
  { name: "sparse, two groups, single scattering", query: "?probe=frame&source=sparse&groups=2&ms=0", cascades: 1, mutations: { slots: "stats" } },
];

for (const p of FRAME_PROBES) {
  test(`K6.2 probe: 'frame' lighting (${p.name}) matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.frame!.cascades, detail(r)).toBe(p.cascades);
    const redraw = r.frame!.redraw as { shadowDensity: { drawn: number; skipped: number }; transmittance: { drawn: number; skipped: number } };
    expect(redraw.shadowDensity, detail(r)).toEqual({ drawn: 1, skipped: 1 });
    expect(redraw.transmittance, detail(r)).toEqual({ drawn: 1, skipped: 1 });
    for (const key of ["gridStats", "transmittanceStats", "stats"] as const) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "stats" ? 200 : 250);
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] frame ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// K6.3 (design §3.6): phase functions (the Lorenz-Mie LUT, per-channel g) and every multiple scattering mode in both
// lightings, each kernel against its twin from the read-back input before it ('mlp' / 'multi-scale': grid → mips →
// transmittance). `mutations` maps a twin term to the stats it must break.
type K63Key = "stats" | "gridStats" | "transmittanceStats" | "environmentGridStats" | "mipStats";
const K63_PROBES: { name: string; query: string; keys: K63Key[]; mutations: Record<string, K63Key> }[] = [
  { name: "gather, 'lorenz-mie'", query: "?probe=gather&phase=lorenz-mie", keys: ["stats"], mutations: { mie: "stats" } },
  { name: "gather, 'lorenz-mie-rgb'", query: "?probe=gather&phase=lorenz-mie-rgb&source=sparse&lod=none", keys: ["stats"], mutations: { mie: "stats" } },
  { name: "gather, 'octaves'", query: "?probe=gather&ms=octaves", keys: ["stats"], mutations: { octaves: "stats" } },
  { name: "gather, 'octaves' with the environment ('directional')", query: "?probe=gather&ms=octaves&env=directional", keys: ["stats"], mutations: { octaves: "stats" } },
  { name: "gather, 'two-sided'", query: "?probe=gather&ms=two-sided", keys: ["stats"], mutations: { "two-sided": "stats" } },
  { name: "frame, 'lorenz-mie'", query: "?probe=frame&phase=lorenz-mie", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { mie: "stats" } },
  { name: "frame, 'lorenz-mie-rgb'", query: "?probe=frame&phase=lorenz-mie-rgb", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { mie: "stats" } },
  { name: "frame, henyey-greenstein, phaseGExponent 1.5", query: "?probe=frame&phase=henyey-greenstein&exponent=1.5", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { slots: "stats" } },
  { name: "frame, 'octaves'", query: "?probe=frame&ms=octaves", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { octaves: "transmittanceStats" } },
  {
    name: "frame, 'octaves', environment 'directional' (slot + grid)",
    query: "?probe=frame&ms=octaves&env=directional",
    keys: ["gridStats", "transmittanceStats", "environmentGridStats", "stats"],
    mutations: { octaves: "environmentGridStats" },
  },
  { name: "frame, 'two-sided'", query: "?probe=frame&ms=two-sided", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "two-sided": "transmittanceStats" } },
  { name: "frame, 'two-sided', sparse, environment 'spherical-harmonics'", query: "?probe=frame&ms=two-sided&source=sparse&env=spherical-harmonics", keys: ["gridStats", "transmittanceStats", "environmentGridStats", "stats"], mutations: { "two-sided": "transmittanceStats" } },
  {
    name: "frame, 'mlp' (his network)",
    query: "?probe=frame&ms=mlp",
    keys: ["gridStats", "mipStats", "transmittanceStats", "stats"],
    mutations: { "mlp-mip": "mipStats", "mlp-weights": "transmittanceStats", "mlp-depth": "transmittanceStats" },
  },
  { name: "frame, 'neural' (= 'mlp'), sparse, two groups", query: "?probe=frame&ms=neural&source=sparse&groups=2", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "mlp-weights": "transmittanceStats" } },
  { name: "frame, 'multi-scale' (edge zero)", query: "?probe=frame&ms=multi-scale", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "ms-edge": "mipStats", "ms-levels": "transmittanceStats" } },
  { name: "frame, 'multi-scale' (edge clamp, 4 levels)", query: "?probe=frame&ms=multi-scale&edge=clamp&levels=4", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "ms-edge": "mipStats" } },
  // K6.3a: the paper's wider levels (`multiScaleSigma: 'paper'`: level k reads mip k + 1, σ ≈ 2^k voxels; one more mip)
  {
    name: "frame, 'multi-scale' (σ = 2^k, paper)",
    query: "?probe=frame&ms=multi-scale&sigma=paper",
    keys: ["gridStats", "mipStats", "transmittanceStats", "stats"],
    mutations: { "ms-sigma": "transmittanceStats", "ms-edge": "mipStats" },
  },
  { name: "frame, 'multi-scale' (σ = 2^k, 8 levels: the last level clamps to the chain)", query: "?probe=frame&ms=multi-scale&sigma=paper&levels=8", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "ms-sigma": "transmittanceStats" } },
  // K6.5b × K6.3: host occluder maps with each mode's own scattered term (octaves: occluded in the gather)
  { name: "frame, 'octaves' with occluder maps", query: "?probe=frame&ms=octaves&occluders=1", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { occluder: "stats" } },
  { name: "frame, 'two-sided' with occluder maps", query: "?probe=frame&ms=two-sided&occluders=1", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "occluder-scattered": "transmittanceStats" } },
  { name: "frame, 'mlp' with occluder maps", query: "?probe=frame&ms=mlp&occluders=1", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "occluder-scattered": "transmittanceStats" } },
  { name: "frame, 'multi-scale' with occluder maps", query: "?probe=frame&ms=multi-scale&occluders=1", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "occluder-scattered": "transmittanceStats" } },
  // K6.11: the emission colour in the fused 'frame' gather (volume B's LUT row / user hook; the twin was white before; a dense source needs groups=2: one group gives B the source of A, which has no emission)
  { name: "frame, emission colour LUT", query: "?probe=frame&ms=0&groups=2&emissionColor=lut", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "emission-color": "stats" } },
  { name: "frame, emission colour hook, sparse", query: "?probe=frame&ms=0&source=sparse&emissionColor=hook", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "emission-color": "stats" } },
  { name: "gather, 'two-sided' with occluder maps", query: "?probe=gather&ms=two-sided&occluders=1", keys: ["stats"], mutations: { occluder: "stats" } },
];

for (const p of K63_PROBES) {
  test(`K6.3 probe: ${p.name} matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.warnings, detail(r)).toEqual([]);
    for (const key of p.keys) {
      const s = r[key]!;
      expect(s, `${key} missing: ${detail(r)}`).toBeDefined();
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "mipStats" ? 64 : 512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(minCovered(key, p.query));
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] K6.3 ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
      console.log(`[k6-core] K6.3 ${p.name} mutation ${m}: ${key} ${bad[key]!.over} / ${bad[key]!.compared} over`);
    }
    expect(errors).toEqual([]);
  });
}

// K6.4 (design §3.3–§3.4, §3.7): spectral media in every kernel with a spectral variant, each against its twin from the
// read-back input of the kernel before it (extinction + absorption stacks → every slot group's transmittance → the
// environment grid and the fused gather's two scatter textures; the 'volume' gather; resolve → two prefixes; the spectral
// composite; spectral volume shadow maps and their rgb receivers). `mutations` maps a twin term to the stats it must break.
type K64Key = "stats" | "renderStats" | "gridStats" | "transmittanceStats" | "environmentGridStats" | "receiverStats";
const K64_PROBES: { name: string; query: string; keys: K64Key[]; mutations: Record<string, K64Key> }[] = [
  { name: "gather spectral, dense", query: "?probe=gather&spectral=1", keys: ["stats", "renderStats"], mutations: { spectrum: "stats", extinction: "stats", "wgsl-gather-extinction": "stats", "wgsl-gather-phase": "stats" } },
  { name: "gather spectral, two groups (partials)", query: "?probe=gather&spectral=1&groups=2", keys: ["stats", "renderStats"], mutations: { mix: "stats" } },
  { name: "gather spectral, 'octaves', environment 'directional'", query: "?probe=gather&spectral=1&ms=octaves&env=directional", keys: ["stats"], mutations: { octaves: "stats" } },
  { name: "gather spectral, 'two-sided', sparse, occluder maps", query: "?probe=gather&spectral=1&ms=two-sided&source=sparse&occluders=1", keys: ["stats"], mutations: { "two-sided": "stats", occluder: "stats" } },
  { name: "gather spectral, single scattering, 'outward' SH", query: "?probe=gather&spectral=1&ms=0&env=outward", keys: ["stats"], mutations: { spectrum: "stats" } },
  {
    name: "frame spectral, dense, 1 cascade",
    query: "?probe=frame&spectral=1",
    keys: ["gridStats", "transmittanceStats", "stats"],
    mutations: { "spectral-absorption": "gridStats", spectrum: "gridStats", diffusion: "transmittanceStats", "spectral-pack": "transmittanceStats", slots: "stats", sky: "stats" },
  },
  { name: "frame spectral, dense, two groups (partials), 2 cascades", query: "?probe=frame&spectral=1&groups=2&cascades=2", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { slots: "stats" } },
  {
    name: "frame spectral, 'octaves', environment 'directional' (slot + grid)",
    query: "?probe=frame&spectral=1&ms=octaves&env=directional",
    keys: ["gridStats", "transmittanceStats", "environmentGridStats", "stats"],
    mutations: { octaves: "environmentGridStats", "env-slot": "stats" },
  },
  {
    name: "frame spectral, 'two-sided', sparse, environment 'spherical-harmonics'",
    query: "?probe=frame&spectral=1&ms=two-sided&source=sparse&env=spherical-harmonics",
    keys: ["gridStats", "transmittanceStats", "environmentGridStats", "stats"],
    mutations: { "two-sided": "transmittanceStats" },
  },
  { name: "frame spectral, 'outward' environment, single scattering", query: "?probe=frame&spectral=1&ms=0&env=outward", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "env-outward": "stats" } },
  { name: "frame spectral, 'mlp' runs as 1D diffusion", query: "?probe=frame&spectral=1&ms=mlp", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { diffusion: "transmittanceStats" } },
  { name: "frame spectral, occluder maps", query: "?probe=frame&spectral=1&occluders=1", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { occluder: "stats", "occluder-scattered": "transmittanceStats" } },
  { name: "frame spectral, emission colour LUT", query: "?probe=frame&spectral=1&ms=0&groups=2&emissionColor=lut", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "emission-color": "stats" } },
  { name: "resolve spectral", query: "?probe=resolve&spectral=1", keys: ["stats"], mutations: { "spectral-alpha": "stats", "scan-stop": "stats", "wgsl-resolve-order": "stats" } },
  { name: "resolve spectral, temporal", query: "?probe=resolve&spectral=1&temporal=1", keys: ["stats"], mutations: { "spectral-alpha": "stats" } },
  { name: "resolve spectral, rg16float alpha prefix (texture-formats-tier1 when the adapter has it)", query: "?probe=resolve&spectral=1&tier1=1", keys: ["stats"], mutations: { "spectral-alpha": "stats", "wgsl-resolve-order": "stats" } },
  { name: "integrate spectral (linear)", query: "?probe=integrate&spectral=1", keys: ["stats"], mutations: { "spectral-alpha": "stats", coverage: "stats" } },
  { name: "integrate spectral (sRGB)", query: "?probe=integrate&spectral=1&srgb=1", keys: ["stats"], mutations: { "spectral-alpha": "stats" } },
  { name: "shadow map spectral, directional (2 cascades)", query: "?probe=shadow&spectral=1&light=dir&cascades=2", keys: ["stats", "receiverStats"], mutations: { spectrum: "stats", "receiver-far": "receiverStats" } },
  { name: "shadow map spectral, point (cube)", query: "?probe=shadow&spectral=1&light=point", keys: ["stats", "receiverStats"], mutations: { spectrum: "stats", "receiver-far": "receiverStats" } },
  { name: "shadow map spectral, spot, sparse", query: "?probe=shadow&spectral=1&light=spot&source=sparse", keys: ["stats", "receiverStats"], mutations: { spectrum: "stats" } },
];

for (const p of K64_PROBES) {
  test(`K6.4 probe: ${p.name} matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.warnings, detail(r)).toEqual([]);
    for (const key of p.keys) {
      const s = r[key]!;
      expect(s, `${key} missing: ${detail(r)}`).toBeDefined();
      // render(): one value per pixel of the 24 x 16 froxel grid
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "renderStats" ? 384 : 512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(minCovered(key, p.query));
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] K6.4 ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// K6.5 (design §2.6, §3.8): volume shadow maps (each light kind) and the emission kernels, each against its twin from
// the read-back input of the kernel before it (grid → map → receivers; directions → reduce → probes → receivers).
type K65Key = "stats" | "receiverStats" | "reduceStats" | "probeStats" | "irradianceStats";
const K65_PROBES: { name: string; query: string; keys: K65Key[]; mutations: Record<string, K65Key> }[] = [
  { name: "shadow map, directional (2 cascades), dense", query: "?probe=shadow&light=dir&cascades=2", keys: ["stats", "receiverStats"], mutations: { "map-steps": "stats", "map-far": "stats", "receiver-far": "receiverStats", "receiver-layer": "receiverStats" } },
  { name: "shadow map, spot, dense", query: "?probe=shadow&light=spot", keys: ["stats", "receiverStats"], mutations: { "map-far": "stats", "receiver-far": "receiverStats" } },
  { name: "shadow map, point (cube), dense", query: "?probe=shadow&light=point", keys: ["stats", "receiverStats"], mutations: { "map-steps": "stats", "receiver-far": "receiverStats" } },
  { name: "shadow map, directional (4 cascades), sparse", query: "?probe=shadow&light=dir&cascades=4&source=sparse", keys: ["stats", "receiverStats"], mutations: { "receiver-layer": "receiverStats" } },
  {
    name: "emission, dense, soot LUT",
    query: "?probe=emission",
    keys: ["stats", "reduceStats", "probeStats", "irradianceStats"],
    mutations: { "emission-direction": "stats", "emission-color": "stats", "emission-reduce": "reduceStats", "emission-probe": "probeStats", "emission-irradiance": "irradianceStats" },
  },
  { name: "emission, sparse, 2 probe cascades", query: "?probe=emission&source=sparse&cascades=2", keys: ["stats", "reduceStats", "probeStats", "irradianceStats"], mutations: { "emission-color": "stats" } },
  { name: "emission, dense, user WGSL hook", query: "?probe=emission&hook=1", keys: ["stats", "reduceStats", "probeStats", "irradianceStats"], mutations: { "emission-color": "stats" } },
];

for (const p of K65_PROBES) {
  test(`K6.5 probe: ${p.name} matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.warnings, detail(r)).toEqual([]);
    for (const key of p.keys) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "reduceStats" ? 2 : key === "stats" && p.query.includes("emission") ? 32 : 512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(minCovered(key, p.query));
      expect(s.over, `${key}: ${detail(r)}`).toBe(0);
      console.log(`[k6-core] K6.5 ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toExponential(2)} (${s.bound}), max |d| ${s.maxAbs.toExponential(2)}, ${s.over} over`);
    }
    if (r.shadow) {
      console.log(`[k6-core] K6.5 ${p.name}: ${JSON.stringify(r.shadow)}`);
      expect(r.shadow.enabled, detail(r)).toBe(true);
      expect(r.shadow.shadowed, detail(r)).toBeGreaterThan(20);
      expect(r.shadow.receiverShadowed, detail(r)).toBeGreaterThan(20);
      // nothing changed: no piece; the light moved: one piece (a directional layout draws its cascades in one piece)
      expect(r.shadow.same, detail(r)).toEqual({ drawn: 0, waiting: 0 });
      expect(r.shadow.moved.drawn, detail(r)).toBeGreaterThanOrEqual(1);
    }
    if (r.emission) {
      console.log(`[k6-core] K6.5 ${p.name}: ${JSON.stringify(r.emission)}`);
      expect(r.emission.first, detail(r)).toEqual({ directions: 1, probes: true });
      expect(r.emission.second, detail(r)).toEqual({ directions: 0, probes: false });
      expect(r.emission.outside, detail(r)).toBeGreaterThan(20);
      if (p.query.includes("hook=1")) {
        expect(r.emission.hookErrors.length, detail(r)).toBe(3);
        expect(r.emission.hookErrors[0], detail(r)).toMatch(/VDB_BAD_WGSL.*line 1: top-level name 'emissionColour'/);
        expect(r.emission.hookErrors[1], detail(r)).toMatch(/VDB_BAD_WGSL.*line 3/);
        expect(r.emission.hookErrors[2], detail(r)).toMatch(/VDB_BAD_WGSL.*vdbUserEmissionColor\(f32, f32\)/);
      }
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(0);
    }
    expect(errors).toEqual([]);
  });
}

// K6.10b: the branches the K6 gate review found untested (docs/archive/handoffs/K6-10-gate-review.md finding 4c), each
// against its twin, with WGSL-side mutations (k6-core.ts WGSL_PATCHES) that must fail. `mutations` maps each to the stats
// it must break.
type K610bKey = "stats" | "receiverStats";
const K610B_PROBES: { name: string; query: string; keys: K610bKey[]; mutations: Record<string, K610bKey> }[] = [
  { name: "composite, grey sRGB blend over the scene colour", query: "?probe=integrate&srgb=1", keys: ["stats"], mutations: { "wgsl-composite-srgb": "stats", coverage: "stats" } },
  { name: "composite, orthographic scene depth", query: "?probe=integrate&depthMode=ortho", keys: ["stats"], mutations: { "wgsl-composite-ortho": "stats" } },
  { name: "composite, reversed perspective scene depth", query: "?probe=integrate&depthMode=reversed", keys: ["stats"], mutations: { "wgsl-composite-reversed": "stats" } },
  { name: "composite, depth32float scene depth", query: "?probe=integrate&depthFormat=depth32float", keys: ["stats"], mutations: { "wgsl-composite-depth": "stats" } },
  {
    name: "composite, reversed orthographic depth32float, sRGB",
    query: "?probe=integrate&depthMode=ortho-reversed&depthFormat=depth32float&srgb=1",
    keys: ["stats"],
    mutations: { "wgsl-composite-ortho": "stats", "wgsl-composite-reversed": "stats", "wgsl-composite-depth": "stats", "wgsl-composite-srgb": "stats" },
  },
  // the emission colour inside the 'volume' gathers (volume B, the one with an emission channel (dense: groups=2): a synthetic LUT row or a user hook; the twins read the same)
  { name: "gather, emission colour LUT row", query: "?probe=gather&groups=2&emissionColor=lut", keys: ["stats"], mutations: { "wgsl-emission-lut": "stats", "emission-color": "stats" } },
  { name: "gather sparse, emission colour user WGSL hook", query: "?probe=gather&source=sparse&lod=none&emissionColor=hook", keys: ["stats"], mutations: { "wgsl-emission-hook": "stats", "emission-color": "stats" } },
  { name: "gather spectral, emission colour LUT row", query: "?probe=gather&spectral=1&groups=2&emissionColor=lut", keys: ["stats"], mutations: { "wgsl-emission-lut": "stats" } },
  // the temporal resolve with a moving camera: the history read between texels, a per-froxel motion blend
  { name: "resolve (temporal), the camera moved", query: "?probe=resolve&move=1", keys: ["stats"], mutations: { "wgsl-resolve-reproject": "stats", "wgsl-resolve-motion": "stats", "temporal-clamp": "stats" } },
];

for (const p of K610B_PROBES) {
  test(`K6.10b probe: ${p.name} matches its CPU twins; WGSL mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.warnings, detail(r)).toEqual([]);
    for (const key of p.keys) {
      const s = r[key]!;
      expect(s, `${key} missing: ${detail(r)}`).toBeDefined();
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(minCovered(key, p.query));
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] K6.10b ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    if (p.query.includes("move=1")) {
      // most froxels keep some history (blend < 1), read off the texel centres
      console.log(`[k6-core] K6.10b ${p.name}: ${JSON.stringify(r.motion)}`);
      expect(r.motion!.partial, detail(r)).toBeGreaterThan(r.motion!.valid / 2);
      expect(r.motion!.offCentre, detail(r)).toBeGreaterThan(r.motion!.valid / 4);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
      console.log(`[k6-core] K6.10b ${p.name} mutation ${m}: ${key} ${bad[key]!.over} / ${bad[key]!.compared} over`);
    }
    expect(errors).toEqual([]);
  });
}

test("K6.1 a second identical update builds no pipeline or module; two loads agree", async ({ page }) => {
  test.setTimeout(600_000);
  const r = await run(page, "?probe=gather&swap=1");
  expect(r.error, detail(r)).toBeUndefined();
  expect(r.builds!.first, detail(r)).toBeGreaterThan(0);
  expect(r.builds!.second, detail(r)).toBe(0);
  const again = await run(page, "?probe=gather&swap=1");
  expect(again.stats, detail(again)).toEqual(r.stats);
});

// ---------------------------------------------------------------------------------------------------------------
// K6.8: the core-vs-TSL A/B rows (K6.1 Step 1, K6.2 Step 3, K6.3 Step 4, K6.4 Step 5, K6.1d WDAS, K6.2b environment) are
// gone with the TSL path. Their scenes render through the core in e2e/k6-frames.spec.ts, which compares them with the
// frames before a change (K6_FRAMES_BASELINE) and writes artifacts/k6/ours-core/ for scripts/k3/compare.mjs against his
// captures (the regression report). The last core-vs-TSL numbers are in docs/archive/handoffs/K6-*.md and K6-8.md.
// ---------------------------------------------------------------------------------------------------------------
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
interface K3Result {
  done: boolean;
  error?: string;
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  uncaptured: string[];
  warnings: string[];
}

async function k3(page: Page, query: string): Promise<K3Result> {
  await page.goto(`/src/harness/k3-frame.html${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_FRAME__?: { done: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 600_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_FRAME__?: unknown }).__VDB_K3_FRAME__)) as K3Result;
}
const k3Image = (r: K3Result) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });
const k3Detail = (r: K3Result) => JSON.stringify({ ...r, rgba: undefined });

// K6.2b (design §3.5, Q3): the environment reduction on the GPU, each stage against its twin from the read-back input of
// the stage before it: the work map (exact-area equirect / supersampled cube), his 64 x 32 map (the GGX roughness-0.4
// convolution) and the environment buffer (his lights and the SH9). f32 outputs: 1e-3 relative (SwiftShader's
// transcendental and division precision; the twins are f64). The stages rerun only for their inputs.
type EnvKey = "workStats" | "mapStats" | "bufferStats";
const ENV_PROBES: { name: string; query: string; count: number; mutations: Record<string, EnvKey> }[] = [
  { name: "equirect rgba16float, 'directional'", query: "?probe=environment&mode=directional", count: 8, mutations: { "env-area": "workStats", "env-blur": "mapStats", "env-seeds": "bufferStats", "env-sh": "bufferStats" } },
  { name: "equirect, rotated and reflected, 'directional' pairs", query: "?probe=environment&mode=directional&pairs=1&rotate=1", count: 8, mutations: { "env-seeds": "bufferStats", "env-sh": "bufferStats" } },
  { name: "equirect rgba32float, rotated, SH L2 lights", query: "?probe=environment&mode=spherical-harmonics&order=2&rotate=1&format=rgba32float", count: 12, mutations: { "env-blur": "mapStats" } },
  { name: "equirect, SH L1 lights, 3 samples", query: "?probe=environment&mode=spherical-harmonics&order=1&samples=3", count: 6, mutations: { "env-sh": "bufferStats", "env-blur": "mapStats" } },
  { name: "cube rgba16float, 'directional'", query: "?probe=environment&projection=cube&mode=directional&samples=5", count: 5, mutations: { "env-area": "workStats" } },
  { name: "equirect, 'outward' (SH only)", query: "?probe=environment&mode=outward", count: 0, mutations: { "env-sh": "bufferStats" } },
];

for (const p of ENV_PROBES) {
  test(`K6.2b probe: environment (${p.name}) matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    for (const key of ["workStats", "mapStats", "bufferStats"] as const) console.log(`[k6-core] environment ${p.name} ${key}: ${JSON.stringify(r[key])}`);
    expect(r.environment!.valid, detail(r)).toBe(1);
    expect(r.environment!.count, detail(r)).toBe(p.count);
    for (const key of ["workStats", "mapStats", "bufferStats"] as const) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "bufferStats" ? 10 : 2048);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(minCovered(key, p.query));
      // cube: a sub-sample near a face texel edge may pick the neighbour texel (its direction comes from WGSL sin / cos,
      // 2^-11 absolute): one sub-sample of k^2 per flipped texel, a few % of the work texels at most
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(key === "workStats" && p.query.includes("cube") ? Math.floor(s.compared * 0.02) : 0);
      console.log(`[k6-core] environment ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toExponential(2)} relative, ${s.over} over`);
    }
    // identical update: nothing; intensity: reduce; rotation: blur + reduce; revision: all three
    const runs = r.environment!.runs;
    expect(runs, detail(r)).toEqual([
      { prefilter: 1, blur: 1, reduce: 1 },
      { prefilter: 1, blur: 1, reduce: 1 },
      { prefilter: 1, blur: 1, reduce: 2 },
      { prefilter: 1, blur: 2, reduce: 3 },
      { prefilter: 2, blur: 3, reduce: 4 },
    ]);
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(0);
    }
    expect(errors).toEqual([]);
  });
}

// K6.2b: 'frame' lighting with an environment (a synthetic sky, rotated and reflected): the dominant 'directional' light in
// the next free slot (single scattering), the environment grid (vdbFrameEnvironmentGridVoxel3d on the read-back grid and
// buffer) and the 'outward' SH term of the fused gather; 'volume' lighting marches every environment light.
type FrameEnvKey = "stats" | "transmittanceStats" | "environmentGridStats";
const FRAME_ENV_PROBES: { name: string; query: string; grid: boolean; mutations: Record<string, FrameEnvKey> }[] = [
  { name: "'directional', single scattering (slot + grid)", query: "?probe=frame&env=directional&ms=0", grid: true, mutations: { "env-slot": "transmittanceStats", "env-grid": "environmentGridStats" } },
  { name: "'directional', diffusion (grid only), sparse", query: "?probe=frame&env=directional&source=sparse", grid: true, mutations: { "env-grid": "environmentGridStats" } },
  { name: "SH lights, single scattering, two groups", query: "?probe=frame&env=spherical-harmonics&ms=0&groups=2", grid: true, mutations: { "env-grid": "environmentGridStats" } },
  { name: "'outward' SH irradiance", query: "?probe=frame&env=outward", grid: false, mutations: { "env-outward": "stats" } },
];

for (const p of FRAME_ENV_PROBES) {
  test(`K6.2b probe: 'frame' lighting with an environment (${p.name}) matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.environment!.valid, detail(r)).toBe(1);
    const redraw = r.frame!.redraw as { environmentGrid: { drawn: number; skipped: number } };
    expect(redraw.environmentGrid, detail(r)).toEqual(p.grid ? { drawn: 1, skipped: 1 } : { drawn: 0, skipped: 0 });
    for (const key of ["transmittanceStats", "stats", ...(p.grid ? (["environmentGridStats"] as const) : [])] as const) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(200);
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] frame env ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

const GATHER_ENV_PROBES: { name: string; query: string; mutations: string[] }[] = [
  { name: "'directional' lights, diffusion", query: "?probe=gather&env=directional", mutations: ["env-lights", "env-outward"] },
  { name: "SH lights, single scattering, sparse", query: "?probe=gather&env=spherical-harmonics&ms=0&source=sparse&lod=none", mutations: ["env-lights"] },
  { name: "'outward' SH irradiance", query: "?probe=gather&env=outward", mutations: ["env-outward"] },
];

for (const p of GATHER_ENV_PROBES) {
  test(`K6.2b probe: 'volume' gather with an environment (${p.name}) matches its CPU twin; mutations fail`, async ({ page }) => {
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.environment!.valid, detail(r)).toBe(1);
    expect(r.stats!.covered, detail(r)).toBeGreaterThanOrEqual(200);
    expect(r.stats!.over, detail(r)).toBeLessThanOrEqual(Math.floor(r.stats!.compared * 0.001));
    console.log(`[k6-core] gather env ${p.name} on ${r.adapter}: ${r.stats!.compared} texels, ${r.stats!.covered} non-zero, max ${r.stats!.maxUlp.toFixed(2)} f16 ulp, ${r.stats!.over} over`);
    for (const m of p.mutations) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad.stats!.over, `mutation ${m} must fail: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad.stats!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// K6.5b (design §3.7): host occluder maps (mesh → volume shadows). Three synthetic maps of the probe lights
// (k6-occluders-synth.ts): a depth32float orthographic map drawn by a render pass (directional, bilinear PCF, a bias), an
// r32float cube (point, bilinear) and an r32float reversed-depth perspective map (spot, one texel). The transmittance,
// the fused 'frame' gather and the 'volume' gather against their twins with `vdbFrameOccluderVisibility`; the twin
// without the maps, with the scattered part unoccluded, or with one texel everywhere must fail.
const OCCLUDER_PROBES: { name: string; query: string; frame: boolean; mutations: Record<string, StatsKey> }[] = [
  { name: "'frame' lighting, dense", query: "?probe=frame&occluders=1", frame: true, mutations: { occluder: "stats", "occluder-scattered": "transmittanceStats", "occluder-filter": "stats" } },
  { name: "'frame' lighting, sparse, two groups, single scattering", query: "?probe=frame&occluders=1&source=sparse&groups=2&ms=0", frame: true, mutations: { occluder: "stats" } },
  { name: "'volume' gather, dense", query: "?probe=gather&occluders=1", frame: false, mutations: { occluder: "stats", "occluder-filter": "stats" } },
  { name: "'volume' gather, sparse", query: "?probe=gather&occluders=1&source=sparse&lod=none", frame: false, mutations: { occluder: "stats" } },
];

for (const p of OCCLUDER_PROBES) {
  test(`K6.5b probe: occluder maps in ${p.name} match the CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.occluders!.maps, detail(r)).toBe(3);
    // the maps shadow something in every kernel the probe checks
    if (p.frame) {
      expect(r.occluders!.transmittance, detail(r)).toBeGreaterThan(50);
      expect(r.occluders!.density, detail(r)).toBeGreaterThan(20);
    } else expect(r.occluders!.gather, detail(r)).toBeGreaterThan(20);
    for (const key of p.frame ? (["transmittanceStats", "stats"] as const) : (["stats"] as const)) {
      const s = r[key]!;
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(200);
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] occluders ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over; occluded lookups ${JSON.stringify(r.occluders)}`);
    }
    if (p.frame) {
      // a map with a revision that does not move lets the transmittance skip its redraw
      const redraw = r.frame!.redraw as { transmittance: { drawn: number; skipped: number } };
      expect(redraw.transmittance, detail(r)).toEqual({ drawn: 1, skipped: 1 });
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// ---------------------------------------------------------------------------------------------------------------
// K6.1c: the raw `?scene=` mode (no three.js; the host frame loop of demo 17) vs the same scene through `./three`'s
// core path (`k3-frame`), same device and kernels: the host seams (camera, lights, targets, composite)
// must add nothing, so the bar is the K6.1 A/B's (PSNR ≥ 45 dB, ≤ 0.2 % pixels > 4/255). `smoke` is k3-frame's own
// CC0 scene (runs on CI); his Step 1 / Step 3 isolated scenes need the bunny and run with K6_HIS_SCENES=1 (1280 × 720,
// 512 × 512 × 24), writing PNGs to artifacts/k6/ours-raw/ for scripts/k3/compare.mjs against his captures.
// ---------------------------------------------------------------------------------------------------------------
interface RawScene {
  name: string;
  placement: string;
  volumes: number;
  nonBackground: number;
  inPageIdentical: boolean;
  secondBuilds: number;
}
const RAW_DIR = path.join(REPO, "artifacts", "k6", "raw");
const rawImage = (r: Result & { width?: number; height?: number; rgba?: string }) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });

async function rawVsCore(page: Page, scene: string, query: string, k3Query: string): Promise<void> {
  const errors = collectErrors(page);
  const raw = (await run(page, `?scene=${scene}${query}`, 1_200_000)) as Result & { scene?: RawScene; width?: number; height?: number; rgba?: string };
  const rawDetail = JSON.stringify({ ...raw, rgba: undefined }, null, 2);
  expect(raw.error, rawDetail).toBeUndefined();
  expect(raw.uncaptured, rawDetail).toEqual([]);
  expect(raw.scene!.inPageIdentical, rawDetail).toBe(true);
  expect(raw.scene!.secondBuilds, rawDetail).toBe(0);
  expect(raw.scene!.nonBackground, rawDetail).toBeGreaterThan(500);
  const core = await k3(page, `?scene=${scene}${k3Query}`);
  expect(core.error, k3Detail(core)).toBeUndefined();
  const cmp = compareImages(rawImage(raw), k3Image(core), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
  mkdirSync(RAW_DIR, { recursive: true });
  const name = `${scene}${query.replace(/[&=]/g, "-")}`;
  writeFileSync(path.join(RAW_DIR, `${name}-raw.png`), encodePng(rawImage(raw)));
  writeFileSync(path.join(RAW_DIR, `${name}-three-core.png`), encodePng(k3Image(core)));
  writeFileSync(path.join(RAW_DIR, `${name}-diff.png`), encodePng(diffHeatmap(rawImage(raw), k3Image(core), { threshold: 4, fullScale: 16 })));
  writeFileSync(path.join(RAW_DIR, `${name}.json`), JSON.stringify({ scene, query, ...cmp, raw: raw.scene, rawWarnings: raw.warnings }, null, 2));
  if (!query) {
    mkdirSync(path.join(REPO, "artifacts", "k6", "ours-raw"), { recursive: true });
    writeFileSync(path.join(REPO, "artifacts", "k6", "ours-raw", `${scene}.png`), encodePng(rawImage(raw)));
  }
  console.log(`[k6-raw] ${name} (${raw.scene!.placement}): raw vs three core PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff}); ${raw.scene!.nonBackground} volume px`);
  expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
  expect(errors, rawDetail).toEqual([]);
}

test("K6.1c raw scene: CC0 smoke, no three, equals the ./three core path", async ({ page }) => {
  test.setTimeout(900_000);
  await rawVsCore(page, "smoke", "", "");
});

test("K6.1c raw scene: CC0 smoke through the sparse source equals the ./three core path", async ({ page }) => {
  test.setTimeout(900_000);
  await rawVsCore(page, "smoke", "&source=sparse", "&source=sparse");
});

const RAW_HIS = [
  "step1-volume-dir-ms-off-isolated",
  "step1-volume-dir-ms-on-isolated",
  "step3-default-dir-isolated",
  "step3-shadow-cascades-4-dir-isolated",
  "step3-shadow-grid-high-dir-isolated",
  "step3-two-bunnies-dir-isolated",
];
for (const scene of RAW_HIS) {
  test(`K6.1c raw scene vs ./three core: ${scene}`, async ({ page }) => {
    test.skip(process.env.K6_HIS_SCENES !== "1", "set K6_HIS_SCENES=1 (needs fixtures/k3/bunny_cloud.wvdb)");
    test.skip(!existsSync(path.join(REPO, "fixtures", "k3", "bunny_cloud.wvdb")), "fixtures/k3/bunny_cloud.wvdb missing");
    test.setTimeout(1_800_000);
    // k3-frame renders his scenes with 4× MSAA (his page); the volume-only isolated frames have no mesh edges
    await rawVsCore(page, scene, "", "");
  });
}

// K6.8: `VDBFrameRenderer.passHook` brackets each core update: with `timestamp-query` the begin / end timestamps of the
// update's kernels (two empty compute passes carry the writes), else the CPU callbacks. A check that the bracket is
// recorded and resolves, on SwiftShader; no timing is taken (D25).
test("K6.8 passHook: the core update is bracketed (timestamp writes when the adapter has timestamp-query)", async ({ page }) => {
  test.setTimeout(600_000);
  const errors = collectErrors(page);
  await page.goto("/src/harness/k3-timing.html?asset=cloud04&source=dense&lighting=frame&w=320&h=180&froxels=96x64x16&timerCheck=1");
  await page.waitForFunction(() => (window as { __VDB_K3_TIMING__?: { done: boolean } }).__VDB_K3_TIMING__?.done === true, undefined, { timeout: 600_000 });
  const r = (await page.evaluate(() => (window as { __VDB_K3_TIMING__?: unknown }).__VDB_K3_TIMING__)) as { error?: string; uncaptured: string[]; timestampQuery?: boolean; timerCheck?: { updates: number[]; core: (number | null)[] } };
  const d = JSON.stringify(r);
  expect(r.error, d).toBeUndefined();
  expect(r.uncaptured, d).toEqual([]);
  // one core update per frame
  expect(r.timerCheck!.updates, d).toEqual([1, 1, 1]);
  if (r.timestampQuery) for (const ms of r.timerCheck!.core) expect(Number.isFinite(ms) && ms! >= 0, d).toBe(true);
  else expect(r.timerCheck!.core, d).toEqual([null, null, null]);
  console.log(`[k6-core] passHook on SwiftShader: timestamp-query ${r.timestampQuery}; ${JSON.stringify(r.timerCheck)}`);
  expect(errors).toEqual([]);
});
