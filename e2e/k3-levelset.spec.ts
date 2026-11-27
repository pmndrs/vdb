import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { encodePng } from "../scripts/k3/png.mjs";

// K3.7 (R1 §6 Step 7): level-set surfaces traced on the sparse pool (VDBLevelSetMaterial) and point grids
// (VDBPoints), examples/src/harness/k3-levelset.ts. Gates without his captures: the GPU trace against the CPU
// twin (silhouette IoU >= 0.99, hit distance, normal angle), the sphere against its analytic normal, the
// points' drawn count and coverage, the colour attribute, determinism (in page and across loads).
// PNGs go to artifacts/k3/ours-step7/ (scripts/k3/compare-step7.mjs compares them with his Step 7 captures).
const PAGE_URL = "/src/harness/k3-levelset.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OURS = path.join(repoRoot, "artifacts", "k3", "ours-step7");
const has = (rel: string) => existsSync(path.join(repoRoot, "fixtures", rel));

interface Stats {
  pixels: number;
  gpuHits: number;
  cpuHits: number;
  both: number;
  iou: number;
  hitMeanVoxels: number;
  hitMaxVoxels: number;
  normalMeanDeg: number;
  normalP99Deg: number;
  normalMaxDeg: number;
  normalInteriorMaxDeg: number;
  analytic?: { meanDeg: number; maxDeg: number; radiusSpreadVoxels: number };
  twinMs: number;
  avgSamples: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  nonBackground?: number;
  twin?: Stats;
  points?: { count: number; totalCount: number; covered: number; centresCovered: number; centresTotal: number; stray: number; drawn?: number };
  colors?: { checked: number; bad: number; firstBad?: unknown };
  timing?: { medianMs: number };
  warnings: string[];
  uncaptured: string[];
  [k: string]: unknown;
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_LEVELSET__?: { done: boolean } }).__VDB_K3_LEVELSET__?.done === true, undefined, {
    timeout: 400_000,
  });
  return (await page.evaluate(() => (window as { __VDB_K3_LEVELSET__?: unknown }).__VDB_K3_LEVELSET__)) as Result;
}

const detailOf = (r: Result) => JSON.stringify({ ...r, rgba: r.rgba ? `${r.rgba.length} b64 chars` : undefined }, null, 2);

function save(name: string, r: Result): Buffer {
  const png = Buffer.from(encodePng({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) }) as Uint8Array);
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, `${name}.png`), png);
  return png;
}

test("K3.7 level set: the GPU trace of the teapot matches the CPU twin (IoU, hits, normals) and is deterministic", async ({ page }) => {
  test.skip(!has("vdb-samples/utahteapot.vdb"), "fixtures/vdb-samples/utahteapot.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(600_000);
  const first = await run(page, "?scene=teapot&twin=1");
  const detail = detailOf(first);
  expect(first.error, detail).toBeUndefined();
  expect(first.uncaptured, detail).toEqual([]);
  expect(first.inPageIdentical, detail).toBe(true);
  const t = first.twin!;
  // His camera and placement: the teapot covers ~6,000 px of the 384x216 frame.
  expect(t.gpuHits, detail).toBeGreaterThan(5_000);
  expect(t.iou, detail).toBeGreaterThanOrEqual(0.99);
  // GPU f32 vs the f32 twin on the same pixel rays: the same walk, so the roots agree to the noise of the sample
  // positions (SwiftShader and Apple M5 Pro: IoU 1, max 0.002 voxels, normals within 0.004 degrees). Without the
  // bisection the hits move 0.016 voxels; rays from interpolated box positions (his positionViewDirection) 4 voxels.
  expect(t.hitMaxVoxels, detail).toBeLessThan(0.01);
  expect(t.normalP99Deg, detail).toBeLessThan(0.01);
  expect(t.normalInteriorMaxDeg, detail).toBeLessThan(0.1);

  const second = await run(page, "?scene=teapot");
  expect(second.error).toBeUndefined();
  const a = save("k3-levelset-teapot-run1", first);
  const b = save("k3-levelset-teapot-run2", second);
  expect(a.equals(b), "two page loads render byte-identical images").toBe(true);
  console.log(
    `[k3-levelset] ${first.adapter}: teapot IoU ${t.iou.toFixed(5)} (${t.gpuHits} GPU / ${t.cpuHits} CPU px), hit max ${t.hitMaxVoxels.toExponential(2)} voxels, normal mean ${t.normalMeanDeg.toFixed(3)}° p99 ${t.normalP99Deg.toFixed(3)}° max ${t.normalMaxDeg.toFixed(2)}° (interior ${t.normalInteriorMaxDeg.toFixed(2)}°), twin ${t.avgSamples.toFixed(1)} samples/px`,
  );
});

test("K3.7 level set: the sphere's traced normals match the analytic normal; its hits lie on one radius", async ({ page }) => {
  test.skip(!has("vdb-samples/sphere.vdb"), "fixtures/vdb-samples/sphere.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(300_000);
  const r = await run(page, "?scene=sphere&twin=1");
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  const t = r.twin!;
  expect(t.gpuHits, detail).toBeGreaterThan(10_000);
  expect(t.iou, detail).toBeGreaterThanOrEqual(0.99);
  expect(t.normalP99Deg, detail).toBeLessThan(0.1);
  // A sampled level set of radius ~59 voxels: the zero crossings sit on one sphere and the normals point away from
  // its centre (SwiftShader: spread 0.004 voxels, normals within 0.013 degrees).
  expect(t.analytic!.meanDeg, detail).toBeLessThan(0.1);
  expect(t.analytic!.maxDeg, detail).toBeLessThan(1);
  expect(t.analytic!.radiusSpreadVoxels, detail).toBeLessThan(0.1);
  save("k3-levelset-sphere", r);
  console.log(`[k3-levelset] sphere: analytic normal mean ${t.analytic!.meanDeg.toFixed(3)}° max ${t.analytic!.maxDeg.toFixed(3)}°, radius spread ${t.analytic!.radiusSpreadVoxels.toFixed(3)} voxels; twin IoU ${t.iou.toFixed(5)}`);
});

test("K3.7 level set: under a rotated, non-uniformly scaled instance the normals follow W^T (twin and analytic)", async ({ page }) => {
  test.skip(!has("vdb-samples/sphere.vdb"), "fixtures/vdb-samples/sphere.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(300_000);
  const r = await run(page, "?scene=sphere&twin=1&aniso=1&w=192&h=108");
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  const t = r.twin!;
  expect(t.gpuHits, detail).toBeGreaterThan(2_000);
  expect(t.iou, detail).toBeGreaterThanOrEqual(0.99);
  expect(t.normalP99Deg, detail).toBeLessThan(0.1);
  expect(t.analytic!.meanDeg, detail).toBeLessThan(0.1);
  expect(t.analytic!.maxDeg, detail).toBeLessThan(1);
  console.log(`[k3-levelset] sphere, anisotropic instance: analytic W^T normal mean ${t.analytic!.meanDeg.toFixed(3)}° max ${t.analytic!.maxDeg.toFixed(3)}°; twin IoU ${t.iou.toFixed(5)}`);
});

// Fail-before runs: K3_LS_MUTATE=<name> appends `&mutate=<name>` to the G13a / K3.7b queries below (each must fail).
const MUTATE_QUERY = process.env.K3_LS_MUTATE ? `&mutate=${process.env.K3_LS_MUTATE}` : "";

test("G13a physical material: transmission + clearcoat trace the same surface; the refraction starts at the hit", async ({ page }) => {
  test.skip(!has("vdb-samples/sphere.vdb"), "fixtures/vdb-samples/sphere.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(600_000);
  const r = await run(page, `?scene=sphere&twin=1&material=physical&refraction=1&vsStandard=1${MUTATE_QUERY}`);
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  const t = r.twin!;
  // The probe drawn by VDBLevelSetPhysicalMaterial (transmission 1, clearcoat 1): the same gates as the standard one.
  expect(t.gpuHits, detail).toBeGreaterThan(10_000);
  expect(t.iou, detail).toBeGreaterThanOrEqual(0.99);
  expect(t.hitMaxVoxels, detail).toBeLessThan(0.01);
  expect(t.normalP99Deg, detail).toBeLessThan(0.1);
  // Mask and depth: physical (transmission on, drawn in three's transparent pass) = standard, over a cutting plane.
  const v = r.vsStandard as { differing: number; maxDiff: number; planePixels: number };
  expect(v.planePixels, detail).toBeGreaterThan(1_000);
  // Only depth ties on the plane/surface intersection may differ (SwiftShader: 2 px): the physical material draws in
  // three's transparent pass, after the plane, so an equal depth keeps the later fragment. A depth from the box
  // (?mutate=physical-depth) differs on thousands.
  expect(v.differing, detail).toBeLessThanOrEqual(8);
  // The refracted backdrop uv vs refract() from the CPU twin's hit and normal (three's model starts at the box's
  // back face instead: tens of pixels off).
  const f = r.refraction as { pixels: number; meanPx: number; p99Px: number; maxPx: number };
  expect(f.pixels, detail).toBeGreaterThan(5_000);
  expect(f.p99Px, detail).toBeLessThan(0.5);
  expect(f.maxPx, detail).toBeLessThan(1.5);
  console.log(
    `[k3-levelset] G13a physical: twin IoU ${t.iou.toFixed(5)}, hit max ${t.hitMaxVoxels.toExponential(2)} voxels; vs standard ${v.differing} px differ (plane ${v.planePixels} px); refraction ${f.pixels} px mean ${f.meanPx.toFixed(3)} p99 ${f.p99Px.toFixed(3)} max ${f.maxPx.toFixed(3)} px`,
  );
});

for (const [mode, query] of [
  ["fp8", "quantize=fp8"],
  ["fpn", "quantize=fpn&tol=0.02"],
] as const) {
  test(`K3.7b level set: an ${mode} teapot traces as the CPU twin traces the same quantized image`, async ({ page }) => {
    test.skip(!has("vdb-samples/utahteapot.vdb"), "fixtures/vdb-samples/utahteapot.vdb missing (pnpm fixtures:vdb-samples)");
    test.setTimeout(600_000);
    const r = await run(page, `?scene=teapot&twin=1&${query}${MUTATE_QUERY}`);
    const detail = detailOf(r);
    expect(r.error, detail).toBeUndefined();
    expect(r.uncaptured, detail).toEqual([]);
    const q = r.quantized as { mode: string; imageBytes: number; floatImageBytes: number };
    expect(q.mode).toBe(mode);
    expect(q.imageBytes, detail).toBeLessThan(q.floatImageBytes);
    const t = r.twin!;
    expect(t.gpuHits, detail).toBeGreaterThan(5_000);
    expect(t.iou, detail).toBeGreaterThanOrEqual(0.99);
    expect(t.hitMaxVoxels, detail).toBeLessThan(0.01);
    expect(t.normalP99Deg, detail).toBeLessThan(0.01);
    console.log(
      `[k3-levelset] ${mode} teapot: ${(q.imageBytes / 1048576).toFixed(1)} MiB (float ${(q.floatImageBytes / 1048576).toFixed(1)} MiB), twin IoU ${t.iou.toFixed(5)}, hit max ${t.hitMaxVoxels.toExponential(2)} voxels, normal p99 ${t.normalP99Deg.toFixed(4)}°`,
    );
  });
}

test("K3.7b level set: a directional light's shadow map traces the surface (orthographic rays) onto a ground plane", async ({ page }) => {
  test.skip(!has("vdb-samples/sphere.vdb"), "fixtures/vdb-samples/sphere.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(600_000);
  const r = await run(page, `?scene=sphere&shadow=1${MUTATE_QUERY}`);
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  save(`k3-levelset-shadow${process.env.K3_LS_MUTATE ? `-${process.env.K3_LS_MUTATE}` : ""}`, r);
  const s = r.shadow as { planePixels: number; gpuShadow: number; cpuShadow: number; iou: number; mismatched: number; offEdge: number };
  expect(s.cpuShadow, detail).toBeGreaterThan(1_500);
  expect(s.iou, detail).toBeGreaterThanOrEqual(0.97);
  // Every disagreement sits on the CPU shadow's edge (map texels and pixel centres straddle it).
  expect(s.offEdge, detail).toBe(0);
  console.log(`[k3-levelset] shadow map: ${s.gpuShadow} GPU / ${s.cpuShadow} CPU shadow px of ${s.planePixels}, IoU ${s.iou.toFixed(4)}, ${s.mismatched} mismatched (${s.offEdge} off the edge)`);
});

test("K3.7b points: a .wvdb points grid streamed LOD by LOD into VDBPoints draws as the whole list at each count", async ({ page }) => {
  test.skip(!has("vdb-samples/sphere_points.vdb"), "fixtures/vdb-samples/sphere_points.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(300_000);
  const r = await run(page, `?scene=points-wvdb${MUTATE_QUERY}`);
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  const w = r.wvdb as { points: number; lods: number; events: number; drawn: number[]; identical: number; differing: number[] };
  expect(w.points, detail).toBe(9802);
  // LOD 0 (every 64th point), LODs 0-1 (every 8th), all: each drawn as it arrives.
  expect(w.drawn, detail).toContain(Math.ceil(9802 / 64));
  expect(w.drawn[w.drawn.length - 1], detail).toBe(9802);
  expect(w.identical, detail).toBe(w.events);
  save("k3-points-wvdb-stream", r);
  console.log(`[k3-levelset] points-wvdb: ${w.events} events, drawn ${w.drawn.join(" → ")}, ${w.identical}/${w.events} frames identical to the whole list`);
});

test("K3.7 points: sphere_points draws every point where it projects, nothing else; deterministic", async ({ page }) => {
  test.skip(!has("vdb-samples/sphere_points.vdb"), "fixtures/vdb-samples/sphere_points.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(300_000);
  const r = await run(page, "?scene=points");
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.inPageIdentical, detail).toBe(true);
  const p = r.points!;
  expect(p.totalCount, detail).toBe(9802);
  expect(p.count, detail).toBe(9802);
  expect(p.drawn, detail).toBe(9802);
  expect(p.centresTotal, detail).toBeGreaterThan(9000);
  expect(p.centresCovered, detail).toBe(p.centresTotal);
  expect(p.stray, detail).toBe(0);
  const again = await run(page, "?scene=points");
  expect(save("k3-points-sphere-run1", r).equals(save("k3-points-sphere-run2", again))).toBe(true);
  console.log(`[k3-levelset] points: ${p.count} drawn, ${p.centresCovered}/${p.centresTotal} centres covered, ${p.covered} px covered, ${p.stray} stray`);
});

test("K3.7 points: the Cd colour attribute reaches each point's pixels", async ({ page }) => {
  test.setTimeout(120_000);
  const r = await run(page, "?scene=points-color");
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.colors!.checked, detail).toBe(64);
  expect(r.colors!.bad, detail).toBe(0);
});

// His Step 7 parity scenes (capture-reference.mjs `step7-*`): 1280x720, slow on SwiftShader. Opt in with
// K3_FRAME_PARITY=1; scripts/k3/compare-step7.mjs compares the PNGs with fixtures/k3-reference/75e7345/.
const PARITY_SCENES = [
  "step7-utah-teapot-surface-normals",
  "step7-venus-statue-surface-normals",
  "step7-sphere-points-isolated",
  "step7-utah-teapot-surface-asis",
  "step7-venus-statue-surface-asis",
  "step7-utah-teapot-mesh-asis",
  "step7-sphere-points-asis",
];
for (const scene of PARITY_SCENES) {
  test(`K3.7 parity render: ${scene}`, async ({ page }) => {
    test.skip(process.env.K3_FRAME_PARITY !== "1", "set K3_FRAME_PARITY=1");
    test.skip(scene.includes("venus") && !has("k3/venusstatue.vdb"), "fixtures/k3/venusstatue.vdb missing (copy src/docs/examples/public/assets/venusstatue.vdb from his clone)");
    test.skip(!scene.includes("venus") && !has(scene.includes("points") ? "vdb-samples/sphere_points.vdb" : "vdb-samples/utahteapot.vdb"), "fixture missing");
    test.setTimeout(1_200_000);
    const r = await run(page, `?scene=${scene}`);
    expect(r.error, detailOf(r)).toBeUndefined();
    expect(r.uncaptured).toEqual([]);
    expect(r.inPageIdentical).toBe(true);
    save(scene, r);
    writeFileSync(path.join(OURS, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
  });
}
