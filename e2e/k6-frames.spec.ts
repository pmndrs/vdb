import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { diffHeatmap } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { decodePng, encodePng } from "../scripts/k3/png.mjs";

// K6.8: the core frame regression set. Every frame the core-vs-TSL A/Bs of K6.1–K6.Q1 rendered through `./three`'s core
// path (his K3 Steps 1, 3–6 and the environment scenes through k3-frame, the WDAS cloud through k3-timing, the K6.5 mesh
// shadow and emission cases through k3-frame-shadows, the K6.5b occluder gate scenes through k6-occluders), plus the CC0
// smoke (CI). Two modes, opt in:
//   K6_FRAMES_OUT=<dir>       render and write each frame as <dir>/<name>[-<field>].png (the "before" set of a change);
//   K6_FRAMES_BASELINE=<dir>  render and compare each frame against that set: identical, or within float noise
//                             (max ≤ K6_FRAMES_MAX/255, default 2; ≤ K6_FRAMES_SHARE of pixels over 1/255, default 0.1 %).
// Both together write and compare. A frame whose baseline file is missing is reported, not failed. `K6_FRAMES_ONLY=<substring>`
// narrows the list; frames whose fixture is missing skip. Report: <out or baseline>/../k6-frames-report.jsonl.
// His K3 scenes also go to artifacts/k6/ours-core/<his scenario>.png with the harness JSON (volumeRect, parityExpected) for
// the regression report against his captures (K6.8: this replaces the core-vs-TSL A/B rows of k6-core / k6-shadows):
//   node scripts/k3/compare.mjs --ours artifacts/k6/ours-core --reference fixtures/k3-reference/75e7345
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = process.env.K6_FRAMES_OUT ? path.resolve(process.env.K6_FRAMES_OUT) : null;
const BASE = process.env.K6_FRAMES_BASELINE ? path.resolve(process.env.K6_FRAMES_BASELINE) : null;
const MAX = Number(process.env.K6_FRAMES_MAX ?? 2);
const SHARE = Number(process.env.K6_FRAMES_SHARE ?? 0.001);
const ONLY = process.env.K6_FRAMES_ONLY ?? "";
// K6.11: `K6_FRAMES_QUERY=ao=0` appends a query parameter to every harness page (the harnesses read `?ao=`: the AO-off identity check)
const EXTRA_QUERY = process.env.K6_FRAMES_QUERY ?? "";

interface Frame {
  name: string;
  url: string;
  global: string;
  fields: string[];
  fixture?: string;
  timeout?: number;
  /** His capture's scenario name: the frame is also written to artifacts/k6/ours-core/ (compare.mjs). */
  capture?: string;
}

const BUNNY = "fixtures/k3/bunny_cloud.wvdb";
const SKY = "fixtures/k3/sky-midday-HDR.jpg";
const k3 = (name: string, query: string, fixture = BUNNY, capture?: string): Frame => ({ name, url: `/src/harness/k3-frame.html${query}`, global: "__VDB_K3_FRAME__", fields: ["rgba"], fixture, ...(capture ? { capture } : {}) });

const STEP1 = ["step1-lighting-volume", "step1-lights-directional", "step1-multiple-scattering-off", "step1-volume-dir-ms-off", "step1-volume-dir-ms-off-isolated", "step1-volume-dir-ms-on", "step1-volume-dir-ms-on-isolated"];
const STEP3: [string, string][] = [
  ["step3-default-dir-isolated", "dense"],
  ["step3-shadow-cascades-4-dir-isolated", "dense"],
  ["step3-shadow-grid-high-dir-isolated", "dense"],
  ["step3-two-bunnies-dir-isolated", "dense"],
  ["step3-default-dir-isolated", "sparse"],
  ["step3-two-bunnies-dir-isolated", "sparse"],
  ["step1-volume-dir-ms-on-isolated", "sparse"],
];
const STEP4 = [
  "step4-default-neural-dir-isolated",
  "step4-frame-phase-dual-lobe-dir-isolated",
  "step4-frame-phase-dual-lobe-back-g-high-dir-isolated",
  "step4-frame-phase-dual-lobe-back-weight-high-dir-isolated",
  "step4-frame-phase-g-exponent-dir-isolated",
  "step4-frame-phase-g-low-dir-isolated",
  "step4-frame-phase-isotropic-dir-isolated",
  "step4-frame-phase-lorenz-mie-dir-isolated",
  "step4-frame-phase-lorenz-mie-rgb-dir-isolated",
  "step4-lighting-volume-phase-g-low-dir-isolated",
  "step4-multiple-scattering-octaves-dir-isolated",
  "step4-multiple-scattering-two-sided-dir-isolated",
  "step4-octave-contribution-high-dir-isolated",
  "step4-octave-extinction-high-dir-isolated",
  "step4-octave-phase-low-dir-isolated",
  "step4-octaves-1-dir-isolated",
  "step4-octaves-8-dir-isolated",
];
const STEP5 = [
  "step5-spectral-dir-isolated",
  ...["cloud", "fog", "ice", "soot", "white-smoke", "dark-smoke", "haze", "dust"].map((m) => `step5-medium-${m}-dir-isolated`),
  "step5-angstrom-air-dir-isolated",
  "step5-lighting-volume-medium-dark-smoke-dir-isolated",
];
const STEP6 = [
  "step6-cast-shadow-shadow-only-dir-isolated",
  "step6-cast-shadow-obstacle-shadow-only-dir-isolated",
  "step6-cast-shadow-obstacle-cascades-4-shadow-only-dir-isolated",
  "step6-cast-shadow-dir-isolated",
  "step6-cast-shadow-obstacle-dir-isolated",
  "step6-cast-shadow-obstacle-cascades-4-dir-isolated",
  "step6-cast-shadow-obstacle-coarse-dir-isolated",
  "step6-cast-shadow-two-bunnies-dir-isolated",
  "step6-explosion-emission-env-isolated",
  "step6-explosion-emission-cascades-4-env-isolated",
  "step6-explosion-emission-probes-8-env-isolated",
];
const ENV: [string, string][] = [
  ["step3-environment-sky-midday-env-isolated", ""],
  ["step3-environment-sky-midday-env-isolated", "&source=sparse"],
  ["step5-lighting-volume-sky-midday-env-isolated", ""],
  ["step5-lighting-volume-medium-cloud-sky-midday-env-isolated", ""],
  ["step3-environment-sky-midday-env-isolated", "&ms=false"],
  ["step3-environment-sky-midday-env-isolated", "&envMode=spherical-harmonics"],
  ["step3-environment-sky-midday-env-isolated", "&envMode=outward"],
  ["step5-lighting-volume-sky-midday-env-isolated", "&envMode=outward"],
  ["step3-environment-sky-midday-env-isolated", "&envPmrem=1"],
];
const slug = (s: string) => s.replace(/[&=?]+/g, "-").replace(/^-+|-+$/g, "");

const FRAMES: Frame[] = [
  // CC0 (CI): k3-frame's own smoke scene, 'volume' and 'frame' lighting, dense and sparse
  k3("smoke-volume", "?w=320&h=192&froxels=160x96x24&lighting=volume", ""),
  k3("smoke-frame", "?w=320&h=192&froxels=160x96x24&lighting=frame", ""),
  k3("smoke-frame-sparse", "?w=320&h=192&froxels=160x96x24&lighting=frame&source=sparse", ""),
  ...STEP1.map((s) => k3(`${s}-volume`, `?scene=${s}&lighting=volume`, BUNNY, s)),
  ...STEP3.map(([s, src]) => k3(`${s}-${src}`, `?scene=${s}&source=${src}`, BUNNY, src === "dense" ? s : undefined)),
  ...STEP4.map((s) => k3(s, `?scene=${s}`, BUNNY, s)),
  ...STEP5.map((s) => k3(s, `?scene=${s}`, BUNNY, s)),
  // his renderer has no mesh → volume shadows (K6.5b): his Step 6 scenes with occlusion off, and two with it on (report)
  ...STEP6.map((s) => k3(s, `?scene=${s}&occlusion=0`, BUNNY, s)),
  ...STEP6.slice(3, 5).map((s) => k3(`${s}-occlusion`, `?scene=${s}`)),
  ...ENV.map(([s, extra]) => k3(slug(`${s}${extra}`), `?scene=${s}${extra}`, SKY, extra ? undefined : s)),
  ...(["volume", "frame"] as const).map((lighting) => ({
    name: `wdas-sparse-${lighting}`,
    url: `/src/harness/k3-timing.html?asset=wdas&source=sparse&lod=pyramid&hdda=1&lighting=${lighting}&w=1280&h=720&froxels=512x512x24&image=1`,
    global: "__VDB_K3_TIMING__",
    fields: ["rgba"],
    fixture: "fixtures/wdas/wdas_cloud_quarter_fp8.nvdb",
  })),
  // K6.11b: ambient-class light (hemisphere, ambient, light probe) so the shadowed-ambient default (D35 (1)) has a
  // before/after gate beyond the 'outward' environment frame: k3-frame's CC0 smoke with his hemisphere + ambient lights
  // (CI), and the K6.Q1 report scenes through k6-scene (the probe on the WDAS cloud, the hemisphere on the smoke and the
  // absorbing bunny; local fixtures). `K6_FRAMES_QUERY=ao=0` renders them unshadowed.
  k3("smoke-volume-hemi", "?w=320&h=192&froxels=160x96x24&lighting=volume&lights=dir-hemi", ""),
  k3("smoke-frame-hemi", "?w=320&h=192&froxels=160x96x24&lighting=frame&lights=dir-hemi", ""),
  ...(
    [
      ["wdas-probe", "frame", "fixtures/wdas/wdas_cloud_quarter_fp8.nvdb"],
      ["wdas-probe", "volume", "fixtures/wdas/wdas_cloud_quarter_fp8.nvdb"],
      ["smoke-hemi", "frame", "fixtures/vdb-samples/smoke.vdb"],
      ["bunny-absorbing-hemi", "frame", BUNNY],
      ["bunny-absorbing-hemi", "volume", BUNNY],
      ["bunny-absorbing-ambient", "frame", BUNNY],
    ] as const
  ).map(([scene, lighting, fixture]) => ({
    name: `scene-${scene}-${lighting}`,
    url: `/src/harness/k6-scene.html?scene=${scene}&lighting=${lighting}&w=640&h=360`,
    global: "__VDB_K6_SCENE__",
    fields: ["rgba"],
    fixture,
  })),
  ...["?case=shadow", "?case=shadow&cascades=4", "?case=shadow&light=spot", "?case=shadow&source=sparse", "?case=shadow&lighting=frame", "?case=emission&source=dense", "?case=emission&source=sparse"].map((q) => ({
    name: `shadows-${slug(q)}`,
    url: `/src/harness/k3-frame-shadows.html${q}`,
    global: "__VDB_K3_SHADOWS__",
    fields: ["rgba", "rgbaOff"],
  })),
  ...[
    ["sun", "frame"],
    ["sun", "volume"],
    ["spot", "frame"],
    ["point", "frame"],
    ["point", "volume"],
  ].map(([light, lighting]) => ({
    name: `occluders-${light}-${lighting}`,
    url: `/src/harness/k6-occluders.html?light=${light}&lighting=${lighting}`,
    global: "__VDB_K6_OCCLUDERS__",
    fields: ["on", "off", "moved"],
  })),
].filter((f) => !ONLY || f.name.includes(ONLY));

interface Row {
  name: string;
  field: string;
  status: "written" | "identical" | "noise" | "differs" | "no-baseline";
  maxDiff?: number;
  over1?: number;
  share?: number;
  pixels?: number;
}
const rows: Row[] = [];

async function render(page: Page, f: Frame): Promise<Record<string, unknown>> {
  await page.goto(EXTRA_QUERY ? `${f.url}${f.url.includes("?") ? "&" : "?"}${EXTRA_QUERY}` : f.url);
  await page.waitForFunction((g) => (window as unknown as Record<string, { done?: boolean } | undefined>)[g]?.done === true, f.global, { timeout: f.timeout ?? 1_800_000 });
  return (await page.evaluate((g) => (window as unknown as Record<string, unknown>)[g], f.global)) as Record<string, unknown>;
}

function compare(a: Uint8Array, b: Uint8Array): { maxDiff: number; over1: number; pixels: number } {
  let maxDiff = 0;
  let over1 = 0;
  for (let i = 0; i < a.length; i += 4) {
    let m = 0;
    for (let c = 0; c < 3; c++) m = Math.max(m, Math.abs(a[i + c]! - b[i + c]!));
    maxDiff = Math.max(maxDiff, m);
    if (m > 1) over1++;
  }
  return { maxDiff, over1, pixels: a.length / 4 };
}

test.describe.configure({ mode: "parallel" });

for (const f of FRAMES) {
  test(`K6.8 core frame: ${f.name}`, async ({ page }) => {
    test.skip(!OUT && !BASE, "set K6_FRAMES_OUT=<dir> and/or K6_FRAMES_BASELINE=<dir>");
    test.skip(!!f.fixture && !existsSync(path.join(repoRoot, f.fixture)), `${f.fixture} missing`);
    test.setTimeout(2_400_000);
    const r = await render(page, f);
    const detail = JSON.stringify({ ...r, rgba: undefined, rgbaOff: undefined, on: undefined, off: undefined, moved: undefined, settled: undefined, noCaster: undefined, mask: undefined, coverage: undefined }).slice(0, 4000);
    expect(r.error, detail).toBeUndefined();
    const width = (r.width as number) ?? 0;
    const height = (r.height as number) ?? 0;
    const failures: string[] = [];
    for (const field of f.fields) {
      const b64 = r[field] as string | undefined;
      expect(b64, `${f.name}: no ${field} (${detail})`).toBeTruthy();
      const data = new Uint8Array(Buffer.from(b64!, "base64"));
      const file = `${f.name}${field === "rgba" ? "" : `-${field}`}.png`;
      if (OUT) {
        mkdirSync(OUT, { recursive: true });
        writeFileSync(path.join(OUT, file), Buffer.from(encodePng({ width, height, data }) as Uint8Array));
      }
      if (!BASE) {
        rows.push({ name: f.name, field, status: "written" });
        continue;
      }
      const basePath = path.join(BASE, file);
      if (!existsSync(basePath)) {
        rows.push({ name: f.name, field, status: "no-baseline" });
        continue;
      }
      const before = decodePng(readFileSync(basePath)) as { width: number; height: number; data: Uint8Array };
      expect([before.width, before.height], `${file} size`).toEqual([width, height]);
      const c = compare(data, before.data);
      const share = c.over1 / c.pixels;
      const status = c.maxDiff === 0 ? "identical" : c.maxDiff <= MAX && share <= SHARE ? "noise" : "differs";
      rows.push({ name: f.name, field, status, ...c, share });
      console.log(`[k6-frames] ${file}: ${status}, max ${c.maxDiff}/255, ${(share * 100).toFixed(4)} % px > 1/255`);
      if (status === "differs") {
        const diffDir = path.join(path.dirname(BASE), "k6-frames-diff");
        mkdirSync(diffDir, { recursive: true });
        writeFileSync(path.join(diffDir, file), Buffer.from(encodePng(diffHeatmap({ width, height, data }, before, { threshold: 1, fullScale: 16 })) as Uint8Array));
        failures.push(`${file}: max ${c.maxDiff}/255, ${(share * 100).toFixed(4)} % px > 1/255`);
      }
    }
    if (f.capture) {
      const dir = path.join(repoRoot, "artifacts", "k6", "ours-core");
      mkdirSync(dir, { recursive: true });
      writeFileSync(path.join(dir, `${f.capture}.png`), Buffer.from(encodePng({ width, height, data: new Uint8Array(Buffer.from(r.rgba as string, "base64")) }) as Uint8Array));
      writeFileSync(path.join(dir, `${f.capture}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
    }
    const report = path.join(path.dirname(BASE ?? OUT!), "k6-frames-report.jsonl");
    for (const row of rows.splice(0)) writeFileSync(report, `${JSON.stringify(row)}\n`, { flag: "a" });
    expect(failures).toEqual([]);
  });
}
