import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K3.6 / K6.5 / K6.8: volume shadows on meshes and the emission light through `./three` (examples/src/harness/
// k3-frame-shadows.ts), the core's maps and probes received by three. The core kernels have their twins in
// e2e/k6-core.spec.ts (`?probe=shadow`, `?probe=emission`); these frames are in the K6.8 regression set
// (e2e/k6-frames.spec.ts), and his Step 6 scenes too. Gates here: the shadows darken the ground (and nothing else moves
// when castShadows goes off but the ground), every case's map is drawn with the expected layout, the sparse source and
// 'frame' lighting give the same shadows as dense 'volume', the emission light lights the ground (black without it),
// two page loads are byte-identical, no errors or warnings. (The last core-vs-TSL numbers: docs/archive/handoffs/K6-5.md,
// K6-5b.md, K6-8.md.) PNGs: artifacts/k6/shadows/.
const PAGE_URL = "/src/harness/k3-frame-shadows.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(repoRoot, "artifacts", "k6", "shadows");

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  width?: number;
  height?: number;
  rgba?: string;
  rgbaOff?: string;
  inPageIdentical?: boolean;
  warnings: string[];
  uncaptured: string[];
  entry?: { size: number; cascades: number; drawn: number; waiting: number } | null;
  emitters?: number;
  shadowStats?: { groundPixels: number; covered: number; shadowedPixels: number; meanDarkening: number };
  lightOff?: { groundPixels: number; nonBlack: number };
  backend?: { requested: string; core: boolean; maps: number; probes: boolean };
  coverage?: string;
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_SHADOWS__?: { done: boolean } }).__VDB_K3_SHADOWS__?.done === true, undefined, { timeout: 380_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_SHADOWS__?: unknown }).__VDB_K3_SHADOWS__)) as Result;
}

const bytes = (b64: string) => new Uint8Array(Buffer.from(b64, "base64"));
const image = (r: Result, b64: string) => ({ width: r.width!, height: r.height!, data: bytes(b64) });
const detailOf = (r: Result) => JSON.stringify({ ...r, rgba: r.rgba ? `${r.rgba.length} b64` : undefined, rgbaOff: r.rgbaOff ? `${r.rgbaOff.length} b64` : undefined, coverage: undefined }, null, 2);

function save(name: string, r: Result, b64: string): Buffer {
  const png = Buffer.from(encodePng(image(r, b64)) as Uint8Array);
  mkdirSync(OUT, { recursive: true });
  writeFileSync(path.join(OUT, `${name}.png`), png);
  return png;
}

/** Pixels outside the volume's dilated coverage where on and off differ (the term on meshes), and the largest change. */
function term(r: Result): { pixels: number; affected: number; maxDelta: number } {
  const on = bytes(r.rgba!);
  const off = bytes(r.rgbaOff!);
  const cov = bytes(r.coverage!);
  let pixels = 0;
  let affected = 0;
  let maxDelta = 0;
  for (let i = 0; i < cov.length; i++) {
    if (cov[i]) continue;
    pixels++;
    let d = 0;
    for (let c = 0; c < 3; c++) d = Math.max(d, Math.abs(on[i * 4 + c]! - off[i * 4 + c]!));
    if (d > 0) affected++;
    maxDelta = Math.max(maxDelta, d);
  }
  return { pixels, affected, maxDelta };
}

function expectClean(r: Result): void {
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.warnings, detail).toEqual([]);
  expect(r.inPageIdentical, detail).toBe(true);
}

const SHADOW_CASES: { name: string; query: string; cascades: number }[] = [
  { name: "sun, dense", query: "?case=shadow", cascades: 1 },
  { name: "sun, 4 cascades", query: "?case=shadow&cascades=4", cascades: -1 },
  { name: "spot", query: "?case=shadow&light=spot", cascades: 0 },
  { name: "sun, sparse", query: "?case=shadow&source=sparse", cascades: 1 },
  { name: "sun, 'frame' lighting", query: "?case=shadow&lighting=frame", cascades: 1 },
];

for (const c of SHADOW_CASES) {
  test(`K6.5 volume shadows on meshes through ./three (${c.name}): the core's map darkens the ground`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, c.query);
    expectClean(r);
    expect(r.backend, detailOf(r)).toMatchObject({ core: true, maps: 1 });
    const slug = c.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
    save(`shadows-${slug}`, r, r.rgba!);
    save(`shadows-${slug}-off`, r, r.rgbaOff!);
    const t = term(r);
    console.log(`[k6-shadows] ${c.name} on ${r.adapter}: ${JSON.stringify(r.shadowStats)}; term ${JSON.stringify(t)}; entry ${JSON.stringify(r.entry)}`);
    expect(r.shadowStats!.shadowedPixels, detailOf(r)).toBeGreaterThan(100);
    expect(r.shadowStats!.meanDarkening, detailOf(r)).toBeGreaterThan(0);
    expect(t.affected, JSON.stringify(t)).toBeGreaterThan(100);
    expect(r.entry, detailOf(r)).toMatchObject({ size: 512 });
    if (c.cascades > 0) expect(r.entry!.cascades, detailOf(r)).toBe(c.cascades);
    if (c.cascades < 0) expect(r.entry!.cascades, detailOf(r)).toBeGreaterThanOrEqual(1);
    expect(errors).toEqual([]);
  });
}

test("K6.5 volume shadows: the sparse source and 'frame' lighting shadow the ground as dense 'volume'; two loads agree", async ({ page }) => {
  test.setTimeout(1_200_000);
  const dense = await run(page, "?case=shadow");
  expectClean(dense);
  const again = await run(page, "?case=shadow");
  expectClean(again);
  expect(save("shadows-dense-run2", again, again.rgba!).equals(save("shadows-dense-run1", dense, dense.rgba!)), "two page loads byte-identical").toBe(true);
  for (const q of ["&source=sparse", "&lighting=frame"]) {
    const other = await run(page, `?case=shadow${q}`);
    expectClean(other);
    // the ground's shadow (castShadows on − off) from the same shadow grid: same pixels within a few %
    const a = dense.shadowStats!.shadowedPixels;
    const b = other.shadowStats!.shadowedPixels;
    console.log(`[k6-shadows] dense vs ${q}: shadowed ${a} vs ${b}; frames ${compareImages(image(dense, dense.rgbaOff!), image(other, other.rgbaOff!)).psnr.toFixed(2)} dB (shadows off)`);
    expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(a * 0.05);
  }
});

test("K6.5 the emission light (explosion frame 10 and a smooth blob, dense and sparse): lights the ground; black without it; two loads agree", async ({ page }) => {
  test.setTimeout(1_200_000);
  const errors = collectErrors(page);
  let first: Buffer | null = null;
  for (const [emitter, source] of [
    ["explosion", "dense"],
    ["explosion", "sparse"],
    ["blob", "dense"],
  ] as const) {
    const r = await run(page, `?case=emission&emitter=${emitter}&source=${source}`);
    expectClean(r);
    const detail = detailOf(r);
    expect(r.emitters, detail).toBe(1);
    expect(r.backend, detail).toMatchObject({ core: true, probes: true });
    expect(r.lightOff!.groundPixels, detail).toBeGreaterThan(2000);
    expect(r.lightOff!.nonBlack, detail).toBe(0);
    const t = term(r);
    console.log(`[k6-emission] ${emitter} ${source} on ${r.adapter}: term ${JSON.stringify(t)}`);
    expect(t.affected, JSON.stringify(t)).toBeGreaterThan(1000);
    const png = save(`emission-${emitter}-${source}`, r, r.rgba!);
    if (emitter === "explosion" && source === "dense") first = png;
  }
  const again = await run(page, "?case=emission&emitter=explosion&source=dense");
  expectClean(again);
  expect(save("emission-explosion-dense-run2", again, again.rgba!).equals(first!), "two page loads byte-identical").toBe(true);
  expect(errors).toEqual([]);
});
