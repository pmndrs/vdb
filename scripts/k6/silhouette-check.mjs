#!/usr/bin/env node
// K6.3a: where do the core's pixels over his strict bar sit? For each K3 scene: the core and TSL frames at his size
// (1280 × 720, his 512 × 512 × 24 froxels) vs his capture, every pixel > 8/255 inside the volume rect classified as
//   - silhouette band: within `--band` px (default 3) of a faint pixel of his capture (within `--bg` = 16/255 of the
//     background: the cloud's fringe is several pixels of low coverage, so "exactly background" sits further out);
//   - inner edge: elsewhere, where his capture's local contrast (max channel step to a pixel within 2 px) ≥ `--edge`;
//   - smooth interior: the rest.
// and the edge orientation at each pixel (his capture's |d/dx| vs |d/dy|): his froxels are 2.5 px wide and 1.4 px tall
// at 1280 × 720, so an x, y filter of the froxel grid moves vertical edges more.
// Then the mechanism test of V80 (x, y filtering of the integrated prefix vs his per-slice composite): core vs TSL at
// 640 × 360 with his froxel ratio (256 × 256 × 24) and with one froxel per pixel (640 × 360 × 24, no x, y filtering).
// If the differences are V80's they vanish with one froxel per pixel; a lighting or MS defect would not.
//
//   node scripts/k6/silhouette-check.mjs --scenes=a,b [--port=5317] [--base=http://127.0.0.1:5317] [--band=3] [--edge=24] [--bg=16]
//
// SwiftShader (bundled Chromium; his captures reproduce there: TSL = his to 1/255). Quality only, no timings.
// Writes artifacts/k6-3a/silhouette/<scene>/ (frames, his capture, class masks, heat maps) and summary.json.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { decodePng, encodePng } from "../k3/png.mjs";
import { diffHeatmap } from "../k3/metrics.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.join(ROOT, "artifacts", "k6-3a", "silhouette");
const CAPTURES = path.join(ROOT, "fixtures", "k3-reference", "75e7345");
const arg = (name, fallback) => {
  const hit = process.argv.find((a) => a === `--${name}` || a.startsWith(`--${name}=`));
  if (!hit) return fallback;
  return hit.includes("=") ? hit.slice(hit.indexOf("=") + 1) : true;
};
const SCENES = String(arg("scenes", "step4-lighting-volume-phase-g-low-dir-isolated,step4-octave-extinction-high-dir-isolated,step4-octaves-1-dir-isolated")).split(",");
const PORT = Number(arg("port", 5317));
const BAND = Number(arg("band", 3));
const EDGE = Number(arg("edge", 24));
const BG_TOL = Number(arg("bg", 16));
let BASE = arg("base", "");

async function startServer() {
  if (BASE) return null;
  BASE = `http://127.0.0.1:${PORT}`;
  const child = spawn("pnpm", ["--filter", "@webvdb/examples", "dev", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"], { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] });
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(BASE)).ok) return child;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  child.kill();
  throw new Error(`Vite did not start on ${BASE}`);
}

async function render(browser, query) {
  const p = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await p.goto(`${BASE}/src/harness/k3-frame.html?${query}`);
  await p.waitForFunction(() => window.__VDB_K3_FRAME__?.done === true, undefined, { timeout: 900_000, polling: 500 });
  const r = await p.evaluate(() => window.__VDB_K3_FRAME__);
  await p.close();
  if (r.error) throw new Error(`k3-frame?${query}: ${r.error}`);
  if ((query.includes("backend=core") ? "core" : "tsl") !== (r.backend ?? "tsl")) throw new Error(`k3-frame?${query} drew ${r.backend}`);
  return { width: r.width, height: r.height, data: new Uint8Array(Buffer.from(r.rgba, "base64")), rect: r.volumeRect, adapter: r.adapter };
}

const maxDiff = (a, b, i) => Math.max(Math.abs(a.data[i * 4] - b.data[i * 4]), Math.abs(a.data[i * 4 + 1] - b.data[i * 4 + 1]), Math.abs(a.data[i * 4 + 2] - b.data[i * 4 + 2]));

/** City-block distance (two passes) from every pixel to the nearest pixel where `seed` is set. */
function distance(seed, w, h) {
  const d = new Float64Array(w * h).fill(1e9);
  for (let i = 0; i < w * h; i++) if (seed[i]) d[i] = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (x > 0) d[i] = Math.min(d[i], d[i - 1] + 1);
    if (y > 0) d[i] = Math.min(d[i], d[i - w] + 1);
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x;
    if (x < w - 1) d[i] = Math.min(d[i], d[i + 1] + 1);
    if (y < h - 1) d[i] = Math.min(d[i], d[i + w] + 1);
  }
  return d;
}

/** Pixels > `t` of a vs b inside `rect` (all pixels when null): count, fraction, max, the list. */
function over(a, b, rect, t = 8) {
  const [x0, y0, x1, y1] = rect ?? [0, 0, a.width, a.height];
  const list = [];
  let max = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const i = y * a.width + x;
    const m = maxDiff(a, b, i);
    max = Math.max(max, m);
    if (m > t) list.push(i);
  }
  const n = (x1 - x0) * (y1 - y0);
  return { count: list.length, fraction: list.length / n, max, list };
}

function classify(his, core, rect) {
  const { width: w, height: h } = his;
  const bg = [his.data[0], his.data[1], his.data[2]];
  const isBg = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) isBg[i] = Math.max(...[0, 1, 2].map((c) => Math.abs(his.data[i * 4 + c] - bg[c]))) <= BG_TOL ? 1 : 0;
  const dist = distance(isBg, w, h);
  const contrast = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const xx = x + dx;
      const yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      for (let c = 0; c < 3; c++) m = Math.max(m, Math.abs(his.data[(y * w + x) * 4 + c] - his.data[(yy * w + xx) * 4 + c]));
    }
    contrast[y * w + x] = m;
  }
  const fails = over(core, his, rect);
  const cls = { outer: 0, inner: 0, smooth: 0 };
  const mags = { outer: [], inner: [], smooth: [] };
  const mask = { width: w, height: h, data: new Uint8Array(w * h * 4) };
  for (let i = 0; i < w * h; i++) {
    const g = Math.round((his.data[i * 4] + his.data[i * 4 + 1] + his.data[i * 4 + 2]) / 6);
    mask.data.set([g, g, g, 255], i * 4);
  }
  const colour = { outer: [255, 0, 0], inner: [255, 220, 0], smooth: [255, 0, 255] };
  let brighter = 0;
  let vertical = 0;
  const dists = [];
  const lum = (i) => his.data[i * 4] + his.data[i * 4 + 1] + his.data[i * 4 + 2];
  for (const i of fails.list) {
    const k = dist[i] <= BAND ? "outer" : contrast[i] >= EDGE ? "inner" : "smooth";
    cls[k]++;
    mags[k].push(maxDiff(core, his, i));
    dists.push(dist[i]);
    mask.data.set([...colour[k], 255], i * 4);
    const yc = core.data[i * 4] + core.data[i * 4 + 1] + core.data[i * 4 + 2];
    const yh = his.data[i * 4] + his.data[i * 4 + 1] + his.data[i * 4 + 2];
    if (yc > yh) brighter++;
    const x = i % w;
    const y = Math.floor(i / w);
    const gx = x > 0 && x < w - 1 ? Math.abs(lum(i + 1) - lum(i - 1)) : 0;
    const gy = y > 0 && y < h - 1 ? Math.abs(lum(i + w) - lum(i - w)) : 0;
    if (gx > gy) vertical++;
  }
  // context: how much of the rect is band / inner edge / smooth
  const [x0, y0, x1, y1] = rect;
  const area = { outer: 0, inner: 0, smooth: 0, background: 0 };
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const i = y * w + x;
    if (isBg[i]) area.background++;
    else area[dist[i] <= BAND ? "outer" : contrast[i] >= EDGE ? "inner" : "smooth"]++;
  }
  const stat = (v) => (v.length ? { n: v.length, median: v.sort((a, b) => a - b)[Math.floor(v.length / 2)], max: v[v.length - 1] } : { n: 0 });
  dists.sort((a, b) => a - b);
  return {
    fails: { count: fails.count, fraction: fails.fraction, max: fails.max },
    classes: cls,
    magnitudes: { outer: stat(mags.outer), inner: stat(mags.inner), smooth: stat(mags.smooth) },
    distanceToBackground: { median: dists[Math.floor(dists.length / 2)] ?? null, p90: dists[Math.floor(dists.length * 0.9)] ?? null, max: dists[dists.length - 1] ?? null },
    coreBrighter: brighter,
    verticalEdges: vertical,
    rectArea: area,
    mask,
  };
}

const server = await startServer();
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-unsafe-swiftshader"] });
const summary = { at: new Date().toISOString(), band: BAND, edge: EDGE, backgroundTolerance: BG_TOL, scenes: [] };
try {
  for (const scene of SCENES) {
    const dir = path.join(OUT, scene);
    mkdirSync(dir, { recursive: true });
    const capture = path.join(CAPTURES, `${scene}.png`);
    if (!existsSync(capture)) throw new Error(`${capture} missing`);
    const his = decodePng(readFileSync(capture));
    console.log(`== ${scene}`);
    const core = await render(browser, `scene=${scene}&backend=core`);
    const tsl = await render(browser, `scene=${scene}&backend=tsl`);
    const rect = core.rect;
    const c = classify(his, core, rect);
    const t = over(tsl, his, rect);
    const ct = over(core, tsl, rect);
    for (const [name, img] of Object.entries({ core, tsl, his })) writeFileSync(path.join(dir, `${name}.png`), encodePng(img));
    writeFileSync(path.join(dir, "core-vs-his-classes.png"), encodePng(c.mask));
    writeFileSync(path.join(dir, "core-vs-his-diff.png"), encodePng(diffHeatmap(core, his, { threshold: 8, fullScale: 32 })));
    writeFileSync(path.join(dir, "core-vs-tsl-diff.png"), encodePng(diffHeatmap(core, tsl, { threshold: 8, fullScale: 32 })));
    // Mechanism: x, y filtering on (his froxel ratio) vs off (one froxel per pixel), core vs TSL, 640 × 360.
    const mech = {};
    for (const [label, froxels] of [["his ratio 256x256x24", "256x256x24"], ["one froxel per pixel 640x360x24", "640x360x24"]]) {
      const q = `scene=${scene}&w=640&h=360&froxels=${froxels}`;
      const a = await render(browser, `${q}&backend=core`);
      const b = await render(browser, `${q}&backend=tsl`);
      const o8 = over(a, b, a.rect, 8);
      const o4 = over(a, b, a.rect, 4);
      mech[label] = { rect: a.rect, over8: o8.count, over8Fraction: o8.fraction, over4: o4.count, over4Fraction: o4.fraction, max: o8.max };
      writeFileSync(path.join(dir, `core-vs-tsl-640-${froxels}.png`), encodePng(diffHeatmap(a, b, { threshold: 8, fullScale: 32 })));
    }
    const entry = {
      scene,
      adapter: core.adapter,
      rect,
      coreVsHis: { ...c, mask: undefined },
      tslVsHis: { count: t.count, fraction: t.fraction, max: t.max },
      coreVsTsl: { count: ct.count, fraction: ct.fraction, max: ct.max },
      mechanism: mech,
    };
    summary.scenes.push(entry);
    const pc = (x) => `${(x * 100).toFixed(3)} %`;
    console.log(`  core vs his: ${c.fails.count} px > 8/255 (${pc(c.fails.fraction)} of the rect, max ${c.fails.max}); silhouette band ${c.classes.outer}, inner edges ${c.classes.inner}, smooth interior ${c.classes.smooth}; core brighter at ${c.coreBrighter}; on mostly vertical edges ${c.verticalEdges}`);
    console.log(`  distance to background: median ${c.distanceToBackground.median}, p90 ${c.distanceToBackground.p90}, max ${c.distanceToBackground.max}; magnitudes ${JSON.stringify(c.magnitudes)}`);
    console.log(`  TSL vs his ${t.count} px (max ${t.max}); core vs TSL ${ct.count} px > 8/255 (max ${ct.max})`);
    for (const [label, m] of Object.entries(mech)) console.log(`  640x360 ${label}: core vs TSL ${m.over8} px > 8/255 (${pc(m.over8Fraction)}), ${m.over4} > 4/255, max ${m.max}`);
  }
} finally {
  await browser.close();
  server?.kill();
}
writeFileSync(path.join(OUT, "summary.json"), JSON.stringify(summary, null, 2));
console.log(`wrote ${path.relative(ROOT, path.join(OUT, "summary.json"))}`);
