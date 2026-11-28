// Issue #2 step 2 (K6.S2): the froxel renderer's density filter ('trilinear' vs 'stochastic', one voxel per slice
// sub-step) on installed Chrome headless (the real GPU; the adapter is printed), against a running examples Vite server
// (preferably on a frozen copy: Vite serves sources live). Usage:
//   node scripts/k6/stochastic-froxel-ab.mjs http://127.0.0.1:<port> [--assets=wdas,smoke] [--lods=pyramid,none]
//     [--lighting=frame] [--rounds=4] [--frames=30] [--warmup=10] [--check=32] [--size=hd|mid|small] [--extra=a=b&c=d]
//     [--skip-timing] [--skip-quality]
// Per asset and lod:
// - timing (k3-timing, paired ABBA rounds): the core update's GPU ms (`froxelsMs`: in a still 'frame' scene the cached
//   grid passes are skipped, so it is the gather + the resolve) and the wall frame; the ratio trilinear / stochastic per
//   round (> 1: stochastic is faster);
// - quality (k3-timing `?densityCheck=N`): the deterministic trilinear frame against N stochastic frames without temporal
//   reprojection (their mean: the bias, Jensen + the composite, as an image z and a relative luminance bias) and with it
//   (the last frame: what a viewer sees; PSNR in display bytes, as one stochastic frame).
// Timings are provisional unless the machine is quiet (D25): note the load and the device.
import { chromium } from "@playwright/test";
import os from "node:os";

const [base, ...rest] = process.argv.slice(2);
if (!base) throw new Error("usage: stochastic-froxel-ab.mjs <base url> [--assets=] [--lods=] [--rounds=]");
const opt = Object.fromEntries(rest.map((a) => a.replace(/^--/, "").split("=")).map(([k, ...v]) => [k, v.length ? v.join("=") : "1"]));
const assets = (opt.assets ?? "wdas,smoke").split(",");
const lods = (opt.lods ?? "pyramid,none").split(",");
const lighting = opt.lighting ?? "frame";
const rounds = Number(opt.rounds ?? 4);
const frames = Number(opt.frames ?? 30);
const warmup = Number(opt.warmup ?? 10);
const check = Number(opt.check ?? 32);
const size = opt.size === "small" ? "w=320&h=180&froxels=64x64x16&substeps=16&shadowSteps=16" : opt.size === "mid" ? "w=960&h=540&froxels=256x256x24&substeps=32&shadowSteps=32" : "w=1920&h=1080&froxels=512x512x24&substeps=32&shadowSteps=32";
const extra = opt.extra ? `&${opt.extra}` : "";

const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features"] });
const page = await browser.newPage();
const logs = [];
page.on("console", (m) => {
  if (m.type() === "error") logs.push(m.text());
});
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));

async function load(query) {
  await page.goto(`${base}/src/harness/k3-timing.html?${query}`);
  await page.waitForFunction(() => window.__VDB_K3_TIMING__?.done === true, undefined, { timeout: 900_000 });
  const r = await page.evaluate(() => window.__VDB_K3_TIMING__);
  if (r.error) throw new Error(`${query}: ${r.error}`);
  return r;
}

const load1 = () => os.loadavg().map((x) => x.toFixed(2)).join(" ");
const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)] : null;
};
console.log(JSON.stringify({ host: `${os.cpus()[0]?.model} × ${os.cpus().length}`, loadBefore: load1(), lighting, rounds, frames, check, size: opt.size ?? "hd" }));
for (const asset of assets)
  for (const lod of lods) {
    const q = `asset=${asset}&source=sparse&lod=${lod}&lighting=${lighting}&${size}${extra}`;
    const out = { asset, lod, lighting };
    if (!opt["skip-timing"]) {
      const runs = { trilinear: [], stochastic: [] };
      for (let k = 0; k < rounds; k++) {
        const order = k % 2 === 0 ? ["trilinear", "stochastic"] : ["stochastic", "trilinear"];
        for (const d of order) runs[d].push(await load(`${q}&density=${d}&frames=${frames}&warmup=${warmup}`));
      }
      const ms = (r) => r.froxelsMs;
      out.adapter = runs.trilinear[0].adapter;
      out.froxelsMs = { trilinear: runs.trilinear.map((r) => +ms(r)?.toFixed(3)), stochastic: runs.stochastic.map((r) => +ms(r)?.toFixed(3)) };
      out.wallMs = { trilinear: runs.trilinear.map((r) => +r.frameWallMs.toFixed(2)), stochastic: runs.stochastic.map((r) => +r.frameWallMs.toFixed(2)) };
      out.speedup = runs.trilinear.map((r, k) => +(ms(r) / ms(runs.stochastic[k])).toFixed(3));
      out.speedupMedian = median(out.speedup);
    }
    if (!opt["skip-quality"]) {
      const r = await load(`${q}&densityCheck=${check}`);
      out.adapter ??= r.adapter;
      out.quality = r.densityCheck;
    }
    out.load = load1();
    console.log(JSON.stringify(out));
  }
if (logs.length) console.log("console errors:", JSON.stringify(logs.slice(0, 10)));
await browser.close();
