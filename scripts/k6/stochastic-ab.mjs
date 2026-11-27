// Issue #2: paired A/B of the reference renderer's density filter ('trilinear' vs 'stochastic', one voxel per lookup)
// at equal time, on installed Chrome headless (the real GPU; the adapter is printed). Usage:
//   node scripts/k6/stochastic-ab.mjs http://127.0.0.1:<port> [--scenes=wdas,smoke] [--budget=20] [--rounds=2]
//     [--w=160] [--h=90] [--extra=a=b&c=d]
// against a running examples Vite server (preferably on a frozen copy: Vite serves sources live). Each round renders
// every scene with both filters for `budget` seconds (`?target=0`), alternating the order (ABBA). Per scene it prints:
// - throughput: batches per second of each filter and their ratio (paired per round);
// - efficiency: median over volume pixels of (stderr² · seconds)_trilinear / (stderr² · seconds)_stochastic of the
//   luminance (> 1: the stochastic filter reaches a given error sooner; both are unbiased estimators of one image, so
//   equal-time MSE is variance · time);
// - bias: the luminance difference of the two pooled means as an image z (Σ e / √Σ σ²) and the pixels over 4σ.
// Timings are provisional unless the machine is quiet (D25): note the load and the device.
import { chromium } from "@playwright/test";
import os from "node:os";

const [base, ...rest] = process.argv.slice(2);
if (!base) throw new Error("usage: stochastic-ab.mjs <base url> [--scenes=] [--budget=] [--rounds=]");
const opt = Object.fromEntries(rest.map((a) => a.replace(/^--/, "").split("=")).map(([k, ...v]) => [k, v.join("=")]));
const scenes = (opt.scenes ?? "wdas,smoke").split(",");
const budget = Number(opt.budget ?? 20);
const rounds = Number(opt.rounds ?? 2);
const W = Number(opt.w ?? 160);
const H = Number(opt.h ?? 90);
const extra = opt.extra ? `&${opt.extra}` : "";

const fromBase64 = (s) => new Float32Array(Uint8Array.from(Buffer.from(s, "base64")).buffer);
const lum = (a, i) => 0.2126 * a[i] + 0.7152 * a[i + 1] + 0.0722 * a[i + 2];

const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu"] });
const page = await browser.newPage();
const logs = [];
page.on("console", (m) => {
  if (m.type() === "error") logs.push(m.text());
});
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));

// Each round has its own seed: the kernel is deterministic, so pooling two rounds of one seed would count the same
// paths twice.
async function render(scene, density, seed) {
  const q = `?case=scene&scene=${scene}&density=${density}&seed=${seed}&w=${W}&h=${H}&target=0&budget=${budget}&batches=1000000${extra}`;
  await page.goto(`${base}/src/harness/k6-reference.html${q}`);
  await page.waitForFunction(() => window.__VDB_K6_REFERENCE__?.done === true, undefined, { timeout: (budget + 600) * 1000 });
  const r = await page.evaluate(() => window.__VDB_K6_REFERENCE__);
  if (r.error) throw new Error(`${scene} ${density}: ${r.error}`);
  const rep = r.report;
  return { adapter: r.adapter, batches: rep.batches, seconds: rep.seconds, noise: rep.noise.mean, mean: fromBase64(rep.mean), stderr: fromBase64(rep.stderr) };
}

const load = () => os.loadavg().map((x) => x.toFixed(2)).join(" ");
console.log(JSON.stringify({ host: `${os.cpus()[0]?.model} × ${os.cpus().length}`, loadBefore: load(), budget, rounds, size: `${W}x${H}` }));
for (const scene of scenes) {
  const runs = { trilinear: [], stochastic: [] };
  for (let k = 0; k < rounds; k++) {
    const order = k % 2 === 0 ? ["trilinear", "stochastic"] : ["stochastic", "trilinear"];
    for (const d of order) runs[d].push(await render(scene, d, k + 1));
  }
  const t = runs.trilinear;
  const s = runs.stochastic;
  const rate = (r) => r.batches / r.seconds;
  const throughput = t.map((a, k) => rate(s[k]) / rate(a));
  // Efficiency and bias per pixel, from the pooled runs of each filter (batch-weighted means; stderr pooled as 1/Σ(1/se²)).
  const pool = (rs) => {
    const n = rs[0].mean.length;
    const mean = new Float64Array(n);
    const se2 = new Float64Array(n);
    const cost = rs.reduce((a, r) => a + r.seconds, 0);
    for (let i = 0; i < n; i++) {
      let wsum = 0;
      let m = 0;
      for (const r of rs) {
        const w = r.batches;
        m += w * r.mean[i];
        wsum += w;
      }
      mean[i] = m / wsum;
      let inv = 0;
      for (const r of rs) inv += 1 / Math.max(r.stderr[i] ** 2, 1e-30);
      se2[i] = 1 / inv;
    }
    return { mean, se2, cost };
  };
  const pt = pool(t);
  const ps = pool(s);
  const ratios = [];
  let zSum = 0;
  let zVar = 0;
  let zOver = 0;
  let pixels = 0;
  for (let i = 0; i < pt.mean.length; i += 4) {
    const y = lum(pt.mean, i);
    if (!(pt.mean[i + 3] < 0.98 && y > 1e-3)) continue;
    const vt = lum(pt.se2.map(Math.sqrt), i) ** 2;
    const vs = lum(ps.se2.map(Math.sqrt), i) ** 2;
    ratios.push((vt * pt.cost) / Math.max(vs * ps.cost, 1e-30));
    const e = lum(ps.mean, i) - y;
    zSum += e;
    zVar += vt + vs;
    if (Math.abs(e) / Math.sqrt(vt + vs) > 4) zOver++;
    pixels++;
  }
  ratios.sort((a, b) => a - b);
  const q = (p) => ratios[Math.floor(ratios.length * p)];
  console.log(
    JSON.stringify({
      scene,
      adapter: t[0].adapter,
      batches: { trilinear: t.map((r) => r.batches), stochastic: s.map((r) => r.batches) },
      seconds: { trilinear: t.map((r) => +r.seconds.toFixed(2)), stochastic: s.map((r) => +r.seconds.toFixed(2)) },
      noise: { trilinear: t.map((r) => +r.noise.toFixed(4)), stochastic: s.map((r) => +r.noise.toFixed(4)) },
      throughputRatio: throughput.map((x) => +x.toFixed(3)),
      efficiency: { median: +q(0.5).toFixed(3), p10: +q(0.1).toFixed(3), p90: +q(0.9).toFixed(3) },
      // image z: Σ e / √Σ σ² (|z| < 4: no bias the runs can see); over4: pixels with |e| > 4σ
      bias: { pixels, imageZ: +(zSum / Math.sqrt(Math.max(zVar, 1e-30))).toFixed(3), over4: zOver },
      load: load(),
    }),
  );
}
if (logs.length) console.log("console errors:", JSON.stringify(logs.slice(0, 10)));
await browser.close();
