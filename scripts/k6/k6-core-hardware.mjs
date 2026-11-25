// K6.4a: run k6-core harness probes on installed Chrome headless (the real GPU; check the adapter it prints). Usage:
//   node scripts/k6/k6-core-hardware.mjs http://127.0.0.1:<port> "?probe=resolve&spectral=1&tier1=1" [<query> ...]
// against a running examples Vite server (preferably on a frozen copy: Vite serves sources live). Prints each probe's stats.
import { chromium } from "@playwright/test";
const [base, ...queries] = process.argv.slice(2);
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: ["--enable-unsafe-webgpu"],
});
const page = await browser.newPage();
const logs = [];
page.on("console", (m) => {
  if (m.type() === "error" || m.type() === "warning")
    logs.push(`${m.type()}: ${m.text()}`);
});
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
for (const q of queries) {
  await page.goto(`${base}/src/harness/k6-core.html${q}`);
  await page.waitForFunction(
    () => window.__VDB_K6_CORE__?.done === true,
    undefined,
    { timeout: 1_800_000 },
  );
  const r = await page.evaluate(() => window.__VDB_K6_CORE__);
  // compile seconds are not timings (a loaded machine, one run): drop them
  if (r.compile) delete r.compile.seconds;
  const brief = {
    query: q,
    adapter: r.adapter,
    error: r.error,
    warnings: r.warnings,
    uncaptured: r.uncaptured,
    spectral: r.spectral,
  };
  for (const k of Object.keys(r))
    if (/Stats$|^stats$/.test(k) && r[k])
      brief[k] = {
        compared: r[k].compared,
        covered: r[k].covered,
        over: r[k].over,
        maxUlp: r[k].maxUlp,
      };
  for (const k of Object.keys(r))
    if (!(k in brief) && !/Stats$|^stats$|^done$/.test(k)) brief[k] = r[k];
  console.log(JSON.stringify(brief).slice(0, 4000));
}
if (logs.length) console.log("console:", JSON.stringify(logs.slice(0, 20)));
await browser.close();
