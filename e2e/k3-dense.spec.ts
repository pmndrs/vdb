import { expect, test, type Page } from "@playwright/test";

// K3.0 (R1 §6 Step 0), the core half. The page is examples/src/harness/k3-dense.ts: the committed CC0
// cloud_04_variant_0000.wvdb as the dense source, and the core vdbDenseSource's generated WGSL (vdbSrcDensity at probe
// points and LODs) against its CPU twin sampleWvdbDenseTexture. K6.11 (D35): the three-texture TSL render (and its two
// determinism gates) went with vdbVolumeNodes; the dense frame renders are covered by e2e/k6-frames.spec.ts and the
// k6-core gather / scene probes.
const PAGE_URL = "/src/harness/k3-dense.html";

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  volume?: { dimensions: number[]; lod: number; meanDensity: number; mipLevels: number; statsSource: string };
  probes?: { count: number; maxAbs: number; meanAbs: number; over: number; tolerance: number; lodLevels: number; texels: number[]; firstBad?: unknown };
  uncaptured: string[];
}

async function run(page: Page, query = ""): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_DENSE__?: { done: boolean } }).__VDB_K3_DENSE__?.done === true, undefined, {
    timeout: 110_000,
  });
  return (await page.evaluate(() => (window as { __VDB_K3_DENSE__?: unknown }).__VDB_K3_DENSE__)) as Result;
}

test("K3.0 dense source: the core vdbDenseSource WGSL matches its CPU twin", async ({ page }) => {
  test.setTimeout(240_000);
  const first = await run(page);
  const detail = JSON.stringify(first, null, 2);
  expect(first.error, detail).toBeUndefined();
  expect(first.uncaptured, detail).toEqual([]);
  expect(first.volume, detail).toMatchObject({ dimensions: [84, 128, 43], lod: 2, mipLevels: 7, statsSource: "texels" });
  const p = first.probes!;
  expect(p.count, detail).toBe(2048);
  expect(p.texels, detail).toEqual([84, 128, 43]);
  expect(p.lodLevels, detail).toBe(7);
  expect(p.over, detail).toBe(0);
  console.log(`[k3-dense] ${first.adapter}: probes max |d| ${p.maxAbs.toExponential(2)} (mean ${p.meanAbs.toExponential(2)})`);
});

test("K3.0 dense source without mips (lod ignored): the core WGSL still matches", async ({ page }) => {
  test.setTimeout(240_000);
  const first = await run(page, "?mips=0");
  expect(first.error).toBeUndefined();
  expect(first.volume?.mipLevels).toBe(0);
  expect(first.probes?.lodLevels).toBe(0);
  expect(first.probes?.over).toBe(0);
});
