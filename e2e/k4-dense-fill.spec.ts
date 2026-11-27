import { expect, test, type Page } from "@playwright/test";

// K4.22: the core dense source's GPU fill of a partial `.wvdb` LOD (examples/src/harness/k4-dense-fill.ts). Seeded
// files stream one byte per push; every event (partial pages and completed LODs) goes into one `vdbDenseSource` by
// `replace`, and the source's textures (level 0 and mip atlas) read back must equal the CPU twin (`wvdbDenseResolve`)
// byte for byte: halving and odd sizes, an RGBA8 albedo beside the density, a larger grid, a dense layout without
// mips; level 1 of the pending bricks from the coarser LOD (default) or his box mips. Integer kernels: no tolerance.
// The WGSL mutations (fail-before) must each break the comparison.
const PAGE_URL = "/src/harness/k4-dense-fill.html";

interface CaseResult {
  name: string;
  dims: number[];
  events: number;
  filled: number;
  bytes: number;
  mismatches: number;
  maxDiff: number;
  fills: { gpu: number; cpu: number };
  upsampledTexels: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  cases: CaseResult[];
  uncaptured: string[];
}

async function run(page: Page, query = ""): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K4_DENSE_FILL__?: { done: boolean } }).__VDB_K4_DENSE_FILL__?.done === true, undefined, { timeout: 240_000 });
  return (await page.evaluate(() => (window as { __VDB_K4_DENSE_FILL__?: unknown }).__VDB_K4_DENSE_FILL__)) as Result;
}

test.describe.configure({ timeout: 300_000 });

test("K4.22 dense fill: GPU textures = CPU twin byte for byte, every page of every LOD", async ({ page }) => {
  const r = await run(page);
  const detail = JSON.stringify(r, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.cases.map((c) => c.name), detail).toEqual(["halving", "halving-box-mips", "odd", "albedo", "large", "dense-nomip"]);
  for (const c of r.cases) {
    expect(c.mismatches, `${c.name}\n${detail}`).toBe(0);
    // The probe exercises the GPU path: pages with pending bricks, filled on the GPU (none fell back to the CPU),
    // and the upsample changed texels against the background fill.
    expect(c.filled, `${c.name}\n${detail}`).toBeGreaterThan(2);
    expect(c.fills.gpu, `${c.name}\n${detail}`).toBeGreaterThanOrEqual(c.filled);
    expect(c.fills.cpu, `${c.name}\n${detail}`).toBe(0);
    expect(c.upsampledTexels, `${c.name}\n${detail}`).toBeGreaterThan(100);
  }
  console.log(`[k4.22] ${r.adapter}: ${r.cases.map((c) => `${c.name} ${c.events} events, ${c.fills.gpu} GPU fills, ${c.bytes} B equal`).join("; ")}`);
});

for (const mutate of ["wgsl-fill-round", "wgsl-mip-round", "wgsl-fill-pending", "wgsl-mip-coarse"]) {
  test(`K4.22 dense fill: the mutation ${mutate} breaks the comparison`, async ({ page }) => {
    const r = await run(page, `?mutate=${mutate}`);
    const detail = JSON.stringify(r, null, 2);
    expect(r.error, detail).toBeUndefined();
    const off = r.cases.reduce((s, c) => s + c.mismatches, 0);
    expect(off, detail).toBeGreaterThan(0);
    console.log(`[k4.22] ${mutate}: ${off} bytes differ (${r.cases.map((c) => `${c.name} ${c.mismatches}`).join(", ")})`);
  });
}
