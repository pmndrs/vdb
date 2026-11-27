import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";

// K6.Q1b: the size-adaptive 'frame' shadow grid (`settings.shadowResolutionMax`) through `./three` on the CC0 smoke
// (k3-frame's own scene, CI; `?extinction=400` makes it dense). Off (the default 0, or a max at or below `shadowResolution`) the frame is byte-identical to
// the fixed grid; on, the core picks a larger side (`shadowGridResolution`) and the frame changes. The side itself is
// unit-tested (`test/tools/frame-grid.test.ts`, vdbFrameAdaptiveGridResolution); this checks the renderer wiring.
const BASE = "/src/harness/k3-frame.html?w=320&h=192&froxels=160x96x24&lighting=frame";

interface Result {
  done: boolean;
  error?: string;
  rgba?: string;
  shadowGridResolution?: number;
  warnings: string[];
  uncaptured: string[];
}

async function render(page: Page, query: string): Promise<Result> {
  await page.goto(`${BASE}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_FRAME__?: { done?: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 600_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_FRAME__?: unknown }).__VDB_K3_FRAME__)) as Result;
}

function maxDiff(a: string, b: string): number {
  const x = Buffer.from(a, "base64");
  const y = Buffer.from(b, "base64");
  let m = 0;
  for (let i = 0; i < x.length; i++) if (i % 4 !== 3) m = Math.max(m, Math.abs(x[i]! - y[i]!));
  return m;
}

test("K6.Q1b: shadowResolutionMax off keeps the fixed grid byte for byte; on, the grid grows and the frame changes", async ({ page }) => {
  test.setTimeout(1_200_000);
  const errors = collectErrors(page);
  const fixed = await render(page, "");
  expect(fixed.error, JSON.stringify(fixed.warnings)).toBeUndefined();
  expect(fixed.shadowGridResolution).toBe(64);
  // a max at or below shadowResolution is off
  const atBase = await render(page, "&shadowResMax=64");
  expect(atBase.shadowGridResolution).toBe(64);
  expect(maxDiff(atBase.rgba!, fixed.rgba!)).toBe(0);
  // the default rule ('both': 16 voxels, 2 optical depths) on this small smoke: its side, and the frame moves iff it grew
  const both = await render(page, "&shadowResMax=256");
  expect(both.error).toBeUndefined();
  expect(both.shadowGridResolution! >= 64 && both.shadowGridResolution! <= 256).toBe(true);
  expect(maxDiff(both.rgba!, fixed.rgba!) === 0).toBe(both.shadowGridResolution === 64);
  // a dense smoke (extinction 400): 2 optical depths per cell asks for a finer grid, the side grows, the frame changes
  const dense = await render(page, "&extinction=400");
  expect(dense.shadowGridResolution).toBe(64);
  const adaptive = await render(page, "&extinction=400&shadowResMax=256");
  expect(adaptive.error).toBeUndefined();
  const side = adaptive.shadowGridResolution!;
  expect(side).toBeGreaterThan(64);
  expect(side % 16 === 0 || side < 256).toBe(true);
  const moved = maxDiff(adaptive.rgba!, dense.rgba!);
  expect(moved).toBeGreaterThan(0);
  // a lower max clamps the side
  const capped = await render(page, "&extinction=400&shadowResMax=96");
  expect(capped.shadowGridResolution).toBe(Math.min(96, side));
  console.log(`[k6-adaptive-grid] default rule side ${both.shadowGridResolution}; dense smoke side ${side} (max 256), capped ${capped.shadowGridResolution} (max 96); frame moved by up to ${moved}/255`);
  for (const r of [fixed, atBase, both, dense, adaptive, capped]) expect(r.uncaptured).toEqual([]);
  expect(errors).toEqual([]);
});
