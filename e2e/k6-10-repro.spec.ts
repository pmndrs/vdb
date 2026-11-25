import { expect, test } from "@playwright/test";
import { collectErrors } from "./console-errors";

// K6.10 gate review finding 1, regression test since K6.10a: `settings.shadowResolution` above what the transmittance
// stack can tile in maxTextureDimension3D must clamp (warned once) instead of throwing. Before K6.10a the clamp loop in
// renderer.ts #gridResolution called vdbFrameStackLayout, which threw a RangeError ("4 sub-volumes of 1100³ do not fit
// 2048³") on every update, so no frame rendered. The unit test of the clamp is `vdbFrameFitGridResolution` in
// packages/vdb/test/core/frame/frame.test.ts.
test("K6.10a: shadowResolution 1100 with 'frame' lighting clamps (warned once) instead of throwing", async ({ page }) => {
  test.setTimeout(600_000);
  const errors = collectErrors(page);
  const clampWarnings: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "warning" && m.text().includes("settings.shadowResolution 1100")) clampWarnings.push(m.text());
  });
  await page.goto("/src/harness/k3-frame.html?w=320&h=192&froxels=160x96x24&lighting=frame&shadowRes=1100");
  await page.waitForFunction(() => (window as unknown as { __VDB_K3_FRAME__?: { done?: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 540_000 });
  const r = (await page.evaluate(() => (window as unknown as { __VDB_K3_FRAME__: { error?: string } }).__VDB_K3_FRAME__)) as { error?: string };
  console.log(`[k6-10a] error: ${String(r.error).split("\n")[0]}; clamp warnings: ${clampWarnings.join(" | ")}; console errors: ${errors.length ? errors.join(" | ").slice(0, 400) : "none"}`);
  expect(r.error).toBeUndefined();
  expect(errors).toEqual([]);
  // one warning for the whole run (several updates), naming the reason and the clamped resolution
  expect(clampWarnings.length).toBe(1);
  expect(clampWarnings[0]).toMatch(/do not tile in maxTextureDimension3D|maxBufferSize/);
  expect(clampWarnings[0]).toMatch(/clamped to \d+/);
});
