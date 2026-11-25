import { expect, test } from "@playwright/test";
import { collectErrors } from "./console-errors";

// K6.10 gate review, repro only (not a product test): `settings.shadowResolution` above what the transmittance stack can
// tile in maxTextureDimension3D must clamp (renderer.ts #gridResolution says it clamps and warns), but the clamp loop
// calls vdbFrameStackLayout, which throws a RangeError before the loop can lower R. Marked `test.fail()`: this test
// passes while the bug is present and fails once it is fixed (then delete it, or turn it into a regression test).
test.fail();
test("K6.10 repro: shadowResolution 1100 with 'frame' lighting clamps instead of throwing", async ({ page }) => {
  test.setTimeout(600_000);
  const errors = collectErrors(page);
  await page.goto("/src/harness/k3-frame.html?w=320&h=192&froxels=160x96x24&lighting=frame&shadowRes=1100");
  await page.waitForFunction(() => (window as unknown as { __VDB_K3_FRAME__?: { done?: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 540_000 });
  const r = (await page.evaluate(() => (window as unknown as { __VDB_K3_FRAME__: { error?: string } }).__VDB_K3_FRAME__)) as { error?: string };
  console.log(`[k6-10 repro] error: ${String(r.error).split("\n")[0]}; console errors: ${errors.length ? errors.join(" | ").slice(0, 400) : "none"}`);
  expect(r.error).toBeUndefined();
  expect(errors).toEqual([]);
});
