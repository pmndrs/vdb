import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";

/**
 * Gallery demo 16 — coarse-first streaming into the froxel renderer. smoke.vdb is baked coarse-first in the page,
 * streamed in 1 KiB chunks (`?test=1`: unthrottled, the stream held until each grid's frame is read back) through
 * `vdbStreamCoarseFirst` into one pool id. The preview must draw from the first ~14 KiB and differ from the final
 * frame, and the final frame must equal the full grid added directly (`?mode=direct`, as demo 09's golden checks
 * its full grid). Runs with and without the gallery assets (`?gallery=0`: procedural sky). G16a: the bake has an FpN
 * preview and an Fp8 full grid, and the swap goes into the node, source and volume made for the preview (`followImage`)
 * with no shader module or pipeline built.
 */
const DEMO = "/src/demos/16-streaming/index.html";
const SOURCE = "/fixtures/vdb-samples/smoke.vdb";
const BASE = "presentation=readback&test=1&still=1&asset=smoke&quality=low&temporal=0";
/** His smoke.wvdb needs 15,155 B before LOD0 can draw (demo 09's gate); ours with whole 1 KiB chunks. */
const GATE = Math.floor(14.8 * 1024);

interface Snapshot {
  width: number;
  height: number;
  mean: [number, number, number];
  thumb: number[];
  fine: number[];
}

test.use({ viewport: { width: 480, height: 320 } });
test.setTimeout(600_000);

/** Mean absolute difference of the 96 × 54 grey thumbnails: a preview's lost detail averages out at 32 × 18. */
const fineDiff = (a: Snapshot, b: Snapshot): number => a.fine.reduce((s, v, i) => s + Math.abs(v - b.fine[i]!), 0) / a.fine.length;
const thumbSpread = (s: Snapshot): number => Math.max(...s.thumb) - Math.min(...s.thumb);

async function openDone(page: Page, query: string): Promise<NonNullable<Window["__DEMO16__"]>> {
  await page.goto(`${DEMO}?${query}`);
  await page.waitForFunction(() => window.__DEMO16__?.error !== undefined || window.__DEMO16__?.done === true, undefined, { timeout: 240_000 });
  return (await page.evaluate(() => window.__DEMO16__))!;
}

test.describe("demo 16 — gallery: streaming", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  for (const gallery of [true, false]) {
    test(`preview from the first ~14 KiB, then the full grid in place, equal to the direct render (gallery assets ${gallery ? "on" : "off"})`, async ({ page }) => {
      const fixture = await page.request.head(SOURCE);
      test.skip(!fixture.ok(), `fixtures missing (${SOURCE} -> ${fixture.status()}): pnpm fixtures:vdb-samples`);
      const errors = collectErrors(page);
      const g = gallery ? "" : "&gallery=0";

      const state = await openDone(page, `${BASE}${g}`);
      const detail = () => JSON.stringify({ ...state, phases: state.phases.map((p) => ({ ...p, snapshot: p.snapshot ? "…" : null })), errors }, null, 2);
      expect(state.error, detail()).toBeUndefined();
      expect(state.presentation, detail()).toBe("readback");
      expect(state.asset, detail()).toBe("smoke");
      if (!gallery) expect(state.sky, detail()).toBe("procedural");

      // Two grids, the preview first, in one pool id with no growth.
      expect(state.phases.map((p) => [p.phase, p.grid]), detail()).toEqual([
        ["preview", "density.lod48"],
        ["full", "density"],
      ]);
      const [preview, full] = state.phases as [(typeof state.phases)[number], (typeof state.phases)[number]];
      expect(state.sameId, detail()).toBe(true);
      expect(state.inPlace, detail()).toBe(true);
      expect(preview.transferBytes, detail()).toBeGreaterThan(0);
      expect(preview.transferBytes, detail()).toBeLessThanOrEqual(GATE);
      expect(preview.voxelRatio, detail()).toBeGreaterThan(1);
      // G16a: the swap changes the grid type (FpN preview, Fp8 full grid) and the box, into the same node, source and
      // volume (made at the preview with every type of the family and `followImage`): no shader or pipeline is built.
      expect([preview.gridType, full.gridType], detail()).toEqual(["FpN", "Fp8"]);
      expect(preview.builds, detail()).toBeGreaterThan(0);
      expect(full.builds, detail()).toBe(0);
      expect(full.transferBytes, detail()).toBe(state.totalBytes);
      expect(state.transferred, detail()).toBe(state.totalBytes);
      await expect(page.getByTestId("phase")).toHaveAttribute("data-phase", "full");

      // Both phases draw the volume; the preview is visibly coarser than the full grid.
      const p = preview.snapshot!;
      const f = full.snapshot!;
      expect(thumbSpread(p), detail()).toBeGreaterThan(10);
      expect(thumbSpread(f), detail()).toBeGreaterThan(10);
      const swap = fineDiff(p, f);

      // The streamed final frame equals the full grid added directly (no stream), placed and drawn the same way.
      const direct = await openDone(page, `${BASE}${g}&mode=direct`);
      expect(direct.error, JSON.stringify(direct)).toBeUndefined();
      const reference = direct.reference!;
      const final = fineDiff(f, reference);
      console.log(
        `demo 16 (gallery ${gallery ? "on" : "off"}): preview at ${preview.transferBytes} B of ${state.totalBytes}; preview->full diff ${swap.toFixed(2)}, ` +
          `full vs direct ${final.toFixed(3)}; shader/pipeline builds: preview ${preview.builds}, swap ${full.builds}`,
      );
      // Low-quality froxels blur both a lot (a 0.3 mean difference); the direct render is bit-identical (0.000), so the
      // preview must stand clearly above that floor and the streamed final at it.
      expect(swap, detail()).toBeGreaterThan(0.15);
      expect(final, detail()).toBeLessThan(swap * 0.1);

      expect(state.uncaptured, detail()).toEqual([]);
      expect(direct.uncaptured, detail()).toEqual([]);
      expect(errors, detail()).toEqual([]);
    });
  }

  // K4.22: smoke.vdb baked as a 3-LOD `.wvdb` with small pages streams page by page into one dense source. One frame per
  // stream event (a page or a completed LOD) at the same file offsets in every mode, each against the final frame by
  // PSNR over the volume's pixels: `upsample` (pending bricks filled from the coarser LOD on the GPU, their mip level 1
  // the coarser LOD) starts within 2 dB of the coarser LOD it replaces (`off` keeps that until the LOD is whole) and
  // gains from there; on average it beats `off`, his mips (`upsample-box`: every level from the upsample) and empty
  // bricks (`background`) by 3 dB or more. The final frame is byte-identical in every mode and to the whole file added
  // directly.
  test("K4.22 .wvdb partial LODs: upsampled pages beat whole LODs only, his mips and empty bricks; the final frame identical in every mode", async ({ page }) => {
    const fixture = await page.request.head(SOURCE);
    test.skip(!fixture.ok(), `fixtures missing (${SOURCE} -> ${fixture.status()}): pnpm fixtures:vdb-samples`);
    const errors = collectErrors(page);
    const WVDB = `${BASE}&gallery=0&format=wvdb`;
    type Run = NonNullable<NonNullable<Window["__DEMO16__"]>["wvdb"]>;
    const runs = {} as Record<"upsample" | "upsample-box" | "background" | "off", Run>;
    for (const partial of ["upsample", "upsample-box", "background", "off"] as const) {
      const state = await openDone(page, `${WVDB}&partial=${partial}`);
      const detail = JSON.stringify({ ...state, phases: [] }, null, 2);
      expect(state.error, detail).toBeUndefined();
      expect(state.uncaptured, detail).toEqual([]);
      expect(state.wvdb?.partial, detail).toBe(partial);
      runs[partial] = state.wvdb!;
    }
    const direct = await openDone(page, `${WVDB}&mode=direct`);
    expect(direct.error, JSON.stringify(direct)).toBeUndefined();
    const detail = JSON.stringify(runs, null, 2);
    const { upsample, background, off } = runs;
    const his = runs["upsample-box"];
    // The same events in every mode (the reader always reports pages; `off` only keeps the last whole LOD on screen).
    expect(background.frames.map((f) => f.bytes), detail).toEqual(upsample.frames.map((f) => f.bytes));
    expect(off.frames.map((f) => f.bytes), detail).toEqual(upsample.frames.map((f) => f.bytes));
    expect(his.frames.map((f) => f.bytes), detail).toEqual(upsample.frames.map((f) => f.bytes));
    expect(upsample.lods.length, detail).toBe(3);
    // The completed LOD is byte-identical whatever showed mid-stream: the final frames, and the direct render.
    expect(background.finalHash, detail).toBe(upsample.finalHash);
    expect(off.finalHash, detail).toBe(upsample.finalHash);
    expect(his.finalHash, detail).toBe(upsample.finalHash);
    expect(direct.referenceHash, detail).toBe(upsample.finalHash);
    // The fills ran on the GPU (none on the CPU), on several pages of LODs 1 and 2.
    expect(upsample.fills?.cpu, detail).toBe(0);
    expect(upsample.fills?.gpu ?? 0, detail).toBeGreaterThanOrEqual(upsample.frames.filter((f) => f.filled).length);
    const partials = upsample.frames.map((f, i) => ({ i, f })).filter(({ f }) => f.filled);
    expect(new Set(partials.map(({ f }) => f.lod)), detail).toEqual(new Set([1, 2]));
    expect(partials.length, detail).toBeGreaterThanOrEqual(6);
    expect(upsample.region, detail).toBeGreaterThan(1000);
    const rows = upsample.frames.map((f, i) => {
      const b = background.frames[i]!;
      const o = off.frames[i]!;
      const h = his.frames[i]!;
      const what = f.complete ? "complete" : `${String(f.decoded).padStart(3)}/${f.stored} bricks`;
      return (
        `LOD ${f.lod} ${what.padEnd(14)} @ ${(f.bytes / 1024).toFixed(1).padStart(5)} KiB: upsample ${f.regionPsnr.toFixed(2)} dB, his mips ${h.regionPsnr.toFixed(2)}, ` +
        `whole LODs only (LOD ${o.shownLod}) ${o.regionPsnr.toFixed(2)}, empty bricks ${b.regionPsnr.toFixed(2)}; builds ${f.builds}/${h.builds}/${o.builds}/${b.builds}`
      );
    });
    const mean = (r: Run) => partials.reduce((s, { i }) => s + r.frames[i]!.regionPsnr, 0) / partials.length;
    console.log(
      `demo 16 .wvdb (${upsample.lods.map((d) => d.join("x")).join(" → ")}, region ${upsample.region} px, GPU fills ${upsample.fills?.gpu}), PSNR vs the final frame over the volume's pixels:\n  ${rows.join("\n  ")}\n` +
        `  mean over the ${partials.length} partial pages: upsample ${mean(upsample).toFixed(2)} dB, his mips ${mean(his).toFixed(2)} dB, whole LODs only ${mean(off).toFixed(2)} dB, empty bricks ${mean(background).toFixed(2)} dB`,
    );
    // SwiftShader, quality low (2026-11-28): upsample 49.4 dB mean, his mips 42.4, whole LODs only 42.9, empty 34.2.
    for (const { i, f } of partials) {
      const u = f.regionPsnr;
      const o = off.frames[i]!.regionPsnr;
      // The first page of a LOD shows the coarser LOD at the finer size: within 1.3 dB of it (the march and the level-0
      // reads follow the texture size); every later page only adds detail.
      expect(u, `${rows[i]}\n${detail}`).toBeGreaterThanOrEqual(o - 2);
      // While at most half the bricks are in, empty bricks are far worse (later the missing ones may be near empty).
      if (f.decoded * 2 <= f.stored) expect(u, `${rows[i]}\n${detail}`).toBeGreaterThan(background.frames[i]!.regionPsnr + 3);
    }
    expect(mean(upsample), detail).toBeGreaterThan(mean(off) + 3);
    expect(mean(upsample), detail).toBeGreaterThan(mean(his) + 3);
    expect(mean(upsample), detail).toBeGreaterThan(mean(background) + 3);
    // The finest LOD's pages build no shader module or pipeline (the fill kernels and the frame kernels exist by then).
    expect(upsample.frames.filter((f) => f.lod === 2).map((f) => f.builds), detail).toEqual(upsample.frames.filter((f) => f.lod === 2).map(() => 0));
    expect(errors, detail).toEqual([]);
  });
});
