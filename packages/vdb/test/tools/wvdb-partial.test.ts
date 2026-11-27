/**
 * K4.22: partial `.wvdb` LODs while streaming. `wvdbDensePartialVolume` turns a `WvdbStreamReader({ partialLods })`
 * event into a dense volume whose undecoded stored bricks carry a `fill` (the coarser LOD to upsample);
 * `wvdbDenseResolve` (the CPU twin of the core dense source's GPU fill) applies it. Checks:
 *  - the twin against his worker kernel (upsample + clearMissingBricks + page writes, re-implemented from his
 *    source in his-wvdb-kernel.ts) chunk by chunk: byte for byte on halving LOD sizes (his bake's), within one code
 *    on odd sizes (8-bit integer weights, V-row in DIVERGES);
 *  - the completed LOD is byte-identical with any fill (and through the core source after partial replaces);
 *  - the core source's CPU fill uploads the twin's bytes, mip atlas included; its GPU path is the k4-dense-fill probe.
 */
import { describe, expect, it } from "vitest";
import {
  WVDB_UPSAMPLE_WEIGHT_ONE,
  WvdbStreamReader,
  readWvdb,
  writeWvdb,
  wvdbDensePartialVolume,
  wvdbDenseResolve,
  wvdbDenseVolume,
  wvdbMipAtlas,
  wvdbMipAtlasLayout,
  wvdbPendingBricks,
  wvdbUpsampleAxis,
  wvdbUpsampleTexels,
  type WvdbDenseVolume,
  type WvdbFile,
  type WvdbGridInput,
  type WvdbLodEvent,
  type WvdbVec3,
  type WvdbVolumeLevel,
} from "../../src/tools/internal.js";
import { vdbDenseSource } from "../../src/core/index.js";
import { bytesEqual } from "./wvdb-helpers.js";
import { hisKernel, hisToByte, type HisKernelGrid } from "./his-wvdb-kernel.js";

/** A soft blob with ripples, values in [0, 1.2] (some saturate), zero outside a sphere (sparse bricks). */
function blob(dims: WvdbVec3, seed = 0): Float32Array {
  const [X, Y, Z] = dims;
  const v = new Float32Array(X * Y * Z);
  const R = Math.min(X, Y, Z) * 0.45;
  for (let z = 0; z < Z; z++)
    for (let y = 0; y < Y; y++)
      for (let x = 0; x < X; x++) {
        const r = Math.hypot(x - X / 2, y - Y / 2, z - Z / 2);
        v[x + X * (y + Y * z)] = r < R ? 1.2 * (1 - r / R) + 0.1 * Math.sin(x * 0.9 + y * 0.5 + z * 0.3 + seed) : 0;
      }
  return v;
}

function bake(dims: WvdbVec3, o: { layout?: "sparse" | "dense"; pageBricks?: number; lods?: number; color?: boolean } = {}): Uint8Array {
  const grids: WvdbGridInput[] = [{ name: "density", class: "fog volume", background: 0, bbox: [[-1, -1, -1], [1, 1, 1]], finest: { dimensions: dims, values: blob(dims) } }];
  if (o.color) {
    const n = dims[0] * dims[1] * dims[2];
    const c = new Float32Array(n * 3);
    const b = blob(dims, 1);
    for (let i = 0; i < n; i++) {
      c[i] = b[i]!;
      c[n + i] = 0.5 * b[i]!;
      c[2 * n + i] = b[i]! > 0 ? 0.8 : 0;
    }
    grids.push({ name: "Cd", channels: 3, background: [0, 0, 0], bbox: [[-1, -1, -1], [1, 1, 1]], finest: { dimensions: dims, values: c } });
  }
  return writeWvdb(grids, { layout: o.layout ?? "sparse", pageBricks: o.pageBricks ?? 4, lods: o.lods ?? 3 }).bytes;
}

/**
 * Streams `bytes` one byte per push (so a push completes at most one chunk) and calls `on` for each event at once:
 * events share the reader's live levels (Traps), so a snapshot must be taken before the next push.
 */
function stream<T>(bytes: Uint8Array, on: (e: WvdbLodEvent, file: WvdbFile) => T): { reader: WvdbStreamReader; events: WvdbLodEvent[]; out: T[] } {
  const reader = new WvdbStreamReader({ partialLods: true });
  const events: WvdbLodEvent[] = [];
  const out: T[] = [];
  for (let i = 0; i < bytes.length; i++) {
    for (const e of reader.push(bytes.subarray(i, i + 1))) {
      events.push(e);
      out.push(on(e, reader.file!));
    }
  }
  reader.end();
  return { reader, events, out };
}

/** His kernel's texel data for grid `gridName` after each chunk (null where his kernel posts nothing), in file order. */
async function hisChunkResults(bytes: Uint8Array, gridName: string): Promise<Map<number, { lod: number; complete: boolean; data: Uint8Array }>> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dataStart = 12 + view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, dataStart)));
  // WVDBReader.js getBackgroundTexel L131–137, getKernelGrid L140–151 (scale 1)
  const kernelGrids: HisKernelGrid[] = header.grids.map((grid: any) => ({
    class: grid.class,
    background: grid.background,
    channels: grid.channels || 1,
    scale: 1,
    backgroundTexel: (grid.channels || 1) === 3 ? [...grid.background.map(hisToByte), 255] : [grid.class === "level set" ? 0 : hisToByte(grid.background)],
    lods: grid.lods.map(({ dimensions, mask, pages }: any) => ({ dimensions, layout: mask ? "sparse" : "dense", pages: pages.length })),
  }));
  const chunks: { gridIndex: number; lod: number; kind: string; offset: number; length: number; firstIndex?: number }[] = [];
  header.grids.forEach((grid: any, gridIndex: number) => {
    grid.lods.forEach(({ mask, pages }: any, lod: number) => {
      if (mask) chunks.push({ gridIndex, lod, kind: "mask", offset: mask.offset, length: mask.length });
      let firstIndex = 0;
      pages.forEach(({ offset, length, bricks }: any) => {
        chunks.push({ gridIndex, lod, kind: "page", offset, length, firstIndex });
        firstIndex += bricks;
      });
    });
  });
  chunks.sort((a, b) => a.offset - b.offset);
  const kernel = hisKernel(header.bake.bits, kernelGrids);
  const want = header.grids.findIndex((g: any) => g.name === gridName);
  const out = new Map<number, { lod: number; complete: boolean; data: Uint8Array }>();
  for (const c of chunks) {
    const r = await kernel.chunk({ ...c, bytes: bytes.slice(dataStart + c.offset, dataStart + c.offset + c.length) });
    if (r && c.gridIndex === want) out.set(c.offset, { lod: r.lod, complete: r.complete, data: r.data });
  }
  return out;
}

/** Bytes that differ, and the largest difference. */
function diff(a: Uint8Array, b: Uint8Array): { count: number; max: number } {
  expect(a.length).toBe(b.length);
  let count = 0;
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i]! - b[i]!);
    if (d) count++;
    max = Math.max(max, d);
  }
  return { count, max };
}

describe("K4.22 upsample axis (his wvdbKernel.js L125–135, integer weights)", () => {
  /** His axis: f64 position, f32 weight. */
  const hisAxis = (size: number, sourceSize: number) =>
    Array.from({ length: size }, (_, i) => {
      const position = Math.min(Math.max(((i + 0.5) / size) * sourceSize - 0.5, 0), sourceSize - 1);
      const lower = Math.floor(position);
      return { lower, upper: Math.min(lower + 1, sourceSize - 1), weight: Math.fround(position - lower) };
    });

  it("halving sizes: the same taps and exactly his weights (1/4, 3/4, edges clamped)", () => {
    for (const [size, src] of [[64, 32], [32, 16], [16, 8], [2, 1], [8, 4]] as const) {
      const ours = wvdbUpsampleAxis(size, src);
      hisAxis(size, src).forEach((h, i) => {
        expect([ours.lower[i], ours.upper[i]]).toEqual([h.lower, h.upper]);
        expect(ours.weight[i]! / WVDB_UPSAMPLE_WEIGHT_ONE).toBe(h.weight);
      });
    }
  });

  it("any sizes: the same taps, weights within half a step of 1/256", () => {
    for (const [size, src] of [[37, 19], [19, 10], [29, 15], [21, 11], [97, 49], [127, 64], [5, 3], [3, 2], [7, 1]] as const) {
      const ours = wvdbUpsampleAxis(size, src);
      hisAxis(size, src).forEach((h, i) => {
        expect([ours.lower[i], ours.upper[i]]).toEqual([h.lower, h.upper]);
        expect(Math.abs(ours.weight[i]! / WVDB_UPSAMPLE_WEIGHT_ONE - h.weight)).toBeLessThanOrEqual(0.5 / WVDB_UPSAMPLE_WEIGHT_ONE + 1e-7);
      });
    }
  });
});

describe("K4.22 partial LODs: the CPU twin against his worker kernel, chunk by chunk", () => {
  const cases: { name: string; dims: WvdbVec3; layout: "sparse" | "dense"; exact: boolean }[] = [
    { name: "sparse, halving sizes", dims: [48, 40, 32], layout: "sparse", exact: true },
    { name: "dense, halving sizes", dims: [32, 24, 16], layout: "dense", exact: true },
    { name: "sparse, odd sizes", dims: [37, 29, 21], layout: "sparse", exact: false },
  ];
  for (const c of cases) {
    it(`${c.name} ${c.dims.join("x")}: every page of every LOD`, async () => {
      const bytes = bake(c.dims, { layout: c.layout, pageBricks: 3 });
      const his = await hisChunkResults(bytes, "density");
      const { events, out } = stream(bytes, (e, file) => (e.complete ? null : wvdbDensePartialVolume(file, e, { mipmaps: false })));
      let compared = 0;
      let upsampled = 0;
      let off = { count: 0, max: 0, bytes: 0 };
      for (const [k, e] of events.entries()) {
        if (e.complete || !e.chunk || e.chunk.kind !== "page") continue;
        const h = his.get(e.chunk.offset);
        if (!h) continue; // his kernel posts nothing for a one-page LOD
        expect(h.lod).toBe(e.lod);
        const partial = out[k]!;
        const resolved = wvdbDenseResolve(partial);
        // The last page completes the LOD in his kernel too: then ours is the plain LOD (no pending bricks).
        if (h.complete) expect(partial.density.fill ?? null).toBeNull();
        if (partial.density.fill) upsampled++;
        const d = diff(resolved.density.data, h.data);
        if (c.exact) expect(d, `LOD ${e.lod} chunk at ${e.chunk.offset}`).toEqual({ count: 0, max: 0 });
        off = { count: off.count + d.count, max: Math.max(off.max, d.max), bytes: off.bytes + h.data.length };
        compared++;
      }
      expect(compared).toBeGreaterThan(5);
      expect(upsampled).toBeGreaterThan(3);
      if (!c.exact) {
        // 8-bit weights vs his f32 ones on non-dyadic sizes: a code at most, on a small share of the texels.
        expect(off.max).toBeLessThanOrEqual(1);
        expect(off.count / off.bytes).toBeLessThan(0.02);
      }
      console.log(`[k4.22] ${c.name}: ${compared} partial pages vs his kernel, ${upsampled} with a fill; ${off.count} of ${off.bytes} bytes off (max ${off.max})`);
    });
  }

  it("an albedo (vector, RGBA8) grid upsamples every component, alpha stays 255", async () => {
    const dims: WvdbVec3 = [32, 32, 16];
    const bytes = bake(dims, { pageBricks: 2, color: true });
    const his = await hisChunkResults(bytes, "Cd");
    const { reader, events, out } = stream(bytes, (e, file) => (e.complete ? null : wvdbDensePartialVolume(file, e, { albedo: "Cd", mipmaps: false })));
    let compared = 0;
    for (const [k, e] of events.entries()) {
      if (e.complete || !e.chunk || e.chunk.kind !== "page" || reader.file!.grids[e.chunk.gridIndex]!.name !== "Cd") continue;
      const h = his.get(e.chunk.offset);
      if (!h || h.complete) continue;
      const v = wvdbDenseResolve(out[k]!);
      expect(v.albedo!.format).toBe("rgba8unorm");
      expect(diff(v.albedo!.data, h.data)).toEqual({ count: 0, max: 0 });
      for (let i = 3; i < v.albedo!.data.length; i += 4) if (v.albedo!.data[i] !== 255) throw new Error(`alpha ${v.albedo!.data[i]} at ${i}`);
      compared++;
    }
    expect(compared).toBeGreaterThan(2);
  });
});

describe("K4.22 partial volumes", () => {
  const dims: WvdbVec3 = [48, 40, 32];
  const bytes = bake(dims, { pageBricks: 3 });
  const snap = (l: WvdbVolumeLevel): WvdbVolumeLevel => ({ ...l, mask: l.mask && l.mask.slice(), decoded: l.decoded.slice(), brickIndices: l.brickIndices.slice() });
  const { reader, events, out } = stream(bytes, (e, file) => ({ e, level: snap(e.levels[0] as WvdbVolumeLevel), upsample: wvdbDensePartialVolume(file, e), boxMips: wvdbDensePartialVolume(file, e, { fillMips: "box" }), background: wvdbDensePartialVolume(file, e, { fill: "background" }) }));
  const file = reader.file!;
  const partials = out.filter((o) => !o.e.complete && o.e.chunk?.kind === "page" && o.e.lod > 0);

  it("pending bricks: stored and not decoded; none before a sparse LOD's mask", () => {
    const level = partials[0]!.level;
    const { pending, count } = wvdbPendingBricks(level);
    let stored = 0;
    for (let b = 0; b < level.brickTotal; b++) if (level.mask![b >>> 5]! & (1 << (b & 31))) stored++;
    expect(count).toBe(stored - level.bricksDecoded);
    for (let i = 0; i < level.bricksDecoded; i++) {
      const b = level.brickIndices[i]!;
      expect(pending[b >>> 5]! & (1 << (b & 31))).toBe(0);
    }
    expect(wvdbPendingBricks({ ...level, mask: null }).count).toBe(0);
  });

  it("the fill: coarse texels shared by every page of a LOD, max bounds the upsample, mip atlas lazy and the CPU one", () => {
    // The last page's event has every brick in: no fill (the LOD is whole, its complete event follows).
    const filled = partials.filter((o) => o.e.lod === 2 && o.upsample.density.fill);
    const vols = filled.map((o) => o.upsample);
    expect(partials.filter((o) => o.e.lod === 2).at(-1)!.upsample.density.fill ?? null).toBeNull();
    expect(vols.length).toBeGreaterThan(2);
    for (const v of vols) {
      const f = v.density.fill!;
      expect(f.from.data).toBe(vols[0]!.density.fill!.from.data);
      expect(f.from.dimensions).toEqual(file.grids[0]!.levels[1]!.dimensions as WvdbVec3);
      expect(v.density.mip!.dimensions).toEqual(wvdbMipAtlasLayout(v.dimensions)!.dimensions);
      const r = wvdbDenseResolve(v);
      expect(r.density.fill).toBeUndefined();
      expect(f.mips).toBe("coarse");
      // Level 1 of the atlas (x offset 0): the coarse texel where the brick is pending, the box mean elsewhere.
      const box = wvdbMipAtlas(r.density.data, r.dimensions, 1)!;
      const [aw, ah] = box.dimensions;
      const [lw, lh, ld] = f.from.dimensions;
      const [cx, cy] = [Math.ceil(dims[0] / 8), Math.ceil(dims[1] / 8)];
      let fromCoarse = 0;
      for (let z = 0; z < ld; z++)
        for (let y = 0; y < lh; y++)
          for (let x = 0; x < lw; x++) {
            const b = ((2 * x) >> 3) + cx * (((2 * y) >> 3) + cy * ((2 * z) >> 3));
            const pending = (f.pending[b >>> 5]! & (1 << (b & 31))) !== 0;
            const got = r.density.mip!.data[x + aw * (y + ah * z)]!;
            expect(got).toBe(pending ? f.from.data[x + lw * (y + lh * z)]! : box.data[x + aw * (y + ah * z)]!);
            if (pending) fromCoarse++;
          }
      expect(fromCoarse).toBeGreaterThan(0);
      // fillMips 'box' (his): every level from the upsampled texels.
      const his = wvdbDenseResolve(filled.find((o) => o.upsample === v)!.boxMips);
      expect(bytesEqual(his.density.data, r.density.data)).toBe(true);
      expect(bytesEqual(his.density.mip!.data, box.data)).toBe(true);
      expect(Math.max(...r.density.data) / 255).toBeLessThanOrEqual(v.maxDensity);
      // Baked stats (the header's mean): the same on every page.
      expect(v.meanDensity).toBe(wvdbDenseVolume(readWvdb(bytes)).meanDensity);
      // The lazy atlas is the background-filled one (a host that ignores `fill`).
      expect(bytesEqual(v.density.mip!.data, wvdbMipAtlas(v.density.data, v.dimensions, 1)!.data)).toBe(true);
    }
    // 'background': no fill; the pending bricks hold the background.
    const bg = partials.filter((o) => o.e.lod === 2 && o.upsample.density.fill).at(-1)!.background;
    expect(bg.density.fill).toBeUndefined();
    expect(bytesEqual(bg.density.data, vols.at(-1)!.density.data)).toBe(true);
    expect(diff(wvdbUpsampleTexels(vols.at(-1)!.density), bg.density.data).count).toBeGreaterThan(0);
  });

  it("the completed LOD is byte-identical with fill 'upsample', 'background' or none (and to readWvdb's)", () => {
    const whole = readWvdb(bytes);
    const complete = out.filter((o) => o.e.complete);
    expect(complete.map((o) => o.e.lod)).toEqual([0, 1, 2]);
    for (const o of complete) {
      const want = wvdbDenseVolume(whole, { lod: o.e.lod });
      for (const got of [o.upsample, o.background]) {
        expect(got.density.fill ?? null).toBeNull();
        expect(bytesEqual(got.density.data, want.density.data)).toBe(true);
        expect(bytesEqual(got.density.mip!.data, want.density.mip!.data)).toBe(true);
      }
    }
  });

  it("errors: a bad fill, an event of another file", () => {
    expect(() => wvdbDensePartialVolume(file, partials[0]!.e, { fill: "nearest" as never })).toThrow(/fill must be/);
    const other = stream(bake([16, 16, 16], { pageBricks: 1 }), () => 0).events.find((e) => !e.complete && e.lod === 1)!;
    expect(() => wvdbDensePartialVolume(file, { ...other, lod: 2 })).toThrow(/no LOD 2 level/);
  });
});

// ---------------------------------------------------------------------------
// The core dense source: CPU fill = the twin; the final LOD after partial replaces = a direct upload
// ---------------------------------------------------------------------------

function textureFake() {
  const textures: { desc: GPUTextureDescriptor; data: Uint8Array | null; destroyed: boolean }[] = [];
  const device = {
    limits: {},
    createTexture(desc: GPUTextureDescriptor) {
      const t = { desc, data: null as Uint8Array | null, destroyed: false };
      textures.push(t);
      return { record: t, destroy: () => (t.destroyed = true), createView: () => ({ texture: t }) };
    },
    createSampler: (desc: GPUSamplerDescriptor) => ({ desc }),
    queue: {
      writeTexture(dst: { texture: { record: (typeof textures)[number] } }, data: Uint8Array) {
        dst.texture.record.data = new Uint8Array(data);
      },
    },
  };
  const live = () => textures.filter((t) => !t.destroyed);
  return { device: device as unknown as GPUDevice, textures, live };
}

describe("K4.22 core dense source, CPU fill", () => {
  const dims: WvdbVec3 = [48, 40, 32];
  const bytes = bake(dims, { pageBricks: 3 });
  const volumes: WvdbDenseVolume[] = stream(bytes, (e, file) => wvdbDensePartialVolume(file, e)).out;

  it("uploads the twin's bytes for every partial LOD, then the final LOD exactly as a direct upload", async () => {
    const fake = textureFake();
    const source = vdbDenseSource(fake.device, volumes[0]!, { fill: "cpu" });
    let filled = 0;
    for (const v of volumes.slice(1)) {
      source.replace(v);
      await new Promise((r) => setTimeout(r, 0)); // retired textures go in a later task (K6.10a)
      const [map, mip] = fake.live();
      const want = wvdbDenseResolve(v);
      expect(bytesEqual(map!.data!, want.density.data)).toBe(true);
      expect(bytesEqual(mip!.data!, want.density.mip!.data)).toBe(true);
      if (v.density.fill) filled++;
    }
    expect(filled).toBeGreaterThan(5);
    expect(source.fills).toEqual({ gpu: 0, cpu: filled });
    const direct = textureFake();
    vdbDenseSource(direct.device, wvdbDenseVolume(readWvdb(bytes)));
    const [a, am] = fake.live();
    const [b, bm] = direct.live();
    expect(bytesEqual(a!.data!, b!.data!)).toBe(true);
    expect(bytesEqual(am!.data!, bm!.data!)).toBe(true);
    // A new LOD size makes new textures and bumps the version; pages of one LOD don't.
    expect(source.version).toBe(2);
  });

  it("rejects an unknown fill mode", () => {
    expect(() => vdbDenseSource(textureFake().device, volumes[0]!, { fill: "wasm" as never })).toThrow(/fill must be/);
  });
});
