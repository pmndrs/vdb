// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/wvdb/wvdbKernel.js upsample L122–188, clearMissingBricks L193–211, chunk L230–344 (a
//   multi-page LOD starts from the previous LOD upsampled to its size, then each page overwrites its bricks).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: per pending brick instead of the whole level; integer weights (8 fractional bits, round half up once at
// the end) so the core's GPU kernel and this twin agree byte for byte; recomputed from the event's levels on every
// page (no state). Divergences: DIVERGES.md (K4.22 row).

/**
 * K4.22: show a `.wvdb` LOD while its pages stream in (spec §12.2). `WvdbStreamReader({ partialLods: true })` emits
 * the LOD's levels after every page; {@link wvdbDensePartialVolume} turns such an event into the dense volume the
 * frame renderer's dense source uploads. With `fill: 'upsample'` (default) the bricks that are stored but not
 * decoded yet are filled from the coarser LOD, as his worker kernel does, so the picture only gains detail; with
 * `fill: 'background'` they stay empty until their page lands.
 *
 * The fill itself runs on the GPU in the core's dense source (`vdbDenseSource`: a compute kernel fills the pending
 * bricks and rebuilds the mip atlas); {@link wvdbDenseResolve} is its CPU twin, byte for byte, and the fallback for
 * hosts without the core.
 */

import { VDBToolsError } from "../errors.js";
import { WVDB_BRICK_DIM, brickCounts, isMaskBitSet, type WvdbVec3 } from "./format.js";
import type { WvdbFile, WvdbLodEvent, WvdbVolumeGrid, WvdbVolumeLevel } from "./read.js";
import {
  wvdbDenseAssemble,
  wvdbDenseFillTexture,
  wvdbDenseVolume,
  wvdbMipAtlas,
  wvdbTexels,
  type WvdbDenseFill,
  type WvdbDenseOptions,
  type WvdbDenseTexture,
  type WvdbDenseVolume,
} from "./dense.js";

/** How {@link wvdbDensePartialVolume} shows the bricks of a partial LOD that have not arrived yet. */
export type WvdbDensePartialFill = "upsample" | "background";

/** Options of {@link wvdbDensePartialVolume}: those of `wvdbDenseVolume` (but the LOD: the event's) and the fill. */
export interface WvdbDensePartialOptions extends Omit<WvdbDenseOptions, "lod"> {
  /**
   * Bricks stored but not decoded yet: `'upsample'` (default) fills them from the finest complete coarser LOD
   * (his kernel's trilinear upsample; a `fill` on each texture, done by the core's dense source on the GPU or by
   * {@link wvdbDenseResolve}); `'background'` leaves the background there until their page lands.
   */
  fill?: WvdbDensePartialFill;
  /**
   * The mip atlas of an upsampled partial LOD (`WvdbDenseFill.mips`): `'coarse'` (default) takes level 1 of the
   * pending bricks from the coarser LOD, `'box'` builds it from the upsampled texels as his renderer does.
   */
  fillMips?: "coarse" | "box";
}

/** Weights are integers in [0, 256]: 8 fractional bits, as a GPU's trilinear filter. */
export const WVDB_UPSAMPLE_WEIGHT_ONE = 256;

/**
 * One axis of the upsample from `sourceSize` texels to `size` (his `axis`, wvdbKernel.js L125–135): target texel `i`
 * reads source position `clamp((i + ½)·sourceSize/size − ½, 0, sourceSize − 1)`; `lower`/`upper` are the two taps
 * and `weight` the upper tap's weight in 1/256 (the exact fraction rounded half up), in integers.
 */
export function wvdbUpsampleAxis(size: number, sourceSize: number): { lower: Uint32Array; upper: Uint32Array; weight: Uint32Array } {
  const lower = new Uint32Array(size);
  const upper = new Uint32Array(size);
  const weight = new Uint32Array(size);
  const den = 2 * size;
  for (let i = 0; i < size; i++) {
    // position · den = (2i + 1)·sourceSize − size, clamped to [0, (sourceSize − 1)·den]
    const num = Math.min(Math.max((2 * i + 1) * sourceSize - size, 0), (sourceSize - 1) * den);
    const lo = Math.floor(num / den);
    const r = num - lo * den;
    lower[i] = lo;
    upper[i] = Math.min(lo + 1, sourceSize - 1);
    weight[i] = Math.floor((r * WVDB_UPSAMPLE_WEIGHT_ONE + size) / den);
  }
  return { lower, upper, weight };
}

/** The bricks of `level` that are stored and not decoded yet (one bit per brick), and their count. */
export function wvdbPendingBricks(level: WvdbVolumeLevel): { pending: Uint32Array; count: number } {
  const words = Math.ceil(level.brickTotal / 32);
  const pending = new Uint32Array(words);
  let count = 0;
  // A sparse LOD whose mask has not arrived: nothing is known to be stored.
  if (level.layout === "sparse" && !level.mask) return { pending, count };
  for (let w = 0; w < words; w++) {
    const valid = w === words - 1 && level.brickTotal % 32 !== 0 ? (1 << level.brickTotal % 32) - 1 : 0xffffffff;
    const stored = level.layout === "sparse" ? level.mask![w]! : valid;
    const bits = (stored & ~level.decoded[w]! & valid) >>> 0;
    pending[w] = bits;
    let b = bits;
    while (b) {
      b &= b - 1;
      count++;
    }
  }
  return { pending, count };
}

/**
 * CPU twin of the core's fill kernel (`vdbDenseFillCs`): `tex.data` with every pending brick replaced by the
 * coarser texels upsampled to its size, per component, in integers: `c = Σ v · wx · wy · wz` over the 8 taps
 * (weights in 1/256), then `(c + 2²³) >> 24`. Without a fill, a copy of `tex.data`.
 */
export function wvdbUpsampleTexels(tex: WvdbDenseTexture): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(tex.data);
  const fill = tex.fill;
  if (!fill || fill.count === 0) return out;
  const [w, h, d] = tex.dimensions;
  const [sw, sh, sd] = fill.from.dimensions;
  const tb = tex.texelBytes;
  const src = fill.from.data;
  if (src.length !== sw * sh * sd * tb) throw new VDBToolsError("VDB_BAD_OPTION", `wvdbUpsampleTexels: the coarse texels are ${src.length} bytes, expected ${sw}x${sh}x${sd}x${tb}`);
  const ax = wvdbUpsampleAxis(w, sw);
  const ay = wvdbUpsampleAxis(h, sh);
  const az = wvdbUpsampleAxis(d, sd);
  const [cx, cy, cz] = brickCounts(tex.dimensions);
  const ONE = WVDB_UPSAMPLE_WEIGHT_ONE;
  for (let bz = 0; bz < cz; bz++) {
    for (let by = 0; by < cy; by++) {
      for (let bx = 0; bx < cx; bx++) {
        if (!isMaskBitSet(fill.pending, bx + cx * (by + cy * bz))) continue;
        for (let z = bz * WVDB_BRICK_DIM; z < Math.min(d, (bz + 1) * WVDB_BRICK_DIM); z++) {
          const z0 = az.lower[z]! * sh;
          const z1 = az.upper[z]! * sh;
          const wz = az.weight[z]!;
          for (let y = by * WVDB_BRICK_DIM; y < Math.min(h, (by + 1) * WVDB_BRICK_DIM); y++) {
            const y0 = (ay.lower[y]! + z0) * sw;
            const y1 = (ay.upper[y]! + z0) * sw;
            const y2 = (ay.lower[y]! + z1) * sw;
            const y3 = (ay.upper[y]! + z1) * sw;
            const wy = ay.weight[y]!;
            for (let x = bx * WVDB_BRICK_DIM; x < Math.min(w, (bx + 1) * WVDB_BRICK_DIM); x++) {
              const x0 = ax.lower[x]!;
              const x1 = ax.upper[x]!;
              const wx = ax.weight[x]!;
              const o = (x + (y + z * h) * w) * tb;
              for (let c = 0; c < tb; c++) {
                const at = (i: number) => src[i * tb + c]!;
                const c00 = at(x0 + y0) * (ONE - wx) + at(x1 + y0) * wx;
                const c10 = at(x0 + y1) * (ONE - wx) + at(x1 + y1) * wx;
                const c01 = at(x0 + y2) * (ONE - wx) + at(x1 + y2) * wx;
                const c11 = at(x0 + y3) * (ONE - wx) + at(x1 + y3) * wx;
                const c0 = c00 * (ONE - wy) + c10 * wy;
                const c1 = c01 * (ONE - wy) + c11 * wy;
                out[o + c] = Math.floor((c0 * (ONE - wz) + c1 * wz + 0x800000) / 0x1000000);
              }
            }
          }
        }
      }
    }
  }
  return out;
}

/** True when a fill's mip level 1 comes from its coarser LOD: `mips: 'coarse'` and the coarse size is level 1's. */
export function wvdbFillCoarseMips(tex: WvdbDenseTexture): boolean {
  const f = tex.fill;
  return !!f && (f.mips ?? "coarse") === "coarse" && f.from.dimensions.every((v, a) => v === Math.max(1, Math.ceil(tex.dimensions[a]! / 2)));
}

/**
 * CPU twin of a texture's GPU fill: the filled texels and (with a mip atlas) the atlas built from them, level 1 of
 * the pending bricks from the coarser LOD with `mips: 'coarse'` (`vdbDenseMipCs`); no `fill`.
 */
export function wvdbDenseResolveTexture(tex: WvdbDenseTexture): WvdbDenseTexture {
  const fill = tex.fill;
  if (!fill) return tex;
  const data = wvdbUpsampleTexels(tex);
  const tb = tex.texelBytes;
  const [cx, cy] = brickCounts(tex.dimensions);
  // Level-1 texel (x, y, z) box-filters level-0 texels 2x..2x+1 (clamped): one brick holds them all (bricks are 8 wide).
  const coarse = fill.count > 0 && wvdbFillCoarseMips(tex)
    ? (l: { data: Uint8Array; width: number; height: number; depth: number }) => {
        for (let z = 0; z < l.depth; z++)
          for (let y = 0; y < l.height; y++)
            for (let x = 0; x < l.width; x++) {
              const b = ((2 * x) >> 3) + cx * (((2 * y) >> 3) + cy * ((2 * z) >> 3));
              if (!isMaskBitSet(fill.pending, b)) continue;
              const i = (x + l.width * (y + l.height * z)) * tb;
              for (let c = 0; c < tb; c++) l.data[i + c] = fill.from.data[i + c]!;
            }
      }
    : undefined;
  return { data, dimensions: tex.dimensions, format: tex.format, texelBytes: tb, mip: tex.mip ? wvdbMipAtlas(data, tex.dimensions, tb, coarse) : null };
}

/**
 * The CPU twin of the core dense source's fill (K4.22): `volume` with every texture's pending bricks upsampled
 * from its coarser LOD and its mip atlas rebuilt from the result, as plain textures (no `fill`). Byte for byte
 * what `vdbDenseSource` uploads for `volume` on the GPU. A host without the core uploads this instead. The
 * stats are `volume`'s (baked, or from the texels with the pending bricks at background).
 */
export function wvdbDenseResolve(volume: WvdbDenseVolume): WvdbDenseVolume {
  const r = (t: WvdbDenseTexture | null) => (t ? wvdbDenseResolveTexture(t) : null);
  return { ...volume, density: r(volume.density)!, mask: r(volume.mask), albedo: r(volume.albedo), emission: r(volume.emission) };
}

/** Coarse texels per level and scale, so every page of a LOD shares one array (the core uploads it once). */
const coarseTexels = new WeakMap<WvdbVolumeLevel, Map<number, Uint8Array<ArrayBuffer>>>();

function coarseOf(grid: WvdbVolumeGrid, level: WvdbVolumeLevel, scale: number): Uint8Array<ArrayBuffer> {
  let byScale = coarseTexels.get(level);
  if (!byScale) coarseTexels.set(level, (byScale = new Map()));
  let data = byScale.get(scale);
  if (!data) byScale.set(scale, (data = wvdbTexels(grid, level, scale)));
  return data;
}

/**
 * The dense volume of a `.wvdb` stream event (`WvdbStreamReader({ partialLods: true })`): for a complete LOD,
 * `wvdbDenseVolume(file, { lod })`; for a partial one, the LOD's levels as decoded so far, with (by default) a
 * `fill` on each texture: its stored, undecoded bricks and the finest complete coarser LOD's texels to upsample
 * there. Pass the result to `vdbDenseSource.replace` / `vdbDenseTextures(…).replace` like any LOD; the core fills
 * on the GPU. The texels are copied now: later pages do not change the returned volume.
 *
 * The density mean is the header's baked one when the file has it (scale 1); else it comes from the decoded
 * texels (pending bricks at background), so `normalizeDensity` may move while a LOD streams in. `maxDensity`
 * also bounds the coarse texels (an upsampled texel never exceeds them).
 *
 *     const reader = new WvdbStreamReader({ partialLods: true });
 *     for (const e of reader.push(bytes)) source.replace(wvdbDensePartialVolume(reader.file!, e));
 *
 * @param file The reader's file (`reader.file`): its grids hold the complete coarser LODs.
 * @param event A stream event (partial or complete).
 * @param o Grids, scales, mipmaps (as `wvdbDenseVolume`) and the fill.
 */
export function wvdbDensePartialVolume(file: WvdbFile, event: WvdbLodEvent, o: WvdbDensePartialOptions = {}): WvdbDenseVolume {
  if (!file || !Array.isArray(file.grids)) throw new VDBToolsError("VDB_BAD_OPTION", "wvdbDensePartialVolume: pass the stream reader's file (reader.file)");
  if (!event || !Array.isArray(event.levels)) throw new VDBToolsError("VDB_BAD_OPTION", "wvdbDensePartialVolume: pass a WvdbStreamReader event");
  const fill = o.fill ?? "upsample";
  if (fill !== "upsample" && fill !== "background") throw new VDBToolsError("VDB_BAD_OPTION", `wvdbDensePartialVolume: fill must be 'upsample' or 'background', got ${JSON.stringify(o.fill)}`);
  const fillMips = o.fillMips ?? "coarse";
  if (fillMips !== "coarse" && fillMips !== "box") throw new VDBToolsError("VDB_BAD_OPTION", `wvdbDensePartialVolume: fillMips must be 'coarse' or 'box', got ${JSON.stringify(o.fillMips)}`);
  const { fill: _fill, fillMips: _fillMips, ...dense } = o;
  if (event.complete) return wvdbDenseVolume(file, { ...dense, lod: event.lod });
  const levelOf = (grid: WvdbVolumeGrid): WvdbVolumeLevel => {
    const level = event.levels[file.grids.indexOf(grid)];
    if (!level || level.kind !== "volume" || level.lod !== event.lod) {
      throw new VDBToolsError("VDB_BAD_OPTION", `wvdbDensePartialVolume: the event has no LOD ${event.lod} level of "${grid.name}" (is it this file's event?)`);
    }
    return level;
  };
  return wvdbDenseAssemble(file, dense, event.lod, {
    level: levelOf,
    texture(grid, level, data, texelBytes, mipmaps, scale) {
      let f: WvdbDenseFill | null = null;
      // The finest complete LOD below this one (the reader completes LODs coarse first).
      const coarse = fill === "upsample" ? grid.levels[Math.min(event.lod, grid.levels.length) - 1] : undefined;
      if (coarse) {
        const { pending, count } = wvdbPendingBricks(level);
        if (count > 0) f = { pending, count, from: { data: coarseOf(grid, coarse, scale), dimensions: [...coarse.dimensions] as WvdbVec3 }, mips: fillMips };
      }
      return wvdbDenseFillTexture(data, [...level.dimensions] as WvdbVec3, texelBytes, mipmaps, f);
    },
  });
}
