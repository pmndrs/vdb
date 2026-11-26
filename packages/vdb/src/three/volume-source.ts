// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameVolume.js L14–22, L357–405 (the dense texture setup this handle describes);
//   src/three-openvdb/frame/webgpu/frameMipmaps.js L183–189 (mip atlas texture).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; K6.11 (D35): a handle only. His `createVolumeUniforms` / `createVolumeNodes` shape (the TSL half,
// `vdbVolumeUniforms` / `vdbVolumeNodes`) and the three textures are deleted: the core's `vdbDenseSource` reads the volume.
// Divergences: DIVERGES.md V23–V25.

/**
 * The dense `.wvdb` volume source in three (K3.0, R1 §4.2): a handle `VDBFrameVolume` draws.
 *
 *   const file = readWvdb(bytes);
 *   const volume = wvdbDenseVolume(file);                 // @pmndrs/vdb/tools: his 8-bit texels + mip atlas
 *   const source = vdbDenseTextures(volume);              // the handle
 *   const frameVolume = new VDBFrameVolume(source, { extinction: 100 });
 *
 * K6.11: the handle holds no GPU object. `VDBFrameRenderer` reads `source.volume` into the core's dense source
 * (`vdbDenseSource`, which makes the textures and the `vdbSrc*` WGSL). A custom TSL material samples a pooled grid
 * through `VDBGridNode`; custom WGSL or compute uses the core source's `wgsl()`.
 */

import { VDBError } from "../core/errors.js";
import type { VDBVolumeChannels, VDBVolumeSourceInfo } from "../core/source/types.js";
import type { WvdbDenseTexture, WvdbDenseVolume } from "../tools/wvdb/dense.js";
import type { VDBThreeSparseSource } from "./sparse-source.js";

type ChannelName = "density" | "mask" | "albedo" | "emission";
const CHANNELS: readonly ChannelName[] = ["density", "mask", "albedo", "emission"];

/** A dense source as a handle (the descriptive part of `VDBVolumeSource`; no raw bindings). */
export interface VDBThreeDenseSource extends VDBVolumeSourceInfo {
  /** Always `"dense"`. */
  readonly kind: "dense";
  /** The `.wvdb` volume currently set (`VDBFrameVolume`: the core frame renderer uploads it into its own textures). */
  readonly volume: WvdbDenseVolume;
  /**
   * Swap in another LOD or sequence frame. New sizes bump `version`, and so does a frame with another `.wvdb` box
   * (a `VDBFrameVolume` follows by itself, its box too unless it was given a `bbox`). A channel the source lacks is
   * ignored (warned once). The density's mip atlas must stay if the source has one.
   */
  replace(volume: WvdbDenseVolume): void;
  /** Releases the handle (it holds no GPU object). */
  dispose(): void;
}

/** A volume source `VDBFrameVolume` reads: a dense `.wvdb` volume or a pool grid. */
export type VDBThreeVolumeSource = VDBThreeDenseSource | VDBThreeSparseSource;

const channelsOf = (v: WvdbDenseVolume): Record<ChannelName, WvdbDenseTexture | null> => ({
  density: v.density,
  mask: v.mask,
  albedo: v.albedo,
  emission: v.emission,
});

/** Which channels (and whether each has a mip atlas) a dense source was made with; fixed for its life. */
type DenseShape = Record<ChannelName, { mip: boolean } | null>;

const sameDims = (a: readonly number[] | undefined, b: readonly number[] | undefined): boolean => !!a && !!b && a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

class ThreeDenseSource implements VDBThreeDenseSource {
  readonly kind = "dense" as const;
  readonly boxToSource = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0.5, 0.5, 1]);
  readonly channels: VDBVolumeChannels;
  volume: WvdbDenseVolume;
  texels: [number, number, number];
  lodLevels: number;
  stats: { meanNonZero: number; max: number; activeVoxels: number };
  version = 0;
  private readonly warnedExtra = new Set<ChannelName>();
  readonly #shape: DenseShape;

  constructor(volume: WvdbDenseVolume) {
    this.volume = volume;
    this.channels = { mask: !!volume.mask, albedo: !!volume.albedo, emission: !!volume.emission };
    const c = channelsOf(volume);
    this.#shape = Object.fromEntries(CHANNELS.map((n) => [n, c[n] ? { mip: !!c[n]!.mip } : null])) as DenseShape;
    this.texels = [...volume.dimensions] as [number, number, number];
    this.lodLevels = volume.density.mip?.levels ?? 0;
    this.stats = { meanNonZero: volume.meanDensity, max: volume.maxDensity, activeVoxels: volume.activeTexels };
  }

  replace(volume: WvdbDenseVolume): void {
    if (!volume || volume.kind !== "dense" || !volume.density) {
      throw new VDBError("VDB_BAD_OPTION", "vdbDenseTextures.replace: pass a dense volume from wvdbDenseVolume() (@pmndrs/vdb/tools).");
    }
    const shape = this.#shape;
    const next = channelsOf(volume);
    const densityMip = next.density!.mip;
    if (!!shape.density!.mip !== !!densityMip && shape.density!.mip) {
      throw new VDBError("VDB_BAD_OPTION", "vdbDenseTextures.replace: the frame's density has no mip atlas but the source's has; bake every frame with mipmaps, or make the source without.");
    }
    // K5.9 (sequences): a channel the source has and the frame lacks reads its absent value (mask and albedo 1,
    // emission 0; the core's dense source does that); a channel (or mip atlas) the source lacks is ignored, since the
    // shaders were built without it (warned once: make the source from a frame that has every channel).
    for (const name of CHANNELS) {
      const s = shape[name];
      const v = next[name];
      if (!s && v) {
        if (!this.warnedExtra.has(name)) {
          this.warnedExtra.add(name);
          console.warn(`vdbDenseTextures.replace: the frame has a ${name} channel the source was made without; it is ignored.`);
        }
        next[name] = null;
      } else if (s && v && s.mip !== !!v.mip && s.mip) {
        throw new VDBError("VDB_BAD_OPTION", `vdbDenseTextures.replace: the frame's ${name} channel has no mip atlas but the source's has; make a new source.`);
      }
    }
    // WVDBLoader.js L37–48 updateTexture: a new size is a new texture (here: bumps `version`).
    const before = channelsOf(this.volume);
    let resized = false;
    for (const name of CHANNELS) {
      const o = before[name];
      const v = next[name];
      if (!o || !v) continue;
      if (!sameDims(o.dimensions, v.dimensions) || (!!o.mip && !!v.mip && !sameDims(o.mip.dimensions, v.mip.dimensions))) resized = true;
    }
    // A frame with another world box (his WVDB sequences give each frame its own) also bumps `version`, so a
    // VDBFrameVolume whose box is the source's moves its box (an empty frame's null box keeps the last one).
    const a = this.volume.bbox;
    const b = volume.bbox;
    const moved = !!b && (!a || a.some((corner, i) => corner.some((v, k) => v !== b[i]![k])));
    if (resized || moved) this.version++;
    this.volume = volume;
    this.texels = [...volume.dimensions] as [number, number, number];
    this.lodLevels = shape.density!.mip ? (volume.density.mip?.levels ?? 0) : 0;
    this.stats = { meanNonZero: volume.meanDensity, max: volume.maxDensity, activeVoxels: volume.activeTexels };
  }

  dispose(): void {
    // No GPU object: the core's dense source (the frame renderer's) owns the textures.
  }
}

/** A dense `.wvdb` volume (`wvdbDenseVolume`, `@pmndrs/vdb/tools`) as a handle for `VDBFrameVolume`. */
export function vdbDenseTextures(volume: WvdbDenseVolume): VDBThreeDenseSource {
  if (!volume || volume.kind !== "dense" || !volume.density) {
    throw new VDBError("VDB_BAD_OPTION", "vdbDenseTextures: pass a dense volume from wvdbDenseVolume() (@pmndrs/vdb/tools).");
  }
  return new ThreeDenseSource(volume);
}
