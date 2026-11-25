/**
 * Device-limit clamps of the core frame renderer's large textures (K6.8, K6.10a): the transmittance stack's grid
 * resolution and the volume shadow map size. Pure functions of the device's limits, so the unit tests run them on a
 * fake device's limits without allocating anything.
 *
 * Dawn zero-fills a new texture through one staging buffer of its size, with rows padded to 256 bytes. A texture
 * whose staging buffer would exceed the device's `maxBufferSize` fails the host's submit and the frame is lost, so
 * both clamps bound that staging size, not only the texture's dimensions.
 */

import { vdbFrameStackLayout } from "../../tools/frame/stack.js";
import { VDB_FRAME_SPECTRAL_GROUPS } from "./uniforms.js";
import { VDB_FRAME_SPECTRAL_GRID_WORDS } from "./wgsl/lighting.js";

/** The limits the clamps read (a `GPUSupportedLimits` subset); a missing one takes the WebGPU default. */
export interface VDBFrameLimits {
  readonly maxBufferSize?: number;
  readonly maxTextureDimension2D?: number;
  readonly maxTextureDimension3D?: number;
}

/** WebGPU's default limits (the floor every adapter meets). */
export const VDB_FRAME_DEFAULT_LIMITS = { maxBufferSize: 268435456, maxTextureDimension2D: 8192, maxTextureDimension3D: 2048 } as const;

/** Bytes of Dawn's zero-fill staging buffer for a texture: rows padded to 256 bytes, times rows and depth (or layers). */
export function vdbFrameStagingBytes(width: number, height: number, depthOrLayers: number, texelBytes: number): number {
  return Math.ceil((width * texelBytes) / 256) * 256 * height * depthOrLayers;
}

/** Why a clamp lowered a size: the stack does not tile, the texture is too large, or its staging buffer is. */
export type VDBFrameClampReason = "tile" | "dimension" | "bytes";

/** A clamped size and the reason it was lowered (`null`: unchanged). */
export interface VDBFrameClamp {
  readonly size: number;
  readonly reason: VDBFrameClampReason | null;
}

/** Targets in the transmittance stack: 4 per cascade, 3 per slot group and cascade when spectral (every group counted). */
export function vdbFrameStackTargets(cascades: number, spectral: boolean): number {
  return spectral ? 3 * VDB_FRAME_SPECTRAL_GROUPS * cascades : 4 * cascades;
}

/**
 * Why `side` does not fit the grid textures, or `null` when it does: the transmittance stack (the largest, rgba16float)
 * must tile in `maxTextureDimension3D` and its staging buffer fit `maxBufferSize`; a spectral grid's partials are the
 * shadow stack 3 words deep, which must fit `maxTextureDimension3D` too.
 */
function gridMisfit(side: number, cascades: number, spectral: boolean, maxDim: number, maxBytes: number): VDBFrameClampReason | null {
  const targets = vdbFrameStackTargets(cascades, spectral);
  const perAxis = Math.floor(maxDim / side);
  if (perAxis < 1 || targets > perAxis ** 3) return "tile";
  const stack = vdbFrameStackLayout(side, targets, maxDim).size;
  if (spectral && vdbFrameStackLayout(side, cascades, maxDim).size[2] * VDB_FRAME_SPECTRAL_GRID_WORDS > maxDim) return "tile";
  if (vdbFrameStagingBytes(stack[0], stack[1], stack[2], 8) > maxBytes) return "bytes";
  return null;
}

/**
 * K6.8 / K6.10a: the largest shadow grid resolution ≤ `resolution` (≥ 2) whose textures the device can hold. Before
 * K6.10a a resolution whose stack did not tile (above 1024 with one cascade, 1024 with 3–4 cascades) threw a
 * `RangeError` from `vdbFrameStackLayout` instead of clamping.
 */
export function vdbFrameFitGridResolution(resolution: number, cascades: number, spectral: boolean, limits: VDBFrameLimits = {}): VDBFrameClamp {
  const maxDim = limits.maxTextureDimension3D ?? VDB_FRAME_DEFAULT_LIMITS.maxTextureDimension3D;
  const maxBytes = limits.maxBufferSize ?? VDB_FRAME_DEFAULT_LIMITS.maxBufferSize;
  const start = Math.max(2, Math.floor(resolution));
  let reason: VDBFrameClampReason | null = null;
  // Start at the side the byte budget allows at all (a cube's 8 bytes per texel per target), then step down
  const targets = vdbFrameStackTargets(cascades, spectral);
  let r = Math.min(start, maxDim, Math.max(2, Math.floor(Math.cbrt(maxBytes / (8 * targets))) + 1));
  if (r < start) reason = gridMisfit(start, cascades, spectral, maxDim, maxBytes) ?? "bytes";
  for (;;) {
    const misfit = gridMisfit(r, cascades, spectral, maxDim, maxBytes);
    if (!misfit || r <= 2) break;
    reason ??= misfit;
    r--;
  }
  return { size: r, reason: r === start ? null : reason };
}

/**
 * K6.10a: the largest volume shadow map size ≤ `size` (≥ 1) the device can hold: at most `maxTextureDimension2D`, and
 * one map texture (`layers` rgba16float layers: 6 for a point light's cube, the cascades for a directional light) with
 * a staging buffer within `maxBufferSize`. A spectral map's rgb twin is a second texture of the same size.
 */
export function vdbFrameFitShadowMapSize(size: number, layers: number, limits: VDBFrameLimits = {}): VDBFrameClamp {
  const maxDim = limits.maxTextureDimension2D ?? VDB_FRAME_DEFAULT_LIMITS.maxTextureDimension2D;
  const maxBytes = limits.maxBufferSize ?? VDB_FRAME_DEFAULT_LIMITS.maxBufferSize;
  const start = Math.max(1, Math.round(size));
  let s = start;
  let reason: VDBFrameClampReason | null = null;
  if (s > maxDim) {
    s = maxDim;
    reason = "dimension";
  }
  const bytes = (side: number) => vdbFrameStagingBytes(side, side, layers, 8);
  if (bytes(s) > maxBytes) {
    reason ??= "bytes";
    s = Math.min(s, Math.max(1, Math.floor(Math.sqrt(maxBytes / (8 * layers)))));
    while (s > 1 && bytes(s) > maxBytes) s--;
  }
  return { size: s, reason: s === start ? null : reason };
}
