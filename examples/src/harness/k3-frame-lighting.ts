/**
 * Shared helpers of the k3 harnesses (k3-frame, k3-frame-shadows, k6-occluders): readback as tight rows and as a
 * bottom-first float atlas, twin statistics, and two synthetic environments (a sky with a sun; his white-furnace
 * map). K6.8: the TSL pass twins that lived here are gone with the TSL passes; the core's kernels have their CPU
 * twins in the k6-core harness (e2e/k6-core.spec.ts).
 */
import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { VDBFrameAtlas } from "../../../packages/vdb/src/tools/internal.js";

type V3 = [number, number, number];

export interface TwinStats {
  compared: number;
  maxAbs: number;
  maxRel: number;
  over: number;
  /** Fraction of compared values within tolerance. */
  within: number;
  firstBad?: unknown;
}

/** Readback as tight rows (r187: rows top first, padded to 256 bytes). */
export function tightRows<T extends Uint16Array | Float32Array>(pixels: T, w: number, h: number, channels: number): T {
  const tight = w * channels;
  const stride = h > 1 ? (pixels.length - tight) / (h - 1) : tight;
  if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${pixels.length} elements for ${w}x${h}x${channels}`);
  const out = new (pixels.constructor as new (n: number) => T)(w * h * channels);
  for (let y = 0; y < h; y++) out.set(pixels.subarray(y * stride, y * stride + tight) as never, y * tight);
  return out;
}

/** A half-float render target texture as a bottom-first float atlas (the shaders' frameFragCoord rows). */
export async function readAtlas(renderer: WebGPURenderer, rt: THREE.RenderTarget, channels: number, textureIndex = 0): Promise<VDBFrameAtlas> {
  const w = rt.width;
  const h = rt.height;
  const raw = (await (renderer.readRenderTargetPixelsAsync as (...a: unknown[]) => Promise<unknown>)(rt, 0, 0, w, h, textureIndex)) as Uint16Array | Float32Array;
  const rows = tightRows(raw, w, h, channels);
  const out = new Float32Array(w * h * channels);
  for (let y = 0; y < h; y++) {
    const src = (h - 1 - y) * w * channels;
    for (let i = 0; i < w * channels; i++) {
      const v = rows[src + i]!;
      out[y * w * channels + i] = rows instanceof Uint16Array ? THREE.DataUtils.fromHalfFloat(v) : v;
    }
  }
  return { data: out, width: w, height: h, channels };
}

export function accumulate(stats: TwinStats, gpu: number, cpu: number, absTol: number, relTol: number, where: unknown): void {
  const d = Math.abs(gpu - cpu);
  stats.compared++;
  stats.maxAbs = Math.max(stats.maxAbs, d);
  stats.maxRel = Math.max(stats.maxRel, d / Math.max(1e-3, Math.abs(cpu)));
  if (d > absTol + relTol * Math.abs(cpu)) {
    stats.over++;
    stats.firstBad ??= { ...((where as object) ?? {}), gpu, cpu };
  }
}

export const newStats = (): TwinStats => ({ compared: 0, maxAbs: 0, maxRel: 0, over: 0, within: 1 });
export const finish = (s: TwinStats) => {
  s.within = s.compared > 0 ? 1 - s.over / s.compared : 0;
  return s;
};

/**
 * A deterministic equirect sky (512 × 256 half floats): a vertical gradient (blue sky, darker ground)
 * and a small bright sun at `sun` (unit, world). Mapping: EquirectangularReflectionMapping.
 */
export function syntheticEnvironment(sun: V3 = [0.45, 0.75, 0.48]): THREE.DataTexture {
  const w = 512;
  const h = 256;
  const data = new Uint16Array(w * h * 4);
  const s = Math.hypot(...sun);
  const sd = sun.map((v) => v / s) as V3;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // three's equirect lookup: u = atan(dir.z, dir.x) / 2π + 0.5, v = asin(dir.y) / π + 0.5 (flipY false: row 0 is v = 0, the bottom).
      const u = (x + 0.5) / w;
      const v = (y + 0.5) / h;
      const phi = (u - 0.5) * 2 * Math.PI;
      const theta = (v - 0.5) * Math.PI;
      const dir: V3 = [Math.cos(theta) * Math.cos(phi), Math.sin(theta), Math.cos(theta) * Math.sin(phi)];
      const up = dir[1];
      const sky: V3 = up >= 0 ? [0.35 + 0.25 * (1 - up), 0.55 + 0.2 * (1 - up), 1.0] : [0.25, 0.22, 0.2];
      const cos = dir[0] * sd[0] + dir[1] * sd[1] + dir[2] * sd[2];
      const sunGlow = cos > 0.995 ? 60 : 0;
      const o = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) data[o + c] = THREE.DataUtils.toHalfFloat(sky[c]! + sunGlow);
      data[o + 3] = THREE.DataUtils.toHalfFloat(1);
    }
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

/** His white furnace (utils/furnace.js): a uniform equirect environment of radiance `radiance` (linear, float). */
export function furnaceEnvironment(radiance = 0.5): THREE.DataTexture {
  const w = 64;
  const h = 32;
  const data = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h; i++) data.set([radiance, radiance, radiance, 1.0], i * 4);
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}
