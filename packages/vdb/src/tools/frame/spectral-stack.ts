// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameShadowDensityMaterial.js L111–121 (the spectral shadow density),
//   FrameTransmittanceMaterial.js L278–459 (buildSpectralNodes), FrameEnvironmentGridMaterial.js L104–197 (the spectral
//   branch), FrameDensityMaterial.js L363–537 and FrameLightingMaterial.js L114–275 (the spectral density and lighting
//   passes), FrameInjectMaterial.js L200–532 (the spectral inject pass), FrameTemporalMaterial.js and
//   FrameIntegrateMaterial.js L140–217 (the spectral second target and composite).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: f64 CPU twins of the core's spectral kernels (K6.4) on stacked 3D textures (K6.2's layout): the spectral
// shadow grid (two stacks), the transmittance per slot group, the environment grid, the fused density + lighting
// gather, the 'volume' gather's combine, the resolve scan and the composite. Divergences: DIVERGES.md (K6.4 rows).

/**
 * CPU twins (f64) of the core frame renderer's spectral kernels (K6.4): the oracles of `e2e/k6-core.spec.ts`'s
 * spectral probes. The pass maths are tools/frame/spectral.ts (his K3.5 twins); these assemble them on the core's
 * stacked textures, slot-group packing and combine rules.
 */

import { vdbFrameDistanceFade, vdbFrameNoise, vdbFrameOctavesOff, vdbFrameTwoSidedDiffusion, vdbSrgbEotf, vdbSrgbOetf, type Vec2, type Vec3, type Vec4 } from "./index.js";
import { vdbFrameIntegratePrefix, vdbFrameMaxOpticalDepth } from "./froxel.js";
import { vdbFrameFindCascade, vdbFrameGridUvw, vdbFrameInsideGrid, vdbFrameDensityFroxelSteps, vdbFrameIsotropicDiffusion, type VDBFrameDensityInputs, type VDBFrameGridLayout, type VDBFrameShadowLight } from "./grid.js";
import { vdbFrameGridFragCoord, vdbFrameGridVoxelWorld } from "./stack.js";
import {
  vdbFrameMeanComponent,
  vdbFrameSpectralGridAlbedo,
  vdbFrameSpectralGridMarch,
  vdbFrameSpectralShadowMarchDepth,
  vdbFrameSpectralShadowStep,
  vdbFrameSpectralShadowTotal,
  vdbFrameSpectralShadowTransmittance,
  vdbFrameUnpackSpectralGroup,
  vdbFramePackSpectralGroup,
  type VDBFrameSpectralDepths,
  type VDBFrameSpectralMarch,
  type VDBFrameSpectralShadow,
} from "./spectral.js";
import { vdbFrameOccludedTotal } from "./occluders.js";
import { vdbFrameSpectralAmbientPairTotals, vdbFrameSpectralAmbientVisibility } from "./ambient.js";
import { vdbFrameEnvironmentIrradiance, type VDBFrameEnvironmentFlags, type VDBFrameEnvironmentGpuResult } from "./environment-gpu.js";

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const map3 = (f: (c: number) => number): Vec3 => [f(0), f(1), f(2)];
const normalize3 = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
};
const unmarched = (): VDBFrameSpectralMarch => ({ x: [1, 1, 1], y: [1, 1, 1], z: [0, 0, 0], w: [0, 0, 0] });

/** One caster as the spectral shadow density pass reads it (`VDBFrameGridCaster` + the volume's extinction spectrum). */
export interface VDBFrameSpectralGridCaster {
  readonly worldToLocal: ArrayLike<number>;
  readonly maxTexels: number;
  readonly worldScale: number;
  readonly mipmaps: boolean;
  readonly absorbance: number;
  readonly extinctionSpectrum: Vec3;
  extinctionAt(local: Vec3, lod: number): number;
  albedoAt(local: Vec3, lod: number): Vec3;
}

/**
 * The spectral shadow density pass at one voxel (`vdbFrameGridDensityCs`, spectral): the summed extinction
 * (spectrum × extinction) and absorption (× (1 − albedo) of each channel) of the casters whose box holds the voxel.
 */
export function vdbFrameSpectralGridDensityVoxel(world: Vec3, voxelSize: number, casters: readonly VDBFrameSpectralGridCaster[]): { extinction: Vec3; absorption: Vec3 } {
  const extinction: Vec3 = [0, 0, 0];
  const absorption: Vec3 = [0, 0, 0];
  for (const c of casters) {
    const m = c.worldToLocal;
    const local = [0, 1, 2].map((r) => m[r]! * world[0] + m[4 + r]! * world[1] + m[8 + r]! * world[2] + m[12 + r]!) as Vec3;
    if (Math.max(Math.abs(local[0]), Math.abs(local[1]), Math.abs(local[2])) > 0.5) continue;
    const lod = Math.max(0, Math.log2((voxelSize / c.worldScale) * c.maxTexels));
    const readLod = c.mipmaps ? lod : 0;
    const e = c.extinctionAt(local, readLod) * c.absorbance;
    if (!(e > 0)) continue;
    const albedo = c.albedoAt(local, readLod);
    for (let k = 0; k < 3; k++) {
      const s = c.extinctionSpectrum[k]! * e;
      extinction[k] = extinction[k]! + s;
      absorption[k] = absorption[k]! + s * (1 - clamp01(albedo[k]!));
    }
  }
  return { extinction, absorption };
}

/** The spectral grid at a cascade's uvw: (extinction, absorption) of each channel (the two stacks). */
export type VDBFrameSpectralGridAt = (cascade: number, uvw: Vec3) => { extinction: Vec3; absorption: Vec3 };

/**
 * The spectral transmittance kernel at voxel `cell` of `cascade` for slot group `group` (his buildSpectralNodes on the
 * core's stacks): the 3 packed targets. Group 0: (sky up, sky down) and slot 0; group g: slots 2g − 1 and 2g; a slot
 * entry is (total, direct). `lights` are the slots (≤ 6, incl. the environment slot); `visibility(slot, world)` the K6.5b
 * occluder of the scene slots (< `sceneSlots`). Voxels with no extinction within half a voxel keep 1.
 */
export function vdbFrameSpectralTransmittanceVoxel3d(
  gridAt: VDBFrameSpectralGridAt,
  grid: VDBFrameGridLayout,
  cascade: number,
  cell: Vec3,
  lights: readonly VDBFrameShadowLight[],
  skyDirection: Vec3,
  multipleScattering: number,
  jitter: number,
  group: number,
  o: {
    octaves?: Vec4;
    twoSided?: boolean;
    visibility?: ((slot: number, world: Vec3) => number) | null;
    sceneSlots?: number;
    /**
     * K6.Q1a shadowed ambient: the pairs' upper directions (`vdbFrameAmbientDirections`; [0] the sky direction): group
     * 0's sky entry becomes the hemispherical visibilities of each channel (V up, V down). Default none (the sky totals).
     */
    ambientDirections?: readonly Vec3[] | null;
  } = {},
): [Vec4, Vec4, Vec4] {
  const ones: [Vec4, Vec4, Vec4] = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
  ];
  const octaves = o.octaves ?? vdbFrameOctavesOff;
  const world = vdbFrameGridVoxelWorld(grid, cascade, cell);
  const voxel = grid.gridVoxel[cascade]!;
  const sampleAt = (p: Vec3) => {
    const c = vdbFrameFindCascade(grid, p, 0);
    return c < 0 ? { extinction: [0, 0, 0] as Vec3, absorption: [0, 0, 0] as Vec3 } : gridAt(c, vdbFrameGridUvw(grid, c, p));
  };
  let near = 0;
  for (let i = 0; i < 8; i++) {
    const c: Vec3 = [(i & 1) - 0.5, ((i >> 1) & 1) - 0.5, ((i >> 2) & 1) - 0.5];
    const e = sampleAt([world[0] + c[0] * voxel[0], world[1] + c[1] * voxel[1], world[2] + c[2] * voxel[2]]).extinction;
    near += e[0] + e[1] + e[2];
  }
  if (!(near > 0)) return ones;
  const noise = vdbFrameNoise(jitter, vdbFrameGridFragCoord(grid.resolution, grid.columns, cascade, cell));
  const march = (dir: Vec3, maxDistance: number) => vdbFrameSpectralGridMarch(sampleAt, grid, world, dir, maxDistance, noise, multipleScattering, octaves);
  const slot = (i: number): VDBFrameSpectralMarch => {
    if (i >= Math.min(lights.length, 6)) return unmarched();
    const l = lights[i]!;
    const isDirection = l[3] < 1.5;
    const toLight: Vec3 = [l[0] - world[0], l[1] - world[1], l[2] - world[2]];
    const dir = isDirection ? normalize3([l[0], l[1], l[2]]) : normalize3(toLight);
    const front = march(dir, isDirection ? 1e20 : Math.hypot(...toLight));
    if (o.twoSided) {
      const back = march([-dir[0], -dir[1], -dir[2]], 1e20);
      front.y = map3((c) => vdbFrameTwoSidedDiffusion(front.x[c]!, front.z[c]!, front.w[c]!, back.z[c]!));
    }
    // K6.5b: the scene slot's scattered part × the voxel's occluder visibility (octaves: the gather occludes)
    if (o.visibility && i < (o.sceneSlots ?? lights.length) && !(octaves[0] > 0)) {
      const v = o.visibility(i, world);
      front.y = map3((c) => vdbFrameOccludedTotal(front.x[c]!, front.y[c]!, v));
    }
    return front;
  };
  const entry = (m: VDBFrameSpectralMarch): VDBFrameSpectralShadow => ({ x: m.y, y: m.x });
  if (group === 0) {
    const sky = normalize3(skyDirection);
    const up = march(sky, 1e20);
    const down = march([-sky[0], -sky[1], -sky[2]], 1e20);
    const total = (f: VDBFrameSpectralMarch, b: VDBFrameSpectralMarch) => (multipleScattering > 0 ? map3((c) => vdbFrameIsotropicDiffusion(f.z[c]!, b.z[c]!, f.w[c]!)) : f.y);
    const dirs = o.ambientDirections;
    if (dirs && dirs.length > 0) {
      // K6.Q1a: each pair's isotropic-light totals of each channel (pair 0: the sky marches), averaged per side; octaves
      // at the voxel's grid albedo (its own cascade's texel)
      const centre = gridAt(cascade, vdbFrameGridUvw(grid, cascade, world));
      const albedo = vdbFrameSpectralGridAlbedo(centre.extinction, centre.absorption);
      const totals = dirs.map((d, k) => {
        const front = k === 0 ? up : march(d, 1e20);
        const back = k === 0 ? down : march([-d[0], -d[1], -d[2]], 1e20);
        return vdbFrameSpectralAmbientPairTotals(front, back, multipleScattering > 0, octaves, albedo);
      });
      const v = vdbFrameSpectralAmbientVisibility(totals);
      return vdbFramePackSpectralGroup({ x: v[0], y: v[1] }, entry(slot(0)));
    }
    return vdbFramePackSpectralGroup({ x: total(up, down), y: total(down, up) }, entry(slot(0)));
  }
  return vdbFramePackSpectralGroup(entry(slot(2 * group - 1)), entry(slot(2 * group)));
}

/**
 * The spectral environment grid kernel at voxel `cell` of the last cascade (his spectral branch on the core's stacks):
 * one-voxel-step marches of each channel that stay in the last cascade, the voxel's albedo for the octaves; diffusion:
 * the two-sided total of each channel. Outside the casters: the unshadowed light × the octave total of 1.
 */
export function vdbFrameSpectralEnvironmentGridVoxel3d(
  gridAt: VDBFrameSpectralGridAt,
  grid: VDBFrameGridLayout,
  cell: Vec3,
  lights: readonly { direction: Vec3; color: Vec3 }[],
  multipleScattering: number,
  octaves: Vec4 = vdbFrameOctavesOff,
): Vec4 {
  const cascade = grid.count - 1;
  const world = vdbFrameGridVoxelWorld(grid, cascade, cell);
  const voxel = grid.gridVoxel[cascade]!;
  const uvwOf = (p: Vec3) => vdbFrameGridUvw(grid, cascade, p);
  const unshadowed: Vec3 = [0, 0, 0];
  for (const l of lights) for (let c = 0; c < 3; c++) unshadowed[c] = unshadowed[c]! + l.color[c]!;
  let near = 0;
  for (let i = 0; i < 8; i++) {
    const c: Vec3 = [(i & 1) - 0.5, ((i >> 1) & 1) - 0.5, ((i >> 2) & 1) - 0.5];
    const e = gridAt(cascade, uvwOf([world[0] + c[0] * voxel[0], world[1] + c[1] * voxel[1], world[2] + c[2] * voxel[2]])).extinction;
    near += e[0] + e[1] + e[2];
  }
  const outside = vdbFrameSpectralShadowTotal([1, 1, 1], octaves, [1, 1, 1]);
  if (!(near > 0)) return [Math.max(unshadowed[0], 0) * outside[0], Math.max(unshadowed[1], 0) * outside[1], Math.max(unshadowed[2], 0) * outside[2], 1];
  const centre = gridAt(cascade, uvwOf(world));
  const albedo = vdbFrameSpectralGridAlbedo(centre.extinction, centre.absorption);
  const marchDepths = (dir: Vec3): VDBFrameSpectralDepths => {
    const step = 1 / Math.max(Math.abs(dir[0]) / voxel[0], Math.abs(dir[1]) / voxel[1], Math.abs(dir[2]) / voxel[2]);
    const probe: Vec3 = [world[0] + dir[0] * step * 0.5, world[1] + dir[1] * step * 0.5, world[2] + dir[2] * step * 0.5];
    const d: VDBFrameSpectralDepths = { x: [0, 0, 0], y: [0, 0, 0], z: [0, 0, 0] };
    for (let i = 0; i < 1024; i++) {
      const uvw = uvwOf(probe);
      if (uvw.some((v) => v < 0 || v > 1) || Math.min(...vdbFrameSpectralShadowMarchDepth(d, multipleScattering, octaves)) > 6) break;
      const s = gridAt(cascade, uvw);
      const st = vdbFrameSpectralShadowStep(s.extinction, s.absorption, multipleScattering);
      for (let c = 0; c < 3; c++) {
        d.x[c] = d.x[c]! + st.x[c]! * step;
        d.y[c] = d.y[c]! + st.y[c]! * step;
        d.z[c] = d.z[c]! + st.z[c]! * step;
        probe[c] = probe[c]! + dir[c]! * step;
      }
    }
    return d;
  };
  const irradiance: Vec3 = [0, 0, 0];
  for (const l of lights) {
    const d = marchDepths(l.direction);
    const total =
      multipleScattering > 0
        ? map3((c) => vdbFrameIsotropicDiffusion(d.z[c]!, marchDepths([-l.direction[0], -l.direction[1], -l.direction[2]]).z[c]!, d.y[c]!))
        : vdbFrameSpectralShadowTotal(vdbFrameSpectralShadowTransmittance(d, multipleScattering, octaves).y, octaves, albedo);
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + l.color[c]! * total[c]!;
  }
  return [Math.max(irradiance[0], 0), Math.max(irradiance[1], 0), Math.max(irradiance[2], 0), 1];
}

const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
};

/**
 * sampleShadows of every slot group in use at a world position (the core's spectral stack: `targetAt(cascade, group, k,
 * uvw)`), 1 outside the grid, blended toward the next cascade within 10% of a cascade's sides. Groups past `groups`: 1.
 */
export function vdbFrameSpectralStackGroups(targetAt: (cascade: number, group: number, k: number, uvw: Vec3) => number[], grid: VDBFrameGridLayout, world: Vec3, groups: number): number[][][] {
  const out = [0, 1, 2, 3].map(() => [0, 1, 2].map(() => [1, 1, 1, 1]));
  const cascade = vdbFrameFindCascade(grid, world, 0);
  if (cascade < 0) return out;
  const uvw = vdbFrameGridUvw(grid, cascade, world);
  const side = Math.min(uvw[0], uvw[1], uvw[2], 1 - uvw[0], 1 - uvw[1], 1 - uvw[2]);
  const blend = 1 - smoothstep(0, 0.1, side);
  const next = blend > 0 ? vdbFrameFindCascade(grid, world, cascade + 1) : -1;
  for (let g = 0; g < Math.min(groups, 4); g++) {
    for (let k = 0; k < 3; k++) {
      let v = targetAt(cascade, g, k, uvw);
      if (next >= 0) {
        const w = targetAt(next, g, k, vdbFrameGridUvw(grid, next, world));
        v = v.map((x, c) => x + (w[c]! - x) * blend);
      }
      out[g]![k] = v;
    }
  }
  return out;
}

/** The spectral density sums of one volume in one froxel (the core's `vdbFrameDensitySums`, spectral). */
export interface VDBFrameSpectralSums {
  /** (albedo · extinction rgb, weight = the channel mean of the extinction). */
  a: Vec4;
  /** (emission · weight, visible t · weight). */
  e: Vec4;
  /** (extinction rgb, 0). */
  x: Vec4;
}

/** The inputs of {@link vdbFrameSpectralDensitySums}: the grey density inputs (hooks), the extinction spectrum, the environment term. */
export interface VDBFrameSpectralDensityInputs3d extends Pick<VDBFrameDensityInputs, "froxelGrid" | "columns" | "range" | "projectionInverse" | "cameraWorld" | "viewToLocal" | "maxTexels" | "mipLevels" | "worldScale" | "subSteps" | "jitter" | "fadeRadius" | "opacity" | "baseColor" | "scatterColor" | "extinctionAt" | "albedoAt" | "emissionAt"> {
  extinctionSpectrum: Vec3;
  /** The environment irradiance added as albedo · E / π at the visible world and local position, given the albedo. */
  environmentAt?: ((world: Vec3, local: Vec3, albedo: Vec3) => Vec3) | null;
}

/**
 * The core's spectral density sums of one volume in one froxel (his spectral density pass): the pre-integration with the
 * transmittance of each channel (the visible position weighs the channel mean), the extinction-weighted sums; null when
 * the weight is not > 0. `fragCoord` is his atlas pixel.
 */
export function vdbFrameSpectralDensitySums(fragCoord: Vec2, p: VDBFrameSpectralDensityInputs3d): VDBFrameSpectralSums | null {
  const f = vdbFrameDensityFroxelSteps(fragCoord, p);
  if (!f) return null;
  const { viewNear, viewFar, localNear, localFar, t0, segment, steps, subStepLength, lod, jitter, mul } = f;
  const lerp3 = (x: Vec3, y: Vec3, t: number): Vec3 => [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t];
  const spectrum = p.extinctionSpectrum;
  let extinctionSum = 0;
  const transmittance: Vec3 = [1, 1, 1];
  let visibleWeight = 0;
  let visibleTSum = 0;
  for (let i = 0; i < steps; i++) {
    const t = t0 + segment * ((i + jitter) / steps);
    const e = p.extinctionAt(lerp3(localNear, localFar, t), lod) * p.opacity * vdbFrameDistanceFade(p.fadeRadius, lerp3(viewNear, viewFar, t));
    const stepT = map3((c) => Math.exp(-spectrum[c]! * e * subStepLength));
    const w = vdbFrameMeanComponent(map3((c) => transmittance[c]! * (1 - stepT[c]!)));
    visibleWeight += w;
    visibleTSum += w * t;
    extinctionSum += e;
    for (let c = 0; c < 3; c++) transmittance[c] = transmittance[c]! * stepT[c]!;
  }
  const extinction = map3((c) => spectrum[c]! * ((extinctionSum / steps) * segment));
  const weight = vdbFrameMeanComponent(extinction);
  if (!(weight > 0)) return null;
  const visibleT = visibleWeight > 0 ? visibleTSum / visibleWeight : 0.5;
  const local = lerp3(localNear, localFar, visibleT);
  const world = mul(p.cameraWorld, lerp3(viewNear, viewFar, visibleT));
  const albedo: Vec3 = p.albedoAt ? p.albedoAt(local, lod) : [...p.baseColor];
  const own = p.emissionAt ? p.emissionAt(local, lod) : ([0, 0, 0] as Vec3);
  const emission: Vec3 = map3((c) => own[c]! + p.scatterColor[c]!);
  if (p.environmentAt) {
    const env = p.environmentAt(world, local, albedo);
    for (let c = 0; c < 3; c++) emission[c] = emission[c]! + (albedo[c]! * env[c]!) / Math.PI;
  }
  return {
    a: [albedo[0] * extinction[0], albedo[1] * extinction[1], albedo[2] * extinction[2], weight],
    e: [emission[0] * weight, emission[1] * weight, emission[2] * weight, visibleT * weight],
    x: [extinction[0], extinction[1], extinction[2], 0],
  };
}

/**
 * The core's spectral environment term of the density pass (`vdbFrameEnvironmentDensityTerm` of each channel): the
 * environment grid at the visible position (outside: the unshadowed light × the spectral octave total of 1), nothing
 * with the environment slot, else 'outward' SH irradiance shadowed between the sky down and up totals of each channel
 * (`sky` (up, down) are group 0's entry a at the visible position, before the totals; K6.Q1a `ambientOcclusion`: the
 * hemispherical visibilities, read as they are).
 */
export function vdbFrameSpectralEnvironmentDensityTerm(
  env: VDBFrameEnvironmentGpuResult,
  flags: VDBFrameEnvironmentFlags,
  outward: Vec3,
  skyDirection: Vec3,
  sky: VDBFrameSpectralShadow,
  albedo: Vec3,
  gridAt: () => Vec3 | null,
  octaves: Vec4 = vdbFrameOctavesOff,
  ambientOcclusion = false,
): Vec3 {
  if (!env.valid) return [0, 0, 0];
  const first = flags.slot ? 1 : 0;
  if (flags.useShadows && flags.grid && env.count > first) {
    const inside = gridAt();
    if (inside) return inside;
    const sum: Vec3 = [0, 0, 0];
    for (let i = first; i < env.count; i++) for (let c = 0; c < 3; c++) sum[c] = sum[c]! + env.colors[i]![c]!;
    const total = vdbFrameSpectralShadowTotal([1, 1, 1], octaves, albedo);
    return map3((c) => Math.max(sum[c]!, 0) * total[c]!);
  }
  if (flags.useShadows && flags.slot && env.count > 0) return [0, 0, 0];
  const blend = (outward[0] * skyDirection[0] + outward[1] * skyDirection[1] + outward[2] * skyDirection[2]) * 0.5 + 0.5;
  // K6.Q1a: with the shadowed ambient the entry holds the hemispherical visibilities (totals already)
  const up = ambientOcclusion ? sky.x : vdbFrameSpectralShadowTotal(sky.x, octaves, albedo);
  const down = ambientOcclusion ? sky.y : vdbFrameSpectralShadowTotal(sky.y, octaves, albedo);
  const shadow = map3((c) => down[c]! + (up[c]! - down[c]!) * blend);
  const e = vdbFrameEnvironmentIrradiance(env.sh, outward);
  if (!flags.diffusion) return map3((c) => e[c]! * shadow[c]!);
  const inward = map3((c) => up[c]! + (down[c]! - up[c]!) * blend);
  const back = vdbFrameEnvironmentIrradiance(env.sh, [-outward[0], -outward[1], -outward[2]]);
  return map3((c) => (e[c]! * shadow[c]! + back[c]! * inward[c]!) * 0.5);
}

/** One volume's spectral froxel in the 'volume' gather: unpremultiplied radiance and the alpha of each channel. */
export interface VDBFrameSpectralVolumeFroxel {
  radiance: Vec3;
  alpha: Vec3;
}

/**
 * The 'volume' gather's spectral combine of the volumes in one froxel: (L · α rgb, α rgb). `'over'` per channel in record
 * order; `'mixture'` per channel (τ = −log(1 − α), α = 1 − exp(−Στ), L = ΣτL / Στ); one volume gives (Lα, α).
 */
export function vdbFrameGatherFroxelSpectral(volumes: readonly VDBFrameSpectralVolumeFroxel[], mode: "mixture" | "over"): { color: Vec3; alpha: Vec3 } {
  const lit = volumes.filter((v) => Math.max(...v.alpha) > 0);
  if (lit.length === 0) return { color: [0, 0, 0], alpha: [0, 0, 0] };
  if (lit.length === 1) {
    const v = lit[0]!;
    return { color: map3((c) => v.radiance[c]! * v.alpha[c]!), alpha: [...v.alpha] };
  }
  if (mode === "over") {
    const color: Vec3 = [0, 0, 0];
    const alpha: Vec3 = [0, 0, 0];
    for (const v of lit) {
      for (let c = 0; c < 3; c++) {
        const k = 1 - v.alpha[c]!;
        color[c] = v.radiance[c]! * v.alpha[c]! + color[c]! * k;
        alpha[c] = v.alpha[c]! + alpha[c]! * k;
      }
    }
    return { color, alpha };
  }
  const tau: Vec3 = [0, 0, 0];
  const tl: Vec3 = [0, 0, 0];
  for (const v of lit) {
    for (let c = 0; c < 3; c++) {
      const t = Math.min(-Math.log(Math.max(1 - v.alpha[c]!, 0)), vdbFrameMaxOpticalDepth);
      tau[c] = tau[c]! + t;
      tl[c] = tl[c]! + t * v.radiance[c]!;
    }
  }
  const alpha = map3((c) => 1 - Math.exp(-tau[c]!));
  return { color: map3((c) => (tau[c]! > 0 ? (tl[c]! / tau[c]!) * alpha[c]! : 0)), alpha };
}

/**
 * The spectral resolve scan of one column (his spectral integrate loop over whole slices): `froxels[z]` = (scatter
 * (L·α rgb, α green), alpha (α red, α blue, 0, α green)). Boundary k holds (Σ radiance rgb, 1 − T green) and the alpha
 * prefix (1 − T red, 1 − T blue); a slice adds nothing once every channel's T < 0.001 (his maxComponent break).
 */
export function vdbFrameScanColumnSpectral(froxels: readonly { scatter: Vec4; alpha: Vec4 }[], o: { stop?: number } = {}): { prefix: Vec4[]; alpha: Vec2[] } {
  const stop = o.stop ?? 0.001;
  const prefix: Vec4[] = [[0, 0, 0, 0]];
  const alpha: Vec2[] = [[0, 0]];
  const L: Vec3 = [0, 0, 0];
  const T: Vec3 = [1, 1, 1];
  for (const f of froxels) {
    const a: Vec3 = [clamp01(f.alpha[0]), clamp01(f.scatter[3]), clamp01(f.alpha[1])];
    if (Math.max(...T) >= stop && Math.max(...a) > 0) {
      for (let c = 0; c < 3; c++) {
        L[c] = L[c]! + T[c]! * f.scatter[c]!;
        T[c] = T[c]! * (1 - a[c]!);
      }
    }
    prefix.push([L[0], L[1], L[2], 1 - T[1]]);
    alpha.push([1 - T[0], 1 - T[2]]);
  }
  return { prefix, alpha };
}

/**
 * The spectral composite's two taps (`vdbFrameIntegrate` spectral): the grey two-tap formula for each channel, the red
 * and blue opacity from the alpha prefix. Returns (radiance, transmittance of each channel).
 */
export function vdbFrameIntegratePrefixSpectral(a: Vec4, aAlpha: Vec2, b: Vec4, bAlpha: Vec2, c: number): { radiance: Vec3; transmittance: Vec3 } {
  const r = vdbFrameIntegratePrefix([a[0], 0, 0, aAlpha[0]], [b[0], 0, 0, bAlpha[0]], c);
  const g = vdbFrameIntegratePrefix([a[1], 0, 0, a[3]], [b[1], 0, 0, b[3]], c);
  const bl = vdbFrameIntegratePrefix([a[2], 0, 0, aAlpha[1]], [b[2], 0, 0, bAlpha[1]], c);
  return { radiance: [r.radiance[0], g.radiance[0], bl.radiance[0]], transmittance: [r.transmittance, g.transmittance, bl.transmittance] };
}

/**
 * His spectral composite (`vdbFrameCompositeSpectralColor`): the scene colour × the transmittance of each channel + the
 * radiance, or (sRGB) the per-channel mix in sRGB (clamped to [0, 1] when `clampColors`); alpha from green.
 */
export function vdbFrameCompositeSpectral(radiance: Vec3, transmittance: Vec3, scene: Vec4, srgb: boolean, clampColors = true): Vec4 {
  const alpha = map3((c) => 1 - transmittance[c]!);
  const outAlpha = alpha[1] + scene[3] * (1 - alpha[1]);
  if (!srgb) return [radiance[0] + scene[0] * transmittance[0], radiance[1] + scene[1] * transmittance[1], radiance[2] + scene[2] * transmittance[2], outAlpha];
  const enc = (x: number) => (clampColors ? clamp01(vdbSrgbOetf(x)) : vdbSrgbOetf(x));
  const out = map3((c) => {
    const bg = enc(scene[c]!);
    const vol = enc(radiance[c]! / Math.max(alpha[c]!, 1e-6));
    return vdbSrgbEotf(bg + (vol - bg) * alpha[c]!);
  });
  return [out[0], out[1], out[2], outAlpha];
}

/** Re-exported for the spectral 'frame' lighting twin: group entries of the core's 3 packed targets. */
export { vdbFrameUnpackSpectralGroup };
