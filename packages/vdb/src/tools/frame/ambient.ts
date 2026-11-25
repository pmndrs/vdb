/**
 * K6.Q1 shadowed ambient (V85; DIVERGES V125–V128): the CPU twins of the core renderer's ambient occlusion. Ambient,
 * hemisphere and light-probe light (distant light with no shadow slot) is attenuated by the medium: transmittance is
 * marched along `pairs` opposite direction pairs, (ω_k, −ω_k) with ω_k in the hemisphere around the sky direction, and
 * averaged into two hemispherical visibilities (V_up over the ω_k, V_down over the −ω_k). Each direction's value is the
 * lighting mode's isotropic-light total (as his sky up / down slots): the direct transmittance without multiple
 * scattering, the octave total over an open direction's with octaves, the two-sided isotropic diffusion of the pair (white furnace: V_up + V_down
 * = 2 at albedo 1) with any other mode. The distant light then reads the visibilities as his sky up / down totals:
 * ambient × (V_up + V_down) / 2, hemisphere sky × V_up / 2 + ground × V_down / 2, the probe's SH irradiance along ± the
 * sky × V / 2. Pairs are an equal-area quadrature of each hemisphere: the sky direction, then a ring at the centroid
 * of the rest. 'frame' lighting marches the pairs in the transmittance pass (two spare channels of the fourth target);
 * 'volume' lighting marches them per froxel through the volume's own shadow march (fewer steps).
 */

import { vdbFrameIsotropicDiffusion } from "./grid.js";
import { vdbFrameOctavesOff, vdbFrameShadowTotal, type Vec3, type Vec4 } from "./index.js";
import { vdbFrameSpectralShadowTotal, vdbShHemisphereRadianceAt, type VDBFrameSpectralMarch } from "./spectral.js";

/** Most direction pairs (the uniform table's length). */
export const VDB_FRAME_AMBIENT_MAX_PAIRS = 8;

const normalize = (v: readonly number[]): Vec3 => {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!);
  return l > 0 ? [v[0]! / l, v[1]! / l, v[2]! / l] : [0, 1, 0];
};
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];

/**
 * The upper direction of each pair (world, unit): [0] the sky direction, then `pairs − 1` at cos θ = (1 − 1/pairs) / 2
 * from it (the ring's area centroid when the sky direction's cap holds 1/pairs of the hemisphere), evenly in azimuth
 * from a tangent of the sky direction. The lower directions are their negatives. K6.Q1a `phase` (0–1) turns the ring by
 * that fraction of its azimuth step (the renderer passes the frame's jitter with `jitter` and `temporal`).
 */
export function vdbFrameAmbientDirections(sky: readonly number[], pairs: number, phase = 0): Vec3[] {
  const n = Math.min(VDB_FRAME_AMBIENT_MAX_PAIRS, Math.max(1, Math.round(pairs)));
  const s = normalize(sky);
  const out: Vec3[] = [s];
  if (n === 1) return out;
  const helper: Vec3 = Math.abs(s[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const t = normalize(cross(helper, s));
  const b = cross(s, t);
  const cosTheta = (1 - 1 / n) / 2;
  const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
  for (let k = 0; k < n - 1; k++) {
    const phi = (2 * Math.PI * (k + phase)) / (n - 1);
    const c = Math.cos(phi) * sinTheta;
    const d = Math.sin(phi) * sinTheta;
    out.push(normalize([s[0] * cosTheta + t[0] * c + b[0] * d, s[1] * cosTheta + t[1] * c + b[1] * d, s[2] * cosTheta + t[2] * c + b[2] * d]));
  }
  return out;
}

/**
 * One pair's isotropic-light totals (up, down) from its two marches (direct, total, reduced depth, effective depth):
 * the two-sided isotropic diffusion with `diffusion`, else each march's total through the octave sum (octaves off:
 * the total as given) divided by the octave total of an open direction (a visibility, 1 unoccluded). The 'frame'
 * transmittance's sky up / down rule, but normalised.
 */
export function vdbFrameAmbientPairTotals(front: Vec4, back: Vec4, diffusion: boolean, octaves: Vec4 = vdbFrameOctavesOff): [number, number] {
  if (diffusion) return [vdbFrameIsotropicDiffusion(front[2], back[2], front[3]), vdbFrameIsotropicDiffusion(back[2], front[2], back[3])];
  // octaves: normalised by the octave total of no medium (1 + Σ bⁿ), so an open direction is 1 as without octaves
  const open = vdbFrameShadowTotal(1, octaves);
  return [vdbFrameShadowTotal(front[1], octaves) / open, vdbFrameShadowTotal(back[1], octaves) / open];
}

/**
 * K6.Q1a: one pair's isotropic-light totals of each channel (up, down) from its two spectral marches (direct, total,
 * reduced depth, effective depth of each channel): the two-sided isotropic diffusion of each channel with `diffusion`,
 * else each march's spectral total (octaves: at `albedo`, the spectral octave light) over an open direction's. A grey
 * medium (equal channels; octaves: albedo 1) gives {@link vdbFrameAmbientPairTotals} in every channel.
 */
export function vdbFrameSpectralAmbientPairTotals(front: VDBFrameSpectralMarch, back: VDBFrameSpectralMarch, diffusion: boolean, octaves: Vec4 = vdbFrameOctavesOff, albedo: Vec3 = [1, 1, 1]): [Vec3, Vec3] {
  const map3 = (f: (c: number) => number): Vec3 => [f(0), f(1), f(2)];
  if (diffusion) return [map3((c) => vdbFrameIsotropicDiffusion(front.z[c]!, back.z[c]!, front.w[c]!)), map3((c) => vdbFrameIsotropicDiffusion(back.z[c]!, front.z[c]!, back.w[c]!))];
  const open = vdbFrameSpectralShadowTotal([1, 1, 1], octaves, albedo);
  const up = vdbFrameSpectralShadowTotal(front.y, octaves, albedo);
  const down = vdbFrameSpectralShadowTotal(back.y, octaves, albedo);
  return [map3((c) => up[c]! / open[c]!), map3((c) => down[c]! / open[c]!)];
}

/** K6.Q1a: the hemispherical visibilities of each channel (V_up, V_down): the mean of each side's pair totals. */
export function vdbFrameSpectralAmbientVisibility(totals: readonly (readonly [Vec3, Vec3])[]): [Vec3, Vec3] {
  if (totals.length === 0) return [[1, 1, 1], [1, 1, 1]];
  const up: Vec3 = [0, 0, 0];
  const down: Vec3 = [0, 0, 0];
  for (const t of totals) for (let c = 0; c < 3; c++) {
    up[c] = up[c]! + t[0][c]! / totals.length;
    down[c] = down[c]! + t[1][c]! / totals.length;
  }
  return [up, down];
}

/** The hemispherical visibilities (V_up, V_down): the mean of each side's pair totals. */
export function vdbFrameAmbientVisibility(totals: readonly (readonly [number, number])[]): [number, number] {
  if (totals.length === 0) return [1, 1];
  let up = 0;
  let down = 0;
  for (const t of totals) {
    up += t[0];
    down += t[1];
  }
  return [up / totals.length, down / totals.length];
}

/** The distant light of a frame (renderer units: irradiance on a surface facing it, three colour × intensity). */
export interface VDBFrameDistantLight {
  ambient: Vec3;
  /** Sums of the hemisphere lights' sky and ground colours (they share the first one's up: the sky direction). */
  hemiSky: Vec3;
  hemiGround: Vec3;
  /** Light probe SH9 (three's order, RGB), or none. */
  probe?: readonly Vec3[] | null;
  /** Unit sky direction (the first hemisphere light's up, else +y). */
  sky: Vec3;
  /** K6.Q1a `ambientProbe`: the probe along ± the sky as three's irradiance (default, his) or the hemisphere's mean radiance. */
  probeMode?: VDBFrameAmbientProbe;
}

/** K6.Q1a `ambientProbe`: how the light probe's sky up / down terms read it. */
export type VDBFrameAmbientProbe = "irradiance" | "radiance";

/** three's getShIrradianceAt. */
function shIrradiance(n: Vec3, sh: readonly Vec3[]): Vec3 {
  const [x, y, z] = n;
  const k = [0.886227, 2 * 0.511664 * y, 2 * 0.511664 * z, 2 * 0.511664 * x, 2 * 0.429043 * x * y, 2 * 0.429043 * y * z, z * z * 0.743125 - 0.247708, 2 * 0.429043 * x * z, 0.429043 * (x * x - y * y)];
  const out: Vec3 = [0, 0, 0];
  for (let i = 0; i < 9; i++) for (let c = 0; c < 3; c++) out[c] = out[c]! + (sh[i]?.[c] ?? 0) * k[i]!;
  return out;
}

/** K6.Q1a: the probe's light from the hemisphere around unit `n` (the WGSL `vdbFrameSkyProbe`). */
export function vdbFrameSkyProbe(n: Vec3, sh: readonly Vec3[], mode: VDBFrameAmbientProbe = "irradiance"): Vec3 {
  return mode === "radiance" ? vdbShHemisphereRadianceAt(n, sh) : shIrradiance(n, sh);
}

/**
 * The shadowed distant-light irradiance at visibilities (V_up, V_down) (both lightings): ambient × (V_up + V_down) / 2
 * + hemisphere sky × V_up / 2 + ground × V_down / 2 + the probe's irradiance along ± the sky direction × V / 2.
 * Unoccluded (1, 1) gives the ambient and hemisphere light of the unshadowed renderer. K6.Q1a: a visibility may be per
 * channel (spectral); `probeMode` picks the probe's term.
 */
export function vdbFrameAmbientIrradiance(light: VDBFrameDistantLight, visibility: readonly [number | Vec3, number | Vec3]): Vec3 {
  const ch = (v: number | Vec3, c: number) => (typeof v === "number" ? v : v[c]!) * 0.5;
  const out: Vec3 = [0, 1, 2].map((c) => light.ambient[c]! * (ch(visibility[0], c) + ch(visibility[1], c)) + light.hemiSky[c]! * ch(visibility[0], c) + light.hemiGround[c]! * ch(visibility[1], c)) as Vec3;
  if (light.probe) {
    const a = vdbFrameSkyProbe(light.sky, light.probe, light.probeMode);
    const b = vdbFrameSkyProbe([-light.sky[0], -light.sky[1], -light.sky[2]], light.probe, light.probeMode);
    for (let c = 0; c < 3; c++) out[c] = out[c]! + a[c]! * ch(visibility[0], c) + b[c]! * ch(visibility[1], c);
  }
  return out;
}
