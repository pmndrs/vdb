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
import { vdbFrameMieAsymmetry, vdbFrameMieRgbMode } from "./mie.js";
import { vdbFrameSpectralShadowTotal, type VDBFrameSpectralMarch } from "./spectral.js";

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
 * from a tangent of the sky direction. The lower directions are their negatives. The ring is fixed (K6.Q1c, D37 (5):
 * K6.Q1a turned it per frame with `jitter` and `temporal`).
 */
export function vdbFrameAmbientDirections(sky: readonly number[], pairs: number): Vec3[] {
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
    const phi = (2 * Math.PI * k) / (n - 1);
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

/**
 * K6.Q1c (V128): the phase weight of the distant light. The pair totals are fitted with a dipole, T(ω) ≈ a + b·ω (least
 * squares over the 2N sides ±ω_k: a their mean, b = M⁺ Σ_k (T(ω_k) − T(−ω_k)) ω_k / 2 with M = Σ_k ω_k ω_kᵀ). Under an HG
 * lobe of mean cosine g, ∫ p T dω = a − g b·ω_cam (ω_cam toward the camera) = a + g b·r (r the view ray's direction),
 * clamped to [0, 1] per lobe and mixed by the lobe weight (a dual lobe's mean cosine in one clamp darkens a backlit
 * position too much: the back lobe's share is lost to the clamp at 0). The renderer packs M⁺ / 2 (`ambientDipole`); with
 * 1–3 pairs M is singular and the pseudo-inverse fits b in the pairs' span.
 */

/** Eigen-decomposition of a symmetric 3×3 (row-major) by Jacobi rotations: eigenvalues and eigenvector columns. */
function symmetricEigen3(m: readonly number[]): { values: Vec3; vectors: number[] } {
  const a = [...m];
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 32; sweep++) {
    const off = Math.abs(a[1]!) + Math.abs(a[2]!) + Math.abs(a[5]!);
    if (off < 1e-15) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = a[p * 3 + q]!;
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (a[q * 3 + q]! - a[p * 3 + p]!) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k * 3 + p]!;
        const akq = a[k * 3 + q]!;
        a[k * 3 + p] = c * akp - s * akq;
        a[k * 3 + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p * 3 + k]!;
        const aqk = a[q * 3 + k]!;
        a[p * 3 + k] = c * apk - s * aqk;
        a[q * 3 + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k * 3 + p]!;
        const vkq = v[k * 3 + q]!;
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: [a[0]!, a[4]!, a[8]!], vectors: v };
}

/**
 * K6.Q1c: M⁺ / 2 of the pairs' upper directions (row-major 3×3, symmetric; M = Σ_k ω_k ω_kᵀ, eigenvalues below 1e-6 of
 * the largest dropped). `b · r = Σ_k (T(ω_k) − T(−ω_k)) ω_k · (M⁺/2 · r)`.
 */
export function vdbFrameAmbientDipoleMatrix(directions: readonly (readonly number[])[]): number[] {
  const m = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (const w of directions) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) m[i * 3 + j] = m[i * 3 + j]! + w[i]! * w[j]!;
  const { values, vectors } = symmetricEigen3(m);
  const top = Math.max(...values.map(Math.abs));
  const out = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let e = 0; e < 3; e++) {
    if (!(top > 0) || Math.abs(values[e]!) <= 1e-6 * top) continue;
    const inv = 0.5 / values[e]!;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[i * 3 + j] = out[i * 3 + j]! + vectors[i * 3 + e]! * vectors[j * 3 + e]! * inv;
  }
  return out;
}

/** K6.Q1c: a phase's two HG lobes, each one's mean cosine per channel, and the second lobe's weight. */
export interface VDBFramePhaseLobes {
  g: [Vec3, Vec3];
  weight: number;
}

/**
 * K6.Q1c: the phase's lobes (the WGSL `vdbFrameAmbientPhaseFactor` reads the same): forward g and back g (spectral: each
 * scaled per channel and clamped ±0.99, as `vdbFramePhaseColor`) with the back weight; both lobes at the Lorenz-Mie table's
 * asymmetry for 'lorenz-mie' and 'lorenz-mie-rgb'.
 */
export function vdbFramePhaseLobes(phase: Vec4, phaseSpectrum: Vec3): VDBFramePhaseLobes {
  const mieG: Vec3 = [vdbFrameMieAsymmetry, vdbFrameMieAsymmetry, vdbFrameMieAsymmetry];
  if (phase[3] > vdbFrameMieRgbMode - 0.5) return { g: [mieG, mieG], weight: phase[2] };
  const scale: Vec3 = phaseSpectrum[2] > 0.5 ? [phaseSpectrum[0], 1, phaseSpectrum[1]] : [1, 1, 1];
  const clampG = (g: number) => Math.min(0.99, Math.max(-0.99, g));
  const mie = Math.min(phase[3], 1);
  const lobe = (g: number) => scale.map((s) => clampG(s * g) + (vdbFrameMieAsymmetry - clampG(s * g)) * mie) as Vec3;
  return { g: [lobe(phase[0]), lobe(phase[1])], weight: phase[2] };
}

/**
 * K6.Q1c: the scale of the distant light at one position (both visibilities × it): the dipole's phase-weighted visibility
 * mix(clamp(a + g₁ b·r, 0, 1), clamp(a + g₂ b·r, 0, 1), weight) over the isotropic one a (1 where a ≤ 1e-6). `totals` are
 * the pairs' (up, down) totals in `directions` order (grey: numbers; spectral: per channel), `matrix`
 * {@link vdbFrameAmbientDipoleMatrix} of the directions the renderer packed, `ray` the unit view ray (camera → position),
 * `lobes` {@link vdbFramePhaseLobes}.
 */
export function vdbFrameAmbientPhaseFactor(
  totals: readonly (readonly [number | Vec3, number | Vec3])[],
  directions: readonly (readonly number[])[],
  matrix: readonly number[],
  ray: readonly number[],
  lobes: VDBFramePhaseLobes,
): Vec3 {
  if (totals.length === 0) return [1, 1, 1];
  const u = [0, 1, 2].map((i) => matrix[i * 3]! * ray[0]! + matrix[i * 3 + 1]! * ray[1]! + matrix[i * 3 + 2]! * ray[2]!);
  const ch = (v: number | Vec3, c: number) => (typeof v === "number" ? v : v[c]!);
  return [0, 1, 2].map((c) => {
    let sum = 0;
    let dipole = 0;
    totals.forEach((t, k) => {
      const w = directions[k]!;
      sum += ch(t[0], c) + ch(t[1], c);
      dipole += (ch(t[0], c) - ch(t[1], c)) * (w[0]! * u[0]! + w[1]! * u[1]! + w[2]! * u[2]!);
    });
    const a = sum / (2 * totals.length);
    const lobe = (g: Vec3) => Math.min(Math.max(a + g[c]! * dipole, 0), 1);
    const v = lobe(lobes.g[0]) + (lobe(lobes.g[1]) - lobe(lobes.g[0])) * lobes.weight;
    return a > 1e-6 ? v / a : 1;
  }) as Vec3;
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
}

/** three's getShIrradianceAt. */
function shIrradiance(n: Vec3, sh: readonly Vec3[]): Vec3 {
  const [x, y, z] = n;
  const k = [0.886227, 2 * 0.511664 * y, 2 * 0.511664 * z, 2 * 0.511664 * x, 2 * 0.429043 * x * y, 2 * 0.429043 * y * z, z * z * 0.743125 - 0.247708, 2 * 0.429043 * x * z, 0.429043 * (x * x - y * y)];
  const out: Vec3 = [0, 0, 0];
  for (let i = 0; i < 9; i++) for (let c = 0; c < 3; c++) out[c] = out[c]! + (sh[i]?.[c] ?? 0) * k[i]!;
  return out;
}

/**
 * The shadowed distant-light irradiance at visibilities (V_up, V_down) (both lightings): ambient × (V_up + V_down) / 2
 * + hemisphere sky × V_up / 2 + ground × V_down / 2 + the probe's irradiance along ± the sky direction × V / 2.
 * Unoccluded (1, 1) gives the ambient and hemisphere light of the unshadowed renderer. K6.Q1a: a visibility may be per
 * channel (spectral).
 */
export function vdbFrameAmbientIrradiance(light: VDBFrameDistantLight, visibility: readonly [number | Vec3, number | Vec3]): Vec3 {
  const ch = (v: number | Vec3, c: number) => (typeof v === "number" ? v : v[c]!) * 0.5;
  const out: Vec3 = [0, 1, 2].map((c) => light.ambient[c]! * (ch(visibility[0], c) + ch(visibility[1], c)) + light.hemiSky[c]! * ch(visibility[0], c) + light.hemiGround[c]! * ch(visibility[1], c)) as Vec3;
  if (light.probe) {
    const a = shIrradiance(light.sky, light.probe);
    const b = shIrradiance([-light.sky[0], -light.sky[1], -light.sky[2]], light.probe);
    for (let c = 0; c < 3; c++) out[c] = out[c]! + a[c]! * ch(visibility[0], c) + b[c]! * ch(visibility[1], c);
  }
  return out;
}
