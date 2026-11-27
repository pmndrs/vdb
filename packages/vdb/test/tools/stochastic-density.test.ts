import { describe, expect, it } from "vitest";
import {
  indexBBox,
  vdbBuildFromDense,
  vdbRefInvert4,
  vdbReferenceRenderCpu,
  vdbReferenceSparseMediumCpu,
  vdbReferenceStatistics,
  vdbSparseDensityCpu,
  vdbSparseDensityStochasticCpu,
  type VDBReferenceMediumCpu,
  type VDBReferenceScene,
  type VDBSparseVolumeCpu,
} from "../../src/tools/internal.js";

// Issue #2 (stochastic texture filtering, Pharr et al. 2024): one voxel per density lookup, picked with probability
// equal to its trilinear weight. The CPU twin of `vdbSrcDensityStochastic`, its exact expectation, and the reference
// tracer's ratio tracking with it (unbiased: tracking is linear in sigma). The GPU kernel = this twin in
// e2e/k6-reference.spec.ts.

const f = Math.fround;
const N: [number, number, number] = [24, 20, 12];

/** A smooth field with gradients on every axis (values in [0, 1]), x-major as vdbBuildFromDense reads it. */
function field(): Uint32Array {
  const values = new Float32Array(N[0] * N[1] * N[2]);
  for (let x = 0; x < N[0]; x++)
    for (let y = 0; y < N[1]; y++)
      for (let z = 0; z < N[2]; z++) {
        // The z factor stays positive, so a ray's integral keeps the x, y pattern (a sin(z) factor integrates to ~0).
        const v = (0.5 + 0.45 * Math.sin(x * 0.9 + 0.3) * Math.cos(y * 0.7)) * (0.3 + 0.7 * Math.sin(z * 0.6 + 0.5) ** 2);
        values[(x * N[1] + y) * N[2] + z] = v > 0.05 ? v : 0;
      }
  return vdbBuildFromDense(values, N, { voxelSize: 0.25 });
}

const image = field();
const vol: VDBSparseVolumeCpu = { lod: "none", density: { image, box: indexBBox(image) } };

/** `vdbSrcSamplePos` (f32): uvw → index position, clamped to the box. */
function samplePos(uvw: readonly number[]): number[] {
  const { min, max } = vol.density.box;
  return [0, 1, 2].map((a) => {
    const size = max[a]! - min[a]! + 1;
    const p = f(f(min[a]! + f(f(uvw[a]!) * size)) - 0.5);
    return Math.min(Math.max(p, min[a]!), min[a]! + size - 1);
  });
}

/** E[stochastic] by enumerating the 8 voxels: xi picks each corner, weighted by its probability. */
function expectation(uvw: readonly number[], swap = false): number {
  const p = samplePos(uvw);
  const frac = p.map((v) => v - Math.floor(v));
  let sum = 0;
  for (let k = 0; k < 8; k++) {
    const bits = [k & 1, (k >> 1) & 1, (k >> 2) & 1];
    let w = 1;
    const xi = bits.map((bit, a) => {
      const pu = swap ? 1 - frac[a]! : frac[a]!; // the mutation: upper with probability 1 − frac
      w *= bit ? pu : 1 - pu;
      return bit ? frac[a]! / 2 : (1 + frac[a]!) / 2; // xi < frac picks the upper voxel
    });
    if (w > 0) sum += w * vdbSparseDensityStochasticCpu(vol, uvw, 0, xi).value;
  }
  return sum;
}

const lcg = (seed: number) => () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);

describe("issue #2: stochastic trilinear density (CPU twin of vdbSrcDensityStochastic)", () => {
  it("reads exactly one voxel, and its expectation over xi is the trilinear value", () => {
    const rand = lcg(7);
    let worst = 0;
    for (let i = 0; i < 400; i++) {
      const uvw = [rand() * 1.1 - 0.05, rand() * 1.1 - 0.05, rand() * 1.1 - 0.05];
      const tri = vdbSparseDensityCpu(vol, uvw, 0).value;
      worst = Math.max(worst, Math.abs(expectation(uvw) - tri));
    }
    expect(worst).toBeLessThan(1e-6);
  });

  it("fail-before: the swapped pick (upper voxel with probability 1 − frac) does not have the trilinear mean", () => {
    const rand = lcg(7);
    let worst = 0;
    for (let i = 0; i < 400; i++) {
      const uvw = [rand(), rand(), rand()];
      worst = Math.max(worst, Math.abs(expectation(uvw, true) - vdbSparseDensityCpu(vol, uvw, 0).value));
    }
    expect(worst).toBeGreaterThan(1e-2);
  });

  it("lod > 0 falls back to the deterministic sample", () => {
    expect(vdbSparseDensityStochasticCpu(vol, [0.3, 0.4, 0.5], 1, [0.5, 0.5, 0.5]).value).toBe(vdbSparseDensityCpu(vol, [0.3, 0.4, 0.5], 1).value);
  });
});

describe("issue #2: reference ratio tracking with the stochastic density", () => {
  const W = 8;
  const H = 8;
  const HALF = 1.2;
  const SLAB = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const SIGMA = 2.5;
  const n = 0.1;
  const far = 100;
  const proj = [1 / HALF, 0, 0, 0, 0, 1 / HALF, 0, 0, 0, 0, -2 / (far - n), 0, 0, 0, -(far + n) / (far - n), 1];
  const scene: VDBReferenceScene = {
    camera: { cameraToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1], projectionInverse: vdbRefInvert4(proj), orthographic: true },
    volumes: [{ localToWorld: SLAB, sigma: SIGMA, majorant: 1, albedo: [0, 0, 0] }],
  };

  /** exp(−σ ∫ trilinear density) along the pixel's ray (z from +½ to −½ of the unit box), midpoint quadrature. */
  function expectedT(px: number, py: number): number {
    const u = ((((px + 0.5) / W) * 2 - 1) * HALF) / 2 + 0.5;
    const v = ((((1 - (py + 0.5) / H) * 2 - 1) * HALF) / 2) + 0.5;
    if (u < 0 || u > 1 || v < 0 || v > 1) return 1;
    const steps = 4096;
    let tau = 0;
    for (let k = 0; k < steps; k++) tau += vdbSparseDensityCpu(vol, [u, v, 1 - (k + 0.5) / steps], 0).value;
    return Math.exp((-SIGMA * tau) / steps);
  }

  function run(medium: VDBReferenceMediumCpu, densityFilter: "trilinear" | "stochastic") {
    const accum = vdbReferenceRenderCpu(scene, [medium], { width: W, height: H, spp: 8, maxBounces: 1, seed: 3 }, { batches: 24, densityFilter });
    const { mean, stderr } = vdbReferenceStatistics(accum);
    let over = 0;
    let sumE = 0;
    let sumS2 = 0;
    for (let py = 0; py < H; py++)
      for (let px = 0; px < W; px++) {
        const i = (py * W + px) * 4 + 3;
        const e = mean[i]! - expectedT(px, py);
        if (Math.abs(e) > 4.5 * stderr[i]! + 1e-6) over++;
        sumE += e;
        sumS2 += stderr[i]! ** 2;
      }
    return { over, meanZ: sumE / Math.sqrt(Math.max(sumS2, 1e-30)), violations: accum.stats.majorantViolations };
  }
  const passes = (c: { over: number; meanZ: number }) => c.over <= 1 && Math.abs(c.meanZ) < 4;

  it("T = exp(−σ ∫ trilinear) with one voxel per lookup (and with eight), no majorant violation", () => {
    const medium = vdbReferenceSparseMediumCpu(vol);
    const s = run(medium, "stochastic");
    expect(passes(s), JSON.stringify(s)).toBe(true);
    expect(s.violations).toBe(0);
    const t = run(medium, "trilinear");
    expect(passes(t), JSON.stringify(t)).toBe(true);
  });

  it("fail-before: always the lower voxel (a half-voxel shift) is biased", () => {
    const medium = vdbReferenceSparseMediumCpu(vol);
    const lower: VDBReferenceMediumCpu = { ...medium, densityStochastic: (uvw) => medium.densityStochastic!(uvw, [0.9999999, 0.9999999, 0.9999999]) };
    const r = run(lower, "stochastic");
    if (process.env.REF_DEBUG) console.log("MUT", JSON.stringify(r));
    expect(passes(r)).toBe(false);
  });
});
