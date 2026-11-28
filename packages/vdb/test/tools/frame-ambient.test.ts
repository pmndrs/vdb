import { describe, expect, it } from "vitest";
import {
  VDB_FRAME_AMBIENT_MAX_PAIRS,
  vdbFrameAmbientDipoleMatrix,
  vdbFrameAmbientDirections,
  vdbFrameAmbientPhaseFactor,
  vdbFramePhaseLobes,
  vdbFrameAmbientIrradiance,
  vdbFrameAmbientPairTotals,
  vdbFrameAmbientVisibility,
  vdbFrameShadowMarch,
} from "../../src/tools/internal.js";

// K6.Q1 shadowed ambient (V85): the direction set, the per-pair totals and the distant-light irradiance the core
// kernels twin. Each test pins a property a port error (or a mutation) would break.

type V3 = [number, number, number];
const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

describe("K6.Q1 vdbFrameAmbientDirections", () => {
  it("gives unit directions, the sky first, the ring at the equal-area centroid, evenly in azimuth", () => {
    const sky: V3 = [0.2, 0.9, -0.3];
    const s = sky.map((v) => v / Math.hypot(...sky));
    for (let n = 1; n <= VDB_FRAME_AMBIENT_MAX_PAIRS; n++) {
      const d = vdbFrameAmbientDirections(sky, n);
      expect(d.length).toBe(n);
      for (const v of d) expect(Math.hypot(...v)).toBeCloseTo(1, 12);
      expect(dot(d[0]!, s)).toBeCloseTo(1, 12);
      for (const v of d.slice(1)) expect(dot(v, s)).toBeCloseTo((1 - 1 / n) / 2, 12);
      // the ring's projections onto the tangent plane sum to zero (even azimuths)
      if (n > 2) {
        const sum = d.slice(1).reduce((a, v) => a.map((x, i) => x + v[i]! - s[i]! * dot(v, s)), [0, 0, 0]);
        expect(Math.hypot(...sum)).toBeLessThan(1e-12);
      }
    }
    expect(vdbFrameAmbientDirections([0, 1, 0], 99).length).toBe(VDB_FRAME_AMBIENT_MAX_PAIRS);
    expect(vdbFrameAmbientDirections([0, 1, 0], 0).length).toBe(1);
    // a sky along +y still gets a tangent frame
    expect(vdbFrameAmbientDirections([0, 1, 0], 4).every((v) => v.every(Number.isFinite))).toBe(true);
  });

  it("is a quadrature of the spherical mean of transmittance: a homogeneous ball, off-centre (error shrinks with pairs)", () => {
    // T(ω) = exp(−σ · distance to the unit sphere's surface from p along ω); exact mean by dense quadrature
    const p: V3 = [0.35, -0.2, 0.15];
    const sigma = 2.5;
    const exit = (w: readonly number[]) => {
      const b = dot(p, w);
      return -b + Math.sqrt(b * b - (dot(p, p) - 1));
    };
    let exact = 0;
    const M = 400;
    for (let i = 0; i < M; i++)
      for (let j = 0; j < 2 * M; j++) {
        const u = -1 + (2 * (i + 0.5)) / M;
        const phi = (Math.PI * (j + 0.5)) / M;
        const r = Math.sqrt(1 - u * u);
        exact += Math.exp(-sigma * exit([r * Math.cos(phi), u, r * Math.sin(phi)]));
      }
    exact /= 2 * M * M;
    const estimate = (n: number) => {
      const totals = vdbFrameAmbientDirections([0, 1, 0], n).map((w) => [Math.exp(-sigma * exit(w)), Math.exp(-sigma * exit(w.map((x) => -x)))] as [number, number]);
      const v = vdbFrameAmbientVisibility(totals);
      return (v[0] + v[1]) / 2;
    };
    const err = [1, 4, 7].map((n) => Math.abs(estimate(n) / exact - 1));
    // 1 pair (his sky up / down) is off by tens of percent here; 4 and 7 pairs within a few
    expect(err[1]!).toBeLessThan(0.05);
    expect(err[2]!).toBeLessThan(0.04);
    expect(err[1]!).toBeLessThan(err[0]!);
  });
});

describe("K6.Q1 pair totals and irradiance", () => {
  it("diffusion pairs pass the white furnace at albedo 1 (V up + V down = 2) and fall below with absorption", () => {
    const steps = Array.from({ length: 24 }, (_, i) => 0.05 + 0.04 * Math.sin(i));
    const back = Array.from({ length: 24 }, (_, i) => 0.08 + 0.02 * Math.cos(i * 0.7));
    const white: V3 = [1, 1, 1];
    const f = vdbFrameShadowMarch(steps, white, 0.2);
    const b = vdbFrameShadowMarch(back, white, 0.2);
    const [up, down] = vdbFrameAmbientPairTotals(f, b, true);
    expect(up + down).toBeCloseTo(2, 12);
    const grey: V3 = [0.5, 0.5, 0.5];
    const [u2, d2] = vdbFrameAmbientPairTotals(vdbFrameShadowMarch(steps, grey, 0.2), vdbFrameShadowMarch(back, grey, 0.2), true);
    expect(u2 + d2).toBeLessThan(1.9);
    // no multiple scattering: each side's own total (the direct transmittance)
    const [u3, d3] = vdbFrameAmbientPairTotals(vdbFrameShadowMarch(steps, white, 0), vdbFrameShadowMarch(back, white, 0), false);
    expect(u3).toBeCloseTo(Math.exp(-steps.reduce((a, x) => a + x, 0)), 12);
    expect(d3).toBeCloseTo(Math.exp(-back.reduce((a, x) => a + x, 0)), 12);
    // octaves: the octave total of each side's 1 / (1 + depth), over the open direction's (1 + 0.5 + 0.25)
    const oct: [number, number, number, number] = [2, 0.5, 0.5, 0.5];
    const depth = steps.reduce((a, x) => a + x, 0);
    const [u4] = vdbFrameAmbientPairTotals(vdbFrameShadowMarch(steps, white, 0, oct), vdbFrameShadowMarch(back, white, 0, oct), false, oct);
    expect(u4).toBeCloseTo((Math.exp(-depth) + 0.5 * Math.exp(-0.5 * depth) + 0.25 * Math.exp(-0.25 * depth)) / 1.75, 6);
  });

  it("unoccluded, the irradiance is the unshadowed renderer's ambient + hemisphere light; occluded it scales per side", () => {
    const light = { ambient: [0.3, 0.2, 0.1] as V3, hemiSky: [0.5, 0.6, 0.7] as V3, hemiGround: [0.1, 0.05, 0.02] as V3, sky: [0, 1, 0] as V3 };
    const open = vdbFrameAmbientIrradiance(light, [1, 1]);
    for (let c = 0; c < 3; c++) expect(open[c]).toBeCloseTo(light.ambient[c]! + (light.hemiSky[c]! + light.hemiGround[c]!) / 2, 12);
    const half = vdbFrameAmbientIrradiance(light, [0.5, 0.2]);
    for (let c = 0; c < 3; c++) expect(half[c]).toBeCloseTo(light.ambient[c]! * 0.35 + light.hemiSky[c]! * 0.25 + light.hemiGround[c]! * 0.1, 12);
    // a constant probe (c0 only): irradiance 0.886227 c0 at any visibility mean
    const probe = Array.from({ length: 9 }, (_, i) => (i === 0 ? ([1, 2, 3] as V3) : ([0, 0, 0] as V3)));
    const p = vdbFrameAmbientIrradiance({ ambient: [0, 0, 0], hemiSky: [0, 0, 0], hemiGround: [0, 0, 0], sky: [0, 1, 0], probe }, [0.6, 0.4]);
    for (let c = 0; c < 3; c++) expect(p[c]).toBeCloseTo(0.886227 * (c + 1) * 0.5, 9);
  });

  it("the probe's sky up / down terms are three's getShIrradianceAt along ± the sky (K6.Q1b: the 'radiance' A/B is deleted)", () => {
    // band 1 along +y: three's irradiance 2 × 0.511664 c₁ on the upper side (×V_up/2), its negative on the lower one
    const l1 = Array.from({ length: 9 }, (_, i) => (i === 1 ? ([1, 1, 1] as V3) : ([0, 0, 0] as V3)));
    const p = vdbFrameAmbientIrradiance({ ambient: [0, 0, 0], hemiSky: [0, 0, 0], hemiGround: [0, 0, 0], sky: [0, 1, 0], probe: l1 }, [1, 0]);
    for (let c = 0; c < 3; c++) expect(p[c]).toBeCloseTo(0.511664, 9);
  });

});

describe("K6.Q1c the distant light's phase weight (V128)", () => {
  const hg = (c: number, g: number) => (1 - g * g) / Math.pow(1 + g * g - 2 * g * c, 1.5);
  // ∫ p(r · ω) T(ω) dω / 4π by dense quadrature (ω: the direction toward the light; r: the view ray; HG normalised to 1)
  const phaseMean = (T: (w: V3) => number, r: V3, g: number, M = 400) => {
    let sum = 0;
    for (let i = 0; i < M; i++)
      for (let j = 0; j < 2 * M; j++) {
        const u = -1 + (2 * (i + 0.5)) / M;
        const phi = (Math.PI * (j + 0.5)) / M;
        const s = Math.sqrt(1 - u * u);
        const w: V3 = [s * Math.cos(phi), u, s * Math.sin(phi)];
        sum += hg(dot(r, w), g) * T(w);
      }
    return sum / (2 * M * M);
  };
  // one HG lobe of mean cosine g (per channel)
  const hgLobe = (g: number | V3) => {
    const v: V3 = typeof g === "number" ? [g, g, g] : g;
    return { g: [v, v] as [V3, V3], weight: 0 };
  };
  const unit = (v: readonly number[]): V3 => {
    const l = Math.hypot(v[0]!, v[1]!, v[2]!);
    return [v[0]! / l, v[1]! / l, v[2]! / l];
  };

  it("a linear T(ω) = a + b·ω: the dipole recovers b exactly (4–8 pairs) and a + g b·r is the phase-weighted mean", () => {
    const a = 0.45;
    const b: V3 = [0.2, -0.15, 0.1];
    const r = unit([0.3, -0.5, 0.8]);
    const g = 0.6;
    // the exact phase-weighted mean of a linear T (sign convention: light from along the ray is forward-scattered)
    expect(phaseMean((w) => a + dot(b, w), r, g)).toBeCloseTo(a + g * dot(b, r), 4);
    for (let n = 4; n <= VDB_FRAME_AMBIENT_MAX_PAIRS; n++) {
      const dirs = vdbFrameAmbientDirections([0.1, 1, -0.2], n);
      const totals = dirs.map((w) => [a + dot(b, w), a - dot(b, w)] as [number, number]);
      const m = vdbFrameAmbientDipoleMatrix(dirs);
      const f = vdbFrameAmbientPhaseFactor(totals, dirs, m, r, hgLobe(g));
      for (let c = 0; c < 3; c++) expect(f[c]! * a).toBeCloseTo(a + g * dot(b, r), 12);
      // g = 0 (isotropic phase): the factor is 1
      expect(vdbFrameAmbientPhaseFactor(totals, dirs, m, r, hgLobe(0))).toEqual([1, 1, 1]);
    }
  });

  it("1–3 pairs: M is singular; the pseudo-inverse fits b in the pairs' span (the sky axis for 1 pair)", () => {
    const sky: V3 = [0, 1, 0];
    const b: V3 = [0.2, 0.3, -0.1];
    const a = 0.5;
    for (const n of [1, 2, 3]) {
      const dirs = vdbFrameAmbientDirections(sky, n);
      const m = vdbFrameAmbientDipoleMatrix(dirs);
      expect(m.every(Number.isFinite)).toBe(true);
      const totals = dirs.map((w) => [a + dot(b, w), a - dot(b, w)] as [number, number]);
      // along the sky every set resolves b's sky component
      const f = vdbFrameAmbientPhaseFactor(totals, dirs, m, sky, hgLobe(1));
      expect(f[0]! * a).toBeCloseTo(a + b[1], 12);
    }
    // 1 pair: a ray in the tangent plane sees no dipole
    const one = vdbFrameAmbientDirections(sky, 1);
    const f = vdbFrameAmbientPhaseFactor([[0.9, 0.1]], one, vdbFrameAmbientDipoleMatrix(one), [1, 0, 0], hgLobe(0.8));
    expect(f[1]).toBeCloseTo(1, 12);
  });

  it("clamps the weighted visibility to [0, 1], gives 1 where a ≤ 1e-6, and works per channel", () => {
    const dirs = vdbFrameAmbientDirections([0, 1, 0], 4);
    const m = vdbFrameAmbientDipoleMatrix(dirs);
    // open above, black below; a ray looking down: forward scattering (g > 0) sees the light from below the position
    // (black), back scattering (g < 0) the open sky
    const totals = dirs.map(() => [1, 0] as [number, number]);
    const f = vdbFrameAmbientPhaseFactor(totals, dirs, m, [0, -1, 0], hgLobe([0.99, 0, -0.99]));
    expect(f[0]!).toBeLessThan(1);
    expect(f[0]!).toBeGreaterThanOrEqual(0);
    expect(f[1]).toBeCloseTo(1, 12);
    expect(f[2]!).toBeGreaterThan(1);
    expect(f[2]! * 0.5).toBeLessThanOrEqual(1);
    expect(vdbFrameAmbientPhaseFactor(dirs.map(() => [0, 0] as [number, number]), dirs, m, [0, -1, 0], hgLobe(0.8))).toEqual([1, 1, 1]);
    // spectral totals: each channel its own fit
    const spectral = dirs.map(() => [[1, 0.5, 0.2], [0, 0.5, 0.1]] as [V3, V3]);
    const fs = vdbFrameAmbientPhaseFactor(spectral, dirs, m, [0, -1, 0], hgLobe(0.5));
    expect(fs[1]).toBeCloseTo(1, 12);
    expect(fs[0]!).toBeLessThan(fs[2]!);
  });

  it("is closer to the phase-weighted mean than the isotropic visibility: a homogeneous ball, off-centre, g 0.8", () => {
    const p: V3 = [0.35, -0.2, 0.15];
    const sigma = 2.5;
    const exit = (w: readonly number[]) => {
      const b = dot(p, w);
      return -b + Math.sqrt(b * b - (dot(p, p) - 1));
    };
    const T = (w: V3) => Math.exp(-sigma * exit(w));
    const dirs = vdbFrameAmbientDirections([0, 1, 0], 4);
    const totals = dirs.map((w) => [T(w), T(w.map((x) => -x) as V3)] as [number, number]);
    const v = vdbFrameAmbientVisibility(totals);
    const iso = (v[0] + v[1]) / 2;
    const m = vdbFrameAmbientDipoleMatrix(dirs);
    for (const r of [unit([0.7, 0.2, 0.3]), unit([-0.5, 0.1, -0.2]), unit([0, -1, 0.1])]) {
      const exact = phaseMean(T, r, 0.8, 300);
      const dip = iso * vdbFrameAmbientPhaseFactor(totals, dirs, m, r, hgLobe(0.8))[0]!;
      expect(Math.abs(dip / exact - 1)).toBeLessThan(Math.abs(iso / exact - 1));
    }
  });

  it("the phase lobes: forward and back g with the back weight, per channel with a spectrum, the Mie table's asymmetry", () => {
    const d = vdbFramePhaseLobes([0.8, -0.3, 0.2, 0], [1, 1, 0]);
    expect(d.g[0]).toEqual([0.8, 0.8, 0.8]);
    expect(d.g[1]).toEqual([-0.3, -0.3, -0.3]);
    expect(d.weight).toBe(0.2);
    expect(vdbFramePhaseLobes([0, 0, 0, 1], [1, 1, 0]).g).toEqual([[0.8526, 0.8526, 0.8526], [0.8526, 0.8526, 0.8526]]);
    expect(vdbFramePhaseLobes([0, 0, 0, 2], [1, 1, 1]).g).toEqual([[0.8526, 0.8526, 0.8526], [0.8526, 0.8526, 0.8526]]);
    // spectral: g × (red, 1, blue) scale, clamped ±0.99 per lobe
    const s = vdbFramePhaseLobes([0.9, -0.3, 0.25, 0], [0.8, 1.2, 1]);
    expect(s.g[0][0]).toBeCloseTo(0.72, 12);
    expect(s.g[0][2]).toBeCloseTo(0.99, 12);
    expect(s.g[1][2]).toBeCloseTo(-0.36, 12);
  });

  it("a dual lobe clamps per lobe: a backlit step (open hemisphere behind the camera) keeps the back lobe's share", () => {
    // T = 1 on the hemisphere around +y, 0 below; the camera looks up (r = +y is forward scattering of the light from
    // above) or down (r = −y: light from above only by back scattering). Reference: the dual lobe 0.8 / −0.3 / 0.2.
    const T = (w: V3) => (w[1] > 0 ? 1 : 0);
    const dual = (c: number) => 0.8 * hg(c, 0.8) + 0.2 * hg(c, -0.3);
    const exactDual = (r: V3) => {
      let sum = 0;
      const M = 300;
      for (let i = 0; i < M; i++)
        for (let j = 0; j < 2 * M; j++) {
          const u = -1 + (2 * (i + 0.5)) / M;
          const phi = (Math.PI * (j + 0.5)) / M;
          const q = Math.sqrt(1 - u * u);
          const w: V3 = [q * Math.cos(phi), u, q * Math.sin(phi)];
          sum += dual(dot(r, w)) * T(w);
        }
      return sum / (2 * M * M);
    };
    const dirs = vdbFrameAmbientDirections([0, 1, 0], 4);
    const m = vdbFrameAmbientDipoleMatrix(dirs);
    const totals = dirs.map((w) => [T(w), T(w.map((x) => -x) as V3)] as [number, number]);
    const lobes = vdbFramePhaseLobes([0.8, -0.3, 0.2, 0], [1, 1, 0]);
    const down: V3 = [0, -1, 0];
    const exact = exactDual(down);
    const perLobe = 0.5 * vdbFrameAmbientPhaseFactor(totals, dirs, m, down, lobes)[0]!;
    const meanCosine = 0.5 * vdbFrameAmbientPhaseFactor(totals, dirs, m, down, hgLobe(0.8 * 0.8 - 0.3 * 0.2))[0]!;
    expect(Math.abs(perLobe - exact)).toBeLessThan(Math.abs(meanCosine - exact));
    expect(Math.abs(perLobe / exact - 1)).toBeLessThan(0.35);
    // forward: both near the exact (most light comes from the open side)
    expect(Math.abs((0.5 * vdbFrameAmbientPhaseFactor(totals, dirs, m, [0, 1, 0], lobes)[0]!) / exactDual([0, 1, 0]) - 1)).toBeLessThan(0.1);
  });
});
