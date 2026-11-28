/**
 * K6.4: the core's spectral twins (tools/frame/spectral-stack.ts) reduce to the grey twins when every channel is the
 * same (a white medium with no Ångström exponent): the slot-group packing, the transmittance, the environment grid, the
 * 'volume' combine, the resolve scan and the composite. The GPU kernels are checked against these twins in
 * e2e/k6-core.spec.ts (K6.4 probes).
 */
import { describe, expect, it } from "vitest";
import {
  vdbFrameCompositeSpectral,
  vdbFrameEnvironmentGridVoxel3d,
  vdbFrameGatherFroxel,
  vdbFrameGatherFroxelSpectral,
  vdbFrameIntegratePrefix,
  vdbFrameIntegratePrefixSpectral,
  vdbFrameScanColumn,
  vdbFrameScanColumnSpectral,
  vdbFrameSpectralEnvironmentGridVoxel3d,
  vdbFrameSpectralGridDensityVoxel,
  vdbFrameSpectralTransmittanceVoxel3d,
  vdbFrameTransmittanceVoxel3d,
  vdbFrameUnpackSpectralGroup,
  vdbFrameGridDensityVoxel,
  vdbFrameAmbientDirections,
  vdbFrameAmbientIrradiance,
  vdbFrameAmbientPairTotals,
  vdbFrameSpectralAmbientPairTotals,
  vdbFrameShadowMarch,
  vdbFrameSpectralShadowMarch,
  vdbFrameOctavesOff,
  type VDBFrameGridLayout,
} from "../../src/tools/internal.js";

type V3 = [number, number, number];
type V4 = [number, number, number, number];

let seed = 7;
const rand = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296;

// one cascade, 8³ voxels over a 2-unit box; a smooth blob of (extinction, absorption)
const grid: VDBFrameGridLayout = { count: 1, resolution: 8, columns: 8, atlasSize: [64, 8], gridMin: [[-1, -1, -1]], gridSize: [[2, 2, 2]], gridVoxel: [[0.25, 0.25, 0.25]] };
const blob = (uvw: readonly number[]) => {
  const d = Math.hypot(uvw[0]! - 0.5, uvw[1]! - 0.45, uvw[2]! - 0.55);
  const e = Math.max(0, 1 - d * 2.4) * 3;
  return [e, e * 0.3] as [number, number];
};
const lights: V4[] = [
  [-0.4, 0.8, 0.45, 1],
  [0.8, 0.9, 0.3, 2],
  [-0.5, 1.2, -0.6, 3],
];

describe("K6.4 spectral stack twins (equal channels = grey)", () => {
  it("the spectral shadow grid of a white medium (spectrum 1, grey albedo) holds the grey extinction in each channel", () => {
    const casters = [
      {
        worldToLocal: [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1],
        maxTexels: 32,
        worldScale: 2,
        mipmaps: false,
        absorbance: 0.8,
        extinctionSpectrum: [1, 1, 1] as V3,
        extinctionAt: (local: V3) => blob([local[0] + 0.5, local[1] + 0.5, local[2] + 0.5])[0],
        albedoAt: (): V3 => [0.7, 0.7, 0.7],
      },
    ];
    for (let i = 0; i < 50; i++) {
      const world: V3 = [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1];
      const s = vdbFrameSpectralGridDensityVoxel(world, 0.25, casters);
      const g = vdbFrameGridDensityVoxel(world, 0.25, casters);
      for (let c = 0; c < 3; c++) {
        expect(s.extinction[c]).toBeCloseTo(g[0], 12);
        // grey albedo: the per-channel absorption equals the luminance absorption
        expect(s.absorption[c]).toBeCloseTo(g[1], 12);
      }
    }
  });

  for (const ms of [0, 0.6]) {
    for (const twoSided of [false, true]) {
      it(`the transmittance's slot groups unpack to the grey targets (ms ${ms}${twoSided ? ", two-sided" : ""})`, () => {
        const greyAt = (_c: number, uvw: V3) => blob(uvw);
        const spectralAt = (_c: number, uvw: V3) => {
          const [e, a] = blob(uvw);
          return { extinction: [e, e, e] as V3, absorption: [a, a, a] as V3 };
        };
        let compared = 0;
        for (let i = 0; i < 40; i++) {
          const cell: V3 = [Math.floor(rand() * 8), Math.floor(rand() * 8), Math.floor(rand() * 8)];
          const grey = vdbFrameTransmittanceVoxel3d(greyAt, grid, 0, cell, lights, [0, 1, 0], ms, -1, null, { twoSided });
          const groups = [0, 1].map((g) => {
            const t = vdbFrameSpectralTransmittanceVoxel3d(spectralAt, grid, 0, cell, lights, [0, 1, 0], ms, -1, g, { twoSided });
            return vdbFrameUnpackSpectralGroup(t[0], t[1], t[2]);
          });
          // grey targets: (sky up, sky down, slot 0, 1 total), (slots 2–5 total), (slots 0–3 direct), (4, 5 direct)
          const sky = groups[0]!.a;
          const slot = (i: number) => (i === 0 ? groups[0]!.b : i === 1 ? groups[1]!.a : groups[1]!.b);
          for (let c = 0; c < 3; c++) {
            expect(sky.x[c]).toBeCloseTo(grey[0][0], 12);
            expect(sky.y[c]).toBeCloseTo(grey[0][1], 12);
            expect(slot(0).x[c]).toBeCloseTo(grey[0][2], 12);
            expect(slot(1).x[c]).toBeCloseTo(grey[0][3], 12);
            expect(slot(2).x[c]).toBeCloseTo(grey[1][0], 12);
            expect(slot(0).y[c]).toBeCloseTo(grey[2][0], 12);
            expect(slot(1).y[c]).toBeCloseTo(grey[2][1], 12);
            expect(slot(2).y[c]).toBeCloseTo(grey[2][2], 12);
          }
          compared++;
        }
        expect(compared).toBe(40);
      });
    }
  }

  it("the environment grid of a white medium equals the grey one (diffusion and single scattering)", () => {
    const envLights = [
      { direction: [0.3, 0.9, 0.2] as V3, color: [0.8, 0.7, 0.6] as V3 },
      { direction: [-0.6, 0.2, 0.75] as V3, color: [0.2, 0.3, 0.5] as V3 },
    ];
    const norm = (v: V3): V3 => {
      const l = Math.hypot(...v);
      return [v[0] / l, v[1] / l, v[2] / l];
    };
    const ls = envLights.map((l) => ({ ...l, direction: norm(l.direction) }));
    const spectralAt = (_c: number, uvw: V3) => {
      const [e, a] = blob(uvw);
      return { extinction: [e, e, e] as V3, absorption: [a, a, a] as V3 };
    };
    for (const ms of [0, 0.5]) {
      for (let i = 0; i < 20; i++) {
        const cell: V3 = [Math.floor(rand() * 8), Math.floor(rand() * 8), Math.floor(rand() * 8)];
        const g = vdbFrameEnvironmentGridVoxel3d((_c, uvw) => blob(uvw), grid, cell, ls, ms);
        const s = vdbFrameSpectralEnvironmentGridVoxel3d(spectralAt, grid, cell, ls, ms);
        for (let c = 0; c < 4; c++) expect(s[c]).toBeCloseTo(g[c]!, 12);
      }
    }
  });

  it("the 'volume' combine, the resolve scan, the two taps and the composite reduce to grey with equal channels", () => {
    for (let trial = 0; trial < 30; trial++) {
      const volumes = Array.from({ length: 1 + Math.floor(rand() * 3) }, () => ({ radiance: [rand() * 2, rand() * 2, rand() * 2] as V3, alpha: rand() * 0.9 }));
      const grey = vdbFrameGatherFroxel(volumes);
      const s0 = vdbFrameGatherFroxelSpectral(volumes.map((v) => ({ radiance: v.radiance, alpha: [v.alpha, v.alpha, v.alpha] as V3 })));
      for (let c = 0; c < 3; c++) {
        expect(s0.color[c]).toBeCloseTo(grey[c]!, 12);
        expect(s0.alpha[c]).toBeCloseTo(grey[3], 12);
      }
      const froxels: V4[] = Array.from({ length: 10 }, () => {
        const a = rand() < 0.2 ? 0 : rand() * 0.8;
        return [rand() * a, rand() * a, rand() * a, a];
      });
      const g = vdbFrameScanColumn(froxels);
      const s = vdbFrameScanColumnSpectral(froxels.map((f) => ({ scatter: f, alpha: [f[3], f[3], 0, f[3]] as V4 })));
      for (let k = 0; k <= 10; k++) {
        for (let c = 0; c < 4; c++) expect(s.prefix[k]![c]).toBeCloseTo(g[k]![c]!, 12);
        expect(s.alpha[k]![0]).toBeCloseTo(g[k]![3], 12);
        expect(s.alpha[k]![1]).toBeCloseTo(g[k]![3], 12);
      }
      const k = Math.floor(rand() * 10);
      const c = rand();
      const gi = vdbFrameIntegratePrefix(g[k]!, g[k + 1]!, c);
      const si = vdbFrameIntegratePrefixSpectral(s.prefix[k]!, s.alpha[k]!, s.prefix[k + 1]!, s.alpha[k + 1]!, c);
      for (let ch = 0; ch < 3; ch++) {
        expect(si.radiance[ch]).toBeCloseTo(gi.radiance[ch]!, 12);
        expect(si.transmittance[ch]).toBeCloseTo(gi.transmittance, 12);
      }
      // the linear composite: radiance + scene × T (the grey premultiplied blend over the scene)
      const scene: V4 = [rand(), rand(), rand(), 1];
      const out = vdbFrameCompositeSpectral(si.radiance, si.transmittance, scene, false);
      for (let ch = 0; ch < 3; ch++) expect(out[ch]).toBeCloseTo(gi.radiance[ch]! + scene[ch]! * gi.transmittance, 12);
      expect(out[3]).toBeCloseTo(1, 12);
    }
  });

  it("a blue-heavy spectrum darkens blue more than red (Beer, per channel)", () => {
    const at = (_c: number, uvw: V3) => {
      const [e, a] = blob(uvw);
      return { extinction: [e * 0.6, e, e * 1.6] as V3, absorption: [a * 0.6, a, a * 1.6] as V3 };
    };
    const t = vdbFrameSpectralTransmittanceVoxel3d(at, grid, 0, [4, 1, 4], lights, [0, 1, 0], 0, -1, 0);
    const slot0 = vdbFrameUnpackSpectralGroup(t[0], t[1], t[2]).b;
    // the direct transmittance toward the sun: red > green > blue (unless nothing is in the way)
    expect(slot0.y[0]).toBeGreaterThanOrEqual(slot0.y[1]);
    expect(slot0.y[1]).toBeGreaterThanOrEqual(slot0.y[2]);
  });
});

// K6.Q1a: the spectral shadowed ambient (group 0's sky entry holds the hemispherical visibilities of each channel)
describe("K6.Q1a spectral ambient visibilities (a grey medium = grey)", () => {
  const octaves: V4 = [3, 0.6, 0.45, 0.7];
  const cases: { name: string; ms: number; oct?: V4; absorption: number }[] = [
    { name: "single scattering, absorbing", ms: 0, absorption: 0.3 },
    { name: "diffusion, absorbing", ms: 0.6, absorption: 0.3 },
    { name: "octaves, white (albedo 1)", ms: 0, oct: octaves, absorption: 0 },
  ];
  for (const k of cases) {
    it(`the spectral sky entry equals the grey fourth target's V up, V down in every channel (${k.name}, 4 pairs, a ring off the axes)`, () => {
      // the fixed ring turned about the sky (off the axes; K6.Q1c deleted the renderer's turn)
      const directions = vdbFrameAmbientDirections([0, 1, 0], 4).map((d, i): V3 => (i === 0 ? d : [Math.cos(0.77) * d[0] - Math.sin(0.77) * d[2], d[1], Math.sin(0.77) * d[0] + Math.cos(0.77) * d[2]]));
      const greyAt = (_c: number, uvw: V3): [number, number] => {
        const [e] = blob(uvw);
        return [e, e * k.absorption];
      };
      const spectralAt = (_c: number, uvw: V3) => {
        const [e] = blob(uvw);
        return { extinction: [e, e, e] as V3, absorption: [e * k.absorption, e * k.absorption, e * k.absorption] as V3 };
      };
      let occluded = 0;
      for (let i = 0; i < 40; i++) {
        const cell: V3 = [Math.floor(rand() * 8), Math.floor(rand() * 8), Math.floor(rand() * 8)];
        const grey = vdbFrameTransmittanceVoxel3d(greyAt, grid, 0, cell, lights, [0, 1, 0], k.ms, -1, null, { ambientDirections: directions, ...(k.oct ? { octaves: k.oct } : {}) });
        const t = vdbFrameSpectralTransmittanceVoxel3d(spectralAt, grid, 0, cell, lights, [0, 1, 0], k.ms, -1, 0, { ambientDirections: directions, ...(k.oct ? { octaves: k.oct } : {}) });
        const sky = vdbFrameUnpackSpectralGroup(t[0], t[1], t[2]).a;
        for (let c = 0; c < 3; c++) {
          expect(sky.x[c]).toBeCloseTo(grey[3][2], 12);
          expect(sky.y[c]).toBeCloseTo(grey[3][3], 12);
        }
        if (grey[3][2] < 0.99) occluded++;
      }
      // the blob shadows some voxels (the comparison is not all ones)
      expect(occluded).toBeGreaterThan(5);
    });
  }

  it("a chromatic medium gives each channel its own visibility (blue-heavy: blue darkest), the white-furnace sum stays 2", () => {
    const at = (_c: number, uvw: V3) => {
      const [e] = blob(uvw);
      return { extinction: [e * 0.6, e, e * 1.6] as V3, absorption: [0, 0, 0] as V3 };
    };
    const directions = vdbFrameAmbientDirections([0, 1, 0], 4);
    const single = vdbFrameUnpackSpectralGroup(...vdbFrameSpectralTransmittanceVoxel3d(at, grid, 0, [4, 3, 4], lights, [0, 1, 0], 0, -1, 0, { ambientDirections: directions })).a;
    expect(single.x[0]).toBeGreaterThan(single.x[1]);
    expect(single.x[1]).toBeGreaterThan(single.x[2]);
    // diffusion at albedo 1: each pair's two sides sum to 2 (white furnace), so V up + V down = 2 in every channel
    const diffusion = vdbFrameUnpackSpectralGroup(...vdbFrameSpectralTransmittanceVoxel3d(at, grid, 0, [4, 3, 4], lights, [0, 1, 0], 0.6, -1, 0, { ambientDirections: directions })).a;
    for (let c = 0; c < 3; c++) expect(diffusion.x[c]! + diffusion.y[c]!).toBeCloseTo(2, 12);
  });

  it("the spectral pair totals and the per-channel irradiance reduce to the grey ones channel by channel", () => {
    const steps = [0.3, 0.5, 0.2, 0.1];
    const back = [0.05, 0.4];
    for (const [ms, oct] of [[0, vdbFrameOctavesOff], [0.6, vdbFrameOctavesOff], [0, octaves]] as const) {
      const albedo: V3 = oct[0] > 0 ? [1, 1, 1] : [0.7, 0.7, 0.7];
      const g = vdbFrameAmbientPairTotals(vdbFrameShadowMarch(steps, albedo, ms, oct), vdbFrameShadowMarch(back, albedo, ms, oct), ms > 0, oct);
      const s = vdbFrameSpectralAmbientPairTotals(vdbFrameSpectralShadowMarch(steps, [1, 1, 1], albedo, ms, oct), vdbFrameSpectralShadowMarch(back, [1, 1, 1], albedo, ms, oct), ms > 0, oct, albedo);
      for (let c = 0; c < 3; c++) {
        expect(s[0][c]).toBeCloseTo(g[0], 12);
        expect(s[1][c]).toBeCloseTo(g[1], 12);
      }
    }
    const light = { ambient: [0.3, 0.2, 0.1] as V3, hemiSky: [0.5, 0.6, 0.7] as V3, hemiGround: [0.1, 0.05, 0.02] as V3, sky: [0, 1, 0] as V3, probe: [[0.4, 0.5, 0.6], [0.1, 0.12, 0.2], [0.02, 0, -0.03], [-0.05, 0.01, 0.04]] as V3[] };
    const grey = vdbFrameAmbientIrradiance(light, [0.6, 0.3]);
    const spectral = vdbFrameAmbientIrradiance(light, [[0.6, 0.6, 0.6], [0.3, 0.3, 0.3]]);
    for (let c = 0; c < 3; c++) expect(spectral[c]).toBeCloseTo(grey[c]!, 12);
  });
});
