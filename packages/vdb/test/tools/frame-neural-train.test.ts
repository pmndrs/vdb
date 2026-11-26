import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  vdbFrameMlpCheck,
  vdbFrameMlpForward,
  vdbFrameMlpFromNeural,
  vdbFrameMlpInputs,
  vdbFrameNeuralWeightsProcedural,
} from "../../src/tools/internal.js";

// K6.N: the trainer of our own 'neural' / 'mlp' weights (scripts/k6/neural/*.mjs, derived from his scripts/neural-indirect).
// The scripts are plain ESM without types: loaded by path.
const dir = new URL("../../../../scripts/k6/neural/", import.meta.url);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const load = async (name: string): Promise<any> => import(/* @vite-ignore */ new URL(name, dir).href);

describe("K6.N neural trainer (scripts/k6/neural)", () => {
  it("its features are the core's inputs (vdbFrameMlpInputs) on a procedural medium", async () => {
    const features = await load("features.mjs");
    const media = await load("media.mjs");
    const medium = media.createMedium("cumulus", 5);
    const { grid, data } = media.buildMediumGrid(medium, { voxelDepth: 3, albedo: 0.8, offset: [0.01, 0.02, 0.03] });
    const levels = features.buildMips(grid, data);
    const sample = (level: number, w: number[]): [number, number] => {
      const u = (w[0]! - grid.min[0]) / grid.size[0];
      const v = (w[1]! - grid.min[1]) / grid.size[1];
      const t = (w[2]! - grid.min[2]) / grid.size[2];
      if (u < 0 || u > 1 || v < 0 || v > 1 || t < 0 || t > 1) return [0, 0];
      const out = features.sampleLevel(levels[level], u, v, t, [0, 0]);
      return [out[0], out[1]];
    };
    let seed = 11;
    const r = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296;
    let checked = 0;
    for (let i = 0; i < 40; i++) {
      const position = [0, 1, 2].map((k) => grid.min[k] + r() * grid.size[k]);
      const z = 1 - 2 * r();
      const phi = 2 * Math.PI * r();
      const direction = [Math.sqrt(1 - z * z) * Math.cos(phi), Math.sqrt(1 - z * z) * Math.sin(phi), z];
      const depths = features.marchShadow(grid, levels, position, direction, 0.2);
      const mine = features.getFeatures(grid, levels, position, direction, depths, new Float32Array(features.neuralInputs));
      const core = vdbFrameMlpInputs(sample as never, 6, grid.unit, position as never, direction as never, depths[0]);
      for (let k = 0; k < core.length; k++) expect(mine[k]).toBeCloseTo(core[k]!, 4);
      checked++;
    }
    expect(checked).toBe(40);
  });

  it("its path tracer matches a quadrature of single scattering in a uniform medium; ratio tracking agrees with the grid", async () => {
    const features = await load("features.mjs");
    const trace = await load("trace.mjs");
    const grid = features.fitGrid([-1, -1, -1], [1, 1, 1]);
    const sigma = 0.5;
    const data = new Float32Array(grid.resolution ** 3 * 2);
    for (let i = 0; i < data.length; i += 2) data[i] = sigma;
    const light = [0.3, 0.5, 0.8].map((v, _, a) => v / Math.hypot(a[0]!, a[1]!, a[2]!));
    const scene = trace.createTraceScene(grid, data, light, 1);
    const centre = [0, 1, 2].map((k) => grid.min[k] + grid.size[k] * [0.5, 0.45, 0.55][k]!);
    const g = 0.8;
    const hg = (c: number) => (1 - g * g) / (4 * Math.PI * Math.pow(1 + g * g - 2 * g * c, 1.5));
    const exitDistance = (p: number[], d: number[]) => {
      let t = Infinity;
      for (let k = 0; k < 3; k++) {
        if (d[k]! > 0) t = Math.min(t, (grid.min[k]! + grid.size[k]! - p[k]!) / d[k]!);
        else if (d[k]! < 0) t = Math.min(t, (grid.min[k]! - p[k]!) / d[k]!);
      }
      return t;
    };
    // 4π · mean over directions of ∫ σ e^{−σ s} HG(l·ω) e^{−σ d_light(q)} ds
    const n = 1200;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const z = 1 - (2 * (i + 0.5)) / n;
      const phi = Math.PI * (3 - Math.sqrt(5)) * i;
      const w = [Math.sqrt(1 - z * z) * Math.cos(phi), Math.sqrt(1 - z * z) * Math.sin(phi), z];
      const smax = exitDistance(centre, w);
      const steps = 160;
      const ds = smax / steps;
      let line = 0;
      for (let s = 0; s < steps; s++) {
        const t = (s + 0.5) * ds;
        const q = centre.map((c, k) => c + w[k]! * t);
        line += sigma * Math.exp(-sigma * t) * Math.exp(-sigma * exitDistance(q, light)) * ds;
      }
      sum += hg(light[0]! * w[0]! + light[1]! * w[1]! + light[2]! * w[2]!) * line;
    }
    const expected = 4 * Math.PI * (sum / n);
    const voxelPoint = centre.map((c, k) => (c - grid.min[k]!) / grid.voxel[k]!);
    const estimate = (nee: string) => {
      const random = features.createRandom(99);
      let total = 0;
      let var2 = 0;
      const batches = 40;
      for (let b = 0; b < batches; b++) {
        const e = trace.estimateScatteredFluence(scene, voxelPoint, 2500, random, g, { maxBounces: 1, nee });
        total += e.mean;
        var2 += e.variance;
      }
      return { mean: total / batches, error: Math.sqrt(var2) / batches };
    };
    const byGrid = estimate("grid");
    const byRatio = estimate("ratio");
    expect(Math.abs(byGrid.mean - expected)).toBeLessThan(4 * byGrid.error + 0.01 * expected);
    expect(Math.abs(byRatio.mean - expected)).toBeLessThan(4 * byRatio.error + 0.01 * expected);
  });

  it("makes media from seeds alone: deterministic, in [0, 1], and its sources read no asset", async () => {
    const media = await load("media.mjs");
    for (const kind of media.mediumKinds as string[]) {
      const a = media.createMedium(kind, 3);
      const b = media.createMedium(kind, 3);
      let any = 0;
      for (let i = 0; i < 200; i++) {
        const p = [Math.sin(i * 1.7) * 0.9, Math.cos(i * 2.3) * 0.7, Math.sin(i * 0.9 + 1) * 0.9];
        const v = a.density(p[0], p[1], p[2]);
        expect(v).toBe(b.density(p[0], p[1], p[2]));
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
        any += v;
      }
      expect(any).toBeGreaterThan(0);
    }
    // D35 (2): procedural media only. A static guard: the training sources import no file reader or decoder.
    for (const name of ["media.mjs", "features.mjs", "trace.mjs"]) {
      const text = readFileSync(new URL(name, dir), "utf8");
      expect(text).not.toMatch(/node:fs|readFileSync|wvdb|\.vdb|\.nvdb|inflateSync/i);
    }
  });

  it("our weights are a valid generic MLP of the same shape as his, with finite outputs", () => {
    const mlp = vdbFrameMlpFromNeural(vdbFrameNeuralWeightsProcedural);
    expect(vdbFrameMlpCheck(mlp)).toBeNull();
    expect(mlp.layers.map((l) => [l.inputs, l.outputs])).toEqual([
      [20, 32],
      [32, 32],
      [32, 1],
    ]);
    const x = Array.from({ length: 20 }, (_, i) => (i % 7) * 0.4);
    const y = vdbFrameMlpForward(mlp, x);
    expect(Number.isFinite(y)).toBe(true);
    expect(y).toBeGreaterThan(0);
  });
});
