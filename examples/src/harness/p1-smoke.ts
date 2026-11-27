/**
 * P1: pmndrs/pathtracer's `vdb-smoke` dense grid, rebuilt here from `fixtures/vdb-samples/smoke.vdb`
 * (CC BY-SA 4.0, git-ignored), and the pathtracer's dense trilinear (its point-sample contract).
 *
 * `smokeDense` follows its `bench/scenes/volumes/vdb-convert.ts` `convertVdbFile` (pmndrs/pathtracer, MIT,
 * read 2026-11-28 at main 06488fa): 2³ voxel blocks averaged (inactive voxels count as 0, non-positive
 * values skipped), a 2-cell zero border, scaled to a peak of 1, x fastest. Its `vdb-smoke.f32`
 * (`bench/assets/generated/volumes/`, git-ignored there too) is this array; the P1.1 test checks that
 * bit for bit when the file is present (`VDB_SMOKE_F32`, or the sibling checkout's path).
 *
 * `denseTrilinear` is its `gridDensity` in index space: voxel centres at integers, zero outside
 * `[0, n - 1]`, a weighted sum of the 8 taps in f64.
 *
 * Browser-safe (no node:fs): the P1 harness page (p1-parity.ts) and the vitest tests (packages/vdb/test/p1) share it.
 */
import type { VdbFile } from "../../../packages/vdb/src/tools/index.js";

export interface SmokeDense {
  resolution: [number, number, number];
  /** x fastest: `density[i + nx * (j + ny * k)]`. */
  density: Float32Array;
  peak: number;
}

/** The pathtracer's `convertVdbFile(vdb, …, { downsample: 2 })` (border 2, first float grid). */
export function smokeDense(vdb: VdbFile, down = 2, border = 2): SmokeDense {
  const grid = vdb.grids.find((g) => g.gridType.startsWith("Tree_float"));
  if (!grid || !("valueType" in grid) || grid.valueType !== "float" || !grid.indexBBox) throw new Error("smokeDense: no float grid with an index bbox");
  const { min, max } = grid.indexBBox;
  const dim = [0, 1, 2].map((a) => max[a]! - min[a]! + 1);
  const cells = dim.map((d) => Math.ceil(d / down));
  const res = cells.map((c) => c + 2 * border) as [number, number, number];
  const sums = new Float64Array(res[0] * res[1] * res[2]);
  for (const leaf of grid.iterLeaves()) {
    for (let lx = 0; lx < 8; lx++)
      for (let ly = 0; ly < 8; ly++)
        for (let lz = 0; lz < 8; lz++) {
          const li = lx * 64 + ly * 8 + lz;
          if (!((leaf.valueMask[li >> 5]! >>> (li & 31)) & 1)) continue;
          const v = leaf.values[li]!;
          if (!(v > 0)) continue;
          const i = Math.floor((leaf.origin[0] + lx - min[0]!) / down) + border;
          const j = Math.floor((leaf.origin[1] + ly - min[1]!) / down) + border;
          const k = Math.floor((leaf.origin[2] + lz - min[2]!) / down) + border;
          if (i < border || j < border || k < border || i >= res[0] - border || j >= res[1] - border || k >= res[2] - border) continue;
          sums[i + res[0] * (j + res[1] * k)]! += v;
        }
  }
  let peak = 0;
  const density = new Float32Array(sums.length);
  for (let i = 0; i < sums.length; i++) {
    density[i] = sums[i]! / down ** 3;
    peak = Math.max(peak, density[i]!);
  }
  for (let i = 0; i < density.length; i++) density[i] = density[i]! / peak;
  return { resolution: res, density, peak };
}

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** P1.1's 10k index points: uniform over the grid padded by 2 voxels, near every face, on voxel centres and half-way between. */
export function contractPoints(res: readonly [number, number, number], count = 10_000, seed = 11): [number, number, number][] {
  const r = rng(seed);
  const pts: [number, number, number][] = [];
  const axis = (a: number) => -2 + r() * (res[a]! + 3);
  while (pts.length < count) {
    const kind = pts.length % 5;
    const p: [number, number, number] = [axis(0), axis(1), axis(2)];
    if (kind === 1) {
      const a = Math.floor(r() * 3);
      p[a] = r() < 0.5 ? -1 + (r() - 0.5) * 0.2 : res[a]! - 1 + (r() - 0.5) * 2.2; // across the low and high faces
    } else if (kind === 2) {
      for (let a = 0; a < 3; a++) p[a] = Math.round(p[a]!); // voxel centres
    } else if (kind === 3) {
      for (let a = 0; a < 3; a++) p[a] = Math.floor(p[a]!) + 0.5; // half-way
    }
    pts.push(p.map(Math.fround) as [number, number, number]); // the GPU reads f32
  }
  return pts;
}

/** x-fastest (`i + nx * (j + ny * k)`) to `vdbBuildFromDense`'s x-major order (`(i * ny + j) * nz + k`). */
export function toXMajor(density: Float32Array, [nx, ny, nz]: readonly [number, number, number]): Float32Array {
  const out = new Float32Array(density.length);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) out[(i * ny + j) * nz + k] = density[i + nx * (j + ny * k)]!;
  return out;
}

/** The pathtracer's `gridDensity` in index space (identity `worldToIndex`): zero outside `[0, n - 1]`. */
export function denseTrilinear(density: ArrayLike<number>, [nx, ny, nz]: readonly [number, number, number], p: readonly [number, number, number]): number {
  const at = (i: number, j: number, k: number) => (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? 0 : density[i + nx * (j + ny * k)]!);
  const i0 = Math.floor(p[0]), j0 = Math.floor(p[1]), k0 = Math.floor(p[2]);
  const fx = p[0] - i0, fy = p[1] - j0, fz = p[2] - k0;
  let acc = 0;
  for (let c = 0; c < 8; c++) {
    const di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
    acc += at(i0 + di, j0 + dj, k0 + dk) * (di ? fx : 1 - fx) * (dj ? fy : 1 - fy) * (dk ? fz : 1 - fz);
  }
  return acc;
}

/** The pathtracer's `brickMajorants(resolution, data, { minima })` (bench and kernels, `packages/pathtracer/src/volume/volume.ts`, MIT). */
export function pathtracerBrickMajorants(
  resolution: readonly [number, number, number],
  data: ArrayLike<number>,
  options: { minima?: boolean } = {},
): { resolution: [number, number, number]; values: Float32Array; minima?: Float32Array } {
  const BRICK = 8;
  const [nx, ny, nz] = resolution;
  const dims = [0, 1, 2].map((a) => Math.floor((resolution[a]! + BRICK) / BRICK)) as [number, number, number];
  const out = new Float32Array(dims[0] * dims[1] * dims[2]);
  const mins = options.minima ? new Float32Array(out.length) : null;
  for (let ck = 0; ck < dims[2]; ck++)
    for (let cj = 0; cj < dims[1]; cj++)
      for (let ci = 0; ci < dims[0]; ci++) {
        let m = 0;
        let lo = Infinity;
        if (mins && (ci * BRICK - 1 < 0 || ci * BRICK + BRICK > nx - 1 || cj * BRICK - 1 < 0 || cj * BRICK + BRICK > ny - 1 || ck * BRICK - 1 < 0 || ck * BRICK + BRICK > nz - 1)) lo = 0;
        for (let k = Math.max(0, ck * BRICK - 1); k <= Math.min(nz - 1, ck * BRICK + BRICK - 1); k++)
          for (let j = Math.max(0, cj * BRICK - 1); j <= Math.min(ny - 1, cj * BRICK + BRICK - 1); j++)
            for (let i = Math.max(0, ci * BRICK - 1); i <= Math.min(nx - 1, ci * BRICK + BRICK - 1); i++) {
              const v = data[i + nx * (j + ny * k)]!;
              m = Math.max(m, v);
              if (v < lo) lo = v;
            }
        out[ci + dims[0] * (cj + dims[1] * ck)] = m;
        if (mins) mins[ci + dims[0] * (cj + dims[1] * ck)] = Math.min(lo, m);
      }
  return { resolution: dims, values: out, ...(mins ? { minima: mins } : {}) };
}

/** The pathtracer's `ddaSegments(dims, a, b, tmax)` (f64; zero-length segments included). */
export function pathtracerDdaSegments(
  dims: readonly [number, number, number],
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  tmax: number,
): { t0: number; t1: number; cell: [number, number, number] }[] {
  const BRICK = 8;
  const ua = a.map((v) => (v + 1) / BRICK), ub = b.map((v) => v / BRICK);
  let tEnter = 0, tExit = tmax;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(ub[k]!) < 1e-20) {
      if (ua[k]! < 0 || ua[k]! >= dims[k]!) return [];
    } else {
      const t0 = (0 - ua[k]!) / ub[k]!, t1 = (dims[k]! - ua[k]!) / ub[k]!;
      tEnter = Math.max(tEnter, Math.min(t0, t1));
      tExit = Math.min(tExit, Math.max(t0, t1));
    }
  }
  if (!(tEnter < tExit)) return [];
  const cell = [0, 1, 2].map((k) => Math.min(dims[k]! - 1, Math.max(0, Math.floor(ua[k]! + ub[k]! * tEnter))));
  const step = ub.map((v) => (v > 0 ? 1 : -1));
  const tNext = [0, 1, 2].map((k) => (Math.abs(ub[k]!) < 1e-20 ? 1e30 : (cell[k]! + (ub[k]! > 0 ? 1 : 0) - ua[k]!) / ub[k]!));
  const dt = ub.map((v) => (Math.abs(v) < 1e-20 ? 1e30 : Math.abs(1 / v)));
  const out: { t0: number; t1: number; cell: [number, number, number] }[] = [];
  let t = tEnter;
  for (let guard = 0; guard < 4096; guard++) {
    let axis = 0;
    if (tNext[1]! < tNext[axis]!) axis = 1;
    if (tNext[2]! < tNext[axis]!) axis = 2;
    const t1 = Math.min(tNext[axis]!, tExit);
    out.push({ t0: t, t1, cell: [cell[0]!, cell[1]!, cell[2]!] });
    t = t1;
    if (t1 >= tExit) break;
    cell[axis] = cell[axis]! + step[axis]!;
    tNext[axis] = tNext[axis]! + dt[axis]!;
    if (cell[axis]! < 0 || cell[axis]! >= dims[axis]!) break;
  }
  return out;
}
