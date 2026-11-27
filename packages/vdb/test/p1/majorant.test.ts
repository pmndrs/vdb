/**
 * P1.2 (brick-majorant builder), P1.3 (the walk's CPU twin) and P1.4 (the nodeMax walk): K1-core §5.4.
 *
 * The property: for every fixture × mode, 1,000 rays × 256 jittered samples, the trilinear sample at a
 * point of the ray is ≤ the bound of the walk segment holding it (+ 1e-6), and ≥ its minimum with minima.
 * The sampled field is the GPU's: f32 tap values (Fp decoded as the WGSL does), `outside` 'background' or
 * 'zero'. A sample within float noise of a segment end may take either neighbour's bound (the walk's t
 * and the sample's cell come from different roundings). Fail-before: a bound one voxel short and the
 * cell's own nodeMax both fail the same check.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { NvdbFile, parseVdb, vdbBrickMajorant, vdbBuildFromDense, vdbSampleTrilinearCpu, type VDBBrickMajorant } from "../../src/tools/index.js";
import { forEachTreeRegion, indexBBox, leafVoxels, majorantWalk, vdbReadValueF32Cpu, type MajorantSegment, type Vec3 } from "../../src/tools/internal.js";
import { pathtracerBrickMajorants, pathtracerDdaSegments, smokeDense, toXMajor } from "../../../../examples/src/harness/p1-smoke.js";

const PRIMITIVES = ["box", "sphere", "torus"].flatMap((shape) => ["float", "fp8", "fpn"].map((type) => `${shape}_fog_${type}`));
const fixtureUrl = (name: string) => new URL(`../../../../fixtures/primitives/${name}.nvdb`, import.meta.url);
const present = PRIMITIVES.every((n) => existsSync(fixtureUrl(n)));
const SMOKE = fileURLToPath(new URL("../../../../fixtures/vdb-samples/smoke.vdb", import.meta.url));
const smokePresent = existsSync(SMOKE);

function load(name: string): Uint32Array {
  const buf = readFileSync(fileURLToPath(fixtureUrl(name)));
  return NvdbFile.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)).gridImage(0);
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The sampled field over the box `[lo, hi]` (voxels) as a dense f32 cache, from the tree's regions, checked
 * against `vdbReadValueF32Cpu` at random voxels (so the cache is not a common-mode copy of the builder's walk).
 */
interface Field {
  lo: Vec3;
  n: Vec3;
  v: Float32Array;
  /** Value of a voxel outside the cache. */
  far: number;
  zero: boolean;
  min: Vec3;
  max: Vec3;
}

function fieldOf(image: Uint32Array, zero: boolean): Field {
  const { min, max } = indexBBox(image);
  // Covers every cell a brick walk visits (its grid ends at most one brick past max + 1).
  const lo: Vec3 = [min[0] - 12, min[1] - 12, min[2] - 12];
  const n: Vec3 = [max[0] - min[0] + 25, max[1] - min[1] + 25, max[2] - min[2] + 25];
  const v = new Float32Array(n[0] * n[1] * n[2]);
  const { background } = forEachTreeRegion(image, () => {});
  v.fill(background);
  const put = (x: number, y: number, z: number, value: number) => {
    const i = x - lo[0], j = y - lo[1], k = z - lo[2];
    if (i < 0 || j < 0 || k < 0 || i >= n[0] || j >= n[1] || k >= n[2]) return;
    v[i + n[0] * (j + n[1] * k)] = value;
  };
  forEachTreeRegion(image, (r) => {
    const [ox, oy, oz] = r.origin;
    if (ox > lo[0] + n[0] || oy > lo[1] + n[1] || oz > lo[2] + n[2] || ox + r.dim < lo[0] || oy + r.dim < lo[1] || oz + r.dim < lo[2]) return;
    if (r.level > 0) {
      for (let z = Math.max(oz, lo[2]); z < Math.min(oz + r.dim, lo[2] + n[2]); z++)
        for (let y = Math.max(oy, lo[1]); y < Math.min(oy + r.dim, lo[1] + n[1]); y++)
          for (let x = Math.max(ox, lo[0]); x < Math.min(ox + r.dim, lo[0] + n[0]); x++) put(x, y, z, r.value);
      return;
    }
    const { values } = leafVoxels(image, r.leaf);
    for (let q = 0; q < 512; q++) put(ox + (q >> 6), oy + ((q >> 3) & 7), oz + (q & 7), values[q]!);
  });
  if (zero) {
    for (let k = 0; k < n[2]; k++)
      for (let j = 0; j < n[1]; j++)
        for (let i = 0; i < n[0]; i++) {
          const x = lo[0] + i, y = lo[1] + j, z = lo[2] + k;
          if (x < min[0] || y < min[1] || z < min[2] || x > max[0] || y > max[1] || z > max[2]) v[i + n[0] * (j + n[1] * k)] = 0;
        }
  }
  // Independent check of the cache: the twin's own reader at random voxels of the box.
  const r = rng(5);
  for (let s = 0; s < 4000; s++) {
    const ijk: Vec3 = [lo[0] + Math.floor(r() * n[0]), lo[1] + Math.floor(r() * n[1]), lo[2] + Math.floor(r() * n[2])];
    const outside = ijk.some((c, a) => c < min[a]! || c > max[a]!);
    const want = zero && outside ? 0 : vdbReadValueF32Cpu(image, ijk);
    const got = v[ijk[0] - lo[0] + n[0] * (ijk[1] - lo[1] + n[1] * (ijk[2] - lo[2]))]!;
    if (!Object.is(want, got) && want !== got) throw new Error(`field cache differs at ${ijk}: ${got} vs ${want}`);
  }
  return { lo, n, v, far: zero ? 0 : background, zero, min, max };
}

function sampleField(F: Field, p: readonly number[]): number {
  const at = (x: number, y: number, z: number) => {
    const i = x - F.lo[0], j = y - F.lo[1], k = z - F.lo[2];
    if (i < 0 || j < 0 || k < 0 || i >= F.n[0] || j >= F.n[1] || k >= F.n[2]) return F.far;
    return F.v[i + F.n[0] * (j + F.n[1] * k)]!;
  };
  const bx = Math.floor(p[0]!), by = Math.floor(p[1]!), bz = Math.floor(p[2]!);
  const fx = p[0]! - bx, fy = p[1]! - by, fz = p[2]! - bz;
  const l = (a: number, b: number, t: number) => a + (b - a) * t;
  const c00 = l(at(bx, by, bz), at(bx + 1, by, bz), fx);
  const c10 = l(at(bx, by + 1, bz), at(bx + 1, by + 1, bz), fx);
  const c01 = l(at(bx, by, bz + 1), at(bx + 1, by, bz + 1), fx);
  const c11 = l(at(bx, by + 1, bz + 1), at(bx + 1, by + 1, bz + 1), fx);
  return l(l(c00, c10, fy), l(c01, c11, fy), fz);
}

interface Ray {
  o: Vec3;
  d: Vec3;
}

/** Rays through the support box: from outside towards a random inner point, some starting inside, some axis-parallel. */
function rays(min: Vec3, max: Vec3, count: number, seed: number): Ray[] {
  const r = rng(seed);
  const out: Ray[] = [];
  const c = [0, 1, 2].map((a) => (min[a]! + max[a]! + 1) / 2);
  const ext = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) + 4;
  for (let s = 0; s < count; s++) {
    const target: Vec3 = [0, 1, 2].map((a) => min[a]! - 1.5 + r() * (max[a]! - min[a]! + 3)) as Vec3;
    let d: Vec3 = [r() - 0.5, r() - 0.5, r() - 0.5];
    if (s % 10 === 7) d[Math.floor(r() * 3)] = 0; // one axis-parallel component
    if (s % 25 === 3) d = [0, 0, 0].map((_, a) => (a === s % 3 ? (r() < 0.5 ? -1 : 1) : 0)) as Vec3; // along an axis
    const len = Math.hypot(...d) || 1;
    d = d.map((v) => (v / len) * (0.5 + r())) as Vec3; // unnormalized on purpose: t is the caller's
    const back = s % 5 === 0 ? 0 : ext / Math.hypot(...d);
    const o: Vec3 = [0, 1, 2].map((a) => Math.fround(target[a]! - d[a]! * back)) as Vec3;
    void c;
    out.push({ o, d: d.map(Math.fround) as Vec3 });
  }
  return out;
}

interface Check {
  samples: number;
  over: number;
  under: number;
  worstOver: number;
  gaps: number;
  outsideNonzero: number;
  segments: number;
  lookups: number;
}

/** The §5.4 property over one walk per ray. */
function check(F: Field, walk: (ray: Ray) => MajorantSegment[], R: Ray[], samplesPerRay: number, seed: number, minima: boolean): Check {
  const r = rng(seed);
  const res: Check = { samples: 0, over: 0, under: 0, worstOver: 0, gaps: 0, outsideNonzero: 0, segments: 0, lookups: 0 };
  for (const ray of R) {
    const segs = walk(ray);
    res.segments += segs.length;
    for (const s of segs) res.lookups += s.lookups;
    for (let i = 1; i < segs.length; i++) if (segs[i]!.t0 !== segs[i - 1]!.t1) res.gaps++;
    for (const s of segs) if (!(s.t1 > s.t0)) res.gaps++;
    const p = (t: number) => [0, 1, 2].map((a) => ray.o[a]! + ray.d[a]! * t);
    if (segs.length === 0) {
      // A ray the walk misses must see a zero field (zero mode): sample along a stretch of it.
      if (F.zero) for (let k = 0; k < 16; k++) if (sampleField(F, p(r() * 400)) !== 0) res.outsideNonzero++;
      continue;
    }
    const T0 = segs[0]!.t0;
    const T1 = segs[segs.length - 1]!.t1;
    let j = 0;
    for (let k = 0; k < samplesPerRay; k++) {
      const t = T0 + ((k + r()) / samplesPerRay) * (T1 - T0);
      while (j < segs.length - 1 && t >= segs[j]!.t1) j++;
      const v = sampleField(F, p(t));
      const eps = 1e-5 * Math.max(1, Math.abs(t));
      let bound = segs[j]!.max;
      let floor = segs[j]!.min;
      if (j > 0 && t - segs[j]!.t0 < eps) {
        bound = Math.max(bound, segs[j - 1]!.max);
        floor = Math.min(floor, segs[j - 1]!.min);
      }
      if (j < segs.length - 1 && segs[j]!.t1 - t < eps) {
        bound = Math.max(bound, segs[j + 1]!.max);
        floor = Math.min(floor, segs[j + 1]!.min);
      }
      res.samples++;
      if (v > bound + 1e-6) {
        res.over++;
        res.worstOver = Math.max(res.worstOver, v - bound);
      }
      if (minima && v < floor - 1e-6) res.under++;
    }
    if (F.zero) {
      // Beyond the walk's coverage the zero-mode field is 0.
      for (let k = 0; k < 8; k++) {
        const t = T1 + 1e-3 + r() * 50;
        if (sampleField(F, p(t)) !== 0) res.outsideNonzero++;
      }
    }
  }
  return res;
}

const RAYS = 1000;
const SAMPLES = 256;

describe.skipIf(!present)("P1.2–P1.4 majorant property (K1-core §5.4): 1000 rays × 256 samples, nine fixtures", () => {
  it.each(PRIMITIVES)("%s: brick / global / nodeMax bound every sample, both outside modes", (name) => {
    const image = load(name);
    const { min, max } = indexBBox(image);
    const R = rays(min, max, RAYS, 17);
    for (const outside of ["background", "zero"] as const) {
      const F = fieldOf(image, outside === "zero");
      const brick = vdbBrickMajorant(image, { minima: true, outside });
      const runs: [string, (ray: Ray) => MajorantSegment[], boolean][] = [
        ["brick", (ray) => majorantWalk(image, brick, ray.o, ray.d, { outside, minima: true }), true],
        ["global", (ray) => majorantWalk(image, null, ray.o, ray.d, { outside, mode: "global" }), false],
        ["nodeMax", (ray) => majorantWalk(image, null, ray.o, ray.d, { outside, mode: "nodeMax" }), false],
      ];
      for (const [mode, walk, minima] of runs) {
        const c = check(F, walk, R, SAMPLES, 23, minima);
        const tag = `${name} ${outside} ${mode}`;
        expect(c.samples, tag).toBeGreaterThan(RAYS * SAMPLES * 0.9);
        expect(c.over, `${tag}: ${c.over} samples over the bound (worst ${c.worstOver})`).toBe(0);
        expect(c.under, `${tag}: samples under the minimum`).toBe(0);
        expect(c.gaps, `${tag}: segments not contiguous`).toBe(0);
        expect(c.outsideNonzero, `${tag}: field non-zero outside the walk`).toBe(0);
      }
    }
  });

  it("fail-before: a brick grid one voxel short, and the cell's own nodeMax, violate the same property", () => {
    const image = load("sphere_fog_float");
    const { min, max } = indexBBox(image);
    const R = rays(min, max, 300, 29);
    const F = fieldOf(image, true);
    const brick = vdbBrickMajorant(image, { outside: "zero" });
    // Shift the cells by one voxel: each cell's bound then misses its last tap layer.
    const short: VDBBrickMajorant = { ...brick, origin: [brick.origin![0] + 1, brick.origin![1] + 1, brick.origin![2] + 1] };
    const a = check(F, (ray) => majorantWalk(image, short, ray.o, ray.d, { outside: "zero" }), R, SAMPLES, 31, false);
    expect(a.over).toBeGreaterThan(0);
    // Per-cell nodeMax (no 2x2x2 support): bound = the cell corner's own node max.
    const own = (ray: Ray) =>
      majorantWalk(image, null, ray.o, ray.d, { outside: "zero", mode: "nodeMax" }).map((s) => {
        const p = [0, 1, 2].map((k) => Math.floor(ray.o[k]! + ray.d[k]! * ((s.t0 + s.t1) / 2))) as Vec3;
        return { ...s, max: vdbReadValueF32Cpu(image, p) };
      });
    const b = check(F, own, R, SAMPLES, 37, false);
    expect(b.over).toBeGreaterThan(0);
  });

  it("the field cache equals vdbSampleTrilinearCpu (Float: exact taps) at random points", () => {
    const image = load("torus_fog_float");
    const { min, max } = indexBBox(image);
    const r = rng(41);
    for (const outside of ["background", "zero"] as const) {
      const F = fieldOf(image, outside === "zero");
      for (let s = 0; s < 3000; s++) {
        const p: Vec3 = [0, 1, 2].map((a) => min[a]! - 2 + r() * (max[a]! - min[a]! + 4)) as Vec3;
        expect(Math.abs(sampleField(F, p) - vdbSampleTrilinearCpu(image, p, { outside }))).toBeLessThanOrEqual(1e-12);
      }
    }
  });
});

describe.skipIf(!smokePresent)("P1.2 align 'pathtracer' equals its brickMajorants; the walk equals its DDA", () => {
  const buf = smokePresent ? readFileSync(SMOKE) : Buffer.alloc(0);
  const dense = smokePresent ? smokeDense(parseVdb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))) : null;

  it("vdb-smoke: dims, maxima and minima bit-equal", () => {
    const { resolution: res, density } = dense!;
    const image = vdbBuildFromDense(toXMajor(density, res), res);
    const ours = vdbBrickMajorant(image, { align: "pathtracer", resolution: res, minima: true });
    const theirs = pathtracerBrickMajorants(res, density, { minima: true });
    expect(ours.dims).toEqual(theirs.resolution);
    expect(ours.origin).toEqual([-1, -1, -1]);
    expect(Array.from(ours.data)).toEqual(Array.from(theirs.values));
    expect(Array.from(ours.minima!)).toEqual(Array.from(theirs.minima!));
    expect(theirs.values.some((v) => v > 0) && theirs.minima!.some((v) => v > 0)).toBe(true);
  });

  it("a random dense block touching every face (its minima edge rule included)", () => {
    const res: [number, number, number] = [19, 9, 24];
    const r = rng(43);
    const density = new Float32Array(res[0] * res[1] * res[2]).map(() => (r() < 0.15 ? 0 : Math.fround(0.05 + r())));
    const image = vdbBuildFromDense(toXMajor(density, res), res, { collapse: false });
    const ours = vdbBrickMajorant(image, { align: "pathtracer", resolution: res, minima: true });
    const theirs = pathtracerBrickMajorants(res, density, { minima: true });
    expect(Array.from(ours.data)).toEqual(Array.from(theirs.values));
    expect(Array.from(ours.minima!)).toEqual(Array.from(theirs.minima!));
  });

  it("rejects a grid outside [0, resolution) and 'background' with align 'pathtracer'", () => {
    const image = vdbBuildFromDense(new Float32Array(8).fill(1), [2, 2, 2], { origin: [-1, 0, 0] });
    expect(() => vdbBrickMajorant(image, { align: "pathtracer" })).toThrow(/inside \[0, resolution - 1\]/);
    expect(() => vdbBrickMajorant(image, { align: "pathtracer", outside: "background" })).toThrow(/drop outside/);
  });

  it("vdb-smoke: the brick walk's segments are its ddaSegments (zero-length ones dropped), t to 1e-5", () => {
    const { resolution: res, density } = dense!;
    const image = vdbBuildFromDense(toXMajor(density, res), res);
    const maj = vdbBrickMajorant(image, { align: "pathtracer", resolution: res });
    const R = rays([0, 0, 0], [res[0] - 1, res[1] - 1, res[2] - 1], 500, 47);
    let compared = 0;
    for (const ray of R) {
      const ours = majorantWalk(image, maj, ray.o, ray.d, { outside: "zero", tMax: 1e6 });
      const theirs = pathtracerDdaSegments(maj.dims, ray.o, ray.d, 1e6).filter((s) => s.t1 > s.t0);
      expect(ours.length).toBe(theirs.length);
      ours.forEach((s, i) => {
        const t = theirs[i]!;
        const ci = t.cell[0] + maj.dims[0] * (t.cell[1] + maj.dims[1] * t.cell[2]);
        expect(s.max).toBe(maj.data[ci]);
        expect(Math.abs(s.t0 - t.t0)).toBeLessThanOrEqual(1e-5 * Math.max(1, Math.abs(t.t0)));
        expect(Math.abs(s.t1 - t.t1)).toBeLessThanOrEqual(1e-5 * Math.max(1, Math.abs(t.t1)));
        compared++;
      });
    }
    expect(compared).toBeGreaterThan(5000);
  });
});
