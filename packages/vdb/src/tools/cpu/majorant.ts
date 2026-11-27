/**
 * Majorants for delta and ratio tracking (P1.2, P1.3, P1.4; docs/dev/design/K1-core.md §5).
 *
 * {@link vdbBrickMajorant} builds the brick grid the core's `majorant` WGSL profile walks: one upper
 * (and optionally lower) bound per cell of `2^brickLog2` index voxels, over the SAMPLED field. A
 * trilinear sample at `p` reads voxels `floor(p)` and `floor(p) + 1`, so a cell that holds the
 * positions `[o + B c, o + B (c + 1))` (origin `o`, edge `B`) bounds the voxels `[o + B c, o + B c + B]`
 * (its "taps", `B + 1` per axis): leaf voxels active or not, tile values, and the background.
 *
 * {@link majorantWalk} is the CPU twin of `vdbMajorantInit` / `vdbMajorantNext` (all three modes),
 * in f32 in the WGSL's evaluation order. Like the other twins, GPU parity compares `t` with a
 * tolerance (a GPU may fuse a multiply-add or divide less exactly) and the bounds exactly.
 */
import { VDBToolsError } from "../errors.js";
import type { VDBVec3 } from "../vdb/types.js";
import {
  cellBoundTrilinear,
  defaultValueMax,
  forEachTreeRegion,
  getDim,
  hddaInit,
  hddaStep,
  hddaUpdate,
  indexBBox,
  leafVoxels,
  type Vec3,
} from "./march.js";

const f = Math.fround;

/**
 * Brick-majorant bounds for one grid: a dense grid of cells, x fastest, cell `c` covering the index
 * positions `[origin + B c, origin + B (c + 1))` with `B = 2^brickLog2`. Built by
 * {@link vdbBrickMajorant}; the core's `VDBGridPool.setMajorant` stores it next to the grid
 * (the core exports this same type).
 */
export interface VDBBrickMajorant {
  /** log2 of the brick edge in index voxels (3 for 8³ bricks). */
  brickLog2: number;
  /** Cells along x, y, z. */
  dims: VDBVec3;
  /** `dims[0] * dims[1] * dims[2]` upper bounds, x fastest. */
  data: Float32Array;
  /**
   * Index position of cell 0's min corner. Default (when a host builds the grid itself and leaves it out):
   * the grid's `indexBBoxMin - 1`, where the trilinear support starts.
   */
  origin?: VDBVec3;
  /** Lower bounds, as `data` (optional; the `minima` module option reads them). */
  minima?: Float32Array;
}

/** Options of {@link vdbBrickMajorant}. */
export interface VDBBrickMajorantOptions {
  /** log2 of the brick edge (0..10). Default 3 (8³ voxels, the pathtracer's brick). */
  brickLog2?: number;
  /** Also build lower bounds (`minima`). Default false. */
  minima?: boolean;
  /**
   * Where the cells sit.
   * - `'support'` (default): origin `indexBBoxMin - 1`, `ceil((max - min + 2) / B)` cells per axis: the
   *   smallest grid that covers the trilinear support `[min - 1, max + 1)`.
   * - `'pathtracer'`: pmndrs/pathtracer's `brickMajorants` layout for a dense grid of `resolution` voxels
   *   at index `[0, n)`: origin -1, `floor((n + B) / B)` cells, cell c over `[B c - 1, B c + B - 1)`. Bounds
   *   are at least 0 and a cell whose taps reach `[0, n - 1]`'s faces has minimum 0 (its rule, kept
   *   bit for bit, so the arrays equal its `brickMajorants` on the same dense array). Implies
   *   `outside: 'zero'`; the grid's bbox must lie in `[0, resolution - 1]`.
   */
  align?: "support" | "pathtracer";
  /**
   * The field the bounds are for, as the module's `outside` option: `'background'` (default) reads the
   * tree everywhere; `'zero'` is 0 outside the index bbox (a tap outside it reads 0).
   */
  outside?: "background" | "zero";
  /** `align: 'pathtracer'`: the dense grid's voxel counts. Default `indexBBoxMax + 1`. */
  resolution?: VDBVec3;
}

function bad(message: string): never {
  throw new VDBToolsError("VDB_BAD_OPTION", `vdbBrickMajorant: ${message}`);
}

/**
 * Builds brick-majorant bounds of a NanoVDB grid image (P1.2) for the core's `majorant` profile
 * (`pool.setMajorant(id, vdbBrickMajorant(image))`, module option `majorant: 'brick'`).
 *
 * A cell's bound is the max (and with `minima`, the min) of every value a trilinear sample inside it
 * can read: its `B + 1` taps per axis, leaf voxels active or not, tiles and the background, or 0 for a
 * tap outside the index bbox with `outside: 'zero'`. So every trilinear (or nearest) sample in the cell
 * lies in `[minima, data]`. Works on every scalar grid type (Fp values as the GPU decodes them).
 *
 * @param image The grid image (little-endian u32 words, `GridData` at word 0).
 * @param options Brick size, minima, alignment and the `outside` field.
 * @throws VDBToolsError `VDB_BAD_OPTION` for invalid options or a bbox outside `resolution`.
 */
export function vdbBrickMajorant(image: Uint32Array, options: VDBBrickMajorantOptions = {}): VDBBrickMajorant {
  const log2 = options.brickLog2 ?? 3;
  if (!Number.isInteger(log2) || log2 < 0 || log2 > 10) bad(`brickLog2 must be an integer in 0..10, got ${String(log2)}.`);
  const align = options.align ?? "support";
  if (align !== "support" && align !== "pathtracer") bad(`align must be 'support' or 'pathtracer', got ${JSON.stringify(align)}.`);
  if (options.outside !== undefined && options.outside !== "background" && options.outside !== "zero") {
    bad(`outside must be 'background' or 'zero', got ${JSON.stringify(options.outside)}.`);
  }
  if (align === "pathtracer" && options.outside === "background") bad("align 'pathtracer' samples zero outside the grid; drop outside: 'background'.");
  const zero = align === "pathtracer" || options.outside === "zero";
  const B = 1 << log2;
  const { min, max } = indexBBox(image);
  const empty = max[0] < min[0] || max[1] < min[1] || max[2] < min[2];

  let origin: Vec3;
  let dims: Vec3;
  let n: Vec3 = [0, 0, 0];
  if (align === "support") {
    origin = empty ? [0, 0, 0] : [min[0] - 1, min[1] - 1, min[2] - 1];
    dims = empty ? [1, 1, 1] : [0, 1, 2].map((a) => Math.ceil((max[a]! - min[a]! + 2) / B)) as Vec3;
  } else {
    const res = options.resolution ?? (empty ? [1, 1, 1] : [max[0] + 1, max[1] + 1, max[2] + 1]);
    if (!Array.isArray(res) || res.length !== 3 || res.some((v) => !Number.isInteger(v) || v < 1)) {
      bad(`resolution must be [nx, ny, nz] positive integers, got ${JSON.stringify(res)}.`);
    }
    n = [res[0], res[1], res[2]];
    if (!empty && (min.some((v) => v < 0) || max.some((v, a) => v >= n[a]!))) {
      bad(`align 'pathtracer' needs the grid's bbox [${min.join(",")}]..[${max.join(",")}] inside [0, resolution - 1] = [0,0,0]..[${n.map((v) => v - 1).join(",")}].`);
    }
    origin = [-1, -1, -1];
    dims = [0, 1, 2].map((a) => Math.floor((n[a]! + B) / B)) as Vec3;
  }

  const cells = dims[0] * dims[1] * dims[2];
  const hi = new Float64Array(cells).fill(-Infinity);
  const lo = options.minima ? new Float64Array(cells).fill(Infinity) : null;
  // Tap extent of the whole grid of cells: [origin, origin + B * dims].
  const extLo: Vec3 = [origin[0], origin[1], origin[2]];
  const extHi: Vec3 = [origin[0] + B * dims[0], origin[1] + B * dims[1], origin[2] + B * dims[2]];
  // Where values come from the tree: the bbox (zero) or the extent (background).
  const domLo: Vec3 = zero ? [Math.max(min[0], extLo[0]), Math.max(min[1], extLo[1]), Math.max(min[2], extLo[2])] : extLo;
  const domHi: Vec3 = zero ? [Math.min(max[0], extHi[0]), Math.min(max[1], extHi[1]), Math.min(max[2], extHi[2])] : extHi;
  const cellLo = (a: number, axis: number) => Math.max(0, Math.ceil((a - origin[axis]! - B) / B));
  const cellHi = (b: number, axis: number) => Math.min(dims[axis]! - 1, Math.floor((b - origin[axis]!) / B));

  /** Max/min `v` into every cell whose taps meet the voxel box [a, b] (inclusive), after clipping to the domain. */
  const add = (a0: number, a1: number, a2: number, b0: number, b1: number, b2: number, v: number) => {
    if (empty && zero) return;
    const x0 = Math.max(a0, domLo[0]), y0 = Math.max(a1, domLo[1]), z0 = Math.max(a2, domLo[2]);
    const x1 = Math.min(b0, domHi[0]), y1 = Math.min(b1, domHi[1]), z1 = Math.min(b2, domHi[2]);
    if (x0 > x1 || y0 > y1 || z0 > z1) return;
    const cx0 = cellLo(x0, 0), cx1 = cellHi(x1, 0);
    const cy0 = cellLo(y0, 1), cy1 = cellHi(y1, 1);
    const cz0 = cellLo(z0, 2), cz1 = cellHi(z1, 2);
    for (let cz = cz0; cz <= cz1; cz++) {
      for (let cy = cy0; cy <= cy1; cy++) {
        let i = cx0 + dims[0] * (cy + dims[1] * cz);
        for (let cx = cx0; cx <= cx1; cx++, i++) {
          if (v > hi[i]!) hi[i] = v;
          if (lo && v < lo[i]!) lo[i] = v;
        }
      }
    }
  };

  const { background, rootTiles } = forEachTreeRegion(image, (r) => {
    const [ox, oy, oz] = r.origin;
    if (ox > domHi[0] || oy > domHi[1] || oz > domHi[2] || ox + r.dim - 1 < domLo[0] || oy + r.dim - 1 < domLo[1] || oz + r.dim - 1 < domLo[2]) return;
    if (r.level > 0) {
      add(ox, oy, oz, ox + r.dim - 1, oy + r.dim - 1, oz + r.dim - 1, r.value);
      return;
    }
    const { values } = leafVoxels(image, r.leaf);
    for (let n0 = 0; n0 < 512; n0++) {
      const x = ox + (n0 >> 6), y = oy + ((n0 >> 3) & 7), z = oz + (n0 & 7);
      add(x, y, z, x, y, z, values[n0]!);
    }
  });

  // The background (space under no root tile) and the zeros a tap outside the bbox reads.
  const roots = new Set(rootTiles.map((o) => `${o[0]},${o[1]},${o[2]}`));
  const covered = (a: Vec3, b: Vec3): boolean => {
    for (let x = Math.floor(a[0] / 4096); x <= Math.floor(b[0] / 4096); x++)
      for (let y = Math.floor(a[1] / 4096); y <= Math.floor(b[1] / 4096); y++)
        for (let z = Math.floor(a[2] / 4096); z <= Math.floor(b[2] / 4096); z++) if (!roots.has(`${x * 4096},${y * 4096},${z * 4096}`)) return false;
    return true;
  };
  for (let cz = 0; cz < dims[2]; cz++) {
    for (let cy = 0; cy < dims[1]; cy++) {
      for (let cx = 0; cx < dims[0]; cx++) {
        const i = cx + dims[0] * (cy + dims[1] * cz);
        const c = [cx, cy, cz];
        const tLo = [0, 1, 2].map((a) => origin[a]! + B * c[a]!) as Vec3;
        const tHi = [0, 1, 2].map((a) => tLo[a]! + B) as Vec3;
        const inLo = [0, 1, 2].map((a) => Math.max(tLo[a]!, domLo[a]!)) as Vec3;
        const inHi = [0, 1, 2].map((a) => Math.min(tHi[a]!, domHi[a]!)) as Vec3;
        const meets = !(empty && zero) && inLo.every((v, a) => v <= inHi[a]!);
        if (meets && !covered(inLo, inHi)) {
          hi[i] = Math.max(hi[i]!, background);
          if (lo) lo[i] = Math.min(lo[i]!, background);
        }
        if (zero) {
          const leaves = empty || tLo.some((v, a) => v < min[a]!) || tHi.some((v, a) => v > max[a]!);
          if (align === "pathtracer") {
            // Its rule: the bound starts at 0, and the minimum is 0 when the taps start below 0 or end at n - 1 or later.
            hi[i] = Math.max(hi[i]!, 0);
            if (lo && (leaves || tLo.some((v) => v < 0) || tHi.some((v, a) => v + 1 > n[a]! - 1))) lo[i] = Math.min(lo[i]!, 0);
          } else if (leaves) {
            hi[i] = Math.max(hi[i]!, 0);
            if (lo) lo[i] = Math.min(lo[i]!, 0);
          }
        }
      }
    }
  }

  const data = Float32Array.from(hi);
  const out: VDBBrickMajorant = { brickLog2: log2, dims, data, origin };
  if (lo) {
    const m = new Float32Array(cells);
    for (let i = 0; i < cells; i++) m[i] = Math.min(lo[i]!, data[i]!);
    out.minima = m;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The walk (twin of the `majorant` WGSL profile)

/** One segment of the majorant walk: every sample on `[t0, t1)` lies in `[min, max]`. */
export interface MajorantSegment {
  t0: number;
  t1: number;
  /** Upper bound of the sampled field on the segment. */
  max: number;
  /** Lower bound: the brick minimum with minima, else 0 (a valid minorant of a non-negative field). */
  min: number;
  /** Accessor queries the segment cost (`vdbGetDim` / `vdbNodeMax`; 1 brick load, 2 with minima). */
  lookups: number;
}

/** Options of {@link majorantWalk}. */
export interface MajorantWalkOptions {
  /** The module's `majorant` option. `'brick'` with no brick majorant walks as `'global'`. Default `'brick'`. */
  mode?: "brick" | "global" | "nodeMax";
  /** Ray interval before clipping. Default `[0, 3.4e38]`. */
  tMin?: number;
  tMax?: number;
  /** The record's `valueMax` (default {@link defaultValueMax}: the pool's default). */
  valueMax?: number;
  /** The module's sampler (the support box of `global` and `nodeMax`). Default `'trilinear'`. */
  sampler?: "trilinear" | "nearest";
  /** The module's `outside`; `'zero'` raises a bound below 0 to 0 (`global`, `nodeMax`). Default `'background'`. */
  outside?: "background" | "zero";
  /** The module's `minima` option (read the brick minima when it has them). Default false. */
  minima?: boolean;
  /** Safety cap on segments. Default 1 << 16. */
  maxSegments?: number;
}

/** f32 1e-20: below it a direction component counts as 0 (the pathtracer's DDA guard). */
const PARALLEL = f(1e-20);
const FAR = f(1e30);

/**
 * Twin of the WGSL `vdb_maj_clip`: the slab test of `[lo, hi]` with a guard for axis-parallel rays (a
 * component below 1e-20 in magnitude: the ray misses unless `lo <= o < hi` on that axis; WGSL may
 * assume no infinities, CLAUDE.md Traps). Returns `[a, b]`; empty when `!(a < b)`.
 */
function clip(lo: Vec3, hi: Vec3, o: Vec3, d: Vec3, t0: number, t1: number): [number, number] {
  let a = t0;
  let b = t1;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(d[k]!) < PARALLEL) {
      if (o[k]! < lo[k]! || o[k]! >= hi[k]!) return [t0, t0];
    } else {
      const p = f(f(lo[k]! - o[k]!) / d[k]!);
      const q = f(f(hi[k]! - o[k]!) / d[k]!);
      a = Math.max(a, Math.min(p, q));
      b = Math.min(b, Math.max(p, q));
    }
  }
  return [a, b];
}

const f3 = (v: readonly number[]): Vec3 => [f(v[0]!), f(v[1]!), f(v[2]!)];

/**
 * Twin of `vdbMajorantInit(g, o, d, tMin, tMax)` + `vdbMajorantNext` until it ends: the segments the
 * GPU walk returns for an index-space ray (`d` unnormalized; `t` in the caller's units), in order,
 * contiguous (each `t0` is the previous `t1`), none of zero length.
 *
 * - `'brick'` with bounds: the DDA over the brick cells (the pathtracer's `ptDdaInit` / `ptDdaNext`
 *   in the module's cell space `(p - origin) / B`), clipped to the cell grid's box.
 * - `'global'` (or `'brick'` without bounds): one segment over the sampler's support box
 *   (`[min - 1, max + 1]` trilinear, `[min - 0.5, max + 0.5]` nearest) with bound `valueMax`.
 * - `'nodeMax'` (P1.4): the vendored HDDA at leaf size 8, widening over tiles and empty space, over the
 *   support box; a cell's bound is {@link cellBoundTrilinear} (the 2x2x2 block of `dim`-cells from the
 *   cell's corner: the "8-leaf support" of a leaf cell). Stats bound ACTIVE values, so this assumes every
 *   inactive leaf voxel is <= its leaf's max (fog). Without stats the bound is `valueMax`.
 */
export function majorantWalk(
  gridWords: Uint32Array,
  majorant: VDBBrickMajorant | null,
  origin: Vec3,
  direction: Vec3,
  options: MajorantWalkOptions = {},
): MajorantSegment[] {
  const mode = options.mode ?? "brick";
  const zero = options.outside === "zero";
  const o = f3(origin);
  const d = f3(direction);
  const tMin = f(options.tMin ?? 0);
  const tMax = f(options.tMax ?? 3.4e38);
  const maxSegments = options.maxSegments ?? 1 << 16;
  const vmaxRecord = f(options.valueMax ?? defaultValueMax(gridWords));
  const vmax = zero ? Math.max(vmaxRecord, 0) : vmaxRecord;
  const out: MajorantSegment[] = [];

  if (mode === "brick" && majorant) {
    const B = 1 << majorant.brickLog2;
    const s = f(1 / B);
    const { min } = indexBBox(gridWords);
    const org = f3(majorant.origin ?? [min[0] - 1, min[1] - 1, min[2] - 1]);
    const n = majorant.dims;
    const u: Vec3 = [f(f(o[0] - org[0]) * s), f(f(o[1] - org[1]) * s), f(f(o[2] - org[2]) * s)];
    const du: Vec3 = [f(d[0] * s), f(d[1] * s), f(d[2] * s)];
    const [tEnter, tExit] = clip([0, 0, 0], f3(n), u, du, tMin, tMax);
    if (!(tEnter < tExit)) return out;
    const cell = [0, 1, 2].map((k) => Math.min(n[k]! - 1, Math.max(0, Math.floor(f(u[k]! + f(du[k]! * tEnter)))))) as Vec3;
    const step = du.map((v) => (v > 0 ? 1 : -1));
    const tn = [0, 1, 2].map((k) => (Math.abs(du[k]!) < PARALLEL ? FAR : f(f(f(cell[k]! + (du[k]! > 0 ? 1 : 0)) - u[k]!) / du[k]!)));
    const dt = du.map((v) => (Math.abs(v) < PARALLEL ? FAR : Math.abs(f(1 / v))));
    const minima = options.minima === true && majorant.minima !== undefined;
    let t = tEnter;
    let te = tExit;
    while (t < te && out.length < maxSegments) {
      let k = 0;
      if (tn[1]! < tn[k]!) k = 1;
      if (tn[2]! < tn[k]!) k = 2;
      const t1 = Math.min(tn[k]!, te);
      const i = cell[0] + n[0] * (cell[1] + n[1] * cell[2]);
      cell[k] = cell[k]! + step[k]!;
      tn[k] = f(tn[k]! + dt[k]!);
      if (cell[k]! < 0 || cell[k]! >= n[k]!) te = t1;
      if (t1 > t) {
        out.push({ t0: t, t1, max: majorant.data[i]!, min: minima ? majorant.minima![i]! : 0, lookups: minima ? 2 : 1 });
        t = t1;
      }
    }
    return out;
  }

  const { min, max } = indexBBox(gridWords);
  const pad = options.sampler === "nearest" ? 0.5 : 1;
  const [t0, t1] = clip(f3([min[0] - pad, min[1] - pad, min[2] - pad]), f3([max[0] + pad, max[1] + pad, max[2] + pad]), o, d, tMin, tMax);
  if (!(t0 < t1)) return out;
  if (mode !== "nodeMax") {
    out.push({ t0, t1, max: vmax, min: 0, lookups: 0 });
    return out;
  }

  const bound = (ijk: Vec3, dim: number): number => {
    const b = cellBoundTrilinear(gridWords, ijk, dim, { valueMax: vmaxRecord });
    return zero ? Math.max(b, 0) : b;
  };
  const start: Vec3 = [f(o[0] + f(d[0] * t0)), f(o[1] + f(d[1] * t0)), f(o[2] + f(d[2] * t0))];
  const h = hddaInit(o, t0, d, t1, Math.max(getDim(gridWords, [Math.floor(start[0]), Math.floor(start[1]), Math.floor(start[2])]), 8));
  let t = t0;
  let te = t1;
  let lookups = 1;
  while (t < te && out.length < maxSegments) {
    let updates = 0;
    for (;;) {
      const want = Math.max(getDim(gridWords, h.voxel), 8);
      lookups++;
      if (want === h.dim || updates >= 2) break;
      hddaUpdate(h, o, d, want);
      updates++;
    }
    const end = Math.min(Math.min(Math.min(h.next[0], h.next[1]), Math.min(h.next[2], h.tmax)), te);
    const voxel: Vec3 = [h.voxel[0], h.voxel[1], h.voxel[2]];
    const dim = h.dim;
    const more = hddaStep(h);
    const e = more ? end : te;
    if (!more) te = Math.max(t, e);
    if (e > t) {
      out.push({ t0: t, t1: e, max: bound(voxel, dim), min: 0, lookups: lookups + 8 });
      lookups = 0;
      t = e;
    }
  }
  return out;
}
