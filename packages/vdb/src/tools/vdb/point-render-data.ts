/**
 * The render-ready point list of a point data grid (K3.7, G14a): world positions,
 * optional `Cd` colours and other attributes, a subsample stride above
 * `maxPoints`, an optional prefix-subsample order, and the mean point spacing
 * that sizes the sprites. The input `VDBPoints` (`./three`) draws.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/openvdb/utils/points.js` `createPointData`: the stride rule, the count,
 * the spacing formula). Copyright (c) mjurczyk/openvdb contributors, MIT (see
 * THIRD_PARTY_NOTICES.md). Changes: built on our parsed `VdbPointDataGrid.points()`
 * (K4.2) in one pass instead of a sliced generator; `Cd` of any decoded
 * 3-component real attribute; other attributes, a prefix order, progress and
 * bounds (G14a) (DIVERGES.md).
 *
 * Worker use: everything here is plain data (typed arrays and numbers), so a
 * worker can `parseVdb` and call {@link vdbPointRenderData}, then post the result
 * with {@link vdbPointRenderDataTransfer} as the transfer list.
 */
import { VDBToolsError } from "../errors.js";
import type { VdbPointAttributeArray, VdbPointDataGrid } from "./types.js";

/** One extra attribute carried by {@link VDBPointRenderData}: values in the kept points' order. */
export interface VDBPointRenderAttribute {
  /** `count * components` values, the decoder's element type (real attributes are `Float32Array`, `double` `Float64Array`). */
  values: VdbPointAttributeArray;
  /** Values per point (vector components times the attribute's stride). */
  components: number;
}

/** What `VDBPoints` draws: world positions (xyz per point), optional linear RGB, the mean spacing. */
export interface VDBPointRenderData {
  /** Points in the arrays. */
  count: number;
  /** Points in the grid before the stride. */
  totalCount: number;
  /** Every `stride`-th point (in the grid's point order) is kept; 1 below `maxPoints`. */
  stride: number;
  /** Mean distance between the kept points in world units: `voxelSize / cbrt(points per active voxel) * cbrt(stride)`. */
  spacing: number;
  /** World positions, `count * 3`. */
  positions: Float32Array;
  /** `Cd` per point (`count * 3`), or null when the grid has no 3-component real `Cd`. */
  colors: Float32Array | null;
  /** The attributes asked for with `attributes` that the grid has, in the same order as `positions`. */
  attributes: Record<string, VDBPointRenderAttribute>;
  /**
   * `order: 'prefix'` only: `prefix[k]` (k = 0..10) is how many points the first k tenths of the arrays hold.
   * `prefix[1]` is every 10th kept point, `prefix[5]` every 2nd, `prefix[10]` = `count`. Null for `order: 'grid'`.
   */
  prefix: number[] | null;
  /** Box of the kept positions (world units), or null without points. */
  bounds: { min: [number, number, number]; max: [number, number, number] } | null;
}

/** Progress of {@link vdbPointRenderData}: the decode (per leaf of the grid), then the copy into render order. */
export interface VDBPointRenderDataProgress {
  /** `'decode'`: the grid's attribute arrays (skipped when `grid.points()` already ran); `'copy'`: the render arrays. */
  stage: "decode" | "copy";
  /** Points done so far in this stage. */
  done: number;
  /** Points in this stage. */
  total: number;
}

/** Options of {@link vdbPointRenderData}. */
export interface VDBPointRenderDataOptions {
  /** Above this many points every `ceil(total / maxPoints)`-th point is kept (his default 4,000,000). */
  maxPoints?: number;
  /** Colour attribute name. Default `'Cd'`. */
  color?: string;
  /** Other attributes to carry, by name (e.g. `['v']`). Names the grid does not have (or could not decode) are left out. */
  attributes?: readonly string[];
  /**
   * `'grid'` (default): the grid's point order. `'prefix'`: residue classes of the kept index mod 10 in the order
   * 0, 2, 4, 6, 8, 1, 3, 5, 7, 9, so every prefix of k tenths (`prefix[k]` points) is an even subsample: drawing
   * fewer points is an instance count, not new buffers.
   */
  order?: "grid" | "prefix";
  /** Called during the decode (per leaf) and the copy (every 2^20 points), and once at the end of each stage. */
  onProgress?: (progress: VDBPointRenderDataProgress) => void;
}

/** His `defaultMaxPoints`. */
export const VDB_POINTS_DEFAULT_MAX = 4_000_000;

/** Residue classes mod 10 in prefix order: 10 % (class 0), then to 50 % (the even classes), then the odd ones. */
const PREFIX_CLASSES = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9] as const;
const classSize = (n: number, c: number): number => (c < n ? Math.floor((n - 1 - c) / 10) + 1 : 0);

/**
 * Builds the {@link VDBPointRenderData} of a parsed point data grid
 * (`parseVdb(...).grids` entry with `valueType: 'ptdataidx32'`). Decodes the
 * grid's points (`grid.points()`, cached by the grid).
 *
 * @param grid A point data grid from `parseVdb`.
 * @param options Point limit, attributes, order and progress ({@link VDBPointRenderDataOptions}).
 */
export function vdbPointRenderData(grid: VdbPointDataGrid, options: VDBPointRenderDataOptions = {}): VDBPointRenderData {
  if (grid?.valueType !== "ptdataidx32" || typeof grid.points !== "function") {
    throw new VDBToolsError("VDB_UNSUPPORTED_GRID_TYPE", "vdbPointRenderData: grid must be a parsed point data grid (valueType 'ptdataidx32').");
  }
  const maxPoints = options.maxPoints ?? VDB_POINTS_DEFAULT_MAX;
  if (!(maxPoints >= 1)) throw new VDBToolsError("VDB_BAD_OPTION", `vdbPointRenderData: maxPoints must be >= 1, got ${maxPoints}.`);
  const order = options.order ?? "grid";
  if (order !== "grid" && order !== "prefix") throw new VDBToolsError("VDB_BAD_OPTION", `vdbPointRenderData: order must be 'grid' or 'prefix', got ${JSON.stringify(order)}.`);
  const onProgress = options.onProgress;
  const list = grid.points(onProgress ? { onProgress: (done, total) => onProgress({ stage: "decode", done, total }) } : undefined);
  const totalCount = list.count;
  const stride = Math.max(1, Math.ceil(totalCount / maxPoints));
  const count = totalCount > 0 ? Math.floor((totalCount - 1) / stride) + 1 : 0;
  const positions = new Float32Array(count * 3);
  const name = options.color ?? "Cd";
  const cd = list.attributes[name];
  const shape = list.attributeShapes[name];
  const hasColors = cd !== undefined && shape?.components === 3 && shape.stride === 1 && (cd instanceof Float32Array || cd instanceof Float64Array);
  const colors = hasColors ? new Float32Array(count * 3) : null;
  const extra: { src: VdbPointAttributeArray; dst: VdbPointAttributeArray; k: number }[] = [];
  const attributes: Record<string, VDBPointRenderAttribute> = {};
  for (const a of options.attributes ?? []) {
    const src = list.attributes[a];
    const s = list.attributeShapes[a];
    if (!src || !s || attributes[a]) continue;
    const k = s.components * s.stride;
    const dst = new (src.constructor as new (n: number) => VdbPointAttributeArray)(count * k);
    attributes[a] = { values: dst, components: k };
    extra.push({ src, dst, k });
  }
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  const src = list.positions;
  const copy = (i: number, j: number): void => {
    for (let c = 0; c < 3; c++) {
      const p = src[i * 3 + c]!;
      positions[j * 3 + c] = p;
      if (p < min[c]!) min[c] = p;
      if (p > max[c]!) max[c] = p;
    }
    if (colors) {
      const c = cd as Float32Array | Float64Array;
      colors[j * 3] = c[i * 3]!;
      colors[j * 3 + 1] = c[i * 3 + 1]!;
      colors[j * 3 + 2] = c[i * 3 + 2]!;
    }
    for (const e of extra) for (let c = 0; c < e.k; c++) (e.dst as Float32Array)[j * e.k + c] = (e.src as Float32Array)[i * e.k + c]!;
  };
  const tick = (j: number): void => {
    if (onProgress && (j & 0xfffff) === 0 && j > 0) onProgress({ stage: "copy", done: j, total: count });
  };
  let prefix: number[] | null = null;
  if (order === "prefix") {
    // Kept index q = 0..count-1 (grid point q * stride), output position by residue class of q.
    prefix = [0];
    let j = 0;
    for (const c of PREFIX_CLASSES) {
      for (let q = c; q < count; q += 10, j++) {
        copy(q * stride, j);
        tick(j);
      }
      prefix.push(prefix[prefix.length - 1]! + classSize(count, c));
    }
  } else {
    for (let i = 0, j = 0; i < totalCount; i += stride, j++) {
      copy(i, j);
      tick(j);
    }
  }
  onProgress?.({ stage: "copy", done: count, total: count });
  const [sx, sy, sz] = grid.transform.voxelSize;
  const voxelSize = Math.cbrt(Math.abs(sx * sy * sz));
  const activeVoxels = Number(grid.activeVoxelCount);
  const pointsPerVoxel = activeVoxels > 0 ? totalCount / activeVoxels : 1;
  const spacing = (voxelSize / Math.cbrt(pointsPerVoxel)) * Math.cbrt(stride);
  return { count, totalCount, stride, spacing, positions, colors, attributes, prefix, bounds: count > 0 ? { min, max } : null };
}

/**
 * The buffers of a {@link VDBPointRenderData} to pass as a `postMessage` transfer list (each once), so a worker hands
 * the arrays to the page without a copy: `self.postMessage(data, vdbPointRenderDataTransfer(data))`.
 *
 * @param data Render data from {@link vdbPointRenderData}.
 */
export function vdbPointRenderDataTransfer(data: VDBPointRenderData): ArrayBuffer[] {
  const out = new Set<ArrayBuffer>();
  const add = (a: ArrayBufferView | null): void => {
    if (a && a.buffer instanceof ArrayBuffer) out.add(a.buffer);
  };
  add(data.positions);
  add(data.colors);
  for (const a of Object.values(data.attributes)) add(a.values);
  return [...out];
}
