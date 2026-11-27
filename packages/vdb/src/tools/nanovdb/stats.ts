/**
 * stats.ts — per-node active-value statistics, computed the way NanoVDB's
 * `tools/GridStats.h` does: min/max are the extrema of *active* voxel values in
 * a node's subtree, `ave` is their arithmetic mean, and `stddev` is the
 * *population* standard deviation (divide by N, not N-1). A parent node's stats
 * are the exact aggregate of its children's plus its own active tiles, each
 * tile counted with GridStats' weight (see {@link StatsAccumulator.addTile}).
 *
 * Two accumulations (K7.H1, DIVERGES V73), chosen by
 * {@link vdbSetStatsAccumulation} when an accumulator is made:
 * - `'welford'` (default since K7.H1): GridStats' own f64 update, ported
 *   operation for operation (`Stats<T, 0>::add(val)`, `add(val, n)`,
 *   `add(other)`). Fed in GridStats' order (leaf voxels by offset, an internal
 *   node's active tiles then its children by slot, the root table in order) it
 *   reproduces native ave/stddev bit for bit, ±∞ leaves included (NaN there).
 * - `'sums'` (K1–K7.H): f64 count, sum and sum of squares; ave/stddev within a
 *   few f32 ulp of native, ±∞ where Welford gives NaN.
 * Either way ave/stddev are narrowed to f32 only when stored.
 */

const INT_MIN = -0x80000000;
const INT_MAX = 0x7fffffff;

/** How {@link StatsAccumulator} accumulates ave/stddev (K7.H1 A/B; see the file header). */
export type VDBStatsAccumulation = "welford" | "sums";

let accumulation: VDBStatsAccumulation = "welford";

/**
 * Sets how accumulators made from now on compute ave/stddev, and returns the previous setting.
 * Internal (tests and A/B scripts); builds made under one setting are consistent.
 */
export function vdbSetStatsAccumulation(mode: VDBStatsAccumulation): VDBStatsAccumulation {
  if (mode !== "welford" && mode !== "sums") throw new RangeError(`stats accumulation must be 'welford' or 'sums', got ${String(mode)}`);
  const previous = accumulation;
  accumulation = mode;
  return previous;
}

/** Running active-value statistics plus the tight index-space bbox of the
 *  active voxels they cover. */
export class StatsAccumulator {
  /** Samples (with tiles: GridStats weight, see {@link addTile}). */
  count = 0;
  readonly mode: VDBStatsAccumulation = accumulation;
  // 'sums'
  private sum = 0;
  private sumSq = 0;
  // 'welford': GridStats' mAvg, mAux.
  private avg = 0;
  private aux = 0;
  min = Number.POSITIVE_INFINITY;
  max = Number.NEGATIVE_INFINITY;
  // Tight bbox of active voxels (inclusive). Empty when count === 0.
  bboxMin: [number, number, number] = [INT_MAX, INT_MAX, INT_MAX];
  bboxMax: [number, number, number] = [INT_MIN, INT_MIN, INT_MIN];

  /** Adds one active voxel at index coord (x,y,z) with the given value. */
  addVoxel(value: number, x: number, y: number, z: number): void {
    this.count += 1;
    if (this.mode === "welford") {
      // Stats<T, 0>::add(val)
      const delta = value - this.avg;
      this.avg += delta / this.count;
      this.aux += delta * (value - this.avg);
    } else {
      this.sum += value;
      this.sumSq += value * value;
    }
    if (value < this.min) this.min = value;
    if (value > this.max) this.max = value;
    if (x < this.bboxMin[0]) this.bboxMin[0] = x;
    if (y < this.bboxMin[1]) this.bboxMin[1] = y;
    if (z < this.bboxMin[2]) this.bboxMin[2] = z;
    if (x > this.bboxMax[0]) this.bboxMax[0] = x;
    if (y > this.bboxMax[1]) this.bboxMax[1] = y;
    if (z > this.bboxMax[2]) this.bboxMax[2] = z;
  }

  /**
   * Adds an active tile: `value` counted `weight` times, with the tile's
   * inclusive index bbox. NanoVDB GridStats passes the child node type's slot
   * count as the weight, so once tiles are involved `count` is a stats weight,
   * not a voxel count.
   */
  addTile(
    value: number,
    weight: number,
    bboxMin: readonly [number, number, number],
    bboxMax: readonly [number, number, number],
  ): void {
    if (this.mode === "welford") {
      // Stats<T, 0>::add(val, n), left to right as C++ evaluates it.
      const denom = 1.0 / (this.count + weight);
      const delta = value - this.avg;
      this.avg += denom * delta * weight;
      this.aux += denom * delta * delta * this.count * weight;
    } else {
      this.sum += value * weight;
      this.sumSq += value * value * weight;
    }
    this.count += weight;
    if (value < this.min) this.min = value;
    if (value > this.max) this.max = value;
    for (let a = 0; a < 3; a++) {
      if (bboxMin[a]! < this.bboxMin[a]!) this.bboxMin[a] = bboxMin[a]!;
      if (bboxMax[a]! > this.bboxMax[a]!) this.bboxMax[a] = bboxMax[a]!;
    }
  }

  /** Merges a child accumulator's totals into this one (both must use the same accumulation). */
  merge(child: StatsAccumulator): void {
    if (child.count === 0) return;
    if (child.mode !== this.mode) throw new RangeError("StatsAccumulator.merge: the accumulators use different accumulations");
    if (this.mode === "welford") {
      // Stats<T, 0>::add(const Stats& other): mAux += other.mAux + (…).
      const denom = 1.0 / (this.count + child.count);
      const delta = child.avg - this.avg;
      this.avg += denom * delta * child.count;
      this.aux += child.aux + denom * delta * delta * this.count * child.count;
    } else {
      this.sum += child.sum;
      this.sumSq += child.sumSq;
    }
    this.count += child.count;
    if (child.min < this.min) this.min = child.min;
    if (child.max > this.max) this.max = child.max;
    for (let a = 0; a < 3; a++) {
      if (child.bboxMin[a]! < this.bboxMin[a]!) this.bboxMin[a] = child.bboxMin[a]!;
      if (child.bboxMax[a]! > this.bboxMax[a]!) this.bboxMax[a] = child.bboxMax[a]!;
    }
  }

  get average(): number {
    if (this.mode === "welford") return this.avg;
    return this.count > 0 ? this.sum / this.count : 0;
  }

  /** Population standard deviation (GridStats convention). */
  get stdDev(): number {
    if (this.mode === "welford") return this.count < 2 ? 0 : Math.sqrt(this.aux / this.count);
    if (this.count === 0) return 0;
    const mean = this.sum / this.count;
    const variance = this.sumSq / this.count - mean * mean;
    return Math.sqrt(Math.max(0, variance));
  }

  get isEmpty(): boolean {
    return this.count === 0;
  }

  /**
   * `min` as GridStats' `Extrema` stores it: Extrema starts at the value type's largest finite value
   * (`finiteMax`: FLT_MAX for Float, 65504 for Half), so a node whose values are all +∞ keeps that
   * start value as its minimum (K7.H1). 0 when empty.
   */
  storedMin(finiteMax: number): number {
    if (this.isEmpty) return 0;
    return this.min === Number.POSITIVE_INFINITY ? finiteMax : this.min;
  }

  /** `max` as GridStats stores it: −`finiteMax` when every value is −∞ (see {@link storedMin}). 0 when empty. */
  storedMax(finiteMax: number): number {
    if (this.isEmpty) return 0;
    return this.max === Number.NEGATIVE_INFINITY ? -finiteMax : this.max;
  }
}

/** FLT_MAX, `std::numeric_limits<float>::max()`: GridStats' starting minimum for f32 values. */
export const STATS_FLOAT_MAX = 3.4028234663852886e38;
/** 65504, the largest finite binary16: GridStats' starting minimum for Half values (raw 0x7BFF). */
export const STATS_HALF_MAX = 65504;

const fround = Math.fround;

/** `Vec3<float>::lengthSqr()` evaluated in f32: `x*x + y*y + z*z`, left to right. */
export function lengthSqrF32(x: number, y: number, z: number): number {
  return fround(fround(fround(x * x) + fround(y * y)) + fround(z * z));
}

/**
 * Vector statistics, NanoVDB `tools/GridStats.h` `Stats<Vec3f, 1>` (rank 1):
 * every value is ranked by its f32 `lengthSqr()`. `min`/`max` (inherited) are
 * the extreme squared lengths and {@link minVec}/{@link maxVec} the vectors
 * that hold them (first one wins on ties, as `Pair::operator<` is strict);
 * `average` and `stdDev` are the mean and population standard deviation of the
 * squared lengths, not of the components.
 */
export class Vec3StatsAccumulator extends StatsAccumulator {
  minVec: [number, number, number] = [0, 0, 0];
  maxVec: [number, number, number] = [0, 0, 0];

  /** Adds one active voxel with vector value (vx, vy, vz) at index coord (x, y, z). */
  addVoxelVec(vx: number, vy: number, vz: number, x: number, y: number, z: number): void {
    const s = lengthSqrF32(vx, vy, vz);
    if (s < this.min) this.minVec = [vx, vy, vz];
    if (s > this.max) this.maxVec = [vx, vy, vz];
    this.addVoxel(s, x, y, z);
  }

  /** Adds an active vector tile with GridStats weight `weight` (see {@link StatsAccumulator.addTile}). */
  addTileVec(
    value: readonly [number, number, number],
    weight: number,
    bboxMin: readonly [number, number, number],
    bboxMax: readonly [number, number, number],
  ): void {
    const s = lengthSqrF32(value[0], value[1], value[2]);
    if (s < this.min) this.minVec = [value[0], value[1], value[2]];
    if (s > this.max) this.maxVec = [value[0], value[1], value[2]];
    this.addTile(s, weight, bboxMin, bboxMax);
  }

  override merge(child: StatsAccumulator): void {
    if (child.count === 0) return;
    if (child instanceof Vec3StatsAccumulator) {
      if (child.min < this.min) this.minVec = [...child.minVec];
      if (child.max > this.max) this.maxVec = [...child.maxVec];
    }
    super.merge(child);
  }
}
