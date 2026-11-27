/**
 * K7.H1: node statistics accumulate as NanoVDB `tools/GridStats.h` does (f64 Welford, `Stats<T, 0>`),
 * the default since K7.H1; `'sums'` (the K1–K7.H f64 sum / sum of squares) stays for the A/B
 * (`scripts/ab-stats-accumulation.mjs`). Native bit parity is checked in `half-native.test.ts` (Half
 * sidecars, every node) and `native-order.test.ts` (Float images, byte for byte); these are the unit cases.
 */
import { afterEach, describe, expect, it } from "vitest";

import { VDB_FLOAT_LEAF_CODEC, VDB_HALF_LEAF_CODEC, vdbBuildFromLeavesDetailed } from "../../src/tools/index.js";
import { nodeStats, treeNodes } from "../../src/tools/cpu/march.js";
import { STATS_FLOAT_MAX, StatsAccumulator, vdbSetStatsAccumulation } from "../../src/tools/nanovdb/stats.js";
import { buildTreeFromLeaves } from "../../src/tools/nanovdb/tree.js";

afterEach(() => {
  vdbSetStatsAccumulation("welford");
});

/** GridStats `Stats<T, 0>` as written in C++ (the oracle: straight from the header, no shared code). */
class CppStats {
  n = 0;
  avg = 0;
  aux = 0;
  add(v: number): void {
    this.n += 1;
    const delta = v - this.avg;
    this.avg += delta / this.n;
    this.aux += delta * (v - this.avg);
  }
  addN(v: number, n: number): void {
    const denom = 1.0 / (this.n + n);
    const delta = v - this.avg;
    this.avg += denom * delta * n;
    this.aux += denom * delta * delta * this.n * n;
    this.n += n;
  }
  addStats(o: CppStats): void {
    if (o.n === 0) return;
    const denom = 1.0 / (this.n + o.n);
    const delta = o.avg - this.avg;
    this.avg += denom * delta * o.n;
    this.aux += o.aux + denom * delta * delta * this.n * o.n;
    this.n += o.n;
  }
  get std(): number {
    return this.n < 2 ? 0 : Math.sqrt(this.aux / this.n);
  }
}

const Z: [number, number, number] = [0, 0, 0];

describe("StatsAccumulator ('welford', GridStats' update)", () => {
  it("is the default", () => {
    expect(new StatsAccumulator().mode).toBe("welford");
  });

  it("voxels, weighted tiles and merges follow GridStats' f64 update operation for operation", () => {
    let seed = 3;
    const rnd = (): number => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 7 - 2;
    const a = new StatsAccumulator();
    const b = new StatsAccumulator();
    const ca = new CppStats();
    const cb = new CppStats();
    for (let i = 0; i < 500; i++) {
      const v = Math.fround(rnd());
      a.addVoxel(v, 0, 0, 0);
      ca.add(v);
      const w = Math.fround(rnd());
      if (i % 50 === 0) {
        b.addTile(w, 512, Z, Z);
        cb.addN(w, 512);
      } else {
        b.addVoxel(w, 0, 0, 0);
        cb.add(w);
      }
    }
    a.merge(b);
    ca.addStats(cb);
    expect(Object.is(a.average, ca.avg)).toBe(true);
    expect(Object.is(a.stdDev, ca.std)).toBe(true);
  });

  it("population mean and deviation; one value has deviation 0", () => {
    const s = new StatsAccumulator();
    for (const v of [1, 2, 3, 4]) s.addVoxel(v, 0, 0, 0);
    expect(s.average).toBe(2.5);
    expect(s.stdDev).toBe(Math.sqrt(1.25));
    const one = new StatsAccumulator();
    one.addVoxel(7, 0, 0, 0);
    expect([one.average, one.stdDev]).toEqual([7, 0]);
  });

  it("a node holding +-inf: NaN deviation (and NaN mean once a finite value follows), as native", () => {
    const s = new StatsAccumulator();
    s.addVoxel(1, 0, 0, 0);
    s.addVoxel(Infinity, 0, 0, 0);
    expect(s.average).toBe(Infinity);
    expect(s.stdDev).toBeNaN();
    const t = new StatsAccumulator();
    t.addVoxel(Infinity, 0, 0, 0);
    t.addVoxel(1, 0, 0, 0);
    expect(t.average).toBeNaN();
    // 'sums' gave +-inf here (DIVERGES V73 before K7.H1).
    vdbSetStatsAccumulation("sums");
    const u = new StatsAccumulator();
    u.addVoxel(1, 0, 0, 0);
    u.addVoxel(Infinity, 0, 0, 0);
    expect(u.average).toBe(Infinity);
  });

  it("refuses to merge accumulators of different modes", () => {
    const w = new StatsAccumulator();
    vdbSetStatsAccumulation("sums");
    const s = new StatsAccumulator();
    s.addVoxel(1, 0, 0, 0);
    expect(() => w.merge(s)).toThrow(RangeError);
    expect(() => vdbSetStatsAccumulation("kahan" as never)).toThrow(RangeError);
  });
});

describe("stored extrema: Extrema starts at the type's largest finite value", () => {
  const leaf = (value: number) => ({
    origin: [0, 0, 0] as [number, number, number],
    values: new Float32Array(512).fill(value),
    valueMask: new Uint32Array(16).fill(0xffffffff),
  });
  it("Float: a node of +inf only stores min FLT_MAX, of -inf only max -FLT_MAX (native's start values)", () => {
    const pos = vdbBuildFromLeavesDetailed([leaf(Infinity)], VDB_FLOAT_LEAF_CODEC, { background: 0, gridName: "p", collapse: false }).image;
    const neg = vdbBuildFromLeavesDetailed([leaf(-Infinity)], VDB_FLOAT_LEAF_CODEC, { background: 0, gridName: "n", collapse: false }).image;
    for (const level of [0, 1, 2, 3] as const) {
      const p = nodeStats(pos, level, treeNodes(pos).find((n) => n.level === level)!.address);
      const n = nodeStats(neg, level, treeNodes(neg).find((x) => x.level === level)!.address);
      expect([p.min, p.max]).toEqual([STATS_FLOAT_MAX, Infinity]);
      expect([n.min, n.max]).toEqual([-Infinity, -STATS_FLOAT_MAX]);
    }
  });
  it("Half: a leaf whose inputs overflow to +inf stores min 0x7BFF (65504) and max 0x7C00", () => {
    const img = vdbBuildFromLeavesDetailed([leaf(1e6)], VDB_HALF_LEAF_CODEC, { background: 0, gridName: "h", collapse: false }).image;
    const l = treeNodes(img).find((n) => n.level === 0)!;
    const dv = new DataView(img.buffer, img.byteOffset, img.byteLength);
    expect([dv.getUint16(l.address + 80, true), dv.getUint16(l.address + 82, true)]).toEqual([0x7bff, 0x7c00]);
  });
});

describe("tree stats follow GridStats' order (f64, before the f32 store hides it)", () => {
  it("leaf voxels by offset, lowers' leaves and uppers' lowers by slot, the root by signed origin", () => {
    let seed = 11;
    const rnd = (): number => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32;
    // Leaves in two uppers (one at negative x), several lowers each, several leaves per lower.
    const leaves: { origin: [number, number, number]; values: Float32Array; valueMask: Uint32Array }[] = [];
    for (const ux of [-4096, 0])
      for (let l = 0; l < 5; l++)
        for (let k = 0; k < 6; k++) {
          const origin: [number, number, number] = [ux + l * 256 + k * 8, ((l * 3) % 4) * 128 + k * 16, (k % 3) * 8];
          const values = Float32Array.from({ length: 512 }, () => Math.fround(rnd() * 10 - 3));
          const valueMask = Uint32Array.from({ length: 16 }, () => (rnd() * 2 ** 32) >>> 0);
          leaves.push({ origin, values, valueMask });
        }
    // Feed them shuffled: the builder must order them itself.
    const shuffled = [...leaves].sort(() => rnd() - 0.5);
    const tree = buildTreeFromLeaves(shuffled);

    const leafStats = (l: (typeof leaves)[number]): CppStats => {
      const s = new CppStats();
      for (let n = 0; n < 512; n++) if ((l.valueMask[n >>> 5]! >>> (n & 31)) & 1) s.add(l.values[n]!);
      return s;
    };
    const signed = (a: number[], b: number[]): number => a[0]! - b[0]! || a[1]! - b[1]! || a[2]! - b[2]!;
    const floor = (o: readonly number[], d: number): number[] => o.map((c) => Math.floor(c / d) * d);
    const slot = (o: readonly number[], parent: number, child: number, log2: number): number => {
      const p = floor(o, parent);
      return (((o[0]! - p[0]!) / child) << (2 * log2)) | (((o[1]! - p[1]!) / child) << log2) | ((o[2]! - p[2]!) / child);
    };
    const groupBy = <T,>(items: T[], key: (t: T) => string): Map<string, T[]> => {
      const m = new Map<string, T[]>();
      for (const it of items) m.set(key(it), [...(m.get(key(it)) ?? []), it]);
      return m;
    };
    const uppers = [...groupBy(leaves, (l) => floor(l.origin, 4096).join(",")).entries()].sort((a, b) =>
      signed(a[0].split(",").map(Number), b[0].split(",").map(Number)),
    );
    const root = new CppStats();
    for (const [, inUpper] of uppers) {
      const upper = new CppStats();
      const lowers = [...groupBy(inUpper, (l) => floor(l.origin, 128).join(",")).values()].sort(
        (a, b) => slot(a[0]!.origin, 4096, 128, 5) - slot(b[0]!.origin, 4096, 128, 5),
      );
      for (const inLower of lowers) {
        const lower = new CppStats();
        for (const l of [...inLower].sort((a, b) => slot(a.origin, 128, 8, 4) - slot(b.origin, 128, 8, 4))) lower.addStats(leafStats(l));
        upper.addStats(lower);
      }
      root.addStats(upper);
    }
    expect(tree.uppers.length).toBe(2);
    expect(tree.lowers.length).toBe(10);
    expect(Object.is(tree.root.average, root.avg)).toBe(true);
    expect(Object.is(tree.root.stdDev, root.std)).toBe(true);
  });
});

describe("Welford merges, many and uneven (f64 bits, as GridStats)", () => {
  it("300 merges of accumulators of 1 to 600 samples and weighted tiles", () => {
    let seed = 5;
    const rnd = (): number => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32;
    const total = new StatsAccumulator();
    const oracle = new CppStats();
    for (let m = 0; m < 300; m++) {
      const part = new StatsAccumulator();
      const cpart = new CppStats();
      const n = 1 + Math.floor(rnd() * 600);
      for (let i = 0; i < n; i++) {
        const v = Math.fround(rnd() * 100 - 20);
        if (rnd() < 0.02) {
          part.addTile(v, 4096, Z, Z);
          cpart.addN(v, 4096);
        } else {
          part.addVoxel(v, 0, 0, 0);
          cpart.add(v);
        }
      }
      total.merge(part);
      oracle.addStats(cpart);
    }
    expect(Object.is(total.average, oracle.avg)).toBe(true);
    expect(Object.is(total.stdDev, oracle.std)).toBe(true);
  });

  it("the merge's product runs left to right (denom * delta * delta * n_this * n_other): a case where order shows", () => {
    // Found by search: (… * 237) * 370 and (… * 370) * 237 round differently in f64.
    const make = (v: number, n: number): [StatsAccumulator, CppStats] => {
      const s = new StatsAccumulator();
      const c = new CppStats();
      for (let i = 0; i < n; i++) {
        s.addVoxel(v, 0, 0, 0);
        c.add(v);
      }
      return [s, c];
    };
    const [a, ca] = make(5.042420387268066, 237);
    const [b, cb] = make(7.048832416534424, 370);
    a.merge(b);
    ca.addStats(cb);
    expect(Object.is(a.stdDev, ca.std)).toBe(true);
    expect(Object.is(a.average, ca.avg)).toBe(true);
  });
});
