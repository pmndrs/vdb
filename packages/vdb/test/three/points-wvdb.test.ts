/**
 * K3.7b: `.wvdb` points LOD streaming into `VDBPoints` (`VDBPoints.fromWvdbGrid`, `addWvdbLevel`), and the G14a
 * `pointCount` / `capacity` / `writePoints` it rests on. Oracle: the file decoded at once (`readWvdb`), its levels
 * concatenated coarse first (the stored LOD order, §3.5), against which every streamed state must be a prefix.
 * The GPU upload of the ranges runs in e2e/k3-levelset.spec.ts (`?scene=points-wvdb`).
 */

import { zlibSync } from "fflate";
import { describe, expect, it } from "vitest";
import { VDBError } from "../../src/core/index.js";
import { VDBPoints } from "../../src/three/internal.js";
import { readWvdb, WvdbStreamReader, writeWvdb, type WvdbPointsGrid, type WvdbPointsLevel } from "../../src/tools/internal.js";

/** A points file: n points with colours, 3 LODs (LOD 2 spans two 2^18-point pages). */
function pointsFile(n: number): Uint8Array {
  let s = 12345;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  const positions = Float32Array.from({ length: n * 3 }, () => rnd() * 10 - 5);
  const colors = Float32Array.from({ length: n * 3 }, () => rnd());
  return writeWvdb([{ type: "points", name: "pts", positions, colors, spacing: 0.04 }], { lods: 3, compress: (b) => zlibSync(b, { level: 1 }) }).bytes;
}

const N = 300_000;
const BYTES = pointsFile(N);
const FULL = readWvdb(BYTES).grids[0] as WvdbPointsGrid;
const CONCAT = (() => {
  const positions = new Float32Array(FULL.count * 3);
  const colors = new Float32Array(FULL.count * 3);
  for (const l of FULL.levels) {
    positions.set(l.positions, l.firstPoint * 3);
    colors.set(l.colors!, l.firstPoint * 3);
  }
  return { positions, colors };
})();

/** The first index below `n` where `a` and `b` differ, or -1 (a loop: deep equality on 10^6 floats is slow). */
function firstDiff(a: Float32Array, b: Float32Array, n: number): number {
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return -1;
}

/** Sums the update ranges added since the last call (three clears them after an upload; nothing uploads here). */
function takeRanges(attr: { updateRanges: { start: number; count: number }[]; clearUpdateRanges(): void }): { start: number; count: number }[] {
  const out = attr.updateRanges.map((r) => ({ ...r }));
  attr.clearUpdateRanges();
  return out;
}

describe("VDBPoints: .wvdb points LOD streaming (K3.7b)", () => {
  it("streamed in small pushes with partial LODs, the drawn points are always a prefix of the coarse-first list", () => {
    const reader = new WvdbStreamReader({ partialLods: true });
    let points: VDBPoints | null = null;
    let last = 0;
    let events = 0;
    let uploaded = 0;
    const counts: number[] = [];
    for (let at = 0; at < BYTES.length; at += 65_537) {
      for (const e of reader.push(BYTES.subarray(at, Math.min(BYTES.length, at + 65_537)))) {
        points ??= VDBPoints.fromWvdbGrid(reader.header!.grids[0] as never);
        points.addWvdbLevel(e.levels[0] as WvdbPointsLevel);
        events++;
        // Only the new range goes up: each point is written once over the whole stream.
        const ranges = takeRanges(points.positionAttribute as never);
        for (const r of ranges) {
          expect(r.start).toBe(uploaded * 3);
          uploaded += r.count / 3;
        }
        takeRanges(points.colorAttribute as never);
        const n = points.pointCount;
        expect(n).toBeGreaterThanOrEqual(last);
        last = n;
        counts.push(n);
        expect(firstDiff(points.positionAttribute.array as Float32Array, CONCAT.positions, n * 3)).toBe(-1);
        expect(firstDiff(points.colorAttribute!.array as Float32Array, CONCAT.colors, n * 3)).toBe(-1);
        expect((points as unknown as { count: number }).count).toBe(n);
        expect(points.spacing).toBeCloseTo(0.04 * Math.cbrt(N / Math.max(1, n)), 6);
      }
    }
    reader.end();
    expect(points!.pointCount).toBe(N);
    expect(points!.capacity).toBe(N);
    expect(uploaded).toBe(N);
    expect(points!.spacing).toBeCloseTo(0.04, 7);
    // LOD 2 spans 2 pages: partial events draw it before it completes.
    expect(events).toBeGreaterThan(FULL.levels.length);
    // The coarse levels are prefixes of the stored order: LOD 0 = every 64th point, LODs 0-1 = every 8th.
    expect(counts).toContain(FULL.levels[0]!.count);
    expect(counts).toContain(FULL.levels[0]!.count + FULL.levels[1]!.count);
    expect(FULL.levels[0]!.count).toBe(Math.ceil(N / 64));
  });

  it("a decoded file's levels work too; a level arriving after a gap is written but not drawn until the gap fills", () => {
    const points = VDBPoints.fromWvdbGrid(FULL);
    expect(points.pointCount).toBe(0);
    expect(points.colorAttribute).not.toBeNull();
    points.addWvdbLevel(FULL.levels[1]!);
    expect(points.pointCount).toBe(0);
    points.addWvdbLevel(FULL.levels[0]!);
    expect(points.pointCount).toBe(FULL.levels[0]!.count);
    points.addWvdbLevel(FULL.levels[1]!);
    expect(points.pointCount).toBe(FULL.levels[0]!.count + FULL.levels[1]!.count);
    expect(firstDiff(points.positionAttribute.array as Float32Array, CONCAT.positions, points.pointCount * 3)).toBe(-1);
  });

  it("rejects levels without fromWvdbGrid, non-points grids and out-of-range writes", () => {
    const plain = new VDBPoints({ positions: new Float32Array(30) });
    expect(() => plain.addWvdbLevel(FULL.levels[0]!)).toThrow(VDBError);
    expect(() => VDBPoints.fromWvdbGrid({ type: "volume" } as never)).toThrow(VDBError);
    expect(() => plain.writePoints(8, new Float32Array(9))).toThrow(/capacity/);
    expect(() => plain.writePoints(0, new Float32Array(3), new Float32Array(3))).toThrow(/colours/);
    expect(() => {
      plain.pointCount = 11;
    }).toThrow(VDBError);
    plain.pointCount = 4;
    expect((plain as unknown as { count: number }).count).toBe(4);
    plain.writePoints(9, new Float32Array([1, 2, 3]));
    expect([...(plain.positionAttribute.array as Float32Array).subarray(27, 30)]).toEqual([1, 2, 3]);
  });
});
