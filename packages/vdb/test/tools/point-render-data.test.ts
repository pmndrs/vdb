/**
 * G14a: `vdbPointRenderData` options beyond K3.7 (other attributes, the prefix-subsample order, progress, bounds)
 * and `vdbPointRenderDataTransfer`. The oracle is the grid's own point list: index arithmetic on it, not the
 * implementation's loops.
 */

import { describe, expect, it } from "vitest";
import { vdbPointRenderData, vdbPointRenderDataTransfer, type VDBPointRenderDataProgress, type VdbPointDataGrid, type VdbPointList } from "../../src/tools/internal.js";

/** A stand-in grid: n points at (i, 2i, -i), Cd = (i, i, i) / n, v = (i, 0, 1) (f32), id = i (int32), 4 per voxel. */
function fakeGrid(n: number): VdbPointDataGrid {
  const list: VdbPointList = {
    count: n,
    positions: Float32Array.from({ length: n * 3 }, (_, k) => [1, 2, -1][k % 3]! * Math.floor(k / 3)),
    attributes: {
      Cd: Float32Array.from({ length: n * 3 }, (_, k) => Math.floor(k / 3) / n),
      v: Float32Array.from({ length: n * 3 }, (_, k) => [Math.floor(k / 3), 0, 1][k % 3]!),
      id: Int32Array.from({ length: n }, (_, i) => i),
    },
    attributeShapes: { Cd: { components: 3, stride: 1 }, v: { components: 3, stride: 1 }, id: { components: 1, stride: 1 } },
    groups: {},
    skipped: [],
  };
  return {
    valueType: "ptdataidx32",
    pointCount: n,
    activeVoxelCount: BigInt(Math.ceil(n / 4)),
    transform: { type: "UniformScaleMap", matrix: [], voxelSize: [0.5, 0.5, 0.5] },
    points: (options?: { onProgress?: (done: number, total: number) => void }) => {
      options?.onProgress?.(n, n);
      return list;
    },
  } as unknown as VdbPointDataGrid;
}

/** The grid point index at each output slot (from the x coordinate, which is the index). */
const indices = (positions: Float32Array): number[] => Array.from({ length: positions.length / 3 }, (_, j) => positions[j * 3]!);

describe("vdbPointRenderData: order 'prefix' (G14a)", () => {
  it("the 10 % prefix is every 10th point, 50 % every 2nd, 100 % all of them once", () => {
    const n = 1003;
    const d = vdbPointRenderData(fakeGrid(n), { order: "prefix" });
    const order = indices(d.positions);
    expect(d.prefix).toHaveLength(11);
    expect(d.prefix![0]).toBe(0);
    expect(d.prefix![10]).toBe(n);
    expect(order.slice(0, d.prefix![1])).toEqual(Array.from({ length: Math.ceil(n / 10) }, (_, k) => k * 10));
    expect(order.slice(0, d.prefix![5]).sort((a, b) => a - b)).toEqual(Array.from({ length: Math.ceil(n / 2) }, (_, k) => k * 2));
    expect([...order].sort((a, b) => a - b)).toEqual(Array.from({ length: n }, (_, i) => i));
    // Every tenth prefix is a union of residue classes mod 10.
    for (let k = 1; k <= 10; k++) {
      const classes = new Set(order.slice(0, d.prefix![k]).map((i) => i % 10));
      expect(classes.size).toBe(k);
    }
  });

  it("attribute arrays and colours follow the order; extra attributes keep their element type", () => {
    const d = vdbPointRenderData(fakeGrid(257), { order: "prefix", attributes: ["v", "id", "missing"] });
    const order = indices(d.positions);
    expect(Object.keys(d.attributes).sort()).toEqual(["id", "v"]);
    expect(d.attributes.v!.components).toBe(3);
    expect(d.attributes.id!.components).toBe(1);
    expect(d.attributes.id!.values).toBeInstanceOf(Int32Array);
    expect(d.attributes.v!.values).toBeInstanceOf(Float32Array);
    order.forEach((i, j) => {
      expect(d.attributes.id!.values[j]).toBe(i);
      expect([...(d.attributes.v!.values as Float32Array).subarray(j * 3, j * 3 + 3)]).toEqual([i, 0, 1]);
      expect(d.colors![j * 3 + 2]).toBeCloseTo(i / 257, 6);
      expect([...d.positions.subarray(j * 3, j * 3 + 3)]).toEqual([i, 2 * i, -i]);
    });
  });

  it("with a stride, the order is over the kept points (every stride-th), and the spacing is the kept points'", () => {
    const d = vdbPointRenderData(fakeGrid(1000), { order: "prefix", maxPoints: 300 });
    expect(d.stride).toBe(4);
    expect(d.count).toBe(250);
    const order = indices(d.positions);
    expect(order.slice(0, d.prefix![1])).toEqual(Array.from({ length: 25 }, (_, k) => k * 40));
    expect([...order].sort((a, b) => a - b)).toEqual(Array.from({ length: 250 }, (_, q) => q * 4));
    expect(d.spacing).toBeCloseTo(vdbPointRenderData(fakeGrid(1000), { maxPoints: 300 }).spacing, 12);
  });

  it("order 'grid' (default) keeps the grid's order and has no prefix table; bad orders throw", () => {
    const d = vdbPointRenderData(fakeGrid(30), { attributes: ["id"] });
    expect(d.prefix).toBeNull();
    expect(indices(d.positions)).toEqual(Array.from({ length: 30 }, (_, i) => i));
    expect([...d.attributes.id!.values]).toEqual(Array.from({ length: 30 }, (_, i) => i));
    expect(() => vdbPointRenderData(fakeGrid(30), { order: "random" as never })).toThrow(/order/);
  });
});

describe("vdbPointRenderData: bounds, progress, transfer (G14a)", () => {
  it("bounds are the kept positions' box; null without points", () => {
    const d = vdbPointRenderData(fakeGrid(100), { maxPoints: 50 });
    // (+ 0 turns the -0 of point 0's z into 0)
    expect({ min: d.bounds!.min.map((v) => v + 0), max: d.bounds!.max.map((v) => v + 0) }).toEqual({ min: [0, 0, -98], max: [98, 196, 0] });
    expect(vdbPointRenderData(fakeGrid(0)).bounds).toBeNull();
  });

  it("reports the decode, then the copy, each ending at its total", () => {
    const seen: VDBPointRenderDataProgress[] = [];
    vdbPointRenderData(fakeGrid(64), { order: "prefix", onProgress: (p) => seen.push({ ...p }) });
    expect(seen[0]).toEqual({ stage: "decode", done: 64, total: 64 });
    expect(seen[seen.length - 1]).toEqual({ stage: "copy", done: 64, total: 64 });
    expect(seen.filter((p) => p.stage === "decode")).toHaveLength(1);
  });

  it("the transfer list holds each output buffer once", () => {
    const d = vdbPointRenderData(fakeGrid(40), { attributes: ["v", "id"] });
    const t = vdbPointRenderDataTransfer(d);
    expect(t).toHaveLength(4);
    expect(new Set(t).size).toBe(4);
    expect(t).toContain(d.positions.buffer);
    expect(t).toContain(d.colors!.buffer);
    expect(t).toContain(d.attributes.v!.values.buffer);
    // A structured clone with the transfer list moves the arrays (the worker's postMessage).
    const moved = structuredClone(d, { transfer: t });
    expect(moved.positions.length).toBe(120);
    expect(d.positions.length).toBe(0);
  });
});
