import { describe, expect, it } from "vitest";
import { parseVdb } from "../../src/tools/index.js";
import { VDBUnsupportedError } from "../../src/tools/index.js";
import { writeSyntheticVdb, type SynthSpec } from "./synthetic-vdb.js";

const leafValues = new Float32Array(512).fill(2);
const leafMask = new Uint32Array(16).fill(0xffffffff);

function spec(extra: Partial<SynthSpec> = {}): SynthSpec {
  return {
    name: "density",
    background: 0,
    voxelSize: 0.5,
    rootTiles: [],
    uppers: [{ origin: [0, 0, 0], tiles: [], lowers: [{ origin: [0, 0, 0], tiles: [], leaves: [{ origin: [0, 0, 0], values: leafValues, valueMask: leafMask }] }] }],
    ...extra,
  };
}

/** Same-length patch of the grid type string, so a valid file reads as an unsupported value type. */
function patchType(buf: ArrayBuffer, from: string, to: string): ArrayBuffer {
  const bytes = new Uint8Array(buf.slice(0));
  const f = new TextEncoder().encode(from);
  const t = new TextEncoder().encode(to);
  expect(f.length).toBe(t.length);
  outer: for (let i = 0; i + f.length <= bytes.length; i++) {
    for (let j = 0; j < f.length; j++) if (bytes[i + j] !== f[j]) continue outer;
    bytes.set(t, i);
    return bytes.buffer;
  }
  throw new Error("pattern not found");
}

describe("parseVdb descriptors and select", () => {
  const buf = writeSyntheticVdb(spec({ compression: { zip: true } }));

  it("default call is unchanged and also lists descriptors", () => {
    const file = parseVdb(buf);
    expect(file.grids).toHaveLength(1);
    expect(file.descriptors).toHaveLength(1);
    expect(file.descriptors[0]).toMatchObject({
      name: "density",
      gridType: "Tree_float_5_4_3",
      valueType: "float",
      gridClass: "fog volume",
      unsupported: null,
      compression: { zip: true, blosc: false },
    });
    expect(file.descriptors[0]!.transform!.voxelSize[0]).toBeCloseTo(0.5);
  });

  it("select: () => false lists the grid without decoding it", () => {
    const file = parseVdb(buf, { select: () => false });
    expect(file.grids).toHaveLength(0);
    expect(file.descriptors.map((d) => d.name)).toEqual(["density"]);
  });

  it("select decodes only what it accepts and sees the descriptor", () => {
    const seen: string[] = [];
    const file = parseVdb(buf, { select: (d) => (seen.push(d.name), d.valueType === "float") });
    expect(seen).toEqual(["density"]);
    expect(file.grids.map((g) => g.name)).toEqual(["density"]);
    expect(file.grids[0]!.readValue([1, 1, 1])).toEqual({ value: 2, active: true });
  });

  it("flags blosc in the descriptor", () => {
    const b = parseVdb(writeSyntheticVdb(spec({ compression: { blosc: true } })), { select: () => false });
    expect(b.descriptors[0]!.compression.blosc).toBe(true);
  });

  it("an unsupported grid throws without select, and is reported with it", () => {
    const bad = patchType(buf, "Tree_float_5_4_3", "Tree_int64_5_4_3");
    expect(() => parseVdb(bad)).toThrow(VDBUnsupportedError);
    let selectCalls = 0;
    const file = parseVdb(bad, { select: () => (selectCalls++, true) });
    expect(selectCalls).toBe(0);
    expect(file.grids).toHaveLength(0);
    expect(file.descriptors[0]).toMatchObject({ name: "density", valueType: "int64", unsupported: expect.stringContaining("int64") });
    expect(file.descriptors[0]!.transform).toBeNull();
  });

  it("an unsupported tree layout is reported too", () => {
    const bad = patchType(buf, "Tree_float_5_4_3", "Tree_float_5_4_4");
    expect(() => parseVdb(bad)).toThrow(/5-4-4/);
    expect(parseVdb(bad, { select: () => true }).descriptors[0]!.unsupported).toMatch(/5-4-4/);
  });
});
