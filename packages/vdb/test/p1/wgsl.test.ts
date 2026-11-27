/**
 * P1.3–P1.5 generator checks without a GPU (the GPU parity is e2e/p1-parity.spec.ts): what each new variant pulls
 * in, and the lean reader's baked offsets against the vendored stride tables (so a NanoVDB ABI change that moves a
 * field fails here, not in a render).
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { vdbModuleWgsl, type VDBModuleOptions } from "../../src/core/index.js";
import { LEAN_OFFSETS } from "../../src/core/wgsl/lean.js";
import { vdbModuleDeclarations } from "../../src/core/wgsl/generate.js";

const STRIDES = JSON.parse(readFileSync(new URL("../../vendor/stride-tables.json", import.meta.url), "utf8")) as {
  defines: Record<string, number | string>;
  gridTypeConstants: Record<string, Record<string, number>>;
};
const slot = { group: 0, firstBinding: 0 } as const;

describe("P1.5 lean reader", () => {
  it("bakes the vendored stride tables' offsets (Float, Half and every Fp type above the leaf)", () => {
    const d = (k: string) => Number(STRIDES.defines[`PNANOVDB_${k}`]);
    expect(LEAN_OFFSETS.gridSize).toBe(d("GRID_SIZE"));
    expect(LEAN_OFFSETS.treeRootOffset).toBe(d("TREE_OFF_NODE_OFFSET_ROOT"));
    expect(LEAN_OFFSETS.rootTableSize).toBe(d("ROOT_OFF_TABLE_SIZE"));
    expect(LEAN_OFFSETS.rootTileChild).toBe(d("ROOT_TILE_OFF_CHILD"));
    expect(LEAN_OFFSETS.rootTileState).toBe(d("ROOT_TILE_OFF_STATE"));
    expect(LEAN_OFFSETS.upperValueMask).toBe(d("UPPER_OFF_VALUE_MASK"));
    expect(LEAN_OFFSETS.upperChildMask).toBe(d("UPPER_OFF_CHILD_MASK"));
    expect(LEAN_OFFSETS.lowerValueMask).toBe(d("LOWER_OFF_VALUE_MASK"));
    expect(LEAN_OFFSETS.lowerChildMask).toBe(d("LOWER_OFF_CHILD_MASK"));
    expect(LEAN_OFFSETS.leafValueMask).toBe(d("LEAF_OFF_VALUE_MASK"));
    expect(LEAN_OFFSETS.leafFlags).toBe(d("LEAF_OFF_BBOX_DIF_AND_FLAGS"));
    for (const t of ["FLOAT", "HALF", "FP4", "FP8", "FP16", "FPN"]) {
      const g = STRIDES.gridTypeConstants[t]!;
      expect(g.root_off_background, t).toBe(LEAN_OFFSETS.rootBackground);
      expect(g.root_size, t).toBe(LEAN_OFFSETS.rootTiles);
      expect(g.root_tile_size, t).toBe(LEAN_OFFSETS.rootTileSize);
      expect(g.root_tile_off_value, t).toBe(LEAN_OFFSETS.rootTileValue);
      expect(g.upper_off_table, t).toBe(LEAN_OFFSETS.upperTable);
      expect(g.lower_off_table, t).toBe(LEAN_OFFSETS.lowerTable);
      expect(g.table_stride, t).toBe(LEAN_OFFSETS.tableStride);
      expect(g.leaf_off_table, t).toBe(LEAN_OFFSETS.leafTable);
    }
    // Fp leaves: minimum and quantum sit just before the table (LeafFnBase).
    expect(LEAN_OFFSETS.leafTable - d("LEAF_TABLE_NEG_OFF_MINIMUM")).toBe(LEAN_OFFSETS.leafMinimum);
    expect(LEAN_OFFSETS.leafTable - d("LEAF_TABLE_NEG_OFF_QUANTUM")).toBe(LEAN_OFFSETS.leafQuantum);
  });

  it("pulls in no vendored function, only the grid-type constants", () => {
    for (const types of [["float"], ["float", "fp8", "fpn"], ["half"]] as const) {
      for (const profile of ["read", "sample"] as const) {
        const decls = vdbModuleDeclarations({ slot, types, profile, reader: "lean", outside: "zero" });
        const fns = decls.filter((n) => !/^PNANOVDB_GRID_TYPE_/.test(n));
        // sample's vdbRayClip keeps the vendored slab test (its one vendored function).
        expect(fns, `${types}/${profile}`).toEqual(profile === "sample" ? ["pnanovdb_hdda_ray_clip"] : []);
      }
    }
    const code = vdbModuleWgsl({ slot, types: ["float"], profile: "read", reader: "lean" });
    expect(code).not.toContain("vdb_readaccessor");
    expect(code).toContain("fn vdbReadValue(");
    expect(code).toContain("fn vdbIsActive(");
    expect(vdbModuleWgsl({ slot, types: ["float"], profile: "read", reader: "lean", isActive: false })).not.toContain("fn vdbIsActive(");
  });
});

describe("P1.3 / P1.4 majorant profile", () => {
  const m = (o: Partial<VDBModuleOptions>) => vdbModuleWgsl({ slot, types: ["float"], profile: "majorant", ...o } as VDBModuleOptions);
  it("brick and global are sample + the walk (no march layer); nodeMax adds it", () => {
    for (const majorant of ["brick", "global"] as const) {
      const code = m({ majorant });
      expect(code).toContain("fn vdbMajorantInit(");
      expect(code).toContain("fn vdbMajorantNext(");
      expect(code).toContain("fn vdbMajorantGlobal(");
      expect(code).toContain("fn vdbSample(");
      expect(code).not.toContain("vdb_hdda_init");
      expect(code).not.toContain("fn vdbNodeMax(");
    }
    expect(m({ majorant: "brick" })).toContain("VDB_MAJORANT_MAGIC");
    expect(m({ majorant: "global" })).not.toContain("VDB_MAJORANT_MAGIC");
    const node = m({ majorant: "nodeMax" });
    expect(node).toContain("fn vdbNodeMax(");
    expect(node).toContain("fn vdb_maj_bound(");
  });

  it("minima adds vdbMajorantMinimum only when asked; outside 'zero' clamps the global bound at 0", () => {
    expect(m({})).not.toContain("vdbMajorantMinimum");
    expect(m({ minima: true })).toContain("fn vdbMajorantMinimum(");
    expect(m({ outside: "zero" })).toContain("max(bitcast<f32>(vdb_rec(g, VDB_REC_VALUE_MAX)), 0.0)");
    expect(m({ sampler: "nearest", majorant: "global" })).toContain("vec3f(vdbIndexMin(g)) - vec3f(0.5)");
  });

  it("default modules keep their bytes: a sample module never mentions the majorant", () => {
    expect(vdbModuleWgsl({ slot, types: ["float"], profile: "sample" })).not.toMatch(/Majorant|VDB_MAJ/);
  });
});
