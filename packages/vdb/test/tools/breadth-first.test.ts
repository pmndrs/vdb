/**
 * NanoVDB breadth-first node order (K4.17 / K7.7 native finding).
 *
 * Our grids set `GridFlags::IsBreadthFirst`, which promises what NanoVDB's
 * `tools::checkGrid` (`nanovdb_validate`, v14.0.0 `GridValidator.h`) checks:
 * walking the root table in stored (key) order, then each node's children in
 * child-mask order, visits the upper, lower and leaf nodes in the order they are
 * stored. Before the fix, `assembleTree` stored the upper nodes by SIGNED origin
 * while it sorted the root table by the unsigned key (K4.12), so every grid with
 * upper nodes on both sides of 0 (bunny_cloud, any build with a negative origin)
 * failed `nanovdb_validate` ("Upper node is not stored breadth-first"). Both now
 * follow the signed origin, as native NanoVDB stores them (OpenVDB's root map).
 *
 * The check here is a twin of the validator's: per level, the walk order's node
 * addresses must ascend. Native v14 files pass it (sanity of the twin).
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  NvdbFile,
  vdbBakeCoarseFirst,
  vdbBuildFromDense,
  vdbCollapseGrid,
  vdbQuantize,
  vdbResampleGrid,
} from "../../src/tools/index.js";
import { treeNodes } from "../../src/tools/cpu/march.js";

/** Levels whose nodes the root-first walk does not visit in storage order (empty: breadth-first). */
function breadthFirstErrors(image: Uint32Array): string[] {
  const last = [-1, -1, -1];
  const errors = new Set<string>();
  for (const node of treeNodes(image)) {
    if (node.level === 3) continue;
    if (node.address <= last[node.level]!) errors.add(["leaf", "lower", "upper"][node.level]!);
    last[node.level] = node.address;
  }
  return [...errors];
}

/** Upper nodes per root tile: 20³ voxels centred on the origin touch all eight octants. */
function straddlingGrid(): Uint32Array {
  const n = 20;
  const values = new Float32Array(n * n * n);
  for (let i = 0; i < values.length; i++) values[i] = 0.25 + (i % 7) / 8;
  return vdbBuildFromDense(values, [n, n, n], { origin: [-10, -10, -10], gridName: "octants" });
}

function nvdbImages(rel: string): Uint32Array[] {
  const b = readFileSync(new URL(rel, import.meta.url));
  const file = NvdbFile.fromArrayBuffer(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
  return file.grids.map((_, i) => file.gridImage(i));
}

describe("NanoVDB breadth-first storage order (nanovdb_validate twin)", () => {
  it("the twin accepts native NanoVDB v14 files", () => {
    for (const rel of ["./fixtures/native-v14/sphere_fog_blosc.nvdb", "./fixtures/native-v14/two_grids_blosc.nvdb"]) {
      for (const image of nvdbImages(rel)) expect(breadthFirstErrors(image), rel).toEqual([]);
    }
  });

  it("a build whose upper nodes straddle 0 stores them in root-table order", () => {
    const image = straddlingGrid();
    expect(treeNodes(image).filter((n) => n.level === 2)).toHaveLength(8);
    expect(breadthFirstErrors(image)).toEqual([]);
  });

  it("quantize, collapse, resample and coarse-first keep the order", () => {
    const image = straddlingGrid();
    for (const mode of ["fp8", "fpn", "half"] as const) expect(breadthFirstErrors(vdbQuantize(image, mode)), mode).toEqual([]);
    expect(breadthFirstErrors(vdbCollapseGrid(image).image), "collapse").toEqual([]);
    const resampled = vdbResampleGrid(image, { voxelSize: 0.5, bbox: [[-10, -10, -10], [9, 9, 9]] });
    expect(breadthFirstErrors(resampled.image), "resample").toEqual([]);
    const bytes = vdbBakeCoarseFirst(image, { previewResolution: 8 });
    const file = NvdbFile.fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    for (let i = 0; i < file.grids.length; i++) expect(breadthFirstErrors(file.gridImage(i)), file.grids[i]!.name).toEqual([]);
  });
});
