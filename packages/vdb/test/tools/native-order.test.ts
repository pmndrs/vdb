/**
 * K7.C1 / K7.H1: our NanoVDB images against native NanoVDB, byte for byte.
 *
 * Two fixes made them equal: the root table and the upper nodes in native's signed origin order (K4.12's
 * unsigned key order reverted; v14's `nanovdb_validate` rejected our multi-upper grids with negative
 * coordinates: "Upper node is not stored breadth-first"), and node ave/stddev from GridStats' Welford
 * update (K7.H1). Bytes allowed to differ: GridData's checksum and version (8..20: we write our ABI
 * version and no checksum) and, for re-layouts, the root background (`effectiveBackground` takes the
 * near-field inactive value, by design).
 *
 * Always: the committed native v14 grid (`fixtures/native-v14/sphere_fog_blosc.nvdb`) re-laid out as Float.
 * With `pnpm parity:native` fixtures: `buildFromVdb` of the four samples vs the native Float images.
 */
import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { NvdbFile, buildFromVdb, parseVdb, vdbQuantize } from "../../src/tools/index.js";
import { nativeImages } from "./blosc-corpus.js";

const ROOT = new URL("../../../../", import.meta.url).pathname;
const ab = (b: Buffer): ArrayBuffer => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
const u8 = (img: Uint32Array | Uint8Array): Uint8Array => new Uint8Array(img.buffer, img.byteOffset, img.byteLength);

/** Offsets where `ours` differs from `native`, outside GridData's checksum and version and the given ranges. */
function differing(ours: Uint32Array | Uint8Array, native: Uint32Array | Uint8Array, skip: [number, number][] = []): number[] {
  const a = u8(ours);
  const b = u8(native);
  expect(a.length).toBe(b.length);
  const out: number[] = [];
  for (let i = 0; i < b.length; i++) {
    if (a[i] === b[i] || (i >= 8 && i < 20) || skip.some(([s, e]) => i >= s && i < e)) continue;
    out.push(i);
    if (out.length > 20) break;
  }
  return out;
}

const rootOffset = (img: Uint32Array): number => 672 + Number(new DataView(img.buffer, img.byteOffset).getBigUint64(672 + 24, true));

describe("native NanoVDB v14 grid (committed): our Float re-layout equals it", () => {
  const native = nativeImages(new URL("./fixtures/native-v14/sphere_fog_blosc.nvdb", import.meta.url).pathname)[0]!;
  const image = new Uint32Array(native.buffer, native.byteOffset, native.byteLength / 4);
  it("byte for byte but the root background", () => {
    const ours = vdbQuantize(image, "float");
    const bg = rootOffset(ours) + 28;
    expect(differing(ours, image, [[bg, bg + 4]])).toEqual([]);
  });
});

const PARITY = `${ROOT}fixtures/parity`;
describe.each(["cube", "sphere", "smoke", "utahteapot"])("%s: buildFromVdb vs native Float (pnpm parity:native)", (sample) => {
  const src = `${PARITY}/${sample}.normalized.vdb`;
  const nat = `${PARITY}/${sample}.native.float.nvdb`;
  it.skipIf(!existsSync(src) || !existsSync(nat))("byte for byte (cube and sphere have negative coordinates and 8 upper nodes)", () => {
    const grid = parseVdb(ab(readFileSync(src))).grids.find((g) => g.gridType === "Tree_float_5_4_3")!;
    const native = NvdbFile.fromArrayBuffer(ab(readFileSync(nat))).gridImage(0);
    expect(differing(buildFromVdb(grid), native)).toEqual([]);
  });
});
