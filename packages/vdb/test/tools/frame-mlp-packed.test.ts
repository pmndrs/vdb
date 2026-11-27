import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import {
  vdbFrameMlpForward,
  vdbFrameMlpForwardPacked,
  vdbFrameMlpFromNeural,
  vdbFrameMlpFromTza,
  vdbFrameMlpPack,
  vdbFrameMlpPackedOffsets,
  vdbFrameMlpPackedShape,
  vdbFrameMlpShape,
  vdbFrameMlpToTza,
  vdbFrameNeuralWeights,
  vdbRoundToHalf,
  vdbTzaRead,
  vdbTzaWrite,
  VDBToolsError,
  type VDBFrameMlp,
} from "../../src/tools/internal.js";

// K6.Na (pmndrs/vdb#3): the 'neural' network on the pmndrs neural tooling's layer layout and weight container.
let seed = 11;
const r = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

/** A random network of any widths (not multiples of 4), so the packing's padding is exercised. */
function randomMlp(widths: number[], output: VDBFrameMlp["output"] = "linear"): VDBFrameMlp {
  const layers = widths.slice(1).map((outputs, i) => ({ inputs: widths[i]!, outputs, activation: (i === widths.length - 2 ? "linear" : "relu") as "relu" | "linear" }));
  const total = layers.reduce((s, l) => s + l.outputs * (l.inputs + 1), 0);
  return { name: "random", levels: (widths[0]! - 2) / 3, layers, output, weights: Float32Array.from({ length: total }, () => r() * 2 - 1) };
}

describe("K6.Na packed MLP layout (denoiser-wgsl packWeights, one tap)", () => {
  const ours = vdbFrameMlpFromNeural(vdbFrameNeuralWeights);

  it("the default network: 465 vec4s (20 → 32: 8 × 5 mat4x4 + 8 bias groups; 32 → 32: 8 × 8 + 8; 32 → 1: 1 × 8 + 1)", () => {
    expect(vdbFrameMlpPackedOffsets(ours)).toEqual([0, 168, 432]);
    expect(vdbFrameMlpPack(ours).length).toBe(465 * 4);
    const shape = vdbFrameMlpPackedShape(ours);
    const flat = vdbFrameMlpShape(ours);
    // the same info and layer shapes; only the offsets (vec4s, not floats) differ
    expect([...shape].filter((_, i) => i < 4 || i % 4 !== 3)).toEqual([...flat].filter((_, i) => i < 4 || i % 4 !== 3));
    expect([shape[7], shape[11], shape[15]]).toEqual([0, 168, 432]);
  });

  it("loses nothing: the packed forward pass equals the generic one (f64) for our network and odd widths", () => {
    const nets = [ours, randomMlp([11, 7, 5, 1], "exp"), randomMlp([20, 13, 1]), randomMlp([8, 64, 3, 2], "exp")];
    for (const net of nets) {
      const packed = vdbFrameMlpPack(net);
      const shape = vdbFrameMlpPackedShape(net);
      for (let i = 0; i < 100; i++) {
        const x = Array.from({ length: net.layers[0]!.inputs }, () => r() * 3);
        const a = vdbFrameMlpForward(net, x);
        expect(vdbFrameMlpForwardPacked(packed, shape, x)).toBeCloseTo(a, 9);
      }
    }
  });

  it("fail-before: the first layer's matrices transposed or a shifted bias group changes the output", () => {
    const packed = vdbFrameMlpPack(ours);
    const shape = vdbFrameMlpPackedShape(ours);
    const transposed = Float32Array.from(packed);
    for (let m = 0; m < 40; m++) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) transposed[m * 16 + 4 * j + i] = packed[m * 16 + 4 * i + j]!;
    const biasShift = Float32Array.from(packed);
    biasShift[(432 + 32) * 4] = packed[(432 + 32) * 4]! + 0.5; // the output bias (every voxel reads it)
    let moved = 0;
    for (let i = 0; i < 50; i++) {
      const x = Array.from({ length: 20 }, () => r() * 0.5);
      const a = vdbFrameMlpForwardPacked(packed, shape, x);
      if (Math.abs(vdbFrameMlpForwardPacked(transposed, shape, x) - a) > 1e-6 * Math.max(1, a)) moved++;
      expect(Math.abs(vdbFrameMlpForwardPacked(biasShift, shape, x) - a)).toBeGreaterThan(1e-4 * a);
    }
    expect(moved).toBeGreaterThan(40);
  });
});

describe("K6.Na .tza weight container (pmndrs/denoiser parseTZA format)", () => {
  const ours = vdbFrameMlpFromNeural(vdbFrameNeuralWeights);

  it("round-trips our network exactly in f32 (layers.<i>.weight [out, in], layers.<i>.bias [out])", () => {
    const blob = vdbFrameMlpToTza(ours);
    const tensors = vdbTzaRead(blob);
    expect([...tensors.keys()]).toEqual(["layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias", "layers.2.weight", "layers.2.bias"]);
    expect(tensors.get("layers.0.weight")!.shape).toEqual([32, 20]);
    expect(tensors.get("layers.0.weight")!.layout).toBe("oi");
    const back = vdbFrameMlpFromTza(blob, { name: ours.name });
    expect(back.layers).toEqual(ours.layers);
    expect(back.levels).toBe(6);
    expect(back.output).toBe("exp");
    expect(back.weights).toEqual(ours.weights);
  });

  it("f16 tensors round each weight to the nearest half (half the bytes; a different network)", () => {
    const blob = vdbFrameMlpToTza(ours, { type: "h" });
    expect(blob.byteLength).toBeLessThan(vdbFrameMlpToTza(ours).byteLength);
    const back = vdbFrameMlpFromTza(blob);
    expect(Array.from(back.weights)).toEqual(Array.from(ours.weights, (w) => Math.fround(vdbRoundToHalf(w))));
  });

  it("any tensor set round-trips; bad blobs throw VDBToolsError", () => {
    const t = new Map([
      ["a", { data: Float32Array.of(1, -2.5, 3), shape: [3], layout: "o" }],
      ["b.c", { data: Float32Array.of(0.25, 4, 5, 6), shape: [2, 2], type: "h" as const }],
    ]);
    const back = vdbTzaRead(vdbTzaWrite(t));
    expect(Array.from(back.get("a")!.data)).toEqual([1, -2.5, 3]);
    expect(back.get("b.c")!.shape).toEqual([2, 2]);
    expect(back.get("b.c")!.type).toBe("h");
    expect(Array.from(back.get("b.c")!.data)).toEqual([0.25, 4, 5, 6]);
    const blob = vdbTzaWrite(t);
    const wrongMagic = Uint8Array.from(blob);
    wrongMagic[0] = 0;
    expect(() => vdbTzaRead(wrongMagic)).toThrow(VDBToolsError);
    expect(() => vdbTzaRead(blob.subarray(0, blob.byteLength - 3))).toThrow(VDBToolsError);
    expect(() => vdbFrameMlpFromTza(vdbTzaWrite(new Map([["x", { data: Float32Array.of(1), shape: [1] }]])))).toThrow(/layers\.0\.weight/);
  });

  // The sibling's own reader (read-only checkout, found by PMNDRS_DENOISER_DIR or beside the main checkout): our blob
  // gives it the same tensors. Skipped where the checkout is absent (CI).
  const here = path.dirname(fileURLToPath(import.meta.url));
  const candidates = [process.env.PMNDRS_DENOISER_DIR, path.resolve(here, "../../../../../Denoiser"), path.resolve(here, "../../../../../../../../Denoiser")].filter((p): p is string => !!p);
  const denoiserTza = candidates.map((d) => path.join(d, "packages/denoiser-core/src/tza.ts")).find((f) => existsSync(f));
  it.skipIf(!denoiserTza)("pmndrs/denoiser's parseTZA reads our blob to the same tensors", async () => {
    const { parseTZA } = (await import(/* @vite-ignore */ pathToFileURL(denoiserTza!).href)) as { parseTZA: (b: ArrayBuffer) => Map<string, { data: Float32Array; shape: number[] }> };
    for (const type of ["f", "h"] as const) {
      const blob = vdbFrameMlpToTza(ours, { type });
      const theirs = parseTZA(blob.slice().buffer);
      const mine = vdbTzaRead(blob);
      expect([...theirs.keys()]).toEqual([...mine.keys()]);
      for (const [name, t] of mine) {
        expect(theirs.get(name)!.shape).toEqual(t.shape);
        expect(Array.from(theirs.get(name)!.data)).toEqual(Array.from(t.data));
      }
    }
  });
});
