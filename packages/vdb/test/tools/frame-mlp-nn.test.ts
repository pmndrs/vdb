import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import {
  vdbFrameMlpForward,
  vdbFrameMlpFromNeural,
  vdbFrameMlpFromNn,
  vdbFrameMlpNnInputs,
  vdbFrameMlpToNn,
  vdbFrameNeuralWeights,
  VDBToolsError,
  type VDBFrameMlp,
  type VDBFrameMlpNn,
} from "../../src/tools/internal.js";

// K6.Nc (pmndrs/vdb#3): the 'neural' network in the bias-free weight convention of the pmndrs neural radiance cache
// (pmndrs/pathtracer `@pathtracer/mlp`).
let seed = 23;
const r = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

function randomMlp(widths: number[]): VDBFrameMlp {
  const layers = widths.slice(1).map((outputs, i) => ({ inputs: widths[i]!, outputs, activation: (i === widths.length - 2 ? "linear" : "relu") as "relu" | "linear" }));
  const total = layers.reduce((s, l) => s + l.outputs * (l.inputs + 1), 0);
  return { name: "random", levels: (widths[0]! - 2) / 3, layers, output: "exp", weights: Float32Array.from({ length: total }, () => r() * 2 - 1) };
}

/** Feature vectors in the range of our inputs (log(1 + depth) values, a √albedo in [0, 1]); f32 values, as the input vector holds. */
const featureSets = (n: number, count: number) => Array.from({ length: count }, () => Array.from({ length: n }, (_, i) => Math.fround(i === n - 1 ? r() : r() * 4)));

/** `@pathtracer/mlp`'s network as its oracle documents it (no biases, hidden ReLU, last linear), f64; first output. */
function nnForward(nn: Pick<VDBFrameMlpNn, "shape" | "weights">, x: ArrayLike<number>): number {
  const { inputs, width, hidden, outputs } = nn.shape;
  let a = Array.from(x);
  let at = 0;
  for (let l = 0; l <= hidden; l++) {
    const nIn = l === 0 ? inputs : width;
    const nOut = l === hidden ? outputs : width;
    const z: number[] = [];
    for (let o = 0; o < nOut; o++) {
      let s = 0;
      for (let i = 0; i < nIn; i++) s += nn.weights[at + o * nIn + i]! * a[i]!;
      z.push(l === hidden ? s : Math.max(s, 0));
    }
    at += nIn * nOut;
    a = z;
  }
  return a[0]!;
}

const expOut = (o: number) => Math.exp(Math.min(o, 4));
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-30);

describe("K6.Nc 'neural' in @pathtracer/mlp's bias-free convention", () => {
  const ours = vdbFrameMlpFromNeural(vdbFrameNeuralWeights);
  const nn = vdbFrameMlpToNn(ours);

  it("the default network: 24 → 2 × 40 → 1, a constant one at input 20, 2 600 weights", () => {
    expect(nn.shape).toEqual({ inputs: 24, width: 40, hidden: 2, outputs: 1 });
    expect(nn.features).toBe(20);
    expect(nn.ones).toEqual([20]);
    expect(nn.weights.length).toBe(24 * 40 + 40 * 40 + 40);
    // the carriers: layer 0 unit 39 reads the one with weight 1; layer 1 unit 39 reads unit 39 with weight 1
    expect(nn.weights[39 * 24 + 20]).toBe(1);
    expect(nn.weights[960 + 39 * 40 + 39]).toBe(1);
  });

  it("gives our outputs (f64) on 2 000 feature vectors", () => {
    let worst = 0;
    for (const f of featureSets(20, 2000)) worst = Math.max(worst, rel(expOut(nnForward(nn, vdbFrameMlpNnInputs(nn, f))), vdbFrameMlpForward(ours, f)));
    expect(worst).toBeLessThan(1e-12);
  });

  it("round trip ours → convention → ours is bit-identical (carriers and dead units folded away)", () => {
    const back = vdbFrameMlpFromNn(nn, { levels: ours.levels, output: ours.output });
    expect(back.layers).toEqual(ours.layers);
    expect(Array.from(back.weights)).toEqual(Array.from(ours.weights));
    for (const widths of [[20, 7, 13, 1], [14, 33, 1], [8, 5, 5, 5, 3]]) {
      const m = randomMlp(widths);
      const b = vdbFrameMlpFromNn(vdbFrameMlpToNn(m), { levels: m.levels });
      expect(b.layers).toEqual(m.layers);
      expect(Array.from(b.weights)).toEqual(Array.from(m.weights));
    }
  });

  it("reads a network trained in the convention (two constant ones, no carriers, padding inputs)", () => {
    // the NRC encoding's form: features, then zero padding, then ones (its inputs 62 and 63)
    const shape = { inputs: 24, width: 16, hidden: 2, outputs: 1 };
    const trained: VDBFrameMlpNn = { shape, weights: Float32Array.from({ length: 24 * 16 + 16 * 16 + 16 }, () => r() - 0.5), features: 20, ones: [22, 23] };
    const m = vdbFrameMlpFromNn(trained, { output: "linear" });
    expect(m.layers.map((l) => [l.inputs, l.outputs])).toEqual([[20, 16], [16, 16], [16, 1]]);
    let worst = 0;
    // the two ones' columns become ONE f32 bias (one rounding of their sum): absolute error on a linear output near 0
    for (const f of featureSets(20, 500)) {
      const want = nnForward(trained, vdbFrameMlpNnInputs(trained, f));
      worst = Math.max(worst, Math.abs(vdbFrameMlpForward(m, f) - want) / (1 + Math.abs(want)));
    }
    expect(worst).toBeLessThan(1e-6);
  });

  it("fail-before: a carrier or a bias in the wrong place moves the outputs", () => {
    const moved = (mutate: (w: Float32Array) => void) => {
      const w = nn.weights.slice();
      mutate(w);
      let over = 0;
      const sets = featureSets(20, 500);
      for (const f of sets) if (rel(expOut(nnForward({ shape: nn.shape, weights: w }, vdbFrameMlpNnInputs(nn, f))), vdbFrameMlpForward(ours, f)) > 1e-4) over++;
      return over / sets.length;
    };
    expect(moved((w) => (w[960 + 39 * 40 + 39] = 0))).toBeGreaterThan(0.5); // the layer-1 carrier dropped
    expect(moved((w) => { // layer 1's biases read from unit 38 (dead) instead of the carrier
      for (let j = 0; j < 32; j++) [w[960 + j * 40 + 38], w[960 + j * 40 + 39]] = [w[960 + j * 40 + 39]!, 0];
    })).toBeGreaterThan(0.5);
  });

  it("rejects networks the convention cannot hold", () => {
    const linearHidden = { ...ours, layers: ours.layers.map((l) => ({ ...l, activation: "linear" as const })) };
    expect(() => vdbFrameMlpToNn(linearHidden)).toThrow(VDBToolsError);
    expect(() => vdbFrameMlpFromNn({ ...nn, weights: nn.weights.subarray(1) })).toThrow(VDBToolsError);
    expect(() => vdbFrameMlpFromNn({ ...nn, ones: [3] })).toThrow(VDBToolsError);
    expect(() => vdbFrameMlpNnInputs(nn, [1, 2, 3])).toThrow(VDBToolsError);
  });

  // The sibling's own code, when a pmndrs/pathtracer checkout sits beside this repo (read-only; skipped otherwise).
  const here = path.dirname(fileURLToPath(import.meta.url));
  const candidates = [process.env.PMNDRS_PATHTRACER_DIR, path.resolve(here, "../../../../../pmndrs-pathtracer"), path.resolve(here, "../../../../../../../../pmndrs-pathtracer")].filter((p): p is string => !!p);
  const mlpSrc = candidates.map((d) => path.join(d, "packages/mlp/src")).find((d) => existsSync(path.join(d, "oracle.ts")));
  it.skipIf(!mlpSrc)("@pathtracer/mlp's oracle and WGSL generator take the network as is", async () => {
    type Shape = VDBFrameMlpNn["shape"];
    const oracle = (await import(/* @vite-ignore */ pathToFileURL(path.join(mlpSrc!, "oracle.ts")).href)) as { forward: (s: Shape, w: ArrayLike<number>, x: ArrayLike<number>) => { output: Float64Array } };
    const config = (await import(/* @vite-ignore */ pathToFileURL(path.join(mlpSrc!, "config.ts")).href)) as { validateShape: (s: Shape) => void; weightCount: (s: Shape) => number };
    const wgsl = (await import(/* @vite-ignore */ pathToFileURL(path.join(mlpSrc!, "wgsl.ts")).href)) as { inferWgsl: (o: { shape: Shape; precision: "f32" | "f16" }, hooks: string) => string; BUFFER_HOOKS_INFER: string };
    expect(() => config.validateShape(nn.shape)).not.toThrow();
    expect(config.weightCount(nn.shape)).toBe(nn.weights.length);
    let worst = 0;
    for (const f of featureSets(20, 2000)) worst = Math.max(worst, rel(expOut(oracle.forward(nn.shape, nn.weights, vdbFrameMlpNnInputs(nn, f)).output[0]!), vdbFrameMlpForward(ours, f)));
    expect(worst).toBeLessThan(1e-12);
    // its per-record inference kernel generates for our shape (not run here: it is a dispatch with its own bindings)
    const code = wgsl.inferWgsl({ shape: nn.shape, precision: "f32" }, wgsl.BUFFER_HOOKS_INFER);
    expect(code).toContain("const NN_IN = 24u;");
    expect(code).toContain("const NN_W = 40u;");
    // fail-before through the sibling's oracle: a transposed first layer is caught
    const t = nn.weights.slice();
    for (let o = 0; o < 24; o++) for (let i = 0; i < 24; i++) t[o * 24 + i] = nn.weights[i * 24 + o]!;
    let over = 0;
    for (const f of featureSets(20, 200)) if (rel(expOut(oracle.forward(nn.shape, t, vdbFrameMlpNnInputs(nn, f)).output[0]!), vdbFrameMlpForward(ours, f)) > 1e-4) over++;
    expect(over).toBeGreaterThan(100);
  });
});
