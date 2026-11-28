// K6.Nc (pmndrs/vdb#3): our code (MIT). It writes the weight convention of pmndrs/pathtracer's `@pathtracer/mlp`
// (research track N, MIT, Copyright (c) 2026 Dennis Smolek; read at 06488fa, same code as 0e4435b): `config.ts` (layer
// shapes, `layerOffset`) and `oracle.ts` (`forward`: no biases, hidden ReLU, last layer linear). No code is taken from
// it. Divergences: DIVERGES.md (K6.Nc row).

/**
 * K6.Nc: the 'neural' network in the weight convention of the pmndrs neural radiance cache (`@pathtracer/mlp`, the
 * network pmndrs/pathtracer's NRC and pmndrs/gi's neural tail train on the GPU). That convention has **no biases**: a
 * network is `{ inputs, width, hidden, outputs }` (inputs and width multiples of 8, every hidden layer `width` wide),
 * its weights one flat f32 array, layer by layer, each row-major [out][in]; hidden layers are ReLU, the last is linear,
 * and the input encoding carries constant-one inputs that act as the bias (`NrcCache.loadWeights` takes such an array).
 *
 * Ours has biases, so {@link vdbFrameMlpToNn} embeds it exactly: input `features` (our 3 L + 2 features come first)
 * holds a constant 1, and the last unit of every hidden layer is a **carrier** fed only by that one (weight 1, so it
 * stays 1 through each ReLU); each layer's biases become its weights from the one or the carrier. Our 20 → 32 → 32 → 1
 * becomes 24 → 2 × 40 → 1 (2 600 weights; ours 1 761 with biases). {@link vdbFrameMlpFromNn} is the inverse for any
 * network in that convention: the constant-one columns of the first layer become its biases, then every hidden unit
 * with no input weights (a carrier, or a dead unit) is a constant and is folded into the next layer's biases and
 * removed. Ours → convention → ours gives back the same weights bit for bit. The output transform (our 'exp') is not
 * part of the convention: it stays an option, as in the `.tza` map (`mlp-tza.ts`).
 */

import { VDBToolsError } from "../errors.js";
import { vdbFrameMlpCheck, vdbFrameMlpOffsets, type VDBFrameMlp, type VDBFrameMlpLayer } from "./mlp.js";

/** A network in `@pathtracer/mlp`'s convention (its `MlpShape` and flat weights), with where our inputs sit. */
export interface VDBFrameMlpNn {
  /** `@pathtracer/mlp`'s `MlpShape`: `hidden` ReLU layers of `width`, then a linear layer to `outputs`. */
  shape: { inputs: number; width: number; hidden: number; outputs: number };
  /** Layer by layer, row-major [out][in], no biases (`layerOffset` order). */
  weights: Float32Array;
  /** Inputs 0 … features − 1 are our feature vector (3 L + 2 values). */
  features: number;
  /** Inputs that hold a constant 1 (the bias inputs); every other input past `features` is 0. */
  ones: number[];
}

const bad = (message: string): never => {
  throw new VDBToolsError("VDB_BAD_OPTION", `mlp (nn convention): ${message}`);
};

const ceil8 = (n: number) => Math.ceil(n / 8) * 8;

/** The weight count of a network in the convention (`@pathtracer/mlp` `weightCount`). */
export function vdbFrameMlpNnWeightCount(shape: VDBFrameMlpNn["shape"]): number {
  return shape.inputs * shape.width + (shape.hidden - 1) * shape.width * shape.width + shape.width * shape.outputs;
}

/**
 * Our network in `@pathtracer/mlp`'s bias-free convention (exact; module comment). Needs ReLU hidden layers and a
 * linear last layer (the convention has no other activations). The output transform is not carried.
 */
export function vdbFrameMlpToNn(mlp: VDBFrameMlp): VDBFrameMlpNn {
  const reason = vdbFrameMlpCheck(mlp);
  if (reason) bad(reason);
  const layers = mlp.layers;
  if (layers.length < 2) bad("needs at least one hidden layer");
  layers.forEach((l, i) => {
    const want = i === layers.length - 1 ? "linear" : "relu";
    if (l.activation !== want) bad(`layer ${i} is ${l.activation}; the convention has ReLU hidden layers and a linear last layer`);
  });
  const features = layers[0]!.inputs;
  const hidden = layers.length - 1;
  const inputs = ceil8(features + 1);
  const width = ceil8(Math.max(...layers.slice(0, hidden).map((l) => l.outputs)) + 1);
  const outputs = layers[hidden]!.outputs;
  if (outputs > width) bad(`${outputs} outputs exceed the hidden width ${width}`);
  const shape = { inputs, width, hidden, outputs };
  const carrier = width - 1;
  const w = new Float32Array(vdbFrameMlpNnWeightCount(shape));
  const offsets = vdbFrameMlpOffsets(mlp);
  let at = 0;
  for (let li = 0; li <= hidden; li++) {
    const l = layers[li]!;
    const src = offsets[li]!;
    const nIn = li === 0 ? inputs : width;
    const nOut = li === hidden ? outputs : width;
    // the bias input: our constant one (layer 0) or the previous layer's carrier
    const one = li === 0 ? features : carrier;
    for (let j = 0; j < l.outputs; j++) {
      for (let i = 0; i < l.inputs; i++) w[at + j * nIn + i] = mlp.weights[src + j * l.inputs + i]!;
      w[at + j * nIn + one] = mlp.weights[src + l.outputs * l.inputs + j]!;
    }
    if (li < hidden) w[at + carrier * nIn + one] = 1;
    at += nIn * nOut;
  }
  return { shape, weights: w, features, ones: [features] };
}

/** The convention's input vector for our features: the features, 1 at each `ones` input, 0 elsewhere. */
export function vdbFrameMlpNnInputs(nn: Pick<VDBFrameMlpNn, "shape" | "features" | "ones">, features: ArrayLike<number>): Float32Array {
  if (features.length !== nn.features) bad(`expected ${nn.features} features, got ${features.length}`);
  const x = new Float32Array(nn.shape.inputs);
  for (let i = 0; i < nn.features; i++) x[i] = features[i]!;
  for (const c of nn.ones) x[c] = 1;
  return x;
}

/**
 * A network in `@pathtracer/mlp`'s convention as ours (exact): the first layer's `ones` columns summed into its
 * biases (other inputs past `features` are zero and dropped), then every hidden unit with no input weights folded into
 * the next layer's biases as the constant relu(its bias) and removed (carriers and dead units).
 */
export function vdbFrameMlpFromNn(nn: VDBFrameMlpNn, o: { name?: string; output?: VDBFrameMlp["output"]; levels?: number } = {}): VDBFrameMlp {
  const { shape, features, ones } = nn;
  for (const [k, v] of Object.entries(shape)) if (!Number.isInteger(v) || v < 1) bad(`shape.${k} must be a positive integer (got ${v})`);
  if (nn.weights.length !== vdbFrameMlpNnWeightCount(shape)) bad(`expected ${vdbFrameMlpNnWeightCount(shape)} weights, got ${nn.weights.length}`);
  if (!(features >= 1 && features <= shape.inputs)) bad(`features must be 1 to ${shape.inputs} (got ${features})`);
  for (const c of ones) if (!(Number.isInteger(c) && c >= features && c < shape.inputs)) bad(`a constant-one input must be past the features (got ${c})`);
  // dense f64 layers: W [out][in] and b [out]
  const W: Float64Array[][] = [];
  const B: Float64Array[] = [];
  let at = 0;
  for (let l = 0; l <= shape.hidden; l++) {
    const nIn = l === 0 ? shape.inputs : shape.width;
    const nOut = l === shape.hidden ? shape.outputs : shape.width;
    const rows: Float64Array[] = [];
    const b = new Float64Array(nOut);
    for (let j = 0; j < nOut; j++) {
      const row = Float64Array.from(nn.weights.subarray(at + j * nIn, at + (j + 1) * nIn));
      if (l === 0) {
        for (const c of ones) b[j]! += row[c]!;
        rows.push(row.slice(0, features));
      } else rows.push(row);
    }
    W.push(rows);
    B.push(b);
    at += nIn * nOut;
  }
  // fold the constant hidden units (no input weights) into the next layer, last unit first so indices stay valid
  for (let l = 0; l < shape.hidden; l++) {
    for (let j = W[l]!.length - 1; j >= 0; j--) {
      if (W[l]![j]!.some((v) => v !== 0)) continue;
      const k = Math.max(B[l]![j]!, 0);
      const next = W[l + 1]!;
      next.forEach((row, r) => {
        if (k !== 0) B[l + 1]![r]! += row[j]! * k;
        W[l + 1]![r] = Float64Array.from([...row.subarray(0, j), ...row.subarray(j + 1)]);
      });
      W[l]!.splice(j, 1);
      B[l] = Float64Array.from([...B[l]!.subarray(0, j), ...B[l]!.subarray(j + 1)]);
    }
    if (W[l]!.length === 0) bad(`hidden layer ${l} has no unit that depends on the inputs`);
  }
  const layers: VDBFrameMlpLayer[] = W.map((rows, l) => ({ inputs: l === 0 ? features : W[l - 1]!.length, outputs: rows.length, activation: l === shape.hidden ? "linear" : "relu" }));
  const weights: number[] = [];
  W.forEach((rows, l) => {
    for (const row of rows) weights.push(...row);
    weights.push(...B[l]!);
  });
  const mlp: VDBFrameMlp = { name: o.name ?? "nn", levels: o.levels ?? (features - 2) / 3, layers, output: o.output ?? "exp", weights: Float32Array.from(weights) };
  const reason = vdbFrameMlpCheck(mlp);
  if (reason) bad(reason);
  return mlp;
}
