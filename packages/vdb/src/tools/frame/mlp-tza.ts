// Portions derived from pmndrs/denoiser (MIT, Copyright (c) 2024 Dennis Smolek), commit 3ad5e98:
//   packages/denoiser-core/src/tza.ts (`parseTZA`, `halfToFloat`: the OIDN TZA v2 weight container reader).
// Changes: a writer (the reader's inverse; binary16 through tools/nanovdb/half.ts), errors as VDBToolsError, and the
// map between a TZA tensor set and the core's generic MLP (`VDBFrameMlp`). Divergences: DIVERGES.md (K6.Na rows).

/**
 * K6.Na: the 'neural' network in the weight container of the pmndrs neural tooling (pmndrs/denoiser reads Intel OIDN's
 * `.tza` blobs with `parseTZA`, its Python converter with `tza.py`). A `.tza` is a list of named tensors (f32 'f' or
 * f16 'h', any shape); a network is stored as PyTorch `nn.Linear` names: `layers.<i>.weight` [outputs, inputs] (row-major,
 * layout "oi") and `layers.<i>.bias` [outputs] ("o"). Hidden layers are ReLU and the last is linear (the trainer's net);
 * the output transform and the feature levels are not in the file: `levels` = (inputs − 2) / 3, `output` defaults to
 * 'exp' (both options). Our weights stay a TypeScript data module too (the default network needs no fetch).
 */

import { VDBToolsError } from "../errors.js";
import { vdbFloatToHalf, vdbHalfToFloat } from "../nanovdb/half.js";
import type { VDBFrameMlp, VDBFrameMlpLayer } from "./mlp.js";

const MAGIC = 0x41d7;
const MAJOR_VERSION = 2;
/** Tensor data alignment of the writer (OIDN's blobs align tensors to 64 bytes). */
const ALIGN = 64;

/** One tensor of a `.tza` blob (f32 values; f16 tensors are widened on read). */
export interface VDBTzaTensor {
  data: Float32Array;
  shape: number[];
  /** The layout string (one char per dimension, e.g. "oi"); "" when the blob has none. */
  layout: string;
  /** The stored type: 'f' f32, 'h' f16. */
  type: "f" | "h";
}

const bad = (message: string): never => {
  throw new VDBToolsError("VDB_BAD_OPTION", `tza: ${message}`);
};

/** Reads a `.tza` v2 blob (the format `parseTZA` reads): tensor name → tensor, in file order. */
export function vdbTzaRead(bytes: ArrayBuffer | Uint8Array): Map<string, VDBTzaTensor> {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (buf.byteLength < 12) bad("too short for a header");
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (view.getUint16(0, true) !== MAGIC) bad("invalid or corrupted weights blob (magic)");
  const major = view.getUint8(2);
  if (major !== MAJOR_VERSION) bad(`unsupported weights blob version ${major}`);
  let offset = Number(view.getBigUint64(4, true));
  const need = (n: number) => {
    if (offset + n > buf.byteLength) bad("table runs past the end of the blob");
  };
  need(4);
  const count = view.getUint32(offset, true);
  offset += 4;
  const decoder = new TextDecoder();
  const tensors = new Map<string, VDBTzaTensor>();
  for (let t = 0; t < count; t++) {
    need(2);
    const nameLength = view.getUint16(offset, true);
    offset += 2;
    need(nameLength + 1);
    const name = decoder.decode(buf.subarray(offset, offset + nameLength));
    offset += nameLength;
    const dims = view.getUint8(offset);
    offset += 1;
    need(dims * 5 + 9);
    const shape: number[] = [];
    for (let d = 0; d < dims; d++) {
      shape.push(view.getUint32(offset, true));
      offset += 4;
    }
    const layout = decoder.decode(buf.subarray(offset, offset + dims));
    offset += dims;
    const type = String.fromCharCode(view.getUint8(offset));
    offset += 1;
    const at = Number(view.getBigUint64(offset, true));
    offset += 8;
    const n = shape.reduce((a, b) => a * b, 1);
    if (type !== "f" && type !== "h") bad(`tensor ${name}: invalid data type '${type}'`);
    const size = n * (type === "f" ? 4 : 2);
    if (at + size > buf.byteLength) bad(`tensor ${name} runs past the end of the blob`);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = type === "f" ? view.getFloat32(at + 4 * i, true) : vdbHalfToFloat(view.getUint16(at + 2 * i, true));
    tensors.set(name, { data, shape, layout, type: type as "f" | "h" });
  }
  return tensors;
}

/** Writes tensors as a `.tza` v2 blob (data 64-byte aligned after the 12-byte header, then the table). */
export function vdbTzaWrite(tensors: ReadonlyMap<string, Pick<VDBTzaTensor, "data" | "shape"> & Partial<Pick<VDBTzaTensor, "layout" | "type">>>): Uint8Array {
  const encoder = new TextEncoder();
  const entries = [...tensors].map(([name, t]) => {
    const type = t.type ?? "f";
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (n !== t.data.length) bad(`tensor ${name}: shape [${t.shape.join(", ")}] holds ${n} values, data has ${t.data.length}`);
    const layout = t.layout ?? "";
    if (layout && layout.length !== t.shape.length) bad(`tensor ${name}: layout "${layout}" for ${t.shape.length} dimensions`);
    return { name: encoder.encode(name), shape: t.shape, layout: encoder.encode(layout.padEnd(t.shape.length, "x")), type, data: t.data, bytes: n * (type === "f" ? 4 : 2) };
  });
  let at = 12;
  const placed = entries.map((e) => {
    at = Math.ceil(at / ALIGN) * ALIGN;
    const offset = at;
    at += e.bytes;
    return offset;
  });
  const table = at;
  const tableBytes = 4 + entries.reduce((s, e) => s + 2 + e.name.length + 1 + 5 * e.shape.length + 1 + 8, 0);
  const out = new Uint8Array(table + tableBytes);
  const view = new DataView(out.buffer);
  view.setUint16(0, MAGIC, true);
  view.setUint8(2, MAJOR_VERSION);
  view.setUint8(3, 0);
  view.setBigUint64(4, BigInt(table), true);
  entries.forEach((e, i) => {
    const o = placed[i]!;
    for (let k = 0; k < e.data.length; k++) {
      if (e.type === "f") view.setFloat32(o + 4 * k, e.data[k]!, true);
      else view.setUint16(o + 2 * k, vdbFloatToHalf(e.data[k]!), true);
    }
  });
  let p = table;
  view.setUint32(p, entries.length, true);
  p += 4;
  entries.forEach((e, i) => {
    view.setUint16(p, e.name.length, true);
    p += 2;
    out.set(e.name, p);
    p += e.name.length;
    view.setUint8(p, e.shape.length);
    p += 1;
    for (const d of e.shape) {
      view.setUint32(p, d, true);
      p += 4;
    }
    out.set(e.layout, p);
    p += e.shape.length;
    view.setUint8(p, e.type.charCodeAt(0));
    p += 1;
    view.setBigUint64(p, BigInt(placed[i]!), true);
    p += 8;
  });
  return out;
}

/**
 * K6.Na: a network as `.tza` tensors (`layers.<i>.weight` [outputs, inputs], `layers.<i>.bias` [outputs]). `type` 'h'
 * stores binary16 (half the bytes; the weights round to f16, which changes the network: an A/B, not a format detail).
 * Only networks whose hidden layers are ReLU and whose last layer is linear map onto the convention.
 */
export function vdbFrameMlpToTza(mlp: VDBFrameMlp, o: { type?: "f" | "h" } = {}): Uint8Array {
  const tensors = new Map<string, Pick<VDBTzaTensor, "data" | "shape" | "layout" | "type">>();
  let at = 0;
  mlp.layers.forEach((l, i) => {
    const wanted = i === mlp.layers.length - 1 ? "linear" : "relu";
    if (l.activation !== wanted) bad(`layer ${i} is '${l.activation}'; the .tza convention has ReLU hidden layers and a linear last layer`);
    tensors.set(`layers.${i}.weight`, { data: mlp.weights.slice(at, at + l.outputs * l.inputs), shape: [l.outputs, l.inputs], layout: "oi", type: o.type ?? "f" });
    at += l.outputs * l.inputs;
    tensors.set(`layers.${i}.bias`, { data: mlp.weights.slice(at, at + l.outputs), shape: [l.outputs], layout: "o", type: o.type ?? "f" });
    at += l.outputs;
  });
  return vdbTzaWrite(tensors);
}

/** K6.Na: the network of a `.tza` blob written by {@link vdbFrameMlpToTza} (or any `layers.<i>.weight/bias` set). */
export function vdbFrameMlpFromTza(bytes: ArrayBuffer | Uint8Array, o: { name?: string; output?: VDBFrameMlp["output"]; levels?: number } = {}): VDBFrameMlp {
  const tensors = vdbTzaRead(bytes);
  const layers: VDBFrameMlpLayer[] = [];
  const parts: Float32Array[] = [];
  for (let i = 0; tensors.has(`layers.${i}.weight`); i++) {
    const w = tensors.get(`layers.${i}.weight`)!;
    const b = tensors.get(`layers.${i}.bias`) ?? bad(`layers.${i}.bias is missing`);
    if (w.shape.length !== 2) bad(`layers.${i}.weight has ${w.shape.length} dimensions (want [outputs, inputs])`);
    const [outputs, inputs] = w.shape as [number, number];
    if (b.shape.length !== 1 || b.shape[0] !== outputs) bad(`layers.${i}.bias has shape [${b.shape.join(", ")}] (want [${outputs}])`);
    layers.push({ inputs, outputs, activation: "relu" });
    parts.push(w.data, b.data);
  }
  if (layers.length === 0) bad("no layers.0.weight tensor");
  layers[layers.length - 1]!.activation = "linear";
  const weights = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
  let at = 0;
  for (const p of parts) {
    weights.set(p, at);
    at += p.length;
  }
  const levels = o.levels ?? (layers[0]!.inputs - 2) / 3;
  return { name: o.name ?? "tza", levels, layers, output: o.output ?? "exp", weights };
}
