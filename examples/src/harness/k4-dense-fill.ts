/**
 * K4.22 harness: the core dense source's GPU fill of a partially streamed `.wvdb` LOD (raw WebGPU, no three). Seeded
 * `.wvdb` files stream through `WvdbStreamReader({ partialLods: true })` one byte per push (a push completes at most
 * one chunk, so each event is a snapshot); every event becomes `wvdbDensePartialVolume(...)` and goes into ONE
 * `vdbDenseSource` by `replace` (the GPU path: `vdbDenseFillCs` + `vdbDenseMipCs`, own submit). After each replace the
 * source's textures (level 0 and the mip atlas) are read back and compared byte for byte with the CPU twin
 * (`wvdbDenseResolve`), the completed LODs included (= `wvdbDenseVolume`).
 *
 * Cases: halving sizes (his bake), the same with his mips (`fillMips: 'box'`: every level from the upsampled texels),
 * odd sizes, an RGBA8 albedo grid beside the density, a larger grid, a dense layout without mipmaps. `?mutate=wgsl-*`
 * patches the kernels (fail-before): the upsample's rounding, the mip's rounding, the pending test, the coarse level 1. Results: `window.__VDB_K4_DENSE_FILL__` (e2e/k4-dense-fill.spec.ts).
 */
import { installWgslPatch, type VDBWgslPatches } from "./wgsl-patch.js";
import { vdbDenseSource } from "../../../packages/vdb/src/core/index.js";
import { vdbDenseSourceTextures } from "../../../packages/vdb/src/core/internal.js";
import {
  WvdbStreamReader,
  writeWvdb,
  wvdbDensePartialVolume,
  wvdbDenseResolve,
  type WvdbDenseTexture,
  type WvdbDenseVolume,
  type WvdbGridInput,
  type WvdbVec3,
} from "../../../packages/vdb/src/tools/internal.js";

interface CaseResult {
  name: string;
  dims: number[];
  /** Volumes replaced into the source (partial and complete). */
  events: number;
  /** Of them, partial ones whose density had pending bricks (the GPU fill ran). */
  filled: number;
  /** Texture bytes compared (level 0 and atlas, every channel, every event). */
  bytes: number;
  /** Bytes that differ from the twin, and the largest difference. */
  mismatches: number;
  maxDiff: number;
  /** The first mismatch. */
  first?: { event: number; channel: string; which: "map" | "mip"; index: number; gpu: number; cpu: number };
  /** The source's own count of GPU and CPU fills (all GPU here). */
  fills: { gpu: number; cpu: number };
  /** Texels the upsample changed against the background fill (the probe exercises it). */
  upsampledTexels: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  mutate: string;
  cases: CaseResult[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K4_DENSE_FILL__?: Result;
  }
}

const MUTATE = new URLSearchParams(location.search).get("mutate") ?? "none";
const WGSL_PATCHES: VDBWgslPatches = {
  "wgsl-fill-round": ["c1 * az.z + 0x800000u) >> 24u", "c1 * az.z) >> 24u"],
  "wgsl-mip-round": ["(sum + 4u) >> 3u", "sum >> 3u"],
  "wgsl-fill-pending": ["if pending == 0u {", "if pending != 2u {"],
  "wgsl-mip-coarse": ["if p.coarseMip != 0u {", "if p.coarseMip == 7u {"],
};

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, mutate: MUTATE, cases: [], uncaptured: [] };
window.__VDB_K4_DENSE_FILL__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

/** Mulberry32. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A seeded cloud: soft blobs times noise, values up to ~1.3 (some saturate), empty corners (sparse bricks). */
function cloud(dims: WvdbVec3, seed: number): Float32Array {
  const r = rng(seed);
  const [X, Y, Z] = dims;
  const blobs = Array.from({ length: 4 }, () => ({ c: [r() * X, r() * Y, r() * Z], radius: (0.2 + r() * 0.25) * Math.min(X, Y, Z) }));
  const v = new Float32Array(X * Y * Z);
  for (let z = 0; z < Z; z++)
    for (let y = 0; y < Y; y++)
      for (let x = 0; x < X; x++) {
        let d = 0;
        for (const b of blobs) d = Math.max(d, 1 - Math.hypot(x - b.c[0]!, y - b.c[1]!, z - b.c[2]!) / b.radius);
        v[x + X * (y + Y * z)] = d > 0 ? d * 1.3 * (0.6 + 0.4 * r()) : 0;
      }
  return v;
}

interface Spec {
  name: string;
  dims: WvdbVec3;
  layout: "sparse" | "dense";
  pageBricks: number;
  albedo?: boolean;
  mipmaps: boolean;
  fillMips?: "coarse" | "box";
  seed: number;
}

function bake(spec: Spec): Uint8Array {
  const grids: WvdbGridInput[] = [{ name: "density", class: "fog volume", background: 0, bbox: [[-1, -1, -1], [1, 1, 1]], finest: { dimensions: spec.dims, values: cloud(spec.dims, spec.seed) } }];
  if (spec.albedo) {
    const n = spec.dims[0] * spec.dims[1] * spec.dims[2];
    const a = cloud(spec.dims, spec.seed + 100);
    const b = cloud(spec.dims, spec.seed + 200);
    const c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      c[i] = a[i]!;
      c[n + i] = b[i]!;
      c[2 * n + i] = 0.5 * (a[i]! + b[i]!);
    }
    grids.push({ name: "Cd", channels: 3, background: [0.25, 0.5, 0.75], bbox: [[-1, -1, -1], [1, 1, 1]], finest: { dimensions: spec.dims, values: c } });
  }
  return writeWvdb(grids, { layout: spec.layout, pageBricks: spec.pageBricks, lods: 3 }).bytes;
}

/** Reads a 3D texture back, tight (rows top first as stored, x fastest). */
async function readTexture(device: GPUDevice, texture: GPUTexture, texelBytes: number): Promise<Uint8Array> {
  const [w, h, d] = [texture.width, texture.height, texture.depthOrArrayLayers];
  const row = Math.ceil((w * texelBytes) / 256) * 256;
  const buffer = device.createBuffer({ size: row * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: h }, [w, h, d]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buffer.getMappedRange());
  const out = new Uint8Array(w * h * d * texelBytes);
  for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) out.set(padded.subarray((z * h + y) * row, (z * h + y) * row + w * texelBytes), (z * h + y) * w * texelBytes);
  buffer.unmap();
  buffer.destroy();
  return out;
}

async function runCase(device: GPUDevice, spec: Spec): Promise<CaseResult> {
  const bytes = bake(spec);
  const reader = new WvdbStreamReader({ partialLods: true });
  const volumes: WvdbDenseVolume[] = [];
  for (let i = 0; i < bytes.length; i++) {
    for (const e of reader.push(bytes.subarray(i, i + 1))) {
      volumes.push(wvdbDensePartialVolume(reader.file!, e, { mipmaps: spec.mipmaps, ...(spec.fillMips ? { fillMips: spec.fillMips } : {}), ...(spec.albedo ? { albedo: "Cd" } : {}) }));
    }
  }
  reader.end();
  const res: CaseResult = { name: spec.name, dims: [...spec.dims], events: volumes.length, filled: 0, bytes: 0, mismatches: 0, maxDiff: 0, fills: { gpu: 0, cpu: 0 }, upsampledTexels: 0 };
  const source = vdbDenseSource(device, volumes[0]!, { label: `k4.22 ${spec.name}` });
  const compare = (event: number, channel: string, which: "map" | "mip", gpu: Uint8Array, cpu: Uint8Array) => {
    if (gpu.length !== cpu.length) throw new Error(`${spec.name} event ${event} ${channel} ${which}: ${gpu.length} bytes read back, twin ${cpu.length}`);
    res.bytes += gpu.length;
    for (let i = 0; i < gpu.length; i++) {
      const d = Math.abs(gpu[i]! - cpu[i]!);
      if (d === 0) continue;
      res.mismatches++;
      res.maxDiff = Math.max(res.maxDiff, d);
      res.first ??= { event, channel, which, index: i, gpu: gpu[i]!, cpu: cpu[i]! };
    }
  };
  for (const [k, v] of volumes.entries()) {
    if (k > 0) source.replace(v);
    if (v.density.fill) res.filled++;
    const twin = wvdbDenseResolve(v);
    const textures = vdbDenseSourceTextures(source)!;
    for (const [name, tex] of [["density", twin.density], ["albedo", twin.albedo]] as [string, WvdbDenseTexture | null][]) {
      if (!tex) continue;
      const up = textures[name as "density" | "albedo"]!;
      compare(k, name, "map", await readTexture(device, up.map, tex.texelBytes), tex.data);
      if (tex.mip) compare(k, name, "mip", await readTexture(device, up.mip!, tex.texelBytes), tex.mip.data);
      const holes = (name === "density" ? v.density : v.albedo)!.data;
      for (let i = 0; i < holes.length; i++) if (holes[i] !== tex.data[i]) res.upsampledTexels++;
    }
  }
  res.fills = { ...source.fills };
  source.dispose();
  return res;
}

const CASES: Spec[] = [
  { name: "halving", dims: [48, 40, 32], layout: "sparse", pageBricks: 3, mipmaps: true, seed: 1 },
  { name: "halving-box-mips", dims: [48, 40, 32], layout: "sparse", pageBricks: 3, mipmaps: true, fillMips: "box", seed: 1 },
  { name: "odd", dims: [37, 29, 21], layout: "sparse", pageBricks: 2, mipmaps: true, seed: 2 },
  { name: "albedo", dims: [32, 24, 40], layout: "sparse", pageBricks: 4, albedo: true, mipmaps: true, seed: 3 },
  { name: "large", dims: [128, 96, 80], layout: "sparse", pageBricks: 64, mipmaps: true, seed: 4 },
  { name: "dense-nomip", dims: [40, 16, 24], layout: "dense", pageBricks: 5, mipmaps: false, seed: 5 },
];

let checkWgslPatch = (): void => {};

async function main(): Promise<void> {
  root.textContent = "";
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  checkWgslPatch = installWgslPatch(device, WGSL_PATCHES, MUTATE);
  for (const spec of CASES) {
    const r = await runCase(device, spec);
    result.cases.push(r);
    log(
      `${r.name} ${r.dims.join("x")}: ${r.events} events (${r.filled} filled on the GPU: ${r.fills.gpu} fills, ${r.fills.cpu} on the CPU), ` +
        `${r.bytes} bytes vs the twin, ${r.mismatches} differ (max ${r.maxDiff}); ${r.upsampledTexels} upsampled texels`,
      r.mismatches === 0 ? "ok" : "fail",
    );
  }
}

main()
  .then(() => checkWgslPatch())
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
