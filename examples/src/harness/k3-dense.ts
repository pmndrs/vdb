/**
 * K3.0 harness (R1 §6 Step 0), the core half. K6.11 (D35): the three `Data3DTexture` upload and the TSL march through
 * `vdbVolumeNodes` are deleted with the TSL half; what stays is the probe of the core dense source.
 *
 * Reads a `.wvdb` (default: the committed CC0 `cloud_04_variant_0000.wvdb`; `?file=<url>` for another), builds the dense
 * volume his renderer would upload (`wvdbDenseVolume`), and runs a raw-WebGPU compute kernel that calls the core
 * `vdbDenseSource`'s generated `vdbSrcDensity` (x `vdbSrcMask`, + `vdbSrcEmission`) at probe points and LODs; the values
 * are compared with the CPU twin `sampleWvdbDenseTexture`.
 *
 * Results go to `window.__VDB_K3_DENSE__` for e2e/k3-dense.spec.ts. Query: `?mips=0` (no mip atlas), `?lod=`, `?file=`.
 */
import { vdbDenseSource } from "@pmndrs/vdb";
import { readWvdb, sampleWvdbDenseTexture, wvdbDenseVolume } from "../../../packages/vdb/src/tools/internal.js";
import defaultWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

interface ProbeStats {
  count: number;
  maxAbs: number;
  meanAbs: number;
  over: number;
  tolerance: number;
  lodLevels: number;
  texels: number[];
  firstBad?: { uvw: number[]; lod: number; gpu: number; cpu: number };
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  file?: string;
  volume?: { dimensions: number[]; lod: number; meanDensity: number; maxDensity: number; activeTexels: number; mipLevels: number; statsSource: string };
  probes?: ProbeStats;
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_DENSE__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const MIPS = params.get("mips") !== "0";
const FILE = params.get("file") ?? defaultWvdbUrl;

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, uncaptured: [] };
window.__VDB_K3_DENSE__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

/** Deterministic xorshift in [0, 1). */
function rng(seed: number): () => number {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x / 4294967296;
  };
}

async function main(): Promise<void> {
  root.textContent = "";
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  // --- the dense volume (his bytes) -------------------------------------------------------
  const response = await fetch(FILE);
  if (!response.ok) throw new Error(`fetch ${FILE}: ${response.status}`);
  const file = readWvdb(new Uint8Array(await response.arrayBuffer()));
  const lodParam = params.get("lod");
  const volume = wvdbDenseVolume(file, { mipmaps: MIPS, ...(lodParam !== null ? { lod: Number(lodParam) } : {}) });
  result.file = FILE;
  result.volume = {
    dimensions: [...volume.dimensions],
    lod: volume.lod,
    meanDensity: volume.meanDensity,
    maxDensity: volume.maxDensity,
    activeTexels: volume.activeTexels,
    mipLevels: volume.density.mip?.levels ?? 0,
    statsSource: volume.densityStatsSource,
  };
  log(`${FILE.split("/").pop()}: LOD ${volume.lod}, ${volume.dimensions.join("x")} texels, mean ${volume.meanDensity.toFixed(4)}, ${result.volume.mipLevels} mip levels`);

  // --- core: vdbDenseSource WGSL vs the CPU twin --------------------------------------------
  result.probes = await probeCore(device, volume);
  log(
    `core vdbDenseSource: ${result.probes.count} probes, max |GPU - CPU twin| ${result.probes.maxAbs.toExponential(2)} (tolerance ${result.probes.tolerance}), ${result.probes.over} over`,
    result.probes.over === 0 ? "ok" : "fail",
  );
}

async function probeCore(device: GPUDevice, volume: ReturnType<typeof wvdbDenseVolume>): Promise<ProbeStats> {
  const src = vdbDenseSource(device, volume, { label: "k3 probe" });
  const slot = { group: 0, firstBinding: 0 };
  const code = `${src.wgsl({ slot, profile: "march" })}
@group(1) @binding(0) var<storage, read> probes: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outValues: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = arrayLength(&probes);
  if (id.x == 0u) {
    let t = vdbSrcTexels(0u);
    outValues[n] = t.x;
    outValues[n + 1u] = t.y;
    outValues[n + 2u] = t.z;
    outValues[n + 3u] = vdbSrcLodLevels(0u);
  }
  if (id.x >= n) {
    return;
  }
  let p = probes[id.x];
  outValues[id.x] = vdbSrcDensity(0u, p.xyz, p.w) * vdbSrcMask(0u, p.xyz, p.w) + vdbSrcEmission(0u, p.xyz, p.w);
}`;
  const random = rng(0x6b33);
  const lods = [0, 0, 0.25, 0.5, 1, 1.5, 2, 2.75, 4, 20];
  const N = 2048;
  const probes = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    for (let a = 0; a < 3; a++) probes[i * 4 + a] = -0.05 + 1.1 * random();
    probes[i * 4 + 3] = lods[i % lods.length]!;
  }
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code, label: "k3 probe" });
  const layout0 = device.createBindGroupLayout({ entries: src.layoutEntries(slot, 0x4) });
  const layout1 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: 0x4, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: 0x4, buffer: { type: "storage" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] }),
    compute: { module, entryPoint: "main" },
  });
  const probeBuf = device.createBuffer({ size: probes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(probeBuf, 0, probes);
  const outBytes = (N + 4) * 4;
  const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: layout0, entries: src.bindEntries(slot) }));
  pass.setBindGroup(1, device.createBindGroup({ layout: layout1, entries: [{ binding: 0, resource: { buffer: probeBuf } }, { binding: 1, resource: { buffer: outBuf } }] }));
  pass.dispatchWorkgroups(Math.ceil(N / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
  device.queue.submit([encoder.finish()]);
  const err = await device.popErrorScope();
  if (err) throw new Error(`probe kernel: ${err.message}`);
  await readBuf.mapAsync(GPUMapMode.READ);
  const gpu = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  // Hardware filtering quantizes the weights (8 fractional bits is common); allow one 8-bit step plus that.
  const tolerance = 2 / 255;
  let maxAbs = 0;
  let sum = 0;
  let over = 0;
  let firstBad: ProbeStats["firstBad"];
  for (let i = 0; i < N; i++) {
    const uvw = [probes[i * 4]!, probes[i * 4 + 1]!, probes[i * 4 + 2]!];
    const lod = probes[i * 4 + 3]!;
    const cpu = sampleWvdbDenseTexture(volume.density, uvw, lod);
    const d = Math.abs(gpu[i]! - cpu);
    maxAbs = Math.max(maxAbs, d);
    sum += d;
    if (d > tolerance) {
      over++;
      firstBad ??= { uvw, lod, gpu: gpu[i]!, cpu };
    }
  }
  src.dispose();
  [probeBuf, outBuf, readBuf].forEach((b) => b.destroy());
  return {
    count: N,
    maxAbs,
    meanAbs: sum / N,
    over,
    tolerance,
    lodLevels: gpu[N + 3]!,
    texels: [gpu[N]!, gpu[N + 1]!, gpu[N + 2]!],
    ...(firstBad ? { firstBad } : {}),
  };
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
