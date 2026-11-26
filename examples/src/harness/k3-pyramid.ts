/**
 * K3.2a harness: the sparse source's coarse pyramid built on the GPU (core `vdbCreateBoxPyramid`)
 * against its CPU twin (tools `vdbBoxPyramid` + three's `DataUtils.toHalfFloat`, the K3.2 path),
 * texel for texel on the r16f textures, plus build times and bytes.
 *
 * Query:
 * - `?assets=` comma list (default `cloud04,cloud04@32,sphere,sphere8`): `cloud04` (the committed CC0
 *   `.wvdb` as a NanoVDB FLOAT grid), `bunny` (his `.wvdb`, `/fixtures/k3/`), `smoke`
 *   (`/fixtures/vdb-samples/smoke.vdb`, full resolution: fractional box weights), `sphere` / `sphere8`
 *   / `spherefpn` (the Float / Fp8 / FpN primitives: inactive tiles and quantized leaves), `spherels`
 *   (`/fixtures/vdb-samples/sphere.vdb`, a level set: inactive tiles at ±3, so blocks with no active
 *   voxel must read 0 as on the CPU), `wdas` (the
 *   Fp8 quarter cloud). `@N` sets `pyramidMax` (cloud04@32: L₀ = 2, a fractional z axis).
 * - `?repeat=N` (default 3): timed rebuilds per asset (median; GPU timestamps when the adapter has
 *   `timestamp-query`, else queue-synchronised wall time) and CPU builds (median).
 * - `?sequence=1`: replace the grid with another image (pool.replace), check `stale`, rebuild, compare
 *   with the CPU pyramid of the new image over the same index box.
 * - `?three=1`: the `vdbSparseSource` handle on cloud04 across a `pool.replace`: its content `version` moves and
 *   `refreshStats()` reads the new image back (K3.2e). K6.11 (D35): the three pyramid textures, their GPU-vs-CPU
 *   comparison and the TSL render that rebuilt them went with `vdbVolumeNodes`; the core source's own pyramids (build
 *   on creation, rebuild once per image change) are checked by e2e/k6-sparse.spec.ts.
 * Results: `window.__VDB_K3_PYRAMID__` (e2e/k3-pyramid.spec.ts, scripts/k3/time-hardware.mjs).
 */
import * as THREE from "three";
import { installWgslPatch, type VDBWgslPatches } from "./wgsl-patch.js";
import { WebGPURenderer } from "three/webgpu";
import { VDBGridPool, vdbCreateBoxPyramid, type VDBBoxPyramidGpu } from "@pmndrs/vdb";
import { VDBGridNode, vdbSparseSource } from "@pmndrs/vdb/three";
import { NvdbFile, buildFromVdb, parseVdb, readWvdb, vdbBoxPyramid, wvdbDenseToNvdb, wvdbDenseVolume } from "../../../packages/vdb/src/tools/internal.js";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

type Vec3 = [number, number, number];

interface Compare {
  texels: number;
  exact: number;
  ulp1: number;
  over: number;
  maxUlp: number;
  firstBad?: { index: number; gpu: number; cpu: number };
}

interface AssetResult {
  name: string;
  type: string;
  indexBox: { min: number[]; max: number[] };
  texels: number[];
  level: number;
  dimensions: number[];
  atlasDimensions: number[] | null;
  atlasLevels: number;
  base: Compare;
  atlas: Compare | null;
  gpuBytes: { textures: number; scratch: number };
  cpuBytesUploaded: number;
  /** Median GPU build ms (timestamps around the compute pass; copies excluded) or null. */
  gpuMs: number | null;
  /** Median wall ms of record + submit + onSubmittedWorkDone (queue-synchronised). */
  gpuWallMs: number;
  /** Median CPU ms: vdbBoxPyramid + the f16 conversion (the K3.2 path, upload excluded). */
  cpuMs: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  timestampQuery?: boolean;
  assets: AssetResult[];
  sequence?: { staleBefore: boolean; rebuilt: boolean; staleAfter: boolean; again: boolean; base: Compare; atlas: Compare | null };
  three?: {
    /** The source's content version before and after the replace (the 'frame' caster signature reads it). */
    versions: [number, number];
    /** K3.2e: the density stats before the replace, after it (read back from the pool), and a fresh source's on the new image. */
    stats: {
      before: { meanNonZero: number; max: number; activeVoxels: number };
      after: { meanNonZero: number; max: number; activeVoxels: number };
      expected: { meanNonZero: number; max: number; activeVoxels: number };
      updates: number;
    };
  };
  warnings: string[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_K3_PYRAMID__?: Result;
  }
}

const params = new URLSearchParams(location.search);
/**
 * K6.10b `?mutate=wgsl-*` (fail-before, GPU side; wgsl-patch.ts): the base box filter scaled by 3/4; the empty-block mask
 * off; the 2x2x2 level mean divided by 7; the pack writes each word's low texel twice.
 */
const MUTATE = params.get("mutate") ?? "none";
const WGSL_PATCHES: VDBWgslPatches = {
  "wgsl-pyr-base": ["sz += wz * sy;", "sz += wz * sy * 0.75;"],
  "wgsl-pyr-mask": ["if atomicLoad(&pyrFlags[pyrBlock(j)]) == 0u {", "if false {"],
  "wgsl-pyr-level": ["] = sum / 8.0;", "] = sum / 7.0;"],
  "wgsl-pyr-pack": ["= lo | (hi << 16u);", "= lo | (lo << 16u);"],
};
let checkWgslPatch = (): void => {};
const ASSETS = (params.get("assets") ?? "cloud04,cloud04@32,sphere,sphere8").split(",").filter(Boolean);
const REPEAT = Math.max(1, Number(params.get("repeat") ?? 3));
const SEQUENCE = params.get("sequence") === "1";
const THREE_CHECK = params.get("three") === "1";

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, assets: [], warnings: [], uncaptured: [] };
window.__VDB_K3_PYRAMID__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} (fixtures are git-ignored: pnpm fixtures)`);
  return r.arrayBuffer();
}

interface Asset {
  image: Uint32Array;
  indexBox?: { min: Vec3; max: Vec3 };
}

async function loadAsset(name: string): Promise<Asset> {
  if (name === "cloud04" || name === "bunny") {
    const url = name === "cloud04" ? cloudWvdbUrl : "/fixtures/k3/bunny_cloud.wvdb";
    const grid = wvdbDenseToNvdb(wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(url))), { mipmaps: false }));
    return { image: grid.image, indexBox: grid.textureIndexBox };
  }
  if (name === "smoke" || name === "spherels") {
    // spherels: the openvdb.org sphere level set (background 3, inactive interior tiles at -3): the active-block rule.
    const file = parseVdb(await fetchBytes(`/fixtures/vdb-samples/${name === "smoke" ? "smoke" : "sphere"}.vdb`));
    return { image: buildFromVdb(file.grids.find((g) => g.valueType === "float")!) };
  }
  const primitive: Record<string, string> = { sphere: "sphere_fog_float", sphere8: "sphere_fog_fp8", spherefpn: "sphere_fog_fpn", torus: "torus_fog_float" };
  if (primitive[name]) return { image: NvdbFile.fromArrayBuffer(await fetchBytes(`/fixtures/primitives/${primitive[name]}.nvdb`)).gridImage(0) };
  if (name === "wdas") return { image: NvdbFile.fromArrayBuffer(await fetchBytes("/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb")).gridImage(0) };
  throw new Error(`unknown asset ${name}`);
}

const toHalf = (data: Float32Array) => {
  const out = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = THREE.DataUtils.toHalfFloat(data[i]!);
  return out;
};

/** f16 bit patterns → ordered integers (ulp distance = difference). */
const ordered = (h: number) => (h & 0x8000 ? -(h & 0x7fff) : h);

function compare(gpu: Uint16Array, cpu: Uint16Array): Compare {
  if (gpu.length !== cpu.length) throw new Error(`compare: ${gpu.length} GPU texels vs ${cpu.length} CPU`);
  const c: Compare = { texels: cpu.length, exact: 0, ulp1: 0, over: 0, maxUlp: 0 };
  for (let i = 0; i < cpu.length; i++) {
    const d = Math.abs(ordered(gpu[i]!) - ordered(cpu[i]!));
    if (d === 0) c.exact++;
    else if (d === 1) c.ulp1++;
    else {
      c.over++;
      c.firstBad ??= { index: i, gpu: THREE.DataUtils.fromHalfFloat(gpu[i]!), cpu: THREE.DataUtils.fromHalfFloat(cpu[i]!) };
    }
    c.maxUlp = Math.max(c.maxUlp, d);
  }
  return c;
}

/** Reads an r16float 3D texture back (tight rows). */
async function readHalfTexture(device: GPUDevice, texture: GPUTexture): Promise<Uint16Array> {
  const [w, h, d] = [texture.width, texture.height, texture.depthOrArrayLayers];
  const bytesPerRow = Math.ceil((w * 2) / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * h * d, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: h }, [w, h, d]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Uint16Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  const out = new Uint16Array(w * h * d);
  const stride = bytesPerRow / 2;
  for (let r = 0; r < h * d; r++) out.set(raw.subarray(r * stride, r * stride + w), r * w);
  return out;
}

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)]! : NaN;
};

/** CPU twin: the K3.2 path (vdbBoxPyramid, then three's f16 conversion). */
function cpuPyramid(image: Uint32Array, indexBox: { min: Vec3; max: Vec3 }, pyramidMax: number | undefined) {
  const t0 = performance.now();
  const p = vdbBoxPyramid(image, { indexBox, ...(pyramidMax !== undefined ? { pyramidMax } : {}) });
  const base = toHalf(p.data);
  const atlas = p.atlas ? toHalf(p.atlas.data) : null;
  return { p, base, atlas, ms: performance.now() - t0 };
}

async function checkAgainstCpu(device: GPUDevice, pyramid: VDBBoxPyramidGpu, cpu: { base: Uint16Array; atlas: Uint16Array | null }) {
  const base = compare(await readHalfTexture(device, pyramid.base), cpu.base);
  const atlas = pyramid.atlas && cpu.atlas ? compare(await readHalfTexture(device, pyramid.atlas), cpu.atlas) : null;
  if (!!pyramid.atlas !== !!cpu.atlas) throw new Error("atlas presence differs between GPU and CPU");
  return { base, atlas };
}

async function main(): Promise<void> {
  root.textContent = "";
  const warn = console.warn.bind(console);
  console.warn = (...a: unknown[]) => {
    result.warnings.push(a.map(String).join(" "));
    warn(...a);
  };
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const timestamps = adapter.features.has("timestamp-query");
  result.timestampQuery = timestamps;
  const device = await adapter.requestDevice({ requiredFeatures: timestamps ? ["timestamp-query"] : [], requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  checkWgslPatch = installWgslPatch(device, WGSL_PATCHES, MUTATE);
  const querySet = timestamps ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const resolveBuffer = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const readBuffer = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;

  for (const spec of ASSETS) {
    const [name, maxText] = spec.split("@");
    const pyramidMax = maxText ? Number(maxText) : undefined;
    const asset = await loadAsset(name!);
    const pool = await VDBGridPool.create(device, { label: `k3-pyramid ${spec}` });
    const id = pool.add(asset.image);
    pool.flush();
    await pool.settled();
    const gridInfo = pool.info(id);
    const indexBox = asset.indexBox ?? { min: [...gridInfo.indexBBox.min] as Vec3, max: [...gridInfo.indexBBox.max] as Vec3 };

    const cpuRuns: number[] = [];
    let cpu = cpuPyramid(asset.image, indexBox, pyramidMax);
    cpuRuns.push(cpu.ms);
    for (let r = 1; r < REPEAT; r++) {
      cpu = cpuPyramid(asset.image, indexBox, pyramidMax);
      cpuRuns.push(cpu.ms);
    }

    const pyramid = vdbCreateBoxPyramid(device, pool, id, { indexBox, ...(pyramidMax !== undefined ? { pyramidMax } : {}) });
    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder();
    pyramid.record(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const scopeError = await device.popErrorScope();
    const pipelineError = await pyramid.error;
    if (scopeError || pipelineError) throw new Error(`vdbCreateBoxPyramid ${spec}: ${(scopeError ?? pipelineError)!.message}`);
    const { base, atlas } = await checkAgainstCpu(device, pyramid, cpu);

    // Timed rebuilds (the first build above was the warm-up: pipeline creation).
    const wall: number[] = [];
    const gpu: number[] = [];
    for (let r = 0; r < REPEAT; r++) {
      const e = device.createCommandEncoder();
      const t0 = performance.now();
      pyramid.record(e, querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
      if (querySet) {
        e.resolveQuerySet(querySet, 0, 2, resolveBuffer!, 0);
        e.copyBufferToBuffer(resolveBuffer!, 0, readBuffer!, 0, 16);
      }
      device.queue.submit([e.finish()]);
      await device.queue.onSubmittedWorkDone();
      wall.push(performance.now() - t0);
      if (readBuffer) {
        await readBuffer.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(readBuffer.getMappedRange());
        const ns = Number(t[1]! - t[0]!);
        readBuffer.unmap();
        if (ns > 0) gpu.push(ns / 1e6);
      }
    }
    const { layout } = pyramid;
    const r: AssetResult = {
      name: spec,
      type: gridInfo.type,
      indexBox,
      texels: [...layout.texels],
      level: layout.level,
      dimensions: [...layout.dimensions],
      atlasDimensions: layout.atlas ? [...layout.atlas.dimensions] : null,
      atlasLevels: layout.atlas?.levels.length ?? 0,
      base,
      atlas,
      gpuBytes: { ...pyramid.gpuBytes },
      cpuBytesUploaded: 2 * (cpu.base.length + (cpu.atlas?.length ?? 0)),
      gpuMs: gpu.length ? median(gpu) : null,
      gpuWallMs: median(wall),
      cpuMs: median(cpuRuns),
    };
    result.assets.push(r);
    const ok = base.over === 0 && (atlas?.over ?? 0) === 0;
    log(
      `${spec} (${r.type}): L₀ ${r.level}, base ${r.dimensions.join("x")}, atlas ${r.atlasDimensions?.join("x") ?? "—"}; ` +
        `base ${base.exact}/${base.texels} exact, ${base.ulp1} at 1 ulp, ${base.over} over; ` +
        (atlas ? `atlas ${atlas.exact}/${atlas.texels} exact, ${atlas.ulp1} at 1 ulp, ${atlas.over} over; ` : "") +
        `GPU ${r.gpuMs?.toFixed(3) ?? "—"} ms (wall ${r.gpuWallMs.toFixed(2)} ms), CPU ${r.cpuMs.toFixed(1)} ms; bytes ${JSON.stringify(r.gpuBytes)}`,
      ok ? "ok" : "fail",
    );

    if (SEQUENCE && spec === ASSETS[0]) {
      // A sequence frame: another image in the same grid id; the pyramid keeps its index box.
      const next = await loadAsset(name === "sphere" ? "torus" : "sphere");
      const staleBefore = pyramid.stale;
      pool.replace(id, next.image);
      const e = device.createCommandEncoder();
      pool.update(e);
      const rebuilt = pyramid.update(e);
      device.queue.submit([e.finish()]);
      await device.queue.onSubmittedWorkDone();
      const staleAfter = pyramid.stale;
      const e2 = device.createCommandEncoder();
      const again = pyramid.update(e2);
      device.queue.submit([e2.finish()]);
      const cpuNext = cpuPyramid(next.image, indexBox, pyramidMax);
      result.sequence = { staleBefore, rebuilt, staleAfter, again, ...(await checkAgainstCpu(device, pyramid, cpuNext)) };
      log(`sequence: ${JSON.stringify(result.sequence)}`, rebuilt && !again && result.sequence.base.over === 0 ? "ok" : "fail");
    }
    pyramid.destroy();
    pool.dispose();
  }

  if (THREE_CHECK) await threeCheck(device);
  for (const w of result.warnings) log(`warning: ${w}`, "warn");
}

/** The vdbSparseSource handle across a pool.replace: version moves, refreshStats reads the new image back. */
async function threeCheck(device: GPUDevice): Promise<void> {
  const cloud = await loadAsset("cloud04");
  const renderer = new WebGPURenderer({ device });
  await renderer.init();
  const pool = await VDBGridPool.create(device, { label: "k3-pyramid three" });
  const id = pool.add(cloud.image);
  pool.flush();
  await pool.settled();
  const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd"] });
  const source = vdbSparseSource(grids, { density: id }, { images: { density: cloud.image }, indexBox: cloud.indexBox!, pyramidMax: 32 });
  await device.queue.onSubmittedWorkDone();

  // A sequence frame: replace the grid (same box: the cloud with its density halved), update the pool.
  const half = wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(cloudWvdbUrl))), { mipmaps: false });
  const halfData = half.density.data as Uint8Array;
  for (let i = 0; i < halfData.length; i++) halfData[i] = halfData[i]! >> 1;
  const next = wvdbDenseToNvdb(half).image;
  const versionBefore = source.version;
  pool.replace(id, next);
  pool.flush();
  grids.update();
  await device.queue.onSubmittedWorkDone();
  // K3.2e: nothing refreshes the stats by itself now; the read-back of the new image gives the new mean.
  const statsBefore = { ...source.stats };
  await source.refreshStats();
  const fresh = vdbSparseSource(grids, { density: id }, { images: { density: next }, indexBox: cloud.indexBox!, lod: "none" });
  const stats = { before: statsBefore, after: { ...source.stats }, expected: { ...fresh.stats }, updates: source.statsUpdates };
  fresh.dispose();
  result.three = { versions: [versionBefore, source.version], stats };
  log(`three: ${JSON.stringify(result.three)}`, source.version > versionBefore && source.statsUpdates === 1 ? "ok" : "fail");
  source.dispose();
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
