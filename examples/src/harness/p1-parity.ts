/**
 * P1 GPU parity (pmndrs/pathtracer integration, our side): generated modules + one VDBGridPool against the CPU
 * twins in `tools`. Results go to `window.__P1_PARITY__` for `e2e/p1-parity.spec.ts`.
 *
 * - P1.1 point-sample contract: `vdbSample` (`outside: 'zero'`) on a `vdbBuildFromDense` image of the
 *   pathtracer's `vdb-smoke` dense grid (rebuilt from fixtures/vdb-samples/smoke.vdb) at the 10k points of
 *   `packages/vdb/test/p1/contract.test.ts`, vs the CPU twin and vs its dense trilinear; vendored and lean readers.
 * - P1.3 / P1.4 majorant walks (`brick` with minima, `global`, `nodeMax`, `brick` on the lean reader, `brick` with
 *   `outside: 'background'`) vs `majorantWalk` (tools internal): the segment boundaries agree to
 *   |Δt| ≤ 1e-5 · max(1, |t|) and the bounds (and minima) are equal, compared as step functions so a near-tie
 *   at a cell corner (a sliver one side has and the other not) is not a failure; exact sequences are counted.
 * - P1.5 lean reader: `vdbReadValue` / `vdbIsActive` / `vdbSample` on Float, Fp8, FpN and Half grids (multi-type
 *   and single-type modules, both `outside` modes) vs `vdbReadValueF32Cpu`, `isActive`, `vdbSampleTrilinearCpu`.
 * - P1.6 several grids at one binding: three grids (Float, Fp8, FpN), two with instance transforms, sampled in
 *   world space by one kernel (`vdbWorldToIndex(g, p)` + `vdbSample`, grid id per query) vs their per-grid twins.
 */
import { VDBGridPool, vdbModuleEntries, vdbModuleWgsl } from "@pmndrs/vdb";
import type { VDBGridTypeName, VDBSlotModuleOptions } from "@pmndrs/vdb";
import {
  NvdbFile,
  indexBBox,
  isActive,
  majorantWalk,
  parseVdb,
  vdbBrickMajorant,
  vdbBuildFromDense,
  vdbQuantize,
  vdbReadValueF32Cpu,
  vdbSampleTrilinearCpu,
} from "../../../packages/vdb/src/tools/internal.js";
import type { MajorantSegment, VDBBrickMajorant, Vec3 } from "../../../packages/vdb/src/tools/internal.js";
import { contractPoints, denseTrilinear, rng, smokeDense, toXMajor } from "./p1-smoke.js";
import { installWgslPatch, type VDBWgslPatches } from "./wgsl-patch.js";

const SLOT = { group: 1, firstBinding: 0 } as const;
const MAXSEG = 128;
const WALK_OUT = 1 + MAXSEG * 4;
/** Ray parameter agreement (the P1.3 acceptance): |Δt| ≤ T_REL · max(1, |t|). */
const T_REL = 1e-5;
/** Read values: f32 decode of the same op order; 1 ulp for GPUs that fuse the Fp multiply-add. */
const VALUE_ULP = 1;
/** Trilinear: f32 lerps on the GPU vs f64 lerps of the same taps on the CPU; absolute, scaled by max(1, |ref|). */
const SAMPLE_TOL = 2e-6;
/** P1.1: the pathtracer's dense trilinear (f64 weighted sum) vs the GPU (f32 lerps); the acceptance's 1e-6. */
const CONTRACT_TOL = 1e-6;

/**
 * Fail-before mutations (`?mutate=wgsl-*`, ./wgsl-patch.ts): each patches the generated WGSL so one P1 entry computes
 * something else; e2e/p1-parity.spec.ts runs each and expects its rows to fail.
 */
const WGSL_PATCHES: VDBWgslPatches = {
  // P1.3 brick DDA: step the wrong axis on a y/x tie-break.
  "wgsl-brick-axis": ["if tn.y < tn[k] {", "if tn.y > tn[k] {"],
  // P1.3 minima: read the upper bounds as the lower ones.
  "wgsl-brick-minima": ["(*it).lo = bitcast<f32>(vdb_load((*it).mn + w));", "(*it).lo = bitcast<f32>(vdb_load((*it).base + w));"],
  // P1.3 global / P1.4: the support box one voxel short on the min side.
  "wgsl-support-pad": ["vec3f(vdbIndexMin(g)) - vec3f(1.0)", "vec3f(vdbIndexMin(g)) - vec3f(0.0)"],
  // P1.4: the cell's own node max instead of its 2x2x2 support.
  "wgsl-nodemax-support": ["for (var k = 1u; k < 8u; k++) {", "for (var k = 1u; k < 1u; k++) {"],
  // P1.5: x and y swapped in the leaf voxel index of vdbReadValue, and of the active-mask bit vdbIsActive reads.
  "wgsl-lean-leaf-index": ["((ijk.x & 7) << 6u) | ((ijk.y & 7) << 3u)", "((ijk.y & 7) << 6u) | ((ijk.x & 7) << 3u)"],
  "wgsl-lean-active-index": ["((u.x & 7u) << 6u) | ((u.y & 7u) << 3u)", "((u.y & 7u) << 6u) | ((u.x & 7u) << 3u)"],
  // P1.5: the root key's x field one bit off (negative coordinates miss their root tile).
  "wgsl-lean-root-key": ["((u.x >> 12u) << 10u)", "((u.x >> 12u) << 11u)"],
  // P1.1: outside 'zero' starts one voxel late (the trilinear support's min-side shell reads 0).
  "wgsl-contract-zero": ["any(b < (*acc).lo - vec3i(1))", "any(b < (*acc).lo)"],
  // P1.6: every grid read through grid 0's world-to-index affine.
  "wgsl-multi-grid": ["vdb_affine(g, VDB_REC_WORLD_TO_INDEX", "vdb_affine(0u, VDB_REC_WORLD_TO_INDEX"],
};
const MUTATE = new URLSearchParams(location.search).get("mutate") ?? "";

const FIXTURES = ["box_fog", "sphere_fog", "torus_fog"].flatMap((p) => ["float", "fp8", "fpn"].map((t) => `${p}_${t}`));

interface Row {
  total: number;
  failed: number;
  maxDelta: number;
}
interface P1Parity {
  done: boolean;
  error?: string;
  adapter?: string;
  isSwiftShader?: boolean;
  missing?: string[];
  modules: { id: string; bytes: number; compileErrors: string[] }[];
  rows: Record<string, Row>;
  failures: string[];
  info: Record<string, unknown>;
  validationError?: string;
  totalMs: number;
}

declare global {
  interface Window {
    __P1_PARITY__?: P1Parity;
  }
}

const root = document.getElementById("demo-root")!;
function log(text: string, cls?: "ok" | "fail"): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.appendChild(p);
}

const fr = Math.fround;
const ulp = (v: number) => {
  const a = Math.abs(fr(v));
  if (a === 0) return 2 ** -149;
  return 2 ** (Math.floor(Math.log2(a)) - 23);
};

class Collector {
  readonly rows: Record<string, Row> = {};
  readonly failures: string[] = [];
  add(cat: string, ok: boolean, delta: number, detail: () => string): void {
    const r = (this.rows[cat] ??= { total: 0, failed: 0, maxDelta: 0 });
    r.total++;
    if (Number.isFinite(delta)) r.maxDelta = Math.max(r.maxDelta, delta);
    if (!ok) {
      r.failed++;
      if (this.failures.length < 60) this.failures.push(`${cat}: ${detail()}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Modules and the kernel around them.

interface ModuleSpec {
  id: string;
  options: VDBSlotModuleOptions;
  walk: boolean;
}

const ALL: VDBGridTypeName[] = ["float", "fp8", "fpn"];
const MODULES: ModuleSpec[] = [
  { id: "vendored/sample/zero", options: { slot: SLOT, types: ALL, profile: "sample", outside: "zero" }, walk: false },
  { id: "lean/sample/zero", options: { slot: SLOT, types: ALL, profile: "sample", outside: "zero", reader: "lean" }, walk: false },
  { id: "lean/sample/fpn+half", options: { slot: SLOT, types: ["fpn", "half"], profile: "sample", reader: "lean" }, walk: false },
  { id: "lean/sample/float", options: { slot: SLOT, types: ["float"], profile: "sample", reader: "lean" }, walk: false },
  { id: "majorant/brick+minima/zero", options: { slot: SLOT, types: ALL, profile: "majorant", majorant: "brick", minima: true, outside: "zero" }, walk: true },
  { id: "majorant/global/zero", options: { slot: SLOT, types: ALL, profile: "majorant", majorant: "global", outside: "zero" }, walk: true },
  { id: "majorant/nodeMax/zero", options: { slot: SLOT, types: ALL, profile: "majorant", majorant: "nodeMax", outside: "zero" }, walk: true },
  {
    id: "majorant/brick+minima/zero/lean",
    options: { slot: SLOT, types: ALL, profile: "majorant", majorant: "brick", minima: true, outside: "zero", reader: "lean" },
    walk: true,
  },
  { id: "majorant/brick/background", options: { slot: SLOT, types: ALL, profile: "majorant", majorant: "brick" }, walk: true },
];

function kernel(spec: ModuleSpec): string {
  const minima = spec.options.minima === true;
  const walk = spec.walk
    ? `
@compute @workgroup_size(16)
fn walkMain(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if 2u * i + 1u >= arrayLength(&inq) {
        return;
    }
    let r0 = inq[2u * i];
    let r1 = inq[2u * i + 1u];
    var it = vdbMajorantInit(u32(r0.w), r0.xyz, r1.xyz, 0.0, r1.w);
    let base = i * ${WALK_OUT}u;
    var n = 0u;
    loop {
        let s = vdbMajorantNext(&it);
        if s.y <= s.x {
            break;
        }
        if n >= ${MAXSEG}u {
            n = n + 1u;
            break;
        }
        let q = base + 1u + n * 4u;
        outv[q] = s.x;
        outv[q + 1u] = s.y;
        outv[q + 2u] = s.z;
        outv[q + 3u] = ${minima ? "vdbMajorantMinimum(&it)" : "0.0"};
        n = n + 1u;
    }
    outv[base] = f32(n);
}`
    : "";
  return `${vdbModuleWgsl(spec.options)}
@group(0) @binding(0) var<storage, read_write> outv: array<f32>;
@group(0) @binding(1) var<storage, read> inq: array<vec4f>;

@compute @workgroup_size(64)
fn readMain(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&inq) {
        return;
    }
    let q = inq[i];
    var a = vdbAccessorInit(u32(q.w));
    let ijk = vec3i(q.xyz);
    outv[2u * i] = vdbReadValue(&a, ijk);
    outv[2u * i + 1u] = select(0.0, 1.0, vdbIsActive(&a, ijk));
}

@compute @workgroup_size(64)
fn sampleMain(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&inq) {
        return;
    }
    let q = inq[i];
    var a = vdbAccessorInit(u32(q.w));
    outv[i] = vdbSample(&a, q.xyz);
}

@compute @workgroup_size(64)
fn worldMain(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&inq) {
        return;
    }
    let q = inq[i];
    let g = u32(q.w);
    let p = vdbWorldToIndex(g, q.xyz);
    var a = vdbAccessorInit(g);
    outv[4u * i] = p.x;
    outv[4u * i + 1u] = p.y;
    outv[4u * i + 2u] = p.z;
    outv[4u * i + 3u] = vdbSample(&a, p);
}
${walk}`;
}

interface Gpu {
  device: GPUDevice;
  layout0: GPUBindGroupLayout;
  pipelineLayout: GPUPipelineLayout;
  poolGroup: GPUBindGroup;
}

async function dispatch(gpu: Gpu, pipeline: GPUComputePipeline, input: Float32Array, outFloats: number, invocations: number, workgroup: number): Promise<Float32Array> {
  const { device } = gpu;
  const inBuf = device.createBuffer({ size: Math.max(16, input.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, input.buffer as ArrayBuffer, input.byteOffset, input.byteLength);
  const outBuf = device.createBuffer({ size: Math.max(16, outFloats * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: Math.max(16, outFloats * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: gpu.layout0,
      entries: [
        { binding: 0, resource: { buffer: outBuf } },
        { binding: 1, resource: { buffer: inBuf, size: input.byteLength } },
      ],
    }),
  );
  pass.setBindGroup(1, gpu.poolGroup);
  pass.dispatchWorkgroups(Math.ceil(invocations / workgroup));
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, Math.max(16, outFloats * 4));
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const res = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  inBuf.destroy();
  outBuf.destroy();
  readBuf.destroy();
  return res;
}

// ---------------------------------------------------------------------------
// Grids and queries.

interface Grid {
  id: number;
  name: string;
  image: Uint32Array;
  type: VDBGridTypeName;
  majorant: VDBBrickMajorant | null;
  /** Instance transform (column-major object-to-world), if any. */
  transform: number[] | null;
}

/** Column-major 4x4: uniform scale `s`, rotation `deg` about `axis`, translation `t`. */
function trs(s: number, deg: number, axis: Vec3, t: Vec3): number[] {
  const len = Math.hypot(...axis);
  const [x, y, z] = axis.map((v) => v / len) as Vec3;
  const th = (deg * Math.PI) / 180;
  const cs = Math.cos(th);
  const sn = Math.sin(th);
  const k = 1 - cs;
  const r = [
    [cs + x * x * k, x * y * k - z * sn, x * z * k + y * sn],
    [y * x * k + z * sn, cs + y * y * k, y * z * k - x * sn],
    [z * x * k - y * sn, z * y * k + x * sn, cs + z * z * k],
  ];
  const m = new Array<number>(16).fill(0);
  for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++) m[col * 4 + row] = s * r[row]![col]!;
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  m[15] = 1;
  return m;
}

interface Ray {
  o: Vec3;
  d: Vec3;
}

/** Index-space rays through the support box (from outside, some from inside, some axis-parallel), f32 values. */
function rays(image: Uint32Array, count: number, seed: number): Ray[] {
  const { min, max } = indexBBox(image);
  const r = rng(seed);
  const ext = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) + 4;
  const out: Ray[] = [];
  for (let s = 0; s < count; s++) {
    const target = [0, 1, 2].map((a) => min[a]! - 1.5 + r() * (max[a]! - min[a]! + 3)) as Vec3;
    let d: Vec3 = [r() - 0.5, r() - 0.5, r() - 0.5];
    if (s % 10 === 7) d[Math.floor(r() * 3)] = 0;
    if (s % 16 === 3) d = [0, 0, 0].map((_, a) => (a === s % 3 ? (r() < 0.5 ? -1 : 1) : 0)) as Vec3;
    const len = Math.hypot(...d) || 1;
    d = d.map((v) => (v / len) * (0.5 + r())) as Vec3;
    const back = s % 5 === 0 ? 0 : ext / Math.hypot(...d);
    out.push({ o: target.map((v, a) => fr(v - d[a]! * back)) as Vec3, d: d.map(fr) as Vec3 });
  }
  return out;
}

/** Step-function comparison of two walks (see the header). */
function compareWalk(c: Collector, cat: string, gpu: MajorantSegment[], cpu: MajorantSegment[], minima: boolean, overflow: boolean, label: string, counters: { exact: number; rays: number; segs: number; maxRel: number }): void {
  counters.rays++;
  counters.segs += cpu.length;
  if (overflow) {
    c.add(cat, false, Infinity, () => `${label}: more than ${MAXSEG} segments`);
    return;
  }
  const tol = (t: number) => T_REL * Math.max(1, Math.abs(t));
  if (gpu.length === 0 || cpu.length === 0) {
    const ok = gpu.length === cpu.length || (gpu.length === 0 && cpu.every((s) => s.t1 - s.t0 <= tol(s.t1))) || (cpu.length === 0 && gpu.every((s) => s.t1 - s.t0 <= tol(s.t1)));
    c.add(cat, ok, 0, () => `${label}: ${gpu.length} GPU vs ${cpu.length} CPU segments`);
    if (ok && gpu.length === cpu.length) counters.exact++;
    return;
  }
  const bounds = (s: MajorantSegment[]) => [s[0]!.t0, ...s.map((x) => x.t1)];
  const bg = bounds(gpu);
  const bc = bounds(cpu);
  let worst = 0;
  const nearest = (t: number, list: number[]) => list.reduce((m, u) => Math.min(m, Math.abs(u - t)), Infinity);
  for (const t of bg) worst = Math.max(worst, nearest(t, bc) / Math.max(1, Math.abs(t)));
  for (const t of bc) worst = Math.max(worst, nearest(t, bg) / Math.max(1, Math.abs(t)));
  counters.maxRel = Math.max(counters.maxRel, worst);
  let ok = worst <= T_REL;
  let detail = `${label}: boundary off by ${worst.toExponential(2)} (relative)`;
  const at = (s: MajorantSegment[], t: number) => s.find((x) => t >= x.t0 && t < x.t1);
  for (const t of [...gpu, ...cpu].filter((s) => s.t1 - s.t0 > 2 * tol(s.t1)).map((s) => (s.t0 + s.t1) / 2)) {
    if (nearest(t, bc) <= tol(t) || nearest(t, bg) <= tol(t)) continue;
    const a = at(gpu, t);
    const b = at(cpu, t);
    if (!a || !b || a.max !== b.max || (minima && a.min !== b.min)) {
      ok = false;
      detail = `${label}: at t=${t}: GPU ${a ? `${a.max}/${a.min}` : "none"} vs CPU ${b ? `${b.max}/${b.min}` : "none"}`;
      break;
    }
  }
  c.add(cat, ok, worst, () => detail);
  if (ok && gpu.length === cpu.length && gpu.every((s, i) => s.max === cpu[i]!.max && Math.abs(s.t0 - cpu[i]!.t0) <= tol(s.t0))) counters.exact++;
}

async function run(): Promise<P1Parity> {
  const t00 = performance.now();
  const c = new Collector();
  const result: P1Parity = { done: false, modules: [], rows: c.rows, failures: c.failures, info: {}, totalMs: 0 };
  if (!navigator.gpu) return { ...result, done: true, error: "WebGPU unavailable (navigator.gpu undefined)." };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ...result, done: true, error: "No WebGPU adapter." };
  const ai = (adapter as unknown as { info?: Record<string, string> }).info ?? {};
  result.adapter = [ai.vendor, ai.architecture, ai.device, ai.description].filter(Boolean).join(" / ") || "unknown";
  result.isSwiftShader = /swiftshader/i.test(result.adapter);
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  const uncaptured: string[] = [];
  device.addEventListener("uncapturederror", (e) => uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  const checkWgslPatch = installWgslPatch(device, WGSL_PATCHES, MUTATE);
  result.info.mutate = MUTATE || null;

  // Fixtures: the nine primitives and the smoke sample.
  const files = await Promise.all(FIXTURES.map((n) => fetch(`/fixtures/primitives/${n}.nvdb`)));
  const smokeRes = await fetch("/fixtures/vdb-samples/smoke.vdb");
  const missing = [...FIXTURES.filter((_, k) => !files[k]!.ok).map((n) => `${n}.nvdb`), ...(smokeRes.ok ? [] : ["vdb-samples/smoke.vdb"])];
  if (missing.length > 0) return { ...result, done: true, missing };
  const images = await Promise.all(files.map(async (f) => NvdbFile.fromArrayBuffer(await f.arrayBuffer()).gridImage(0)));
  const smoke = smokeDense(parseVdb(await smokeRes.arrayBuffer()));
  const smokeImage = vdbBuildFromDense(toXMajor(smoke.density, smoke.resolution), smoke.resolution);
  const sphereFloat = images[FIXTURES.indexOf("sphere_fog_float")]!;
  const halfImage = vdbQuantize(sphereFloat, "half");

  device.pushErrorScope("validation");
  const pool = await VDBGridPool.create(device, { initialBytes: 4096, maxGrids: 2, label: "p1 parity" });
  const grids: Grid[] = [];
  const add = (name: string, image: Uint32Array, type: VDBGridTypeName, majorant: VDBBrickMajorant | null, transform: number[] | null = null) => {
    const id = pool.add(image, { label: name, ...(majorant ? { majorant } : {}), ...(transform ? { transform } : {}) });
    grids.push({ id, name, image, type, majorant, transform });
    pool.flush();
    return id;
  };
  FIXTURES.forEach((name, k) => add(name, images[k]!, name.endsWith("float") ? "float" : name.endsWith("fp8") ? "fp8" : "fpn", vdbBrickMajorant(images[k]!, { minima: true, outside: "zero" })));
  const smokeId = add("vdb-smoke", smokeImage, "float", vdbBrickMajorant(smokeImage, { align: "pathtracer", resolution: smoke.resolution, minima: true }));
  const globalId = add("sphere_fog_float (no brick)", sphereFloat, "float", null);
  const halfId = add("sphere_fog_half", halfImage, "half", null);
  const dead = pool.add(sphereFloat, { label: "dead" });
  pool.flush();
  const t1 = trs(0.75, 30, [1, 2, 2], [40, -25, 17]);
  const t2 = trs(1.5, -70, [0, 1, 0.3], [-120, 8, 64]);
  const fp8Inst = add("sphere_fog_fp8 (instance)", images[FIXTURES.indexOf("sphere_fog_fp8")]!, "fp8", null, t1);
  const fpnInst = add("torus_fog_fpn (instance)", images[FIXTURES.indexOf("torus_fog_fpn")]!, "fpn", null, t2);
  pool.remove(dead);
  pool.flush();
  pool.flush();
  result.info.pool = { version: pool.version, gridCapacity: pool.gridCapacity, byteLength: pool.byteLength, grids: grids.length, smokeId, globalId, halfId, dead };

  const layout0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const layout1 = device.createBindGroupLayout({ entries: vdbModuleEntries({ slot: SLOT, visibility: GPUShaderStage.COMPUTE }) });
  const gpu: Gpu = {
    device,
    layout0,
    pipelineLayout: device.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] }),
    poolGroup: device.createBindGroup({ layout: layout1, entries: [pool.bindGroupEntry(SLOT)] }),
  };

  // Query sets, shared by the modules.
  const contract = contractPoints(smoke.resolution);
  const readSets = new Map<number, Vec3[]>();
  const sampleSets = new Map<number, Vec3[]>();
  for (const g of grids) {
    const { min, max } = indexBBox(g.image);
    const r = rng(500 + g.id);
    const reads: Vec3[] = [];
    for (let k = 0; k < 400; k++) reads.push([0, 1, 2].map((a) => Math.floor(min[a]! - 4 + r() * (max[a]! - min[a]! + 9))) as Vec3);
    reads.push([9000, -9000, 9000], [-5000, 100, 7], [0, 0, 0]); // the background and root tiles
    readSets.set(g.id, reads);
    sampleSets.set(g.id, Array.from({ length: 300 }, () => [0, 1, 2].map((a) => fr(min[a]! - 2 + r() * (max[a]! - min[a]! + 4))) as Vec3));
  }
  const walkRays = new Map<number, Ray[]>(grids.map((g) => [g.id, rays(g.image, 48, 900 + g.id)]));
  walkRays.set(dead, rays(sphereFloat, 4, 77));
  const counters: Record<string, { exact: number; rays: number; segs: number; maxRel: number }> = {};

  for (const spec of MODULES) {
    const code = kernel(spec);
    const compileErrors: string[] = [];
    const sm = device.createShaderModule({ code, label: spec.id });
    for (const m of (await sm.getCompilationInfo()).messages) if (m.type === "error") compileErrors.push(`${m.lineNum}:${m.linePos} ${m.message}`);
    result.modules.push({ id: spec.id, bytes: code.length, compileErrors });
    if (compileErrors.length > 0) {
      log(`${spec.id}: ${compileErrors.length} compile errors: ${compileErrors[0]}`, "fail");
      continue;
    }
    const pipe = (entryPoint: string) => device.createComputePipelineAsync({ layout: gpu.pipelineLayout, compute: { module: sm, entryPoint } });
    const o = spec.options;
    const zero = o.outside === "zero";
    const compiled = new Set(o.types);
    const readable = grids.filter((g) => compiled.has(g.type));
    const lean = o.reader === "lean";

    // Reads (every module: the vendored rows are the baseline the lean rows are held to).
    {
      const rows: { g: Grid; ijk: Vec3 }[] = [];
      for (const g of readable) for (const ijk of readSets.get(g.id)!) rows.push({ g, ijk });
      const inp = new Float32Array(rows.length * 4);
      rows.forEach(({ g, ijk }, k) => inp.set([...ijk, g.id], k * 4));
      const out = await dispatch(gpu, await pipe("readMain"), inp, rows.length * 2, rows.length, 64);
      const cat = `${lean ? "lean" : "vendored"}-read`;
      rows.forEach(({ g, ijk }, k) => {
        const want = vdbReadValueF32Cpu(g.image, ijk);
        const got = out[2 * k]!;
        const d = Math.abs(got - want);
        c.add(`${cat}-${g.type}`, d <= VALUE_ULP * ulp(want), d / ulp(want), () => `${spec.id} ${g.name} ${ijk}: ${got} vs ${want}`);
        const act = isActive(g.image, ijk);
        c.add(`${cat}-active`, (out[2 * k + 1] === 1) === act, 0, () => `${spec.id} ${g.name} ${ijk}: active ${out[2 * k + 1]} vs ${act}`);
      });
    }
    // Samples at random index points.
    {
      const rows: { g: Grid; p: Vec3 }[] = [];
      for (const g of readable) for (const p of sampleSets.get(g.id)!) rows.push({ g, p });
      const inp = new Float32Array(rows.length * 4);
      rows.forEach(({ g, p }, k) => inp.set([...p, g.id], k * 4));
      const out = await dispatch(gpu, await pipe("sampleMain"), inp, rows.length, rows.length, 64);
      rows.forEach(({ g, p }, k) => {
        const want = vdbSampleTrilinearCpu(g.image, p, { outside: zero ? "zero" : "background" });
        const d = Math.abs(out[k]! - want);
        c.add(`${lean ? "lean" : "vendored"}-sample`, d <= SAMPLE_TOL * Math.max(1, Math.abs(want)), d, () => `${spec.id} ${g.name} ${p}: ${out[k]} vs ${want}`);
      });
    }
    // P1.1: the contract points on vdb-smoke.
    if (zero && compiled.has("float")) {
      const inp = new Float32Array(contract.length * 4);
      contract.forEach((p, k) => inp.set([...p, smokeId], k * 4));
      const out = await dispatch(gpu, await pipe("sampleMain"), inp, contract.length, contract.length, 64);
      const cat = `contract-${lean ? "lean" : "vendored"}`;
      let nonzero = 0;
      contract.forEach((p, k) => {
        const twin = vdbSampleTrilinearCpu(smokeImage, p, { outside: "zero" });
        const dense = denseTrilinear(smoke.density, smoke.resolution, p);
        if (dense > 0) nonzero++;
        c.add(`${cat}-twin`, Math.abs(out[k]! - twin) <= SAMPLE_TOL, Math.abs(out[k]! - twin), () => `${spec.id} ${p}: ${out[k]} vs twin ${twin}`);
        c.add(`${cat}-dense`, Math.abs(out[k]! - dense) <= CONTRACT_TOL, Math.abs(out[k]! - dense), () => `${spec.id} ${p}: ${out[k]} vs dense ${dense}`);
      });
      result.info[`${cat}-nonzero`] = nonzero;
    }
    // P1.6: three grids (Float, Fp8 + transform, FpN + transform) in one kernel, world space.
    if (compiled.has("float") && compiled.has("fp8") && compiled.has("fpn") && !spec.walk) {
      const three = [smokeId, fp8Inst, fpnInst].map((id) => grids.find((g) => g.id === id)!);
      const rows: { g: Grid; p: Vec3 }[] = [];
      const r = rng(4242);
      for (let k = 0; k < 600; k++) {
        const g = three[k % 3]!;
        const wb = pool.info(g.id).worldBBox;
        rows.push({ g, p: [0, 1, 2].map((a) => fr(wb.min[a]! - 3 + r() * (wb.max[a]! - wb.min[a]! + 6))) as Vec3 });
      }
      const inp = new Float32Array(rows.length * 4);
      rows.forEach(({ g, p }, k) => inp.set([...p, g.id], k * 4));
      const out = await dispatch(gpu, await pipe("worldMain"), inp, rows.length * 4, rows.length, 64);
      const cat = `multi-grid-${lean ? "lean" : "vendored"}`;
      const nonzero = new Map<number, number>();
      rows.forEach(({ g, p }, k) => {
        const m = pool.info(g.id).worldToIndex;
        const idx = [0, 1, 2].map((row) => m[row * 4]! * p[0] + m[row * 4 + 1]! * p[1] + m[row * 4 + 2]! * p[2] + m[row * 4 + 3]!);
        const scale = [0, 1, 2].map((row) => Math.abs(m[row * 4]! * p[0]) + Math.abs(m[row * 4 + 1]! * p[1]) + Math.abs(m[row * 4 + 2]! * p[2]) + Math.abs(m[row * 4 + 3]!));
        const gi = [out[4 * k]!, out[4 * k + 1]!, out[4 * k + 2]!] as Vec3;
        const dIdx = Math.max(...gi.map((v, a) => Math.abs(v - idx[a]!) / Math.max(1, scale[a]!)));
        c.add(`${cat}-index`, dIdx <= 4e-6, dIdx, () => `${g.name} ${p}: index ${gi} vs ${idx}`);
        const want = vdbSampleTrilinearCpu(g.image, gi, { outside: "zero" });
        const d = Math.abs(out[4 * k + 3]! - want);
        c.add(`${cat}-sample`, d <= SAMPLE_TOL * Math.max(1, Math.abs(want)), d, () => `${g.name} ${p}: ${out[4 * k + 3]} vs ${want}`);
        if (want !== 0) nonzero.set(g.id, (nonzero.get(g.id) ?? 0) + 1);
      });
      result.info[`${cat}-nonzero`] = three.map((g) => `${g.name}: ${nonzero.get(g.id) ?? 0}`);
      c.add(`${cat}-sees-all-three`, three.every((g) => (nonzero.get(g.id) ?? 0) > 20), 0, () => `nonzero samples per grid ${JSON.stringify(result.info[`${cat}-nonzero`])}`);
    }
    // P1.3 / P1.4: majorant walks.
    if (spec.walk) {
      const mode = o.majorant ?? "brick";
      const minima = o.minima === true;
      const rows: { g: Grid | null; ray: Ray; id: number }[] = [];
      for (const g of grids) if (compiled.has(g.type)) for (const ray of walkRays.get(g.id)!) rows.push({ g, ray, id: g.id });
      for (const ray of walkRays.get(dead)!) rows.push({ g: null, ray, id: dead });
      const inp = new Float32Array(rows.length * 8);
      rows.forEach(({ ray, id }, k) => inp.set([...ray.o, id, ...ray.d, 1e30], k * 8));
      const out = await dispatch(gpu, await pipe("walkMain"), inp, rows.length * WALK_OUT, rows.length, 16);
      const cat = `walk-${spec.id.replace(/^majorant\//, "")}`;
      const ctr = (counters[cat] ??= { exact: 0, rays: 0, segs: 0, maxRel: 0 });
      rows.forEach(({ g, ray, id }, k) => {
        const base = k * WALK_OUT;
        const n = out[base]!;
        const segs: MajorantSegment[] = [];
        for (let s = 0; s < Math.min(n, MAXSEG); s++) {
          const q = base + 1 + s * 4;
          segs.push({ t0: out[q]!, t1: out[q + 1]!, max: out[q + 2]!, min: out[q + 3]!, lookups: 0 });
        }
        const cpu = g
          ? majorantWalk(g.image, g.majorant, ray.o, ray.d, { mode, outside: zero ? "zero" : "background", minima, tMax: 1e30 })
          : [];
        compareWalk(c, cat, segs, cpu, minima, n > MAXSEG, `${spec.id} ${g?.name ?? "dead"} #${id} ray ${ray.o}/${ray.d}`, ctr);
      });
    }
    log(`${spec.id}: done`);
  }
  result.info.walks = counters;
  checkWgslPatch();

  const scope = await device.popErrorScope();
  const validationError = [scope?.message, ...uncaptured].filter(Boolean).join("; ");
  pool.dispose();
  device.destroy();
  return { ...result, done: true, ...(validationError ? { validationError } : {}), totalMs: performance.now() - t00 };
}

root.textContent = "";
run()
  .catch((e: unknown): P1Parity => ({
    done: true,
    error: e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ""}` : String(e),
    modules: [],
    rows: {},
    failures: [],
    info: {},
    totalMs: 0,
  }))
  .then((r) => {
    window.__P1_PARITY__ = r;
    if (r.error) log(`error: ${r.error}`, "fail");
    else if (r.missing) log(`fixtures missing: ${r.missing.join(", ")} (pnpm fixtures:bake, pnpm fixtures:vdb-samples)`, "fail");
    else {
      log(`done in ${Math.round(r.totalMs)} ms on ${r.adapter ?? "?"}`);
      for (const [cat, s] of Object.entries(r.rows)) log(`${cat}: ${s.total - s.failed}/${s.total} (max delta ${s.maxDelta.toExponential(2)})`, s.failed === 0 ? "ok" : "fail");
      for (const f of r.failures) log(f, "fail");
    }
  });
