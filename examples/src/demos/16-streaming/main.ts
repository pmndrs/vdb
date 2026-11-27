/**
 * Gallery demo 16 — Streaming. A coarse-first `.nvdb` (`vdbBakeCoarseFirst`: `[density.lod48, density]`, ZIP) is
 * streamed through a throttled link into the froxel renderer with `vdbStreamCoarseFirst`: the 48-voxel preview is
 * drawn once its ~14 KiB are in, and the full-resolution grid replaces it in place (the same pool id) when its last
 * byte lands. A byte counter and a timeline (bytes received against time, with the preview and full markers) show
 * it; the caption flashes at each swap.
 *
 * Assets: the WDAS cloud (quarter, Fp8 full grid) from `fixtures/wdas/wdas_cloud_quarter.coarse.nvdb` (baked at dev
 * time: `pnpm build && node scripts/bake-coarse-first.mjs fixtures/wdas/wdas_cloud_quarter_fp8.nvdb --out
 * fixtures/wdas/wdas_cloud_quarter.coarse.nvdb`), else baked in the page from `wdas_cloud_quarter_fp8.nvdb`; and
 * `smoke.vdb` (openvdb.org, CC BY-SA 4.0) baked in the page as demo 09. Sky: `fixtures/gallery/sky-midday-HDR.jpg`
 * when present, else a procedural sky. Public API only.
 *
 * One `VDBGridNode`, sparse source and `VDBFrameVolume` per stream, made at the first grid (G16a): the node decodes
 * every grid type of the family (`event.types`: smoke and the WDAS cloud both have an FpN preview and an Fp8 full
 * grid), and the source is made with `followImage`, so after the swap it re-reads the full grid's boxes, rebuilds its
 * pyramid and the volume moves its box, with no new shader. The frame renderer integrates in world space, so a
 * preview needs no `voxelRatio` density scale, unlike `VDBVolumeMaterial`.
 *
 * K4.22, `?format=wvdb`: smoke.vdb baked as a `.wvdb` in the page (3 LODs, small pages) streams through
 * `WvdbStreamReader({ partialLods: true })` into one dense source (`vdbDenseTextures`, the core's `vdbDenseSource`),
 * page by page: `?partial=upsample` (default) shows each LOD as its pages land, the bricks not in yet upsampled from the
 * coarser LOD on the GPU (`wvdbDensePartialVolume`, the source's fill kernel; mip level 1 of those bricks is the coarser
 * LOD); `upsample-box` builds every mip from the upsample (his renderer); `background` shows them empty; `off` keeps
 * the last complete LOD until the next one is whole. `?test=1` records every page's frame and its PSNR against
 * the final frame (`window.__DEMO16__.wvdb`).
 *
 * Query (plus the shell's `presentation`, `panel`, `still`, `gallery`): `?asset=wdas|smoke`, `?speed=<KiB/s>`
 * (0: unthrottled), `?quality=low|medium|high`, `?temporal=0|1`, `?mode=direct` (no stream: the full grid added
 * directly, the reference the e2e compares the streamed result with), `?test=1` (with `presentation=readback`: still, unthrottled
 * 1 KiB chunks, the stream held until each grid's frame is read back, the shader and pipeline builds counted per
 * grid). `window.__DEMO16__` is the e2e state.
 *
 * Frame times are submit-to-done wall times on this page: provisional (D25), not a benchmark.
 */
import "../webgpu-gate.js";
import { fixtureUrl } from "../assets.js";
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VDBGridPool, type VDBGridTypeName } from "@pmndrs/vdb";
import {
  NvdbFile,
  WvdbStreamReader,
  parseVdb,
  readWvdb,
  vdbBakeCoarseFirst,
  vdbToWvdb,
  wvdbDensePartialVolume,
  wvdbDenseVolume,
  type NvdbGridMetadata,
  type WvdbDenseVolume,
  type WvdbVolumeLevel,
} from "@pmndrs/vdb/tools";
import {
  VDBFrameRenderer,
  VDBFrameVolume,
  VDBGridNode,
  getDevice,
  vdbDenseTextures,
  vdbSparseSource,
  vdbStreamCoarseFirst,
  type VDBCoarseFirstEvent,
  type VDBThreeDenseSource,
  type VDBThreeSparseSource,
} from "@pmndrs/vdb/three";
import { createDeviceRenderer } from "../device-renderer.js";
import { KiB, MiB, PRESENTATION as SHELL_PRESENTATION, Presenter, QUALITY, autoOrbit, initialQuality, mountShowcase, params, probe, type Quality, type Snapshot } from "../showcase-shell.js";
import { loadHdrSky, proceduralSky } from "../gallery-sky.js";

type AssetId = "wdas" | "smoke";
type Mode = "stream" | "direct";
/** K4.22: the coarse-first `.nvdb` into a sparse source, or a `.wvdb` LOD stream into a dense one. */
type Format = "nvdb" | "wvdb";
/** K4.22: what a `.wvdb` LOD shows while its pages stream in. */
type PartialMode = "upsample" | "upsample-box" | "background" | "off";

/** K4.22 `?test=1`: one frame per `.wvdb` stream event (a page or a completed LOD). */
interface WvdbFrameRecord {
  lod: number;
  /** The event completed its LOD. */
  complete: boolean;
  /** Bricks of the LOD decoded and stored (density grid). */
  decoded: number;
  stored: number;
  /** File bytes received at the event. */
  bytes: number;
  /** The LOD shown (the previous complete one with `partial=off` during a partial LOD). */
  shownLod: number;
  /** The shown volume had pending bricks filled from the coarser LOD. */
  filled: boolean;
  /** Against the final frame: mean squared error and PSNR (dB) over RGB, the whole frame and the volume's pixels. */
  mse: number;
  psnr: number;
  regionMse: number;
  regionPsnr: number;
  /** Shader modules and pipelines built for this frame. */
  builds: number;
  /** Frames drawn until two in a row were identical (11: never within 10). */
  drawn: number;
}

interface WvdbRun {
  partial: PartialMode;
  frames: WvdbFrameRecord[];
  /** Pixels where the final frame differs from the frame without a volume (the PSNR region). */
  region: number;
  /** FNV-1a of the final frame's bytes (equal for every `partial` mode and the direct render). */
  finalHash: string;
  /** LOD sizes of the density grid. */
  lods: number[][];
  /** The dense source's fills (K4.22): on the GPU and on the CPU. */
  fills: { gpu: number; cpu: number } | null;
}

interface PhaseRecord {
  phase: "preview" | "full";
  grid: string;
  gridType: string;
  /** File bytes received when the grid completed (the wire bytes: no HTTP compression on this link). */
  transferBytes: number;
  /** Seconds from the start of the stream to the grid's arrival. */
  seconds: number;
  /** Pool image bytes of the grid. */
  imageBytes: number;
  voxelRatio: number;
  /** The first settled frame with this grid (readback only). */
  snapshot: Snapshot | null;
  /** `?test=1`: shader modules and pipelines the device created from the grid's arrival to its snapshot. */
  builds: number;
}

interface Demo16State {
  ready: boolean;
  error?: string;
  adapter?: string;
  presentation: "direct" | "readback";
  mode: Mode;
  asset?: AssetId;
  available: AssetId[];
  missing: string[];
  sky?: string;
  totalBytes?: number;
  transferred: number;
  phases: PhaseRecord[];
  /** Same pool id for every grid shown, and no pool growth after the preview. */
  sameId?: boolean;
  inPlace?: boolean;
  done: boolean;
  frames: number;
  uncaptured: string[];
  /** `?mode=direct`: the reference frame. */
  reference?: Snapshot | null;
  /** K4.22: `?format=wvdb`. */
  format: Format;
  wvdb?: WvdbRun;
  /** K4.22 `?mode=direct&format=wvdb`: FNV-1a of the reference frame's bytes. */
  referenceHash?: string;
}

declare global {
  interface Window {
    __DEMO16__?: Demo16State;
    __DEMO16_API__?: { capture(): Snapshot | null; restart(): void };
  }
}

const TEST = params.get("test") === "1";
const PRESENTATION = SHELL_PRESENTATION;
const MODE: Mode = params.get("mode") === "direct" ? "direct" : "stream";
const PARTIAL_MODES: readonly PartialMode[] = ["upsample", "upsample-box", "background", "off"];
const PARTIAL_LABEL: Record<PartialMode, string> = {
  upsample: "partial LODs: upsample",
  "upsample-box": "partial LODs: upsample, his mips",
  background: "partial LODs: empty",
  off: "whole LODs only",
};
/** K4.22 bake of smoke.vdb as `.wvdb`: finest LOD 96 voxels on the long side, small pages so a LOD lands in several. */
const WVDB_BAKE = { resolution: 96, pageBricks: 24, lods: 3 } as const;
const STILL_MODE = TEST || params.get("still") === "1";
const SPEEDS = [256, 1024, 2048, 4096, 10240] as const;
const speedLabel = (k: number) => (k === 0 ? "unthrottled" : k < 1024 ? `${k} KiB/s` : `${k / 1024} MiB/s`);
const SKY_URL = fixtureUrl("gallery/sky-midday-HDR.jpg");
const WDAS_BAKED = fixtureUrl("wdas/wdas_cloud_quarter.coarse.nvdb");
const WDAS_FP8 = fixtureUrl("wdas/wdas_cloud_quarter_fp8.nvdb");
const SMOKE_VDB = fixtureUrl("vdb-samples/smoke.vdb");
/** Longest side of the volume in scene units. */
const SIZE = 100;

const shell = mountShowcase({ strip: "row",
  number: "16",
  title: "Streaming",
  blurb:
    "A coarse-first .nvdb streams into the froxel renderer: a 48-voxel preview draws from the first ~14 KiB, and the full-resolution grid swaps in place when its last byte lands.",
});
const state: Demo16State = {
  ready: false,
  presentation: PRESENTATION,
  mode: MODE,
  available: [],
  missing: [],
  transferred: 0,
  phases: [],
  done: false,
  frames: 0,
  uncaptured: [],
  format: params.get("format") === "wvdb" ? "wvdb" : "nvdb",
};
window.__DEMO16__ = state;
shell.onFail((message) => {
  state.error = message;
});

// --- the caption's counter, phase line and timeline ---------------------------------------------------------

const phaseEl = document.createElement("div");
phaseEl.dataset.testid = "phase";
phaseEl.style.cssText = "margin:8px 0 2px;font-weight:650;font-size:13.5px;letter-spacing:0.02em";
const counterEl = document.createElement("div");
counterEl.dataset.testid = "counter";
counterEl.style.cssText = "font-size:12.5px;opacity:0.85;font-variant-numeric:tabular-nums;white-space:pre-line";
const timeline = document.createElement("canvas");
timeline.style.cssText = "display:block;width:100%;height:68px;margin-top:6px;border-radius:6px;background:rgba(255,255,255,0.04)";
shell.extra.append(phaseEl, counterEl, timeline);
const PHASE_COLOR = { loading: "#c9d1e6", preview: "#ffd479", full: "#7be3a1" };
const setPhase = (kind: keyof typeof PHASE_COLOR, text: string) => {
  phaseEl.textContent = text;
  phaseEl.style.color = PHASE_COLOR[kind];
  phaseEl.dataset.phase = kind;
};

interface Timeline {
  samples: { t: number; bytes: number }[];
  total: number;
  expectedSeconds: number;
  markers: { t: number; bytes: number; kind: "preview" | "full" }[];
}
function drawTimeline(tl: Timeline): void {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(timeline.clientWidth * dpr));
  const h = Math.max(1, Math.round(timeline.clientHeight * dpr));
  if (timeline.width !== w || timeline.height !== h) {
    timeline.width = w;
    timeline.height = h;
  }
  const g = timeline.getContext("2d")!;
  g.clearRect(0, 0, w, h);
  const pad = { l: 6 * dpr, r: 6 * dpr, t: 14 * dpr, b: 6 * dpr };
  const last = tl.samples[tl.samples.length - 1];
  const tMax = Math.max(0.5, tl.expectedSeconds, last?.t ?? 0, ...tl.markers.map((m) => m.t)) * 1.04;
  const x = (t: number) => pad.l + (t / tMax) * (w - pad.l - pad.r);
  const y = (b: number) => h - pad.b - (tl.total > 0 ? b / tl.total : 0) * (h - pad.t - pad.b);
  // the link's ideal line
  g.strokeStyle = "rgba(255,255,255,0.12)";
  g.setLineDash([3 * dpr, 3 * dpr]);
  g.beginPath();
  g.moveTo(x(0), y(0));
  g.lineTo(x(tl.expectedSeconds), y(tl.total));
  g.stroke();
  g.setLineDash([]);
  // bytes received
  if (tl.samples.length) {
    g.strokeStyle = "#a9bcff";
    g.lineWidth = 1.6 * dpr;
    g.beginPath();
    g.moveTo(x(0), y(0));
    for (const s of tl.samples) g.lineTo(x(s.t), y(s.bytes));
    g.stroke();
    g.fillStyle = "rgba(169,188,255,0.12)";
    g.lineTo(x(last!.t), y(0));
    g.closePath();
    g.fill();
  }
  // markers
  g.font = `${10.5 * dpr}px system-ui, sans-serif`;
  for (const m of tl.markers) {
    const color = m.kind === "full" ? PHASE_COLOR.full : PHASE_COLOR.preview;
    g.strokeStyle = color;
    g.lineWidth = 1.2 * dpr;
    g.beginPath();
    g.moveTo(x(m.t), pad.t - 2 * dpr);
    g.lineTo(x(m.t), h - pad.b);
    g.stroke();
    g.fillStyle = color;
    g.beginPath();
    g.arc(x(m.t), y(m.bytes), 3 * dpr, 0, Math.PI * 2);
    g.fill();
    const label = `${m.kind} · ${m.bytes < 1048576 ? KiB(m.bytes) : MiB(m.bytes)}`;
    const tw = g.measureText(label).width;
    const lx = m.kind === "preview" ? Math.min(x(m.t) + 4 * dpr, w - tw - 2 * dpr) : Math.max(2 * dpr, x(m.t) - tw - 4 * dpr);
    g.fillText(label, lx, 10.5 * dpr);
  }
}

// --- the throttled link (demo 09's pull-driven stream) -------------------------------------------------------

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
function throttle(response: Response, opts: { chunk: number; bytesPerSec: number; gate: () => Promise<void>; onBytes: (sent: number) => void; signal: AbortSignal }): Response {
  const upstream = response.body!.getReader();
  let pending = new Uint8Array(0);
  let sent = 0;
  const body = new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        await opts.gate();
        if (opts.signal.aborted) {
          controller.close();
          return;
        }
        while (pending.length < opts.chunk) {
          const { done, value } = await upstream.read();
          if (done) break;
          const merged = new Uint8Array(pending.length + value.length);
          merged.set(pending);
          merged.set(value, pending.length);
          pending = merged;
        }
        if (pending.length === 0) {
          controller.close();
          return;
        }
        const chunk = pending.slice(0, opts.chunk);
        pending = pending.slice(chunk.length);
        if (opts.bytesPerSec > 0) await sleep((chunk.length / opts.bytesPerSec) * 1000);
        sent += chunk.length;
        opts.onBytes(sent);
        controller.enqueue(chunk);
      },
      cancel: () => upstream.cancel(),
    },
    { highWaterMark: 0 },
  );
  return new Response(body);
}

// --- assets -------------------------------------------------------------------------------------------------

interface AssetDef {
  label: string;
  /** The coarse-first file as a fresh Response each run (a real fetch, or a page bake behind a Blob URL). */
  open(): Promise<{ response: Response; total: number; note: string }>;
}

async function run(): Promise<void> {
  const [bakedOk, fp8Ok, smokeOk, skyOk] = await Promise.all([probe(WDAS_BAKED), probe(WDAS_FP8), probe(SMOKE_VDB), probe(SKY_URL)]);
  const bakes = new Map<AssetId, Uint8Array>();
  const bakeOnce = async (id: AssetId, make: () => Promise<Uint8Array>): Promise<Uint8Array> => {
    let bytes = bakes.get(id);
    if (!bytes) {
      bytes = await make();
      bakes.set(id, bytes);
    }
    return bytes;
  };
  let wvdbBake: Uint8Array | null = null;
  const bakeWvdbOnce = async (make: () => Promise<Uint8Array>): Promise<Uint8Array> => (wvdbBake ??= await make());
  const blobResponse = async (bytes: Uint8Array): Promise<Response> => fetch(URL.createObjectURL(new Blob([bytes as BlobPart], { type: "application/octet-stream" })));
  const ASSETS: Partial<Record<AssetId, AssetDef>> = {};
  if (bakedOk || fp8Ok) {
    ASSETS.wdas = {
      label: "WDAS cloud (quarter)",
      async open() {
        if (bakedOk) {
          const response = await fetch(WDAS_BAKED);
          if (!response.ok) throw new Error(`${WDAS_BAKED}: ${response.status}`);
          return { response, total: Number(response.headers.get("content-length") ?? 0), note: "wdas_cloud_quarter.coarse.nvdb (baked at dev time)" };
        }
        const bytes = await bakeOnce("wdas", async () => {
          shell.status("baking the WDAS cloud coarse-first in the page (a few seconds; bake it once with scripts/bake-coarse-first.mjs)…");
          const image = NvdbFile.fromArrayBuffer(await (await fetch(WDAS_FP8)).arrayBuffer()).gridImage(0);
          await sleep(30); // paint the status before the bake blocks
          return vdbBakeCoarseFirst(image, { previewResolution: 48 });
        });
        shell.status("");
        return { response: await blobResponse(bytes), total: bytes.length, note: "wdas_cloud_quarter_fp8.nvdb, baked in the page" };
      },
    };
  } else state.missing.push("WDAS cloud: pnpm fixtures:wdas, then nanovdb_convert --fp8 (fixtures/README.md)");
  if (smokeOk) {
    ASSETS.smoke = {
      label: "smoke.vdb",
      async open() {
        const bytes = await bakeOnce("smoke", async () => {
          shell.status("baking smoke.vdb coarse-first in the page…");
          const vdb = parseVdb(await (await fetch(SMOKE_VDB)).arrayBuffer());
          await sleep(30);
          // As the WDAS bake: an FpN preview, an Fp8 full grid (the swap changes the grid type).
          return vdbBakeCoarseFirst(vdb, { previewResolution: 48, fullQuantize: "fp8" });
        });
        shell.status("");
        return { response: await blobResponse(bytes), total: bytes.length, note: "smoke.vdb, baked in the page" };
      },
    };
  } else state.missing.push("smoke.vdb: pnpm fixtures:vdb-samples");
  state.available = Object.keys(ASSETS) as AssetId[];
  if (!skyOk) state.missing.push("sky-midday-HDR.jpg (fixtures/gallery/): procedural sky");
  if (state.missing.length) shell.note(`Not found:\n${state.missing.map((m) => `• ${m}`).join("\n")}`);
  if (state.available.length === 0) throw new Error("No streaming asset is present: pnpm fixtures:vdb-samples (smoke.vdb) or pnpm fixtures:wdas.");

  const asked = params.get("asset") as AssetId | null;
  const controls = {
    asset: asked && ASSETS[asked] ? asked : state.available[0]!,
    speed: params.has("speed") ? Math.max(0, Number(params.get("speed"))) : TEST ? 0 : 2048,
    quality: TEST ? ("low" as Quality) : initialQuality(),
    temporal: params.has("temporal") ? params.get("temporal") === "1" : PRESENTATION === "direct",
    format: state.format,
    partial: (PARTIAL_MODES as readonly string[]).includes(params.get("partial") ?? "") ? (params.get("partial") as PartialMode) : ("upsample" as PartialMode),
  };

  // --- renderer and stage -----------------------------------------------------------------------------------
  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;
  const created = await createDeviceRenderer(directCanvas ? { canvas: directCanvas } : {});
  const { renderer, device } = created;
  device.addEventListener("uncapturederror", (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    state.uncaptured.push(message);
    console.error(`WebGPU: ${message}`);
  });
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") shell.fail(`WebGPU device lost: ${info.message}`);
  });
  state.adapter = created.adapterName || "unreported adapter";
  // `?test=1`: count shader modules and pipelines (a swap must build none).
  let builds = 0;
  if (TEST) {
    const d = device as unknown as Record<string, (...args: unknown[]) => unknown>;
    for (const name of ["createShaderModule", "createRenderPipeline", "createRenderPipelineAsync", "createComputePipeline", "createComputePipelineAsync"]) {
      const original = d[name]!.bind(device);
      d[name] = (...args: unknown[]) => {
        builds++;
        return original(...args);
      };
    }
  }
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.9;
  const presenter = new Presenter(shell.app, renderer, directCanvas, device);
  const frameRenderer = new VDBFrameRenderer();

  const scene = new THREE.Scene();
  const sun = new THREE.DirectionalLight(0xfff4e6, 1.8);
  sun.position.set(250, 110, 60); // from the side: the cloud shades across its face
  scene.add(sun, sun.target);
  if (skyOk) {
    const sky = await loadHdrSky(SKY_URL);
    scene.background = sky;
    scene.environment = sky;
    scene.environmentIntensity = 0.3;
    state.sky = "sky-midday-HDR.jpg";
  } else {
    const sky = proceduralSky({ zenith: "#2f6fd6", horizon: "#b8d4f2", ground: "#59636e" });
    scene.background = sky;
    scene.environment = sky;
    scene.environmentIntensity = 0.6;
    state.sky = "procedural";
  }

  const camera = new THREE.PerspectiveCamera(40, 1, 1, 8000);
  const orbit = new OrbitControls(camera, presenter.canvas);
  orbit.enableDamping = PRESENTATION === "direct";
  orbit.minDistance = 60;
  orbit.maxDistance = 900;
  /** The start view; `far` > 1 backs off for a tall asset (the smoke plume). */
  const home = (far = 1) => {
    camera.position.set(-95 * far, -15, 100 * far);
    orbit.target.set(0, -5, 0);
    orbit.update();
  };
  home();
  if (!STILL_MODE) autoOrbit(orbit, 0.4);
  const resize = () => presenter.resize(QUALITY[controls.quality].pixelRatio, camera);
  resize();
  window.addEventListener("resize", resize);

  /** The volume's parent: places the grid's world box (longest side SIZE) at the origin. */
  const holder = new THREE.Group();
  scene.add(holder);
  const place = (box: NvdbGridMetadata["worldBBox"]) => {
    const size = box.max.map((v, i) => v - box.min[i]!);
    const s = SIZE / Math.max(...size);
    holder.scale.setScalar(s);
    holder.position.set(...(box.min.map((v, i) => -(v + size[i]! / 2) * s) as [number, number, number]));
  };

  function applySettings(volume: VDBFrameVolume | null): void {
    const q = QUALITY[controls.quality];
    const st = frameRenderer.settings;
    st.width = q.width;
    st.height = q.height;
    st.depth = q.depth;
    st.subSteps = q.subSteps;
    st.shadowResolution = q.shadowResolution;
    st.lighting = "frame";
    st.multipleScattering = "neural";
    st.temporal = controls.temporal;
    st.jitter = controls.temporal;
    st.phase = "dual-lobe";
    st.phaseG = 0.8;
    st.phaseBackG = -0.3;
    st.phaseBackWeight = 0.2;
    if (volume) volume.shadowSteps = q.shadowSteps;
  }

  // --- one run: a pool, the stream, one node + source + volume made at the first grid --------------------------
  interface Shown {
    volume: VDBFrameVolume;
    /** The sparse path's node, source and pool id (`.nvdb`). */
    sparse?: { grids: VDBGridNode; source: VDBThreeSparseSource; id: number };
    /** The dense path's source (`.wvdb`, K4.22). */
    dense?: VDBThreeDenseSource;
  }
  let pool: VDBGridPool | null = null;
  let shown: Shown | null = null;
  let abort: AbortController | null = null;
  let generation = 0;

  // One render at a time: the swap work and the loop both draw.
  let chain: Promise<unknown> = Promise.resolve();
  const serial = <T>(fn: () => Promise<T>): Promise<T> => {
    const next = chain.then(fn, fn);
    chain = next.catch(() => undefined);
    return next;
  };

  let frameMs = 0;
  /** Draws a frame; with the readback presentation, returns its sRGB bytes. */
  const present = async (): Promise<Uint8ClampedArray<ArrayBuffer> | null> => {
    const t0 = performance.now();
    const bytes = await presenter.present(scene, camera);
    const ms = performance.now() - t0;
    frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
    state.frames++;
    return bytes;
  };

  /**
   * Show pool id `id`: one node (decoding `types`, every grid type the id will hold), a source that follows the id's
   * image (`followImage`: a later swap re-reads its boxes) and a volume, kept for the rest of the run.
   */
  function showGrid(p: VDBGridPool, id: number, image: Uint32Array, types?: VDBGridTypeName[]): void {
    const grids = new VDBGridNode(renderer, p, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"], types });
    const source = vdbSparseSource(grids, { density: id }, { lod: "coarse-pyramid", images: { density: image }, followImage: true });
    const volume = new VDBFrameVolume(source, { frameRenderer, extinction: 100, angstromExponent: 0, normalizeDensity: true });
    applySettings(volume);
    holder.add(volume);
    shown = { volume, sparse: { grids, source, id } };
  }
  /** K4.22: show a `.wvdb` LOD (whole or partial): one dense source for the stream, each LOD or page swapped in. */
  function showDense(v: WvdbDenseVolume): void {
    if (shown?.dense) {
      shown.dense.replace(v);
      return;
    }
    const dense = vdbDenseTextures(v);
    const volume = new VDBFrameVolume(dense, { frameRenderer, extinction: 100, angstromExponent: 0, normalizeDensity: true });
    applySettings(volume);
    holder.add(volume);
    shown = { volume, dense };
  }
  function disposeShown(s: Shown): void {
    holder.remove(s.volume);
    s.volume.dispose();
    s.sparse?.source.dispose();
    s.sparse?.grids.dispose();
    s.dense?.dispose();
  }

  const tl: Timeline = { samples: [], total: 0, expectedSeconds: 1, markers: [] };
  let t0 = 0;
  let total = 0;
  const showCounter = () => {
    const pct = total > 0 ? Math.min(100, (state.transferred / total) * 100) : 0;
    const unit = total < 1048576 ? KiB : MiB;
    counterEl.textContent = `received ${unit(state.transferred)} of ${unit(total)} (${pct.toFixed(pct < 10 ? 1 : 0)}%) · link ${speedLabel(controls.speed)}`;
  };

  async function startRun(): Promise<void> {
    const gen = ++generation;
    abort?.abort();
    abort = new AbortController();
    const signal = abort.signal;
    // Let the previous run's swap work drain, then clear its pieces.
    await serial(async () => {
      if (shown) disposeShown(shown);
      shown = null;
      pool?.dispose();
      pool = null;
    });
    state.ready = false;
    state.done = false;
    state.phases = [];
    state.transferred = 0;
    tl.samples = [];
    tl.markers = [];
    delete state.wvdb;
    state.format = controls.format;
    if (controls.format === "wvdb") {
      await startWvdbRun(gen, signal);
      return;
    }
    const asset = ASSETS[controls.asset]!;
    if (state.asset !== controls.asset) home(controls.asset === "smoke" ? 1.5 : 1);
    state.asset = controls.asset;
    setPhase("loading", MODE === "direct" ? "loading the full grid…" : "connecting…");
    const opened = await asset.open();
    if (gen !== generation) return;
    total = opened.total;
    state.totalBytes = total;
    tl.total = total;
    tl.expectedSeconds = controls.speed > 0 ? total / (controls.speed * 1024) : 1;
    const p = await VDBGridPool.create(getDevice(renderer), { label: "demo-16" });
    pool = p;

    if (MODE === "direct") {
      // The reference: the full grid added directly, placed by the preview's box as the stream places it.
      const file = NvdbFile.fromArrayBuffer(await opened.response.arrayBuffer());
      const fullIndex = file.grids.findIndex((g) => !/\.lod\d+$/.test(g.name));
      place(file.grids[0]!.worldBBox);
      const image = file.gridImage(fullIndex);
      const id = p.add(image);
      p.flush();
      await p.settled();
      await serial(async () => {
        showGrid(p, id, image); // the pool's types (the full grid's only)
        await present();
        await present();
      });
      state.reference = presenter.capture();
      state.transferred = total;
      setPhase("full", `FULL RESOLUTION · ${file.grids[fullIndex]!.gridType} (direct, no stream)`);
      showCounter();
      state.done = true;
      state.ready = true;
      return;
    }

    let held: Promise<void> | null = null;
    let versionAfterPreview = -1;
    const speed = controls.speed * 1024;
    t0 = performance.now();
    const throttled = throttle(opened.response, {
      chunk: TEST ? 1024 : speed > 0 ? Math.max(1024, Math.round(speed / 60)) : 65536,
      bytesPerSec: speed,
      gate: async () => {
        if (held) await held;
      },
      onBytes: (sent) => {
        if (gen !== generation) return;
        state.transferred = sent;
        tl.samples.push({ t: (performance.now() - t0) / 1000, bytes: sent });
        showCounter();
      },
      signal,
    });
    setPhase("loading", "streaming… waiting for the preview grid");
    showCounter();

    const onGrid = (e: VDBCoarseFirstEvent): void => {
      // Arrival: the file bytes the reader had when the grid completed (no HTTP compression here: wire = file bytes).
      const arrived = (performance.now() - t0) / 1000;
      const bytes = e.bytesReceived;
      const phase = e.final ? "full" : "preview";
      tl.markers.push({ t: arrived, bytes, kind: phase });
      held = serial(async () => {
        if (gen !== generation) return;
        const buildsBefore = builds;
        // The new grid is in the pool's staging: write it.
        p.flush();
        await p.settled();
        if (!e.final) versionAfterPreview = p.version;
        if (e.action === "add") {
          // The first grid: the node (every type of the family), the following source and the volume.
          place(e.grid.metadata.worldBBox);
          showGrid(p, e.id, e.grid.image, e.types);
        } else if (shown?.sparse) {
          // A swap: the source re-reads the boxes at the next frame by itself. Rebind after a growth (none here: the
          // full size is reserved), and read the new density mean back first so the first frame is normalized by it.
          shown.sparse.grids.update();
          await shown.sparse.source.refreshStats();
        }
        await present(); // the first frame (compiles at the first grid)
        await present(); // a settled frame
        // `?test=1`: let the first grid's lazy passes (the environment lights' readback) finish compiling before the swap.
        for (let i = 0; TEST && !e.final && i < 8; i++) {
          const before = builds;
          await present();
          if (builds === before) break;
        }
        const info = p.info(e.id);
        state.phases.push({
          phase,
          grid: e.grid.metadata.name,
          gridType: e.grid.metadata.gridType,
          transferBytes: bytes,
          seconds: arrived,
          imageBytes: info.imageBytes,
          voxelRatio: e.voxelRatio,
          snapshot: presenter.capture(),
          builds: builds - buildsBefore,
        });
        if (e.final) {
          setPhase("full", `FULL RESOLUTION · ${e.grid.metadata.gridType}, ${MiB(info.imageBytes)} · in at ${MiB(bytes)}, ${arrived.toFixed(1)} s`);
          shell.flash("full", "full-resolution grid swapped in");
        } else {
          const res = Math.max(...e.grid.metadata.indexBBox.max.map((v, i) => v - e.grid.metadata.indexBBox.min[i]! + 1));
          setPhase("preview", `PREVIEW · ${res}-voxel ${e.grid.metadata.gridType} · in at ${KiB(bytes)}, ${(arrived * 1000).toFixed(0)} ms`);
          shell.flash("preview", `preview from the first ${KiB(bytes)}`);
        }
      });
      held.catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
    };

    try {
      const result = await vdbStreamCoarseFirst(p, throttled, { onGrid, signal });
      await held;
      if (gen !== generation) return;
      state.sameId = result.shown.every((s) => s.id === result.id);
      state.inPlace = p.version === versionAfterPreview;
      state.done = true;
      state.ready = true;
    } catch (err) {
      if (signal.aborted || gen !== generation) return;
      throw err;
    }
  }

  // --- K4.22: a .wvdb LOD stream into a dense source, page by page -------------------------------------------

  /** FNV-1a of bytes (a frame's identity across runs). */
  const fnv = (bytes: ArrayLike<number>): string => {
    let h = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i]!, 0x01000193);
    return (h >>> 0).toString(16).padStart(8, "0");
  };
  /** Mean squared error over RGB of two sRGB frames, over every pixel or those in `mask`. */
  const mseOf = (a: ArrayLike<number>, b: ArrayLike<number>, mask: Uint8Array | null): number => {
    let sum = 0;
    let n = 0;
    for (let p = 0; p * 4 < a.length; p++) {
      if (mask && !mask[p]) continue;
      for (let c = 0; c < 3; c++) {
        const d = a[p * 4 + c]! - b[p * 4 + c]!;
        sum += d * d;
      }
      n += 3;
    }
    return n > 0 ? sum / n : 0;
  };
  const psnrOf = (mse: number) => (mse > 0 ? 10 * Math.log10((255 * 255) / mse) : 100);
  /**
   * Draws until two frames in a row are byte-identical (the frame renderer's lighting state catches up over a few
   * frames), at most `max`; returns the last frame and how many were drawn. Direct presentation: two frames.
   */
  const settle = async (max = 10): Promise<{ pixels: Uint8ClampedArray | null; frames: number }> => {
    let pixels = await present();
    let hash = pixels ? fnv(pixels) : "";
    for (let n = 2; n <= max; n++) {
      const next = await present();
      if (!next) return { pixels: null, frames: n };
      const h = fnv(next);
      pixels = next;
      if (h === hash) return { pixels, frames: n };
      hash = h;
    }
    return { pixels, frames: max + 1 };
  };

  async function startWvdbRun(gen: number, signal: AbortSignal): Promise<void> {
    if (!smokeOk) throw new Error("format=wvdb streams smoke.vdb baked as .wvdb: pnpm fixtures:vdb-samples");
    if (state.asset !== "smoke") home(1.5);
    state.asset = "smoke";
    const partial = controls.partial;
    setPhase("loading", "baking smoke.vdb as .wvdb in the page…");
    const bytes = await bakeWvdbOnce(async () => {
      const vdb = parseVdb(await (await fetch(SMOKE_VDB)).arrayBuffer());
      await sleep(30);
      return vdbToWvdb(vdb, { ...WVDB_BAKE, fileName: "smoke.vdb" }).bytes;
    });
    if (gen !== generation) return;
    total = bytes.length;
    state.totalBytes = total;
    tl.total = total;
    tl.expectedSeconds = controls.speed > 0 ? total / (controls.speed * 1024) : 1;
    const header = readWvdb(bytes, { maxLod: 0 }).header;
    const g0 = header.grids[0]!;
    const lods = (g0.lods as { dimensions?: number[] }[]).map((l) => [...(l.dimensions ?? [])]);
    const box = (v: WvdbDenseVolume) => ({ min: v.bbox![0], max: v.bbox![1] }) as unknown as NvdbGridMetadata["worldBBox"];

    if (MODE === "direct") {
      const v = wvdbDenseVolume(readWvdb(bytes), { mipmaps: true });
      place(box(v));
      let pixels: Uint8ClampedArray | null = null;
      await serial(async () => {
        showDense(v);
        pixels = (await settle()).pixels;
      });
      state.reference = presenter.capture();
      if (pixels) state.referenceHash = fnv(pixels);
      state.transferred = total;
      setPhase("full", `FULL RESOLUTION · LOD ${lods.length - 1}, ${lods.at(-1)!.join("×")} (direct, no stream)`);
      showCounter();
      state.done = true;
      state.ready = true;
      return;
    }

    const run: WvdbRun = { partial, frames: [], region: 0, finalHash: "", lods, fills: null };
    state.wvdb = run;
    let held: Promise<void> | null = null;
    const speed = controls.speed * 1024;
    t0 = performance.now();
    const throttled = throttle(await blobResponse(bytes), {
      chunk: TEST ? 1024 : speed > 0 ? Math.max(1024, Math.round(speed / 60)) : 65536,
      bytesPerSec: speed,
      gate: async () => {
        if (held) await held;
      },
      onBytes: (sent) => {
        if (gen !== generation) return;
        state.transferred = sent;
        tl.samples.push({ t: (performance.now() - t0) / 1000, bytes: sent });
        showCounter();
      },
      signal,
    });
    setPhase("loading", "streaming… waiting for LOD 0");
    showCounter();
    // `?test=1`: the frame with no volume, so the PSNR region is the volume's pixels.
    const empty = TEST ? (await serial(() => settle())).pixels : null;
    const pixels: Uint8ClampedArray[] = [];
    const reader = new WvdbStreamReader({ partialLods: true });
    const body = throttled.body!.getReader();
    let shownLod = -1;
    for (;;) {
      const { done, value } = await body.read();
      if (done || signal.aborted || gen !== generation) break;
      const events = reader.push(value);
      if (events.length === 0) continue;
      // Events share the reader's live levels (Traps): build the volume now, before the next push.
      const e = events.at(-1)!;
      const level = e.levels[0] as WvdbVolumeLevel;
      let stored = level.brickIndices.length;
      if (level.mask) {
        stored = 0;
        for (let b = 0; b < level.brickTotal; b++) if (level.mask[b >>> 5]! & (1 << (b & 31))) stored++;
      }
      // `off`: only whole LODs (the last one this push completed, if any); else the push's last event, partial or not.
      const showEvent = partial === "off" ? events.filter((x) => x.complete).at(-1) : e;
      const show = showEvent
        ? wvdbDensePartialVolume(reader.file!, showEvent, { mipmaps: true, fill: partial === "background" ? "background" : "upsample", fillMips: partial === "upsample-box" ? "box" : "coarse" })
        : null;
      const record = { lod: e.lod, complete: e.complete, decoded: level.bricksDecoded, stored, bytes: reader.bytesReceived };
      const arrived = (performance.now() - t0) / 1000;
      if (e.complete) tl.markers.push({ t: arrived, bytes: reader.bytesReceived, kind: e.lod === lods.length - 1 ? "full" : "preview" });
      held = serial(async () => {
        if (gen !== generation) return;
        const buildsBefore = builds;
        if (show) {
          if (!shown) place(box(show));
          showDense(show);
          shownLod = showEvent!.lod;
        }
        // Nothing whole yet (`off` before LOD 0): the frame without a volume, so every mode has a frame per event.
        const { pixels: settled, frames: drawn } = await settle();
        const page = e.complete ? "complete" : `${record.decoded}/${record.stored} bricks`;
        const kind = e.complete && e.lod === lods.length - 1 ? "full" : "preview";
        setPhase(kind, `LOD ${e.lod} (${lods[e.lod]!.join("×")}) · ${page}${!e.complete ? ` · ${PARTIAL_LABEL[partial]}` : ""} · in at ${KiB(record.bytes)}`);
        if (TEST && settled) {
          pixels.push(new Uint8ClampedArray(settled));
          run.frames.push({ ...record, shownLod, filled: !!show?.density.fill, mse: 0, psnr: 0, regionMse: 0, regionPsnr: 0, builds: builds - buildsBefore, drawn });
        }
      });
      held.catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
    }
    await held;
    if (signal.aborted || gen !== generation) return;
    reader.end();
    const core = shown ? (frameRenderer.coreVolumeOf(shown.volume)?.slot.source as { fills?: { gpu: number; cpu: number } } | undefined) : undefined;
    run.fills = core?.fills ? { ...core.fills } : null;
    shell.flash("full", "every LOD in");
    if (TEST && pixels.length > 0) {
      const final = pixels.at(-1)!;
      let mask: Uint8Array | null = null;
      if (empty) {
        mask = new Uint8Array(final.length / 4);
        for (let p = 0; p < mask.length; p++) {
          for (let c = 0; c < 3; c++) if (Math.abs(final[p * 4 + c]! - empty[p * 4 + c]!) > 2) mask[p] = 1;
          run.region += mask[p]!;
        }
      }
      run.frames.forEach((f, i) => {
        f.mse = mseOf(pixels[i]!, final, null);
        f.psnr = psnrOf(f.mse);
        f.regionMse = mseOf(pixels[i]!, final, mask);
        f.regionPsnr = psnrOf(f.regionMse);
      });
      run.finalHash = fnv(final);
    }
    state.done = true;
    state.ready = true;
  }

  // --- controls ---------------------------------------------------------------------------------------------
  const restart = () => {
    startRun().catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
  };
  shell.select<AssetId>("asset", "asset", state.available.map((id) => ({ value: id, label: ASSETS[id]!.label })), controls.asset, (v) => {
    controls.asset = v;
    restart();
  });
  const speeds = [...new Set([...SPEEDS, controls.speed])].sort((a, b) => a - b);
  shell.select("speed", "link", speeds.map((k) => ({ value: String(k), label: speedLabel(k) })), String(controls.speed), (v) => {
    controls.speed = Number(v);
    restart();
  });
  shell.select<Format>("format", "format", [{ value: "nvdb", label: ".nvdb coarse-first" }, { value: "wvdb", label: ".wvdb LODs (dense)" }], controls.format, (v) => {
    controls.format = v;
    restart();
  });
  shell.select<PartialMode>("partial", "while a LOD streams", PARTIAL_MODES.map((m) => ({ value: m, label: PARTIAL_LABEL[m] })), controls.partial, (v) => {
    controls.partial = v;
    if (controls.format === "wvdb") restart();
  });
  shell.select<Quality>("quality", "quality", (["low", "medium", "high"] as const).map((q) => ({ value: q, label: QUALITY[q].label })), controls.quality, (v) => {
    controls.quality = v;
    resize();
    applySettings(shown?.volume ?? null);
  });
  shell.button("restart", "↻ restart", restart);
  window.__DEMO16_API__ = { capture: () => presenter.capture(), restart };
  // `?debug=1`: the scene's pieces, for tuning from the console.
  if (params.get("debug") === "1") {
    Object.assign(window, {
      __demo16: { THREE, scene, camera, orbit, sun, controls, applySettings, frameRenderer, renderer, shown: () => shown },
    });
  }

  // --- loop -------------------------------------------------------------------------------------------------
  let inFlight = false;
  let lastStats = 0;
  const loop = (now: number): void => {
    requestAnimationFrame(loop);
    drawTimeline(tl);
    if (!shown || state.error || inFlight || TEST) return;
    inFlight = true;
    orbit.update();
    serial(present)
      .catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)))
      .finally(() => {
        inFlight = false;
      });
    if (now - lastStats > 400) {
      lastStats = now;
      const info = pool && shown?.sparse ? pool.info(shown.sparse.id) : null;
      shell.stats(
        `${state.adapter} · ${presenter.width}×${presenter.height} px · ${frameMs.toFixed(1)} ms/frame (provisional, D25)\n` +
        (info ? `pool grid ${info.type}, ${MiB(info.imageBytes)} (capacity ${MiB(info.capacity)}) · ` : "") +
        (controls.format === "wvdb" ? "smoke.vdb as .wvdb (dense source)" : `${ASSETS[controls.asset]!.label}`),
      );
    }
  };
  requestAnimationFrame(loop);
  applySettings(null);
  await startRun();
}

run().catch((err: unknown) => shell.fail(err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err)));
