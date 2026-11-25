/**
 * K3.6 harness (R1 §6 Step 6): volume shadows on meshes (`settings.castShadows`) and the emission light
 * (`VDBFrameEmissionLight`) on three r187. K6.8: every frame is the core frame renderer through `./three` (the core's
 * volume shadow maps and emission probes, received by three); the TSL pass twins that read three's targets are gone with
 * the TSL passes (the core kernels have their CPU twins in the k6-core harness: `?probe=shadow`, `?probe=emission`).
 *
 * `?case=shadow` (default): the committed CC0 cloud above a ground plane (`ShadowMaterial` opacity 0.5, or
 * `?ground=standard`), a sun (`?light=spot`: a spot light) with `castShadow`. Frames: shadows on, shadows off
 * (`castShadows` false), the volume alone (ground hidden: its screen coverage); `shadowStats` over the ground.
 * `?source=sparse` (the same texels as a NanoVDB grid), `?lighting=frame`, `?cascades=N`, `?mapSize=`, `?steps=`.
 *
 * `?case=emission`: the committed CC0 ground explosion frame 10 (EmberGen; his explosion settings: Z-up, 100 units,
 * flames ÷ 6 as emission, 0–3000 K, his 'planck' spectrum (`emissionColor: { spectrum: 'planck' }`), intensity 15) over a
 * white `MeshLambertMaterial` ground lit only by a `VDBFrameEmissionLight` (intensity 25). `?emit=0` hides the light.
 * `?source=sparse`, `?probes=`, `?spectrum=white`, `?emitter=blob` (a smooth synthetic Gaussian emitter).
 *
 * `?case=stale` (K7.1, his 62a6e3e): the shadow case's cloud and ground with a spot (`?light=spot`, default) or point
 * (`?light=point`) light, and a decoy spot light whose empty map is older. Both lights move in one frame; the budget
 * (1) draws the decoy, so the main light's map waits. Receivers must read that map with the state it was drawn with:
 * the waiting frame's ground equals the frame before the move, pixel for pixel; the frame after (the map drawn) does not.
 * Then the main light alone moves (its map drawn in that render): that frame equals the next one, nothing changed.
 *
 * Results: `window.__VDB_K3_SHADOWS__` (e2e/k3-frame-shadows.spec.ts, k6-shadows.spec.ts, k7-volume-shadow-stale.spec.ts).
 */
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBFrameEmissionLight, VDBFrameRenderer, VDBFrameVolume, VDBGridNode, vdbDenseTextures, vdbSparseSource, type VDBThreeVolumeSource } from "../../../packages/vdb/src/three/internal.js";
import { readWvdb, wvdbDenseToNvdb, wvdbDenseVolume, writeWvdb, vdbSrgbOetf, type WvdbDenseVolume } from "../../../packages/vdb/src/tools/internal.js";
import { readAtlas } from "./k3-frame-lighting.js";
import cloudUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";
import explosionUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/ground_explosion_0010.wvdb?url";

type V3 = [number, number, number];

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  case?: string;
  width?: number;
  height?: number;
  rgba?: string;
  /** The castShadows-off frame (shadow case) or the light-off frame (emission), display RGBA8. */
  rgbaOff?: string;
  inPageIdentical?: boolean;
  warnings: string[];
  uncaptured: string[];
  settings?: Record<string, unknown>;
  /** K6.5: the volume's screen coverage dilated by 2 px (W × H bytes, 1 = volume), for A/Bs of the terms on meshes. */
  coverage?: string;
  /** K6.5: which backend drew (the core renderer exists, its maps and probes). */
  backend?: { requested: string; core: boolean; maps: number; probes: boolean };
  sourceKind?: string;
  // shadow case
  entry?: { size: number; cascades: number; breaks: number[]; far: number[]; drawn: number; waiting: number } | null;
  shadowStats?: { groundPixels: number; covered: number; shadowedPixels: number; meanDarkening: number };
  /**
   * K6.8a `?mutate=`: the frame after the light (or map size) came back vs the first; the shadow while it was away. K6.8b
   * (emission): `awayShadowed` counts the pixels that changed while away; `mapsAway` is 1 when the core had probes then.
   */
  mutate?: { kind: string; identical: boolean; maxDiff: number; awayShadowed: number; mapsAway: number };
  // emission case
  emitters?: number;
  probeGrid?: { resolution: number; count: number; min: number[][]; size: number[][]; center: number[] };
  lightOff?: { groundPixels: number; nonBlack: number };
  // stale case
  stale?: {
    groundPixels: number;
    shadowedBefore: number;
    staleChanged: number;
    staleShadowedChanged: number;
    afterChanged: number;
    movedChanged: number;
    order: { decoy: number; main: number };
    staleUpdate: { drawn: number; waiting: number };
    afterUpdate: { drawn: number; waiting: number };
    movedUpdate: { drawn: number; waiting: number };
    mainDrawnAtStale: boolean;
  };
}

declare global {
  interface Window {
    __VDB_K3_SHADOWS__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const CASE = (params.get("case") ?? "shadow") as "shadow" | "emission" | "stale";
const W = Number(params.get("w") ?? 160);
const H = Number(params.get("h") ?? 120);
const SOURCE_KIND = (params.get("source") ?? "dense") as "dense" | "sparse";
const LIGHTING = (params.get("lighting") ?? "volume") as "volume" | "frame";
const LIGHT = (params.get("light") ?? (CASE === "stale" ? "spot" : "dir")) as "dir" | "spot" | "point";
const GROUND = (params.get("ground") ?? "shadow") as "shadow" | "standard";
const CAST = params.get("cast") !== "0";
const CASCADES = Number(params.get("cascades") ?? 1);
const MAP_SIZE = Number(params.get("mapSize") ?? 512);
const STEPS = Number(params.get("steps") ?? 32);
// K6.8: the core is the only renderer (`?backend=` is ignored)
const BACKEND = "core";
const EMIT = params.get("emit") !== "0";
const PROBES = Number(params.get("probes") ?? 16);
const SPECTRUM = params.get("spectrum") ?? "planck";
const MIPS = params.get("mips") !== "0";
const BLOB = params.get("emitter") === "blob";
// K6.8a: `?spectral=1` spectral maps (the receivers' rgb WGSL; per-channel extinction); `?mutate=toggle|resize` after the
// measured frames: hide the light for some renders (the core frees its map and retires its texture and view buffer) or
// change the map size, then restore it; the frame after must equal the first and nothing may bind a retired object
// (K6.10 finding 8). K6.8b, the emission case: `?mutate=probes` (the probe resolution halved: the core makes a new probe
// stack and retires the old one), `?mutate=light` (the emission light hidden: the core releases its probes) or
// `?mutate=emitter` (the emissive volume hidden: no emitters, then back), each for some renders, then restored.
const SPECTRAL = params.get("spectral") === "1";
const MUTATE = params.get("mutate") as "toggle" | "resize" | "probes" | "light" | "emitter" | null;
const BACKGROUND = new THREE.Color(CASE === "emission" ? 0x000000 : 0x598eff);
const SHADOW_OPACITY = 0.5;

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [] };
window.__VDB_K3_SHADOWS__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

function toBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** Linear frame, rows top first (from readAtlas's bottom-first rows). */
async function readLinear(renderer: WebGPURenderer, rt: THREE.RenderTarget): Promise<Float32Array> {
  const a = await readAtlas(renderer, rt, 4);
  const out = new Float32Array(a.data.length);
  const row = W * 4;
  for (let y = 0; y < H; y++) out.set((a.data as Float32Array).subarray((H - 1 - y) * row, (H - y) * row), y * row);
  return out;
}

const toByte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);

function toDisplay(linear: Float32Array): Uint8Array {
  const out = new Uint8Array(linear.length);
  for (let i = 0; i < linear.length; i += 4) {
    out[i] = toByte(vdbSrgbOetf(linear[i]!));
    out[i + 1] = toByte(vdbSrgbOetf(linear[i + 1]!));
    out[i + 2] = toByte(vdbSrgbOetf(linear[i + 2]!));
    out[i + 3] = 255;
  }
  return out;
}

/** The world point of pixel (x, y top) on the plane y = planeY, or null. */
function groundPoint(camera: THREE.PerspectiveCamera, x: number, y: number, planeY: number, half: number, sub: [number, number] = [0.5, 0.5]): V3 | null {
  const ndc = new THREE.Vector3(((x + sub[0]) / W) * 2 - 1, 1 - ((y + sub[1]) / H) * 2, 0.5).unproject(camera);
  const origin = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
  const dir = ndc.sub(origin).normalize();
  if (Math.abs(dir.y) < 1e-9) return null;
  const t = (planeY - origin.y) / dir.y;
  if (!(t > 0)) return null;
  const p = origin.addScaledVector(dir, t);
  if (Math.abs(p.x) > half || Math.abs(p.z) > half) return null;
  return [p.x, p.y, p.z];
}

const v3 = (v: THREE.Vector3): V3 => [v.x, v.y, v.z];

async function makeSource(renderer: WebGPURenderer, device: GPUDevice, dense: WvdbDenseVolume, emission: boolean): Promise<VDBThreeVolumeSource> {
  if (SOURCE_KIND !== "sparse") return vdbDenseTextures(dense);
  // The same texels as NanoVDB FLOAT grids (voxel for voxel), the K3.2 default LOD; the emission grid shares the texture box
  const grid = wvdbDenseToNvdb(dense);
  const pool = await VDBGridPool.create(device, { label: "k3-shadows" });
  const id = pool.add(grid.image);
  const flames = emission ? wvdbDenseToNvdb(dense, { channel: "emission" }) : null;
  const emissionId = flames ? pool.add(flames.image) : undefined;
  pool.flush();
  await pool.settled();
  const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "fastTrilinear"] });
  return vdbSparseSource(grids, emissionId === undefined ? { density: id } : { density: id, emission: emissionId }, {
    lod: "coarse-pyramid",
    images: { density: grid.image },
    indexBox: grid.textureIndexBox,
    ...(flames ? { auxIndexBoxes: { emission: flames.textureIndexBox } } : {}),
    stats: { meanNonZero: dense.meanDensity, max: dense.maxDensity, activeVoxels: dense.activeTexels },
    ...(dense.bbox ? { bbox: dense.bbox as [V3, V3] } : {}),
  });
}

// ── shadow case ──────────────────────────────────────────────────────────────────────────────────────

async function runShadow(renderer: WebGPURenderer, device: GPUDevice): Promise<void> {
  const wvdb = readWvdb(new Uint8Array(await (await fetch(cloudUrl)).arrayBuffer()));
  const dense = wvdbDenseVolume(wvdb, { mipmaps: MIPS });
  const source = await makeSource(renderer, device, dense, false);

  const fr = new VDBFrameRenderer();
  Object.assign(fr.settings, {
    width: 96,
    height: 64,
    depth: 24,
    temporal: false,
    jitter: false,
    lighting: LIGHTING,
    multipleScattering: true,
    castShadows: CAST,
    castShadowMapSize: MAP_SIZE,
    castShadowCascades: CASCADES,
    castShadowSteps: STEPS,
    spectral: SPECTRAL,
  });
  result.settings = { ...fr.settings };

  // The cloud at 100 units (his demo size), its box bottom 20 above the ground (y = 0)
  const bbox = new THREE.Box3(new THREE.Vector3(...(dense.bbox?.[0] ?? [-0.5, -0.5, -0.5])), new THREE.Vector3(...(dense.bbox?.[1] ?? [0.5, 0.5, 0.5])));
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());
  const scale = 100 / Math.max(size.x, size.y, size.z);
  const volume = new VDBFrameVolume(source, {
    frameRenderer: fr,
    bbox,
    extinction: 60,
    shadowSteps: 24,
    mipmaps: MIPS,
    // K6.8a spectral: channels that differ, so the rgb receivers' channels differ too
    ...(SPECTRAL ? { extinctionScale: [1.6, 1.0, 0.5] as [number, number, number], angstromExponent: 0 } : {}),
  });
  volume.scale.setScalar(scale);
  volume.position.copy(center).multiplyScalar(-scale).add(new THREE.Vector3(0, 20 + (size.y * scale) / 2, 0));

  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  scene.add(volume);

  const GROUND_HALF = 300;
  const ground = new THREE.Mesh(
    // Subdivided: SwiftShader interpolates varyings across a large triangle with ~1e-4 relative error (K3.6)
    new THREE.PlaneGeometry(GROUND_HALF * 2, GROUND_HALF * 2, Number(params.get("groundSegments") ?? 64), Number(params.get("groundSegments") ?? 64)),
    GROUND === "standard" ? new THREE.MeshStandardMaterial({ color: 0xbbbbbb, roughness: 1, metalness: 0 }) : new THREE.ShadowMaterial({ opacity: SHADOW_OPACITY }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  let light: THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight;
  if (LIGHT === "spot") {
    const spot = new THREE.SpotLight(0xffffff, 40000, 0, 0.6, 0.0, 2);
    spot.position.set(-60, 260, -110);
    spot.target.position.set(30, 0, 60);
    spot.shadow.camera.near = 10;
    spot.shadow.camera.far = 600;
    light = spot;
  } else if (LIGHT === "point") {
    // K6.8a: a point light where the spot is (its map is a cube: the receivers' cube WGSL)
    const point = new THREE.PointLight(0xffffff, 40000, 0, 2);
    point.position.set(-60, 260, -110);
    point.shadow.camera.near = 10;
    point.shadow.camera.far = 600;
    light = point;
  } else {
    // The sun low enough that the shadow falls beside the volume; its three shadow camera covers the ground (his fitSunShadow)
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(-90, 240, -150);
    sun.target.position.set(0, 0, 0);
    Object.assign(sun.shadow.camera, { left: -450, right: 450, top: 450, bottom: -450, near: -1000, far: 1000 });
    light = sun;
  }
  light.castShadow = true;
  light.shadow.mapSize.setScalar(512);
  light.shadow.camera.updateProjectionMatrix();
  scene.add(light);
  if ("target" in light) scene.add(light.target);
  if (GROUND === "standard") scene.add(new THREE.AmbientLight(0xffffff, 0.2));

  const camera = new THREE.PerspectiveCamera(55, W / H, 1, 5000);
  camera.position.set(70, 190, 220);
  camera.lookAt(25, 10, 45);
  camera.updateProjectionMatrix();
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  const frame = async () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    return readLinear(renderer, rt);
  };

  // The first frame puts the filter on the light (in the proxy's onBeforeRender, after the ground drew); the receivers
  // build with it on the next frame, whose first receiver draws the maps (budget 1: one piece per render).
  for (let i = 0; i < 2 + CASCADES; i++) await frame();
  const on = await frame();
  const onAgain = await frame();
  const display = toDisplay(on);
  const displayAgain = toDisplay(onAgain);
  result.inPageIdentical = display.every((v, i) => v === displayAgain[i]);
  result.width = W;
  result.height = H;
  result.rgba = toBase64(display);
  result.backend = { requested: BACKEND, core: !!fr.core, maps: fr.core?.outputs.volumeShadows.size ?? 0, probes: false };

  // K6.8a (K6.10 finding 8): take the map away and bring it back; the receivers built with it must follow the core
  if (MUTATE === "toggle" || MUTATE === "resize") {
    const away = MUTATE === "toggle" ? () => (light.visible = false) : () => (fr.settings.castShadowMapSize = MAP_SIZE / 2);
    const back = MUTATE === "toggle" ? () => (light.visible = true) : () => (fr.settings.castShadowMapSize = MAP_SIZE);
    away();
    // Several renders (each a task: the readback awaits): the core retires the old texture and view buffer, then destroys them
    let awayFrame = on;
    for (let i = 0; i < 4; i++) awayFrame = await frame();
    const mapsAway = fr.core?.outputs.volumeShadows.size ?? 0;
    let awayShadowed = 0;
    for (let i = 0; i < W * H; i++) if (Math.abs(awayFrame[i * 4 + 1]! - on[i * 4 + 1]!) > 1e-3) awayShadowed++;
    back();
    for (let i = 0; i < 2 + CASCADES; i++) await frame();
    const restored = toDisplay(await frame());
    let maxDiff = 0;
    for (let i = 0; i < restored.length; i++) maxDiff = Math.max(maxDiff, Math.abs(restored[i]! - display[i]!));
    result.mutate = { kind: MUTATE, identical: maxDiff === 0, maxDiff, awayShadowed, mapsAway };
  }

  const entry = fr.volumeShadows.entries.get(light) ?? null;
  // K6.8a: the receivers bind the core's map and its view buffer; the drawn state is the core map's
  const drawnState = entry?.coreMap?.drawn ?? null;
  result.entry =
    entry && drawnState
      ? {
          size: entry.size,
          cascades: drawnState.cascadeCount,
          breaks: [...drawnState.breaks],
          far: drawnState.cascadeCount > 0 ? drawnState.far.slice(0, drawnState.cascadeCount) : [drawnState.far[0] ?? 1],
          drawn: fr.volumeShadows.lastUpdate.drawn,
          waiting: fr.volumeShadows.lastUpdate.waiting,
        }
      : null;

  // The volume's screen coverage: the volume alone (ground hidden) against the background
  ground.visible = false;
  const alone = await frame();
  ground.visible = true;
  const bgLinear = [BACKGROUND.r, BACKGROUND.g, BACKGROUND.b];
  const covered = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (Math.abs(alone[i * 4]! - bgLinear[0]!) + Math.abs(alone[i * 4 + 1]! - bgLinear[1]!) + Math.abs(alone[i * 4 + 2]! - bgLinear[2]!) > 1e-4) covered[i] = 1;
  }
  // Dilate by 2 pixels (the froxel composite's soft edge)
  const coveredWide = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
      if (covered[y * W + x]) for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) if (x + dx >= 0 && x + dx < W && y + dy >= 0 && y + dy < H) coveredWide[(y + dy) * W + x + dx] = 1;
  result.coverage = toBase64(coveredWide);

  // castShadows off: the shadow-only reference (three's own shadow of the light stays: no mesh casts, so 1)
  fr.settings.castShadows = false;
  await frame();
  const off = await frame();
  result.rgbaOff = toBase64(toDisplay(off));

  // Shadow stats over the visible ground (ShadowMaterial: out = background × (1 − opacity × (1 − T)))
  let groundPixels = 0;
  let shadowedPixels = 0;
  let darkening = 0;
  const points: (V3 | null)[] = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const p = coveredWide[i] ? null : groundPoint(camera, x, y, 0, GROUND_HALF - 1);
      points.push(p);
      if (!p) continue;
      groundPixels++;
      const d = off[i * 4 + 1]! - on[i * 4 + 1]!;
      if (d > 1e-3) shadowedPixels++;
      darkening += d;
    }
  }
  result.shadowStats = { groundPixels, covered: covered.reduce((a, b) => a + b, 0), shadowedPixels, meanDarkening: darkening / Math.max(1, groundPixels) };

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(display), W, H), 0, 0);
  root.append(canvas);
  log(`shadow (${LIGHT}, ${SOURCE_KIND}, '${LIGHTING}' lighting): ${JSON.stringify(result.shadowStats)}; entry ${JSON.stringify(result.entry)}; repeat identical ${result.inPageIdentical}`, result.inPageIdentical ? "ok" : "fail");
}

// ── stale case (K7.1) ────────────────────────────────────────────────────────────────────────────────

async function runStale(renderer: WebGPURenderer, device: GPUDevice): Promise<void> {
  const wvdb = readWvdb(new Uint8Array(await (await fetch(cloudUrl)).arrayBuffer()));
  const dense = wvdbDenseVolume(wvdb, { mipmaps: MIPS });
  const source = await makeSource(renderer, device, dense, false);

  const fr = new VDBFrameRenderer();
  Object.assign(fr.settings, {
    width: 96,
    height: 64,
    depth: 24,
    temporal: false,
    jitter: false,
    lighting: LIGHTING,
    multipleScattering: true,
    castShadows: true,
    castShadowMapSize: Number(params.get("mapSize") ?? 256),
    castShadowSteps: STEPS,
    castShadowUpdatesPerFrame: 1,
  });
  result.settings = { ...fr.settings };

  const bbox = new THREE.Box3(new THREE.Vector3(...(dense.bbox?.[0] ?? [-0.5, -0.5, -0.5])), new THREE.Vector3(...(dense.bbox?.[1] ?? [0.5, 0.5, 0.5])));
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());
  const scale = 100 / Math.max(size.x, size.y, size.z);
  const volume = new VDBFrameVolume(source, { frameRenderer: fr, bbox, extinction: 60, shadowSteps: 24, mipmaps: MIPS });
  volume.scale.setScalar(scale);
  volume.position.copy(center).multiplyScalar(-scale).add(new THREE.Vector3(0, 20 + (size.y * scale) / 2, 0));

  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  // The decoy first: its entry is made first, so its map is drawn first and the main light's map is the newer one.
  // It points away from the cloud (an empty map, no shadow on the ground).
  const decoy = new THREE.SpotLight(0xffffff, 100, 0, 0.3, 0.0, 2);
  decoy.position.set(250, 60, 250);
  decoy.target.position.set(290, 0, 290);
  decoy.castShadow = true;
  decoy.shadow.camera.near = 10;
  decoy.shadow.camera.far = 200;
  scene.add(decoy, decoy.target);

  let light: THREE.SpotLight | THREE.PointLight;
  if (LIGHT === "point") {
    const point = new THREE.PointLight(0xffffff, 40000, 0, 2);
    point.shadow.camera.near = 10;
    point.shadow.camera.far = 600;
    light = point;
  } else {
    const spot = new THREE.SpotLight(0xffffff, 40000, 0, 0.6, 0.0, 2);
    spot.target.position.set(30, 0, 60);
    spot.shadow.camera.near = 10;
    spot.shadow.camera.far = 600;
    scene.add(spot.target);
    light = spot;
  }
  light.position.set(-60, 260, -110);
  light.castShadow = true;
  light.shadow.mapSize.setScalar(512);
  scene.add(light, volume);

  const GROUND_HALF = 300;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(GROUND_HALF * 2, GROUND_HALF * 2, 64, 64), new THREE.ShadowMaterial({ opacity: SHADOW_OPACITY }));
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const camera = new THREE.PerspectiveCamera(55, W / H, 1, 5000);
  camera.position.set(70, 190, 220);
  camera.lookAt(25, 10, 45);
  camera.updateProjectionMatrix();
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  const frame = async () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    return readLinear(renderer, rt);
  };
  const lastUpdate = () => ({ ...fr.volumeShadows.lastUpdate });

  // Filters go on, receivers build, then the decoy's map and the main light's map are drawn (one piece per render)
  for (let i = 0; i < 5; i++) await frame();
  const before = await frame();
  const mainEntry = fr.volumeShadows.entries.get(light)!;
  const decoyEntry = fr.volumeShadows.entries.get(decoy)!;
  const order = { decoy: decoyEntry.lastDraw, main: mainEntry.lastDraw };

  // Both lights move: the decoy (the older map) takes the budget, the main light's map waits one render
  const move = Number(params.get("move") ?? 40);
  const mainLastDraw = mainEntry.lastDraw;
  light.position.x += move;
  decoy.position.x += 5;
  const stale = await frame();
  const staleUpdate = lastUpdate();
  const mainDrawnAtStale = mainEntry.lastDraw !== mainLastDraw;
  const after = await frame();
  const afterUpdate = lastUpdate();

  // The main light alone moves: its map is drawn in that render, which must equal the next one (nothing changes)
  light.position.x += move;
  const moved = await frame();
  const movedUpdate = lastUpdate();
  const settled = await frame();

  // The volume's screen coverage at the three light positions (ground hidden: no receivers, so no map draws)
  ground.visible = false;
  const covered = new Uint8Array(W * H);
  const bgLinear = [BACKGROUND.r, BACKGROUND.g, BACKGROUND.b];
  for (const dx of [0, -move, -move]) {
    light.position.x += dx;
    const alone = await frame();
    for (let i = 0; i < W * H; i++) {
      if (Math.abs(alone[i * 4]! - bgLinear[0]!) + Math.abs(alone[i * 4 + 1]! - bgLinear[1]!) + Math.abs(alone[i * 4 + 2]! - bgLinear[2]!) > 1e-4) covered[i] = 1;
    }
  }
  ground.visible = true;

  // Ground pixels 2 px or more from the volume (the froxel composite's soft edge)
  let groundPixels = 0;
  let shadowedBefore = 0;
  let staleChanged = 0;
  let staleShadowedChanged = 0;
  let afterChanged = 0;
  let movedChanged = 0;
  const same =(a: Float32Array, b: Float32Array, i: number) => a[i * 4] === b[i * 4] && a[i * 4 + 1] === b[i * 4 + 1] && a[i * 4 + 2] === b[i * 4 + 2];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      let near = false;
      for (let dy = -2; dy <= 2 && !near; dy++) for (let dx = -2; dx <= 2 && !near; dx++) if (x + dx >= 0 && x + dx < W && y + dy >= 0 && y + dy < H && covered[(y + dy) * W + x + dx]) near = true;
      if (near || !groundPoint(camera, x, y, 0, GROUND_HALF - 1)) continue;
      groundPixels++;
      const shadowed = before[i * 4 + 1]! < bgLinear[1]! - 1e-3;
      if (shadowed) shadowedBefore++;
      if (!same(stale, before, i)) {
        staleChanged++;
        if (shadowed) staleShadowedChanged++;
      }
      if (!same(after, before, i)) afterChanged++;
      if (!same(moved, settled, i)) movedChanged++;
    }
  }
  result.backend = { requested: BACKEND, core: !!fr.core, maps: fr.core?.outputs.volumeShadows.size ?? 0, probes: false };
  result.stale = { groundPixels, shadowedBefore, staleChanged, staleShadowedChanged, afterChanged, movedChanged, order, staleUpdate, afterUpdate, movedUpdate, mainDrawnAtStale };
  result.width = W;
  result.height = H;
  result.rgba = toBase64(toDisplay(stale));
  result.rgbaOff = toBase64(toDisplay(before));
  result.inPageIdentical = true;
  log(`stale (${LIGHT}): ${JSON.stringify(result.stale)}`, staleChanged === 0 ? "ok" : "fail");
}

// ── emission case ────────────────────────────────────────────────────────────────────────────────────

/** A smooth synthetic emitter as a `.wvdb`: Gaussian density (σ n/5) and flames (σ n/8, off-centre), 32³, 100 units. */
function blobWvdb(): Uint8Array {
  const n = 32;
  const density = new Float32Array(n * n * n);
  const flames = new Float32Array(n * n * n);
  for (let z = 0; z < n; z++)
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const i = x + n * (y + n * z);
        const c = (n - 1) / 2;
        density[i] = 0.8 * Math.exp(-((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) / (2 * (n / 5) ** 2));
        flames[i] = 0.9 * Math.exp(-((x - c - 3) ** 2 + (y - c + 2) ** 2 + (z - c) ** 2) / (2 * (n / 8) ** 2));
      }
  const bbox: [[number, number, number], [number, number, number]] = [
    [-50, -50, -50],
    [50, 50, 50],
  ];
  return writeWvdb(
    [
      { name: "density", class: "fog volume", background: 0, bbox, finest: { dimensions: [n, n, n], values: density } },
      { name: "flames", class: "fog volume", background: 0, bbox, finest: { dimensions: [n, n, n], values: flames } },
    ],
    { lods: 1 },
  ).bytes;
}

async function runEmission(renderer: WebGPURenderer, device: GPUDevice): Promise<void> {
  // `?emitter=blob`: a smooth synthetic emitter (Gaussian density and flames, 32³): the strict direction-pass twin. SwiftShader's
  // division is ~12 bits (box segments move by ~2e-4), and the explosion's sharp flames under 8² rays × 24 steps amplify
  // that to several % per direction (its mean over the 16 directions stays within ~1.5%).
  const wvdb = readWvdb(BLOB ? blobWvdb() : new Uint8Array(await (await fetch(explosionUrl)).arrayBuffer()));
  // His explosion: the flames grid as emission, ÷ 6 (his emissiveScale), the density grid
  const dense = wvdbDenseVolume(wvdb, { emission: "flames", scales: { flames: BLOB ? 1.0 : 6.0 }, mipmaps: MIPS });
  const source = await makeSource(renderer, device, dense, true);

  const fr = new VDBFrameRenderer();
  Object.assign(fr.settings, { width: 96, height: 64, depth: 24, temporal: false, jitter: false, lighting: "volume", multipleScattering: true });
  result.settings = { ...fr.settings };

  const bbox = new THREE.Box3(new THREE.Vector3(...dense.bbox![0]), new THREE.Vector3(...dense.bbox![1]));
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());
  const scale = 100 / Math.max(size.x, size.y, size.z);
  const hook = SPECTRUM === "white" ? undefined : ({ spectrum: "planck" } as const);
  const volume = new VDBFrameVolume(source, {
    frameRenderer: fr,
    bbox,
    baseColor: "#555555",
    // His explosion: normalized extinction 100 (optically very thick: the first non-empty sample takes almost all; `?extinction=`)
    extinction: Number(params.get("extinction") ?? (BLOB ? 20 : 100)),
    temperatureRange: [0, 3000],
    // His explosion 'planck' 15; the blob is dimmer (smooth, mostly thin flames): × 1000 keeps its probes out of half-float subnormals
    emissiveIntensity: Number(params.get("emissive") ?? (BLOB ? 15000 : 15)),
    shadowSteps: 24,
    mipmaps: MIPS,
    ...(hook ? { emissionColor: hook } : {}),
  });
  // Z-up, 100 units, the box centred on the stage; the stage stands on the ground at y = −50 (his frame.js)
  volume.scale.setScalar(scale);
  volume.position.copy(center).multiplyScalar(-scale);
  const up = new THREE.Group();
  up.rotation.x = -Math.PI / 2;
  up.add(volume);
  const stage = new THREE.Group();
  const height = size.z * scale;
  const groundY = -50;
  stage.position.y = groundY + height / 2;
  stage.add(up);

  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  scene.add(stage);

  const GROUND_HALF = 400;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(GROUND_HALF * 2, GROUND_HALF * 2, 64, 64), new THREE.MeshLambertMaterial({ color: 0xffffff }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = groundY;
  scene.add(ground);

  const light = new VDBFrameEmissionLight({ intensity: 25, resolution: PROBES, cascades: Number(params.get("emissionCascades") ?? 1) });
  light.visible = EMIT;
  scene.add(light);

  const camera = new THREE.PerspectiveCamera(55, W / H, 1, 5000);
  camera.position.set(0, 160, 330);
  camera.lookAt(0, -50, 0);
  camera.updateProjectionMatrix();
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  const frame = async () => {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    return readLinear(renderer, rt);
  };

  await frame();
  const on = await frame();
  const onAgain = await frame();
  const display = toDisplay(on);
  result.inPageIdentical = display.every((v, i) => v === toDisplay(onAgain)[i]);
  result.width = W;
  result.height = H;
  result.rgba = toBase64(display);
  result.emitters = light.emitterCount;
  result.backend = { requested: BACKEND, core: !!fr.core, maps: 0, probes: !!fr.core?.outputs.emissionProbes };
  result.probeGrid = {
    resolution: light.probeResolution,
    count: light.cascadeCount,
    min: light.cascadeMin.slice(0, light.cascadeCount).map((v) => v.toArray()),
    size: light.cascadeSize.slice(0, light.cascadeCount).map((v) => v.toArray()),
    center: light.probeCenter.toArray(),
  };

  // K6.8b (the K6.10 finding-8 class): the probe stack replaced or released while lit receivers (a plain
  // MeshLambertMaterial: three gives it SHARED refreshes) were built with it, then restored; the frame after must equal the
  // first, and nothing may bind a stack the core retired (an uncaptured "Destroyed texture … used in a submit")
  if (MUTATE === "probes" || MUTATE === "light" || MUTATE === "emitter") {
    const set = (away: boolean) => {
      if (MUTATE === "probes") light.resolution = away ? Math.max(2, PROBES / 2) : PROBES;
      else if (MUTATE === "light") light.visible = away ? false : EMIT;
      else volume.visible = !away;
    };
    set(true);
    // Several renders (each a task: the readback awaits): the core retires the old stack, then destroys it
    let awayFrame = on;
    for (let i = 0; i < 4; i++) awayFrame = await frame();
    const mapsAway = fr.core?.outputs.emissionProbes ? 1 : 0;
    let awayChanged = 0;
    for (let i = 0; i < W * H; i++) if (Math.abs(awayFrame[i * 4]! - on[i * 4]!) > 1e-3) awayChanged++;
    set(false);
    for (let i = 0; i < 2; i++) await frame();
    const restored = toDisplay(await frame());
    let maxDiff = 0;
    for (let i = 0; i < restored.length; i++) maxDiff = Math.max(maxDiff, Math.abs(restored[i]! - display[i]!));
    result.mutate = { kind: MUTATE, identical: maxDiff === 0, maxDiff, awayShadowed: awayChanged, mapsAway };
  }

  const groundPoints: (V3 | null)[] = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) groundPoints.push(groundPoint(camera, x, y, groundY, GROUND_HALF - 1));

  // The light off: the ground is black (no other light), the probes are not read
  light.visible = false;
  await frame();
  const off = await frame();
  result.rgbaOff = toBase64(toDisplay(off));
  volume.visible = false;
  const offNoVolume = await frame();
  volume.visible = true;
  let groundPixels = 0;
  let nonBlack = 0;
  for (let i = 0; i < W * H; i++) {
    if (!groundPoints[i]) continue;
    groundPixels++;
    if (offNoVolume[i * 4]! !== 0 || offNoVolume[i * 4 + 1]! !== 0 || offNoVolume[i * 4 + 2]! !== 0) nonBlack++;
  }
  result.lightOff = { groundPixels, nonBlack };
  // K6.5: the volume's screen coverage (it emits: it differs from black), dilated by 2 px (the composite's soft edge)
  const coverage = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (off[i * 4] === offNoVolume[i * 4] && off[i * 4 + 1] === offNoVolume[i * 4 + 1] && off[i * 4 + 2] === offNoVolume[i * 4 + 2]) continue;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) if (x + dx >= 0 && x + dx < W && y + dy >= 0 && y + dy < H) coverage[(y + dy) * W + x + dx] = 1;
    }
  result.coverage = toBase64(coverage);
  light.visible = EMIT;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(display), W, H), 0, 0);
  root.append(canvas);
  log(`emission (${SOURCE_KIND}): ${light.emitterCount} emitter(s), probes ${JSON.stringify(result.probeGrid)}; light off ${JSON.stringify(result.lightOff)}; repeat identical ${result.inPageIdentical}`, result.inPageIdentical ? "ok" : "fail");
}

async function main(): Promise<void> {
  root.textContent = "";
  result.case = CASE;
  result.sourceKind = SOURCE_KIND;
  const warn = console.warn.bind(console);
  console.warn = (...a: unknown[]) => {
    result.warnings.push(a.map(String).join(" "));
    warn(...a);
  };
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice({ ...(SOURCE_KIND === "sparse" ? { requiredLimits: VDBGridPool.requiredLimits(adapter) } : {}) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const renderer = new WebGPURenderer({ device });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  renderer.shadowMap.enabled = true;

  if (CASE === "emission") await runEmission(renderer, device);
  else if (CASE === "stale") await runStale(renderer, device);
  else await runShadow(renderer, device);
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
