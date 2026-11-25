// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L191–246 (the settings; now the core's), L431–481 (render: the
//   first proxy of a render draws the frame, override passes draw nothing), L1476–1716 (visitLight: the scene's lights
//   in the camera's layers).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes (K6.8): the TSL passes are gone; every frame is the core frame renderer (`vdbCreateFrameRenderer`, compute on
// three's device), recorded into its own encoder and submitted before three's frame. Divergences: DIVERGES.md (K3.1,
// K3.3, K6 rows).

/**
 * The froxel renderer of `./three` (K6.8, design §6): a thin adapter over the core frame renderer. One
 * `VDBFrameRendererGpu` per three renderer device; in the first `VDBFrameVolume` proxy's `onBeforeRender` of a render (or
 * a volume shadow receiver's / the emission light's `updateBefore`, whichever comes first) it collects the volumes in the
 * camera's layers, converts the scene's lights (shadow cameras → volume shadow requests, three's shadow maps →
 * occluders), passes `scene.environment`, records `core.update()` into its own encoder and submits; three's frame
 * submits after it. Every proxy then draws the composite material over three's scene depth and colour.
 */

import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import { vdbFrameMlpFromNeural, type VDBFrameMlp } from "../../tools/frame/mlp.js";
import { vdbFrameNeuralWeights } from "../../tools/frame/neural-weights.js";
import { vdbFrameDefaultSettings, type VDBFrameSettings } from "../../core/frame/settings.js";
import { vdbCreateFrameRenderer, type VDBFrameRendererGpu } from "../../core/frame/renderer.js";
import type { VDBEmissionColor, VDBFrameEmissionLightSettings, VDBFrameLight, VDBFrameRedrawStats, VDBFrameTimestamps, VDBFrameUpdate, VDBFrameVolumeGpu, VDBFrameVolumeParams, VDBVolumeShadowRequest } from "../../core/frame/types.js";
import { vdbSparseSourceGpu, type VDBSparseSourceGpu, type VDBSparseVolumeGpu } from "../../core/source/sparse.js";
import { vdbDenseSource, type VDBDenseSource } from "../../core/source/dense.js";
import type { VDBGridPool } from "../../core/pool/pool.js";
import type { VDBGridTypeName } from "../../core/types.js";
import type { VDBThreeDenseSource } from "../volume-source.js";
import type { VDBIndexBox, VDBThreeSparseSource } from "../sparse-source.js";
import { getDevice } from "../device.js";
import { vdbFrameRenderId } from "./three-internals.js";
import { VDBCoreTextureBinding, vdbCoreTextureSelfTest } from "./core-textures.js";
import { VDBFrameCoreIntegrateMaterial } from "./core-integrate-material.js";
import { VDBCoreOccluders } from "./core-occluders.js";
import { VDBCoreEnvironment } from "./core-environment.js";
import { VDBFrameVolumeShadowReceivers } from "./volume-shadows.js";
import { getFrameAngstromExponent } from "./spectrum.js";
import type { VDBFrameVolume } from "./volume.js";

/**
 * Hooks around each core update (K6.8; not his). `timestampWrites` brackets the kernels the update records (a
 * `'timestamp'` query set and two indices: written at the start and the end of the update's GPU work); resolve and
 * read the set yourself. `begin` / `end` run on the CPU around the update's recording and submit.
 */
export interface VDBFramePassHook {
  timestampWrites?: VDBFrameTimestamps | null;
  begin?(): void;
  end?(update: VDBFrameUpdate): void;
}

/** His trained network as the core MLP (the default `mlp`; one object, so the core uploads it once). */
const DEFAULT_MLP = vdbFrameMlpFromNeural(vdbFrameNeuralWeights);

type RendererLike = WebGPURenderer & { isWebGPURenderer?: boolean };
type SceneLike = THREE.Object3D & { overrideMaterial?: THREE.Material | null; environment?: THREE.Texture | null; environmentIntensity?: number; environmentRotation?: THREE.Euler };

/** Instanced, so a proxy skips its draw with `instanceCount = 0` (three draws no instances). A quad the composite places. */
const proxyQuad = new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0], 3);

/** His `FrameRenderer`, on the core: one instance can serve many volumes and cameras. */
export class VDBFrameRenderer {
  /**
   * The settings (the core's `VDBFrameSettings`: his names and v0.26 defaults plus the K6 fields); live, read at the
   * next render and shared by every volume this renderer draws.
   */
  settings: VDBFrameSettings = vdbFrameDefaultSettings();

  /**
   * EXPERIMENTAL: the network of `multipleScattering` `'mlp'` / `'neural'` as `VDBFrameMlp` data (layer shapes,
   * activations, weights; default: mjurczyk's trained weights). Assign another network of that form (no rebuild), or
   * null to run those modes as 1D diffusion (warned).
   */
  mlp: VDBFrameMlp | null = DEFAULT_MLP;

  /** Timing and debug hooks around each core update ({@link VDBFramePassHook}). */
  passHook: VDBFramePassHook | null = null;

  /**
   * Whether the volumes are drawn into the froxels at the next render (default true). False keeps the last froxels
   * (the composite still draws them; volume shadow maps and probes still update): `update(…, { froxels: false })`.
   */
  froxels = true;

  #core: VDBFrameRendererGpu | null = null;
  #device: GPUDevice | null = null;
  #material: VDBFrameCoreIntegrateMaterial | null = null;
  readonly #prefix = new VDBCoreTextureBinding();
  /** K6.4: the stand-in of the spectral alpha prefix (`outputs.froxels.integratedAlpha`). */
  readonly #alphaPrefix = new VDBCoreTextureBinding();
  #denseSources = new WeakMap<object, VDBDenseSource>();
  #sparseSources = new Map<object, Map<string, VDBSparseSourceGpu>>();
  #sparseSlots = new WeakMap<object, VDBSparseVolumeGpu>();
  /** The three source's stats object each core slot last took (a new object: `refreshStats()` ran). */
  #slotStats = new WeakMap<object, object>();
  #volumes = new Map<VDBFrameVolume, VDBFrameVolumeGpu>();
  #frameId = 0;
  #frameKey: { renderId: number; camera: THREE.Camera; scene: object; result: VDBFrameUpdate } | null = null;
  #cachedFrame = -1;
  #cachedCamera: THREE.Camera | null = null;
  #rendererWarned = false;
  /** K6.5b: three's shadow maps as the core's occluder maps. */
  readonly #occluders = new VDBCoreOccluders();
  /** K6.2b: `scene.environment` as the core's environment input. */
  readonly #environment = new VDBCoreEnvironment();
  /** @internal K6.5: volume shadows on meshes: the shadow filter on each shadow light, reading the core's maps. */
  readonly volumeShadows: VDBFrameVolumeShadowReceivers = new VDBFrameVolumeShadowReceivers(this);

  /** The core frame renderer (made at the first render, per device; null before). Read its `outputs`; set nothing on it. */
  get core(): VDBFrameRendererGpu | null {
    return this.#core;
  }

  /** Redraws and skips of the core's signature-cached passes ('frame' lighting) and volume shadow pieces (zeros before the first render). */
  get redrawStats(): VDBFrameRedrawStats {
    return this.#core?.redrawStats ?? { shadowDensity: { drawn: 0, skipped: 0 }, transmittance: { drawn: 0, skipped: 0 }, environmentGrid: { drawn: 0, skipped: 0 }, volumeShadows: { drawn: 0, waiting: 0 } };
  }

  /** The composite material every volume proxy draws (one per renderer). */
  proxyMaterial(): THREE.Material {
    this.#material ??= new VDBFrameCoreIntegrateMaterial();
    return this.#material;
  }

  /** A quad geometry for a volume proxy; the composite places it over the froxel area. */
  createProxyGeometry(): THREE.InstancedBufferGeometry {
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.instanceCount = 1;
    geometry.setAttribute("position", proxyQuad);
    return geometry;
  }

  /**
   * Called by each volume proxy before it draws: the first proxy of a render runs the core update (once per render,
   * camera and scene); every proxy then composites the froxels. Override passes (depth, normals) draw nothing.
   */
  render(renderer: RendererLike, scene: SceneLike, camera: THREE.Camera, proxy: THREE.Mesh<THREE.InstancedBufferGeometry>): void {
    if (!renderer.isWebGPURenderer) {
      if (!this.#rendererWarned) {
        console.error("VDBFrameVolume: needs WebGPURenderer (three/webgpu).");
        this.#rendererWarned = true;
      }
      return;
    }
    if (scene.overrideMaterial) {
      proxy.geometry.instanceCount = 0;
      return;
    }
    const frame = renderer.info.render.calls;
    if (frame === this.#cachedFrame && camera === this.#cachedCamera) {
      proxy.geometry.instanceCount = 0;
      return;
    }
    const material = this.proxyMaterial() as VDBFrameCoreIntegrateMaterial;
    // NOTE three draws the material it chose before onBeforeRender: a swapped proxy draws the composite from the next frame
    if (proxy.material !== material) proxy.material = material;
    const u = this.coreFrame(renderer, scene, camera);
    if (u?.recorded && u.outputs.froxels) {
      if (this.#prefix.set(u.outputs.froxels.integrated)) material.prefix.value = this.#prefix.texture;
      // K6.4: spectral media composite with the alpha prefix (their own blend over three's viewport texture)
      const alpha = u.outputs.froxels.integratedAlpha;
      if (alpha && this.#alphaPrefix.set(alpha)) material.alphaPrefix.value = this.#alphaPrefix.texture;
      material.setSpectral(!!alpha);
      material.setLayout(u.outputs.froxels.layout, camera as THREE.PerspectiveCamera);
      material.setSrgbBlending(this.settings.srgbBlending && renderer.outputColorSpace === THREE.SRGBColorSpace, renderer.toneMapping === THREE.NoToneMapping);
    }
    this.#cachedFrame = renderer.info.render.calls;
    this.#cachedCamera = camera;
    proxy.geometry.instanceCount = u?.recorded ? 1 : 0;
  }

  /**
   * The core update of one render, recorded into its own encoder and submitted (three's frame submits after it, so queue
   * order is right). It runs once per render id, camera and scene; the first caller runs it: a volume shadow receiver's
   * or the emission light's `updateBefore` (before any receiver of the render draws, his order), else the first volume
   * proxy. The volume shadow maps and the emission probes are drawn here.
   */
  coreFrame(renderer: RendererLike, scene: SceneLike, camera: THREE.Camera): VDBFrameUpdate | null {
    if (!renderer.isWebGPURenderer) return null;
    const renderId = vdbFrameRenderId(renderer);
    const key = this.#frameKey;
    if (key && key.renderId === renderId && key.camera === camera && key.scene === scene) return key.result;
    const device = getDevice(renderer as never);
    if (!this.#core || this.#device !== device) {
      vdbCoreTextureSelfTest(renderer as never, device);
      this.#core?.dispose();
      this.#core = vdbCreateFrameRenderer(device, { label: "vdb three frame" });
      this.#device = device;
      this.#denseSources = new WeakMap();
      this.#volumes = new Map();
      this.#disposeSparse();
    }
    const core = this.#core;
    Object.assign(core.settings, this.settings);
    core.mlp = this.mlp;
    // K6.5: the volume shadow filter on each shadow light (receivers build with it from the next render)
    this.volumeShadows.update(renderer, scene);

    const seen = new Set<VDBFrameVolumeGpu>();
    scene.traverseVisible((object) => {
      const frameVolume = object as VDBFrameVolume;
      if (!frameVolume.isVDBFrameVolume || frameVolume.frameRenderer !== this || !frameVolume.proxy.layers.test(camera.layers)) return;
      frameVolume.syncSource();
      const handle = this.#volumeOf(frameVolume, device);
      frameVolume.boxMatrixWorld.multiplyMatrices(frameVolume.matrixWorld, frameVolume.boxMatrix);
      this.#setParams(handle.params, frameVolume, frameVolume.boxMatrixWorld);
      handle.visible = true;
      // his collector: every volume in the camera's layers casts into the shadow grid, also out of view
      handle.castsShadow = true;
      seen.add(handle);
    });
    for (const [frameVolume, handle] of this.#volumes) {
      if (seen.has(handle)) continue;
      // a volume that is hidden (or out of the camera's layers) neither draws nor casts (Traps: the core casts invisible volumes)
      handle.visible = false;
      handle.castsShadow = false;
      if (!frameVolume.parent) {
        handle.remove();
        this.#volumes.delete(frameVolume);
      }
    }

    const cam = camera as THREE.PerspectiveCamera & { isOrthographicCamera?: boolean };
    const buffer = renderer.getDrawingBufferSize(new THREE.Vector2());
    const canvas = renderer.getSize(new THREE.Vector2());
    const { lights, emissionLight, shadowLights } = this.#lights(scene, camera);
    // K6.5b: three draws its pending shadow maps now (before this update's submit), so the occluders are this frame's
    const occluders = this.settings.occlusion ? this.#occluders.input(renderer as never, scene, camera, shadowLights) : [];
    const hook = this.passHook;
    hook?.begin?.();
    const encoder = device.createCommandEncoder({ label: "vdb three frame" });
    const u = core.update(
      encoder,
      {
        frameId: this.#frameId++,
        camera: {
          view: camera.matrixWorldInverse.elements,
          projection: camera.projectionMatrix.elements,
          near: cam.near,
          far: cam.far,
          orthographic: !!cam.isOrthographicCamera,
          reversedDepth: !!(renderer as unknown as { reversedDepthBuffer?: boolean }).reversedDepthBuffer,
          id: camera.id,
        },
        viewport: { drawingBuffer: [buffer.x, buffer.y], canvas: [canvas.x, canvas.y] },
        lights,
        emissionLight,
        environment: this.#environment.input(renderer as never, scene),
        occluders,
      },
      { froxels: this.froxels, ...(hook?.timestampWrites ? { timestampWrites: hook.timestampWrites } : {}) },
    );
    device.queue.submit([encoder.finish()]);
    // K6.5: the receivers read the maps with the state they were drawn with (copied to their uniforms now, before they draw)
    this.volumeShadows.sync(u.outputs.volumeShadows, core.redrawStats.volumeShadows);
    hook?.end?.(u);
    this.#frameKey = { renderId, camera, scene, result: u };
    return u;
  }

  /** The core volume of a frame volume: its dense source uploaded once on the core side, or a slot of a sparse source. */
  #volumeOf(frameVolume: VDBFrameVolume, device: GPUDevice): VDBFrameVolumeGpu {
    if (frameVolume.source.kind !== "dense") return this.#sparseVolumeOf(frameVolume);
    const three = frameVolume.source as VDBThreeDenseSource;
    let source = this.#denseSources.get(three);
    if (!source) {
      source = vdbDenseSource(device, three.volume, { label: "vdb three dense" });
      this.#denseSources.set(three, source);
    } else if (source.volume !== three.volume) {
      source.replace(three.volume);
    }
    let handle = this.#volumes.get(frameVolume);
    if (handle && handle.slot.source !== source) {
      handle.remove();
      handle = undefined;
    }
    if (!handle) {
      handle = this.#core!.addVolume(source);
      this.#volumes.set(frameVolume, handle);
    }
    return handle;
  }

  /**
   * K6.1d: the core volume of a frame volume on a sparse source: one `vdbSparseSourceGpu` per pool and source options
   * (its pyramids are the core's own), one slot per three source with the same grids, boxes and stats.
   */
  #sparseVolumeOf(frameVolume: VDBFrameVolume): VDBFrameVolumeGpu {
    const three = frameVolume.source as VDBThreeSparseSource & {
      auxBoxes?: Record<"mask" | "albedo" | "emission", VDBIndexBox | null>;
      pyramidMaxOption?: { pyramidMax?: number };
    };
    const pool = three.grids.pool as unknown as VDBGridPool;
    const types = (three.grids.options as { types?: readonly VDBGridTypeName[] }).types;
    const options = {
      lod: three.lod,
      normalize: three.normalize,
      followImage: three.followImage,
      fastTrilinear: three.fastTrilinear,
      cacheAccessor: three.cacheAccessor,
      ...(three.pyramidMaxOption?.pyramidMax !== undefined ? { pyramidMax: three.pyramidMaxOption.pyramidMax } : {}),
      ...(types ? { types } : {}),
    };
    const key = JSON.stringify(options);
    let byKey = this.#sparseSources.get(pool);
    if (!byKey) this.#sparseSources.set(pool, (byKey = new Map()));
    let source = byKey.get(key);
    if (!source) {
      source = vdbSparseSourceGpu(pool, { ...options, label: "vdb three sparse" });
      byKey.set(key, source);
    }
    let handle = this.#volumes.get(frameVolume);
    let slot = this.#sparseSlots.get(three);
    if (!slot || slot.source !== source || slot.removed) {
      slot?.remove();
      const aux = three.auxBoxes ?? { mask: null, albedo: null, emission: null };
      const auxIndexBoxes = Object.fromEntries(Object.entries(aux).filter(([, b]) => b)) as Partial<Record<"mask" | "albedo" | "emission", VDBIndexBox>>;
      slot = source.volume(three.ids, three.followImage ? { stats: three.stats } : { indexBox: three.indexBox, auxIndexBoxes, stats: three.stats });
      this.#sparseSlots.set(three, slot);
      this.#slotStats.set(slot, three.stats);
    } else if (this.#slotStats.get(slot) !== three.stats) {
      // the three source re-read its stats (`refreshStats()` after a replace): the core slot takes them now, instead of
      // a frame or two later from its own readback of the same image (a coarse-first swap's first full frame, G16a)
      (slot as { stats: VDBSparseVolumeGpu["stats"] }).stats = { ...three.stats };
      this.#slotStats.set(slot, three.stats);
    }
    if (handle && handle.slot !== slot) {
      handle.remove();
      handle = undefined;
    }
    if (!handle) {
      handle = this.#core!.addVolume(slot);
      this.#volumes.set(frameVolume, handle);
    }
    return handle;
  }

  /** Frees the sparse sources (their slots and pyramids); the pools are the host's. */
  #disposeSparse(): void {
    for (const byKey of this.#sparseSources.values()) for (const s of byKey.values()) s.dispose();
    this.#sparseSources = new Map();
    this.#sparseSlots = new WeakMap();
  }

  /** A frame volume's props as the core's params. */
  #setParams(p: VDBFrameVolumeParams, v: VDBFrameVolume, matrixWorld: THREE.Matrix4): void {
    const wrap = v.wrap3D === THREE.RepeatWrapping ? "repeat" : v.wrap3D === THREE.MirroredRepeatWrapping ? "mirror" : "clamp";
    const params: VDBFrameVolumeParams = {
      matrixWorld: matrixWorld.elements.slice(),
      baseColor: [v.baseColor.r, v.baseColor.g, v.baseColor.b],
      scatterColor: [v.scatterColor.r, v.scatterColor.g, v.scatterColor.b],
      extinction: v.extinction,
      extinctionScale: [v.extinctionScale.x, v.extinctionScale.y, v.extinctionScale.z],
      // K6.4: a preset name ('smoke', 'air', …) or a number (his getFrameAngstromExponent)
      angstromExponent: getFrameAngstromExponent(v.angstromExponent),
      densityScale: v.densityScale,
      normalizeDensity: v.normalizeDensity,
      densityMean: v.densityMean,
      densityCutoff: v.densityCutoff,
      absorbance: v.absorbance,
      opacity: v.opacity,
      emissiveIntensity: v.emissiveIntensity,
      temperatureRange: [v.temperatureRange.x, v.temperatureRange.y],
      offset3D: [v.offset3D.x, v.offset3D.y, v.offset3D.z],
      wrap3D: wrap,
      lights: v.lights,
      shadowSteps: v.shadowSteps,
      mipmaps: v.mipmaps,
      hdda: v.hdda,
      phase: v.phase,
      phaseG: v.phaseG,
      phaseBackG: v.phaseBackG,
      phaseBackWeight: v.phaseBackWeight,
      phaseGExponent: v.phaseGExponent,
      emissionColor: this.#emissionColor(v),
    };
    Object.assign(p, params);
  }

  /** K6.5: the volume's emission colour as core data ('white' when unset). */
  #emissionColor(v: VDBFrameVolume): VDBEmissionColor {
    return v.emissionColor ?? "white";
  }

  /**
   * The scene's lights in the camera's layers as core lights (his visitLight). K6.5: a shadow light with the volume
   * shadow filter on it asks for a volume shadow map (spot: the light's own shadow camera, so both maps share one texel
   * grid; point: its cube near / far; directional: cascades fitted by the core); a `VDBFrameEmissionLight` gives the
   * emission probe settings.
   */
  #lights(scene: THREE.Object3D, camera: THREE.Camera): { lights: VDBFrameLight[]; emissionLight: VDBFrameEmissionLightSettings | null; shadowLights: THREE.Light[] } {
    const out: VDBFrameLight[] = [];
    // K6.5b: the directional, spot and point lights passed (their shadow maps are the occluders)
    const shadowLights: THREE.Light[] = [];
    let emissionLight: VDBFrameEmissionLightSettings | null = null;
    const position = new THREE.Vector3();
    const target = new THREE.Vector3();
    const size = this.volumeShadows.getMapSize();
    scene.traverseVisible((object) => {
      const light = object as THREE.Light & {
        isAmbientLight?: boolean;
        isHemisphereLight?: boolean;
        isLightProbe?: boolean;
        isDirectionalLight?: boolean;
        isSpotLight?: boolean;
        isPointLight?: boolean;
        isVDBFrameEmissionLight?: boolean;
        groundColor?: THREE.Color;
        sh?: { coefficients: THREE.Vector3[] };
        target?: THREE.Object3D;
        angle?: number;
        penumbra?: number;
        distance?: number;
        decay?: number;
      };
      if (!light.isLight || !light.layers.test(camera.layers)) return;
      if (light.isVDBFrameEmissionLight) {
        if (!emissionLight) {
          const e = light as unknown as Required<VDBFrameEmissionLightSettings>;
          emissionLight = { resolution: e.resolution, range: e.range, cascades: e.cascades, distance: e.distance, steps: Math.max(1, Math.round(e.steps)) };
        }
        return;
      }
      const i = light.intensity;
      const color: [number, number, number] = [light.color.r * i, light.color.g * i, light.color.b * i];
      position.setFromMatrixPosition(light.matrixWorld);
      const id = light.id;
      // K6.5: a map for each light the volume shadow filter is on (volumeShadows.update installed it)
      let shadow: VDBVolumeShadowRequest | undefined;
      if (this.settings.castShadows && light.castShadow && this.volumeShadows.entries.has(light)) {
        if (light.isSpotLight) {
          (light as unknown as THREE.SpotLight).shadow.updateMatrices(light as unknown as THREE.SpotLight);
          const c = (light as unknown as THREE.SpotLight).shadow.camera;
          shadow = { size, view: { viewProjection: new THREE.Matrix4().multiplyMatrices(c.projectionMatrix, c.matrixWorldInverse).elements, near: c.near, far: c.far, orthographic: false } };
        } else if (light.isPointLight) {
          const c = (light as unknown as THREE.SpotLight).shadow.camera;
          shadow = { size, view: { near: c.near, far: light.distance || c.far } };
        } else if (light.isDirectionalLight) shadow = { size, view: "fit" };
      }
      if ((light.isDirectionalLight || light.isSpotLight || light.isPointLight) && light.castShadow) shadowLights.push(light);
      if (light.isAmbientLight) out.push({ id, kind: "ambient", color });
      else if (light.isHemisphereLight) {
        const g = light.groundColor!;
        const up = position.lengthSq() > 0 ? position.clone().normalize() : new THREE.Vector3(0, 1, 0);
        out.push({ id, kind: "hemisphere", sky: color, ground: [g.r * i, g.g * i, g.b * i], up: [up.x, up.y, up.z] });
      } else if (light.isLightProbe) {
        out.push({ id, kind: "probe", sh: light.sh!.coefficients.flatMap((c) => [c.x * i, c.y * i, c.z * i]) });
      } else if (light.isDirectionalLight) {
        target.setFromMatrixPosition(light.target!.matrixWorld).sub(position).normalize();
        out.push({ id, kind: "directional", direction: [target.x, target.y, target.z], color, ...(shadow ? { shadow } : {}) });
      } else if (light.isSpotLight) {
        target.setFromMatrixPosition(light.target!.matrixWorld).sub(position).normalize();
        out.push({
          id,
          kind: "spot",
          position: [position.x, position.y, position.z],
          direction: [target.x, target.y, target.z],
          color,
          distance: light.distance!,
          decay: light.decay!,
          cosAngle: Math.cos(light.angle!),
          cosPenumbra: Math.cos(light.angle! * (1.0 - light.penumbra!)),
          ...(shadow ? { shadow } : {}),
        });
      } else if (light.isPointLight) {
        out.push({ id, kind: "point", position: [position.x, position.y, position.z], color, distance: light.distance!, decay: light.decay!, ...(shadow ? { shadow } : {}) });
      }
    });
    return { lights: out, emissionLight, shadowLights };
  }

  /** Frees the core renderer, its stand-ins and the shadow filters. A later render makes them again. */
  dispose(): void {
    // NOTE Puts back the shadow filter of each light; the receivers build again without the volume shadow
    this.volumeShadows.dispose();
    this.#core?.dispose();
    this.#core = null;
    this.#device = null;
    this.#environment.dispose();
    this.#prefix.dispose();
    this.#alphaPrefix.dispose();
    this.#material?.dispose();
    this.#material = null;
    this.#volumes = new Map();
    this.#disposeSparse();
    this.#frameKey = null;
    this.#cachedFrame = -1;
    this.#cachedCamera = null;
  }
}

/** Per three renderer shared instances (`vdbFrameRenderer(renderer)`). */
const sharedByRenderer = new WeakMap<object, VDBFrameRenderer>();
let shared: VDBFrameRenderer | null = null;

/**
 * The shared {@link VDBFrameRenderer} that a `VDBFrameVolume` uses when you pass no `frameRenderer`.
 * Its `settings` are global to every volume that uses it (the froxel grid, shadows and scattering are
 * built once for all of them). Pass a three renderer to get the instance shared by that renderer only
 * (a WeakMap, so two three renderers never share targets or settings); with no argument you get the
 * one process-wide default. For volumes that need different settings, make your own
 * `new VDBFrameRenderer()` and pass it as `VDBFrameVolume({ frameRenderer })`.
 */
export function vdbFrameRenderer(renderer?: object): VDBFrameRenderer {
  if (renderer) {
    let r = sharedByRenderer.get(renderer);
    if (!r) sharedByRenderer.set(renderer, (r = new VDBFrameRenderer()));
    return r;
  }
  shared ??= new VDBFrameRenderer();
  return shared;
}
