// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L715–1061 (updateTargets, renderFrame, updateFroxelArea,
//   renderTemporal, renderInject: the per-frame order, jitter, temporal validity, previous matrices),
//   L1476–1716 (light records and their slot order: directional, point, spot), frameShaderChunks.js L800–830
//   (setVolumeUniforms).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: an engine-free renderer on the host's device and encoder (D8, D34): plain-data inputs, compute
// kernels on 3D textures, explicit bind groups, a retire queue (design §2–§4). Divergences: DIVERGES.md
// (K6.1 rows).

/**
 * `vdbCreateFrameRenderer(device, options)`: the core froxel renderer ('volume' lighting, K6.1; 'frame' lighting's
 * shadow grid, transmittance and fused gather, K6.2; dense and sparse sources). Per frame:
 *
 *   const enc = device.createCommandEncoder();
 *   const u = frame.update(enc, { frameId, camera, viewport, lights });   // gather per source group, resolve
 *   frame.render(enc, { color, size, depth, sceneColor });               // or vdbFrameIntegrate in your pass
 *   device.queue.submit([enc.finish()]);
 *
 * The renderer records, never submits; it frees only what it made (D8). One `update()` per submitted
 * encoder (its uniforms go through `queue.writeBuffer`), submitted in the task that recorded it: a texture the renderer
 * replaces is destroyed two updates later and never in the task that made it due (K6.10a), so a submit later in that
 * task is safe.
 */

import { VDBError } from "../errors.js";
import type { VDBVolumeSlot, VDBVolumeSource } from "../source/types.js";
import {
  vdbFrameAtlasLayout,
  vdbFrameClipCamera,
  vdbFrameClipRect,
  vdbFrameClipVolume,
  vdbFrameFroxelArea,
  vdbFrameFroxelRect,
  vdbFrameFroxelSize,
  vdbFrameVolumeSlices,
  vdbFroxelAreaUnion,
  vdbFroxelHeldSize,
  vdbMat4Invert,
  vdbMat4MaxScale,
  vdbMat4Multiply,
  vdbUnitBoxBounds,
  type VDBFroxelArea,
  type VDBFroxelRect,
} from "../../tools/frame/view.js";
import { vdbFrameAdaptiveCellSize, vdbFrameAdaptiveGridResolution, vdbFrameGridCascades, vdbFrameGridSignatureState, vdbFrameUpdateGridSignatures, vdbSameValues, type VDBBox3Like } from "../../tools/frame/grid.js";
import { vdbFrameStackLayout, type VDBFrameStackLayout } from "../../tools/frame/stack.js";
import { GATHER_BINDINGS, VDB_FRAME_GATHER_WORDS } from "./wgsl/gather.js";
import { FRAME_GATHER_BINDINGS, GRID_DENSITY_BINDINGS, TRANSMITTANCE_BINDINGS, VDB_FRAME_SPECTRAL_GRID_WORDS, VDB_FRAME_SUM_WORDS } from "./wgsl/lighting.js";
import { VDB_FRAME_COMPOSITE_ALPHA_BINDING } from "./wgsl/integrate.js";
import { vdbFrameExtinctionSpectrum, vdbFrameSpectralGroups } from "../../tools/frame/spectral.js";
import { RESOLVE_BINDINGS } from "./wgsl/resolve.js";
import { VDB_FRAME_OCCLUDER_BINDING } from "./wgsl/occluders.js";
import { ENV_GRID_BINDINGS } from "./wgsl/environment.js";
import { VDBFrameEnvironmentState } from "./environment.js";
import {
  vdbFrameCompositeSpec,
  vdbFrameEmissionDirectionSpec,
  vdbFrameEmissionProbeSpec,
  vdbFrameEmissionReduceSpec,
  vdbFrameEnvironmentGridSpec,
  vdbFrameGatherSpec,
  vdbFrameGridDensitySpec,
  vdbFrameGridMipSpec,
  vdbFrameResolveSpec,
  vdbFrameTransmittanceSpec,
  vdbFrameVolumeShadowSpec,
  SOURCE_SLOT,
  type VDBFrameCompositeKey,
} from "./kernels.js";
import { VDBFrameShadowMaps } from "./shadows.js";
import { VDBFrameEmissionColors, VDBFrameEmissionProbeSet, type VDBFrameEmissionPlan, type VDBFrameEmitterInput } from "./emission.js";
import { EMISSION_DIRECTION_BINDINGS, EMISSION_PROBE_BINDINGS, EMISSION_REDUCE_BINDINGS, VDB_FRAME_EMISSION_DIRECTIONS, VDB_FRAME_EMITTERS, VDB_FRAME_EMITTERS_STRIDE } from "./wgsl/emission.js";
import { computePipeline, prepareCompute, prepareRender, renderPipeline } from "./pipelines.js";
import { BUFFER_USAGE, createFroxelTexture, VDBRetireQueue } from "./resources.js";
import { VDB_FRAME_DEFAULT_LIMITS, vdbFrameFitGridResolution, vdbFrameStackTargets } from "./limits.js";
import { VDBFrameScattering } from "./scattering.js";
import { vdbFrameScatteringIsMlp, type VDBFrameScatteringKernel } from "./wgsl/lighting.js";
import { vdbFrameMieLutWords } from "../../tools/frame/mie.js";
import { vdbFrameScatteringOctaves } from "../../tools/frame/index.js";
import { VDB_FRAME_AMBIENT_MAX_PAIRS, vdbFrameAmbientDirections } from "../../tools/frame/ambient.js";
import { vdbFrameMlpFromNeural, type VDBFrameMlp } from "../../tools/frame/mlp.js";
import { vdbFrameNeuralWeights } from "../../tools/frame/neural-weights.js";
import { vdbFrameDefaultInternalSettings, type VDBFrameInternalSettings, type VDBFrameSettings } from "./settings.js";
import { vdbFrameMultipleScatteringValue, vdbFramePhaseSpectrum, vdbFramePhaseVector, type VDBFramePhaseFields } from "./phase.js";
import {
  VDB_FRAME_GROUP,
  VDB_FRAME_GROUP_STRIDE,
  VDB_FRAME_LIGHTS,
  VDB_FRAME_MAX_CASCADES,
  VDB_FRAME_MAX_HEMISPHERE_LIGHTS,
  VDB_FRAME_MAX_LIGHTS,
  VDB_FRAME_MAX_OCCLUDERS,
  VDB_FRAME_MAX_VOLUMES,
  VDB_FRAME_OCCLUDER_LAYERS,
  VDB_FRAME_SHADOW_SLOTS,
  VDB_FRAME_SPECTRAL_GROUPS,
  VDB_FRAME_UNIFORMS,
  VDB_FRAME_VOLUME,
  VDB_FRAME_VOLUME_BYTES,
  VDBUniformWriter,
} from "./uniforms.js";
import {
  vdbFrameDefaultVolumeParams,
  type VDBFrameInput,
  type VDBFrameLight,
  type VDBFrameOccluder,
  type VDBFrameOutputs,
  type VDBFrameRedrawStats,
  type VDBFrameTargets,
  type VDBFrameUpdate,
  type VDBFrameVolume,
  type VDBFrameVolumeParams,
  type VDBFrameUpdateOptions,
  type VDBFroxelLayout,
  type VDBFrameEmissionProbes,
  type VDBVolumeShadowMap,
} from "./types.js";

/** Options of {@link vdbCreateFrameRenderer}. */
export interface VDBFrameRendererOptions {
  /** Initial settings (his names and v0.26 defaults, plus the K6 fields). */
  settings?: Partial<VDBFrameSettings>;
  /** The core composite pass (`render()`): the host's colour and depth formats; null: no `render()`. */
  composite?: { colorFormat: GPUTextureFormat; depthFormat: GPUTextureFormat } | null;
  /** Optional device features (K6.1 uses none; fast paths are deferred, design §5.4). */
  features?: { tier1?: boolean; f16?: boolean; subgroups?: boolean };
  /** Label prefix of the GPU objects. */
  label?: string;
}

/** The core frame renderer (design §2.2). */
export interface VDBFrameRenderer {
  readonly device: GPUDevice;
  /** Mutable; read at the next `update()`. */
  readonly settings: VDBFrameSettings;
  readonly outputs: VDBFrameOutputs;
  readonly redrawStats: VDBFrameRedrawStats;
  /**
   * K6.Q1b: the side of the 'frame' shadow grid the last update used (`shadowResolution`, or the size-adaptive side
   * with `shadowResolutionMax`, after the device clamp); 0 before the first 'frame' update.
   */
  readonly shadowGridResolution: number;
  /**
   * K6.3 EXPERIMENTAL: the network of `multipleScattering: 'mlp'` / `'neural'` ('frame' lighting, grey): data for a
   * generic small-MLP evaluator (shapes as uniforms, weights in a storage buffer; a new network needs no rebuild).
   * Default: mjurczyk's trained weights. null: those modes run as 1D diffusion.
   */
  mlp: VDBFrameMlp | null;
  addVolume(slot: VDBVolumeSlot, params?: Partial<VDBFrameVolumeParams>): VDBFrameVolume;
  /** Builds every pipeline the current volumes and settings reach, with the async creators. */
  prepare(options?: { lighting?: ("volume" | "frame")[] }): Promise<void>;
  /**
   * Records the frame's kernels into the host's encoder (one update per submitted encoder). `froxels: false` keeps the
   * last froxels (no froxel work; the composite reads them again). `timestampWrites` brackets every kernel the update
   * records: `beginningOfPassWriteIndex` is written before the first and `endOfPassWriteIndex` after the last (two empty
   * compute passes carry them, so the bracket holds whatever the update records).
   */
  update(encoder: GPUCommandEncoder, frame: VDBFrameInput, options?: VDBFrameUpdateOptions): VDBFrameUpdate;
  render(encoder: GPUCommandEncoder, targets: VDBFrameTargets): void;
  dispose(): void;
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
/** Shadow inputs of an update with nothing to draw (the maps are freed or disabled). */
const EMPTY_SHADOWS = {
  coarse: false,
  steps: 1,
  budget: 1,
  cascades: 1,
  castShadowDistance: 1,
  camera: { view: IDENTITY, cameraWorld: IDENTITY, projectionInverse: IDENTITY, near: 1, far: 2, orthographic: false },
  grid: null,
  castVersion: 0,
} as const;
const DEPTH_FORMATS = new Set<GPUTextureFormat>(["depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8"]);
const WRAP_CODE = { clamp: 0, repeat: 1, mirror: 2 } as const;

class FrameVolume implements VDBFrameVolume {
  params: VDBFrameVolumeParams;
  visible = true;
  castsShadow = true;
  readonly slot: VDBVolumeSlot;
  readonly #owner: FrameRenderer;

  constructor(owner: FrameRenderer, slot: VDBVolumeSlot, params: Partial<VDBFrameVolumeParams>) {
    this.#owner = owner;
    this.slot = slot;
    this.params = { ...vdbFrameDefaultVolumeParams(), ...params };
  }

  remove(): void {
    this.#owner.removeVolume(this);
  }
}

/** One volume after the CPU work of an update: visible (drawn into the froxels) and/or a shadow caster ('frame'). */
interface VolumeRecord {
  volume: FrameVolume;
  localToView: Float64Array;
  viewToLocal: Float64Array;
  worldToLocal: Float64Array;
  rect: VDBFroxelRect;
  first: number;
  count: number;
  visible: boolean;
  caster: boolean;
  /** K6.5: its emission reaches the emission light (drawn, with an emission channel). */
  emitter: boolean;
}

/** One source group of an update: its records (visible ones first, then casters) and where they sit. */
interface SourceGroup {
  source: VDBVolumeSource;
  records: VolumeRecord[];
  /** First record index; visible records [first, first + visible); casters [first + casterStart, + casters). */
  first: number;
  visible: number;
  casterStart: number;
  casters: number;
  /** K6.5: the validated emission colour hook of the group's volumes (a pipeline key), or null. */
  hook: string | null;
}

interface Textures {
  size: [number, number, number];
  scatter: GPUTexture;
  integrated: GPUTexture;
  history: [GPUTexture, GPUTexture] | null;
  partials: GPUTexture | null;
  /** Slices per froxel slice the partials hold (2 for 'volume', 8 for 'frame'; K6.4 spectral 'volume': 3). */
  partialLayers: number;
  /** K6.4 spectral: the second scatter texture (red α, blue α, 0, green α) and the alpha prefix (red, blue opacity). */
  spectral: { scatterAlpha: GPUTexture; integratedAlpha: GPUTexture; alphaFormat: "rg16float" | "rgba16float" } | null;
}

/** The 'frame' lighting grid textures (K6.2): the shadow grid and transmittance stacks (design §3.2). */
interface GridTextures {
  resolution: number;
  cascades: number;
  shadow: GPUTexture;
  shadowLayout: VDBFrameStackLayout;
  transmittance: GPUTexture;
  transmittanceLayout: VDBFrameStackLayout;
  partials: GPUTexture | null;
  /** K6.2b: the environment grid (the last cascade, R³ rgba16float). */
  environment: GPUTexture;
  /** K6.4 spectral: the absorption stack (rgb; the shadow stack holds the extinction rgb), null when grey. */
  absorption: GPUTexture | null;
  /** K6.4: the slot groups the transmittance stack holds (3 targets each per cascade); 0 when grey (4 targets). */
  groups: number;
}

class FrameRenderer implements VDBFrameRenderer {
  readonly device: GPUDevice;
  readonly settings: VDBFrameSettings;
  readonly redrawStats: VDBFrameRedrawStats = {
    shadowDensity: { drawn: 0, skipped: 0 },
    transmittance: { drawn: 0, skipped: 0 },
    environmentGrid: { drawn: 0, skipped: 0 },
    volumeShadows: { drawn: 0, waiting: 0 },
  };
  shadowGridResolution = 0;
  readonly outputs: {
    version: number;
    froxels: { integrated: GPUTexture; integratedAlpha: GPUTexture | null; layout: VDBFroxelLayout } | null;
    uniforms: GPUBuffer;
    volumeShadows: ReadonlyMap<number | string, VDBVolumeShadowMap>;
    emissionProbes: VDBFrameEmissionProbes | null;
  };

  readonly #label: string;
  readonly #composite: { colorFormat: GPUTextureFormat; depthFormat: GPUTextureFormat } | null;
  readonly #maxVolumes: number;
  readonly #retire = new VDBRetireQueue();
  readonly #volumes: FrameVolume[] = [];
  readonly #warned = new Set<string>();
  /** K6.10a: clamped shadow grid resolutions by (resolution, cascades, spectral); the limits do not change. */
  readonly #gridFits = new Map<string, number>();
  readonly #held = vdbFroxelHeldSize();
  readonly #frameWriter = new VDBUniformWriter(VDB_FRAME_UNIFORMS.size);
  readonly #lightsWriter = new VDBUniformWriter(VDB_FRAME_LIGHTS.size);
  readonly #volumesWriter: VDBUniformWriter;
  readonly #frameBuffer: GPUBuffer;
  readonly #lightsBuffer: GPUBuffer;
  readonly #volumesBuffer: GPUBuffer;
  #groupsBuffer: GPUBuffer;
  #groupsWriter: VDBUniformWriter;
  readonly #linear: GPUSampler;
  #textures: Textures | null = null;
  /** The rect each texture last held non-zero values in (the dispatch box also clears it). */
  readonly #written = new WeakMap<GPUTexture, VDBFroxelArea>();
  #historyIndex = 0;
  #historyValid = false;
  #previous: { view: Float64Array; projection: Float64Array; range: [number, number]; cameraId: unknown; frameId: number } | null = null;
  #group0: GPUBindGroup | null = null;
  readonly #sourceGroups = new WeakMap<VDBVolumeSource, { version: number; layout: GPUBindGroupLayout; group: GPUBindGroup }>();
  #recorded = false;
  #disposed = false;
  // K6.2 'frame' lighting: the grid textures, the cascade layout and his change signatures
  #grid: GridTextures | null = null;
  readonly #gridState = vdbFrameGridSignatureState();
  readonly #gridMin = Array.from({ length: VDB_FRAME_MAX_CASCADES }, () => ({ x: 0, y: 0, z: 0 }));
  readonly #gridSize = Array.from({ length: VDB_FRAME_MAX_CASCADES }, () => ({ x: 0, y: 0, z: 0 }));
  readonly #gridVoxel = Array.from({ length: VDB_FRAME_MAX_CASCADES }, () => ({ x: 0, y: 0, z: 0 }));
  /** The grid version the shadow grid texture holds (−1: none). */
  #gridDrawn = -1;
  /** K6.3: shadow grid draws so far (the scattered-light mips follow it). */
  #gridDraws = 0;
  #transmittanceSignature: unknown[] = [];
  // K6.2b: the GPU environment (buffer, maps, kernels) and the environment grid's signature
  readonly #env: VDBFrameEnvironmentState;
  #envGridSignature: unknown[] = [];
  readonly #sourceIds = new WeakMap<object, number>();
  #nextSourceId = 1;
  // K6.5: volume shadow maps, emission colours (LUT, hooks) and the emission probes
  readonly #shadows: VDBFrameShadowMaps;
  readonly #colors: VDBFrameEmissionColors;
  readonly #probes: VDBFrameEmissionProbeSet;
  // K6.3: the Lorenz-Mie LUT and the scattered-light term's resources ('mlp', 'multi-scale')
  readonly #mie: GPUBuffer;
  readonly #scattering: VDBFrameScattering;
  /**
   * K6.3 EXPERIMENTAL: the network of `multipleScattering: 'mlp'` (and `'neural'`) in 'frame' lighting, data for a
   * generic MLP evaluator; default our trained weights (K6.N, D36). null: those modes run as 1D diffusion (warned).
   */
  mlp: VDBFrameMlp | null = vdbFrameMlpFromNeural(vdbFrameNeuralWeights);
  /** K6.4: rg16float storage (`texture-formats-tier1`) for the spectral alpha prefix; else rgba16float. */
  readonly #tier1: boolean;
  // K6.5b: this update's host occluder map views (slots 0–3) and the 1 × 1 placeholder of the empty slots
  #occluderViews: GPUTextureView[] = [];
  #occluderPlaceholder: { texture: GPUTexture; view: GPUTextureView } | null = null;

  constructor(device: GPUDevice, o: VDBFrameRendererOptions) {
    this.device = device;
    this.#label = o.label ?? "vdb frame";
    this.settings = { ...vdbFrameDefaultInternalSettings(), ...(o.settings ?? {}) };
    this.#composite = o.composite ?? null;
    this.#tier1 = o.features?.tier1 ?? !!device.features?.has?.("texture-formats-tier1");
    this.#maxVolumes = Math.min(VDB_FRAME_MAX_VOLUMES, Math.floor((device.limits?.maxUniformBufferBindingSize ?? 65536) / VDB_FRAME_VOLUME_BYTES));
    this.#frameBuffer = device.createBuffer({ label: `${this.#label} uniforms`, size: VDB_FRAME_UNIFORMS.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#lightsBuffer = device.createBuffer({ label: `${this.#label} lights`, size: VDB_FRAME_LIGHTS.size, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#volumesWriter = new VDBUniformWriter(this.#maxVolumes * VDB_FRAME_VOLUME_BYTES);
    this.#volumesBuffer = device.createBuffer({ label: `${this.#label} volumes`, size: this.#maxVolumes * VDB_FRAME_VOLUME_BYTES, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#groupsWriter = new VDBUniformWriter(4 * VDB_FRAME_GROUP_STRIDE);
    this.#groupsBuffer = device.createBuffer({ label: `${this.#label} groups`, size: 4 * VDB_FRAME_GROUP_STRIDE, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    this.#linear = device.createSampler({
      label: `${this.#label} linear clamp`,
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "nearest",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge",
    });
    this.#env = new VDBFrameEnvironmentState(device, this.#label);
    this.#shadows = new VDBFrameShadowMaps(device, this.#label, this.#retire, (key, message) => this.#warnOnce(key, message));
    this.#colors = new VDBFrameEmissionColors(device, this.#label, this.#retire);
    this.#probes = new VDBFrameEmissionProbeSet(device, this.#label, this.#retire);
    // K6.3: the Lorenz-Mie phase tables as one LUT (V47: rgb 'lorenz-mie-rgb', a 'lorenz-mie'), group 0 binding 4
    const mie = vdbFrameMieLutWords();
    this.#mie = device.createBuffer({ label: `${this.#label} Lorenz-Mie table`, size: mie.byteLength, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
    device.queue.writeBuffer(this.#mie, 0, mie.buffer as ArrayBuffer);
    this.#scattering = new VDBFrameScattering(device, this.#label, this.#retire, (key, message) => this.#warnOnce(key, message));
    this.outputs ={ version: 0, froxels: null, uniforms: this.#frameBuffer, volumeShadows: this.#shadows.maps, emissionProbes: null };
  }

  // ---------------------------------------------------------------- volumes

  addVolume(slot: VDBVolumeSlot, params: Partial<VDBFrameVolumeParams> = {}): VDBFrameVolume {
    this.#check();
    if (!slot || !slot.source || typeof slot.source.wgsl !== "function" || !Number.isInteger(slot.v)) {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameRenderer.addVolume: pass a VDBVolumeSlot ({ source, v }); a dense source from vdbDenseSource() is its own slot.");
    }
    const v = new FrameVolume(this, slot, params);
    this.#volumes.push(v);
    return v;
  }

  removeVolume(v: FrameVolume): void {
    const i = this.#volumes.indexOf(v);
    if (i >= 0) this.#volumes.splice(i, 1);
  }

  // ---------------------------------------------------------------- pipelines

  #compositeKey(): VDBFrameCompositeKey | null {
    if (!this.#composite) return null;
    return {
      colorFormat: this.#composite.colorFormat,
      depth: DEPTH_FORMATS.has(this.#composite.depthFormat) ? "depth" : "r32float",
      srgb: this.settings.srgbBlending,
      spectral: !!this.settings.spectral,
    };
  }

  async prepare(options: { lighting?: ("volume" | "frame")[] } = {}): Promise<void> {
    this.#check();
    const modes = options.lighting ?? [this.settings.lighting];
    // K6.5: the emission hooks validate first (a hook pipeline is built only once its hook is valid)
    await this.#colors.settled(this.#volumes.map((v) => v.params.emissionColor));
    const jobs: Promise<void>[] = [];
    const spectral = !!this.settings.spectral;
    const sources = new Set(this.#volumes.map((v) => v.slot.source));
    const shadows = modes.includes("frame") || this.settings.castShadows;
    for (const source of sources) {
      const hooks = new Set<string | null>([null]);
      if (source.channels.emission) for (const v of this.#volumes) if (v.slot.source === source) hooks.add(this.#colors.codeOf(v.params.emissionColor).hook);
      for (const hook of hooks) {
        for (const multi of [false, true]) {
          for (const lighting of modes) jobs.push(prepareCompute(this.device, vdbFrameGatherSpec(source, { multi, maxVolumes: this.#maxVolumes, lighting, hook, spectral })));
        }
        if (source.channels.emission) jobs.push(prepareCompute(this.device, vdbFrameEmissionDirectionSpec(source, { maxVolumes: this.#maxVolumes, hook })));
      }
      if (shadows) for (const multi of [false, true]) jobs.push(prepareCompute(this.device, vdbFrameGridDensitySpec(source, { multi, maxVolumes: this.#maxVolumes, spectral })));
    }
    if (modes.includes("frame")) {
      // K6.3: the transmittance variant of the scattered-light term ('mlp' also its feature mips)
      const scattering = spectral ? "none" : this.#scattering.kind(this.settings, this.mlp);
      jobs.push(prepareCompute(this.device, vdbFrameTransmittanceSpec({ scattering, spectral })), prepareCompute(this.device, vdbFrameEnvironmentGridSpec({ spectral })));
      if (vdbFrameScatteringIsMlp(scattering)) jobs.push(prepareCompute(this.device, vdbFrameGridMipSpec({ first: true })), prepareCompute(this.device, vdbFrameGridMipSpec({ first: false })));
    }
    for (const spec of this.#env.specs(["equirect", "cube"])) jobs.push(prepareCompute(this.device, spec));
    if (this.settings.castShadows) jobs.push(prepareCompute(this.device, vdbFrameVolumeShadowSpec({ spectral })));
    if ([...sources].some((s) => s.channels.emission)) jobs.push(prepareCompute(this.device, vdbFrameEmissionReduceSpec()), prepareCompute(this.device, vdbFrameEmissionProbeSpec()));
    jobs.push(prepareCompute(this.device, vdbFrameResolveSpec(this.#resolveKey())));
    const key = this.#compositeKey();
    if (key) jobs.push(prepareRender(this.device, vdbFrameCompositeSpec(key)));
    await Promise.all(jobs);
  }

  // ---------------------------------------------------------------- update

  update(encoder: GPUCommandEncoder, frame: VDBFrameInput, options: VDBFrameUpdateOptions = {}): VDBFrameUpdate {
    const t = options.timestampWrites;
    if (t && t.beginningOfPassWriteIndex !== undefined) encoder.beginComputePass({ label: `${this.#label} begin`, timestampWrites: { querySet: t.querySet, beginningOfPassWriteIndex: t.beginningOfPassWriteIndex } }).end();
    const u = this.#update(encoder, frame, options);
    if (t && t.endOfPassWriteIndex !== undefined) encoder.beginComputePass({ label: `${this.#label} end`, timestampWrites: { querySet: t.querySet, endOfPassWriteIndex: t.endOfPassWriteIndex } }).end();
    return u;
  }

  #update(encoder: GPUCommandEncoder, frame: VDBFrameInput, options: VDBFrameUpdateOptions): VDBFrameUpdate {
    this.#check();
    this.#retire.advance();
    const s = this.settings;
    const frameLighting = s.lighting === "frame";
    // K6.4: spectral media (a pipeline key of every kernel with a spectral variant)
    const spectral = !!s.spectral;

    if (options.froxels === false && this.#textures) return { outputs: this.outputs, recorded: this.#recorded, volumes: 0 };
    const versions = [this.#shadows.version, this.#probes.version];

    // Froxel size (his getFroxelSize) and the textures
    const size = vdbFrameFroxelSize(s, frame.viewport, this.#held);
    const maxDim = this.device.limits?.maxTextureDimension3D ?? 2048;
    let W = size.width;
    let H = size.height;
    let D = Math.max(1, Math.round(s.depth));
    // multi-group partials stack 2 ('volume'; K6.4 spectral 3) or 8 ('frame') slices per froxel slice along z
    const layers = frameLighting ? VDB_FRAME_SUM_WORDS : spectral ? VDB_FRAME_GATHER_WORDS.spectral : VDB_FRAME_GATHER_WORDS.grey;
    if (W > maxDim || H > maxDim || D + 1 > maxDim || layers * D > maxDim) {
      this.#warnOnce("size", `the froxel grid ${W}x${H}x${D} exceeds maxTextureDimension3D (${maxDim}); clamped.`);
      W = Math.min(W, maxDim);
      H = Math.min(H, maxDim);
      D = Math.min(D, Math.floor(maxDim / layers));
    }

    // Collect and clip the visible volumes (his FrameVolumeCollector)
    const cam = frame.camera;
    const clip = vdbFrameClipCamera(cam.view, cam.projection, { coordinateSystem: "webgpu", reversedDepth: !!cam.reversedDepth });
    let near = Infinity;
    let far = 0;
    const clipped: { volume: FrameVolume; c: NonNullable<ReturnType<typeof vdbFrameClipVolume>> }[] = [];
    for (const volume of this.#volumes) {
      if (!volume.visible) continue;
      const c = vdbFrameClipVolume(clip, volume.params.matrixWorld);
      if (!c) continue;
      clipped.push({ volume, c });
      near = Math.min(near, c.viewNear);
      far = Math.max(far, c.viewFar);
    }
    near = Math.max(near, s.near, cam.near);
    far = Math.min(far, s.far, cam.far);
    const inRange = clipped.length > 0 && far > near;
    if (!inRange) {
      near = cam.near;
      far = cam.far;
    }
    far = Math.max(far, near * 1.0001);
    const range: [number, number] = [near, far];

    // Rects, slices and the froxel area (no froxel work without a volume in view)
    let visible: VolumeRecord[] = [];
    if (inRange) {
      for (const { volume, c } of clipped) {
        const rect = vdbFrameFroxelRect(c.localToView, cam.projection, W, H);
        if (!rect) continue;
        const slices = vdbFrameVolumeSlices(c.viewNear, c.viewFar, range, D);
        visible.push({ volume, localToView: c.localToView, viewToLocal: c.viewToLocal, worldToLocal: c.worldToLocal, rect, first: slices.first, count: slices.count, visible: true, caster: false, emitter: false });
      }
    }
    if (visible.length > this.#maxVolumes) {
      this.#warnOnce("volumes", `${visible.length} visible volumes exceed the ${this.#maxVolumes} a frame holds (uniform binding size); the rest are not drawn.`);
      visible = visible.slice(0, this.#maxVolumes);
    }
    const area = visible.length > 0 ? vdbFrameFroxelArea(visible, W, H, D) : null;
    const froxelWork = !!area && area.count > 0;
    if (!froxelWork) visible = [];

    // K6.5: volume shadow maps need the shadow grid even without froxel work; the emission light needs its emitters
    const shadowsWanted = VDBFrameShadowMaps.wanted(frame.lights, s.castShadows);
    const usesGrid = (frameLighting && froxelWork) || shadowsWanted;
    const emissionLight = frame.emissionLight ?? null;
    if (!froxelWork && !shadowsWanted && !emissionLight) {
      this.#shadows.update({ ...EMPTY_SHADOWS, lights: frame.lights, castShadows: s.castShadows });
      this.#probes.plan(null, [], [0, 0, 0], 1);
      this.#syncOutputs(versions);
      this.#recorded = false;
      return { outputs: this.outputs, recorded: false, volumes: 0 };
    }
    for (const r of visible) r.caster = usesGrid && r.volume.castsShadow;
    // Casters out of view (or not drawn) still shadow (his collector); emitters out of view still light meshes
    const extra: VolumeRecord[] = [];
    const listed = new Set(visible.map((r) => r.volume));
    for (const volume of this.#volumes) {
      const caster = usesGrid && volume.castsShadow;
      const emitter = !!emissionLight && volume.visible && !!volume.slot.channels.emission;
      if (listed.has(volume)) {
        if (emitter) visible.find((r) => r.volume === volume)!.emitter = true;
        continue;
      }
      if (!caster && !emitter) continue;
      const localToView = vdbMat4Multiply(cam.view, volume.params.matrixWorld);
      extra.push({ volume, localToView, viewToLocal: vdbMat4Invert(localToView), worldToLocal: vdbMat4Invert(volume.params.matrixWorld), rect: [0, 0, 0, 0], first: 0, count: 0, visible: false, caster, emitter });
    }
    if (visible.length + extra.length > this.#maxVolumes) {
      this.#warnOnce("casters", `${visible.length + extra.length} volumes, casters and emitters exceed the ${this.#maxVolumes} a frame holds; some are left out.`);
      extra.length = Math.max(0, this.#maxVolumes - visible.length);
    }

    // Source groups, in first-appearance order; in a group the visible non-casters, the visible casters, the
    // casters that are not drawn, then emitters that are neither (gather range: the visible ones; caster range: the casters)
    const bySource = new Map<VDBVolumeSource, VolumeRecord[]>();
    for (const r of [...visible, ...extra]) {
      const list = bySource.get(r.volume.slot.source);
      if (list) list.push(r);
      else bySource.set(r.volume.slot.source, [r]);
    }
    const groups: SourceGroup[] = [];
    const ordered: VolumeRecord[] = [];
    for (const [source, list] of bySource) {
      const sorted = [...list.filter((r) => r.visible && !r.caster), ...list.filter((r) => r.visible && r.caster), ...list.filter((r) => !r.visible && r.caster), ...list.filter((r) => !r.visible && !r.caster)];
      const g: SourceGroup = {
        source,
        records: sorted,
        first: ordered.length,
        visible: sorted.filter((r) => r.visible).length,
        casterStart: sorted.findIndex((r) => r.caster),
        casters: sorted.filter((r) => r.caster).length,
        hook: null,
      };
      if (g.casterStart < 0) g.casterStart = 0;
      groups.push(g);
      ordered.push(...sorted);
    }
    const gatherGroups = groups.filter((g) => g.visible > 0);
    const casterGroups = groups.filter((g) => g.casters > 0);
    const multi = gatherGroups.length > 1;

    // K6.5 emission colour codes (LUT rows, validated hooks; one hook per source group)
    const codes = ordered.map((r) => {
      if (!r.volume.slot.channels.emission) return 0;
      const c = this.#colors.codeOf(r.volume.params.emissionColor);
      const g = groups.find((x) => x.source === r.volume.slot.source)!;
      if (c.hook) {
        if (g.hook && g.hook !== c.hook) throw new VDBError("VDB_BAD_OPTION", "VDBFrameRenderer: volumes of one source use two different emission colour hooks; one hook per source (a pipeline key).");
        g.hook = c.hook;
      }
      return c.code;
    });
    const lut = this.#colors.texture();

    // Textures (a new size or temporal / multi-group need retires the old ones and drops the history)
    const tex = froxelWork ? this.#ensureTextures(W, H, D, s.temporal, multi, layers, spectral) : null;

    // Temporal validity (his historyValid && historyCamera === camera; plus a consecutive frameId)
    const prev = this.#previous;
    const useHistory = s.temporal && this.#historyValid && !!prev && prev.cameraId === (cam.id ?? null) && frame.frameId === prev.frameId + 1;
    if (!s.temporal) this.#historyValid = false;

    // The dispatch box: this area and every rect a texture this update writes still holds
    const historyOut = tex?.history ? tex.history[1 - this.#historyIndex]! : null;
    let box: VDBFroxelArea = area ? { rect: [...area.rect], first: 0, count: D } : { rect: [0, 0, 0, 0], first: 0, count: D };
    if (tex) {
      for (const t of [tex.scatter, tex.integrated, historyOut, tex.spectral?.scatterAlpha, tex.spectral?.integratedAlpha]) {
        const w = t && this.#written.get(t);
        if (w) box = vdbFroxelAreaUnion(box, w);
      }
    }

    // Uniforms
    const projectionInverse = vdbMat4Invert(cam.projection);
    const cameraWorld = vdbMat4Invert(cam.view);
    const fw = this.#frameWriter;
    const U = VDB_FRAME_UNIFORMS;
    fw.clear(0, U.size);
    fw.set(U, 0, "cameraWorld", cameraWorld);
    fw.set(U, 0, "projectionInverse", projectionInverse);
    fw.set(U, 0, "previousView", prev?.view ?? cam.view);
    fw.set(U, 0, "previousProjection", prev?.projection ?? cam.projection);
    fw.set(U, 0, "grid", [W, H, D, vdbFrameAtlasLayout(W, H, D).columns]);
    fw.set(U, 0, "range", [near, far, prev?.range[0] ?? near, prev?.range[1] ?? far]);
    if (area) {
      fw.set(U, 0, "areaRect", area.rect);
      fw.set(U, 0, "areaSlices", [area.first, area.first + area.count]);
    }
    fw.set(U, 0, "dispatchBox", [box.rect[0], box.rect[1], box.rect[2] - box.rect[0], box.rect[3] - box.rect[1]]);
    const jitter = s.jitter ? ((frame.frameId % 1024) * 0.618034) % 1.0 : -1.0;
    const fade = Number.isFinite(s.fadeRadius) && s.fadeRadius > 0 ? s.fadeRadius : 0;
    fw.set(U, 0, "jitterFade", [jitter, fade, Math.min(1, Math.max(0.01, s.temporalBlend)), useHistory ? 1 : 0]);
    fw.set(U, 0, "flags", [Math.max(1, Math.round(s.subSteps)), s.temporal ? 1 : 0, (cam.reversedDepth ? 1 : 0) | (cam.orthographic ? 2 : 0), 0]);
    fw.set(U, 0, "camera", [cam.near, cam.far, 0, 0]);
    // K6.3: octaves (count 0: off) and 'two-sided' for both lightings ('frame' rewrites lighting.xzw in #frameGrid)
    fw.set(U, 0, "octaves", vdbFrameScatteringOctaves(s));
    fw.set(U, 0, "lighting", [0, s.multipleScattering === "two-sided" ? 1 : 0, 0, 0]);
    // K6.4: spectral, slot groups in use ('frame' rewrites the groups in #frameGrid)
    fw.set(U, 0, "spectral", [spectral ? 1 : 0, spectral ? 1 : 0, 0, 0]);
    // K6.2b: the environment kernels this update runs (reduced on the GPU; no readback)
    this.#env.prepare(frame.environment, s);
    // K6.Q1a: with jitter and temporal accumulation the ambient ring turns by the frame's jitter (its azimuth step x
    // the jitter value; a static frame keeps the fixed ring)
    // K6.Q1a: the ambient ring turns with the frame's jitter (jitter + temporal); K6.11b internal A/B: 'fixed' keeps it
    const ringPhase = s.jitter && s.temporal && (s as VDBFrameInternalSettings).ambientOcclusionRing !== "fixed" ? jitter : 0;
    const lightInfo = this.#packLights(frame.lights, s.occlusion ? (frame.occluders ?? []) : [], frameLighting && this.#env.present && this.#env.mode === "outward", ringPhase);
    const grid = usesGrid ? this.#frameGrid(casterGroups, cameraWorld, lightInfo, jitter, frameLighting && froxelWork) : null;
    this.device.queue.writeBuffer(this.#frameBuffer, 0, fw.buffer);

    // K6.5: the volume shadow map pieces (budgeted) and the emission probe work
    const last = grid && grid.count > 0 ? grid.count - 1 : -1;
    this.#shadows.update({
      lights: frame.lights,
      castShadows: s.castShadows,
      coarse: s.castShadowCoarse,
      steps: s.castShadowSteps,
      budget: s.castShadowUpdatesPerFrame,
      cascades: s.castShadowCascades,
      castShadowDistance: s.castShadowDistance,
      camera: { view: cam.view, cameraWorld, projectionInverse, near: cam.near, far: cam.far, orthographic: !!cam.orthographic },
      grid: last >= 0 ? { min: [this.#gridMin[last]!.x, this.#gridMin[last]!.y, this.#gridMin[last]!.z], size: [this.#gridSize[last]!.x, this.#gridSize[last]!.y, this.#gridSize[last]!.z] } : null,
      castVersion: this.#gridState.castVersion,
      spectral,
    });
    this.redrawStats.volumeShadows.drawn = this.#shadows.stats.drawn;
    this.redrawStats.volumeShadows.waiting = this.#shadows.stats.waiting;
    const emitters: VDBFrameEmitterInput[] = [];
    if (emissionLight) {
      ordered.forEach((r, i) => {
        if (!r.emitter) return;
        const p = r.volume.params;
        const slot = r.volume.slot;
        emitters.push({
          record: i,
          group: groups.findIndex((g) => g.source === slot.source),
          matrixWorld: p.matrixWorld,
          key: [this.#sourceId(slot.source), slot.v, slot.version, slot.revision ?? -1, this.#densityScaleOf(p, slot), p.densityCutoff, p.extinction, p.absorbance, p.emissiveIntensity, ...p.temperatureRange, ...p.offset3D, WRAP_CODE[p.wrap3D] ?? 0, p.mipmaps ? 1 : 0, emissionLight.steps ?? 24, codes[i], groups.find((g) => g.source === slot.source)!.hook ?? "", ...Array.from({ length: 16 }, (_, k) => p.matrixWorld[k]!)].join(","),
        });
      });
    }
    const emission = this.#probes.plan(emissionLight, emitters, [cameraWorld[12]!, cameraWorld[13]!, cameraWorld[14]!], emissionLight?.steps ?? 24);

    this.#packVolumes(ordered, codes, spectral);
    this.#packGroups(groups, gatherGroups, casterGroups);

    // Sparse sources (K6.7) record their own work after the host's pool.update(encoder); group 2 follows their version.
    for (const g of groups) (g.source as VDBVolumeSource & { update?: (e: GPUCommandEncoder) => unknown }).update?.(encoder);

    // Record: [shadow grid per caster group, volume shadow maps, transmittance], [emission], gather per source group,
    // then resolve
    let pass = encoder.beginComputePass({ label: `${this.#label} froxels` });
    this.#env.record(pass);
    const group0 = this.#group0Bind();
    if (grid?.drawGrid) {
      const gt = this.#grid!;
      const r = gt.resolution;
      casterGroups.forEach((g) => {
        const built = computePipeline(this.device, vdbFrameGridDensitySpec(g.source, { multi: casterGroups.length > 1, maxVolumes: this.#maxVolumes, spectral }));
        pass.setPipeline(built.pipeline);
        pass.setBindGroup(0, group0);
        pass.setBindGroup(1, this.#gridDensityBind(built.layouts[1]!, gt, casterGroups.length > 1), [groups.indexOf(g) * VDB_FRAME_GROUP_STRIDE]);
        pass.setBindGroup(2, this.#sourceBind(g.source, built.layouts[2]!));
        pass.dispatchWorkgroups(Math.ceil(r / 4), Math.ceil(r / 4), Math.ceil((r * grid.count) / 4));
      });
    }
    if (this.#grid && grid && grid.count > 0) this.#shadows.record(pass, computePipeline(this.device, vdbFrameVolumeShadowSpec({ spectral })), group0, this.#grid.shadow);
    // K6.3: the scattered-light term's grid mips ('mlp': box mips in this pass; 'multi-scale': K7.3's own pass), redrawn
    // with the shadow grid
    if (grid?.drawTransmittance && grid.count > 0) {
      const kind = grid.scattering;
      const version = this.#gridDraws;
      if (vdbFrameScatteringIsMlp(kind) && this.#scattering.mipsStale(kind, version)) this.#scattering.recordMips(pass, group0, this.#grid!.shadow, version);
      if (kind === "multi-scale" && this.#scattering.mipsStale(kind, version)) {
        pass.end();
        this.#scattering.recordMultiScale(encoder, this.#grid!.shadow, this.#grid!.shadowLayout.origins[grid.count - 1]!, version);
        pass = encoder.beginComputePass({ label: `${this.#label} froxels (after the multi-scale mips)` });
      }
    }
    if (grid?.drawTransmittance) {
      const gt = this.#grid!;
      const r = gt.resolution;
      const built = computePipeline(this.device, vdbFrameTransmittanceSpec({ scattering: grid.scattering, spectral }));
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, this.#transmittanceBind(built.layouts[1]!, gt));
      const group2 = spectral ? null : this.#scattering.group2(grid.scattering, built.layouts[2]);
      if (group2) pass.setBindGroup(2, group2);
      // K6.4 spectral: one dispatch over the cascades x the slot groups in use
      pass.dispatchWorkgroups(Math.ceil(r / 4), Math.ceil(r / 4), Math.ceil((r * grid.count * (spectral ? grid.groups : 1)) / 4));
    }
    if (grid?.drawEnvironmentGrid) {
      const gt = this.#grid!;
      const r = gt.resolution;
      const built = computePipeline(this.device, vdbFrameEnvironmentGridSpec({ spectral }));
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, this.#environmentGridBind(built.layouts[1]!, gt));
      pass.dispatchWorkgroups(Math.ceil(r / 4), Math.ceil(r / 4), Math.ceil(r / 4));
    }
    if (emission) this.#recordEmission(pass, group0, groups, emission, lut);
    if (tex && area) {
      gatherGroups.forEach((g) => {
        const built = computePipeline(this.device, vdbFrameGatherSpec(g.source, { multi, maxVolumes: this.#maxVolumes, lighting: s.lighting, hook: g.hook, spectral }));
        pass.setPipeline(built.pipeline);
        pass.setBindGroup(0, group0);
        pass.setBindGroup(1, this.#gatherBind(built.layouts[1]!, tex, multi, frameLighting, g.source.channels.emission ? lut : null), [groups.indexOf(g) * VDB_FRAME_GROUP_STRIDE]);
        pass.setBindGroup(2, this.#sourceBind(g.source, built.layouts[2]!));
        pass.dispatchWorkgroups(Math.ceil((box.rect[2] - box.rect[0]) / 8), Math.ceil((box.rect[3] - box.rect[1]) / 8), D);
      });
      const resolve = computePipeline(this.device, vdbFrameResolveSpec(this.#resolveKey()));
      pass.setPipeline(resolve.pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(1, this.#resolveBind(resolve.layouts[1]!, tex, s.temporal));
      pass.dispatchWorkgroups(Math.ceil((box.rect[2] - box.rect[0]) / 8), Math.ceil((box.rect[3] - box.rect[1]) / 8), 1);
    }
    pass.end();
    this.#syncOutputs(versions);

    if (!tex || !area) {
      this.#recorded = false;
      return { outputs: this.outputs, recorded: false, volumes: 0 };
    }
    // Book-keeping: what each texture holds, the history swap, the previous frame
    for (const t of [tex.scatter, tex.integrated, historyOut, tex.spectral?.scatterAlpha, tex.spectral?.integratedAlpha]) if (t) this.#written.set(t, area);
    if (tex.history) {
      this.#historyIndex = 1 - this.#historyIndex;
      this.#historyValid = true;
    }
    this.#previous = { view: Float64Array.from(cam.view), projection: Float64Array.from(cam.projection), range, cameraId: cam.id ?? null, frameId: frame.frameId };
    const layout: VDBFroxelLayout = { width: W, height: H, depth: D, range, screenRect: vdbFrameClipRect(area.rect, W, H, 1) };
    this.outputs.froxels = { integrated: tex.integrated, integratedAlpha: tex.spectral?.integratedAlpha ?? null, layout };
    this.#recorded = true;
    return { outputs: this.outputs, recorded: true, volumes: visible.length };
  }

  /** K6.5: the shadow maps and emission probes into the outputs (a new texture object bumps `outputs.version`). */
  #syncOutputs(versions: readonly number[]): void {
    if (this.#shadows.version !== versions[0] || this.#probes.version !== versions[1]) this.outputs.version++;
    this.outputs.emissionProbes = this.#probes.outputs;
  }

  /** K6.5: the emission direction passes (per source group with changed emitters), the reduce and the probes. */
  #recordEmission(pass: GPUComputePassEncoder, group0: GPUBindGroup, groups: readonly SourceGroup[], plan: VDBFrameEmissionPlan, lut: GPUTexture): void {
    const P = this.#probes;
    for (const [index, range] of plan.groups) {
      const g = groups[index]!;
      const built = computePipeline(this.device, vdbFrameEmissionDirectionSpec(g.source, { maxVolumes: this.#maxVolumes, hook: g.hook }));
      const b = EMISSION_DIRECTION_BINDINGS;
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, group0);
      const group1 = this.#cached("emission-direction", [built.layouts[1]!, P.listBuffer, lut, P.directionTexture!], () =>
        this.device.createBindGroup({
          label: `${this.#label} emission direction group 1`,
          layout: built.layouts[1]!,
          entries: [
            { binding: b.volumes, resource: { buffer: this.#volumesBuffer } },
            { binding: b.emitters, resource: { buffer: P.listBuffer, size: VDB_FRAME_EMITTERS.size } },
            { binding: b.lut, resource: lut.createView() },
            { binding: b.directions, resource: P.directionTexture!.createView() },
          ],
        }),
      );
      pass.setBindGroup(1, group1, [range.segment * VDB_FRAME_EMITTERS_STRIDE]);
      pass.setBindGroup(2, this.#sourceBind(g.source, built.layouts[2]!));
      pass.dispatchWorkgroups(VDB_FRAME_EMISSION_DIRECTIONS, range.count, 1);
    }
    if (plan.reduce) {
      const built = computePipeline(this.device, vdbFrameEmissionReduceSpec());
      const b = EMISSION_REDUCE_BINDINGS;
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(
        1,
        this.#cached("emission-reduce", [built.layouts[1]!, P.listBuffer, P.directionTexture!, P.emitterTexture!], () =>
          this.device.createBindGroup({
            label: `${this.#label} emission reduce group 1`,
            layout: built.layouts[1]!,
            entries: [
              { binding: b.emitters, resource: { buffer: P.listBuffer, size: VDB_FRAME_EMITTERS.size } },
              { binding: b.directions, resource: P.directionTexture!.createView() },
              { binding: b.out, resource: P.emitterTexture!.createView() },
            ],
          }),
        ),
      );
      pass.dispatchWorkgroups(Math.ceil(P.slots / 64), 1, 1);
    }
    if (plan.probes && P.outputs) {
      const built = computePipeline(this.device, vdbFrameEmissionProbeSpec());
      const b = EMISSION_PROBE_BINDINGS;
      const r = P.outputs.state.resolution;
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(
        1,
        this.#cached("emission-probes", [built.layouts[1]!, P.emitterTexture!, P.probeTexture!], () =>
          this.device.createBindGroup({
            label: `${this.#label} emission probes group 1`,
            layout: built.layouts[1]!,
            entries: [
              { binding: b.grid, resource: { buffer: P.gridBuffer } },
              { binding: b.emitters, resource: P.emitterTexture!.createView() },
              { binding: b.probes, resource: P.probeTexture!.createView({ dimension: "3d" }) },
            ],
          }),
        ),
      );
      pass.dispatchWorkgroups(Math.ceil(r / 4), Math.ceil(r / 4), Math.ceil((r * P.outputs.state.count) / 4));
    }
  }

  /** A stable number per source object (the caster signature's source identity). */
  #sourceId(source: object): number {
    let id = this.#sourceIds.get(source);
    if (id === undefined) this.#sourceIds.set(source, (id = this.#nextSourceId++));
    return id;
  }

  /**
   * 'frame' lighting's CPU work (his updateGrid, updateGridSignatures, transmittanceUnchanged): the cascades around
   * the casters and the camera, the grid textures, the grid uniforms, and which kernels must run.
   */
  #frameGrid(
    casterGroups: readonly SourceGroup[],
    cameraWorld: ArrayLike<number>,
    lights: { sorted: readonly number[][]; sky: readonly number[]; occluderOf: readonly (VDBFrameOccluder | null)[]; ambient: readonly number[] },
    jitter: number,
    transmittance = true,
  ): { count: number; drawGrid: boolean; drawTransmittance: boolean; drawEnvironmentGrid: boolean; scattering: VDBFrameScatteringKernel; groups: number } {
    const s = this.settings;
    const fw = this.#frameWriter;
    const U = VDB_FRAME_UNIFORMS;
    const cascades = Math.min(VDB_FRAME_MAX_CASCADES, Math.max(1, Math.round(s.shadowCascades)));
    const spectral = !!s.spectral;
    const casters = casterGroups.flatMap((g) => g.records.filter((r) => r.caster));
    const boxes = casters.map((r) => {
      const b = vdbUnitBoxBounds(r.volume.params.matrixWorld);
      return { min: { x: b.min[0], y: b.min[1], z: b.min[2] }, max: { x: b.max[0], y: b.max[1], z: b.max[2] } };
    });
    const base = Math.max(2, Math.round(s.shadowResolution));
    // K6.Q1b: the size-adaptive grid (`shadowResolutionMax` above `shadowResolution`), from the casters' boxes and media
    const resolution = this.#gridResolution(s.shadowResolutionMax > base ? this.#adaptiveResolution(casters, boxes, base) : base, cascades, spectral);
    this.shadowGridResolution = resolution;
    // K6.4: the slot groups in use (his getFrameSpectralGroups over the scene slots and the environment slot, set below)
    const sceneSlotCount = Math.min(lights.sorted.length, VDB_FRAME_SHADOW_SLOTS);
    const envSlotWanted = this.#env.present && this.#env.mode === "directional" && sceneSlotCount < VDB_FRAME_SHADOW_SLOTS && !(!!s.multipleScattering && s.multipleScattering !== "octaves");
    const groups = spectral ? vdbFrameSpectralGroups(sceneSlotCount + (envSlotWanted ? 1 : 0)) : 0;
    const gt = this.#ensureGrid(resolution, cascades, casterGroups.length > 1, spectral, groups);
    const cameraPosition = { x: cameraWorld[12]!, y: cameraWorld[13]!, z: cameraWorld[14]! };
    const count = casters.length > 0 ? vdbFrameGridCascades(boxes, cameraPosition, s.shadowDistance, resolution, cascades, this.#gridMin, this.#gridSize, this.#gridVoxel) : 0;
    vdbFrameUpdateGridSignatures(this.#gridState, count > 0 ? casters : [], count, resolution, false, this.#gridMin, this.#gridSize, (values, r) => {
      const p = r.volume.params;
      const slot = r.volume.slot;
      for (let i = 0; i < 16; i++) values.push(p.matrixWorld[i]!);
      // K4.22: `revision` too (a same-size dense replace keeps `version`: a sequence frame, a streamed page)
      values.push(this.#sourceId(slot.source), slot.v, slot.version, slot.revision ?? -1, ...p.baseColor, this.#densityScaleOf(p, slot), p.densityCutoff, p.extinction, p.absorbance, p.mipmaps ? 1 : 0, WRAP_CODE[p.wrap3D] ?? 0, ...p.offset3D);
      // K6.4: the spectral grid holds each channel's extinction
      if (spectral) values.push(...p.extinctionScale, p.angstromExponent);
    });
    const shadowSlots = Math.min(lights.sorted.length, VDB_FRAME_SHADOW_SLOTS);
    const ms = vdbFrameMultipleScatteringValue(s.multipleScattering === "octaves" ? false : s.multipleScattering, this.#phaseOf(null));
    const diffusion = !!s.multipleScattering && s.multipleScattering !== "octaves";
    const phaseFields = this.#phaseOf(null);
    const phase = vdbFramePhaseVector(phaseFields);
    const twoSided = s.multipleScattering === "two-sided";
    const octaves = vdbFrameScatteringOctaves(s);
    const v =(a: { x: number; y: number; z: number }) => [a.x, a.y, a.z, 0];
    for (let c = 0; c < VDB_FRAME_MAX_CASCADES; c++) {
      const live = c < count;
      if (!live) continue;
      const at = (name: string, value: number[]) => {
        const slot = U.fields.get(name)!;
        const base = (slot.offset + c * slot.stride) / 4;
        for (let k = 0; k < value.length; k++) fw.f32[base + k] = value[k]!;
      };
      at("gridMin", v(this.#gridMin[c]!));
      at("gridSize", v(this.#gridSize[c]!));
      at("gridVoxel", v(this.#gridVoxel[c]!));
      at("shadowOrigin", [...gt.shadowLayout.origins[c]!, 0]);
      if (!spectral) {
        for (let k = 0; k < 4; k++) {
          const slot = U.fields.get("transmittanceOrigin")!;
          const base = (slot.offset + (4 * c + k) * slot.stride) / 4;
          const o = gt.transmittanceLayout.origins[4 * c + k]!;
          fw.f32[base] = o[0];
          fw.f32[base + 1] = o[1];
          fw.f32[base + 2] = o[2];
        }
      }
    }
    // K6.4 spectral: target k of slot group g of cascade c at [3 (g count + c) + k]
    if (spectral && count > 0) {
      const slot = U.fields.get("transmittanceOrigin")!;
      for (let g = 0; g < groups; g++)
        for (let c = 0; c < count; c++)
          for (let k = 0; k < 3; k++) {
            const i = 3 * (g * count + c) + k;
            const o = gt.transmittanceLayout.origins[i]!;
            const base = (slot.offset + i * slot.stride) / 4;
            fw.f32[base] = o[0];
            fw.f32[base + 1] = o[1];
            fw.f32[base + 2] = o[2];
          }
    }
    fw.set(U, 0, "spectral", [spectral ? 1 : 0, spectral ? groups : 0, 0, 0]);
    fw.set(U, 0, "gridLayout", [count, resolution, vdbFrameAtlasLayout(resolution, resolution, resolution * cascades).columns, shadowSlots]);
    fw.set(U, 0, "lighting", [ms, twoSided ? 1 : 0, count > 0 ? 1 : 0, diffusion ? 1 : 0]);
    fw.set(U, 0, "framePhase", phase);
    fw.set(U, 0, "framePhaseSpectrum", [...vdbFramePhaseSpectrum(phaseFields), 0]);
    // K6.2b: his light result. The dominant 'directional' environment light takes the next free slot (not with
    // diffusion: then every environment light goes to the environment grid, opposite pairs get the same march);
    // the environment grid holds the others. Its direction and colour are on the GPU only (the buffer).
    const env = this.#env;
    const envSlot = env.present && env.mode === "directional" && shadowSlots < VDB_FRAME_SHADOW_SLOTS && !diffusion;
    const envGrid = env.present && env.mode !== "outward" && count > 0;
    fw.set(U, 0, "environment", [envSlot ? 1 : 0, envSlot ? 1 : 0, envGrid ? 1 : 0, env.present ? 1 : 0]);

    // His redraw skips: the grid when its version or texture changed; the transmittance when its signature did (a
    // jittered grid is drawn every frame)
    const drawGrid = count > 0 && (this.#gridDrawn !== this.#gridState.gridVersion);
    if (count > 0) {
      if (drawGrid) {
        this.redrawStats.shadowDensity.drawn++;
        this.#gridDraws++;
      }
      else this.redrawStats.shadowDensity.skipped++;
      this.#gridDrawn = this.#gridState.gridVersion;
    }
    // K6.5: the volume shadow maps alone ('volume' lighting, or no volume in view) need only the grid
    if (!transmittance) return { count, drawGrid, drawTransmittance: false, drawEnvironmentGrid: false, scattering: "none", groups };
    // K6.3: the scattered-light term of the slots ('mlp', 'multi-scale' as a pipeline key; two-sided, octaves uniforms)
    const scattering = count > 0 && !spectral ? this.#scattering.kind(s, this.mlp) : "none";
    const last = Math.max(0, count - 1);
    const plan =
      count > 0
        ? this.#scattering.plan(
            scattering,
            s,
            {
              resolution,
              min: v(this.#gridMin[last]!),
              size: v(this.#gridSize[last]!),
              voxel: v(this.#gridVoxel[last]!),
              origin: gt.shadowLayout.origins[last]!,
            },
            ms,
          )
        : { kind: scattering, signature: [] };
    // K6.Q1: the shadowed ambient's pairs and directions
    const signature: unknown[] = [this.#gridDraws, gt.shadow, gt.transmittance, shadowSlots, ms, ...lights.sky, envSlot, envSlot ? env.version : -1, twoSided, ...octaves, ...plan.signature, ...lights.ambient, spectral, groups];
    for (let i = 0; i < shadowSlots; i++) signature.push(...lights.sorted[i]!.slice(0, 4));
    // K6.5b: the slot lights' occluder maps (view, matrices, bias, flags, revision); a map without a revision changes
    // every update
    let occluderChanging = false;
    for (let i = 0; i < shadowSlots; i++) {
      const o = lights.occluderOf[i];
      if (!o) continue;
      if (o.revision === undefined) occluderChanging = true;
      signature.push(i, o.depth, o.revision ?? -1, o.bias ?? 0, o.reversedDepth ? 1 : 0, o.filter ?? "bilinear", o.viewProjection.length);
      for (const m of o.viewProjection) for (let k = 0; k < 16; k++) signature.push(m[k]);
    }
    const redraw = jitter >= 0 || occluderChanging;
    const drawTransmittance = count > 0 && (redraw || !vdbSameValues(signature, this.#transmittanceSignature));
    this.#transmittanceSignature = redraw ? [] : signature;
    if (count > 0) {
      if (drawTransmittance) this.redrawStats.transmittance.drawn++;
      else this.redrawStats.transmittance.skipped++;
    }
    // his renderEnvironmentGrid signature: the grid (its version covers the cascades and casters), ms, the lights
    const envSignature: unknown[] = [this.#gridDraws, gt.shadow, gt.environment, count, ms, envSlot, env.version, ...octaves, spectral];
    const drawEnvironmentGrid = envGrid && !vdbSameValues(envSignature, this.#envGridSignature);
    if (envGrid) {
      if (drawEnvironmentGrid) this.redrawStats.environmentGrid.drawn++;
      else this.redrawStats.environmentGrid.skipped++;
      this.#envGridSignature = envSignature;
    }
    return { count, drawGrid, drawTransmittance, drawEnvironmentGrid, scattering: plan.kind, groups };
  }

  /**
   * K6.Q1b: the size-adaptive shadow grid side for these casters (`vdbFrameAdaptiveGridResolution`): each caster's
   * world box, voxels across it and the shadow grid's extinction at its mean density (absorbance included, as the grid
   * density pass) give the cell size of the rule; the side covers the casters' box with it, within [base, max].
   */
  #adaptiveResolution(casters: readonly VolumeRecord[], boxes: readonly VDBBox3Like[], base: number): number {
    const r = (this.settings as VDBFrameInternalSettings).shadowResolutionRule;
    const rule = r === "voxels" || r === "optical" ? r : "both";
    const inputs = casters.map((r, i) => {
      const p = r.volume.params;
      const slot = r.volume.slot;
      const b = boxes[i]!;
      const mean = p.densityMean ?? slot.stats.meanNonZero;
      return {
        size: [b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z] as [number, number, number],
        texels: [...slot.texels] as [number, number, number],
        meanExtinction: (mean * this.#densityScaleOf(p, slot) * p.extinction * p.absorbance) / Math.max(vdbMat4MaxScale(p.matrixWorld), 1e-30),
      };
    });
    return vdbFrameAdaptiveGridResolution(boxes, vdbFrameAdaptiveCellSize(inputs, rule), base, Math.round(this.settings.shadowResolutionMax));
  }

  /**
   * K6.8 / K6.10a: the shadow grid resolution the device can hold. The transmittance stack (4 rgba16float targets per
   * cascade, 3 per slot group and cascade when spectral) is one texture that must tile in `maxTextureDimension3D`; Dawn
   * zero-fills a new texture through one staging buffer of its size, so a stack larger than the device's
   * `maxBufferSize` (256 MiB by default: R = 256 with one cascade is 512 MiB) fails the host's submit and the frame is
   * lost. Clamp R so the stack tiles and fits (warned once; `vdbFrameFitGridResolution`).
   */
  #gridResolution(resolution: number, cascades: number, spectral: boolean): number {
    const key = `${resolution}-${cascades}-${spectral}`;
    const cached = this.#gridFits.get(key);
    if (cached !== undefined) return cached;
    const limits = this.device.limits ?? {};
    const fit = vdbFrameFitGridResolution(resolution, cascades, spectral, limits);
    if (fit.reason) {
      const targets = vdbFrameStackTargets(cascades, spectral);
      const maxDim = limits.maxTextureDimension3D ?? VDB_FRAME_DEFAULT_LIMITS.maxTextureDimension3D;
      const maxMiB = Math.round((limits.maxBufferSize ?? VDB_FRAME_DEFAULT_LIMITS.maxBufferSize) / 1048576);
      const why = fit.reason === "tile" ? `its ${targets} stack targets of ${resolution}³ do not tile in maxTextureDimension3D` : `its transmittance stack of ${Math.round((resolution ** 3 * targets * 8) / 1048576)} MiB exceeds maxBufferSize`;
      this.#warnOnce(`grid-resolution-${key}`, `settings.shadowResolution ${resolution}: ${why} (the device: maxTextureDimension3D ${maxDim}, maxBufferSize ${maxMiB} MiB); clamped to ${fit.size}.`);
    }
    this.#gridFits.set(key, fit.size);
    return fit.size;
  }

  /** The grid textures for this resolution and cascade count (a change retires the old ones and redraws). */
  #ensureGrid(resolution: number, cascades: number, multi: boolean, spectral = false, groups = 0): GridTextures {
    const g = this.#grid;
    // K6.4: spectral grids keep the absorption stack and as many slot groups as were ever used (a growth reallocates)
    const sameKind = !!g && g.resolution === resolution && g.cascades === cascades && !!g.absorption === spectral && (!spectral || g.groups >= groups);
    // spectral multi-group partials: 3 words per voxel (the stack depth × 3)
    const partialSize = (layout: VDBFrameStackLayout): [number, number, number] => [layout.size[0], layout.size[1], layout.size[2] * (spectral ? VDB_FRAME_SPECTRAL_GRID_WORDS : 1)];
    if (g && sameKind && (!multi || g.partials)) return g;
    const maxDim = this.device.limits?.maxTextureDimension3D ?? 2048;
    if (g && sameKind) {
      g.partials = createFroxelTexture(this.device, `${this.#label} grid partials`, partialSize(g.shadowLayout), "r32uint", false);
      return g;
    }
    if (g) {
      this.#retire.retire(g.shadow);
      this.#retire.retire(g.transmittance);
      this.#retire.retire(g.partials);
      this.#retire.retire(g.environment);
      this.#retire.retire(g.absorption);
    }
    const shadowLayout = vdbFrameStackLayout(resolution, cascades, maxDim);
    const keptGroups = spectral ? Math.min(VDB_FRAME_SPECTRAL_GROUPS, Math.max(groups, g?.groups ?? 0, 1)) : 0;
    const transmittanceLayout = vdbFrameStackLayout(resolution, spectral ? 3 * keptGroups * cascades : 4 * cascades, maxDim);
    const next: GridTextures = {
      resolution,
      cascades,
      shadowLayout,
      transmittanceLayout,
      shadow: createFroxelTexture(this.device, `${this.#label} shadow grid`, shadowLayout.size, "rgba16float"),
      transmittance: createFroxelTexture(this.device, `${this.#label} transmittance`, transmittanceLayout.size, "rgba16float"),
      partials: multi ? createFroxelTexture(this.device, `${this.#label} grid partials`, partialSize(shadowLayout), "r32uint", false) : null,
      environment: createFroxelTexture(this.device, `${this.#label} environment grid`, [resolution, resolution, resolution], "rgba16float"),
      absorption: spectral ? createFroxelTexture(this.device, `${this.#label} shadow grid absorption`, shadowLayout.size, "rgba16float") : null,
      groups: keptGroups,
    };
    this.#grid = next;
    // A new texture is empty: draw the grid and the transmittance again
    this.#gridDrawn = -1;
    this.#transmittanceSignature = [];
    this.#envGridSignature = [];
    return next;
  }

  #ensureTextures(W: number, H: number, D: number, temporal: boolean, multi: boolean, layers: number, spectral = false): Textures {
    const t = this.#textures;
    // K6.4: grey and spectral froxel textures differ (the alpha textures, a history twice as deep)
    const sameSize = !!t && t.size[0] === W && t.size[1] === H && t.size[2] === D && !!t.spectral === spectral;
    if (t && sameSize && (!temporal || t.history) && (!multi || (t.partials && t.partialLayers === layers))) {
      return t;
    }
    let changed = false;
    const next: Textures = sameSize && t ? { ...t } : (() => {
      if (t) {
        this.#retire.retire(t.scatter);
        this.#retire.retire(t.integrated);
        t.history?.forEach((h) => this.#retire.retire(h));
        this.#retire.retire(t.partials);
        this.#retire.retire(t.spectral?.scatterAlpha);
        this.#retire.retire(t.spectral?.integratedAlpha);
      }
      changed = true;
      this.#historyValid = false;
      const alphaFormat = this.#tier1 ? "rg16float" : "rgba16float";
      return {
        size: [W, H, D],
        scatter: createFroxelTexture(this.device, `${this.#label} scatter`, [W, H, D], "rgba16float"),
        integrated: createFroxelTexture(this.device, `${this.#label} integrated`, [W, H, D + 1], "rgba16float"),
        history: null,
        partials: null,
        partialLayers: 0,
        spectral: spectral
          ? {
              scatterAlpha: createFroxelTexture(this.device, `${this.#label} scatter alpha`, [W, H, D], "rgba16float"),
              integratedAlpha: createFroxelTexture(this.device, `${this.#label} integrated alpha`, [W, H, D + 1], alphaFormat),
              alphaFormat,
            }
          : null,
      } as Textures;
    })();
    if (temporal && !next.history) {
      // K6.4 spectral: the scatter history at z < D, the alpha history at z >= D
      const depth = spectral ? 2 * D : D;
      next.history = [createFroxelTexture(this.device, `${this.#label} history 0`, [W, H, depth], "rgba16float"), createFroxelTexture(this.device, `${this.#label} history 1`, [W, H, depth], "rgba16float")];
      this.#historyValid = false;
      this.#historyIndex = 0;
    }
    if (multi && (!next.partials || next.partialLayers !== layers)) {
      this.#retire.retire(next.partials);
      next.partials = createFroxelTexture(this.device, `${this.#label} partials`, [W, H, layers * D], "r32uint", false);
      next.partialLayers = layers;
    }
    this.#textures = next;
    if (changed) this.outputs.version++;
    return next;
  }

  /** The resolve kernel's key: temporal, K6.4 spectral and the alpha prefix format. */
  #resolveKey(): { temporal: boolean; spectral: boolean; alphaFormat: "rg16float" | "rgba16float" } {
    return { temporal: this.settings.temporal, spectral: !!this.settings.spectral, alphaFormat: this.#tier1 ? "rg16float" : "rgba16float" };
  }

  #group0Bind(): GPUBindGroup {
    if (!this.#group0) {
      const layout = computePipeline(this.device, vdbFrameResolveSpec({ temporal: false })).layouts[0]!;
      this.#group0 = this.device.createBindGroup({
        label: `${this.#label} group 0`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.#frameBuffer } },
          { binding: 1, resource: { buffer: this.#lightsBuffer } },
          { binding: 2, resource: this.#linear },
          { binding: 3, resource: { buffer: this.#env.buffer } },
          { binding: 4, resource: { buffer: this.#mie } },
        ],
      });
    }
    return this.#group0;
  }

  readonly #bindCache = new Map<string, { keys: readonly object[]; group: GPUBindGroup }>();

  /** A bind group cached on the identity of everything it binds (rebuilt when one changes). */
  #cached(name: string, keys: readonly object[], make: () => GPUBindGroup): GPUBindGroup {
    const hit = this.#bindCache.get(name);
    if (hit && hit.keys.length === keys.length && hit.keys.every((k, i) => k === keys[i])) return hit.group;
    const group = make();
    this.#bindCache.set(name, { keys, group });
    return group;
  }

  /** K6.5b: the 4 occluder slot views of this update (the placeholder in empty slots). */
  #occluderSlots(): GPUTextureView[] {
    if (!this.#occluderPlaceholder) {
      const texture = this.device.createTexture({ label: `${this.#label} occluder placeholder`, size: [1, 1, 1], format: "r32float", usage: 0x04 /* TEXTURE_BINDING */ });
      this.#occluderPlaceholder = { texture, view: texture.createView({ dimension: "2d-array" }) };
    }
    const placeholder = this.#occluderPlaceholder.view;
    return [0, 1, 2, 3].map((k) => this.#occluderViews[k] ?? placeholder);
  }

  /** K6.5b: the group-1 entries of the occluder slots. */
  #occluderEntries(views: readonly GPUTextureView[]): GPUBindGroupEntry[] {
    return views.map((view, k) => ({ binding: VDB_FRAME_OCCLUDER_BINDING + k, resource: view }));
  }

  #gatherBind(layout: GPUBindGroupLayout, tex: Textures, multi: boolean, frameLighting: boolean, lut: GPUTexture | null): GPUBindGroup {
    const b = frameLighting ? FRAME_GATHER_BINDINGS : GATHER_BINDINGS;
    const transmittance = frameLighting ? this.#grid!.transmittance : null;
    const environment = frameLighting ? this.#grid!.environment : null;
    const alpha = tex.spectral?.scatterAlpha ?? null;
    const name = `gather${frameLighting ? "-frame" : ""}${multi ? "-multi" : ""}${lut ? "-lut" : ""}${alpha ? "-spectral" : ""}`;
    const occluders = this.#occluderSlots();
    return this.#cached(name, [layout, tex.scatter, ...(multi ? [tex.partials!] : []), ...(transmittance ? [transmittance, environment!] : []), ...(lut ? [lut] : []), ...(alpha ? [alpha] : []), this.#groupsBuffer, ...occluders], () =>
      this.device.createBindGroup({
        label: `${this.#label} gather group 1`,
        layout,
        entries: [
          { binding: b.volumes, resource: { buffer: this.#volumesBuffer } },
          { binding: b.group, resource: { buffer: this.#groupsBuffer, size: VDB_FRAME_GROUP.size } },
          { binding: b.scatter, resource: tex.scatter.createView({ dimension: "3d" }) },
          ...(multi ? [{ binding: b.partials, resource: tex.partials!.createView({ dimension: "3d" }) }] : []),
          ...(transmittance
            ? [
                { binding: FRAME_GATHER_BINDINGS.transmittance, resource: transmittance.createView({ dimension: "3d" }) },
                { binding: FRAME_GATHER_BINDINGS.environmentGrid, resource: environment!.createView({ dimension: "3d" }) },
              ]
            : []),
          ...(lut ? [{ binding: b.emissionLut, resource: lut.createView() }] : []),
          ...(alpha ? [{ binding: b.scatterAlpha, resource: alpha.createView({ dimension: "3d" }) }] : []),
          ...this.#occluderEntries(occluders),
        ],
      }),
    );
  }

  #gridDensityBind(layout: GPUBindGroupLayout, gt: GridTextures, multi: boolean): GPUBindGroup {
    const b = GRID_DENSITY_BINDINGS;
    const absorption = gt.absorption;
    return this.#cached(`grid-density${multi ? "-multi" : ""}${absorption ? "-spectral" : ""}`, [layout, gt.shadow, ...(multi ? [gt.partials!] : []), ...(absorption ? [absorption] : []), this.#groupsBuffer], () =>
      this.device.createBindGroup({
        label: `${this.#label} grid density group 1`,
        layout,
        entries: [
          { binding: b.volumes, resource: { buffer: this.#volumesBuffer } },
          { binding: b.group, resource: { buffer: this.#groupsBuffer, size: VDB_FRAME_GROUP.size } },
          { binding: b.grid, resource: gt.shadow.createView({ dimension: "3d" }) },
          ...(multi ? [{ binding: b.partials, resource: gt.partials!.createView({ dimension: "3d" }) }] : []),
          ...(absorption ? [{ binding: b.absorption, resource: absorption.createView({ dimension: "3d" }) }] : []),
        ],
      }),
    );
  }

  #transmittanceBind(layout: GPUBindGroupLayout, gt: GridTextures): GPUBindGroup {
    const b = TRANSMITTANCE_BINDINGS;
    const occluders = this.#occluderSlots();
    const absorption = gt.absorption;
    return this.#cached(`transmittance${absorption ? "-spectral" : ""}`, [layout, gt.shadow, gt.transmittance, ...(absorption ? [absorption] : []), ...occluders], () =>
      this.device.createBindGroup({
        label: `${this.#label} transmittance group 1`,
        layout,
        entries: [
          { binding: b.grid, resource: gt.shadow.createView({ dimension: "3d" }) },
          { binding: b.transmittance, resource: gt.transmittance.createView({ dimension: "3d" }) },
          ...(absorption ? [{ binding: b.absorption, resource: absorption.createView({ dimension: "3d" }) }] : []),
          ...this.#occluderEntries(occluders),
        ],
      }),
    );
  }

  #environmentGridBind(layout: GPUBindGroupLayout, gt: GridTextures): GPUBindGroup {
    const b = ENV_GRID_BINDINGS;
    const absorption = gt.absorption;
    return this.#cached(`environment-grid${absorption ? "-spectral" : ""}`, [layout, gt.shadow, gt.environment, ...(absorption ? [absorption] : [])], () =>
      this.device.createBindGroup({
        label: `${this.#label} environment grid group 1`,
        layout,
        entries: [
          { binding: b.grid, resource: gt.shadow.createView({ dimension: "3d" }) },
          { binding: b.out, resource: gt.environment.createView({ dimension: "3d" }) },
          ...(absorption ? [{ binding: b.absorption, resource: absorption.createView({ dimension: "3d" }) }] : []),
        ],
      }),
    );
  }

  #resolveBind(layout: GPUBindGroupLayout, tex: Textures, temporal: boolean): GPUBindGroup {
    const b = RESOLVE_BINDINGS;
    const historyIn = tex.history ? tex.history[this.#historyIndex]! : null;
    const historyOut = tex.history ? tex.history[1 - this.#historyIndex]! : null;
    const sp = tex.spectral;
    return this.#cached(`resolve${temporal ? `-${this.#historyIndex}` : ""}${sp ? "-spectral" : ""}`, [layout, tex.scatter, tex.integrated, ...(temporal ? [historyIn!, historyOut!] : []), ...(sp ? [sp.scatterAlpha, sp.integratedAlpha] : [])], () =>
      this.device.createBindGroup({
        label: `${this.#label} resolve group 1`,
        layout,
        entries: [
          { binding: b.scatter, resource: tex.scatter.createView({ dimension: "3d" }) },
          { binding: b.integrated, resource: tex.integrated.createView({ dimension: "3d" }) },
          ...(temporal
            ? [
                { binding: b.historyIn, resource: historyIn!.createView({ dimension: "3d" }) },
                { binding: b.historyOut, resource: historyOut!.createView({ dimension: "3d" }) },
              ]
            : []),
          ...(sp
            ? [
                { binding: b.scatterAlpha, resource: sp.scatterAlpha.createView({ dimension: "3d" }) },
                { binding: b.integratedAlpha, resource: sp.integratedAlpha.createView({ dimension: "3d" }) },
              ]
            : []),
        ],
      }),
    );
  }

  #sourceBind(source: VDBVolumeSource, layout: GPUBindGroupLayout): GPUBindGroup {
    const hit = this.#sourceGroups.get(source);
    if (hit && hit.version === source.version && hit.layout === layout) return hit.group;
    const group = this.device.createBindGroup({ label: `${this.#label} source group 2`, layout, entries: source.bindEntries(SOURCE_SLOT) });
    this.#sourceGroups.set(source, { version: source.version, layout, group });
    return group;
  }

  // ---------------------------------------------------------------- packing

  /**
   * Packs the lights; returns the sorted records (directional, point, spot; ≤ 16), the sky direction and (K6.5b) each
   * sorted light's occluder map (the first 4 maps of this frame's directional, point and spot lights take slots 0–3).
   */
  #packLights(
    lights: readonly VDBFrameLight[],
    occluders: readonly VDBFrameOccluder[],
    outwardEnvironment = false,
    ringPhase = 0,
  ): { sorted: number[][]; sky: number[]; occluderOf: (VDBFrameOccluder | null)[]; ambient: number[] } {
    const w = this.#lightsWriter;
    const L = VDB_FRAME_LIGHTS;
    w.clear(0, L.size);
    const ambient = [0, 0, 0];
    const probe = new Float32Array(36);
    const hemi: { sky: readonly number[]; ground: readonly number[]; up: number[] }[] = [];
    const hemiSkySum = [0, 0, 0];
    const hemiGroundSum = [0, 0, 0];
    const directional: number[][] = [];
    const point: number[][] = [];
    const spot: number[][] = [];
    // K6.5b: the light id of each record (element 16), for the occluder maps
    const ids = new Map<number[], number | string>();
    const norm = (v: readonly number[]) => {
      const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
      return [v[0]! / l, v[1]! / l, v[2]! / l];
    };
    let hemiTotal = 0;
    let sky: number[] | null = null;
    for (const l of lights) {
      switch (l.kind) {
        case "ambient":
          for (let i = 0; i < 3; i++) ambient[i] = ambient[i]! + l.color[i]!;
          break;
        case "probe":
          for (let i = 0; i < 9; i++) for (let c = 0; c < 3; c++) probe[i * 4 + c] = probe[i * 4 + c]! + (l.sh[i * 3 + c] ?? 0);
          break;
        case "hemisphere":
          hemiTotal++;
          if (hemi.length < VDB_FRAME_MAX_HEMISPHERE_LIGHTS) hemi.push({ sky: l.sky, ground: l.ground, up: norm(l.up) });
          for (let i = 0; i < 3; i++) {
            hemiSkySum[i] = hemiSkySum[i]! + l.sky[i]!;
            hemiGroundSum[i] = hemiGroundSum[i]! + l.ground[i]!;
          }
          // his frameSkyDirection: the first hemisphere light's direction
          if (!sky && Math.hypot(l.up[0], l.up[1], l.up[2]) > 0) sky = norm(l.up);
          break;
        case "directional": {
          const d = norm(l.direction);
          const r = [-d[0]!, -d[1]!, -d[2]!, 1, ...l.color, 0, 0, 0, 0, 0, 0, 0, 0, 0];
          ids.set(r, l.id);
          directional.push(r);
          break;
        }
        case "point": {
          const r = [...l.position, 2, ...l.color, 0, 0, 0, 0, 0, l.distance, l.decay, 0, 0];
          ids.set(r, l.id);
          point.push(r);
          break;
        }
        case "spot": {
          const d = norm(l.direction);
          const r = [...l.position, 3, ...l.color, l.cosPenumbra, -d[0]!, -d[1]!, -d[2]!, l.cosAngle, l.distance, l.decay, 0, 0];
          ids.set(r, l.id);
          spot.push(r);
          break;
        }
      }
    }
    const sorted = [...directional, ...point, ...spot];
    if (sorted.length > VDB_FRAME_MAX_LIGHTS || hemiTotal > VDB_FRAME_MAX_HEMISPHERE_LIGHTS) {
      this.#warnOnce("lights", `too many lights: only ${VDB_FRAME_MAX_LIGHTS} lights and ${VDB_FRAME_MAX_HEMISPHERE_LIGHTS} hemisphere lights are used.`);
    }
    const used = sorted.slice(0, VDB_FRAME_MAX_LIGHTS);
    const skyDirection = sky ?? [0, 1, 0];
    // K6.5b: occluder slots (falloff.z = slot + 1), their layers, bias, flags and matrices
    const occluderOf: (VDBFrameOccluder | null)[] = used.map(() => null);
    const views: GPUTextureView[] = [];
    const info: number[] = [];
    const matrices = new Float32Array(VDB_FRAME_MAX_OCCLUDERS * VDB_FRAME_OCCLUDER_LAYERS * 16);
    for (const o of occluders) {
      const index = used.findIndex((r) => ids.get(r) === o.light);
      if (index < 0) {
        this.#warnOnce(`occluder-light-${String(o.light)}`, `the occluder map of light ${String(o.light)} has no directional, point or spot light of that id in this frame (or it is past the ${VDB_FRAME_MAX_LIGHTS} lights); it is ignored.`);
        continue;
      }
      if (occluderOf[index]) {
        this.#warnOnce("occluder-twice", `light ${String(o.light)} has two occluder maps; the first is used.`);
        continue;
      }
      if (views.length >= VDB_FRAME_MAX_OCCLUDERS) {
        this.#warnOnce("occluders", `more than ${VDB_FRAME_MAX_OCCLUDERS} occluder maps: only the first ${VDB_FRAME_MAX_OCCLUDERS} (in input order) shadow the volumes.`);
        break;
      }
      const layers = o.viewProjection.length;
      if (layers < 1 || layers > VDB_FRAME_OCCLUDER_LAYERS) {
        throw new VDBError("VDB_BAD_OPTION", `VDBFrameRenderer.update: occluder of light ${String(o.light)} has ${layers} view-projections; give 1–${VDB_FRAME_OCCLUDER_LAYERS} (one per layer of its depth view).`);
      }
      const slot = views.length;
      occluderOf[index] = o;
      used[index]![14] = slot + 1;
      views.push(o.depth);
      info.push(layers, o.bias ?? 0, o.reversedDepth ? 1 : 0, o.filter === "nearest" ? 1 : 0);
      o.viewProjection.forEach((m, l) => {
        for (let k = 0; k < 16; k++) matrices[(slot * VDB_FRAME_OCCLUDER_LAYERS + l) * 16 + k] = m[k]!;
      });
    }
    this.#occluderViews = views;
    // K6.Q1: the shadowed ambient's direction pairs, when there is distant light to shadow (ambient, a probe, hemisphere
    // lights; 'frame' lighting's 'outward' environment)
    const s = this.settings;
    const distant = ambient.some((v) => v !== 0) || probe.some((v) => v !== 0) || hemi.length > 0 || outwardEnvironment;
    const pairs = s.ambientOcclusion && distant ? Math.min(VDB_FRAME_AMBIENT_MAX_PAIRS, Math.max(1, Math.round(s.ambientOcclusionPairs))) : 0;
    const directions = pairs > 0 ? vdbFrameAmbientDirections(skyDirection, pairs, ringPhase) : [];
    const aoSteps = Math.max(0, Math.round(s.ambientOcclusionSteps));
    w.set(L, 0, "ambient", [ambient[0]!, ambient[1]!, ambient[2]!, hemi.length]);
    w.set(L, 0, "counts", [used.length, views.length, pairs, aoSteps]);
    if (pairs > 0) w.set(L, 0, "ambientDirections", directions.flatMap((d) => [...d, 0]));
    w.set(L, 0, "probe", probe);
    w.set(L, 0, "hemiSky", hemi.flatMap((h) => [...h.sky, 0]));
    w.set(L, 0, "hemiGround", hemi.flatMap((h) => [...h.ground, 0]));
    w.set(L, 0, "hemiUp", hemi.flatMap((h) => [...h.up, 0]));
    w.set(L, 0, "position", used.flatMap((r) => r.slice(0, 4)));
    w.set(L, 0, "color", used.flatMap((r) => r.slice(4, 8)));
    w.set(L, 0, "spot", used.flatMap((r) => r.slice(8, 12)));
    w.set(L, 0, "falloff", used.flatMap((r) => r.slice(12, 16)));
    w.set(L, 0, "sky", [...skyDirection, 0]);
    w.set(L, 0, "hemiSkySum", [...hemiSkySum, 0]);
    w.set(L, 0, "hemiGroundSum", [...hemiGroundSum, 0]);
    if (views.length > 0) {
      w.set(L, 0, "occluderInfo", info);
      w.set(L, 0, "occluderMatrix", matrices);
    }
    this.device.queue.writeBuffer(this.#lightsBuffer, 0, w.buffer);
    return { sorted: used, sky: skyDirection, occluderOf, ambient: [pairs, ...directions.flat()] };
  }

  /** The phase fields of a volume after its overrides (V65), or the renderer's own ('frame' lighting) for null. */
  #phaseOf(p: VDBFrameVolumeParams | null): VDBFramePhaseFields {
    const s = this.settings;
    return {
      phase: p?.phase ?? s.phase,
      phaseG: p?.phaseG ?? s.phaseG,
      phaseBackG: p?.phaseBackG ?? s.phaseBackG,
      phaseBackWeight: p?.phaseBackWeight ?? s.phaseBackWeight,
      phaseGExponent: p?.phaseGExponent ?? s.phaseGExponent,
    };
  }

  /** densityScale after normalizeDensity (his getDensityScale). */
  #densityScaleOf(p: VDBFrameVolumeParams, slot: VDBVolumeSlot): number {
    const mean = p.densityMean ?? slot.stats.meanNonZero;
    return p.normalizeDensity && mean > 0 ? p.densityScale / mean : p.densityScale;
  }

  #packVolumes(records: readonly VolumeRecord[], codes: readonly number[], spectral = false): void {
    const w = this.#volumesWriter;
    const V = VDB_FRAME_VOLUME;
    records.forEach((r, i) => {
      const base = i * VDB_FRAME_VOLUME_BYTES;
      const p = r.volume.params;
      const slot = r.volume.slot;
      const phaseFields = this.#phaseOf(p);
      const phase = vdbFramePhaseVector(phaseFields);
      const ms = vdbFrameMultipleScatteringValue(this.settings.multipleScattering === "octaves" ? false : this.settings.multipleScattering, phaseFields);
      const levels = slot.lodLevels;
      w.clear(base, VDB_FRAME_VOLUME_BYTES);
      w.set(V, base, "viewToLocal", r.viewToLocal);
      w.set(V, base, "localToView", r.localToView);
      w.set(V, base, "worldToLocal", r.worldToLocal);
      w.set(V, base, "rect", r.rect);
      w.set(V, base, "slices", [r.first, r.count, r.volume.slot.v, p.lights >>> 0]);
      w.set(V, base, "scatterOpacity", [...p.scatterColor, p.opacity]);
      w.set(V, base, "baseExtinction", [...p.baseColor, p.extinction]);
      w.set(V, base, "offsetWrap", [...p.offset3D, WRAP_CODE[p.wrap3D] ?? 0]);
      w.set(V, base, "phase", phase);
      w.set(V, base, "phaseSpectrumMs", [...vdbFramePhaseSpectrum(phaseFields), ms]);
      w.set(V, base, "density", [this.#densityScaleOf(p, slot), p.densityCutoff, vdbMat4MaxScale(p.matrixWorld), p.absorbance]);
      w.set(V, base, "emission", [p.emissiveIntensity, p.temperatureRange[0], p.temperatureRange[1], levels]);
      w.set(V, base, "march", [Math.max(0, Math.round(p.shadowSteps)), p.mipmaps && levels > 0 ? 1 : 0, p.hdda ? 1 : 0, codes[i] ?? 0]);
      // K6.4: each channel's extinction scale (his getFrameExtinctionSpectrum); grey kernels do not read it
      w.set(V, base, "extinctionSpectrum", spectral ? [...vdbFrameExtinctionSpectrum([...p.extinctionScale], p.angstromExponent), 0] : [1, 1, 1, 0]);
    });
    this.device.queue.writeBuffer(this.#volumesBuffer, 0, w.buffer, 0, Math.max(1, records.length) * VDB_FRAME_VOLUME_BYTES);
  }

  /** One record per source group: its gather range and role among the gather groups, its casters and their role. */
  #packGroups(groups: readonly SourceGroup[], gatherGroups: readonly SourceGroup[], casterGroups: readonly SourceGroup[]): void {
    const bytes = Math.max(4, groups.length) * VDB_FRAME_GROUP_STRIDE;
    if (this.#groupsBuffer.size < bytes) {
      this.#retire.retire(this.#groupsBuffer);
      this.#groupsBuffer = this.device.createBuffer({ label: `${this.#label} groups`, size: bytes, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
      this.#groupsWriter = new VDBUniformWriter(bytes);
    }
    const w = this.#groupsWriter;
    const role = (list: readonly SourceGroup[], g: SourceGroup) => {
      const i = list.indexOf(g);
      return i < 0 ? 0 : (i === 0 ? 1 : 0) | (i === list.length - 1 ? 2 : 0);
    };
    groups.forEach((g, i) => {
      w.clear(i * VDB_FRAME_GROUP_STRIDE, VDB_FRAME_GROUP_STRIDE);
      w.set(VDB_FRAME_GROUP, i * VDB_FRAME_GROUP_STRIDE, "range", [g.first, g.visible, role(gatherGroups, g), gatherGroups.indexOf(g)]);
      w.set(VDB_FRAME_GROUP, i * VDB_FRAME_GROUP_STRIDE, "casters", [g.first + g.casterStart, g.casters, role(casterGroups, g), casterGroups.indexOf(g)]);
    });
    this.device.queue.writeBuffer(this.#groupsBuffer, 0, w.buffer, 0, groups.length * VDB_FRAME_GROUP_STRIDE);
  }

  // ---------------------------------------------------------------- render

  render(encoder: GPUCommandEncoder, targets: VDBFrameTargets): void {
    this.#check();
    const key = this.#compositeKey();
    if (!key) throw new VDBError("VDB_BAD_OPTION", "VDBFrameRenderer.render: create the renderer with options.composite ({ colorFormat, depthFormat }).");
    const froxels = this.outputs.froxels;
    if (!this.#recorded || !froxels) return;
    const srgb = key.srgb && (targets.outputColorSpace ?? "srgb") === "srgb";
    // K6.4: the spectral composite reads the alpha prefix of this update's textures
    const spectral = !!froxels.integratedAlpha;
    const finalKey: VDBFrameCompositeKey = { ...key, srgb, spectral };
    if (srgb && !targets.sceneColor) {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameRenderer.render: sRGB blending reads the scene colour: pass targets.sceneColor (a copy of the colour target), or set settings.srgbBlending = false.");
    }
    if (spectral && !targets.sceneColor) {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameRenderer.render: spectral media composite over the scene colour themselves (three transmittances): pass targets.sceneColor (a copy of the colour target).");
    }
    const [sw, sh] = targets.size;
    // Only the screen-size fields move here; the rest is the update's.
    const fw = this.#frameWriter;
    fw.set(VDB_FRAME_UNIFORMS, 0, "screen", [sw, sh, (targets.clampColors ?? true) ? 1 : 0, 0]);
    const screen = VDB_FRAME_UNIFORMS.fields.get("screen")!;
    this.device.queue.writeBuffer(this.#frameBuffer, screen.offset, fw.buffer, screen.offset, 16);
    const built = renderPipeline(this.device, vdbFrameCompositeSpec(finalKey));
    const [x0, y0, x1, y1] = froxels.layout.screenRect;
    // clip → pixels (y down)
    const px0 = Math.max(0, Math.floor((x0 * 0.5 + 0.5) * sw));
    const px1 = Math.min(sw, Math.ceil((x1 * 0.5 + 0.5) * sw));
    const py0 = Math.max(0, Math.floor((1 - (y1 * 0.5 + 0.5)) * sh));
    const py1 = Math.min(sh, Math.ceil((1 - (y0 * 0.5 + 0.5)) * sh));
    if (px1 <= px0 || py1 <= py0) return;
    const group1 = this.device.createBindGroup({
      label: `${this.#label} composite group 1`,
      layout: built.layouts[1]!,
      entries: [
        { binding: 0, resource: froxels.integrated.createView({ dimension: "3d" }) },
        { binding: 1, resource: this.#linear },
        { binding: 2, resource: targets.depth },
        ...(srgb || spectral ? [{ binding: 3, resource: targets.sceneColor! }] : []),
        ...(spectral ? [{ binding: VDB_FRAME_COMPOSITE_ALPHA_BINDING, resource: froxels.integratedAlpha!.createView({ dimension: "3d" }) }] : []),
      ],
    });
    const pass = encoder.beginRenderPass({ label: `${this.#label} composite`, colorAttachments: [{ view: targets.color, loadOp: "load", storeOp: "store" }] });
    pass.setPipeline(built.pipeline);
    pass.setBindGroup(0, this.#group0Bind());
    pass.setBindGroup(1, group1);
    pass.setScissorRect(px0, py0, px1 - px0, py1 - py0);
    pass.draw(3);
    pass.end();
  }

  // ---------------------------------------------------------------- lifecycle

  #warnOnce(key: string, message: string): void {
    if (this.#warned.has(key)) return;
    this.#warned.add(key);
    console.warn(`@pmndrs/vdb frame renderer: ${message}`);
  }

  #check(): void {
    if (this.#disposed) throw new VDBError("VDB_DISPOSED", "VDBFrameRenderer: the renderer was disposed; create a new one.");
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const t = this.#textures;
    if (t) {
      t.scatter.destroy();
      t.integrated.destroy();
      t.history?.forEach((h) => h.destroy());
      t.partials?.destroy();
      t.spectral?.scatterAlpha.destroy();
      t.spectral?.integratedAlpha.destroy();
    }
    this.#textures = null;
    const g = this.#grid;
    if (g) {
      g.shadow.destroy();
      g.transmittance.destroy();
      g.partials?.destroy();
      g.environment.destroy();
      g.absorption?.destroy();
    }
    this.#grid = null;
    this.#shadows.dispose();
    this.#probes.dispose();
    this.#colors.dispose();
    this.outputs.emissionProbes = null;
    this.#env.dispose();
    this.#scattering.dispose();
    this.#mie.destroy();
    this.#occluderPlaceholder?.texture.destroy();
    this.#occluderPlaceholder = null;
    this.#occluderViews = [];
    this.#retire.drain();
    this.#frameBuffer.destroy();
    this.#lightsBuffer.destroy();
    this.#volumesBuffer.destroy();
    this.#groupsBuffer.destroy();
    this.outputs.froxels = null;
    this.#volumes.length = 0;
  }

  /** @internal The textures (harness probes and tests). */
  get debugTextures(): Readonly<Textures> | null {
    return this.#textures;
  }

  /** @internal The emission textures (harness probes): direction and emitter textures, slots, the list buffer. */
  get debugEmission(): { directions: GPUTexture | null; emitters: GPUTexture | null; probes: GPUTexture | null; slots: number } {
    const P = this.#probes;
    return { directions: P.directionTexture, emitters: P.emitterTexture, probes: P.probeTexture, slots: P.slots };
  }

  /** @internal The environment state (harness probes and tests): the buffer, the maps and the kernel counts. */
  /** @internal K6.3 (harness probes): the 'mlp' feature mips, the multi-scale chain and the network in use. */
  get debugScattering(): VDBFrameScattering["debug"] {
    return this.#scattering.debug;
  }

  /** @internal The environment state (harness probes and tests). */
  get debugEnvironment(): VDBFrameEnvironmentState {
    return this.#env;
  }

  /** @internal The 'frame' lighting grid textures and the live cascade layout (harness probes and tests). */
  get debugGrid(): { textures: Readonly<GridTextures>; count: number; gridMin: number[][]; gridSize: number[][]; gridVoxel: number[][] } | null {
    const g = this.#grid;
    if (!g) return null;
    const count = this.#frameWriter.f32[VDB_FRAME_UNIFORMS.fields.get("gridLayout")!.offset / 4]!;
    const v = (a: { x: number; y: number; z: number }[]) => a.slice(0, count).map((p) => [p.x, p.y, p.z]);
    return { textures: g, count, gridMin: v(this.#gridMin), gridSize: v(this.#gridSize), gridVoxel: v(this.#gridVoxel) };
  }
}

/**
 * Creates the core frame renderer on the host's device (design §2.2). It makes its uniform buffers now and
 * its froxel textures at the first `update()` with volumes in view.
 *
 * @param device The host's `GPUDevice`.
 * @param options {@link VDBFrameRendererOptions}.
 */
export function vdbCreateFrameRenderer(device: GPUDevice, options: VDBFrameRendererOptions = {}): VDBFrameRenderer {
  if (!device || typeof (device as { createTexture?: unknown }).createTexture !== "function") {
    throw new VDBError("VDB_UNSUPPORTED_DEVICE", "vdbCreateFrameRenderer: pass the host's GPUDevice.");
  }
  return new FrameRenderer(device, options);
}

/** @internal The froxel textures of a renderer from {@link vdbCreateFrameRenderer} (harness probes). */
export function vdbFrameRendererTextures(r: VDBFrameRenderer): {
  scatter: GPUTexture;
  integrated: GPUTexture;
  history: readonly GPUTexture[] | null;
  partials: GPUTexture | null;
  size: readonly number[];
  spectral: { scatterAlpha: GPUTexture; integratedAlpha: GPUTexture; alphaFormat: "rg16float" | "rgba16float" } | null;
} | null {
  return (r as unknown as { debugTextures: Textures | null }).debugTextures;
}

/** @internal The emission textures of a renderer (harness probes). */
export function vdbFrameRendererEmission(r: VDBFrameRenderer): { directions: GPUTexture | null; emitters: GPUTexture | null; probes: GPUTexture | null; slots: number } {
  return (r as unknown as { debugEmission: ReturnType<typeof vdbFrameRendererEmission> }).debugEmission;
}

/** @internal The 'frame' lighting grid of a renderer (harness probes): stacks, their layouts and the live cascades. */
export function vdbFrameRendererGrid(r: VDBFrameRenderer): {
  textures: { resolution: number; cascades: number; shadow: GPUTexture; shadowLayout: VDBFrameStackLayout; transmittance: GPUTexture; transmittanceLayout: VDBFrameStackLayout; partials: GPUTexture | null; environment: GPUTexture; absorption: GPUTexture | null; groups: number };
  count: number;
  gridMin: number[][];
  gridSize: number[][];
  gridVoxel: number[][];
} | null {
  return (r as unknown as { debugGrid: ReturnType<typeof vdbFrameRendererGrid> }).debugGrid;
}

/** @internal The environment of a renderer (harness probes): its buffer, maps and kernel counts (K6.2b). */
export function vdbFrameRendererEnvironment(r: VDBFrameRenderer): {
  buffer: GPUBuffer;
  maps: { work: GPUTexture; map: GPUTexture } | null;
  runs: { prefilter: number; blur: number; reduce: number };
  version: number;
  present: boolean;
} {
  return (r as unknown as { debugEnvironment: VDBFrameEnvironmentState }).debugEnvironment;
}

/** @internal K6.3 (harness probes): the 'mlp' feature mips, the 'multi-scale' chain and the network in use. */
export function vdbFrameRendererScattering(r: VDBFrameRenderer): VDBFrameScattering["debug"] {
  return (r as unknown as { debugScattering: VDBFrameScattering["debug"] }).debugScattering;
}
