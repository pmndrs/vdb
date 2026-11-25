// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameEmissionLight.js L1–287 (FrameEmissionLightBase: props, probe cascades);
//   src/three-openvdb/frame/webgpu/FrameEmissionLight.js L59–658 (FrameEmissionLightNode: the probe lookup with the
//   cascade blend and the falloff past the last cascade; FrameEmissionLight: props, the node registration).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; K6.8: the emitters and probes are the core frame renderer's (its emission kernels, a stacked 3D
// texture read with hardware trilinear); this light carries the settings and lights meshes; K6.8b: the lookup is the
// core's own WGSL (`vdbEmissionIrradiance` of `vdbEmissionProbeWgsl()`) through a `wgslFn`, the probe grid read from the
// core's buffer through a storage stand-in, instead of a TSL port; every three r187 internal goes through
// `three-internals.ts` or the stand-in self-tests (VDB_ENGINE_MISMATCH on another three). Divergences: DIVERGES.md (K3.6,
// K6.5, K6.8b rows).

/**
 * K3.6 / K6.5: light from the emissive grids of every visible `VDBFrameVolume` (his `FrameEmissionLight`). Add one to
 * the scene; three's lit node materials (`MeshStandardMaterial`, `MeshLambertMaterial`, …) receive it as irradiance,
 * like a light probe. The core frame renderer of the scene's volumes draws the emitters and L1 irradiance probes (up to
 * 4 cascades of ≤ 32³) when a volume or the cascades changed; a lit pixel reads 6 texels (12 near a cascade side).
 *
 *   const light = new VDBFrameEmissionLight({ intensity: 25, resolution: 32 });
 *   scene.add(light);   // volumes need an emission channel; colour: VDBFrameVolume `emissionColor`
 *
 * K6.8b: a lit pixel runs the core's `vdbEmissionIrradiance(probes, smp, grid, world, normal)` (`vdbEmissionProbeWgsl()`,
 * twin `vdbFrameEmissionLookup` + `vdbFrameEmissionIrradiance`) through a `wgslFn`: `probes` is the core's probe stack
 * (`outputs.emissionProbes.texture`, a 3D texture stand-in) and `grid` its `VdbEmissionGrid` buffer
 * (`outputs.emissionProbes.grid`, bound through a storage stand-in and unpacked from `array<vec4u>`; the core rewrites it
 * each update). The texture and storage nodes are made once; their `value` follows the core (placeholders while it has no
 * probes: a 1³ texture and a zero grid, which lights nothing). three r187 re-reads a plain material's texture and storage
 * bindings only on a full refresh, so a new core object (a new probe resolution, a released stack, another frame
 * renderer) also rebuilds the receivers: the light keeps `castShadow` on (it casts no shadow) and its node bumps
 * `version`, which three's lights cache key holds for a `castShadow` light. They rebuild at their next render; the core
 * keeps the old stack alive two more updates (the K6.10 finding-8 class).
 */

import * as THREE from "three";
import { AnalyticLightNode, type StorageBufferAttribute } from "three/webgpu";
import * as TSL from "three/tsl";
import { vdbThreeStorageStandIn, type VDBTslNode } from "../grid-node.js";
import { getDevice } from "../device.js";
import { frameEmissionMaxCascades as maxCascades } from "../../tools/frame/emission.js";
import { vdbFrameCheckLightReset, vdbFramePrecompiling, vdbFrameRegisterLightNode, vdbFrameRenderUpdateType } from "./three-internals.js";
import { VDBCoreTextureBinding, vdbCoreTextureSelfTest } from "./core-textures.js";
import { VDBFrameStructLoaderNode, vdbFrameStructWords, type VDBFrameStorageNode } from "./core-structs.js";
import { VDB_EMISSION_GRID, vdbEmissionProbeWgsl } from "../../core/frame/wgsl/emission.js";
import type { VDBFrameEmissionProbes } from "../../core/frame/types.js";
import type { VDBFrameRenderer } from "./renderer.js";
import type { VDBFrameVolume } from "./volume.js";

const { code, normalWorld, positionWorld, sampler, storage, texture3D, wgslFn } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;

/** `vec4u` words of the grid struct. */
const GRID_WORDS = vdbFrameStructWords(VDB_EMISSION_GRID);

/** The core's receiver function and the grid struct, one code node shared by every receiver (once per shader). */
let receiverCode: N | null = null;
const receiverCodeNode = (): N => (receiverCode ??= code(vdbEmissionProbeWgsl()));

/** Light nodes made so far: names each one's storage binding and WGSL functions. */
let names = 0;

/** What the receivers bind while the core has no probes: a 1³ stack and a zero grid (`info.x` 0: no light, no sample). */
interface Placeholders {
  texture: GPUTexture;
  grid: GPUBuffer;
}

/** A light node's three nodes; made at its first receiver build, their values follow the core. */
interface ReceiverNodes {
  renderer: object;
  placeholders: Placeholders;
  probes: N;
  grid: VDBFrameStorageNode;
  /** The grid buffer bound now (the core's or the placeholder). */
  gridBuffer: GPUBuffer;
  fn: (inputs: Record<string, N>) => N;
}

/**
 * The irradiance at a lit point from the core's probe cascades (`vdbEmissionIrradiance`: the finest cascade that holds
 * it, blended into the next one near its sides; past the last cascade its clamped edge probes fall off with distance
 * squared), times the light's colour × intensity.
 */
export class VDBFrameEmissionLightNode extends AnalyticLightNode<THREE.Light> {
  static get type(): string {
    return "VDBFrameEmissionLightNode";
  }

  /** Unique per light node: names the storage binding and the WGSL functions. */
  readonly k = names++;
  #nodes: ReceiverNodes | null = null;
  // NOTE Replaced grid stand-ins. update() runs mid-render, so the next update() disposes them.
  #retired: StorageBufferAttribute[] = [];

  constructor(light: VDBFrameEmissionLight | null = null) {
    super(light as never);
    // NOTE The core update runs before the first lit mesh of each render, as shadow maps
    const render = vdbFrameRenderUpdateType();
    (this as unknown as { updateType: string; updateBeforeType: string }).updateType = render;
    (this as unknown as { updateType: string; updateBeforeType: string }).updateBeforeType = render;
  }

  private get emissionLight(): VDBFrameEmissionLight {
    return (this as unknown as { light: VDBFrameEmissionLight }).light;
  }

  /** The probe stack and grid bound now (null before the first receiver build); tests and harnesses. @internal */
  get bound(): { texture: GPUTexture | null; grid: GPUBuffer; placeholder: boolean } | null {
    const nodes = this.#nodes;
    if (!nodes) return null;
    return { texture: this.emissionLight.coreBinding.gpu, grid: nodes.gridBuffer, placeholder: nodes.gridBuffer === nodes.placeholders.grid };
  }

  override updateBefore(frame: never): boolean | undefined {
    const { renderer, scene, camera } = frame as { renderer: object; scene: THREE.Scene; camera: THREE.Camera };
    // NOTE No update while three precompiles materials (compileAsync), as shadow maps
    if (vdbFramePrecompiling(renderer)) return undefined;
    this.emissionLight.updateProbes(renderer, scene, camera);
    return undefined;
  }

  override update(frame: never): boolean | undefined {
    super.update(frame);
    this.#retired.forEach((a) => a.dispose());
    this.#retired.length = 0;
    this.#bind(this.emissionLight.coreProbes);
    return undefined;
  }

  /**
   * Points the nodes at the core's probes (null: the placeholders). A new core object gets a new stand-in as the node's
   * `value` and bumps this node's `version`, so the receivers rebuild at their next render (see the file comment).
   */
  #bind(probes: VDBFrameEmissionProbes | null): void {
    const nodes = this.#nodes;
    if (!nodes) return;
    const p = nodes.placeholders;
    let changed = false;
    const binding = this.emissionLight.coreBinding;
    if (binding.set(probes?.texture ?? p.texture)) {
      nodes.probes.value = binding.texture;
      changed = true;
    }
    const grid = probes?.grid ?? p.grid;
    if (grid !== nodes.gridBuffer) {
      this.#retired.push(nodes.grid.value as StorageBufferAttribute);
      nodes.grid.value = vdbThreeStorageStandIn(nodes.renderer, grid, `vdbEmissionGrid${this.k}`);
      nodes.gridBuffer = grid;
      changed = true;
    }
    if (changed) (this as unknown as { needsUpdate: boolean }).needsUpdate = true;
  }

  #nodesOf(renderer: object): ReceiverNodes {
    if (this.#nodes) {
      if (this.#nodes.renderer === renderer) return this.#nodes;
      throw new Error("VDBFrameEmissionLight: its receivers were built by a second renderer; a light (and its scene) belongs to one renderer");
    }
    // NOTE The rebuild on a new core object needs three's lights cache key to follow this node's version (internal 1)
    vdbFrameCheckLightReset();
    const device = getDevice(renderer as never);
    vdbCoreTextureSelfTest(renderer as never, device);
    const placeholders: Placeholders = {
      texture: device.createTexture({ label: "@pmndrs/vdb/three emission probes placeholder", size: [1, 1, 1], dimension: "3d", format: "rgba16float", usage: 0x04 /* TEXTURE_BINDING */ }),
      // A zero grid: info.x (cascades) 0, so vdbEmissionIrradiance returns 0 without sampling
      grid: device.createBuffer({ label: "@pmndrs/vdb/three emission grid placeholder", size: VDB_EMISSION_GRID.size, usage: 0x80 /* STORAGE */ }),
    };
    const light = this.emissionLight;
    const probes = light.coreProbes;
    light.coreBinding.set(probes?.texture ?? placeholders.texture);
    const gridBuffer = probes?.grid ?? placeholders.grid;
    const grid = storage(vdbThreeStorageStandIn(renderer, gridBuffer, `vdbEmissionGrid${this.k}`), "uvec4", GRID_WORDS).toReadOnly() as VDBFrameStorageNode;
    grid.setName(`vdbEmissionGrid${this.k}`);
    const loader = new VDBFrameStructLoaderNode(grid, VDB_EMISSION_GRID, `vdbFrameEmissionGrid${this.k}`);
    const name = `vdbFrameEmissionReceiver${this.k}`;
    const source = `fn ${name}(probes: texture_3d<f32>, smp: sampler, world: vec3f, normal: vec3f) -> vec3f {\n  return vdbEmissionIrradiance(probes, smp, ${loader.fnName}(), world, normal);\n}`;
    const fn = wgslFn(source, [receiverCodeNode(), loader]) as (inputs: Record<string, N>) => N;
    this.#nodes = { renderer, placeholders, probes: texture3D(light.coreBinding.texture), grid, gridBuffer, fn };
    return this.#nodes;
  }

  override setup(builder: never): null {
    const b = builder as unknown as { renderer: object; context: { irradiance: N } };
    const nodes = this.#nodesOf(b.renderer);
    const irradiance = nodes.fn({ probes: nodes.probes, smp: sampler(nodes.probes), world: positionWorld, normal: normalWorld });
    const colorNode = (this as unknown as { colorNode: N }).colorNode;
    b.context.irradiance.addAssign(colorNode.mul(irradiance));
    return null;
  }

  override dispose(): void {
    const nodes = this.#nodes;
    if (nodes) {
      this.#retired.push(nodes.grid.value as StorageBufferAttribute);
      const p = nodes.placeholders;
      // NOTE A bind group of this render may still name them: destroy after the task that submits it
      setTimeout(() => {
        p.texture.destroy();
        p.grid.destroy();
      }, 0);
      this.#nodes = null;
    }
    this.#retired.forEach((a) => a.dispose());
    this.#retired.length = 0;
    super.dispose();
  }
}

/** Props of {@link VDBFrameEmissionLight} (his `FrameEmissionLight` props; same names and defaults). */
export interface VDBFrameEmissionLightProps {
  /** Light colour (default white). */
  color?: THREE.ColorRepresentation;
  /** Light intensity, a multiplier of the volumes' emitted light (default 1). */
  intensity?: number;
  /** Probes on each axis of each cascade, 2–32 (default 16). */
  resolution?: number;
  /** Size of the last cascade as a multiple of the box of all emitting volumes; past it light falls off with distance² (default 3). */
  range?: number;
  /** 1: one grid for all volumes. Up to 4: finer grids around the camera (default 1). */
  cascades?: number;
  /** The finer cascades cover this distance from the camera (default 512). */
  distance?: number;
  /** Steps of each ray in a volume (default 24). */
  steps?: number;
}

/** One registration of the light node class (internal 4), checked at the first construction. */
let registered = false;

/**
 * Light that lights lit meshes with the emission of emissive `VDBFrameVolume`s (his `FrameEmissionLight`): the core
 * frame renderer of the scene's volumes draws the emitters and the probe cascades; this light's node reads them. Add
 * it to the scene.
 */
export class VDBFrameEmissionLight extends THREE.Light {
  /** Type flag. */
  readonly isVDBFrameEmissionLight = true as const;

  /** Probes on each axis of each cascade, 2 to 32. */
  resolution: number;
  /** Size of the last cascade as a multiple of the box of all emitting volumes. */
  range: number;
  /** Probe grid cascades, 1 to 4 (1: one grid for all volumes). */
  cascades: number;
  /** Distance from the camera that the finer cascades cover. */
  distance: number;
  /** Steps of each ray in a volume (rounded, at least 1). */
  steps: number;

  /** Minimum corner of each cascade in world space (valid for the first `cascadeCount`). */
  readonly cascadeMin = Array.from({ length: maxCascades }, () => new THREE.Vector3());
  /** Size of each cascade in world space (valid for the first `cascadeCount`). */
  readonly cascadeSize = Array.from({ length: maxCascades }, () => new THREE.Vector3(1.0, 1.0, 1.0));
  /** Cascades in use after the last probe update. */
  cascadeCount = 0;
  /** Probes on each axis of each cascade after the last update. */
  probeResolution = 2;
  /** World centre of the probe grid after the last update. */
  readonly probeCenter = new THREE.Vector3();
  /** Emissive volumes in the last update. */
  emitterCount = 0;
  /** Volumes drawn in the last probe update and whether the probes were drawn (not his; tests and timing). */
  readonly lastUpdate = { directions: 0, probes: false };
  /** The core's probes after the last update (null: none). */
  coreProbes: VDBFrameEmissionProbes | null = null;
  /** @internal The stand-in of the core probe stack. */
  readonly coreBinding = new VDBCoreTextureBinding("3d");

  /**
   * Makes the light with its probe grid settings.
   * @param props {@link VDBFrameEmissionLightProps}; all fields optional.
   */
  constructor(props: VDBFrameEmissionLightProps = {}) {
    const { color = 0xffffff, intensity = 1.0, resolution = 16, range = 3.0, cascades = 1, distance = 512, steps = 24 } = props;
    super(color, intensity);
    if (!registered) {
      vdbFrameRegisterLightNode(VDBFrameEmissionLight as unknown as typeof THREE.Light, VDBFrameEmissionLightNode);
      registered = true;
    }
    (this as { type: string }).type = "VDBFrameEmissionLight";
    // K6.8b: always on, and the light casts no shadow (its node draws none). three r187's lights cache key holds the node
    // version of a castShadow light only; the node bumps it to rebuild its receivers when the core makes new probes.
    Object.defineProperty(this, "castShadow", { get: () => true, set: () => {}, configurable: true, enumerable: true });
    this.resolution = resolution;
    this.range = range;
    this.cascades = cascades;
    this.distance = distance;
    this.steps = steps;
  }

  /** Steps of each ray, rounded and at least 1. */
  getSteps(): number {
    return Math.max(1, Math.round(this.steps));
  }

  /**
   * The core frame of this render (the frame renderer of the scene's first visible volume) draws the emitters and the
   * probes; this light copies the probe grid for its node. The light node calls it before the lit meshes of each render.
   */
  updateProbes(renderer: object, scene: THREE.Object3D, viewCamera: THREE.Camera): void {
    this.lastUpdate.directions = 0;
    this.lastUpdate.probes = false;
    let frameRenderer: VDBFrameRenderer | null = null;
    scene.traverseVisible((object) => {
      const v = object as VDBFrameVolume;
      if (!frameRenderer && v.isVDBFrameVolume) frameRenderer = v.frameRenderer;
    });
    const fr = frameRenderer as VDBFrameRenderer | null;
    fr?.coreFrame(renderer as never, scene as never, viewCamera);
    const probes = fr?.core?.outputs.emissionProbes ?? null;
    this.coreProbes = probes;
    if (!probes) {
      this.emitterCount = 0;
      this.cascadeCount = 0;
      return;
    }
    const { state } = probes;
    this.emitterCount = state.emitters;
    this.cascadeCount = state.count;
    this.probeResolution = state.resolution;
    this.probeCenter.fromArray(state.center as unknown as number[]);
    for (let i = 0; i < state.count; i++) {
      this.cascadeMin[i]!.fromArray(state.min[i] as unknown as number[]);
      this.cascadeSize[i]!.fromArray(state.size[i] as unknown as number[]);
    }
    this.lastUpdate.directions = probes.lastUpdate.directions;
    this.lastUpdate.probes = probes.lastUpdate.probes;
  }

  override dispose(): void {
    super.dispose();
    this.coreBinding.dispose();
  }
}
