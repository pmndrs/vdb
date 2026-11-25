// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameVolumeShadows.js L34–1027 (the light shadow reset, FrameVolumeShadowNode, the
//   filter install, dispose).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Past the pin (K7.1): his 62a6e3e (lookups from the drawn state) and c13f2d5 (the budget counts only lights of the
// rendered scene; the core's maps).
// Changes: TypeScript; K6.8: the maps are the core's (`VDBVolumeShadowMap`, drawn by its kernels), three only receives
// them; K6.8a: the receivers call the core's WGSL (`vdbVolumeShadowWgsl`) with the core's view buffer bound through a
// storage stand-in, instead of a TSL port of the lookup; every three r187 internal goes through `three-internals.ts`
// or the `VDBGridNode` attribute self-test, which throw VDB_ENGINE_MISMATCH. Divergences: DIVERGES.md (K3.6, K6.5 rows).

/**
 * K3.6 / K6.5: shadows of `VDBFrameVolume`s on meshes (`frameRenderer.settings.castShadows`, his `FrameVolumeShadows`).
 * Each shadow-casting directional, spot or point light gets a Beer shadow map from the core frame renderer (its
 * `outputs.volumeShadows`); the light's `shadow.filterNode` multiplies three's shadow by the volume transmittance at
 * the receiver.
 *
 * K6.8a: the receiver is the core's own WGSL (`vdbVolumeShadow`, `…Rgb`, `…Cube`, `…CubeRgb` of `vdbVolumeShadowWgsl()`)
 * through a `wgslFn`. Its inputs are the map texture (a 2d-array: directional cascades or a spot map; a point map through a
 * cube view), the spectral rgb twin, and the map's `view` buffer, the state the map was drawn with (his 62a6e3e), which
 * the core rewrites each update: three binds that buffer itself through a storage stand-in (the `VDBGridNode` pool-buffer
 * pattern), read as `array<vec4u>` and unpacked into `VdbVolumeShadowView`. Each light's texture and storage nodes are
 * made once and only their `value` follows the core (a new stand-in when the core made a new texture or buffer), so a
 * receiver built at any time binds the core's current objects; while a light has no core map they bind a placeholder
 * (a zero view: transmittance 1), never a texture or buffer the core retired (K6.10 finding 8).
 *
 * three: receivers need `receiveShadow`, the light `castShadow`, and `renderer.shadowMap.enabled`. The core update runs
 * before the first receiver of each render (the node's `updateBefore`), so the maps are this render's.
 */

import * as THREE from "three";
import { Node, type StorageBufferAttribute, type WebGPURenderer } from "three/webgpu";
import * as TSL from "three/tsl";
import { vdbThreeStorageStandIn, type VDBTslNode } from "../grid-node.js";
import { getDevice } from "../device.js";
import { frameShadowCoarseMapSize } from "../../tools/frame/shadows.js";
import { vdbFrameCheckFilterInputs, vdbFrameCheckLightReset, vdbFrameRenderUpdateType, vdbFrameShadowFilters, vdbFrameSharedLightNode } from "./three-internals.js";
import type { VDBFrameRenderer } from "./renderer.js";
import { VDBCoreTextureBinding } from "./core-textures.js";
import { VDBFrameStructLoaderNode, vdbFrameStructLoaderWgsl, vdbFrameStructWords, type VDBFrameStorageNode } from "./core-structs.js";
import { VDB_VOLUME_SHADOW_VIEW, vdbVolumeShadowWgsl } from "../../core/frame/wgsl/shadows.js";
import type { VDBVolumeShadowMap } from "../../core/frame/types.js";

const { code, cubeTexture, dot, positionView, positionWorld, sampler, storage, texture, vec3, wgslFn } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;
type RendererLike = WebGPURenderer & { isWebGPURenderer?: boolean };
type ShadowLight = THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight;
type StorageNode = VDBFrameStorageNode;

const isShadowLight = (light: THREE.Object3D): light is ShadowLight => {
  const l = light as THREE.Light & { isDirectionalLight?: boolean; isSpotLight?: boolean; isPointLight?: boolean };
  return !!(l.isDirectionalLight || l.isSpotLight || l.isPointLight) && l.castShadow;
};

// NOTE The object is in a scene. A removed light is not.
const isInScene = (object: THREE.Object3D): boolean => {
  let root = object;
  while (root.parent) root = root.parent;
  return (root as THREE.Scene).isScene === true;
};

// NOTE A light shadow node is built one time; a new filterNode has no effect until it is built again.
// disposeShadow of the shared light node makes the receivers build a new shadow node (three-internals.ts, internal 1).
// Returns the old shadow node, or null when there is none. Receivers can use it until the end of this render.
const resetLightShadow = (light: ShadowLight): { dispose(): void } | null => {
  vdbFrameCheckLightReset();
  const lightNode = vdbFrameSharedLightNode(light) as { shadowNode: { dispose(): void } | null; disposeShadow(): void } | null;
  if (!lightNode || !lightNode.shadowNode) return null;
  const { shadowNode } = lightNode;
  // NOTE disposeShadow without the GPU dispose
  lightNode.shadowNode = { dispose() {} };
  lightNode.disposeShadow();
  return shadowNode;
};

/** `vec4u` words of the view struct (every field is 16-byte aligned). */
const VIEW_WORDS = vdbFrameStructWords(VDB_VOLUME_SHADOW_VIEW);

/**
 * WGSL that unpacks the view buffer (bound as `array<vec4u>` at `ref`) into the core's `VdbVolumeShadowView`, generated
 * from the struct's layout, so a field the core adds is unpacked too (a type this cannot unpack throws).
 */
export function vdbFrameShadowViewLoaderWgsl(fnName: string, ref: string): string {
  return vdbFrameStructLoaderWgsl(VDB_VOLUME_SHADOW_VIEW, fnName, ref);
}

/** The core's receiver functions and the view struct, one code node shared by every receiver (once per shader). */
let receiverCode: N | null = null;
const receiverCodeNode = (): N => (receiverCode ??= code(vdbVolumeShadowWgsl()));

/** The texture and buffer nodes of one light's receivers; made once, their values follow the core. */
interface EntryNodes {
  /** Unique per receivers object: names the storage binding and the WGSL functions. */
  k: number;
  map: N;
  rgb: N;
  view: StorageNode;
  loader: VDBFrameStructLoaderNode;
  fns: Map<string, (inputs: Record<string, N>) => N>;
}

/** What a light without a core map binds: a 1² 2d-array, a 1² cube (6 layers) and a zero view (enabled 0). */
interface Placeholders {
  device: GPUDevice;
  array: GPUTexture;
  cube: GPUTexture;
  view: GPUBuffer;
}

/** One shadow light's core map, as its receivers bind it. */
export interface VDBFrameShadowEntry {
  light: ShadowLight;
  isPoint: boolean;
  /** The map's size (texels per side) of the last core map; the settings' size before one. */
  size: number;
  /** The core map the receivers bind now (null: none yet, or the light has none this frame; they bind placeholders). */
  coreMap: VDBVolumeShadowMap | null;
  /** The core update count at the last draw of the map's first piece (his `lastDraw`; tests and the stale harness). */
  lastDraw: number;
  userFilter: ((inputs: unknown) => N) | null;
  /** The map texture bound now (the core's, or the placeholder). */
  texture: GPUTexture | null;
  binding: VDBCoreTextureBinding;
  /** K6.4 spectral: the optical depth rgb twin bound now (null: a grey map; the node binds the placeholder). */
  rgbTexture: GPUTexture | null;
  rgbBinding: VDBCoreTextureBinding;
  /** The view buffer bound now (the core's, or the placeholder). */
  viewBuffer: GPUBuffer | null;
  nodes: EntryNodes | null;
}

/**
 * Volume transmittance at a receiver from the light's core Beer shadow map: the core's `vdbVolumeShadow` /
 * `vdbVolumeShadowCube` (K6.4 spectral maps: `…Rgb` / `…CubeRgb`, the channel mean in a ShadowMaterial, his rule) through
 * a `wgslFn`, with the view the map was drawn with (twin `vdbFrameVolumeShadowLookup`). `updateBefore` runs the core
 * update before the first receiver of each render.
 */
export class VDBFrameVolumeShadowNode extends Node {
  static get type(): string {
    return "VDBFrameVolumeShadowNode";
  }

  readonly shadows: VDBFrameVolumeShadowReceivers;
  readonly entry: VDBFrameShadowEntry;

  constructor(shadows: VDBFrameVolumeShadowReceivers, entry: VDBFrameShadowEntry) {
    super("float");
    this.shadows = shadows;
    this.entry = entry;
    (this as unknown as { updateBeforeType: string }).updateBeforeType = vdbFrameRenderUpdateType();
  }

  // NOTE Spectral maps: transmittance of each channel. ShadowMaterial has one shadow mask, so it gets the mean.
  isSpectral(builder: { material?: { isShadowNodeMaterial?: boolean } }): boolean {
    return this.entry.rgbTexture !== null && builder.material?.isShadowNodeMaterial !== true;
  }

  override getNodeType(builder: never): string {
    return this.isSpectral(builder) ? "vec3" : "float";
  }

  override updateBefore(frame: never): boolean | undefined {
    const { renderer, scene, camera, renderId } = frame as { renderer: RendererLike; scene: THREE.Scene; camera: THREE.Camera; renderId: number };
    this.shadows.render(renderer, scene, camera, renderId);
    return undefined;
  }

  override setup(builder: never): N {
    const { entry } = this;
    const nodes = this.shadows.nodesOf(entry);
    const rgbKind = entry.rgbTexture !== null;
    const fn = this.shadows.receiverFn(entry, rgbKind);
    const inputs: Record<string, N> = { map: nodes.map, smp: sampler(nodes.map), world: positionWorld };
    if (rgbKind) inputs.rgb = nodes.rgb;
    if (!entry.isPoint) inputs.viewDepth = positionView.z.negate();
    const value = fn(inputs);
    // NOTE ShadowMaterial with spectral maps: mean transmittance of the channels
    return rgbKind && !this.isSpectral(builder) ? dot(value, vec3(1.0 / 3.0)) : value;
  }
}

/** Shadow filter marker (his `isFrameVolumeShadowFilter`). */
type VolumeFilter = ((inputs: N) => N) & { isVDBFrameVolumeShadowFilter?: true };

/** The receivers of the core's volume shadow maps, owned by a {@link VDBFrameRenderer} (`frameRenderer.volumeShadows`). */
export class VDBFrameVolumeShadowReceivers {
  readonly entries = new Map<THREE.Light, VDBFrameShadowEntry>();
  readonly frameRenderer: VDBFrameRenderer;
  // NOTE Replaced shadow nodes, stand-ins and attributes. update() runs mid-render, so the next update() disposes them.
  retired: { dispose(): void }[] = [];
  renderCount = 0;
  #renderId = -1;
  #camera: THREE.Camera | null = null;
  #renderer: RendererLike | null = null;
  #names = 0;
  #placeholders: Placeholders | null = null;
  /** Map pieces drawn in the last update, and how many waited for the budget (not his; tests and timing). */
  readonly lastUpdate = { drawn: 0, waiting: 0 };

  constructor(frameRenderer: VDBFrameRenderer) {
    this.frameRenderer = frameRenderer;
  }

  /**
   * The map size of the settings (`castShadowMapSize`, or his coarse size). The core clamps it to the device's limits
   * (K6.10a); `sync` takes the size of the map the core made.
   */
  getMapSize(): number {
    const { settings } = this.frameRenderer;
    return settings.castShadowCoarse ? frameShadowCoarseMapSize : Math.max(1, Math.round(settings.castShadowMapSize));
  }

  #entry(light: ShadowLight): VDBFrameShadowEntry {
    let entry = this.entries.get(light);
    if (entry) return entry;
    const isPoint = (light as THREE.PointLight).isPointLight === true;
    entry = {
      light,
      isPoint,
      size: this.getMapSize(),
      coreMap: null,
      lastDraw: -1,
      userFilter: null,
      texture: null,
      binding: new VDBCoreTextureBinding(isPoint ? "cube" : "2d-array"),
      rgbTexture: null,
      rgbBinding: new VDBCoreTextureBinding(isPoint ? "cube" : "2d-array"),
      viewBuffer: null,
      nodes: null,
    };
    this.entries.set(light, entry);
    return entry;
  }

  #placeholdersOf(device: GPUDevice): Placeholders {
    const p = this.#placeholders;
    if (p && p.device === device) return p;
    this.#destroyPlaceholders();
    const make = (layers: number, what: string) =>
      device.createTexture({ label: `@pmndrs/vdb/three volume shadow placeholder ${what}`, size: [1, 1, layers], format: "rgba16float", usage: 0x04 /* TEXTURE_BINDING */ });
    // A zero view: info.w (enabled) 0, so every receiver function returns 1 without sampling
    const view = device.createBuffer({ label: "@pmndrs/vdb/three volume shadow placeholder view", size: VDB_VOLUME_SHADOW_VIEW.size, usage: 0x80 /* STORAGE */ });
    return (this.#placeholders = { device, array: make(1, "2d-array"), cube: make(6, "cube"), view });
  }

  #destroyPlaceholders(): void {
    const p = this.#placeholders;
    if (!p) return;
    // NOTE A bind group of this render may still name them: destroy after the task that submits it
    const destroy = () => {
      p.array.destroy();
      p.cube.destroy();
      p.view.destroy();
    };
    setTimeout(destroy, 0);
    this.#placeholders = null;
  }

  /**
   * The light's texture and view nodes (made at its first receiver build, bound to placeholders until the core draws a
   * map). @internal
   */
  nodesOf(entry: VDBFrameShadowEntry): EntryNodes {
    if (entry.nodes) return entry.nodes;
    const renderer = this.#renderer;
    if (!renderer) throw new Error("VDBFrameVolumeShadowReceivers: a receiver was built before update() saw the renderer");
    const p = this.#placeholdersOf(getDevice(renderer as never));
    entry.binding.set(entry.texture ?? (entry.isPoint ? p.cube : p.array));
    entry.rgbBinding.set(entry.rgbTexture ?? (entry.isPoint ? p.cube : p.array));
    const make = entry.isPoint ? cubeTexture : texture;
    const k = this.#names++;
    const view = storage(this.#viewStandIn(entry.viewBuffer ?? p.view, k), "uvec4", VIEW_WORDS).toReadOnly() as StorageNode;
    view.setName(`vdbShadowView${k}`);
    entry.nodes = {
      k,
      map: make(entry.binding.texture),
      rgb: make(entry.rgbBinding.texture),
      view,
      loader: new VDBFrameStructLoaderNode(view, VDB_VOLUME_SHADOW_VIEW, `vdbFrameShadowView${k}`),
      fns: new Map(),
    };
    return entry.nodes;
  }

  #viewStandIn(buffer: GPUBuffer, k: number): StorageBufferAttribute {
    return vdbThreeStorageStandIn(this.#renderer!, buffer, `vdbShadowView${k}`);
  }

  /** The `wgslFn` of a light's receivers (grey or rgb; the point variants take no view depth). @internal */
  receiverFn(entry: VDBFrameShadowEntry, rgb: boolean): (inputs: Record<string, N>) => N {
    const nodes = this.nodesOf(entry);
    const key = rgb ? "rgb" : "grey";
    let fn = nodes.fns.get(key);
    if (fn) return fn;
    const view = `${nodes.loader.fnName}()`;
    const name = `${nodes.loader.fnName.replace("View", "Receiver")}${rgb ? "Rgb" : ""}`;
    const map = entry.isPoint ? "texture_cube<f32>" : "texture_2d_array<f32>";
    const source = entry.isPoint
      ? rgb
        ? `fn ${name}(map: ${map}, rgb: ${map}, smp: sampler, world: vec3f) -> vec3f {\n  return vdbVolumeShadowCubeRgb(map, rgb, smp, ${view}, world);\n}`
        : `fn ${name}(map: ${map}, smp: sampler, world: vec3f) -> f32 {\n  return vdbVolumeShadowCube(map, smp, ${view}, world);\n}`
      : rgb
        ? `fn ${name}(map: ${map}, rgb: ${map}, smp: sampler, world: vec3f, viewDepth: f32) -> vec3f {\n  return vdbVolumeShadowRgb(map, rgb, smp, ${view}, world, viewDepth);\n}`
        : `fn ${name}(map: ${map}, smp: sampler, world: vec3f, viewDepth: f32) -> f32 {\n  return vdbVolumeShadow(map, smp, ${view}, world, viewDepth);\n}`;
    fn = wgslFn(source, [receiverCodeNode(), nodes.loader]) as (inputs: Record<string, N>) => N;
    nodes.fns.set(key, fn);
    return fn;
  }

  // NOTE Frees a light's entry and puts back its own filter. Receivers build again without the volume shadow.
  #free(entry: VDBFrameShadowEntry): void {
    const { light } = entry;
    const shadow = light.shadow as THREE.LightShadow & { filterNode?: VolumeFilter | null };
    // NOTE Receivers that still have the volume shadow node until they build again read the placeholders
    this.#bind(entry, null);
    if (shadow.filterNode?.isVDBFrameVolumeShadowFilter) shadow.filterNode = entry.userFilter as VolumeFilter | null;
    const old = resetLightShadow(light);
    if (old) this.retired.push(old);
    this.retired.push(entry.binding, entry.rgbBinding);
    if (entry.nodes) this.retired.push(entry.nodes.view.value as StorageBufferAttribute);
    this.entries.delete(light);
  }

  #disposeRetired(): void {
    this.retired.forEach((resource) => resource.dispose());
    this.retired.length = 0;
  }

  /**
   * Puts the volume filter on each shadow light (new lights and new filters are found each render); frees the entries
   * of lights with no shadow or not in a scene, and all entries when `castShadows` is off.
   */
  update(renderer: RendererLike, scene: THREE.Object3D): void {
    this.#disposeRetired();
    this.#renderer = renderer;
    if (!this.frameRenderer.settings.castShadows) {
      this.entries.forEach((entry) => this.#free(entry));
      return;
    }
    // NOTE The light-shadow reset (three internal) is self-tested before the first filter goes on a light
    vdbFrameCheckLightReset();
    this.entries.forEach((entry) => {
      if (!isShadowLight(entry.light) || !isInScene(entry.light)) this.#free(entry);
    });
    scene.traverse((light) => {
      if (isShadowLight(light)) this.#install(renderer, light);
    });
  }

  #install(renderer: RendererLike, light: ShadowLight): void {
    const shadow = light.shadow as THREE.LightShadow & { filterNode?: VolumeFilter | null; map: THREE.RenderTarget | null };
    if (shadow.filterNode?.isVDBFrameVolumeShadowFilter) return;
    const entry = this.#entry(light);
    const userFilter = (shadow.filterNode as VolumeFilter | null | undefined) ?? null;
    const filters = vdbFrameShadowFilters();
    entry.userFilter = userFilter;
    // NOTE ShadowNode calls the filter when it builds. The base filter is three's or the user's.
    // Point lights give the direction from the light (bd3D), other lights give shadowCoord. The volume shadow node
    // does not use them: it finds its map coordinate from the drawn state.
    const filter: VolumeFilter = (inputs) => {
      vdbFrameCheckFilterInputs(inputs, entry.isPoint);
      const type = renderer.shadowMap.type as number;
      const baseFilter = (userFilter ?? (entry.isPoint ? (type === THREE.BasicShadowMap ? filters.basicPoint : filters.point) : filters.maps[type])) as ((i: unknown) => N) | null;
      if (!baseFilter) throw new Error("THREE.WebGPURenderer: Shadow map type not supported yet.");
      // NOTE PointShadowNode calls the filter in a TSL function, and its parameter proxy loses the shadow object.
      // So the base filter gets a plain object with the light shadow.
      const baseInputs: Record<string, unknown> = entry.isPoint
        ? { depthTexture: inputs.depthTexture, bd3D: inputs.bd3D, dp: inputs.dp, distance: inputs.distance }
        : { depthTexture: inputs.depthTexture, shadowCoord: inputs.shadowCoord, depthLayer: inputs.depthLayer };
      baseInputs.shadow = shadow;
      return baseFilter(baseInputs).mul(new VDBFrameVolumeShadowNode(this, entry));
    };
    filter.isVDBFrameVolumeShadowFilter = true;
    shadow.filterNode = filter;
    // NOTE The map exists when the shadow node was built before this filter
    if (shadow.map) {
      const old = resetLightShadow(light);
      if (old) this.retired.push(old);
    }
  }

  /** Called by the receivers of every light: the core update of this render (once per render and camera). */
  render(renderer: RendererLike, scene: THREE.Scene, camera: THREE.Camera, renderId: number): void {
    if (this.#renderId === renderId && this.#camera === camera) return;
    this.#renderId = renderId;
    this.#camera = camera;
    this.frameRenderer.coreFrame(renderer as never, scene as never, camera);
  }

  /**
   * Points a light's nodes at a core map (null: the placeholders). The texture and storage nodes keep their identity;
   * a new core texture or buffer gets a new stand-in as the node's `value`, and the old attribute is disposed at the next
   * update. Returns whether anything changed (the caller rebuilds the receivers: see `sync`).
   */
  #bind(entry: VDBFrameShadowEntry, m: VDBVolumeShadowMap | null): boolean {
    entry.coreMap = m;
    const nodes = entry.nodes;
    const p = nodes ? this.#placeholders : null;
    const texture = m?.texture ?? null;
    const rgb = m?.rgbTexture ?? null;
    const view = m?.view ?? null;
    const changed = texture !== entry.texture || rgb !== entry.rgbTexture || view !== entry.viewBuffer;
    if (texture !== entry.texture) {
      entry.texture = texture;
      if (nodes && p) this.#setTexture(entry.binding, nodes.map, texture ?? (entry.isPoint ? p.cube : p.array));
    }
    if (rgb !== entry.rgbTexture) {
      entry.rgbTexture = rgb;
      if (nodes && p) this.#setTexture(entry.rgbBinding, nodes.rgb, rgb ?? (entry.isPoint ? p.cube : p.array));
    }
    if (view !== entry.viewBuffer) {
      entry.viewBuffer = view;
      if (nodes && p) {
        const old = nodes.view.value as StorageBufferAttribute;
        nodes.view.value = this.#viewStandIn(view ?? p.view, nodes.k);
        this.retired.push(old);
      }
    }
    return changed;
  }

  // NOTE VDBCoreTextureBinding.set disposes the old stand-in at once; three keeps the GPU texture it bound (an
  // ExternalTexture's dispose never destroys it) and rebinds the node's new value at the next draw
  #setTexture(binding: VDBCoreTextureBinding, node: N, gpu: GPUTexture): void {
    if (binding.set(gpu)) node.value = binding.texture;
  }

  /**
   * After a core update, each entry binds its core map (texture, rgb twin, view buffer) and reads the budget counters.
   * The map size, the cascades and the drawn views live in the view buffer, so only a new core object (the first map, a
   * new size, a spectral switch, a map that went away or came back) touches the receivers: their nodes get the new
   * stand-ins and the light's receivers rebuild. The rebuild is needed: three r187 re-reads the object-group bindings
   * (textures, storage buffers) of a plain material only on a full refresh, which a node `value` swap does not cause, so
   * a receiver drawn with a shared refresh would keep the old bind group, and with it a texture the core retires
   * (K6.10 finding 8). They rebuild at their next draw (the next render: this one may still draw with the old objects,
   * which the core keeps alive for two more updates).
   */
  sync(maps: ReadonlyMap<number | string, VDBVolumeShadowMap>, stats: { drawn: number; waiting: number }): void {
    this.renderCount++;
    this.lastUpdate.drawn = stats.drawn;
    this.lastUpdate.waiting = stats.waiting;
    this.entries.forEach((entry) => {
      const m = maps.get(entry.light.id) ?? null;
      if (this.#bind(entry, m) && entry.nodes) {
        const old = resetLightShadow(entry.light);
        if (old) this.retired.push(old);
      }
      if (!m) return;
      entry.size = m.size;
      entry.lastDraw = m.lastDraw[0] ?? entry.lastDraw;
    });
  }

  /** Puts back the filter of each light (a later render installs again). */
  dispose(): void {
    this.entries.forEach((entry) => this.#free(entry));
    this.#disposeRetired();
    this.#destroyPlaceholders();
    this.#renderId = -1;
    this.#camera = null;
    this.#renderer = null;
  }
}
