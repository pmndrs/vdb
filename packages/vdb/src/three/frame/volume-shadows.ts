// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameVolumeShadows.js L34–1027 (the light shadow reset, receiver uniforms, cascade
//   lookup, FrameVolumeShadowNode, the filter install, dispose).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Past the pin (K7.1): his 62a6e3e (lookups from the drawn state) and c13f2d5 (the budget counts only lights of the
// rendered scene; the core's maps).
// Changes: TypeScript; K6.8: the maps are the core's (`VDBVolumeShadowMap`, drawn by its kernels), three only receives
// them; every three r187 internal goes through `three-internals.ts`, which self-tests it and throws
// VDB_ENGINE_MISMATCH. Divergences: DIVERGES.md (K3.6, K6.5 rows).

/**
 * K3.6 / K6.5: shadows of `VDBFrameVolume`s on meshes (`frameRenderer.settings.castShadows`, his `FrameVolumeShadows`).
 * Each shadow-casting directional, spot or point light gets a Beer shadow map from the core frame renderer (its
 * `outputs.volumeShadows`); the light's `shadow.filterNode` multiplies three's shadow by the volume transmittance at
 * the receiver, read with the state the map was drawn with (`VDBVolumeShadowMap.drawn`): directional cascades are
 * layers of a 2d-array, a spot map one layer, a point map a cube view.
 *
 * three: receivers need `receiveShadow`, the light `castShadow`, and `renderer.shadowMap.enabled`. The core update runs
 * before the first receiver of each render (the node's `updateBefore`), so the maps are this render's.
 */

import * as THREE from "three";
import { Node } from "three/webgpu";
import * as TSL from "three/tsl";
import type { WebGPURenderer } from "three/webgpu";
import type { VDBTslNode } from "../grid-node.js";
import { frameMaxShadowMapCascades, frameShadowCoarseMapSize, frameShadowMinOpticalDepth } from "../../tools/frame/shadows.js";
import { vdbFrameCheckFilterInputs, vdbFrameCheckLightReset, vdbFrameRenderUpdateType, vdbFrameShadowFilters, vdbFrameSharedLightNode } from "./three-internals.js";
import type { VDBFrameRenderer } from "./renderer.js";
import { VDBCoreTextureBinding } from "./core-textures.js";
import type { VDBVolumeShadowMap } from "../../core/frame/types.js";

const { Fn, If, clamp, cubeTexture, dot, exp, float, int, length, max, positionView, positionWorld, renderGroup, select, texture, uniform, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;
type RendererLike = WebGPURenderer & { isWebGPURenderer?: boolean };
type ShadowLight = THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight;
type RayUniforms = { frameShadowOrigin: N; frameShadowForward: N; frameShadowPerspective: N; frameShadowFar: N };

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

/** Depth of a world position: the distance from the light (perspective) or from the camera plane (orthographic). */
const shadowDepth = (ray: RayUniforms, worldPosition: N): N => {
  const offset = worldPosition.sub(ray.frameShadowOrigin);
  return select(ray.frameShadowPerspective.equal(1), length(offset), dot(offset, ray.frameShadowForward));
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

// NOTE A uniform that copies a value of the entry one time in each render (the render group).
// A light shadow node is built one time; its object-group uniforms would keep the value of the first build.
const renderUniform = (value: unknown, type: string, read: () => unknown): N => uniform(value, type).setGroup(renderGroup).onRenderUpdate(read);

interface Receiver {
  enabled: N;
  /** Spot lights: the drawn view-projection (null for cascades and point lights). */
  viewProjection: N | null;
  ray: RayUniforms;
  cascades: { count: N; breaks: N; items: { viewProjection: N; origin: N; far: N }[] } | null;
}

/** One shadow light's core map and the state it was drawn with. */
export interface VDBFrameShadowEntry {
  light: ShadowLight;
  isPoint: boolean;
  size: number;
  /** The map's texture node (null until the core drew a map: the receivers read no volume shadow). */
  map: N | null;
  /** K6.4 spectral: the optical depth rgb twin (`rgbTexture`) node. */
  spectralMap: N | null;
  enabled: { value: number };
  drawn: { origin: THREE.Vector3; forward: THREE.Vector3; perspective: number; far: number; viewProjection: THREE.Matrix4; cascadeCount: number; cascadeBreaks: THREE.Vector4 };
  cascades: { drawn: { viewProjection: THREE.Matrix4; origin: THREE.Vector3; far: number }; lastDraw: number }[] | null;
  lastDraw: number;
  receiver: Receiver;
  userFilter: ((inputs: unknown) => N) | null;
  texture: GPUTexture | null;
  binding: VDBCoreTextureBinding;
  rgbTexture: GPUTexture | null;
  rgbBinding: VDBCoreTextureBinding;
}

// NOTE Receivers read the drawn state: the values of the camera that drew each map or cascade layer.
// A map that waits for the frame budget is older, but its values match, so its shadow stays in place.
const createReceiver = (entry: VDBFrameShadowEntry): Receiver => {
  const { drawn, cascades, isPoint } = entry;
  return {
    enabled: renderUniform(0, "int", () => entry.enabled.value),
    viewProjection: cascades || isPoint ? null : renderUniform(new THREE.Matrix4(), "mat4", () => drawn.viewProjection),
    ray: {
      frameShadowOrigin: renderUniform(new THREE.Vector3(), "vec3", () => drawn.origin),
      frameShadowForward: renderUniform(new THREE.Vector3(), "vec3", () => drawn.forward),
      frameShadowPerspective: renderUniform(0, "int", () => drawn.perspective),
      frameShadowFar: renderUniform(1.0, "float", () => drawn.far),
    },
    cascades: cascades
      ? {
          count: renderUniform(0, "int", () => drawn.cascadeCount),
          breaks: renderUniform(new THREE.Vector4(), "vec4", () => drawn.cascadeBreaks),
          items: cascades.map((cascade) => ({
            viewProjection: renderUniform(new THREE.Matrix4(), "mat4", () => cascade.drawn.viewProjection),
            origin: renderUniform(new THREE.Vector3(), "vec3", () => cascade.drawn.origin),
            far: renderUniform(1.0, "float", () => cascade.drawn.far),
          })),
        }
      : null,
  };
};

// NOTE Picks the cascade of the receiver by its view depth, as three's CSMShadowNode (twin: vdbFrameShadowCascadeIndex / Uv).
// Returns the layer uv, the layer, the ray uniforms of the cascade, and whether the receiver is in a cascade.
const cascadeLookup = (receiver: Receiver, size: number): { uv: N; inside: N; index: N; ray: RayUniforms } => {
  const { count, breaks, items } = receiver.cascades!;
  const depth = positionView.z.negate();
  const breakList = [breaks.x, breaks.y, breaks.z, breaks.w];
  // NOTE Cascade index: the number of cascade far distances in front of the receiver
  let index: N = int(0);
  for (let i = 0; i < frameMaxShadowMapCascades - 1; i++) {
    index = index.add(select(depth.greaterThan(breakList[i]).and(int(i + 1).lessThan(count)), int(1), int(0)));
  }
  const pick = (key: "viewProjection" | "origin" | "far"): N => items.reduce<N>((result, item, i) => (i === 0 ? item[key] : select(index.equal(i), item[key], result)), null);
  const clip = pick("viewProjection").mul(vec4(positionWorld, 1.0));
  const tileUv = vec2(clip.x.mul(0.5).add(0.5), float(0.5).sub(clip.y.mul(0.5)));
  const lastBreak = breakList.reduce<N>((result, value, i) => (i === 0 ? value : select(count.equal(i + 1), value, result)), null);
  // NOTE Half a texel in from the layer sides (his tiles of one atlas; the same clamp)
  const inset = float(0.5 / size);
  return {
    uv: clamp(tileUv, vec2(inset), vec2(float(1.0).sub(inset))),
    index,
    inside: depth.lessThanEqual(lastBreak).and(tileUv.x.greaterThanEqual(0.0)).and(tileUv.x.lessThanEqual(1.0)).and(tileUv.y.greaterThanEqual(0.0)).and(tileUv.y.lessThanEqual(1.0)),
    ray: { frameShadowOrigin: pick("origin"), frameShadowForward: receiver.ray.frameShadowForward, frameShadowPerspective: int(0), frameShadowFar: pick("far") },
  };
};

/**
 * Volume transmittance at a receiver from the light's core Beer shadow map (part of the light's shadow filter;
 * `vdbVolumeShadow` / `vdbVolumeShadowCube` in TSL, twin `vdbFrameVolumeShadowLookup`). The map coordinate comes from
 * the drawn state (his 62a6e3e): cascades are layers of a 2d-array, a spot map one layer, a point map a cube view
 * (three negates x of a cube lookup on WebGPU, so the direction goes in with x negated). K6.4 spectral maps: the rgb twin
 * gives each channel's transmittance (`vdbVolumeShadowRgb`), the channel mean in a ShadowMaterial (his rule).
 * `updateBefore` runs the core update before the first receiver of each render.
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
    return this.entry.spectralMap !== null && builder.material?.isShadowNodeMaterial !== true;
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
    const { map, isPoint, receiver } = entry;
    // NOTE Built before the core drew the map: no volume shadow; the first map rebuilds the receivers (sync)
    if (!map) return this.isSpectral(builder) ? vec3(1.0) : float(1.0);
    const spectralMap = entry.rgbTexture ? entry.spectralMap : null;
    const spectral = !!spectralMap && this.isSpectral(builder);
    let { ray } = receiver;
    let inside: N = receiver.enabled.equal(1);
    let texel: N;
    let rgb: N | null = null;
    if (receiver.cascades) {
      const lookup = cascadeLookup(receiver, entry.size);
      ray = lookup.ray;
      inside = inside.and(lookup.inside);
      texel = map.sample(lookup.uv).depth(lookup.index).level(0);
      if (spectralMap) rgb = spectralMap.sample(lookup.uv).depth(lookup.index).level(0);
    } else if (isPoint) {
      const d = positionWorld.sub(ray.frameShadowOrigin);
      texel = map.sample(vec3(d.x.negate(), d.y, d.z)).level(0);
      if (spectralMap) rgb = spectralMap.sample(vec3(d.x.negate(), d.y, d.z)).level(0);
    } else {
      const clip = receiver.viewProjection!.mul(vec4(positionWorld, 1.0));
      const ndc = clip.xy.div(clip.w);
      const uv = vec2(ndc.x.mul(0.5).add(0.5), float(0.5).sub(ndc.y.mul(0.5)));
      inside = inside.and(clip.w.greaterThan(0.0)).and(uv.x.greaterThanEqual(0.0)).and(uv.x.lessThanEqual(1.0)).and(uv.y.greaterThanEqual(0.0)).and(uv.y.lessThanEqual(1.0));
      texel = map.sample(uv).depth(int(0)).level(0);
      if (spectralMap) rgb = spectralMap.sample(uv).depth(int(0)).level(0);
    }
    return Fn(() => {
      const result = (spectral ? vec3(1.0) : float(1.0)).toVar();
      If(inside, () => {
        const value = texel.toVar();
        const opticalDepth = value.w;
        If(opticalDepth.greaterThan(frameShadowMinOpticalDepth), () => {
          const far = ray.frameShadowFar;
          const first = value.x.div(opticalDepth).mul(far);
          const last = value.y.div(opticalDepth).mul(far);
          const depth = shadowDepth(ray, positionWorld);
          // NOTE Beer shadow map: optical depth grows linearly from the first to the last density
          const along = clamp(depth.sub(first).div(max(last.sub(first), 1e-4)), 0.0, 1.0);
          if (spectral) {
            result.assign(exp(rgb!.rgb.mul(along).negate()));
          } else if (rgb) {
            // NOTE ShadowMaterial with spectral maps: mean transmittance of the channels
            result.assign(dot(exp(rgb.rgb.mul(along).negate()), vec3(1.0 / 3.0)));
          } else {
            result.assign(exp(opticalDepth.mul(along).negate()));
          }
        });
      });
      return result;
    })();
  }
}

/** Shadow filter marker (his `isFrameVolumeShadowFilter`). */
type VolumeFilter = ((inputs: N) => N) & { isVDBFrameVolumeShadowFilter?: true };

/** The receivers of the core's volume shadow maps, owned by a {@link VDBFrameRenderer} (`frameRenderer.volumeShadows`). */
export class VDBFrameVolumeShadowReceivers {
  readonly entries = new Map<THREE.Light, VDBFrameShadowEntry>();
  readonly frameRenderer: VDBFrameRenderer;
  // NOTE Replaced shadow nodes and stand-ins. update() runs mid-render, so the next update() disposes them.
  retired: { dispose(): void }[] = [];
  renderCount = 0;
  #renderId = -1;
  #camera: THREE.Camera | null = null;
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
      map: null,
      spectralMap: null,
      enabled: { value: 0 },
      drawn: { origin: new THREE.Vector3(), forward: new THREE.Vector3(0.0, 0.0, -1.0), perspective: 0, far: 1.0, viewProjection: new THREE.Matrix4(), cascadeCount: 0, cascadeBreaks: new THREE.Vector4() },
      cascades: (light as THREE.DirectionalLight).isDirectionalLight
        ? Array.from({ length: frameMaxShadowMapCascades }, () => ({ drawn: { viewProjection: new THREE.Matrix4(), origin: new THREE.Vector3(), far: 1.0 }, lastDraw: -1 }))
        : null,
      lastDraw: -1,
      receiver: null as unknown as Receiver,
      userFilter: null,
      texture: null,
      binding: new VDBCoreTextureBinding(isPoint ? "cube" : "2d-array"),
      rgbTexture: null,
      rgbBinding: new VDBCoreTextureBinding(isPoint ? "cube" : "2d-array"),
    };
    entry.receiver = createReceiver(entry);
    this.entries.set(light, entry);
    return entry;
  }

  // NOTE Frees a light's entry and puts back its own filter. Receivers build again without the volume shadow.
  #free(entry: VDBFrameShadowEntry): void {
    const { light } = entry;
    const shadow = light.shadow as THREE.LightShadow & { filterNode?: VolumeFilter | null };
    // NOTE Receivers that still have the volume shadow node until they build again read no volume shadow
    entry.enabled.value = 0;
    if (shadow.filterNode?.isVDBFrameVolumeShadowFilter) shadow.filterNode = entry.userFilter as VolumeFilter | null;
    const old = resetLightShadow(light);
    if (old) this.retired.push(old);
    this.retired.push(entry.binding, entry.rgbBinding);
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
   * After a core update, each entry reads its core map: the drawn view into the receiver uniforms (they read the state
   * the map was drawn with), the map texture through a stand-in, the budget counters. The first map (or a new size)
   * rebuilds the light's receivers.
   */
  sync(maps: ReadonlyMap<number | string, VDBVolumeShadowMap>, stats: { drawn: number; waiting: number }): void {
    this.renderCount++;
    this.lastUpdate.drawn = stats.drawn;
    this.lastUpdate.waiting = stats.waiting;
    this.entries.forEach((entry) => {
      const m = maps.get(entry.light.id);
      if (!m) {
        entry.enabled.value = 0;
        return;
      }
      entry.enabled.value = m.enabled ? 1 : 0;
      if (m.texture !== entry.texture) {
        const rebuild = entry.texture === null || m.size !== entry.size;
        entry.texture = m.texture;
        entry.binding.set(m.texture);
        const standIn = entry.binding.texture!;
        entry.map = entry.isPoint ? cubeTexture(standIn) : texture(standIn);
        entry.size = m.size;
        if (rebuild) {
          const old = resetLightShadow(entry.light);
          if (old) this.retired.push(old);
        }
      }
      // K6.4: a spectral map's rgb twin; the receivers return rgb (or the channel mean in a ShadowMaterial), so a change of
      // kind rebuilds them
      const rgb = m.rgbTexture ?? null;
      if (rgb !== entry.rgbTexture) {
        const rebuild = !!rgb !== !!entry.rgbTexture;
        entry.rgbTexture = rgb;
        if (rgb) {
          entry.rgbBinding.set(rgb);
          const standIn = entry.rgbBinding.texture!;
          entry.spectralMap = entry.isPoint ? cubeTexture(standIn) : texture(standIn);
        } else {
          entry.spectralMap = null;
        }
        if (rebuild) {
          const old = resetLightShadow(entry.light);
          if (old) this.retired.push(old);
        }
      }
      const d = m.drawn;
      const { drawn } = entry;
      if (entry.cascades) {
        drawn.cascadeCount = d.cascadeCount;
        drawn.cascadeBreaks.fromArray(d.breaks as unknown as number[]);
        drawn.forward.fromArray(d.forward as unknown as number[]);
        for (let i = 0; i < d.cascadeCount; i++) {
          const c = entry.cascades[i]!;
          c.drawn.viewProjection.fromArray(d.viewProjection[i]!);
          c.drawn.origin.fromArray(d.origin[i] as unknown as number[]);
          c.drawn.far = d.far[i]!;
          c.lastDraw = m.lastDraw[i + 1] ?? c.lastDraw;
        }
      } else {
        if (d.viewProjection[0]) drawn.viewProjection.fromArray(d.viewProjection[0]);
        if (d.origin[0]) drawn.origin.fromArray(d.origin[0] as unknown as number[]);
        drawn.forward.fromArray(d.forward as unknown as number[]);
        drawn.perspective = d.perspective ? 1 : 0;
        drawn.far = d.far[0] ?? drawn.far;
      }
      entry.lastDraw = m.lastDraw[0] ?? entry.lastDraw;
    });
  }

  /** Puts back the filter of each light (a later render installs again). */
  dispose(): void {
    this.entries.forEach((entry) => this.#free(entry));
    this.#disposeRetired();
    this.#renderId = -1;
    this.#camera = null;
  }
}
