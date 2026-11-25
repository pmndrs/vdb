// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameEmissionLight.js L1–287 (FrameEmissionLightBase: props, probe cascades);
//   src/three-openvdb/frame/webgpu/FrameEmissionLight.js L59–658 (FrameEmissionLightNode: the probe lookup with the
//   cascade blend and the falloff past the last cascade; FrameEmissionLight: props, the node registration).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; K6.8: the emitters and probes are the core frame renderer's (its emission kernels, a stacked 3D
// texture read with hardware trilinear); this light carries the settings and lights meshes; every three r187 internal
// goes through `three-internals.ts` (VDB_ENGINE_MISMATCH on another three). Divergences: DIVERGES.md (K3.6, K6.5 rows).

/**
 * K3.6 / K6.5: light from the emissive grids of every visible `VDBFrameVolume` (his `FrameEmissionLight`). Add one to
 * the scene; three's lit node materials (`MeshStandardMaterial`, `MeshLambertMaterial`, …) receive it as irradiance,
 * like a light probe. The core frame renderer of the scene's volumes draws the emitters and L1 irradiance probes (up to
 * 4 cascades of ≤ 32³) when a volume or the cascades changed; a lit pixel reads 6 texels (12 near a cascade side).
 *
 *   const light = new VDBFrameEmissionLight({ intensity: 25, resolution: 32 });
 *   scene.add(light);   // volumes need an emission channel; colour: VDBFrameVolume `emissionColor`
 */

import * as THREE from "three";
import { AnalyticLightNode } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { vdbMaxComponent, vdbMinComponent } from "../volume-source.js";
import { frameEmissionMaxCascades as maxCascades } from "../../tools/frame/emission.js";
import { vdbFramePrecompiling, vdbFrameRegisterLightNode, vdbFrameRenderUpdateType } from "./three-internals.js";
import { VDBCoreTextureBinding } from "./core-textures.js";
import type { VDBFrameEmissionProbes } from "../../core/frame/types.js";
import type { VDBFrameRenderer } from "./renderer.js";
import type { VDBFrameVolume } from "./volume.js";

const { Fn, If, clamp, dot, float, int, max, min, mix, normalWorld, positionWorld, renderGroup, select, smoothstep, texture3D, uniform, uniformArray, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;

/** A 1³ zero placeholder until the core binds its probe stack. */
function stackPlaceholder(): THREE.Data3DTexture {
  const t = new THREE.Data3DTexture(new Uint16Array(4), 1, 1, 1);
  t.type = THREE.HalfFloatType;
  t.format = THREE.RGBAFormat;
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

/**
 * The irradiance at a lit point from the finest probe cascade that contains it; near a cascade side the next one
 * blends in (as the shadow grid); past the last cascade its clamped edge probes fall off with distance squared.
 */
export class VDBFrameEmissionLightNode extends AnalyticLightNode<THREE.Light> {
  static get type(): string {
    return "VDBFrameEmissionLightNode";
  }

  readonly gridMin: N;
  readonly gridSize: N;
  readonly gridCount: N;
  readonly gridResolution: N;
  readonly probeCenter: N;
  readonly probeActive: N;
  /** The core's probe stack (a 3D texture, 3 sub-volumes per cascade) and its layout. */
  readonly probeStack: N;
  readonly stackOrigins: N;
  readonly stackSize: N;

  constructor(light: VDBFrameEmissionLight | null = null) {
    super(light as never);
    // NOTE The core update runs before the first lit mesh of each render, as shadow maps
    const render = vdbFrameRenderUpdateType();
    (this as unknown as { updateType: string; updateBeforeType: string }).updateType = render;
    (this as unknown as { updateType: string; updateBeforeType: string }).updateBeforeType = render;
    // NOTE Render group, as three's lights: lit materials without node changes update only shared uniforms
    this.gridMin = uniformArray(
      Array.from({ length: maxCascades }, () => new THREE.Vector3()),
      "vec3",
    ).setGroup(renderGroup);
    this.gridSize = uniformArray(
      Array.from({ length: maxCascades }, () => new THREE.Vector3(1.0, 1.0, 1.0)),
      "vec3",
    ).setGroup(renderGroup);
    this.gridCount = uniform(0, "int").setGroup(renderGroup);
    this.gridResolution = uniform(1.0).setGroup(renderGroup);
    this.probeCenter = uniform(new THREE.Vector3()).setGroup(renderGroup);
    this.probeActive = uniform(0.0).setGroup(renderGroup);
    this.probeStack = texture3D(stackPlaceholder());
    this.stackOrigins = uniformArray(
      Array.from({ length: 3 * maxCascades }, () => new THREE.Vector3()),
      "vec3",
    ).setGroup(renderGroup);
    this.stackSize = uniform(new THREE.Vector3(1, 1, 1)).setGroup(renderGroup);
  }

  private get emissionLight(): VDBFrameEmissionLight {
    return (this as unknown as { light: VDBFrameEmissionLight }).light;
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
    const light = this.emissionLight;
    const gridMin = this.gridMin.array as THREE.Vector3[];
    const gridSize = this.gridSize.array as THREE.Vector3[];
    for (let i = 0; i < maxCascades; i++) {
      gridMin[i]!.copy(light.cascadeMin[i]!);
      gridSize[i]!.copy(light.cascadeSize[i]!);
    }
    this.gridCount.value = light.cascadeCount;
    this.gridResolution.value = light.probeResolution;
    this.probeCenter.value.copy(light.probeCenter);
    this.probeActive.value = light.emitterCount > 0 ? 1.0 : 0.0;
    // the core stack and its sub-volume origins (a new stack: a new stand-in)
    const probes = light.coreProbes;
    if (probes) {
      if (light.coreBinding.set(probes.texture)) this.probeStack.value = light.coreBinding.texture;
      const origins = this.stackOrigins.array as THREE.Vector3[];
      probes.layout.origins.forEach((o, i) => origins[i]?.set(o[0], o[1], o[2]));
      this.stackSize.value.set(probes.layout.size[0], probes.layout.size[1], probes.layout.size[2]);
    }
    return undefined;
  }

  override setup(builder: never): null {
    const { gridMin, gridSize, gridCount, gridResolution, probeCenter, probeActive } = this;
    const gridUvw = (cascade: N, position: N): N => position.sub(gridMin.element(cascade)).div(gridSize.element(cascade));
    // NOTE Half a probe inside the cascade. Then the linear filter does not read the clamped edge.
    const insideGrid = (uvw: N): N => {
      const inset = float(0.5).div(gridResolution);
      return vdbMinComponent(uvw).greaterThanEqual(inset).and(vdbMaxComponent(uvw).lessThanEqual(float(1.0).sub(inset)));
    };
    // NOTE Finest cascade from the first cascade that contains the position, or -1
    const findCascade = (position: N, first: N): N => {
      let cascade: N = int(-1);
      for (let i = maxCascades - 1; i >= 0; i--) cascade = select(int(i).lessThan(gridCount).and(int(i).greaterThanEqual(first)).and(insideGrid(gridUvw(int(i), position))), int(i), cascade);
      return cascade;
    };

    const irradiance = Fn(() => {
      const position = positionWorld.toVar();
      const red = vec4(0.0).toVar();
      const green = vec4(0.0).toVar();
      const blue = vec4(0.0).toVar();
      const falloff = float(1.0).toVar();
      // one channel of a cascade in the stack, hardware trilinear clamped to its half-texel box
      const sampleStack = (cascade: N, channel: number, uvw: N): N => {
        const r = gridResolution;
        const p = this.stackOrigins.element(cascade.mul(3).add(channel)).add(clamp(uvw.mul(r), vec3(0.5), vec3(r.sub(0.5))));
        return this.probeStack.sample(p.div(this.stackSize)).level(0);
      };
      const sampleProbes = (cascade: N, uvw: N, weight: N) => {
        red.assign(mix(red, sampleStack(cascade, 0, uvw), weight));
        green.assign(mix(green, sampleStack(cascade, 1, uvw), weight));
        blue.assign(mix(blue, sampleStack(cascade, 2, uvw), weight));
      };

      If(gridCount.greaterThan(0), () => {
        const cascade = findCascade(position, int(0)).toVar();
        If(cascade.greaterThanEqual(0), () => {
          const uvw = gridUvw(cascade, position).toVar();
          sampleProbes(cascade, uvw, float(1.0));
          const sideDistance = min(uvw, vec3(1.0).sub(uvw));
          const blend = float(1.0).sub(smoothstep(0.0, 0.1, min(sideDistance.x, min(sideDistance.y, sideDistance.z)))).toVar();
          If(blend.greaterThan(0.0), () => {
            const nextCascade = findCascade(position, cascade.add(1)).toVar();
            If(nextCascade.greaterThanEqual(0), () => {
              sampleProbes(nextCascade, gridUvw(nextCascade, position), blend);
            });
          });
        }).Else(() => {
          const lastCascade = gridCount.sub(1).toVar();
          const inset = float(0.5).div(gridResolution);
          const uvw = clamp(gridUvw(lastCascade, position), vec3(inset), vec3(float(1.0).sub(inset))).toVar();
          const toEdge = gridMin.element(lastCascade).add(uvw.mul(gridSize.element(lastCascade))).sub(probeCenter);
          const toPoint = position.sub(probeCenter);
          sampleProbes(lastCascade, uvw, float(1.0));
          falloff.assign(min(1.0, dot(toEdge, toEdge).div(max(dot(toPoint, toPoint), 1e-6))));
        });
      });

      const normal = normalWorld;
      return max(vec3(red.x.add(dot(red.yzw, normal)), green.x.add(dot(green.yzw, normal)), blue.x.add(dot(blue.yzw, normal))), vec3(0.0)).mul(falloff.mul(probeActive));
    })();

    const b = builder as unknown as { context: { irradiance: N } };
    const colorNode = (this as unknown as { colorNode: N }).colorNode;
    b.context.irradiance.addAssign(colorNode.mul(irradiance));
    return null;
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
