/**
 * K3.6 / K6.8: the host side of the volume shadows and the emission light in Node (no GPU): his settings, the three r187
 * internals self-tests (and that they throw VDB_ENGINE_MISMATCH when three behaves otherwise), the shadow filter
 * install / restore on lights, the receivers' copy of the core's drawn state, and the emission light's copy of the
 * core's probe grid. The core's maps, budget and probes are tested in test/core/frame/shadows-emission.test.ts and
 * e2e/k6-core.spec.ts; the frames in e2e/k3-frame-shadows.spec.ts and k6-shadows.spec.ts.
 */

import * as THREE from "three";
import { LightsNode } from "three/webgpu";
import { afterEach, describe, expect, it } from "vitest";
import { VDBFrameEmissionLight, VDBFrameEmissionLightNode, VDBFrameRenderer, vdbFrameCheckLightReset } from "../../src/three/internal.js";
import { vdbFrameRegisterLightNode, vdbFrameRenderId, vdbFrameResetInternalsCache, vdbFrameCheckFilterInputs, vdbFramePrecompiling } from "../../src/three/frame/three-internals.js";
import type { VDBVolumeShadowMap } from "../../src/core/frame/types.js";

const fakeRenderer = () => ({ isWebGPURenderer: true, shadowMap: { type: THREE.PCFShadowMap, enabled: true } }) as never;

afterEach(() => vdbFrameResetInternalsCache());

describe("K3.6 settings and three internals", () => {
  it("his castShadow* settings and defaults", () => {
    const fr = new VDBFrameRenderer();
    expect(fr.settings).toMatchObject({ castShadows: false, castShadowMapSize: 512, castShadowCascades: 1, castShadowDistance: 4000, castShadowSteps: 32, castShadowUpdatesPerFrame: 1, castShadowCoarse: false });
    expect(fr.volumeShadows.getMapSize()).toBe(512);
    fr.settings.castShadowCoarse = true;
    expect(fr.volumeShadows.getMapSize()).toBe(64);
  });

  it("the light-shadow reset self-test passes on three r187", () => {
    expect(() => vdbFrameCheckLightReset()).not.toThrow();
  });

  it("fail-before: a LightsNode that does not share one node per light throws VDB_ENGINE_MISMATCH", () => {
    const proto = LightsNode.prototype as unknown as { setupLightsNode: (b: unknown) => unknown[] };
    const real = proto.setupLightsNode;
    proto.setupLightsNode = function (this: { _lights: THREE.Light[] }) {
      return this._lights.map((light) => ({ isAnalyticLightNode: true, light, shadowNode: null, disposeShadow() {} }));
    };
    try {
      expect(() => vdbFrameCheckLightReset()).toThrow(/VDB_ENGINE_MISMATCH.*AnalyticLightNode/);
      // cached: rethrown without running again
      proto.setupLightsNode = real;
      expect(() => vdbFrameCheckLightReset()).toThrow(/VDB_ENGINE_MISMATCH/);
    } finally {
      proto.setupLightsNode = real;
    }
  });

  it("fail-before: disposeShadow that does not bump the node version (receivers keep the old shadow) throws", async () => {
    const { AnalyticLightNode } = await import("three/webgpu");
    const proto = AnalyticLightNode.prototype as unknown as { disposeShadow: () => void };
    const real = proto.disposeShadow;
    proto.disposeShadow = function (this: { shadowNode: { dispose(): void } | null }) {
      this.shadowNode?.dispose();
      this.shadowNode = null;
    };
    try {
      expect(() => vdbFrameCheckLightReset()).toThrow(/VDB_ENGINE_MISMATCH.*version/);
    } finally {
      proto.disposeShadow = real;
    }
  });

  it("the other internals: render id, precompilation state, filter inputs and light node registration", () => {
    expect(vdbFrameRenderId({ _nodes: { nodeFrame: { renderId: 3 } } })).toBe(3);
    expect(() => vdbFrameRenderId({})).toThrow(/VDB_ENGINE_MISMATCH.*renderId/);
    expect(vdbFramePrecompiling({ _precompilationState: null })).toBe(false);
    expect(() => vdbFramePrecompiling({})).toThrow(/VDB_ENGINE_MISMATCH/);
    expect(() => vdbFrameCheckFilterInputs({ depthTexture: {}, shadowCoord: {} }, false)).not.toThrow();
    expect(() => vdbFrameCheckFilterInputs({ depthTexture: {} }, false)).toThrow(/VDB_ENGINE_MISMATCH.*filterNode/);
    expect(() => vdbFrameCheckFilterInputs({ depthTexture: {}, shadowCoord: {} }, true)).toThrow(/bd3D/);
    class NoRegistry {}
    expect(() => vdbFrameRegisterLightNode(NoRegistry as never, class {})).toThrow(/VDB_ENGINE_MISMATCH.*registerNode/);
  });
});

describe("K3.6 volume shadows: filters on lights", () => {
  it("castShadows puts the volume filter on each shadow light (keeping a user filter as its base); off restores it", () => {
    const fr = new VDBFrameRenderer();
    const scene = new THREE.Scene();
    const sun = new THREE.DirectionalLight();
    const spot = new THREE.SpotLight();
    const quiet = new THREE.PointLight();
    const userFilter = () => null;
    sun.castShadow = true;
    spot.castShadow = true;
    (spot.shadow as unknown as { filterNode: unknown }).filterNode = userFilter;
    scene.add(sun, spot, quiet);
    const renderer = fakeRenderer();

    fr.volumeShadows.update(renderer, scene);
    expect(fr.volumeShadows.entries.size).toBe(0);

    fr.settings.castShadows = true;
    fr.volumeShadows.update(renderer, scene);
    expect(fr.volumeShadows.entries.size).toBe(2);
    const filterOf = (l: THREE.Light) => (l.shadow as unknown as { filterNode: { isVDBFrameVolumeShadowFilter?: boolean } | null }).filterNode;
    expect(filterOf(sun)?.isVDBFrameVolumeShadowFilter).toBe(true);
    expect(filterOf(spot)?.isVDBFrameVolumeShadowFilter).toBe(true);
    expect(filterOf(quiet)).toBeFalsy();
    expect(fr.volumeShadows.entries.get(spot)!.userFilter).toBe(userFilter);
    // directional lights read cascades (layers of the core map); no map before the core drew one
    expect(fr.volumeShadows.entries.get(sun)!.cascades?.length).toBe(4);
    expect(fr.volumeShadows.entries.get(spot)!.cascades).toBeNull();
    expect(fr.volumeShadows.entries.get(sun)!.map).toBeNull();
    // the filter checks what three passes it
    expect(() => (filterOf(sun) as unknown as (i: unknown) => unknown)({})).toThrow(/VDB_ENGINE_MISMATCH/);

    // a removed light frees its entry and gets its own filter back
    scene.remove(spot);
    fr.volumeShadows.update(renderer, scene);
    expect(fr.volumeShadows.entries.size).toBe(1);
    expect(filterOf(spot)).toBe(userFilter);

    fr.settings.castShadows = false;
    fr.volumeShadows.update(renderer, scene);
    expect(fr.volumeShadows.entries.size).toBe(0);
    expect(filterOf(sun)).toBeNull();
    fr.dispose();
  });
});

describe("K6.5 receivers read the core's drawn state", () => {
  it("sync copies each map's drawn view (cascades, spot) and the budget counters; a light without a map reads none", () => {
    const fr = new VDBFrameRenderer();
    fr.settings.castShadows = true;
    const scene = new THREE.Scene();
    const sun = new THREE.DirectionalLight();
    const spot = new THREE.SpotLight();
    sun.castShadow = spot.castShadow = true;
    scene.add(sun, spot);
    fr.volumeShadows.update(fakeRenderer(), scene);
    const vp = (k: number) => Array.from({ length: 16 }, (_, i) => i + k);
    const map = (o: Partial<VDBVolumeShadowMap>): VDBVolumeShadowMap => ({ texture: null as unknown as GPUTexture, rgbTexture: null, size: 256, enabled: true, lastDraw: [3, 4, 5], ...o }) as unknown as VDBVolumeShadowMap;
    const drawn = { viewProjection: [vp(1), vp(2)], origin: [[1, 2, 3], [4, 5, 6]], forward: [0, -1, 0], far: [10, 20], perspective: false, cascadeCount: 2, breaks: [5, 50, 0, 0] };
    const maps = new Map<number | string, VDBVolumeShadowMap>([[sun.id, map({ drawn } as never)]]);
    fr.volumeShadows.sync(maps, { drawn: 1, waiting: 2 });
    const e = fr.volumeShadows.entries.get(sun)!;
    expect(e.enabled.value).toBe(1);
    expect(e.drawn.cascadeCount).toBe(2);
    expect(e.drawn.cascadeBreaks.toArray()).toEqual([5, 50, 0, 0]);
    expect(e.cascades![1]!.drawn.viewProjection.elements).toEqual(vp(2));
    expect(e.cascades![1]!.drawn.origin.toArray()).toEqual([4, 5, 6]);
    expect(e.cascades![1]!.drawn.far).toBe(20);
    expect(e.cascades![1]!.lastDraw).toBe(5);
    expect(e.lastDraw).toBe(3);
    expect(fr.volumeShadows.entries.get(spot)!.enabled.value).toBe(0);
    expect(fr.volumeShadows.lastUpdate).toEqual({ drawn: 1, waiting: 2 });
    fr.dispose();
    expect(fr.volumeShadows.entries.size).toBe(0);
  });
});

describe("K3.6 emission light host side", () => {
  it("registers its light node with three (Light.registerNode) and keeps his props", () => {
    const light = new VDBFrameEmissionLight({ intensity: 25, resolution: 32 });
    expect((VDBFrameEmissionLight.prototype as unknown as { _lightNode: unknown })._lightNode).toBe(VDBFrameEmissionLightNode);
    expect(light).toMatchObject({ intensity: 25, resolution: 32, range: 3, cascades: 1, distance: 512, steps: 24, isLight: true });
    expect(light.getSteps()).toBe(24);
    light.dispose();
  });

  it("updateProbes copies the core's probe grid of the scene's first volume's renderer (none: no probes)", () => {
    const light = new VDBFrameEmissionLight();
    const scene = new THREE.Scene();
    const probes = {
      texture: {} as GPUTexture,
      layout: { origins: [[0, 0, 0]], size: [16, 16, 48] },
      state: { emitters: 2, count: 1, resolution: 16, center: [1, 2, 3], min: [[-5, -6, -7]], size: [[10, 12, 14]] },
      lastUpdate: { directions: 2, probes: true },
    };
    let calls = 0;
    const fr = { coreFrame: () => void calls++, core: { outputs: { emissionProbes: probes } } };
    scene.add(Object.assign(new THREE.Object3D(), { isVDBFrameVolume: true, frameRenderer: fr }));
    light.updateProbes({}, scene, new THREE.PerspectiveCamera());
    expect(calls).toBe(1);
    expect([light.emitterCount, light.cascadeCount, light.probeResolution]).toEqual([2, 1, 16]);
    expect(light.probeCenter.toArray()).toEqual([1, 2, 3]);
    expect(light.cascadeMin[0]!.toArray()).toEqual([-5, -6, -7]);
    expect(light.cascadeSize[0]!.toArray()).toEqual([10, 12, 14]);
    expect(light.lastUpdate).toEqual({ directions: 2, probes: true });
    light.updateProbes({}, new THREE.Scene(), new THREE.PerspectiveCamera());
    expect([light.emitterCount, light.cascadeCount, light.coreProbes]).toEqual([0, 0, null]);
    light.dispose();
  });
});
