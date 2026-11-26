/**
 * K3.1 / K6.8: the froxel renderer's host side in Node (no GPU): the public surface, the shared instances, the volume's
 * props and box, the settings (the core's). The core kernels run against their twins in e2e/k6-core.spec.ts; the
 * adapter's frames in e2e/k6-frames.spec.ts and the k3 specs.
 */

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { readWvdb, wvdbDenseVolume } from "../../src/tools/internal.js";
import { VDBFrameRenderer, VDBFrameVolume, vdbDenseTextures } from "../../src/three/index.js";
import { VDB_FRAME_ALL_LIGHTS, vdbFrameDefaultSettings } from "../../src/core/frame/index.js";
import { fixture } from "../tools/wvdb-helpers.js";

describe("@pmndrs/vdb/three frame exports", () => {
  it("exports the froxel renderer surface; the composite material and stand-ins stay internal (D32); the TSL passes are gone (K6.8)", async () => {
    const three = (await import("../../src/three/index.js")) as Record<string, unknown>;
    const internal = (await import("../../src/three/internal.js")) as Record<string, unknown>;
    for (const name of ["VDBFrameVolume", "VDBFrameRenderer", "vdbFrameRenderer", "VDBFrameEmissionLight", "vdbFrameMedium"]) {
      expect(typeof three[name], name).toBe("function");
    }
    for (const name of ["VDBFrameCoreIntegrateMaterial", "VDBFrameVolumeShadowReceivers", "VDBCoreTextureBinding"]) {
      expect(name in three, name).toBe(false);
      expect(name in internal, name).toBe(true);
    }
    for (const name of ["VDBFrameInjectMaterial", "VDBFrameIntegrateMaterial", "vdbFrameChunks", "VDBFrameVolumeShadows"]) {
      expect(name in internal, name).toBe(false);
    }
  });
});

describe("vdbFrameRenderer sharing (K5.4d)", () => {
  it("is one default instance, and one per three renderer when given one", async () => {
    const { vdbFrameRenderer } = await import("../../src/three/index.js");
    const a = {},
      b = {};
    expect(vdbFrameRenderer()).toBe(vdbFrameRenderer());
    expect(vdbFrameRenderer(a)).toBe(vdbFrameRenderer(a));
    expect(vdbFrameRenderer(a)).not.toBe(vdbFrameRenderer(b));
    expect(vdbFrameRenderer(a)).not.toBe(vdbFrameRenderer());
  });
});

describe("VDBFrameVolume (K3.1, K6.8)", () => {
  const dense = wvdbDenseVolume(readWvdb(fixture("cloud_04_variant_0000.wvdb")));
  const source = vdbDenseTextures(dense);
  const bbox = new THREE.Box3(new THREE.Vector3(...dense.bbox![0]), new THREE.Vector3(...dense.bbox![1]));

  it("has his props and defaults, the source's mean density, and a proxy on the renderer's composite material", () => {
    const frameRenderer = new VDBFrameRenderer();
    const v = new VDBFrameVolume(source, { frameRenderer });
    expect(v.isVDBFrameVolume).toBe(true);
    expect(v.extinction).toBe(100);
    // K5.9: no own phase: the renderer's settings (his volume defaults, HG 0.8, are the renderer's defaults too)
    expect(v.phase).toBe(null);
    expect(v.phaseG).toBe(null);
    expect(v.getPhase()).toEqual({ phase: "henyey-greenstein", phaseG: 0.8, phaseBackG: -0.3, phaseBackWeight: 0.2, phaseGExponent: 0 });
    expect(v.shadowSteps).toBe(32);
    expect(v.lights).toBe(VDB_FRAME_ALL_LIGHTS);
    expect(v.emissionColor).toBe(undefined);
    expect(v.meanDensity).toBeCloseTo(dense.meanDensity, 12);
    expect(v.getDensityScale()).toBeCloseTo(1 / dense.meanDensity, 9);
    v.normalizeDensity = false;
    expect(v.getDensityScale()).toBe(1);
    expect(v.proxy.material).toBe(frameRenderer.proxyMaterial());
    expect(v.proxy.renderOrder).toBe(1_000_000);
    // The box: his FrameVolume matrix of the unit box (centre, size) from the .wvdb bbox.
    const box = new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5)).applyMatrix4(v.boxMatrix);
    expect(box.min.distanceTo(bbox.min)).toBeLessThan(1e-9);
    expect(box.max.distanceTo(bbox.max)).toBeLessThan(1e-9);
    expect(() => new VDBFrameVolume({} as never)).toThrow(/vdbDenseTextures/);
    v.dispose();
    frameRenderer.dispose();
  });

  it("emissionColor is core data (VDBEmissionColor); a TSL hook function throws naming the data form", () => {
    const v = new VDBFrameVolume(source, { frameRenderer: new VDBFrameRenderer(), emissionColor: { spectrum: "soot", exponent: 2 } });
    expect(v.emissionColor).toEqual({ spectrum: "soot", exponent: 2 });
    expect(() => new VDBFrameVolume(source, { emissionColor: (() => null) as never })).toThrow(/VDB_BAD_OPTION|core data/);
  });
});

describe("VDBFrameRenderer settings (K6.8: the core's)", () => {
  it("his defaults (V33: 'frame' lighting, temporal, 'neural'), the K6 fields, the default MLP (ours, K6.N)", () => {
    const fr = new VDBFrameRenderer();
    const s = fr.settings;
    expect(s).toEqual(vdbFrameDefaultSettings());
    expect([s.lighting, s.temporal, s.multipleScattering]).toEqual(["frame", true, "neural"]);
    expect([s.scatteringOctaves, s.scatteringOctaveExtinction, s.scatteringOctaveContribution, s.scatteringOctavePhase]).toEqual([4, 0.5, 0.5, 0.5]);
    expect([s.temporalBlend, s.shadowResolution, s.shadowCascades, s.shadowDistance]).toEqual([0.2, 64, 1, 512]);
    expect([s.phase, s.phaseG, s.phaseBackG, s.phaseBackWeight, s.phaseGExponent]).toEqual(["henyey-greenstein", 0.8, -0.3, 0.2, 0]);
    expect([s.environmentLighting, s.environmentSamples, s.environmentShOrder]).toEqual(["directional", 8, 1]);
    expect([s.occlusion, s.ambientOcclusion]).toEqual([true, true]);
    expect(fr.mlp?.layers.length).toBeGreaterThan(0);
    expect(fr.froxels).toBe(true);
    expect(fr.core).toBe(null);
    expect(fr.redrawStats.transmittance).toEqual({ drawn: 0, skipped: 0 });
  });
});
