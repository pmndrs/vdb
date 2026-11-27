/**
 * K3.7 `VDBLevelSetMaterial` and `VDBPoints` in Node: construction, option
 * validation and the trace entry's text (public `vdb*` API only). The shaders
 * run on a GPU in e2e/k3-levelset.spec.ts.
 */

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { MeshPhysicalNodeMaterial, MeshStandardNodeMaterial, PhysicalLightingModel, PointsNodeMaterial } from "three/webgpu";
import { VDB_GPU_LAYOUT_VERSION, VDBError } from "../../src/core/index.js";
import { vdbPointPixelScale } from "../../src/core/internal.js";
import {
  VDB_LEVEL_SET_SAMPLES_CAP,
  VDBGridNode,
  VDBLevelSetMaterial,
  VDBLevelSetPhysicalLightingModel,
  VDBLevelSetPhysicalMaterial,
  VDBPoints,
  vdbAttenuatedPointSize,
  vdbPointDiscTexture,
} from "../../src/three/internal.js";
import { fakeRenderer } from "./fake-three.js";

const shared = fakeRenderer().renderer;

function pool(info?: (id: number) => { voxelSize: number[] }) {
  return {
    device: shared.backend.device as unknown as GPUDevice,
    buffer: { label: "pool" } as unknown as GPUBuffer,
    version: 1,
    layoutVersion: VDB_GPU_LAYOUT_VERSION,
    types: new Set(["float"]) as ReadonlySet<never>,
    ...(info ? { info } : {}),
  };
}

const code = (fn: () => unknown): string | undefined => {
  try {
    fn();
  } catch (e) {
    return e instanceof VDBError ? e.code : `not a VDBError: ${String(e)}`;
  }
  return undefined;
};

describe("VDBLevelSetMaterial (K3.7)", () => {
  it("is a BackSide MeshStandardNodeMaterial with a mask, depth, normal and the trace as nodes", () => {
    const grids = new VDBGridNode(shared, pool(), { profile: "march", helpers: ["levelSet"] });
    const m = new VDBLevelSetMaterial(shared, grids, 2, { color: 0xff0000, roughness: 0.25 });
    expect(m).toBeInstanceOf(MeshStandardNodeMaterial);
    expect(m.isVDBLevelSetMaterial).toBe(true);
    expect(m.side).toBe(THREE.BackSide);
    expect(m.shadowSide).toBe(THREE.BackSide);
    expect(m.color.getHex()).toBe(0xff0000);
    expect(m.roughness).toBe(0.25);
    expect(m.gridId.value).toBe(2);
    expect(m.maskNode).toBe(m.hitMaskNode);
    expect(m.normalNode).toBe(m.normalViewNode);
    expect(m.depthNode).not.toBeNull();
    expect(m.maxSamples.value).toBe(512);
    expect(m.normalStep.value).toBe(1);
    // No pool info: one index voxel per SDF unit.
    expect(m.sdfToIndex.value).toBe(1);
    // K6.6: the entry calls the core's surface functions (the raw pass's WGSL) and compiles the sample cap in.
    expect(m.traceSource).toContain("vdbLevelSetViewRay(uv, projection_inverse, camera_world, orthographic != 0u)");
    expect(m.traceSource).toContain(`vdbLevelSetSurface(grid_id, ray, view_projection, sdf_to_index, u32(clamp(max_samples, 1.0, ${VDB_LEVEL_SET_SAMPLES_CAP}.0)), normal_step)`);
    expect(m.traceSource).not.toMatch(/pnanovdb|vdb_[a-z]|vdbLevelSetHit/);
    expect(m.surfaceNode).toBeDefined();
  });

  it("takes sdfToIndex from the pool's voxel size, or the parameter", () => {
    const grids = new VDBGridNode(shared, pool(() => ({ voxelSize: [0.05, 0.05, 0.05] })), { profile: "march", helpers: ["levelSet"] });
    expect(new VDBLevelSetMaterial(shared, grids, 0).sdfToIndex.value).toBeCloseTo(20, 12);
    expect(new VDBLevelSetMaterial(shared, grids, 0, { sdfToIndex: 4 }).sdfToIndex.value).toBe(4);
    expect(new VDBLevelSetMaterial(shared, grids, 0, { maxSamplesCap: 64 }).traceSource).toContain("clamp(max_samples, 1.0, 64.0)");
  });

  it("rejects a VDBGridNode without march + levelSet, bad ids and bad options", () => {
    const sample = new VDBGridNode(shared, pool(), { profile: "sample" });
    const march = new VDBGridNode(shared, pool(), { profile: "march", helpers: ["hddaCellEnd"] });
    const ok = new VDBGridNode(shared, pool(), { profile: "march", helpers: ["levelSet"] });
    expect(code(() => new VDBLevelSetMaterial(shared, sample, 0))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, march, 0))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, {} as never, 0))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, ok, -1))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, ok, 0, { sdfToIndex: 0 }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(shared, ok, 0, { maxSamplesCap: 0 }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetMaterial(fakeRenderer().renderer, ok, 0))).toBe("VDB_BAD_OPTION");
  });
});

describe("VDBLevelSetPhysicalMaterial (G13a)", () => {
  it("is a BackSide MeshPhysicalNodeMaterial with the standard material's trace setup and physical parameters", () => {
    const grids = new VDBGridNode(shared, pool(() => ({ voxelSize: [0.05, 0.05, 0.05] })), { profile: "march", helpers: ["levelSet"] });
    const p = new VDBLevelSetPhysicalMaterial(shared, grids, 3, { color: 0xffffff, transmission: 1, ior: 1.45, thickness: 0.4, clearcoat: 0.5, maxSamples: 300, normalStep: 0.5 });
    const s = new VDBLevelSetMaterial(shared, grids, 3, { maxSamples: 300, normalStep: 0.5 });
    expect(p).toBeInstanceOf(MeshPhysicalNodeMaterial);
    expect(p.isVDBLevelSetMaterial).toBe(true);
    expect(p.isVDBLevelSetPhysicalMaterial).toBe(true);
    expect(p.name).toBe("VDBLevelSetPhysicalMaterial");
    expect(p.transmission).toBe(1);
    expect(p.ior).toBeCloseTo(1.45, 6);
    expect(p.thickness).toBe(0.4);
    expect(p.clearcoat).toBe(0.5);
    // One trace setup: the same entry, side, uniforms and node wiring as the standard material.
    expect(p.traceSource).toBe(s.traceSource);
    expect(p.side).toBe(THREE.BackSide);
    expect(p.shadowSide).toBe(THREE.BackSide);
    expect(p.gridId.value).toBe(3);
    expect(p.sdfToIndex.value).toBeCloseTo(20, 12);
    expect(p.maxSamples.value).toBe(300);
    expect(p.normalStep.value).toBe(0.5);
    expect(p.maskNode).toBe(p.hitMaskNode);
    expect(p.normalNode).toBe(p.normalViewNode);
    expect(p.receivedShadowPositionNode).toBe(p.hitWorldNode);
    expect(p.depthNode).not.toBeNull();
    expect(p.vertexNode).not.toBeNull();
    // The clearcoat shades with the trace's normal, not the box's.
    expect((p as unknown as { clearcoatNormalNode: unknown }).clearcoatNormalNode).toBe(p.normalViewNode);
  });

  it("its lighting model is three's physical model with the material's features, refracting from the hit", () => {
    const grids = new VDBGridNode(shared, pool(), { profile: "march", helpers: ["levelSet"] });
    const glass = new VDBLevelSetPhysicalMaterial(shared, grids, 0, { transmission: 1, clearcoat: 1, dispersion: 0.2 });
    const model = glass.setupLightingModel() as VDBLevelSetPhysicalLightingModel & Record<string, boolean>;
    expect(model).toBeInstanceOf(VDBLevelSetPhysicalLightingModel);
    expect(model).toBeInstanceOf(PhysicalLightingModel);
    expect(model.hitWorld).toBe(glass.hitWorldNode);
    expect(model.transmission).toBe(true);
    expect(model.clearcoat).toBe(true);
    expect(model.dispersion).toBe(true);
    expect(model.sheen).toBe(false);
    const plain = new VDBLevelSetPhysicalMaterial(shared, grids, 0).setupLightingModel() as unknown as Record<string, boolean>;
    expect(plain.transmission).toBe(false);
    expect(plain.clearcoat).toBe(false);
  });

  it("validates as the standard material, naming itself", () => {
    const sample = new VDBGridNode(shared, pool(), { profile: "sample" });
    const ok = new VDBGridNode(shared, pool(), { profile: "march", helpers: ["levelSet"] });
    expect(code(() => new VDBLevelSetPhysicalMaterial(shared, sample, 0))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetPhysicalMaterial(shared, ok, 1.5))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetPhysicalMaterial(shared, ok, 0, { maxSamplesCap: 0 }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBLevelSetPhysicalMaterial(fakeRenderer().renderer, ok, 0))).toBe("VDB_BAD_OPTION");
    expect(() => new VDBLevelSetPhysicalMaterial(shared, sample, 0)).toThrow(/\]: VDBLevelSetPhysicalMaterial: /);
  });
});

describe("VDBPoints (K3.7)", () => {
  const positions = Float32Array.from([0, 0, 0, 1, 2, 3, -1, 0, 4]);

  it("draws count instanced sprites with the disc, Cd times the colour, half the spacing as diameter", () => {
    const colors = Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const p = new VDBPoints({ positions, colors, spacing: 0.4 }, { color: 0x808080 });
    expect(p).toBeInstanceOf(THREE.Object3D);
    expect(p.isVDBPoints).toBe(true);
    expect(p.material).toBeInstanceOf(PointsNodeMaterial);
    expect(p.material.alphaMap).toBe(vdbPointDiscTexture());
    expect(p.material.alphaTest).toBe(0.5);
    expect((p as unknown as { count: number }).count).toBe(3);
    expect(p.pointCount).toBe(3);
    expect(p.positionAttribute.array).toBe(positions);
    expect(p.colorAttribute!.array).toBe(colors);
    expect(p.diameter).toBeCloseTo(0.2, 12);
    expect(p.frustumCulled).toBe(false);
    expect(new VDBPoints({ positions }, { size: 3 }).diameter).toBe(3);
    expect(new VDBPoints({ positions, count: 2 }).pointCount).toBe(2);
    expect(new VDBPoints({ positions }).colorAttribute).toBeNull();
    // K6.6: the clip position comes from the core sprite WGSL through the vertex node.
    expect((p.material as unknown as { vertexNode: unknown }).vertexNode).toBeTruthy();
    expect((p.material as unknown as { sizeNode: unknown }).sizeNode).toBeNull();
  });

  it("K6.6: the core's vdbPointPixelScale (projection[5]) equals the pixel scale VDBPoints sets for a perspective camera", () => {
    const p = new VDBPoints({ positions, spacing: 2 });
    p.scale.set(1, 3, 2);
    p.updateMatrixWorld(true);
    const camera = new THREE.PerspectiveCamera(47, 1.7, 0.1, 100);
    camera.zoom = 1.6;
    camera.updateProjectionMatrix();
    p.updatePixelScale({ getSize: (v) => v.set(640, 360), getRenderTarget: () => null }, camera);
    const core = vdbPointPixelScale(p.diameter, camera.projectionMatrix.elements, 360, p.matrixWorld.elements);
    expect(core / p.pixelScale.value).toBeCloseTo(1, 12);
  });

  it("sizes in pixels as his getAttenuatedPointSize, from the render target height when one is set", () => {
    const p = new VDBPoints({ positions, spacing: 2 });
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
    p.updateMatrixWorld(true);
    const perUnit = vdbAttenuatedPointSize(p, camera, 1);
    expect(perUnit).toBeCloseTo(1 / Math.tan(Math.PI / 6), 12);
    p.updatePixelScale({ getSize: (v) => v.set(640, 480), getRenderTarget: () => ({ height: 200 }) }, camera);
    expect(p.pixelScale.value).toBeCloseTo(perUnit * 1 * 200 * 0.5, 9);
    p.updatePixelScale({ getSize: (v) => v.set(640, 480), getRenderTarget: () => null }, camera);
    expect(p.pixelScale.value).toBeCloseTo(perUnit * 1 * 480 * 0.5, 9);
  });

  it("rejects malformed data", () => {
    expect(code(() => new VDBPoints({ positions: new Float32Array(4) }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBPoints({ positions, count: 4 }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBPoints({ positions, colors: new Float32Array(3) }))).toBe("VDB_BAD_OPTION");
    expect(code(() => new VDBPoints({} as never))).toBe("VDB_BAD_OPTION");
  });
});
