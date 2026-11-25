/**
 * K6.5: the CPU halves of the core volume shadow maps and the emission colour, in Node (the kernels run against
 * their twins in e2e/k6-core.spec.ts): the engine-free cascade fit equals the (deleted, K6.8: frozen) TSL port's, the cube face
 * table round-trips, the spot camera from a view-projection, the LUT rows equal his blackbody read, the hook checks,
 * and the shadow budget (oldest first; a waiting map keeps its drawn view; lights not in the frame take no budget).
 */

import { readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  vdbFrameBlackbodyColor,
  vdbFrameBlackbodyTable,
  vdbFrameCubeFaceDirection,
  vdbFrameCubeFaceUv,
  vdbFrameEmissionLutColor,
  vdbFrameEmissionLutRow,
  vdbFrameShadowFitCascades,
  vdbFrameShadowViewCamera,
} from "../../../src/tools/internal.js";
import { VDBFrameShadowMaps, vdbEmissionHookProblems, vdbFrameFitShadowMapSize, VDBRetireQueue, type VDBFrameLight } from "../../../src/core/frame/index.js";

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i]! - b[i]!), `[${i}] ${a[i]} vs ${b[i]}`).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i]!)));
};

describe("K6.5 shadow twins", () => {
  it("vdbFrameShadowFitCascades equals the TSL port's fitCascades (three's maths) on random cameras and lights (K6.8: frozen)", () => {
    // K6.8: the TSL port (three/frame/volume-shadows.ts fitCascades) is deleted; its outputs on the former test's random
    // cameras and lights are frozen in fixtures/three-port-golden.json (dumped from the pre-deletion tree)
    const golden = JSON.parse(readFileSync(new URL("./fixtures/three-port-golden.json", import.meta.url), "utf8")) as {
      cascades: { input: Parameters<typeof vdbFrameShadowFitCascades>[0]; three: { count: number; breaks: number[]; forward: number[]; cascades: { viewProjection: number[]; origin: number[]; far: number }[] } }[];
    };
    expect(golden.cascades.length).toBe(24);
    for (const { input, three } of golden.cascades) {
      const ours = vdbFrameShadowFitCascades(input);
      expect(ours.count).toBe(three.count);
      close(ours.breaks, three.breaks, 1e-12);
      close(ours.forward, three.forward, 1e-9);
      for (let i = 0; i < input.count; i++) {
        const c = three.cascades[i]!;
        close(ours.cascades[i]!.viewProjection, c.viewProjection, 1e-9);
        close(ours.cascades[i]!.origin, c.origin, 1e-9);
        expect(Math.abs(ours.cascades[i]!.far - c.far)).toBeLessThan(1e-9 * c.far);
      }
    }
  });

  it("the cube face table round-trips (a direction's face and uv give it back)", () => {
    let seed = 3;
    const rand = () => ((seed = (seed * 48271) % 2147483647) / 2147483647) * 2 - 1;
    for (let i = 0; i < 500; i++) {
      const d: [number, number, number] = [rand(), rand(), rand()];
      const f = vdbFrameCubeFaceUv(d);
      const back = vdbFrameCubeFaceDirection(f.face, f.s * 2 - 1, f.t * 2 - 1);
      const m = Math.max(Math.abs(d[0]), Math.abs(d[1]), Math.abs(d[2]));
      close(back, [d[0] / m, d[1] / m, d[2] / m], 1e-12);
      expect(f.s).toBeGreaterThanOrEqual(0);
      expect(f.t).toBeLessThanOrEqual(1);
    }
  });

  it("vdbFrameShadowViewCamera recovers the drawing camera of a perspective and an orthographic view-projection", () => {
    const spot = new THREE.PerspectiveCamera(50, 1, 10, 600);
    spot.coordinateSystem = THREE.WebGPUCoordinateSystem;
    spot.position.set(-60, 260, -110);
    spot.lookAt(30, 0, 60);
    spot.updateProjectionMatrix();
    spot.updateMatrixWorld(true);
    const vp = new THREE.Matrix4().multiplyMatrices(spot.projectionMatrix, spot.matrixWorldInverse);
    const cam = vdbFrameShadowViewCamera(vp.elements, spot.near, false);
    close(cam.position, spot.position.toArray(), 1e-9);
    close(cam.forward, spot.getWorldDirection(new THREE.Vector3()).toArray(), 1e-9);
    const ortho = new THREE.OrthographicCamera(-30, 30, 30, -30, 2, 300);
    ortho.coordinateSystem = THREE.WebGPUCoordinateSystem;
    ortho.position.set(5, 80, 7);
    ortho.lookAt(0, 0, 0);
    ortho.updateProjectionMatrix();
    ortho.updateMatrixWorld(true);
    const ovp = new THREE.Matrix4().multiplyMatrices(ortho.projectionMatrix, ortho.matrixWorldInverse);
    const ocam = vdbFrameShadowViewCamera(ovp.elements, ortho.near, true);
    close(ocam.position, ortho.position.toArray(), 1e-9);
    close(ocam.forward, ortho.getWorldDirection(new THREE.Vector3()).toArray(), 1e-9);
  });
});

describe("K6.5 emission colour data and hooks", () => {
  it("a LUT row (the table pre-blended at the soot exponent) reads his blackbody colour", () => {
    const table = vdbFrameBlackbodyTable();
    for (const exponent of [0, 0.4, 1.39, 2.2, 3]) {
      const row = vdbFrameEmissionLutRow(exponent, table);
      for (const t of [300, 500, 812, 1500, 3000, 6500, 12000, 39000, 50000]) close(vdbFrameEmissionLutColor(row, t), vdbFrameBlackbodyColor(t, exponent), 1e-6);
    }
  });

  it("hook checks: names, reserved words, back-ticks, bindings and the entry function, each with its line", () => {
    const ok = "// a hook\nconst vdbUserScale: f32 = 2.0;\nfn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f {\n  let k = t * vdbUserScale;\n  return vec3f(k);\n}\n";
    expect(vdbEmissionHookProblems(ok)).toBeNull();
    expect(vdbEmissionHookProblems("fn emissionColor(t: f32, temperature: f32) -> vec3f { return vec3f(t); }")).toMatch(/line 1: top-level name 'emissionColor'/);
    expect(vdbEmissionHookProblems(`${ok}fn helper() {}\n`)).toMatch(/line 7: top-level name 'helper'/);
    expect(vdbEmissionHookProblems(`${ok}// \`x\`\n`)).toMatch(/back-ticks/);
    expect(vdbEmissionHookProblems(`${ok}@group(0) @binding(0) var<uniform> vdbUserU: vec4f;\n`)).toMatch(/line 7: '@group'/);
    expect(vdbEmissionHookProblems("fn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f {\n  let filter = t;\n  return vec3f(filter);\n}")).toMatch(/line 2: 'filter' is a reserved word/);
    expect(vdbEmissionHookProblems("fn vdbUserOther(t: f32) -> f32 { return t; }")).toMatch(/must define fn vdbUserEmissionColor/);
  });
});

describe("K6.5 volume shadow budget (core; his c13f2d5 and 62a6e3e by construction)", () => {
  const fakeDevice = () => {
    const writes: { buffer: { label: string }; data: Float32Array }[] = [];
    const device = {
      createTexture: (d: GPUTextureDescriptor) => ({ ...d, destroyed: false, createView: () => ({}), destroy() {} }),
      createBuffer: (d: GPUBufferDescriptor) => ({ ...d, destroy() {} }),
      queue: { writeBuffer: (buffer: { label: string }, _o: number, data: ArrayBuffer) => writes.push({ buffer, data: new Float32Array(data.slice(0)) }) },
    };
    return { device: device as unknown as GPUDevice, writes };
  };
  const spot = (id: string, x: number, shadow = true): VDBFrameLight => ({
    id,
    kind: "spot",
    position: [x, 50, 0],
    direction: [0, -1, 0.2],
    color: [1, 1, 1],
    distance: 0,
    decay: 2,
    cosAngle: Math.cos(0.5),
    cosPenumbra: Math.cos(0.5),
    ...(shadow ? { shadow: { size: 64 } } : {}),
  });
  const inputs = (lights: VDBFrameLight[], castVersion = 1) => ({
    lights,
    castShadows: true,
    coarse: false,
    steps: 16,
    budget: 1,
    cascades: 1,
    castShadowDistance: 1000,
    camera: { view: new THREE.Matrix4().elements, cameraWorld: new THREE.Matrix4().elements, projectionInverse: new THREE.Matrix4().elements, near: 1, far: 100, orthographic: false },
    grid: { min: [-10, 0, -10] as [number, number, number], size: [20, 20, 20] as [number, number, number] },
    castVersion,
  });

  it("oldest first within the budget; a waiting map keeps the view it was drawn with; then it draws", () => {
    const { device } = fakeDevice();
    const maps = new VDBFrameShadowMaps(device, "t", new VDBRetireQueue());
    const a = spot("A", 0);
    const b = spot("B", 5);
    maps.update(inputs([a, b]));
    expect(maps.stats).toEqual({ drawn: 1, waiting: 1 });
    expect(maps.maps.get("A")!.enabled).toBe(true);
    expect(maps.maps.get("B")!.enabled).toBe(false); // never drawn: receivers read 1
    maps.update(inputs([a, b]));
    expect(maps.stats).toEqual({ drawn: 1, waiting: 0 });
    const drawnA = Float32Array.from(maps.maps.get("A")!.drawn.viewProjection[0]!);
    // both move: A drew first, so it is the older piece and draws; B waits and keeps the view it was drawn with
    const drawnB = Float32Array.from(maps.maps.get("B")!.drawn.viewProjection[0]!);
    const a2 = spot("A", 1);
    const b2 = spot("B", 6);
    maps.update(inputs([a2, b2]));
    expect(maps.stats).toEqual({ drawn: 1, waiting: 1 });
    expect([...maps.maps.get("A")!.drawn.viewProjection[0]!]).not.toEqual([...drawnA]);
    expect([...maps.maps.get("B")!.drawn.viewProjection[0]!]).toEqual([...drawnB]);
    expect(maps.maps.get("B")!.enabled).toBe(true);
    maps.update(inputs([a2, b2]));
    expect(maps.stats).toEqual({ drawn: 1, waiting: 0 });
    expect([...maps.maps.get("B")!.drawn.viewProjection[0]!]).not.toEqual([...drawnB]);
    // a new grid (castVersion) redraws both pieces, one per update
    maps.update(inputs([a2, b2], 2));
    expect(maps.stats).toEqual({ drawn: 1, waiting: 1 });
    maps.dispose();
  });

  it("fail-before (c13f2d5): a light not in this frame takes no budget and its map is freed; a light without a request has none", () => {
    const { device } = fakeDevice();
    const maps = new VDBFrameShadowMaps(device, "t", new VDBRetireQueue());
    maps.update(inputs([spot("B", 5)]));
    maps.update(inputs([spot("A", 0), spot("C", 3, false)]));
    expect([...maps.maps.keys()]).toEqual(["A"]);
    expect(maps.stats).toEqual({ drawn: 1, waiting: 0 });
    // without casters (no grid) nothing draws and the maps read 1
    maps.update({ ...inputs([spot("A", 0)]), grid: null });
    expect(maps.stats).toEqual({ drawn: 0, waiting: 0 });
    expect(maps.maps.get("A")!.enabled).toBe(false);
    // castShadows off frees every map
    maps.update({ ...inputs([spot("A", 0)]), castShadows: false });
    expect(maps.maps.size).toBe(0);
    maps.dispose();
  });

  it("a point light costs 6 and still draws first when oldest; a directional light draws its cascade layout in one piece", () => {
    const { device } = fakeDevice();
    const maps = new VDBFrameShadowMaps(device, "t", new VDBRetireQueue());
    const point: VDBFrameLight = { id: "P", kind: "point", position: [0, 40, 0], color: [1, 1, 1], distance: 0, decay: 2, shadow: { size: 32 } };
    const sun: VDBFrameLight = { id: "S", kind: "directional", direction: [0.3, -1, 0.2], color: [1, 1, 1], shadow: { size: 32 } };
    const camera = new THREE.PerspectiveCamera(50, 1, 1, 500);
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.position.set(0, 30, 80);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    const cam = { view: camera.matrixWorldInverse.elements, cameraWorld: camera.matrixWorld.elements, projectionInverse: camera.projectionMatrixInverse.elements, near: 1, far: 500, orthographic: false };
    const dispatches = maps.update({ ...inputs([point, sun]), camera: cam, cascades: 3, budget: 3 });
    // the point map (cost 6) is first and always draws; the 3-cascade layout (cost 3) waits for the spent budget
    expect(maps.stats).toEqual({ drawn: 1, waiting: 1 });
    expect(dispatches).toBe(1);
    expect(maps.update({ ...inputs([point, sun]), camera: cam, cascades: 3, budget: 3 })).toBe(3);
    const s = maps.maps.get("S")!;
    expect(s.drawn.cascadeCount).toBe(3);
    expect(s.layers).toBe(4);
    expect(maps.maps.get("P")!.layers).toBe(6);
    maps.dispose();
  });

  it("K6.10a: the map size clamps to maxTextureDimension2D and to a staging buffer within maxBufferSize (a point light's 6 faces count), warned once", () => {
    // fail-before: the size went to createTexture unclamped (a 4096 point map is 805 MB; > 8192 is an invalid texture)
    const made: GPUTextureDescriptor[] = [];
    const { device } = fakeDevice();
    const limited = Object.assign(Object.create(device) as object, {
      limits: { maxTextureDimension2D: 1024, maxBufferSize: 6 * 1024 * 1024 },
      createTexture: (d: GPUTextureDescriptor) => (made.push(d), { ...d, createView: () => ({}), destroy() {} }),
    }) as unknown as GPUDevice;
    const warnings: string[] = [];
    const warned = new Set<string>();
    const maps = new VDBFrameShadowMaps(limited, "t", new VDBRetireQueue(), (key, message) => {
      if (!warned.has(key)) (warned.add(key), warnings.push(message));
    });
    const point: VDBFrameLight = { id: "P", kind: "point", position: [0, 40, 0], color: [1, 1, 1], distance: 0, decay: 2, shadow: { size: 4096 } };
    const big = spot("A", 0);
    (big as { shadow: { size: number } }).shadow.size = 4096;
    const small = spot("B", 5);
    for (let i = 0; i < 3; i++) maps.update(inputs([point, big, small]));
    const side = (d: GPUTextureDescriptor) => (d.size as number[])[0]!;
    const bytes = (d: GPUTextureDescriptor) => Math.ceil((side(d) * 8) / 256) * 256 * side(d) * (d.size as number[])[2]!;
    // point: 6 layers within 6 MiB (rows padded to 256 bytes)
    expect(maps.maps.get("P")!.size).toBe(vdbFrameFitShadowMapSize(4096, 6, { maxTextureDimension2D: 1024, maxBufferSize: 6 * 1024 * 1024 }).size);
    expect(maps.maps.get("P")!.size).toBeLessThan(1024);
    // spot: one layer, the dimension limit binds first (1024² · 8 = 8 MiB > 6 MiB, so the bytes bind too)
    expect(maps.maps.get("A")!.size).toBeLessThanOrEqual(1024);
    expect(maps.maps.get("B")!.size).toBe(64);
    for (const d of made) {
      expect(side(d)).toBeLessThanOrEqual(1024);
      expect(bytes(d)).toBeLessThanOrEqual(6 * 1024 * 1024);
    }
    // three updates, one texture per light, one warning per clamped light
    expect(made.length).toBe(3);
    expect(warnings.length).toBe(2);
    expect(warnings.join("\n")).toMatch(/point light \(6 rgba16float layers, 768 MiB\).*maxBufferSize 6 MiB.*clamped to \d+/);
    maps.dispose();
  });
});

describe("K6.10a vdbFrameFitShadowMapSize", () => {
  it("is the largest side whose padded staging buffer fits, and never above maxTextureDimension2D", () => {
    const stage = (s: number, layers: number) => Math.ceil((s * 8) / 256) * 256 * s * layers;
    // WebGPU defaults: a 4096 point map (805 MB) clamps; spot at 4096 (128 MiB) fits
    const point = vdbFrameFitShadowMapSize(4096, 6);
    expect(point.reason).toBe("bytes");
    expect(stage(point.size, 6)).toBeLessThanOrEqual(268435456);
    expect(stage(point.size + 1, 6)).toBeGreaterThan(268435456);
    expect(vdbFrameFitShadowMapSize(4096, 1)).toEqual({ size: 4096, reason: null });
    expect(vdbFrameFitShadowMapSize(512, 4)).toEqual({ size: 512, reason: null });
    // above maxTextureDimension2D
    expect(vdbFrameFitShadowMapSize(9000, 1)).toEqual({ size: 5792, reason: "dimension" });
    expect(vdbFrameFitShadowMapSize(9000, 1, { maxBufferSize: 2 ** 32 })).toEqual({ size: 8192, reason: "dimension" });
    expect(vdbFrameFitShadowMapSize(0.2, 1)).toEqual({ size: 1, reason: null });
  });
});
