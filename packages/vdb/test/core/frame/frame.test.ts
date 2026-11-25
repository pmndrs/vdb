import { readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { vdbDenseSource } from "../../../src/core/source/dense.js";
import {
  ENV_GROUP0_LAYOUT,
  GROUP0_LAYOUT,
  type VDBFrameInternalSettings,
  VDB_FRAME_GROUP,
  VDB_FRAME_LIGHTS,
  VDB_FRAME_UNIFORMS,
  VDB_FRAME_VOLUME,
  VDB_FRAME_VOLUME_BYTES,
  VDBRetireQueue,
  VDBUniformWriter,
  vdbFrameChunkProbeSpec,
  vdbFrameCompositeSpec,
  vdbFrameDefaultSettings,
  vdbFrameEnvBlurSpec,
  vdbFrameEnvPrefilterSpec,
  vdbFrameEnvReduceSpec,
  vdbFrameEnvironmentGridSpec,
  vdbFrameGatherSpec,
  vdbFrameGridDensitySpec,
  vdbFrameTransmittanceSpec,
  vdbFrameGridMipSpec,
  vdbFrameVolumeShadowSpec,
  vdbFrameEmissionDirectionSpec,
  vdbFrameEmissionReduceSpec,
  vdbFrameEmissionProbeSpec,
  vdbVolumeShadowWgsl,
  vdbEmissionProbeWgsl,
  vdbFrameIntegrateWgsl,
  vdbFrameResolveSpec,
  vdbUniformStruct,
  type VDBComputeSpec,
  type VDBRenderSpec,
} from "../../../src/core/frame/index.js";
import { VDBFrameRenderer } from "../../../src/three/frame/renderer.js";
import {
  vdbFrameClipCamera,
  vdbFrameClipVolume,
  vdbFrameFroxelArea,
  vdbFrameFroxelRect,
  vdbFrameFroxelSize,
  vdbFrameVolumeSlices,
  vdbFroxelHeldSize,
} from "../../../src/tools/frame/view.js";
import { wgslNameProblems } from "../wgsl-names.js";
// @ts-expect-error: repo-level ESM helper without type declarations
import { wgslSizes } from "../../../../../scripts/lib/wgsl-variants.mjs";

/** A device stub that is enough for vdbDenseSource (textures, a sampler, writeTexture). */
const fakeDevice = () =>
  ({
    createTexture: (d: GPUTextureDescriptor) => ({ ...d, createView: () => ({}), destroy() {} }),
    createSampler: () => ({}),
    queue: { writeTexture() {} },
  }) as unknown as GPUDevice;

function denseVolume(o: { albedo?: boolean; mask?: boolean; emission?: boolean; mips?: boolean } = {}) {
  const tex = (k: 1 | 4) => ({
    data: new Uint8Array(8 * 8 * 8 * k).fill(100),
    dimensions: [8, 8, 8] as [number, number, number],
    format: k === 1 ? ("r8unorm" as const) : ("rgba8unorm" as const),
    texelBytes: k,
    mip: o.mips ? { data: new Uint8Array(8 * 4 * 4 * k), dimensions: [8, 4, 4] as [number, number, number], levels: 3 } : null,
  });
  return {
    kind: "dense",
    normalize: "clamp01",
    lod: 0,
    dimensions: [8, 8, 8],
    bbox: null,
    density: tex(1),
    mask: o.mask ? tex(1) : null,
    albedo: o.albedo ? tex(4) : null,
    emission: o.emission ? tex(1) : null,
    densityStats: { sum: 1, count: 1 },
    densityStatsSource: "texels",
    meanDensity: 0.4,
    maxDensity: 0.4,
    activeTexels: 512,
    gridNames: { density: "density" },
  } as never;
}

const sources = () => [vdbDenseSource(fakeDevice(), denseVolume({ albedo: true, mask: true, emission: true, mips: true })), vdbDenseSource(fakeDevice(), denseVolume())];

/** A user emission colour hook (K6.5): every top-level name starts with vdbUser. */
const TEST_HOOK = "fn vdbUserEmissionColor(t: f32, temperature: f32) -> vec3f {\n  return vec3f(t, 0.5 * t, 0.25) * clamp(temperature / 3000.0, 0.0, 1.0);\n}\n";

/** Every K6.1 pipeline (the key matrix): gather × mix × multi × source layout, resolve × temporal, composite × depth × sRGB. */
function allSpecs(): { compute: VDBComputeSpec[]; render: VDBRenderSpec[] } {
  const compute: VDBComputeSpec[] = [];
  for (const s of sources()) {
    for (const mix of ["mixture", "over"] as const) for (const multi of [false, true]) compute.push(vdbFrameGatherSpec(s, { mix, multi }));
    // K6.2 'frame' lighting: the fused gather and the shadow grid density per source layout
    for (const multi of [false, true]) compute.push(vdbFrameGatherSpec(s, { mix: "mixture", multi, lighting: "frame" }), vdbFrameGridDensitySpec(s, { multi }));
    // K6.4 spectral: both gathers and the spectral shadow grid
    for (const multi of [false, true]) {
      for (const mix of ["mixture", "over"] as const) compute.push(vdbFrameGatherSpec(s, { mix, multi, spectral: true }));
      compute.push(vdbFrameGatherSpec(s, { mix: "mixture", multi, lighting: "frame", spectral: true }), vdbFrameGridDensitySpec(s, { multi, spectral: true }));
    }
    // K6.5: the emission direction pass (sources with emission), with and without a user colour hook
    if (s.channels.emission) {
      compute.push(vdbFrameEmissionDirectionSpec(s, {}), vdbFrameEmissionDirectionSpec(s, { hook: TEST_HOOK }));
      compute.push(vdbFrameGatherSpec(s, { mix: "mixture", multi: false, hook: TEST_HOOK }), vdbFrameGatherSpec(s, { mix: "mixture", multi: false, lighting: "frame", hook: TEST_HOOK }));
    }
  }
  compute.push(vdbFrameTransmittanceSpec(), vdbFrameResolveSpec({ temporal: false }), vdbFrameResolveSpec({ temporal: true }));
  // K6.3: the transmittance's scattered-light variants and the 'mlp' feature mips
  compute.push(vdbFrameTransmittanceSpec({ scattering: "mlp" }), vdbFrameTransmittanceSpec({ scattering: "multi-scale" }), vdbFrameGridMipSpec({ first: true }), vdbFrameGridMipSpec({ first: false }));
  // K6.2b: the environment kernels (source-free)
  compute.push(vdbFrameEnvPrefilterSpec("equirect"), vdbFrameEnvPrefilterSpec("cube"), vdbFrameEnvBlurSpec(), vdbFrameEnvReduceSpec(), vdbFrameEnvironmentGridSpec());
  compute.push(vdbFrameVolumeShadowSpec(), vdbFrameEmissionReduceSpec(), vdbFrameEmissionProbeSpec());
  // K6.4: the spectral transmittance, environment grid, shadow map and resolve (both alpha prefix formats)
  compute.push(vdbFrameTransmittanceSpec({ spectral: true }), vdbFrameEnvironmentGridSpec({ spectral: true }), vdbFrameVolumeShadowSpec({ spectral: true }));
  for (const temporal of [false, true]) for (const alphaFormat of ["rgba16float", "rg16float"] as const) compute.push(vdbFrameResolveSpec({ temporal, spectral: true, alphaFormat }));
  const render: VDBRenderSpec[] = [];
  for (const depth of ["depth", "r32float"] as const) for (const srgb of [false, true]) for (const spectral of [false, true]) render.push(vdbFrameCompositeSpec({ colorFormat: "rgba16float", depth, srgb, spectral }));
  return { compute, render };
}

describe("K6.1 uniform structs", () => {
  it("lays fields out by WGSL's uniform rules", () => {
    const s = vdbUniformStruct("T", [
      { name: "a", type: "f32" },
      { name: "b", type: "vec3f" },
      { name: "c", type: "vec2f" },
      { name: "d", type: "mat4x4f" },
      { name: "e", type: "u32" },
      { name: "f", type: "vec4f", count: 3 },
    ]);
    const at = (n: string) => s.fields.get(n)!.offset;
    expect([at("a"), at("b"), at("c"), at("d"), at("e"), at("f")]).toEqual([0, 16, 32, 48, 112, 128]);
    expect(s.size).toBe(176);
    expect(s.wgsl).toContain("f: array<vec4f, 3>,");
  });

  it("the frame structs: the volume record is 512 B, the group record fits its 256 B stride, offsets follow the WGSL order", () => {
    expect(VDB_FRAME_VOLUME.size).toBe(VDB_FRAME_VOLUME_BYTES);
    expect(VDB_FRAME_GROUP.size).toBeLessThanOrEqual(256);
    for (const s of [VDB_FRAME_UNIFORMS, VDB_FRAME_LIGHTS, VDB_FRAME_VOLUME, VDB_FRAME_GROUP]) {
      const names = [...s.wgsl.matchAll(/^ {2}(\w+):/gm)].map((m) => m[1]);
      expect(names.slice(0, s.fields.size)).toEqual([...s.fields.keys()]);
      const offsets = [...s.fields.values()].map((f) => f.offset);
      expect(offsets).toEqual([...offsets].sort((a, b) => a - b));
      for (const f of s.fields.values()) expect(f.offset % (f.type.startsWith("vec3") || f.type.startsWith("vec4") || f.type.startsWith("mat") ? 16 : 4)).toBe(0);
    }
    expect(VDB_FRAME_UNIFORMS.fields.get("grid")!.offset).toBe(256);
    expect(VDB_FRAME_LIGHTS.fields.get("position")!.offset).toBe(32 + 9 * 16 + 3 * 4 * 16);
  });

  it("the packer writes floats and integers at the struct offsets", () => {
    const w = new VDBUniformWriter(2 * VDB_FRAME_VOLUME_BYTES);
    w.set(VDB_FRAME_VOLUME, VDB_FRAME_VOLUME_BYTES, "slices", [3, 4, 5, 6]);
    w.set(VDB_FRAME_VOLUME, VDB_FRAME_VOLUME_BYTES, "density", [0.5, 0.25, 2, 1]);
    const slices = (VDB_FRAME_VOLUME_BYTES + VDB_FRAME_VOLUME.fields.get("slices")!.offset) / 4;
    const density = (VDB_FRAME_VOLUME_BYTES + VDB_FRAME_VOLUME.fields.get("density")!.offset) / 4;
    expect([...w.u32.subarray(slices, slices + 4)]).toEqual([3, 4, 5, 6]);
    expect([...w.f32.subarray(density, density + 4)]).toEqual([0.5, 0.25, 2, 1]);
  });
});

describe("K6.1 composed WGSL", () => {
  const { compute, render } = allSpecs();
  const texts = [...compute, ...render, vdbFrameChunkProbeSpec()].map((s) => [s.label, s.code] as const);
  texts.push(["vdbFrameIntegrateWgsl", vdbFrameIntegrateWgsl()], ["vdbFrameIntegrateWgsl spectral", vdbFrameIntegrateWgsl({ spectral: true })], ["vdbVolumeShadowWgsl", vdbVolumeShadowWgsl()], ["vdbEmissionProbeWgsl", vdbEmissionProbeWgsl()]);

  it.each(texts)("%s: no reserved words, no function/variable clashes, no back-ticks, no duplicate declarations", (_, code) => {
    expect(wgslNameProblems(code)).toEqual([]);
    expect(code).not.toContain("`");
    const fns = [...code.matchAll(/\bfn\s+(\w+)/g)].map((m) => m[1]);
    expect(fns.filter((f, i) => fns.indexOf(f) !== i)).toEqual([]);
    const structs = [...code.matchAll(/\bstruct\s+(\w+)/g)].map((m) => m[1]);
    expect(structs.filter((f, i) => structs.indexOf(f) !== i)).toEqual([]);
    for (const word of ["active", "set", "filter", "target"]) expect(code).not.toMatch(new RegExp(`\\b(?:let|var|fn)\\s+${word}\\b`));
  });

  it("the same options give the same bytes (fixed chunk order)", () => {
    const again = allSpecs();
    expect(again.compute.map((s) => s.code)).toEqual(compute.map((s) => s.code));
  });
});

describe("K6.1 device limits (design §1)", () => {
  const { compute, render } = allSpecs();
  it.each([...compute, ...render].map((s) => [s.label, s] as const))("%s fits the core limits", (_, spec) => {
    expect(spec.groups.length).toBeLessThanOrEqual(3);
    const all = spec.groups.flatMap((g) => [...g.entries]);
    // storage textures (default limit 4): grey kernels write at most 2; K6.4's spectral ones 3 (the gather's scatter,
    // scatterAlpha and partials; the spectral grid's two stacks and partials; the resolve's prefixes and stacked history)
    expect(all.filter((e) => e.storageTexture).length).toBeLessThanOrEqual(/spectral/.test(spec.label) ? 3 : 2);
    // Sampled textures: WebGPU's default (and compat's) maxSampledTexturesPerShaderStage is 16. K6.5b's 4 occluder
    // slots raise the worst kernel (the 'frame' gather of a dense source with mask, albedo and emission and their mips:
    // 8 source + transmittance + environment grid + emission LUT + 4 occluders) to 15; the design's own cap was 12
    // before the occluders (design §3.2).
    expect(all.filter((e) => e.texture).length).toBeLessThanOrEqual(16);
    expect(all.filter((e) => e.buffer?.type === "storage" || e.buffer?.type === "read-only-storage").length).toBeLessThanOrEqual(1);
    // Uniform buffers (default limit 12): frame, lights, environment, volumes, group, and K6.3's Lorenz-Mie table (16 KiB
    // of binary16 in group 0, which keeps the sampled textures at 15); the 'multi-scale' / 'mlp' transmittance adds its
    // group-2 uniform but has no volumes or group record
    expect(all.filter((e) => e.buffer?.type === "uniform").length).toBeLessThanOrEqual(6);
    expect(all.filter((e) => e.sampler).length).toBeLessThanOrEqual(16);
    // the environment prefilter / blur / reduce kernels have their own group 0 (their parameters; K6.2b)
    expect([GROUP0_LAYOUT, ENV_GROUP0_LAYOUT]).toContain(spec.groups[0]);
    if (!/environment (prefilter|blur|reduce)/.test(spec.label)) expect(spec.groups[0]).toBe(GROUP0_LAYOUT);
    for (const g of spec.groups) {
      const bindings = [...g.entries].map((e) => e.binding).sort((a, b) => a - b);
      expect(new Set(bindings).size).toBe(bindings.length);
    }
    for (const m of spec.code.matchAll(/@workgroup_size\((\d+),\s*(\d+),\s*(\d+)\)/g)) expect(Number(m[1]) * Number(m[2]) * Number(m[3])).toBe(64);
    // every declared @group is a layout group (no gaps, nothing past the layouts)
    const groups = new Set([...spec.code.matchAll(/@group\((\d+)\)/g)].map((m) => Number(m[1])));
    for (const g of groups) expect(g).toBeLessThan(spec.groups.length);
  });
});

describe("K6.5b sampled-texture headroom", () => {
  it("the worst kernel samples 15 textures (16 is the WebGPU default limit): one left", () => {
    const { compute } = allSpecs();
    const counts = compute.map((s) => [s.label, s.groups.flatMap((g) => [...g.entries]).filter((e) => e.texture).length] as const);
    const worst = Math.max(...counts.map(([, n]) => n));
    expect(worst).toBe(15);
    expect(counts.filter(([, n]) => n === worst).every(([label]) => label.startsWith("vdb frame gather (frame lighting"))).toBe(true);
  });
});

describe("K6.1 retire queue", () => {
  it("destroys a retired object two updates later; drain destroys the rest", () => {
    const q = new VDBRetireQueue();
    const made = Array.from({ length: 3 }, () => ({ destroyed: 0, destroy() {
      this.destroyed++;
    } }));
    q.retire(made[0]);
    q.advance();
    expect(made[0]!.destroyed).toBe(0);
    q.retire(made[1]);
    q.advance();
    expect(made[0]!.destroyed).toBe(1);
    expect(made[1]!.destroyed).toBe(0);
    q.retire(made[2]);
    q.drain();
    expect(made.map((m) => m.destroyed)).toEqual([1, 1, 1]);
    expect(q.pending).toBe(0);
  });
});

describe("K6.1 settings", () => {
  it("./three's VDBFrameRenderer settings are the core defaults (K6.8: one declaration), with froxelMix, occlusion, the K6.3 / K6.3a multi-scale (internal) and the K6.Q1 ambient-occlusion fields", () => {
    const core = vdbFrameDefaultSettings() as VDBFrameInternalSettings;
    const three = new VDBFrameRenderer().settings as unknown as Record<string, unknown>;
    expect(three).toEqual(core);
    const { froxelMix, occlusion, multiScaleLevels, multiScaleEdge, multiScaleSigma, ambientOcclusion, ambientOcclusionPairs, ambientOcclusionSteps, multipleScattering } = core;
    // the defaults the owner decides (K6.8 keeps them: a one-line change here and in settings.ts)
    expect(multipleScattering).toBe("neural");
    expect([multiScaleLevels, multiScaleEdge, multiScaleSigma]).toEqual([6, "zero", "his"]);
    // K6.Q1: the shadowed ambient is off by default (his unshadowed ambient), 4 pairs, the volume's shadow steps
    expect([ambientOcclusion, ambientOcclusionPairs, ambientOcclusionSteps]).toEqual([false, 4, 0]);
    expect(froxelMix).toBe("mixture");
    // K6.5b: host occluder maps shadow the volumes by default (core only)
    expect(occlusion).toBe(true);
  });
});

// K6.8: the three TSL port of the view maths (three/frame/view.ts, layout.ts) is deleted; its outputs on the inputs of
// the former equality test are frozen in fixtures/three-port-golden.json (dumped from the pre-deletion tree).
describe("K6.1 tools/frame/view.ts equals the (deleted) three/frame/view.ts port, frozen", () => {
  const golden = JSON.parse(readFileSync(new URL("./fixtures/three-port-golden.json", import.meta.url), "utf8")) as {
    view: { view: number[]; projection: number[]; box: number[]; W: number; H: number; D: number; range: [number, number]; ok: boolean; rect: number[]; slices: { first: number; count: number }; area: number[] }[];
    sizes: { input: [number, number, number, number, number]; size: { width: number; height: number }; held: Record<string, number> }[];
  };
  it("clip depths, rects, slices and areas on random cameras and boxes", () => {
    expect(golden.view.length).toBeGreaterThan(40);
    for (const g of golden.view) {
      const clip = vdbFrameClipCamera(g.view, g.projection, { coordinateSystem: "webgpu" });
      const ours = vdbFrameClipVolume(clip, g.box)!;
      expect(ours).not.toBeNull();
      const oursRect = vdbFrameFroxelRect(ours.localToView, g.projection, g.W, g.H);
      expect(!!oursRect).toBe(g.ok);
      if (oursRect) expect(oursRect).toEqual(g.rect);
      const slices = vdbFrameVolumeSlices(ours.viewNear, ours.viewFar, g.range, g.D);
      expect(slices).toEqual(g.slices);
      const a = vdbFrameFroxelArea([{ rect: oursRect, ...slices }], g.W, g.H, g.D);
      expect([...a.rect, a.first, a.count]).toEqual(g.area);
    }
  });

  it("froxel sizes and the held size follow the (deleted) layout.ts getFroxelSize", () => {
    const ours = vdbFroxelHeldSize();
    for (const g of golden.sizes) {
      const [bw, bh, cw, ch, ppf] = g.input;
      expect(vdbFrameFroxelSize({ width: 512, height: 512, pixelsPerFroxel: ppf }, { drawingBuffer: [bw, bh], canvas: [cw, ch] }, ours)).toEqual(g.size);
      expect(ours).toEqual(g.held);
    }
  });
});

describe("K6.1 WGSL sizes (D21; design §5.3)", () => {
  const SNAPSHOT_URL = new URL("../../../src/core/frame/size-snapshot.json", import.meta.url);
  const { compute, render } = allSpecs();
  const sizes = Object.fromEntries([...compute, ...render, vdbFrameChunkProbeSpec()].map((s) => [s.label, (wgslSizes(s.code) as { min: number; gz: number })]));
  const CAP_SOURCE = 56 * 1024;
  const CAP_FREE = 24 * 1024;

  it("matches the committed snapshot (K6_WRITE_SIZES=1 rewrites it) and fits the caps", () => {
    if (process.env.K6_WRITE_SIZES === "1") writeFileSync(SNAPSHOT_URL, `${JSON.stringify({ unit: "bytes of minified WGSL (scripts/lib/wgsl-variants.mjs wgslSizes)", kernels: sizes }, null, 2)}\n`);
    const snap = JSON.parse(readFileSync(SNAPSHOT_URL, "utf8")) as { kernels: Record<string, { min: number; gz: number }> };
    expect(Object.keys(snap.kernels).sort()).toEqual(Object.keys(sizes).sort());
    for (const [label, s] of Object.entries(sizes)) {
      expect(s.min, `${label}: WGSL size changed; if intended run K6_WRITE_SIZES=1 pnpm vitest run packages/vdb/test/core/frame and commit`).toBe(snap.kernels[label]!.min);
      // design §5.3: the transmittance with the MLP (and K7.3's lookup) may take 28 KiB
      const cap = label.includes("gather") || label.includes("grid density") || label.includes("emission direction") ? CAP_SOURCE : label.includes("transmittance (") ? 28 * 1024 : CAP_FREE;
      expect(s.min).toBeLessThanOrEqual(cap);
    }
  });
});
