import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  readWvdb,
  sampleWvdbDenseTexture,
  writeWvdb,
  wvdbDenseVolume,
  wvdbDensityScale,
  type WvdbDenseVolume,
  type WvdbVec3,
} from "../../src/tools/internal.js";
import { vdbDenseSource } from "../../src/core/index.js";
import { vdbDenseTextures } from "../../src/three/index.js";
import { bytesEqual, fixture, hisAsset, readBytes } from "./wvdb-helpers.js";
import { hisKernel, hisToByte, type HisKernelGrid } from "./his-wvdb-kernel.js";

/*
 * K3.0 gate "the dense source uploads exactly his bytes" — STATIC.
 *
 * We may not run his code, so `hisUploadPath` below re-implements, step by step,
 * what his renderer uploads for a `.wvdb` grid, written from his source at
 * mjurczyk/openvdb 75e7345 (read, not executed). Paths are under src/:
 *   openvdb/wvdb/format.js           readWVDBHeader L363–389, getWVDBChunks L392–426,
 *                                    decodeWVDBPage L145–209, decodeWVDBMask L319–328
 *   three-openvdb/wvdb/WVDBReader.js toByte L128, getBackgroundTexel L131–137, getKernelGrid L140–151,
 *                                    prepare L492–546 (getScale L494, densityStats L519–523,
 *                                    one-texel start L524–527), decode L428–489 (file order), apply L557–590
 *   three-openvdb/wvdb/wvdbKernel.js inflate L14–18, toByte L20, toTexels L24–58, fillTexels L76–86,
 *                                    writeBricks L89–119, upsample L122–188, clearMissingBricks L193–211,
 *                                    chunk L230–344 (multi-page LODs start from the upsampled previous LOD)
 *   three-openvdb/wvdb/WVDBLoader.js updateTexture L37–48, attach L116–162 (baked stats L147–152),
 *                                    updateTextures L168–204 (getDensityStats when not baked L196–198)
 *   docs/examples/src/examples/frame.js L519 (density grid), L535 (new FrameVolume(grid.data, { dimensions, bbox }))
 *   three-openvdb/frame/core/FrameVolume.js createTexture3D L14–22, getDensityStats L50–62,
 *                                    getMeanDensity L65–75, raw dimensions L357–364, R8 texture L368–373,
 *                                    Uint8 convert stats L487–493, getDensityScale L561–568
 *   three-openvdb/frame/webgpu/FrameRenderer.js configureTexture L483–487 (linear, no mipmaps)
 *   three-openvdb/frame/webgpu/frameMipmaps.js getLevelSize L14, getMipLevelCount L16–17, downsample L20–62,
 *                                    packAtlas L65–88, buildAtlas L90–106, atlas texture L183–189
 *   three-openvdb/frame/webgpu/frameShaderChunks.js setMipUniforms L773–798
 * and compares it with what `wvdbDenseVolume` + `vdbDenseTextures` (three) and
 * `vdbDenseSource` (raw WebGPU) upload: bytes, sizes, formats, filters, mip
 * atlas, mip uniforms and the normalized density scale.
 */

// ---------------------------------------------------------------------------
// His upload path, re-implemented from his source
// ---------------------------------------------------------------------------

// frameMipmaps.js L14–106
const hisGetLevelSize = (size: number, level: number) => Math.max(1, Math.ceil(size / 2 ** level));
const hisGetMipLevelCount = (w: number, h: number, d: number) => Math.ceil(Math.log2(Math.max(w, h, d, 1)));
function hisDownsample(source: Uint8Array, width: number, height: number, depth: number, channels: number) {
  const targetWidth = hisGetLevelSize(width, 1);
  const targetHeight = hisGetLevelSize(height, 1);
  const targetDepth = hisGetLevelSize(depth, 1);
  const target = new Uint8Array(targetWidth * targetHeight * targetDepth * channels);
  for (let z = 0; z < targetDepth; z++) {
    const z0 = Math.min(z * 2, depth - 1);
    const z1 = Math.min(z * 2 + 1, depth - 1);
    for (let y = 0; y < targetHeight; y++) {
      const y0 = Math.min(y * 2, height - 1);
      const y1 = Math.min(y * 2 + 1, height - 1);
      const row00 = (y0 + z0 * height) * width;
      const row10 = (y1 + z0 * height) * width;
      const row01 = (y0 + z1 * height) * width;
      const row11 = (y1 + z1 * height) * width;
      for (let x = 0; x < targetWidth; x++) {
        const x0 = Math.min(x * 2, width - 1);
        const x1 = Math.min(x * 2 + 1, width - 1);
        const targetIndex = (x + (y + z * targetHeight) * targetWidth) * channels;
        for (let channel = 0; channel < channels; channel++) {
          const s = (i: number) => source[i * channels + channel]!;
          const sum = s(x0 + row00) + s(x1 + row00) + s(x0 + row10) + s(x1 + row10) + s(x0 + row01) + s(x1 + row01) + s(x0 + row11) + s(x1 + row11);
          target[targetIndex + channel] = Math.round(sum / 8);
        }
      }
    }
  }
  return { data: target, width: targetWidth, height: targetHeight, depth: targetDepth };
}
function hisPackAtlas(levels: ReturnType<typeof hisDownsample>[], width: number, height: number, depth: number, channels: number) {
  const atlasWidth = 2 ** Math.max(Math.ceil(Math.log2(width)), levels.length);
  const atlasHeight = hisGetLevelSize(height, 1);
  const atlasDepth = hisGetLevelSize(depth, 1);
  const atlas = new Uint8Array(atlasWidth * atlasHeight * atlasDepth * channels);
  levels.forEach((level, levelIndex) => {
    const offset = atlasWidth - atlasWidth / 2 ** levelIndex;
    for (let z = 0; z < level.depth; z++) {
      for (let y = 0; y < level.height; y++) {
        const sourceStart = (y + z * level.height) * level.width * channels;
        const targetStart = (offset + (y + z * atlasHeight) * atlasWidth) * channels;
        atlas.set(level.data.subarray(sourceStart, sourceStart + level.width * channels), targetStart);
      }
    }
  });
  return { data: atlas, width: atlasWidth, height: atlasHeight, depth: atlasDepth, levels: levels.length };
}
function hisBuildAtlas(data: Uint8Array, width: number, height: number, depth: number, channels: number) {
  const count = hisGetMipLevelCount(width, height, depth);
  if (count < 1) return null;
  const levels = [];
  let level = { data, width, height, depth };
  for (let index = 1; index <= count; index++) {
    level = hisDownsample(level.data, level.width, level.height, level.depth, channels);
    levels.push(level);
  }
  return hisPackAtlas(levels, width, height, depth, channels);
}

// FrameVolume.js L50–62, L65–75
const hisGetDensityStats = (data: Uint8Array) => {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i]! > 0) {
      sum += data[i]!;
      count++;
    }
  }
  return { sum: sum / 255.0, count };
};

interface HisTexture {
  data: Uint8Array;
  width: number;
  height: number;
  depth: number;
  format: number;
  minFilter: number;
  magFilter: number;
  generateMipmaps: boolean;
  unpackAlignment: number;
}

/**
 * What his renderer uploads for grid `name` of a `.wvdb` file once the last LOD
 * has arrived (`maxLod` = Infinity), and the mip uniforms and density scale its
 * shaders get, with FrameVolume defaults `mipmaps: true`, `mipmapFilter: 'box'`,
 * `normalizeDensity: true`, `densityScale: 1`.
 */
async function hisUploadPath(bytes: Uint8Array, opts: { grid?: string; scales?: Record<string, number> } = {}) {
  // format.js readWVDBHeader L363–389
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dataStart = 12 + view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, dataStart)));
  const { grids, bake } = header;
  const scales = opts.scales ?? {};
  const getScale = ({ name }: { name: string }) => scales[name] || 1.0; // WVDBReader.js L494
  const finalLod = Math.min(Math.max(Math.floor(Infinity), 0), bake.lods - 1); // L496
  // WVDBReader.js L131–137
  const getBackgroundTexel = ({ class: gridClass, background, channels = 1 }: any, scale: number): number[] =>
    channels === 3 ? [...background.map(hisToByte), 255] : [gridClass === "level set" ? 0 : hisToByte(background / scale)];
  // WVDBReader.js prepare L504–535 (volume grids)
  const readerGrids: Record<string, any> = {};
  const kernelGrids: HisKernelGrid[] = grids.map((grid: any) => {
    const scale = getScale(grid);
    const backgroundTexel = getBackgroundTexel(grid, scale);
    readerGrids[grid.name] = {
      name: grid.name,
      bbox: grid.bbox,
      densityStats:
        typeof grid.meanDensity === "number" && scale === 1.0 ? { sum: grid.meanDensity * grid.nonZeroVoxels, count: grid.nonZeroVoxels } : null,
      dimensions: [1, 1, 1],
      data: new Uint8Array(backgroundTexel.length).fill(backgroundTexel[0]!),
      lod: -1,
    };
    // getKernelGrid L140–151
    return {
      class: grid.class,
      background: grid.background,
      channels: grid.channels || 1,
      scale,
      backgroundTexel,
      lods: grid.lods.map(({ dimensions, mask, pages }: any) => ({ dimensions, layout: mask ? "sparse" : "dense", pages: pages.length })),
    };
  });
  // format.js getWVDBChunks L392–426 (volume grids only in these files)
  const chunks: { gridIndex: number; lod: number; kind: string; offset: number; length: number; firstIndex?: number }[] = [];
  grids.forEach((grid: any, gridIndex: number) => {
    grid.lods.forEach(({ mask, pages }: any, lod: number) => {
      if (lod > finalLod) return;
      if (mask) chunks.push({ gridIndex, lod, kind: "mask", offset: mask.offset, length: mask.length });
      let firstIndex = 0;
      pages.forEach(({ offset, length, bricks }: any) => {
        chunks.push({ gridIndex, lod, kind: "page", offset, length, firstIndex });
        firstIndex += bricks;
      });
    });
  });
  chunks.sort((a, b) => a.offset - b.offset);
  // WVDBReader.decode L448–481 + apply L557–590: results applied in file order
  const kernel = hisKernel(bake.bits, kernelGrids);
  for (const c of chunks) {
    const result = await kernel.chunk({ ...c, bytes: bytes.slice(dataStart + c.offset, dataStart + c.offset + c.length) });
    if (result) {
      const g = readerGrids[grids[c.gridIndex].name];
      g.data = result.data;
      g.dimensions = grids[c.gridIndex].lods[c.lod].dimensions;
      g.lod = c.lod;
    }
  }
  // frame.js L519: volumeData.grids.density || Object.values(volumeData.grids)[0]
  const grid = opts.grid ? readerGrids[opts.grid] : (readerGrids.density ?? Object.values(readerGrids)[0]);
  const isColor = (grids.find((g: any) => g.name === grid.name).channels || 1) === 3;
  const [w, h, d] = grid.dimensions;
  // FrameVolume.js L14–22 createTexture3D + FrameRenderer.js L483–487 configureTexture
  const texture: HisTexture = {
    data: grid.data,
    width: w,
    height: h,
    depth: d,
    format: isColor ? THREE.RGBAFormat : THREE.RedFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    generateMipmaps: false,
    unpackAlignment: 1,
  };
  // FrameVolume.js L487–490 (Uint8 grid) then WVDBLoader.attach L147–152 (baked) / updateTextures L196–198
  let densityStats = hisGetDensityStats(grid.data);
  if (grid.densityStats) densityStats = grid.densityStats;
  const meanDensity = densityStats.count > 0 ? densityStats.sum / densityStats.count : 0.0; // L65–75
  // frameMipmaps.js L110–123 getAtlas → buildAtlas; L183–189 atlas texture
  const atlas = hisBuildAtlas(grid.data, w, h, d, isColor ? 4 : 1);
  const mipTexture = atlas
    ? { data: atlas.data, width: atlas.width, height: atlas.height, depth: atlas.depth, format: texture.format, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false, unpackAlignment: 1 }
    : null;
  // frameShaderChunks.js setMipUniforms L773–798
  const uniforms = {
    frameGridTexels: [w, h, d],
    frameMipAtlasSize: atlas ? [atlas.width, atlas.height, atlas.depth] : [1, 1, 1],
    frameDensityMipLevels: atlas ? atlas.levels : 0,
    // FrameVolume.js L561–568 (normalizeDensity true, densityScale 1)
    frameDensityScale: meanDensity > 0.0 ? 1.0 / meanDensity : 1.0,
  };
  return { texture, mipTexture, uniforms, meanDensity, lod: grid.lod };
}

// ---------------------------------------------------------------------------
// A recording fake of the WebGPU texture calls vdbDenseSource makes
// ---------------------------------------------------------------------------

function textureFake() {
  const textures: { desc: GPUTextureDescriptor; data: Uint8Array | null; layout: GPUTexelCopyBufferLayout | null; destroyed: boolean }[] = [];
  const samplers: GPUSamplerDescriptor[] = [];
  const device = {
    createTexture(desc: GPUTextureDescriptor) {
      const t = { desc, data: null as Uint8Array | null, layout: null as GPUTexelCopyBufferLayout | null, destroyed: false };
      textures.push(t);
      return { ...t, record: t, destroy: () => (t.destroyed = true), createView: () => ({ texture: t }) };
    },
    createSampler(desc: GPUSamplerDescriptor) {
      samplers.push(desc);
      return { desc };
    },
    queue: {
      writeTexture(dst: { texture: { record: (typeof textures)[number] } }, data: Uint8Array, layout: GPUTexelCopyBufferLayout) {
        dst.texture.record.data = new Uint8Array(data);
        dst.texture.record.layout = layout;
      },
    },
  };
  return { device: device as unknown as GPUDevice, textures, samplers };
}

// ---------------------------------------------------------------------------

async function expectSameUpload(bytes: Uint8Array, opts: { grid?: string; scales?: Record<string, number> } = {}) {
  const his = await hisUploadPath(bytes, opts);
  const file = readWvdb(bytes);
  const ours: WvdbDenseVolume = wvdbDenseVolume(file, { ...(opts.grid ? { density: opts.grid } : {}), ...(opts.scales ? { scales: opts.scales } : {}) });

  // tools: the bytes and sizes
  expect(ours.lod).toBe(his.lod);
  expect(ours.dimensions).toEqual([his.texture.width, his.texture.height, his.texture.depth]);
  expect(bytesEqual(ours.density.data, his.texture.data)).toBe(true);
  expect(ours.meanDensity).toBe(his.meanDensity);
  expect(wvdbDensityScale(ours)).toBe(his.uniforms.frameDensityScale);
  if (his.mipTexture) {
    expect(ours.density.mip!.dimensions).toEqual([his.mipTexture.width, his.mipTexture.height, his.mipTexture.depth]);
    expect(ours.density.mip!.levels).toBe(his.uniforms.frameDensityMipLevels);
    expect(bytesEqual(ours.density.mip!.data, his.mipTexture.data)).toBe(true);
  } else expect(ours.density.mip).toBeNull();

  // three (K6.11: a handle, no textures): the sizes, LOD levels and density mean the frame volume reads from it; the
  // texture bytes, formats and sampler are the core upload checked below
  const three = vdbDenseTextures(ours);
  expect(three.texels).toEqual(his.uniforms.frameGridTexels);
  expect(three.lodLevels).toBe(his.uniforms.frameDensityMipLevels);
  expect(three.stats.meanNonZero).toBe(his.meanDensity);

  // core: the raw WebGPU upload (r8unorm, his bytes, tight rows, linear clamp sampler)
  const fake = textureFake();
  const core = vdbDenseSource(fake.device, ours);
  const [map, mip] = fake.textures;
  expect(map!.desc).toMatchObject({ dimension: "3d", format: "r8unorm", size: [his.texture.width, his.texture.height, his.texture.depth] });
  expect(bytesEqual(map!.data!, his.texture.data)).toBe(true);
  expect(map!.layout).toEqual({ bytesPerRow: his.texture.width, rowsPerImage: his.texture.height });
  if (his.mipTexture) {
    expect(mip!.desc).toMatchObject({ dimension: "3d", format: "r8unorm", size: [his.mipTexture.width, his.mipTexture.height, his.mipTexture.depth] });
    expect(bytesEqual(mip!.data!, his.mipTexture.data)).toBe(true);
  }
  expect(fake.samplers).toEqual([
    expect.objectContaining({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" }),
  ]);
  expect(core.texels).toEqual(ours.dimensions);
  expect(core.lodLevels).toBe(his.uniforms.frameDensityMipLevels);
  expect(core.stats.meanNonZero).toBe(his.meanDensity);
  return { his, ours };
}

describe("dense .wvdb source uploads his bytes (static re-implementation of his upload path)", () => {
  it("cloud_04_variant_0000.wvdb (committed, CC0)", async () => {
    const { ours } = await expectSameUpload(fixture("cloud_04_variant_0000.wvdb"));
    // Baked before his header stats existed: his loader takes the stats from the texels.
    expect(ours.densityStatsSource).toBe("texels");
    expect(ours.activeTexels).toBeGreaterThan(0);
  });

  it("ground_explosion_0010.wvdb: density, and flames with his emissiveScale 6 (scales disable the baked stats)", async () => {
    const bytes = fixture("ground_explosion_0010.wvdb");
    await expectSameUpload(bytes);
    const { ours } = await expectSameUpload(bytes, { grid: "flames", scales: { flames: 6.0 } });
    expect(ours.densityStatsSource).toBe("texels");
  });

  it("ground_explosion_0199.wvdb (empty frame): background only, mean 0", async () => {
    const { ours } = await expectSameUpload(fixture("ground_explosion_0199.wvdb"));
    expect(ours.activeTexels).toBe(0);
    expect(ours.meanDensity).toBe(0);
  });

  it("multi-page sparse and dense LODs (his kernel's upsample + clearMissingBricks path), 8 and 16 bits, values above 1", async () => {
    const dims: WvdbVec3 = [37, 29, 21];
    const values = new Float32Array(dims[0] * dims[1] * dims[2]);
    for (let z = 0; z < dims[2]; z++)
      for (let y = 0; y < dims[1]; y++)
        for (let x = 0; x < dims[0]; x++) {
          const r = Math.hypot(x - 18, y - 14, z - 10);
          values[x + dims[0] * (y + dims[1] * z)] = r < 11 ? 1.4 * (1 - r / 11) + 0.05 * Math.sin(x * 1.3 + y * 0.7) : 0;
        }
    for (const layout of ["sparse", "dense"] as const) {
      for (const bits of [8, 16] as const) {
        const { bytes } = writeWvdb([{ name: "density", class: "fog volume", background: 0, bbox: [[-1, -1, -1], [1, 1, 1]], finest: { dimensions: dims, values } }], {
          layout,
          bits,
          pageBricks: 3,
          lods: 3,
        });
        const header = readWvdb(bytes).header;
        expect(header.grids[0]!.lods[2]!.pages.length).toBeGreaterThan(1);
        const { ours } = await expectSameUpload(bytes);
        // Values above 1 saturate (normalize 'clamp01', R1 §4.3).
        expect(ours.maxDensity).toBe(1);
      }
    }
  });

  it("vector (albedo) and level-set grids follow his toTexels rules", async () => {
    const dims: WvdbVec3 = [12, 9, 10];
    const n = dims[0] * dims[1] * dims[2];
    const color = new Float32Array(n * 3);
    const sdf = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      color[i] = (i % 7) / 6;
      color[n + i] = (i % 5) / 4 + 0.2;
      color[2 * n + i] = -0.1 + (i % 3) * 0.5;
      sdf[i] = ((i % 11) - 5) * 0.1;
    }
    const { bytes } = writeWvdb(
      [
        { name: "density", class: "level set", background: 0.3, bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: dims, values: sdf } },
        { name: "Cd", channels: 3, background: [0.5, 0.25, 1], bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: dims, values: color } },
      ],
      { lods: 2, pageBricks: 1 },
    );
    await expectSameUpload(bytes);
    // The vector grid as albedo: RGBA8, alpha 255, the same size as the density texture.
    const his = await hisUploadPath(bytes, { grid: "Cd" });
    const ours = wvdbDenseVolume(readWvdb(bytes), { albedo: "Cd" });
    expect(ours.albedo!.format).toBe("rgba8unorm");
    expect(bytesEqual(ours.albedo!.data, his.texture.data)).toBe(true);
    expect(bytesEqual(ours.albedo!.mip!.data, his.mipTexture!.data)).toBe(true);
  });

  const his = (name: string) => hisAsset(name);
  for (const name of ["bunny_cloud.wvdb", "smoke.wvdb"]) {
    it.skipIf(!his(name))(`${name} (his example asset, R1 Step 0; MJURCZYK_OPENVDB_DIR)`, async () => {
      const { ours } = await expectSameUpload(readBytes(his(name)!));
      expect(ours.densityStatsSource).toBe("baked");
    });
  }
});

describe("dense source CPU twin (sampleWvdbDenseTexture)", () => {
  const volume = () => {
    const dims: WvdbVec3 = [16, 8, 4];
    const values = new Float32Array(16 * 8 * 4).map((_, i) => ((i * 37) % 101) / 100);
    const { bytes } = writeWvdb([{ name: "density", class: "fog volume", background: 0, bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: dims, values } }], { lods: 1 });
    return wvdbDenseVolume(readWvdb(bytes));
  };

  it("returns texel values at texel centres and clamps at the edges", () => {
    const v = volume();
    const t = v.density;
    for (const [x, y, z] of [[0, 0, 0], [5, 3, 2], [15, 7, 3]]) {
      const uvw = [(x! + 0.5) / 16, (y! + 0.5) / 8, (z! + 0.5) / 4];
      expect(sampleWvdbDenseTexture(t, uvw)).toBeCloseTo(t.data[x! + 16 * (y! + 8 * z!)]! / 255, 12);
    }
    expect(sampleWvdbDenseTexture(t, [-1, -1, -1])).toBeCloseTo(t.data[0]! / 255, 12);
    expect(sampleWvdbDenseTexture(t, [2, 2, 2])).toBeCloseTo(t.data[t.data.length - 1]! / 255, 12);
  });

  it("interpolates linearly between texel centres and blends mip levels", () => {
    const t = volume().density;
    const a = t.data[3 + 16 * (2 + 8 * 1)]! / 255;
    const b = t.data[4 + 16 * (2 + 8 * 1)]! / 255;
    expect(sampleWvdbDenseTexture(t, [4 / 16, 2.5 / 8, 1.5 / 4])).toBeCloseTo((a + b) / 2, 12);
    // lod 1 at a level-1 texel centre is that atlas texel; lod 0.5 is halfway between lod 0 and lod 1.
    const l1 = sampleWvdbDenseTexture(t, [0.5 / 8, 0.5 / 4, 0.5 / 2], 1);
    expect(l1).toBeCloseTo(t.mip!.data[0]! / 255, 12);
    const l0 = sampleWvdbDenseTexture(t, [0.5 / 8, 0.5 / 4, 0.5 / 2], 0);
    expect(sampleWvdbDenseTexture(t, [0.5 / 8, 0.5 / 4, 0.5 / 2], 0.5)).toBeCloseTo((l0 + l1) / 2, 12);
    expect(t.mip!.levels).toBe(4);
  });
});

describe("vdbDenseSource WGSL and bindings", () => {
  const dense = () => {
    const dims: WvdbVec3 = [8, 8, 8];
    const v = new Float32Array(512).fill(0.5);
    const c = new Float32Array(512 * 3).fill(0.25);
    const { bytes } = writeWvdb(
      [
        { name: "density", class: "fog volume", background: 0, bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: dims, values: v } },
        { name: "Cd", channels: 3, background: [0, 0, 0], bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: dims, values: c } },
      ],
      { lods: 1 },
    );
    return wvdbDenseVolume(readWvdb(bytes), { albedo: "Cd" });
  };

  it("declares a sampler then each channel's map and mip atlas from firstBinding, and the R1 §4.2 functions", () => {
    const fake = textureFake();
    const src = vdbDenseSource(fake.device, dense());
    const wgsl = src.wgsl({ slot: { group: 2, firstBinding: 3 }, profile: "march" });
    expect(wgsl).toContain("@group(2) @binding(3) var vdbSrcSampler: sampler;");
    expect(wgsl).toContain("@group(2) @binding(4) var vdbSrcDensityMap: texture_3d<f32>;");
    expect(wgsl).toContain("@group(2) @binding(5) var vdbSrcDensityMip: texture_3d<f32>;");
    expect(wgsl).toContain("@group(2) @binding(6) var vdbSrcAlbedoMap: texture_3d<f32>;");
    expect(wgsl).toContain("@group(2) @binding(7) var vdbSrcAlbedoMip: texture_3d<f32>;");
    for (const fn of ["vdbSrcDensity", "vdbSrcMask", "vdbSrcAlbedo", "vdbSrcEmission", "vdbSrcTexels", "vdbSrcLodLevels", "vdbSrcSkipEmpty", "vdbSrcMaxIn"]) {
      expect(wgsl).toMatch(new RegExp(`fn ${fn}\\(`));
    }
    expect(wgsl).toMatch(/fn vdbSrcMask\(v: u32, uvw: vec3f, lod: f32\) -> f32 \{\n  return 1\.0;/);
    expect(wgsl).toMatch(/fn vdbSrcEmission\(v: u32, uvw: vec3f, lod: f32\) -> f32 \{\n  return 0\.0;/);
    expect(src.wgsl({ slot: { group: 0, firstBinding: 0 }, profile: "sample" })).not.toContain("vdbSrcSkipEmpty");
    // Channel off at compile time: constant, and no sample function for it.
    const off = src.wgsl({ slot: { group: 0, firstBinding: 0 }, profile: "sample", channels: { albedo: false } });
    expect(off).toMatch(/fn vdbSrcAlbedo\(v: u32, uvw: vec3f, lod: f32\) -> vec3f \{\n  return vec3f\(1\.0\);/);
    expect(off).not.toContain("fn vdbSrcSampleAlbedo");
    const layout = src.layoutEntries({ group: 2, firstBinding: 3 });
    expect(layout.map((e) => e.binding)).toEqual([3, 4, 5, 6, 7]);
    expect(layout[0]).toMatchObject({ sampler: { type: "filtering" } });
    expect(layout[1]).toMatchObject({ texture: { sampleType: "float", viewDimension: "3d" } });
    expect(src.bindEntries({ group: 2, firstBinding: 3 }).map((e) => e.binding)).toEqual([3, 4, 5, 6, 7]);
    expect(fake.textures.map((t) => t.desc.format)).toEqual(["r8unorm", "r8unorm", "rgba8unorm", "rgba8unorm"]);
  });

  it("replace: same size rewrites in place; a new size makes new textures and bumps version (the old ones destroyed in a later task); dispose destroys them", async () => {
    const fake = textureFake();
    const v = dense();
    const src = vdbDenseSource(fake.device, v);
    src.replace(v);
    expect(src.version).toBe(0);
    expect(fake.textures).toHaveLength(4);
    const dims: WvdbVec3 = [16, 8, 8];
    const { bytes } = writeWvdb(
      [
        { name: "density", class: "fog volume", background: 0, bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: dims, values: new Float32Array(1024).fill(0.2) } },
        { name: "Cd", channels: 3, background: [0, 0, 0], bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: dims, values: new Float32Array(3072).fill(0.2) } },
      ],
      { lods: 1 },
    );
    src.replace(wvdbDenseVolume(readWvdb(bytes), { albedo: "Cd" }));
    expect(src.version).toBe(1);
    expect(src.texels).toEqual([16, 8, 8]);
    // K6.10a fail-before: destroyed inside replace() before, so a host that replaced between frame.update(encoder) and
    // queue.submit submitted destroyed textures; now they live until a later task
    expect(fake.textures.slice(0, 4).some((t) => t.destroyed)).toBe(false);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fake.textures.slice(0, 4).every((t) => t.destroyed)).toBe(true);
    src.dispose();
    expect(fake.textures.every((t) => t.destroyed)).toBe(true);
    expect(() => src.wgsl({ slot: { group: 0, firstBinding: 0 }, profile: "sample" })).toThrow(/VDB_DISPOSED/);
  });

  it("rejects a non-dense input and a bad slot", () => {
    const fake = textureFake();
    expect(() => vdbDenseSource(fake.device, {} as never)).toThrow(/VDB_BAD_OPTION/);
    const src = vdbDenseSource(fake.device, dense());
    expect(() => src.wgsl({ slot: { group: -1, firstBinding: 0 }, profile: "sample" })).toThrow(/VDB_BAD_OPTION/);
  });
});
