// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L729–763 (createVolumeUniforms, the source
//   half), L834–1028 (createVolumeNodes: getLod, sampleExtinction, sampleAlbedo, sampleEmission and
//   the box helpers, through ./volume-source.ts).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: the density is read from a sparse NanoVDB grid in a VDBGridPool (vdbSample, trilinear)
// instead of an 8-bit 3D texture; three LOD strategies instead of his box mip atlas; an HDDA
// empty-space query for marches. Divergences: DIVERGES.md (K3.2 rows), V23–V25.

/**
 * K6.8a: the TSL half of the sparse source (`vdbSparseSource`, ./sparse-source.ts), split off the handle: the nodes and
 * uniforms of `vdbVolumeNodes` / `vdbVolumeUniforms` (his `createVolumeNodes` shape, for custom TSL materials) and the
 * `'coarse-pyramid'` LOD's own GPU pyramid textures they sample. The handle makes this part the first time a custom
 * material asks for it (`vdbVolumeUniforms`, `vdbVolumeNodes`, the probe `wgslFn`s or `source.pyramid`); a source only a
 * `VDBFrameVolume` reads never makes it: the frame renderer reads the handle's pool, ids, boxes and stats into the core's
 * sparse source, which has its own pyramids.
 *
 * Positions are his local unit box [-0.5, 0.5]³; uvw = local + 0.5 (+ offset, wrapped) maps onto the channel grid's index
 * cell box (see ./sparse-source.ts). The coarse pyramid (`pyramidBuild`): `'gpu'` (default) the core's
 * `vdbCreateBoxPyramid` over the pool grid into three r16f textures, rebuilt when the grid's image changes (checked once
 * per render that samples it); `'cpu'` the tools' `vdbBoxPyramid` from `images.density` (static).
 */

import * as THREE from "three";
import * as TSL from "three/tsl";
import { code, wgslFn } from "three/tsl";
import { vdbCreateBoxPyramid, warmBoxPyramidPipelines } from "../core/compute/pyramid.js";
import { vdbBoxPyramid } from "../tools/frame/sparse.js";
import { vdbBoxPyramidLayout } from "../tools/frame/pyramid-layout.js";
import { VDBError } from "../core/errors.js";
import type { VDBVolumeChannels } from "../core/source/types.js";
import { getGPUTexture, webgpuBackendOf } from "./device.js";
import { vdbSample, type VDBGridNode, type VDBTslNode } from "./grid-node.js";
import { vdbBoxNodes, vdbDensityScaleOf, vdbGridUv, vdbMaxComponent, vdbMipSampler, type VDBVolumeNodes, type VDBVolumeNodesOptions, type VDBVolumeUniformParameters } from "./volume-source.js";
import type { VDBIndexBox, VDBSparseLod, VDBSparsePyramid, VDBSparseSourceGrids, VDBSparseVolumeUniforms, VDBThreeSparseSource } from "./sparse-source.js";

const { If, clamp, float, int, log2, max, mix, pow, select, texture3D, uint, uniform, vec2, vec3 } = TSL as unknown as Record<string, VDBTslNode>;

type Vec3 = [number, number, number];

/** What the TSL half reads from its handle (./sparse-source.ts). */
export interface VDBSparseHandle {
  readonly grids: VDBGridNode;
  readonly ids: Readonly<VDBSparseSourceGrids>;
  readonly lod: VDBSparseLod;
  readonly normalize: "none" | "clamp01";
  readonly channels: VDBVolumeChannels;
  readonly indexBox: VDBIndexBox;
  readonly auxBoxes: Record<"mask" | "albedo" | "emission", VDBIndexBox | null>;
  readonly texels: [number, number, number];
  readonly lodLevels: number;
  readonly followImage: boolean;
  readonly fastTrilinear: boolean;
  readonly cacheAccessor: boolean;
  readonly stats: { readonly meanNonZero: number; readonly max: number; readonly activeVoxels: number };
  /** The pyramid's options: where it is built, `pyramidMax`, and the CPU image (`'cpu'`). */
  readonly pyramidOptions: { readonly build: "gpu" | "cpu"; readonly pyramidMax?: number; readonly image: Uint32Array | null };
  syncLayout(): boolean;
  refreshStats(): Promise<boolean>;
  /** Calls `apply` after each stats readback; returns the unsubscribe. */
  onStats(apply: () => void): () => void;
  /** Calls `apply` after each layout re-read (`followImage`); returns the unsubscribe. */
  onLayout(apply: () => void): () => void;
}

const bad = (message: string) => new VDBError("VDB_BAD_OPTION", `vdbSparseSource: ${message}`);

/** A half-float r16f Data3DTexture (linear, clamp-to-edge, no mipmaps); `data` null: GPU-written (no upload). */
function halfTexture(data: Float32Array | null, dims: readonly number[]): THREE.Data3DTexture {
  let half: Uint16Array | null = null;
  if (data) {
    half = new Uint16Array(data.length);
    for (let i = 0; i < data.length; i++) half[i] = THREE.DataUtils.toHalfFloat(data[i]!);
  }
  const [w, h, d] = dims as Vec3;
  const texture = new THREE.Data3DTexture(half, w, h, d);
  // A GPU-written texture: three allocates it (createTexture) but never uploads (no data to upload).
  if (!half) texture.source.dataReady = false;
  texture.format = THREE.RedFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

// WGSL helpers over the grid module (one module per shader: the VDBGridNode's).
const NODE_AVERAGE_WGSL = /* wgsl */ `fn vdbFrameNodeAverage(g: u32, p: vec3f, dim: u32) -> f32 {
    var acc = vdbAccessorInit(g);
    let fd = f32(dim);
    let q = (p - vec3f(0.5 * fd - 0.5)) / fd;
    let base = floor(q);
    let w = q - base;
    let c = vec3i(base) * i32(dim);
    // One call site in a loop: SwiftShader compiles 8 inlined tree walks very slowly.
    var v: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) {
        v[k] = vdbNodeAverage(&acc, c + vec3i(vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u)) * i32(dim), dim);
    }
    let x00 = v[0] + w.x * (v[1] - v[0]);
    let x10 = v[2] + w.x * (v[3] - v[2]);
    let x01 = v[4] + w.x * (v[5] - v[4]);
    let x11 = v[6] + w.x * (v[7] - v[6]);
    let y0 = x00 + w.y * (x10 - x00);
    let y1 = x01 + w.y * (x11 - x01);
    return y0 + w.z * (y1 - y0);
}`;

// The 'node-average' density at a lod (one call per sample: level 0 → 3 → 7 blends, two taps at most).
const nodeLodWgsl = (sample: string) => /* wgsl */ `fn vdbFrameNodeLod(g: u32, p: vec3f, lod: f32) -> f32 {
    var acc = vdbAccessorInit(g);
    if lod <= 0.0 {
        return ${sample}(&acc, p);
    }
    let fine = lod < 3.0;
    var a = 0.0;
    if fine {
        a = ${sample}(&acc, p);
    } else {
        a = vdbFrameNodeAverage(g, p, 8u);
    }
    if lod >= 7.0 {
        return vdbFrameNodeAverage(g, p, 128u);
    }
    let b = vdbFrameNodeAverage(g, p, select(128u, 8u, fine));
    let t = select((lod - 3.0) / 4.0, lod / 3.0, fine);
    return mix(a, b, t);
}`;

// K3.2d: level-0 density through the source's own sampler: vdbSampleFast (helper 'fastTrilinear') and/or one
// accessor per invocation kept in a private variable (copied in and out: WGSL can't pass a ptr<private> where
// the API takes ptr<function>). Same values as vdbSample with a fresh accessor (the accessor cache only skips
// the descent).
const SAMPLE_CACHE_WGSL = /* wgsl */ `var<private> vdbFrameAcc: VdbAccessor;
var<private> vdbFrameAccGrid: u32 = 0xffffffffu;`;
const sampleWgsl = (fast: boolean, cached: boolean) => {
  const call = fast ? "vdbSampleFast" : "vdbSample";
  return cached
    ? /* wgsl */ `fn vdbFrameSample(g: u32, xyz: vec3f) -> f32 {
    if vdbFrameAccGrid != g {
        vdbFrameAcc = vdbAccessorInit(g);
        vdbFrameAccGrid = g;
    }
    var a = vdbFrameAcc;
    let v = ${call}(&a, xyz);
    vdbFrameAcc = a;
    return v;
}`
    : /* wgsl */ `fn vdbFrameSample(g: u32, xyz: vec3f) -> f32 {
    var a = vdbAccessorInit(g);
    return ${call}(&a, xyz);
}`;
};

// The HDDA empty-space query; CPU twin tools vdbFrameSkipEmpty (see there for the contract).
const SKIP_EMPTY_WGSL = /* wgsl */ `fn vdbFrameSkipEmpty(g: u32, o: vec3f, d: vec3f, t0: f32, t1: f32, cutoff: f32, dilate: u32) -> vec2f {
    var acc = vdbAccessorInit(g);
    var h: VdbHdda;
    vdbHddaInit(&h, o, t0, d, t1, 8u);
    let back = 1e-3 / max(length(d), 1e-6);
    var ts = t0;
    for (var n = 0u; n < 64u; n++) {
        let te = vdbHddaCellEnd(&h);
        var b = 0.0;
        if dilate == 0u {
            b = vdbCellBoundTrilinear(&acc, h.voxel, 8u);
        } else {
            b = -1e30;
            for (var k = 0u; k < 27u; k++) {
                let c = vec3i(i32(k % 3u) - 1, i32((k / 3u) % 3u) - 1, i32(k / 9u) - 1) * 8;
                b = max(b, vdbNodeMax(&acc, h.voxel + c, 8u));
            }
        }
        if b > cutoff {
            return vec2f(max(t0, ts - back), te);
        }
        ts = te;
        if !vdbHddaStep(&h) {
            return vec2f(t1, t1);
        }
    }
    let r = max(t0, ts - back);
    return vec2f(r, r);
}`;

/** The TSL half of one sparse source: its pyramid textures, uniforms and nodes (see the file comment). */
export class VDBSparseNodes {
  readonly #src: VDBSparseHandle;
  pyramid: VDBSparsePyramid | null = null;
  pyramidBuilds = 0;
  /** G16a: the atlas node's texture when the layout has no atlas (a 1³ zero texture of its own). */
  #emptyAtlas: THREE.Data3DTexture | null = null;
  #poolRevisionSeen = -1;
  #nodeAverage: ((...a: unknown[]) => VDBTslNode) | null = null;
  #nodeLod: ((...a: unknown[]) => VDBTslNode) | null = null;
  #skip: ((...a: unknown[]) => VDBTslNode) | null = null;
  #sample: ((...a: unknown[]) => VDBTslNode) | null = null;
  /** G16a: the pyramid texture nodes of a following source's uniforms (their textures change with the layout). */
  readonly #pyramidTextures = new WeakMap<object, { map: VDBTslNode; mip: VDBTslNode }>();

  constructor(src: VDBSparseHandle) {
    this.#src = src;
    if (src.lod !== "coarse-pyramid") return;
    const { build, image } = src.pyramidOptions;
    if (build === "cpu") {
      const p = vdbBoxPyramid(image!, { indexBox: src.indexBox, ...this.#pyramidMax() });
      this.pyramid = {
        build,
        level: p.level,
        dimensions: p.dimensions,
        atlas: p.atlas,
        lodLevels: p.lodLevels,
        data: p.data,
        map: halfTexture(p.data, p.dimensions),
        mip: p.atlas ? halfTexture(p.atlas.data, p.atlas.dimensions) : null,
        gpu: null,
      };
    } else {
      // G16a: a following source compiles the pyramid kernels of every type the node decodes now, so a swap to
      // another grid type creates no pipeline.
      if (src.followImage) warmBoxPyramidPipelines(webgpuBackendOf(src.grids.renderer, "vdbSparseSource").device, src.grids.options.types);
      this.pyramid = this.#gpuPyramid(src.indexBox, src.ids.density);
    }
    this.refreshPyramid();
  }

  #pyramidMax(): { pyramidMax?: number } {
    const m = this.#src.pyramidOptions.pyramidMax;
    return m !== undefined ? { pyramidMax: m } : {};
  }

  /** GPU bytes of the pyramid textures and the GPU build's scratch buffers. */
  get gpuBytes(): { pyramid: number; pyramidScratch: number } {
    const p = this.pyramid;
    const atlasTexels = p?.atlas ? p.atlas.dimensions[0] * p.atlas.dimensions[1] * p.atlas.dimensions[2] : 0;
    return { pyramid: p ? 2 * (p.dimensions[0] * p.dimensions[1] * p.dimensions[2] + atlasTexels) : 0, pyramidScratch: p?.gpu?.gpuBytes.scratch ?? 0 };
  }

  /** The GPU pyramid of `box` (K3.2a): three textures the core's `vdbCreateBoxPyramid` writes into. */
  #gpuPyramid(box: VDBIndexBox, densityId: number): VDBSparsePyramid {
    const grids = this.#src.grids;
    const renderer = grids.renderer as unknown as { initTexture(t: THREE.Texture): void };
    const device = webgpuBackendOf(renderer, "vdbSparseSource").device;
    const texels = [0, 1, 2].map((a) => box.max[a]! - box.min[a]! + 1);
    const layout = vdbBoxPyramidLayout(texels, this.#pyramidMax());
    const map = halfTexture(null, layout.dimensions);
    const mip = layout.atlas ? halfTexture(null, layout.atlas.dimensions) : null;
    map.name = "vdbSparseSource pyramid base";
    if (mip) mip.name = "vdbSparseSource pyramid atlas";
    for (const t of [map, mip]) if (t) renderer.initTexture(t);
    const gpu = vdbCreateBoxPyramid(device, grids.pool as never, densityId, {
      indexBox: box,
      ...this.#pyramidMax(),
      targets: { base: getGPUTexture(renderer as never, map), atlas: mip ? getGPUTexture(renderer as never, mip) : null },
      label: "vdbSparseSource pyramid",
    });
    return {
      build: "gpu",
      level: layout.level,
      dimensions: [...layout.dimensions],
      atlas: layout.atlas ? { dimensions: [...layout.atlas.dimensions], levels: layout.atlas.levels.length } : null,
      lodLevels: layout.lodLevels,
      map,
      mip,
      gpu,
    };
  }

  /** G16a: the atlas texture a following source's shaders bind: the pyramid's, else a 1³ zero texture (never read). */
  atlasTexture(): THREE.Data3DTexture {
    if (this.pyramid?.mip) return this.pyramid.mip;
    if (!this.#emptyAtlas) {
      this.#emptyAtlas = halfTexture(new Float32Array(1), [1, 1, 1]);
      this.#emptyAtlas.name = "vdbSparseSource empty atlas";
    }
    return this.#emptyAtlas;
  }

  /**
   * G16a: the handle re-read its layout. A new density box needs a pyramid of its size (and box min); the same box
   * rebuilds in place (refreshPyramid). Called before the handle's layout listeners (the uniforms) run.
   */
  relayout(moved: boolean): void {
    const previous = this.pyramid;
    if (!moved || !previous?.gpu) return;
    this.pyramid = this.#gpuPyramid(this.#src.indexBox, this.#src.ids.density);
    this.#poolRevisionSeen = -1;
    // After the uniforms point at the new textures, the old ones go, but not yet: three r187's texture.dispose()
    // destroys the GPUTexture at once, and a pass this render already recorded (not submitted yet) may bind them.
    // Renders submit in the task that records them, so wait past this task, then for the queue's submitted work.
    const retire = (): void => {
      previous.gpu?.destroy();
      previous.map.dispose();
      previous.mip?.dispose();
    };
    const device = (this.#src.grids.pool as { device?: GPUDevice }).device;
    if (device?.queue) setTimeout(() => void device.queue.onSubmittedWorkDone().then(retire, retire), 0);
    else retire();
  }

  /** See `VDBThreeSparseSource.refreshPyramid` (the handle syncs its layout first). */
  refreshPyramid(): boolean {
    const gpu = this.pyramid?.gpu;
    if (!gpu) return false;
    // Cheap test first: the pool's revision moves on any change; the grid's imageRevision tells whether this one did.
    const pool = this.#src.grids.pool as unknown as { revision: number; device: GPUDevice };
    if (pool.revision === this.#poolRevisionSeen) return false;
    const device = pool.device;
    const encoder = device.createCommandEncoder({ label: "vdbSparseSource pyramid" });
    if (!gpu.update(encoder)) {
      // Not resident yet (the update that writes it is still to come): look again next time.
      if (!gpu.stale) this.#poolRevisionSeen = pool.revision;
      return false;
    }
    device.queue.submit([encoder.finish()]);
    this.#poolRevisionSeen = pool.revision;
    this.pyramidBuilds++;
    return true;
  }

  nodeAverageFn(): (...a: unknown[]) => VDBTslNode {
    this.#nodeAverage ??= wgslFn(NODE_AVERAGE_WGSL, [this.#src.grids.module] as never) as unknown as (...a: unknown[]) => VDBTslNode;
    return this.#nodeAverage;
  }

  #nodeLodFn(): (...a: unknown[]) => VDBTslNode {
    this.#nodeLod ??= wgslFn(nodeLodWgsl(this.#src.fastTrilinear ? "vdbSampleFast" : "vdbSample"), [this.#src.grids.module, this.nodeAverageFn()] as never) as unknown as (...a: unknown[]) => VDBTslNode;
    return this.#nodeLod;
  }

  /** `vdbFrameSample(g, xyz) -> f32`: level 0 (index space) as configured (`fastTrilinear`, `cacheAccessor`). */
  #sampleFn(): (...a: unknown[]) => VDBTslNode {
    const { grids, fastTrilinear, cacheAccessor } = this.#src;
    this.#sample ??= wgslFn(sampleWgsl(fastTrilinear, cacheAccessor), (cacheAccessor ? [grids.module, code(SAMPLE_CACHE_WGSL, [grids.module])] : [grids.module]) as never) as unknown as (...a: unknown[]) => VDBTslNode;
    return this.#sample;
  }

  skipEmptyFn(): (...a: unknown[]) => VDBTslNode {
    this.#skip ??= wgslFn(SKIP_EMPTY_WGSL, [this.#src.grids.module] as never) as unknown as (...a: unknown[]) => VDBTslNode;
    return this.#skip;
  }

  createUniforms(source: VDBThreeSparseSource, p: VDBVolumeUniformParameters): VDBSparseVolumeUniforms {
    const src = this.#src;
    let densityScale = p.densityScale ?? 1.0;
    let normalize = p.normalizeDensity ?? true;
    const boxUniform = (bx: VDBIndexBox | null) =>
      bx ? { min: uniform(new THREE.Vector3(...bx.min)), size: uniform(new THREE.Vector3(bx.max[0] - bx.min[0] + 1, bx.max[1] - bx.min[1] + 1, bx.max[2] - bx.min[2] + 1)) } : null;
    const levels = uniform(src.lodLevels);
    // K3.2e: the mean follows the density image (refreshStats, checked once per render, reads a replaced image
    // back). K5.9a: the new mean is applied once, when the readback lands; the render update never writes the
    // value, so a value set per draw (VDBFrameVolume's densityScale, through setVolumeUniforms) stays as set.
    const densityScaleU = uniform(vdbDensityScaleOf(source, densityScale, normalize)).onRenderUpdate(() => {
      src.syncLayout();
      void src.refreshStats();
      return undefined;
    });
    const applyStats = (): void => {
      densityScaleU.value = vdbDensityScaleOf(source, densityScale, normalize);
    };
    const offStats = src.onStats(applyStats);
    const pyr = this.pyramid;
    // G16a: a following source samples its pyramid through texture nodes made here (one pair per uniforms), so a new
    // layout swaps their textures; the atlas node always exists (a 1³ texture when the layout has none).
    const textures = src.followImage && pyr ? { map: texture3D(pyr.map), mip: texture3D(this.atlasTexture()) } : null;
    let offLayout = (): void => {};
    const u: VDBSparseVolumeUniforms = {
      worldScale: uniform(p.worldScale ?? 1.0),
      gridTexels: uniform(new THREE.Vector3(...src.texels)),
      mipAtlasSize: uniform(new THREE.Vector3(1, 1, 1)),
      maps: { density: null, mask: null, albedo: null, emission: null },
      mipMaps: { density: null, mask: null, albedo: null, emission: null },
      // getSampledLod reads density (and mask) levels; aux channels are level 0 only but report the same.
      mipLevels: { density: levels, mask: levels, albedo: levels, emission: levels },
      baseColor: uniform(new THREE.Color(p.baseColor ?? 0xffffff)),
      extinction: uniform(p.extinction ?? 100.0),
      densityScale: densityScaleU,
      densityCutoff: uniform(p.densityCutoff ?? 0.0),
      emissiveIntensity: uniform(p.emissiveIntensity ?? 1.0),
      temperatureRange: uniform(new THREE.Vector2(...(p.temperatureRange ?? [0.0, 6500.0]))),
      offset3D: uniform(new THREE.Vector3().copy((p.offset3D ?? { x: 0, y: 0, z: 0 }) as THREE.Vector3)),
      wrap3D: uniform(p.wrap3D ?? THREE.ClampToEdgeWrapping, "int"),
      boxes: { density: boxUniform(src.indexBox), mask: boxUniform(src.auxBoxes.mask), albedo: boxUniform(src.auxBoxes.albedo), emission: boxUniform(src.auxBoxes.emission) },
      // The GPU pyramid rebuilds (on an image change) from here: once per render that samples it.
      pyramidLevel: pyr?.gpu ? uniform(pyr.level).onRenderUpdate(() => void source.refreshPyramid()) : uniform(pyr?.level ?? 0),
      pyramidLevels: uniform(pyr?.atlas?.levels ?? 0),
      pyramidTexels: uniform(new THREE.Vector3(...(pyr?.dimensions ?? [1, 1, 1]))),
      pyramidAtlasSize: uniform(new THREE.Vector3(...(pyr?.atlas?.dimensions ?? [1, 1, 1]))),
      update: (s, o = {}) => {
        if (s !== source) throw bad("these uniforms belong to another source.");
        if (o.densityScale !== undefined) densityScale = o.densityScale;
        if (o.normalizeDensity !== undefined) normalize = o.normalizeDensity;
        densityScaleU.value = vdbDensityScaleOf(source, densityScale, normalize);
        if (src.followImage) applyLayout();
      },
      dispose: () => {
        offStats();
        offLayout();
        this.#pyramidTextures.delete(u);
      },
    };
    // G16a: the layout as uniform values (and texture swaps), applied again after every re-read.
    const setBox = (b: { min: VDBTslNode; size: VDBTslNode } | null, bx: VDBIndexBox | null) => {
      if (!b || !bx) return;
      b.min.value.set(...bx.min);
      b.size.value.set(bx.max[0] - bx.min[0] + 1, bx.max[1] - bx.min[1] + 1, bx.max[2] - bx.min[2] + 1);
    };
    const applyLayout = () => {
      u.gridTexels.value.set(...src.texels);
      setBox(u.boxes.density, src.indexBox);
      setBox(u.boxes.mask, src.auxBoxes.mask);
      setBox(u.boxes.albedo, src.auxBoxes.albedo);
      setBox(u.boxes.emission, src.auxBoxes.emission);
      levels.value = src.lodLevels;
      const p = this.pyramid;
      if (!p) return;
      u.pyramidLevel.value = p.level;
      u.pyramidLevels.value = p.atlas?.levels ?? 0;
      u.pyramidTexels.value.set(...p.dimensions);
      u.pyramidAtlasSize.value.set(...(p.atlas?.dimensions ?? [1, 1, 1]));
      if (textures) {
        textures.map.value = p.map;
        textures.mip.value = this.atlasTexture();
      }
    };
    if (src.followImage) {
      offLayout = src.onLayout(applyLayout);
      if (textures) this.#pyramidTextures.set(u, textures);
    }
    return u;
  }

  createNodes(u0: VDBSparseVolumeUniforms, o: VDBVolumeNodesOptions): VDBVolumeNodes {
    const u = u0;
    if (!u.boxes) throw bad("createNodes needs this source's uniforms (vdbVolumeUniforms(source)).");
    const src = this.#src;
    const { grids, ids, lod } = src;
    const useMipmaps = o.useMipmaps ?? src.lodLevels > 0;
    const gridUv = vdbGridUv(u);
    const clamp01 = src.normalize === "clamp01";
    // G16a: a following source's LOD range is a uniform (it changes with the layout); otherwise a constant.
    const lodMax = src.followImage ? u.mipLevels.density : float(src.lodLevels);
    const followTextures = src.followImage ? this.#pyramidTextures.get(u) : undefined;

    // uvw → sample index space, clamped to the voxels (his clamp-to-edge).
    const samplePos = (box: { min: VDBTslNode; size: VDBTslNode }, uvw: VDBTslNode) =>
      clamp(box.min.add(uvw.mul(box.size)).sub(0.5), box.min, box.min.add(box.size).sub(1.0));
    const level0 =
      src.fastTrilinear || src.cacheAccessor ? (id: number, p: VDBTslNode) => this.#sampleFn()(uint(id), p) : (id: number, p: VDBTslNode) => vdbSample(grids, id, p, { space: "index" });

    const densityAt = (uvw: VDBTslNode, lodIn: VDBTslNode): VDBTslNode => {
      const p = samplePos(u.boxes.density!, uvw).toVar();
      if (!useMipmaps || lod === "none") return level0(ids.density, p);
      const l = clamp(lodIn, 0.0, lodMax).toVar();
      if (lod === "node-average") return this.#nodeLodFn()(uint(ids.density), p, l);
      const r = float(0.0).toVar();
      // coarse-pyramid
      const pyr = this.pyramid!;
      if (src.followImage && !followTextures) throw bad("createNodes needs this source's uniforms (vdbVolumeUniforms(source)).");
      const pyrMap = followTextures ? followTextures.map : texture3D(pyr.map);
      const pyrMip = followTextures ? followTextures.mip : pyr.mip ? texture3D(pyr.mip) : null;
      const { sampleMipmapped } = vdbMipSampler(u.pyramidTexels, u.pyramidAtlasSize);
      const coarse = (rel: VDBTslNode) => (pyrMip ? sampleMipmapped(pyrMap, pyrMip, u.pyramidLevels, uvw, rel).r : pyrMap.sample(uvw).level(float(0.0)).r);
      // K3.2d: one level-0 call site (each inlines the whole tree walk), shared by the two branches below.
      const fine = float(0.0).toVar();
      If(l.lessThan(u.pyramidLevel), () => {
        fine.assign(level0(ids.density, p));
      });
      If(l.lessThanEqual(0.0), () => {
        r.assign(fine);
      })
        .ElseIf(l.lessThan(u.pyramidLevel), () => {
          r.assign(mix(fine, coarse(float(0.0)), l.div(u.pyramidLevel)));
        })
        .Else(() => {
          r.assign(coarse(l.sub(u.pyramidLevel)));
        });
      return r;
    };

    // L892
    const getLod = (localSize: VDBTslNode) => max(0.0, log2(localSize.mul(vdbMaxComponent(u.gridTexels))));

    // L914–936: extinction per world unit; the mask at local + 0.5 (no offset or wrap), as his.
    const sampleExtinction = (localPosition: VDBTslNode, lodIn: VDBTslNode) => {
      const mask = src.channels.mask ? level0(ids.mask!, samplePos(u.boxes.mask!, localPosition.add(0.5))) : float(1.0);
      const raw = densityAt(gridUv(localPosition), lodIn);
      const value = (clamp01 ? clamp(raw, 0.0, 1.0) : raw).mul(mask);
      return select(value.greaterThan(u.densityCutoff), value, float(0.0)).mul(u.densityScale).mul(u.extinction).div(u.worldScale);
    };

    // L960–971 (level 0)
    const sampleAlbedo = (localPosition: VDBTslNode) => {
      if (!src.channels.albedo) return u.baseColor.mul(vec3(1.0));
      const p = samplePos(u.boxes.albedo!, gridUv(localPosition)).toVar();
      const a = ids.albedo!;
      const rgb = Array.isArray(a) ? vec3(level0(a[0], p), level0(a[1], p), level0(a[2], p)) : vec3(level0(a as number, p));
      return u.baseColor.mul(clamp(rgb, 0.0, 1.0));
    };

    // L990–1017 (level 0), colour via the caller's hook (V24).
    const emissionColor = o.emissionColor ?? (() => vec3(1.0));
    const sampleEmission = (localPosition: VDBTslNode) => {
      if (!src.channels.emission) return vec3(0.0);
      const range = u.temperatureRange;
      const value = level0(ids.emission!, samplePos(u.boxes.emission!, gridUv(localPosition)));
      const temperature = mix(range.x, range.y, value);
      const power = pow(temperature.div(max(range.y, 1.0)), 4.0);
      const color = select(temperature.greaterThan(0.0), emissionColor(temperature), vec3(0.0));
      return color.mul(power).mul(u.emissiveIntensity);
    };

    // HDDA empty-space query (R1 §4.3 "HDDA gains"): exact only where the bound covers every voxel a
    // sample reads: level 0 (the trilinear 2x2x2 cell block), or a coarse level whose taps reach at
    // most 8 voxels past the cell (dilated 3x3x3 block): node averages up to level 3, pyramid bases
    // up to level 2. Wrapping or an offset moves samples off the ray's cells: no skipping then.
    const skipEmpty = (origin: VDBTslNode, direction: VDBTslNode, t0: VDBTslNode, t1: VDBTslNode, lodIn: VDBTslNode) => {
      const box = u.boxes.density!;
      const l = useMipmaps && lod !== "none" ? clamp(lodIn, 0.0, lodMax) : float(0.0);
      const dilate = int(0).toVar();
      // Cost rule (K3.2 A/B): a walk visits 8³ cells at 8 node lookups each, so it only pays while samples are at
      // most one cell apart (the march's lod ≤ 3, i.e. steps ≤ 8 voxels); sparser marches sample everything.
      const enabled = u.wrap3D
        .equal(int(THREE.ClampToEdgeWrapping))
        .and(vdbMaxComponent(u.offset3D.abs()).equal(0.0))
        .and(lodIn.lessThanEqual(3.0))
        .toVar();
      if (useMipmaps && lod === "node-average") {
        If(l.greaterThan(0.0), () => {
          dilate.assign(1);
          If(l.greaterThan(3.0), () => {
            enabled.assign(false);
          });
        });
      } else if (useMipmaps && lod === "coarse-pyramid") {
        If(l.greaterThan(0.0), () => {
          dilate.assign(1);
          If(u.pyramidLevel.greaterThan(2.0).or(l.greaterThan(2.0)), () => {
            enabled.assign(false);
          });
        });
      }
      const result = vec2(t0, 1e30).toVar();
      If(enabled, () => {
        const o2 = box.min.sub(0.5).add(origin.add(0.5).mul(box.size));
        const d2 = direction.mul(box.size);
        // The mask multiplies the density, so with a mask only a zero bound is a safe skip.
        const cutoff = src.channels.mask ? float(0.0) : max(u.densityCutoff, 0.0);
        result.assign(this.skipEmptyFn()(uint(ids.density), o2, d2, t0, t1, cutoff, dilate.toUint()));
      });
      return result;
    };

    return { getLod, sampleExtinction, sampleAlbedo, sampleEmission, ...vdbBoxNodes, skipEmpty };
  }

  dispose(): void {
    this.pyramid?.gpu?.destroy();
    this.pyramid?.map.dispose();
    this.pyramid?.mip?.dispose();
    this.#emptyAtlas?.dispose();
  }
}
