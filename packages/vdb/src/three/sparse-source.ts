// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0): the source half of createVolumeUniforms /
// createVolumeNodes (frameShaderChunks.js L729–763, L834–1028), whose sparse-grid port (the TSL half) was deleted in K6.11
// (D35); this file is the handle the core's sparse source is made from.
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Divergences: DIVERGES.md (K3.2 rows), V23–V25, V145.

/**
 * The sparse volume source (K3.2, R1 §4.2–4.3): a `VDBGridPool` grid behind the same seam as the
 * dense `.wvdb` source, so `VDBFrameVolume` renders NanoVDB grids.
 *
 *   const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] });
 *   const source = vdbSparseSource(grids, { density: id }, { lod: "coarse-pyramid", images: { density: image } });
 *   const volume = new VDBFrameVolume(source, { extinction: 100 });   // box: the grid's world box
 *
 * K6.8a / K6.11: the source is a handle (grids, ids, options, boxes, stats, the image-follow state). `VDBFrameRenderer`
 * reads it into the core's sparse source (`vdbSparseSourceGpu`), which builds its own pyramids and reads the grids itself.
 * The handle holds no GPU object. (Custom TSL materials sample a pooled grid through `VDBGridNode`; custom WGSL or compute
 * uses the core source's `wgsl()`.)
 *
 * Positions are his local unit box [-0.5, 0.5]³; uvw = local + 0.5 (+ offset, wrapped) maps onto the
 * channel grid's index cell box `[min − ½, max + ½]` (voxel centres at integers), so uvw
 * (i + 0.5) / N lands on voxel min + i as on a dense texel centre. The sample position is clamped to
 * `[min, max]` (his clamp-to-edge). Level 0 is the pool's trilinear sample; with the node helper
 * `'fastTrilinear'` (recommended, K3.2d) it is `vdbSampleFast` through one accessor per shader invocation
 * (`cacheAccessor`): the same values, 1.5–2× faster on the hardware GPU (docs/archive/handoffs/K3-2d.md). (The core's
 * sparse source reads these options.)
 *
 * LOD (`lod`, his `sampleExtinction(localPos, lod)` at lod ≥ 1; R1 §4.3):
 * - `'none'`: level 0 always (no prefilter; aliases when a froxel spans many voxels). lodLevels 0.
 * - `'coarse-pyramid'` (default, the K3.2 A/B pick): a dense r16f pyramid from L₀ = max(1, ⌈log2(maxTexels / pyramidMax)⌉):
 *   the base is a box resample of the grid, the levels above it his box mip atlas. lod in (0, L₀)
 *   blends level 0 toward L₀; lod ≥ L₀ samples the pyramid as his mip atlas. GPU bytes: ~2 B × (base
 *   + atlas) texels. The core's sparse source builds it on the GPU from the pool grid (K3.2a: the core's
 *   `vdbCreateBoxPyramid`, no CPU image needed) and rebuilds it when the grid's image changes
 *   (`pool.replace`: a sequence frame). K6.11: the CPU build (`pyramidBuild: 'cpu'`, the three textures' static path) is
 *   deleted; `@pmndrs/vdb/tools` `vdbBoxPyramid` stays the GPU build's CPU twin.
 * - `'node-average'`: no extra memory: trilinear over the node averages (`vdbNodeAverage`) at
 *   level 3 (leaves, the exact 8³ box mean for fog leaves) and level 7 (lower nodes, the mean of
 *   their active voxels); lod in (0, 3) blends level 0 → 3, (3, 7) blends 3 → 7. Needs stats.
 *
 * HDDA: `skipEmpty` lets a march jump samples that are provably 0 (`VDBFrameVolume` `hdda`, default
 * off since D30 / D31: it slowed the shadow march on Apple GPUs; the image is the same either way).
 *
 * Following the image (`followImage`, G16a): the index boxes, world box and pyramid are fixed when the source is
 * made unless it follows the grids' images. Then a `pool.replace` that changes a channel grid's box (a coarse-first
 * swap: the preview, then the full grid in the same id) is re-read from `pool.info` once the new image is resident:
 * the new layout, the LOD range and a new `version`, which a `VDBFrameVolume` follows with its box. No shader is rebuilt.
 *
 * Mask, albedo and emission grids (optional, scalar; albedo may be three grids for RGB) are read at
 * level 0 only. One VDB module per shader: every grid goes through the one `VDBGridNode`.
 */

import { VDBError } from "../core/errors.js";
import type { VDBVolumeChannels } from "../core/source/types.js";
import { vdbBoxPyramidLayout } from "../tools/frame/pyramid-layout.js";
import { leafVoxels, treeNodes } from "../tools/cpu/march.js";
import type { VDBGridNode } from "./grid-node.js";
import type { VDBVolumeSourceInfo } from "../core/source/types.js";

type Vec3 = [number, number, number];

/** The LOD strategies of the sparse source (R1 §4.3); see the file comment. */
export type VDBSparseLod = "none" | "coarse-pyramid" | "node-average";

/** The pool grid ids of the channels. `albedo`: one grey grid or three (RGB). */
export interface VDBSparseSourceGrids {
  /** Pool id of the density grid. */
  density: number;
  /** Pool id of an optional mask grid (multiplies the density). */
  mask?: number;
  /** Pool id of one grey albedo grid, or three ids for RGB. */
  albedo?: number | readonly [number, number, number];
  /** Pool id of an optional emission (temperature) grid. */
  emission?: number;
}

/** Inclusive voxel index box. */
export interface VDBIndexBox {
  /** Smallest voxel index (inclusive). */
  min: Vec3;
  /** Largest voxel index (inclusive). */
  max: Vec3;
}

/** Options of {@link vdbSparseSource}. */
export interface VDBSparseSourceOptions {
  /**
   * LOD strategy. Default `'coarse-pyramid'`, the K3.2 A/B's pick (closest to his mips, low flicker, cheap to
   * sample; needs `images.density`); see docs/archive/handoffs/K3-2.md.
   */
  lod?: VDBSparseLod;
  /** `'coarse-pyramid'`: base level texels on the longest side at most (default 128). */
  pyramidMax?: number;
  /** `'none'` (default): raw floats. `'clamp01'`: clamp each density sample to [0, 1] (his 8-bit range; parity A/B). */
  normalize?: "none" | "clamp01";
  /** CPU copies of the grid images: the density stats read `density`. */
  images?: { density?: Uint32Array };
  /**
   * Density stats (`normalizeDensity` divides by `meanNonZero`). Default: computed from `images.density`.
   * Either way they describe the image at creation; after the density grid's image changes (`pool.replace`,
   * its `imageRevision`) the source reads the new image back from the pool and recomputes them (K3.2e; see
   * `statsUpdates`).
   */
  stats?: { meanNonZero: number; max: number; activeVoxels: number };
  /** Density index box the unit box covers (default: the grid's index bbox). Aux grids use their own. */
  indexBox?: VDBIndexBox;
  /**
   * K3.6: index boxes the unit box covers in the aux grids (default: each grid's own index bbox). Grids converted
   * from one dense `.wvdb` level (`wvdbDenseToNvdb`) share the texture's box (`textureIndexBox`); pass it so a
   * sparse emission or mask channel lines up with the density (its active bbox is usually smaller).
   */
  auxIndexBoxes?: Partial<Record<"mask" | "albedo" | "emission", VDBIndexBox>>;
  /** World box (default: the density cell box through the grid's transform, as an AABB). */
  bbox?: [Vec3, Vec3];
  /**
   * K3.2d: level-0 samples through `vdbSampleFast` (the 8 trilinear taps as lower-node child lookups, no
   * tree walk per tap; bit-identical to `vdbSample`). Needs the `VDBGridNode` helper `'fastTrilinear'`.
   * Default: on when the node has the helper.
   */
  fastTrilinear?: boolean;
  /**
   * K3.2d: keep one accessor per shader invocation (a WGSL `var<private>`) across the samples of a march
   * instead of a fresh one per sample, so consecutive samples reuse its cached leaf/lower/upper nodes. The
   * values are the same either way. Default true (the K3.2d hardware A/B: with `fastTrilinear`, whose taps
   * start from the cached lower node, 10–25% off the source passes; alone it changes little).
   */
  cacheAccessor?: boolean;
  /**
   * G16a: follow the grids' images. When a channel grid's `imageRevision` moves (a `pool.replace`, e.g. a
   * `vdbStreamCoarseFirst` swap from the preview to the full grid) and every channel grid is resident again, the
   * source re-reads its index boxes, world box and transform from `pool.info`, rebuilds the coarse pyramid at the
   * new size and bumps `version` (a `VDBFrameVolume` whose box is the source's follows it). Boxes, sizes and LOD
   * levels are uniforms in this mode, so nothing recompiles; give the `VDBGridNode` every grid type the images will
   * have (`types`, e.g. `VDBCoarseFirstEvent.types`): a new image of a type the node does not compile throws
   * `VDB_TYPE_NOT_COMPILED` when the source re-reads it (it would read 0). Default false: the boxes are fixed when the
   * source is made. Not with `indexBox`, `auxIndexBoxes` or `bbox`.
   */
  followImage?: boolean;
}

/** What the source needs from a pool: `info` and the ids' grids (a `VDBGridPool`). */
interface PoolInfoLike {
  info(id: number): { indexBBox: { min: Vec3; max: Vec3 }; indexToWorld: Float64Array; hasStats: boolean; capacity: number; type: string; imageRevision?: number; resident?: boolean };
}

/** The sparse source in three. */
export interface VDBThreeSparseSource extends VDBVolumeSourceInfo {
  /** Always `"sparse"`. */
  readonly kind: "sparse";
  /** World box of the volume (`[min, max]`), the default `VDBFrameVolume` box. */
  readonly bbox: [[number, number, number], [number, number, number]];
  /** The grid node the source reads through. */
  readonly grids: VDBGridNode;
  /** The pool ids of the channels. */
  readonly ids: Readonly<VDBSparseSourceGrids>;
  /** The LOD strategy in use. */
  readonly lod: VDBSparseLod;
  /** Density normalization: `'clamp01'` clamps to [0, 1], `'none'` leaves it. */
  readonly normalize: "none" | "clamp01";
  /** The density index box the unit box covers. */
  readonly indexBox: VDBIndexBox;
  /** K3.2d: level 0 through `vdbSampleFast` (see the option). */
  readonly fastTrilinear: boolean;
  /** K3.2d: one accessor per invocation across samples (see the option). */
  readonly cacheAccessor: boolean;
  /** G16a: the boxes follow the grids' images (see the option). */
  readonly followImage: boolean;
  /** G16a: times the boxes (and the pyramid, if its size changed) were re-read after an image change. */
  readonly layoutUpdates: number;
  /**
   * GPU bytes of the channel grids' pool ranges. The LOD pyramid is the core source's (demo 15:
   * `frameRenderer.coreVolumeOf(volume).slot.source.gpuBytes`).
   */
  readonly gpuBytes: { readonly pool: number };
  /** Density stats (`normalizeDensity`'s `meanNonZero`): of the density image at creation, then of the latest one read back (K3.2e). */
  readonly stats: { readonly meanNonZero: number; readonly max: number; readonly activeVoxels: number };
  /** Times the stats were recomputed after an image change (K3.2e). */
  readonly statsUpdates: number;
  /**
   * K3.2e: if the density grid's image changed since the stats were computed and it is resident, reads it
   * back from the pool (own encoder, submitted) and recomputes the stats (tools' twin of the CPU path);
   * resolves to whether it did. K6.11: nothing calls it for you (the TSL uniforms that did are deleted); `VDBFrameRenderer`
   * copies a changed `stats` into its core slot, which otherwise re-reads the mean itself a frame or two later. Call it
   * after `pool.update` was submitted, as `grids.update()`; await it when the first frame needs the new mean.
   */
  refreshStats(): Promise<boolean>;
  /** Drops the listeners. */
  dispose(): void;
}

const bad = (message: string) => new VDBError("VDB_BAD_OPTION", `vdbSparseSource: ${message}`);

/** The non-zero density stats of an image (all leaf voxels, active or not, and active tiles). */
function imageStats(image: Uint32Array, clamp01: boolean): { meanNonZero: number; max: number; activeVoxels: number } {
  let sum = 0;
  let count = 0;
  let maxValue = 0;
  const add = (v: number, n: number) => {
    const x = clamp01 ? Math.min(Math.max(v, 0), 1) : v;
    if (x !== 0) {
      sum += x * n;
      count += n;
      maxValue = Math.max(maxValue, x);
    }
  };
  for (const node of treeNodes(image)) {
    for (const t of node.activeTiles) add(t.value, t.dim ** 3);
    if (node.level !== 0) continue;
    const { values } = leafVoxels(image, node.address);
    for (let n = 0; n < 512; n++) add(values[n]!, 1);
  }
  return { meanNonZero: count > 0 ? sum / count : 0, max: maxValue, activeVoxels: count };
}

/** World AABB of an index cell box through a 3x4 row-major affine. */
function worldBox(box: VDBIndexBox, m: Float64Array): [Vec3, Vec3] {
  const lo: Vec3 = [Infinity, Infinity, Infinity];
  const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < 8; c++) {
    const p = [c & 1 ? box.max[0] + 0.5 : box.min[0] - 0.5, c & 2 ? box.max[1] + 0.5 : box.min[1] - 0.5, c & 4 ? box.max[2] + 0.5 : box.min[2] - 0.5];
    for (let r = 0; r < 3; r++) {
      const v = m[r * 4]! * p[0]! + m[r * 4 + 1]! * p[1]! + m[r * 4 + 2]! * p[2]! + m[r * 4 + 3]!;
      lo[r] = Math.min(lo[r]!, v);
      hi[r] = Math.max(hi[r]!, v);
    }
  }
  return [lo, hi];
}

/** Local unit box → sample index space: p = min − ½ + (local + ½)·size (column-major). */
function boxToSourceOf(b: VDBIndexBox, texels: Vec3): Float32Array {
  const [sx, sy, sz] = texels;
  return new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, b.min[0] - 0.5 + 0.5 * sx, b.min[1] - 0.5 + 0.5 * sy, b.min[2] - 0.5 + 0.5 * sz, 1]);
}

const texelsOf = (b: VDBIndexBox): Vec3 => [b.max[0] - b.min[0] + 1, b.max[1] - b.min[1] + 1, b.max[2] - b.min[2] + 1];

/**
 * The handle (K6.8a): grids, ids, options, boxes, stats and the image-follow state. The frame renderer reads these into
 * the core's sparse source.
 */
class SparseSource implements VDBThreeSparseSource {
  readonly kind = "sparse" as const;
  readonly grids: VDBGridNode;
  readonly ids: Readonly<VDBSparseSourceGrids>;
  readonly lod: VDBSparseLod;
  readonly normalize: "none" | "clamp01";
  readonly channels: VDBVolumeChannels;
  // G16a: the layout fields below are re-read after an image change when `followImage` is on.
  indexBox: VDBIndexBox;
  auxBoxes: Record<"mask" | "albedo" | "emission", VDBIndexBox | null>;
  texels: [number, number, number];
  lodLevels: number;
  stats: { meanNonZero: number; max: number; activeVoxels: number };
  statsUpdates = 0;
  boxToSource: Float32Array;
  bbox: [Vec3, Vec3];
  readonly fastTrilinear: boolean;
  readonly cacheAccessor: boolean;
  readonly followImage: boolean;
  readonly pyramidOptions: { readonly pyramidMax?: number };
  layoutUpdates = 0;
  /** G16a: the newest channel imageRevision the layout describes, the pool revision last checked. */
  #layoutRevision = -1;
  #layoutPoolRevision = -1;
  readonly #poolIds: number[];
  #versionSeen = { poolRevision: -1, value: 0 };
  /** K3.2e: the density imageRevision the stats describe, the pool revision last checked, and a readback in flight. */
  #statsRevision = -1;
  #statsPoolRevisionSeen = -1;
  #statsReadback: Promise<boolean> | null = null;

  constructor(grids: VDBGridNode, ids: VDBSparseSourceGrids, o: VDBSparseSourceOptions) {
    const pool = grids.pool as unknown as Partial<PoolInfoLike>;
    if (typeof pool.info !== "function") throw bad("the VDBGridNode's pool has no info(); pass a VDBGridPool.");
    const info = (id: number) => (pool as PoolInfoLike).info(id);
    const lod = o.lod ?? "coarse-pyramid";
    if (lod !== "none" && lod !== "coarse-pyramid" && lod !== "node-average") throw bad(`unknown lod ${JSON.stringify(lod)}; use 'none', 'coarse-pyramid' or 'node-average'.`);
    const normalize = o.normalize ?? "none";
    if (normalize !== "none" && normalize !== "clamp01") throw bad(`unknown normalize ${JSON.stringify(normalize)}; use 'none' or 'clamp01'.`);
    const helpers = new Set<string>(grids.options.helpers ?? []);
    if (grids.options.profile !== "march" || !helpers.has("cellBoundTrilinear") || !helpers.has("hddaCellEnd") || (lod === "node-average" && !helpers.has("nodeAverage"))) {
      throw bad(
        `make the VDBGridNode with profile 'march' and helpers ['cellBoundTrilinear', 'hddaCellEnd'${lod === "node-average" ? ", 'nodeAverage'" : ""}] ` +
          `(got profile '${grids.options.profile}', helpers [${[...helpers].join(", ")}]).`,
      );
    }
    const fastTrilinear = o.fastTrilinear ?? helpers.has("fastTrilinear");
    if (fastTrilinear && !helpers.has("fastTrilinear")) throw bad("fastTrilinear needs the VDBGridNode helper 'fastTrilinear' (add it to helpers), or pass fastTrilinear: false.");
    if (fastTrilinear && grids.options.sampler === "nearest") throw bad("fastTrilinear needs the VDBGridNode's sampler 'trilinear'.");
    this.fastTrilinear = fastTrilinear;
    this.cacheAccessor = o.cacheAccessor ?? true;
    const followImage = o.followImage ?? false;
    if (typeof followImage !== "boolean") throw bad("followImage must be a boolean.");
    if (followImage) {
      const fixed = (["indexBox", "auxIndexBoxes", "bbox"] as const).filter((k) => o[k] !== undefined);
      if (fixed.length) throw bad(`followImage re-reads every box from the pool; drop ${fixed.join(", ")} (fixed boxes).`);
    }
    this.followImage = followImage;
    const density = info(ids.density);
    if (lod === "node-average" && !density.hasStats) throw bad("lod 'node-average' reads node statistics; this grid has none (rebuild it with stats).");
    this.grids = grids;
    const channelIds = [ids.density, ids.mask, ids.emission, ...(Array.isArray(ids.albedo) ? ids.albedo : [ids.albedo])].filter((v): v is number => v !== undefined);
    this.checkTypes(
      channelIds.map((id) => ({ id, type: info(id).type })),
      "a channel grid",
    );
    this.ids = Object.freeze({ ...ids, ...(Array.isArray(ids.albedo) ? { albedo: [...ids.albedo] as unknown as [number, number, number] } : {}) });
    this.lod = lod;
    this.normalize = normalize;
    this.indexBox = o.indexBox ? { min: [...o.indexBox.min], max: [...o.indexBox.max] } : { min: [...density.indexBBox.min], max: [...density.indexBBox.max] };
    const albedoId = Array.isArray(ids.albedo) ? ids.albedo[0] : (ids.albedo as number | undefined);
    const box = (id: number | undefined, given?: VDBIndexBox) =>
      id === undefined ? null : given ? { min: [...given.min] as Vec3, max: [...given.max] as Vec3 } : { min: [...info(id).indexBBox.min] as Vec3, max: [...info(id).indexBBox.max] as Vec3 };
    const aux = o.auxIndexBoxes ?? {};
    this.auxBoxes = { mask: box(ids.mask, aux.mask), albedo: box(albedoId, aux.albedo), emission: box(ids.emission, aux.emission) };
    this.channels = { mask: ids.mask !== undefined, albedo: ids.albedo !== undefined, emission: ids.emission !== undefined };
    const b = this.indexBox;
    this.texels = texelsOf(b);
    if (this.texels.some((t) => !(t > 0))) throw bad("the density index box is empty.");
    const image = o.images?.density;
    if (o.stats) this.stats = { ...o.stats };
    else if (image) this.stats = imageStats(image, normalize === "clamp01");
    else throw bad("pass images.density (the CPU image, for the density stats) or stats.");
    this.#statsRevision = (density as { imageRevision?: number }).imageRevision ?? -1;
    this.boxToSource = boxToSourceOf(b, this.texels);
    this.bbox = o.bbox ? [[...o.bbox[0]], [...o.bbox[1]]] : worldBox(b, density.indexToWorld);

    this.pyramidOptions = o.pyramidMax !== undefined ? { pyramidMax: o.pyramidMax } : {};
    this.lodLevels = this.#lodLevelsOf(this.texels);
    this.#poolIds = [...new Set(channelIds)];
    this.#layoutRevision = this.#poolIds.reduce((r, id) => Math.max(r, info(id).imageRevision ?? 0), 0);
  }

  /** The LOD range: 0 (`'none'`), 7 (`'node-average'`), the pyramid layout's levels (`'coarse-pyramid'`). */
  #lodLevelsOf(texels: Vec3): number {
    if (this.lod === "none") return 0;
    if (this.lod === "node-average") return 7;
    const { pyramidMax } = this.pyramidOptions;
    return vdbBoxPyramidLayout(texels, pyramidMax !== undefined ? { pyramidMax } : {}).lodLevels;
  }

  /** GPU bytes of the channel grids' pool ranges. */
  get gpuBytes(): { pool: number } {
    const pool = this.grids.pool as unknown as PoolInfoLike;
    let bytes = 0;
    for (const id of this.#poolIds) {
      try {
        bytes += pool.info(id).capacity;
      } catch {
        // A removed grid holds no pool range.
      }
    }
    return { pool: bytes };
  }

  /**
   * G16a (`followImage`): re-reads the layout when a channel grid's image changed and every channel grid is resident
   * again (the update that writes the new image has run: the GPU must not read it through the old boxes, nor the
   * old image through the new ones). Cheap when the pool's revision did not move; called from `version` (a
   * `VDBFrameVolume` asks once per frame, before it places its box) and from the render updates.
   */
  syncLayout(): boolean {
    if (!this.followImage) return false;
    const pool = this.grids.pool as unknown as { revision?: number } & PoolInfoLike;
    if (typeof pool.revision !== "number" || pool.revision === this.#layoutPoolRevision) return false;
    let revision = 0;
    let resident = true;
    const infos: { id: number; type: string }[] = [];
    for (const id of this.#poolIds) {
      let i: ReturnType<PoolInfoLike["info"]>;
      try {
        i = pool.info(id);
      } catch {
        return false; // a removed grid: keep the last layout
      }
      revision = Math.max(revision, i.imageRevision ?? 0);
      if (i.resident === false) resident = false;
      infos.push({ id, type: i.type });
    }
    if (revision === this.#layoutRevision) {
      this.#layoutPoolRevision = pool.revision;
      return false;
    }
    if (!resident) return false; // look again once the update has written it
    // A new image of a type the module does not decode would read 0 through the new boxes, silently: say so instead
    // (every call until the source is remade over a node that compiles it).
    this.checkTypes(infos, "a channel grid's new image");
    this.#relayout();
    this.#layoutRevision = revision;
    this.#layoutPoolRevision = pool.revision;
    return true;
  }

  /**
   * Throws `VDB_TYPE_NOT_COMPILED` when a channel grid's type is not among the `VDBGridNode`'s compiled `types` (those
   * grids read 0). A node stand-in without `options.types` is not checked.
   */
  private checkTypes(infos: readonly { id: number; type: string }[], what: string): void {
    const compiled = (this.grids.options as { types?: readonly string[] }).types;
    if (!compiled) return;
    const missing = infos.filter((i) => !compiled.includes(i.type));
    if (missing.length === 0) return;
    throw new VDBError(
      "VDB_TYPE_NOT_COMPILED",
      `vdbSparseSource: ${what} is ${missing.map((m) => `grid ${m.id}: ${m.type}`).join(", ")}, but the VDBGridNode compiles only ` +
        `${compiled.join(", ")}; it would read 0. Make the VDBGridNode with types that include every type the images will have ` +
        `(a coarse-first stream lists them in VDBCoarseFirstEvent.types: types: event.types), then make the source over it.`,
    );
  }

  #relayout(): void {
    const pool = this.grids.pool as unknown as PoolInfoLike;
    const ids = this.ids;
    const density = pool.info(ids.density);
    const b: VDBIndexBox = { min: [...density.indexBBox.min], max: [...density.indexBBox.max] };
    const old = this.indexBox;
    const moved = b.min.some((v, a) => v !== old.min[a]) || b.max.some((v, a) => v !== old.max[a]);
    this.indexBox = b;
    this.texels = texelsOf(b);
    this.boxToSource = boxToSourceOf(b, this.texels);
    this.bbox = worldBox(b, density.indexToWorld);
    const auxBox = (id: number | undefined): VDBIndexBox | null => (id === undefined ? null : { min: [...pool.info(id).indexBBox.min] as Vec3, max: [...pool.info(id).indexBBox.max] as Vec3 });
    this.auxBoxes = { mask: auxBox(ids.mask), albedo: auxBox(Array.isArray(ids.albedo) ? ids.albedo[0] : (ids.albedo as number | undefined)), emission: auxBox(ids.emission) };
    if (moved) this.lodLevels = this.#lodLevelsOf(this.texels);
    this.layoutUpdates++;
  }

  /**
   * Content version: the newest `imageRevision` of the channel grids (K3.2a), so a `pool.replace` (a
   * sequence frame) changes it and the 'frame' lighting caster signature (V44) draws the shadow grid
   * again. Read through the pool's `revision` first (cheap when nothing changed).
   */
  get version(): number {
    // G16a: a following source's version is the image revision its layout describes (it moves once the new image is
    // resident and the boxes were re-read), so a VDBFrameVolume re-reads its box and the uniforms then.
    if (this.followImage) {
      this.syncLayout();
      return this.#layoutRevision;
    }
    const pool = this.grids.pool as unknown as { revision?: number; info(id: number): { imageRevision?: number } };
    if (typeof pool.revision !== "number") return 0;
    if (pool.revision !== this.#versionSeen.poolRevision) {
      let value = 0;
      for (const id of this.#poolIds) {
        try {
          value = Math.max(value, pool.info(id).imageRevision ?? 0);
        } catch {
          // A removed grid: keep the others' revision.
        }
      }
      this.#versionSeen = { poolRevision: pool.revision, value };
    }
    return this.#versionSeen.value;
  }

  refreshStats(): Promise<boolean> {
    if (this.#statsReadback) return this.#statsReadback;
    const pool = this.grids.pool as unknown as {
      revision?: number;
      device?: GPUDevice;
      buffer?: GPUBuffer;
      info(id: number): { imageRevision?: number; resident?: boolean; imageOffset?: number | null; imageBytes?: number };
    };
    // Cheap test first (once per render): the pool's revision moves on any change.
    if (typeof pool.revision !== "number" || pool.revision === this.#statsPoolRevisionSeen || !pool.device || !pool.buffer) return Promise.resolve(false);
    let info: ReturnType<typeof pool.info>;
    try {
      info = pool.info(this.ids.density);
    } catch {
      return Promise.resolve(false); // removed: keep the last stats
    }
    const revision = info.imageRevision ?? -1;
    if (revision === this.#statsRevision) {
      this.#statsPoolRevisionSeen = pool.revision;
      return Promise.resolve(false);
    }
    // Not written yet (the update that writes it is still to come): look again next time.
    if (!info.resident || info.imageOffset === null || info.imageOffset === undefined || !info.imageBytes) return Promise.resolve(false);
    const { device, buffer } = pool;
    const bytes = info.imageBytes;
    const staging = device.createBuffer({ label: "vdbSparseSource stats readback", size: bytes, usage: 0x1 | 0x8 }); // MAP_READ | COPY_DST (no WebGPU globals in Node tests)
    const encoder = device.createCommandEncoder({ label: "vdbSparseSource stats readback" });
    encoder.copyBufferToBuffer(buffer, info.imageOffset, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    const seen = pool.revision;
    this.#statsReadback = staging
      .mapAsync(0x1) // GPUMapMode.READ
      .then(() => {
        const image = new Uint32Array(staging.getMappedRange().slice(0));
        this.stats = imageStats(image, this.normalize === "clamp01");
        this.#statsRevision = revision;
        this.#statsPoolRevisionSeen = seen;
        this.statsUpdates++;
        return true;
      })
      .catch(() => false)
      .finally(() => {
        staging.destroy();
        this.#statsReadback = null;
      });
    return this.#statsReadback;
  }

  dispose(): void {
  }
}

/**
 * A sparse volume source over pool grids (see the file comment). `grids` must be a `VDBGridNode`
 * made with profile `'march'` and helpers `['cellBoundTrilinear', 'hddaCellEnd']` (+ `'nodeAverage'`
 * for that LOD). The source holds no GPU object (the pool's grid buffer is the pool's).
 *
 * @param grids A `VDBGridNode` with profile `march` and the helpers the LOD needs.
 * @param ids The pool ids of the density grid and optional channels.
 * @param options LOD, normalization, boxes, pyramid and sampling options.
 */
export function vdbSparseSource(grids: VDBGridNode, ids: VDBSparseSourceGrids, options: VDBSparseSourceOptions = {}): VDBThreeSparseSource {
  if (!grids || typeof grids !== "object" || !("module" in grids)) throw bad("pass a VDBGridNode.");
  if (!ids || !Number.isInteger(ids.density)) throw bad("ids.density must be a pool grid id.");
  return new SparseSource(grids, ids, options);
}
