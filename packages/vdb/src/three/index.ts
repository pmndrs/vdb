/**
 * @pmndrs/vdb/three — the three.js adapter over the engine-free core (D8).
 *
 * `getDevice` / `getGPUTexture` reach the renderer's WebGPU handles, so the
 * pool and the other pmndrs tools share one device. `VDBGridNode` puts a
 * `VDBGridPool` into three's node system (`vdbSample`, `vdbReadValue`, or the
 * module as a `wgslFn` include). `VDBVolumeMaterial` raymarches one pooled
 * grid; `VDBSequence` (the core's, re-exported) plays a frame sequence into one pool id with
 * `pool.replace`. `vdbGridStats` / `vdbDecodeToAtlas` run the core compute
 * tools through the renderer's device. `VDBFrameVolume` / `VDBFrameRenderer`
 * are mjurczyk's froxel renderer on a dense or sparse volume source, drawn by the
 * core frame renderer on three's device (K6.8: a thin adapter; `renderer.core`).
 *
 * There is no renderer bootstrap: create the device yourself
 * (`adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) })`)
 * and pass it, `new WebGPURenderer({ device })`, so the pool outlives
 * `renderer.dispose()`.
 *
 * This barrel is the public API (D32). The frame renderer's composite material, core-texture stand-ins and three
 * internals hooks live in `./internal.ts`, imported by path by tests and harnesses; it is not a package
 * export. Core types the adapter takes or returns (settings, emission colour, light bits) are the core's
 * declarations, re-exported.
 */

// K2.1/K2.2: the adapter over the engine-free core (docs/dev/design/K1-core.md §6). `getDevice` and
// `getGPUTexture` keep the helper names the other pmndrs tools use (D32).
export { getDevice, getGPUTexture } from "./device.js";
export type { VDBThreeBackendLike, VDBThreeRendererLike } from "./device.js";
export { VDBGridNode, vdbReadValue, vdbSample } from "./grid-node.js";
export type { VDBGridNodeOptions, VDBGridNodePoolLike, VDBSampleOptions, VDBTslNode } from "./grid-node.js";

// K2.3: the volume material and the sequence player on the pool.
export {
  VDBVolumeMaterial,
  vdbProxyGeometry,
  VDB_VOLUME_MAX_STEPS_CAP,
  VDB_VOLUME_SHADOW_STEPS_CAP,
  VDB_VOLUME_SAMPLE_BUDGET_CAP,
} from "./material.js";
export type { VDBVolumeMaterialParameters, VDBProxyGridInfo, VDBUintUniform } from "./material.js";
// K6.8: VDBSequence lives in the core (engine-free, Q7); re-exported here.
export { VDBSequence } from "../core/sequence.js";
export type {
  VDBSequenceLoopMode,
  VDBSequenceOptions,
  VDBSequenceRange,
  VDBSequenceStats,
  VDBSequenceTarget,
  VDBSequencePoolLike,
  VDBSequenceFrameLoader,
} from "../core/sequence.js";

// K3.7 (R1 §6 Step 7): level-set surfaces traced on the sparse pool, and point grids as sprites.
export { VDBLevelSetMaterial, VDB_LEVEL_SET_SAMPLES_CAP } from "./level-set-material.js";
export type { VDBLevelSetMaterialParameters } from "./level-set-material.js";
// G13a: the same trace on three's physical material (glass, clearcoat).
export { VDBLevelSetPhysicalMaterial } from "./level-set-physical-material.js";
export type { VDBLevelSetPhysicalMaterialParameters } from "./level-set-physical-material.js";
export { VDBPoints } from "./points.js";
export type { VDBPointsData, VDBPointsOptions } from "./points.js";

// K2.3: compute wrappers over the core tools (renderer in, promise out). The
// core's `vdbGridStats` / `vdbDecodeToAtlas` take a device and an encoder;
// the value transform is the core's `vdbApplyValueTransform` (CPU).
export { vdbGridStats, vdbDecodeToAtlas } from "./compute.js";
export type { VDBThreeGridStatsOptions, VDBGridStatsResult, VDBThreeDecodeToAtlasOptions, VDBDecodeToAtlasResult } from "./compute.js";

// K3.0 / K6.11: the dense `.wvdb` volume source (R1 §4.2), a handle `VDBFrameVolume` draws.
export { vdbDenseTextures } from "./volume-source.js";
export type { VDBThreeDenseSource, VDBThreeVolumeSource } from "./volume-source.js";

// K3.2: the sparse volume source (a VDBGridPool grid behind the same seam) with its LOD strategies.
export { vdbSparseSource } from "./sparse-source.js";
export type {
  VDBSparseLod,
  VDBSparseSourceGrids,
  VDBSparseSourceOptions,
  VDBThreeSparseSource,
  VDBIndexBox,
} from "./sparse-source.js";

// K4.16: coarse-first `.nvdb` streaming into one pool id (add the preview with
// the full size reserved, then replace). Engine-free glue; here because tools
// may not import the core.
export { vdbCoarseFirstApply, vdbStreamCoarseFirst } from "./coarse-first.js";
export type {
  VDBCoarseFirstBind,
  VDBCoarseFirstBindPoolLike,
  VDBCoarseFirstEvent,
  VDBCoarseFirstOptions,
  VDBCoarseFirstPoolLike,
  VDBCoarseFirstResult,
  VDBCoarseFirstSource,
} from "./coarse-first.js";

// K3.1–K3.6, K6.8: the froxel renderer (his FrameVolume / FrameRenderer) over the core frame renderer: the volume, the
// renderer, the measured media and the emission light.
export {
  VDBFrameVolume,
  VDBFrameRenderer,
  vdbFrameRenderer,
  // His helper name stays inside frame/ (diffable with upstream); the public name follows D17.
  getFrameMediumProps as vdbFrameMedium,
  VDBFrameEmissionLight,
} from "./frame/index.js";
export type { VDBFrameVolumeProps, VDBFramePassHook, VDBFramePhaseSettings, VDBFrameMediumVolumeProps, VDBFrameEmissionLightProps } from "./frame/index.js";
// K6.8 (0.2.0): the core's declarations the adapter takes or returns, re-exported (one declaration each).
export { VDB_FRAME_ALL_LIGHTS, VDB_FRAME_LIGHT_BITS } from "../core/frame/types.js";
export type { VDBEmissionColor, VDBFrameRedrawStats, VDBFrameTimestamps, VDBFrameUpdate } from "../core/frame/types.js";
export type { VDBFrameSettings, VDBFramePhaseName, VDBFrameMultipleScatteringMode } from "../core/frame/settings.js";
export type { VDBFrameMlp, VDBFrameMlpLayer } from "../tools/frame/mlp.js";
