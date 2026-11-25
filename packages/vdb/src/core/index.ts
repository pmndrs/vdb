/// <reference types="@webgpu/types" preserve="true" />
/**
 * @pmndrs/vdb (core) - the engine-free WebGPU core: a pool of NanoVDB grids in one storage buffer,
 * the generated `vdb*` WGSL module that reads it, and compute tools that run on it.
 *
 * Everything usable without a GPU (the `.nvdb` loader, the CPU oracle, the `.vdb` tooling) lives in
 * `@pmndrs/vdb/tools`. The core never imports `three` (test/boundaries.test.ts).
 *
 * K6.8 (0.2.0) adds the frame renderer (`vdbCreateFrameRenderer`), the sparse volume source (`vdbSparseSourceGpu`), the
 * level-set and point passes, the receiver WGSL for host materials, and `VDBSequence`.
 *
 * This barrel is the public API (D32). The pool's binary layout helpers and other test oracles live in
 * `./internal.ts`, imported by path by tests and harnesses; it is not a package export.
 */

export { VDBError } from "./errors.js";
export type { VDBErrorCode } from "./errors.js";

export { VDBGridPool } from "./pool/pool.js";
export type {
  VDBAddOptions,
  VDBBrickMajorant,
  VDBGridInfo,
  VDBGridPoolOptions,
  VDBTransformInput,
} from "./pool/pool.js";
/** The pool's GPU layout version, for `vdbCheckModule({ layoutVersion })`. */
export { VDB_GPU_LAYOUT_VERSION } from "./pool/layout.js";
export type { VDBVec3 } from "./pool/layout.js";
// K1.4a: WGSL module generator (docs/dev/design/K1-core.md §3).
export { vdbCheckModule, vdbModuleEntries, vdbModuleWgsl } from "./wgsl/generate.js";
export type {
  VDBBufferModuleOptions,
  VDBModuleCommonOptions,
  VDBModuleHelper,
  VDBModuleOptions,
  VDBModulePoolLike,
  VDBModuleProfile,
  VDBModuleTypeCheck,
  VDBSlotModuleOptions,
} from "./wgsl/types.js";
export { VDB_ALL_SCALAR_TYPES, VDB_GRID_TYPE_IDS, VDB_GRID_TYPE_NAMES } from "./types.js";
export type { VDBGridTypeName, VDBSlot } from "./types.js";
// K2.3: compute tools on the pool. The GPU tools carry the `Gpu` suffix: `@pmndrs/vdb/three` wraps them
// as `vdbGridStats` / `vdbDecodeToAtlas` (renderer in, promise out) and `@pmndrs/vdb/tools` has the
// `*Cpu` versions (D32).
export { vdbApplyValueTransform, vdbDecodeToAtlasGpu, vdbGridStatsGpu, VDB_VALUE_TRANSFORM_PRESETS } from "./compute/index.js";
// K3.2a: the sparse source's coarse pyramid built on the GPU.
export { vdbCreateBoxPyramid } from "./compute/index.js";
export type { VDBBoxPyramidGpu, VDBBoxPyramidGpuOptions, VDBBoxPyramidRecordOptions } from "./compute/index.js";
export type { VDBBoxPyramidLayout, VDBPyramidAtlasLevel } from "../tools/frame/pyramid-layout.js";
export type {
  VDBAtlasFilter,
  VDBAtlasFormat,
  VDBAtlas,
  VDBAtlasReadback,
  VDBDecodeToAtlasOptions,
  VDBGridStats,
  VDBGridStatsOptions,
  VDBReadback,
  VDBValueTransform,
  VDBValueTransformPreset,
} from "./compute/index.js";
// K3.0: the volume-source seam (R1 §4.2) and its dense `.wvdb` source.
export { vdbDenseSource } from "./source/dense.js";
export type { VDBDenseSource, VDBDenseSourceOptions } from "./source/dense.js";
export type { VDBVolumeChannels, VDBVolumeSource, VDBVolumeSourceInfo, VDBVolumeSourceWgslOptions } from "./source/types.js";

// K6.8 (0.2.0, D34): the core frame renderer — compute kernels on 3D textures on the host's device, recorded into the
// host's encoder (design docs/dev/design/K6-core-renderer.md §2). `./three` is a thin adapter over it.
export { vdbCreateFrameRenderer } from "./frame/renderer.js";
export type { VDBFrameRendererGpu, VDBFrameRendererGpuOptions } from "./frame/renderer.js";
export { vdbFrameDefaultSettings } from "./frame/settings.js";
export type { VDBFrameSettings, VDBFramePhaseName, VDBFrameMultipleScatteringMode } from "./frame/settings.js";
export { VDB_FRAME_ALL_LIGHTS, VDB_FRAME_LIGHT_BITS, vdbFrameDefaultVolumeParams } from "./frame/types.js";
export type {
  VDBEmissionColor,
  VDBFrameCamera,
  VDBFrameEmissionLightSettings,
  VDBFrameEmissionProbes,
  VDBFrameEnvironment,
  VDBFrameInput,
  VDBFrameLight,
  VDBFrameMat4,
  VDBFrameOccluder,
  VDBFrameOutputs,
  VDBFrameRedrawStats,
  VDBFrameTargets,
  VDBFrameTimestamps,
  VDBFrameUpdate,
  VDBFrameUpdateOptions,
  VDBFrameVec3,
  VDBFrameVolumeGpu,
  VDBFrameVolumeParams,
  VDBFrameWrap,
  VDBFroxelLayout,
  VDBVolumeShadowDrawn,
  VDBVolumeShadowMap,
  VDBVolumeShadowRequest,
} from "./frame/types.js";
/** EXPERIMENTAL (K6.3): the generic small-MLP data of `multipleScattering` `'mlp'` / `'neural'` (`VDBFrameRendererGpu.mlp`). */
export type { VDBFrameMlp, VDBFrameMlpLayer } from "../tools/frame/mlp.js";
// Host seams (design §2.5): the composite as a function in the host's own pass, and the receivers of the volume shadow
// maps (`vdbVolumeShadow`, `vdbVolumeShadowCube`, K6.4 `vdbVolumeShadowRgb`, `vdbVolumeShadowCubeRgb`) and of the emission
// probes (`vdbEmissionIrradiance`), with the layouts of their uniform buffers.
export { vdbFrameIntegrateEntries, vdbFrameIntegrateWgsl } from "./frame/wgsl/integrate.js";
export type { VDBFrameIntegrateWgslOptions } from "./frame/wgsl/integrate.js";
export { VDB_VOLUME_SHADOW_MAX_CASCADES, VDB_VOLUME_SHADOW_VIEW, vdbVolumeShadowWgsl } from "./frame/wgsl/shadows.js";
export { VDB_EMISSION_GRID, vdbEmissionProbeWgsl } from "./frame/wgsl/emission.js";
// K6.7: one sparse volume source for every volume of a pool (pyramids packed in one atlas).
export { vdbSparseSourceGpu } from "./source/sparse.js";
export type {
  VDBSparseIndexBox,
  VDBSparseLodGpu,
  VDBSparseSourceGpu,
  VDBSparseSourceGpuOptions,
  VDBSparseSourceUpdate,
  VDBSparseVolumeGpu,
  VDBSparseVolumeGpuOptions,
  VDBSparseVolumeGrids,
} from "./source/sparse.js";
export type { VDBVolumeSlot } from "./source/types.js";
// K6.6: level-set surfaces and point sprites (WGSL for host materials, raw passes for engine-free hosts).
export { vdbCreateLevelSetPassGpu, vdbLevelSetSurfaceWgsl, vdbCreatePointPassGpu, vdbPointSpriteWgsl } from "./surface/index.js";
export type { VDBLevelSetPassDraw, VDBLevelSetPassFrame, VDBLevelSetPassGpu, VDBLevelSetPassGpuOptions, VDBLevelSetPassPool, VDBPointPassData, VDBPointPassFrame, VDBPointPassGpu, VDBPointPassGpuOptions } from "./surface/index.js";
// K6.8 (Q7): the sequence player, engine-free (`./three` re-exports it).
export { VDBSequence } from "./sequence.js";
export type { VDBFrameLoader, VDBSequenceLoopMode, VDBSequenceOptions, VDBSequencePoolLike, VDBSequenceRange, VDBSequenceStats, VDBSequenceTarget } from "./sequence.js";
