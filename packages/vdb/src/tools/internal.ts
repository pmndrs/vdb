/**
 * @pmndrs/vdb/tools internals (D32): NOT a package export.
 *
 * Everything public (`./index.ts`) plus what tests, harnesses and scripts need that a user doesn't:
 * the CPU twins of the WGSL march profile and the frame-renderer passes (the parity oracles), the
 * `.nvdb` / `.wvdb` layout constants and chunk codecs, the stride-table accessors, the sparse source's
 * CPU pyramid, and the frame lighting maths. Import it by path (`src/tools/internal.ts`, or
 * `dist/tools/internal.js` from a script after `pnpm build`); its names can change without notice.
 */

export * from "./index.js";

export { FILE_HEADER_SIZE, FILE_METADATA_SIZE, GRID_DATA_SIZE, MAGIC_ASCII } from "./nvdb-file.js";
export {
  cellBoundTrilinear,
  defaultValueMax,
  getDim,
  gridHasStats,
  hddaCellEnd,
  hddaInit,
  hddaStep,
  hddaUpdate,
  hddaWalk,
  indexBBox,
  isActive,
  leafVoxels,
  nodeAverage,
  nodeMax,
  nodeStats,
  rayClip,
  rayClipSupport,
  forEachTreeRegion,
  treeNodes,
} from "./cpu/march.js";
export type {
  Hdda,
  HddaSegment,
  HddaWalkOptions,
  NodeLevel,
  NodeMaxOptions,
  NodeStats,
  RayClipSupportOptions,
  TreeNode,
  TreeRegion,
  Vec3,
} from "./cpu/march.js";
// P1.3 / P1.4: twin of the majorant WGSL profile.
export { majorantWalk } from "./cpu/majorant.js";
export type { MajorantSegment, MajorantWalkOptions } from "./cpu/majorant.js";
export { probeCoords, probePoints } from "./cpu/probe-coords.js";
export type { ProbeOpts } from "./cpu/probe-coords.js";
export {
  levelSetGradient,
  levelSetHit,
  levelSetHitReference,
  levelSetNdcDepth,
  levelSetNormal,
  levelSetPixelUv,
  levelSetSurface,
  levelSetViewRay,
  sampleTrilinearF32,
} from "./cpu/level-set.js";
export type { LevelSetHit, LevelSetHitOptions, LevelSetRay, LevelSetSurface, LevelSetSurfaceOptions } from "./cpu/level-set.js";
// K6.6: twins of the point-sprite WGSL.
export { pointSpriteClip, pointSpriteSize, pointSpriteVertex } from "./cpu/points.js";
export {
  atlasDims,
  atlasFromRaw,
  bboxSize,
  GRID_STATS_SUM_SCALE,
  rootStats,
  transformLeafValues,
} from "./cpu/compute.js";
export type { AtlasFromRawOptions, IndexBBox } from "./cpu/compute.js";
export { defineNumber, gridTypeConstantsForId } from "./cpu/stride-tables.js";
export type { GridTypeConstants } from "./cpu/stride-tables.js";
export {
  brickCounts as wvdbBrickCounts,
  brickOrigin as wvdbBrickOrigin,
  brickTotal as wvdbBrickTotal,
  decodeMask as decodeWvdbMask,
  decodePage as decodeWvdbPage,
  decodePoints as decodeWvdbPoints,
  encodeMask as encodeWvdbMask,
  encodePage as encodeWvdbPage,
  encodePoints as encodeWvdbPoints,
  inflateWvdbChunk,
  normalizeWvdbClass,
  wvdbChunks,
  wvdbChunkByteLimit,
  downsampleWvdbLevel,
  wvdbDensityStats,
  sampleWvdbDense,
  sampleWvdbDenseBox,
  sampleWvdbLevelSetSurfaceMean,
  wvdbFinestDimensions,
  sha256Hex,
  wvdbTexels,
  wvdbTexelStats,
  wvdbMipAtlas,
  wvdbMipLevelCount,
  wvdbMipLevelSize,
  wvdbToByte,
  wvdbDensityScale,
  sampleWvdbDenseTexture,
  // K4.22: partial LODs (the CPU twin of the dense source's GPU fill)
  WVDB_UPSAMPLE_WEIGHT_ONE,
  wvdbUpsampleAxis,
  wvdbUpsampleTexels,
  wvdbPendingBricks,
  wvdbDenseResolveTexture,
  wvdbMipAtlasLayout,
} from "./wvdb/index.js";
export type {
  WvdbBrickInput,
  DecodedPage as WvdbDecodedPage,
  DecodedPoints as WvdbDecodedPoints,
  WvdbLevelSetSource,
} from "./wvdb/index.js";
export {
  vdbFrameSliceDepth,
  vdbFrameFroxelSlice,
  vdbFrameSliceRange,
  vdbFrameViewPosition,
  vdbFrameFroxel,
  vdbFrameDistanceFade,
  vdbFrameBoxSteps,
  vdbFrameNoise,
  vdbFrameRand3,
  vdbFrameHenyeyGreenstein,
  vdbFramePhaseWeight,
  vdbFrameShadowAbsorption,
  vdbFrameShadowStep,
  vdbFrameShadowDepth,
  vdbFrameShadowTransmittance,
  vdbFrameShadowMarch,
  vdbFrameShadowedLight,
  vdbFrameInjectSlice,
  vdbFrameIntegrate,
  vdbSrgbOetf,
  vdbSrgbEotf,
  vdbFrameSrgbBlend,
  vdbFrameSceneViewDepth,
  vdbFrameHenyeyGreensteinColor,
  vdbFramePhaseColor,
  vdbFramePhaseWeightScaled,
  vdbFrameOctavesOff,
  vdbFrameMaxScatteringOctaves,
  vdbFrameScatteringOctaves,
  vdbFrameShadowMarchDepth,
  vdbFrameShadowOpticalDepth,
  vdbFrameOctaveLight,
  vdbFrameShadowTotal,
  vdbFrameTwoSidedDiffusion,
  vdbFrameShadowedLightColor,
} from "./frame/index.js";
export type {
  VDBFrameFroxelSample,
  VDBFrameOctaveSettings,
  Vec2 as VDBFrameVec2,
  Vec3 as VDBFrameVec3,
  Vec4 as VDBFrameVec4,
} from "./frame/index.js";
export {
  vdbPyramidBaseLevel,
  vdbBoxMipAtlasF32,
  vdbBoxPyramid,
  vdbFrameNodeAverageSample,
  vdbFrameSkipEmpty,
  vdbSparseDensityStats,
  vdbPyramidHalfBits,
  vdbPyramidToHalf,
  vdbSparseLodLevelsCpu,
  vdbSparseDensityCpu,
  vdbSparseDensityStochasticCpu,
  vdbSparseMaskCpu,
  vdbSparseAlbedoCpu,
  vdbSparseEmissionCpu,
  vdbSparseSkipSpanCpu,
} from "./frame/sparse.js";
export type { VDBFloatMipAtlas, VDBBoxPyramidOptions, VDBBoxPyramid, VDBSparseChannelCpu, VDBSparseVolumeCpu, VDBSparseSampleCpu } from "./frame/sparse.js";
export { vdbBoxPyramidLayout, vdbPyramidLevelSize, vdbPackPyramidBlocks } from "./frame/pyramid-layout.js";
export type { VDBBoxPyramidLayout, VDBPyramidAtlasLevel, VDBPyramidPacking } from "./frame/pyramid-layout.js";
export {
  vdbFrameGridCascades,
  vdbFrameAdaptiveCellSize,
  vdbFrameAdaptiveGridResolution,
  VDB_FRAME_ADAPTIVE_OPTICAL_DEPTH,
  VDB_FRAME_ADAPTIVE_STEP,
  VDB_FRAME_ADAPTIVE_VOXELS,
  vdbFrameGridRange,
  vdbFrameCascadeCasters,
  vdbFrameGridSignatureState,
  vdbFrameUpdateGridSignatures,
  vdbSameValues,
  vdbSignaturePushVector,
  vdbSignaturePushElements,
  vdbFrameGridUvw,
  vdbFrameInsideGrid,
  vdbFrameFindCascade,
  vdbFrameGridCell,
  vdbFrameAtlasBilinear,
  vdbFrameSampleTiles,
  vdbFrameSampleGrid,
  vdbFrameGridExtinction,
  vdbFrameGridMarch,
  vdbFrameIsotropicDiffusion,
  vdbFrameTransmittanceVoxel,
  vdbFrameReproject,
  vdbFrameLightingFroxel,
  vdbFrameEnvironmentGridVoxel,
  vdbFrameSampleShadows,
  vdbFrameDensityFroxel,
  vdbFrameDensityFroxelSteps,
} from "./frame/grid.js";
export type {
  VDBVec3Like,
  VDBBox3Like,
  VDBFrameGridSignatureState,
  VDBFrameGridLayout,
  VDBFrameAtlas,
  VDBFrameShadowLight,
  VDBFrameLightingInputs,
  VDBFrameDensityInputs,
  VDBFrameTransmittanceModes,
  VDBFrameAdaptiveCaster,
  VDBFrameAdaptiveGridRule,
} from "./frame/grid.js";
export {
  vdbFrameStackLayout,
  vdbFrameTrilinear,
  vdbFrameStackSample,
  vdbFrameGridFragCoord,
  vdbFrameGridVoxelWorld,
  vdbFrameGridDensityVoxel,
  vdbFrameTransmittanceVoxel3d,
  vdbFrameStackShadows,
  vdbFrameEnvironmentGridVoxel3d,
} from "./frame/stack.js";
export {
  VDB_FRAME_ENV_WORK_WIDTH,
  VDB_FRAME_ENV_WORK_HEIGHT,
  VDB_FRAME_ENV_ROUGHNESS,
  VDB_FRAME_ENV_MAX_SPAN,
  VDB_FRAME_ENV_MAX_SUPERSAMPLE,
  VDB_FRAME_ENV_TIE,
  VDB_FRAME_ENV_IDENTITY,
  VDB_FRAME_ENV_MODE,
  vdbFrameEnvToMap,
  vdbFrameEnvToWorld,
  vdbFrameEquirectDirection,
  vdbFrameEnvWorkSolidAngle,
  vdbFrameEnvironmentAreaTexel,
  vdbFrameCubeTexel,
  vdbFrameEnvCubeSupersample,
  vdbFrameEnvironmentCubeTexel,
  vdbFrameEnvGgxWeight,
  vdbFrameEnvironmentBlurTexel,
  vdbFrameSh9Basis,
  vdbFrameEnvironmentIrradianceSh,
  vdbFrameEnvironmentIrradiance,
  vdbFrameEnvironmentReduce,
  vdbFrameEnvironmentDensityTerm,
} from "./frame/environment-gpu.js";
export type { VDBFrameEnvRotation, VDBFrameEnvImage, VDBFrameEnvironmentGpuResult, VDBFrameEnvironmentReduceSettings, VDBFrameEnvironmentFlags } from "./frame/environment-gpu.js";
export type { VDBFrameStackLayout, VDBFrameTexture3d, VDBFrameGridCaster } from "./frame/stack.js";
// K6.Q1 shadowed ambient (twins of the core's ambient occlusion)
export {
  VDB_FRAME_AMBIENT_MAX_PAIRS,
  vdbFrameAmbientDipoleMatrix,
  vdbFrameAmbientDirections,
  vdbFrameAmbientPhaseFactor,
  vdbFramePhaseLobes,
  vdbFrameAmbientIrradiance,
  vdbFrameAmbientPairTotals,
  vdbFrameAmbientVisibility,
  vdbFrameSpectralAmbientPairTotals,
  vdbFrameSpectralAmbientVisibility,
  type VDBFrameDistantLight,
  type VDBFramePhaseLobes,
} from "./frame/ambient.js";
// K6.5b: mesh → volume shadows (host occluder maps)
export { vdbFrameOccluderVisibility, vdbFrameOccluderLayer, vdbFrameOccludedTotal, VDB_FRAME_MAX_OCCLUDERS, VDB_FRAME_OCCLUDER_MAX_LAYERS } from "./frame/occluders.js";
export type { VDBFrameOccluderCpu } from "./frame/occluders.js";
export {
  VDBFrameEnvironmentLights,
  vdbFrameEnvironmentMapDirection,
  vdbFrameEnvironmentMode,
  frameEnvironmentModes as vdbFrameEnvironmentModes,
  frameMaxEnvironmentLights as vdbFrameMaxEnvironmentLights,
  frameEnvironmentMapWidth as vdbFrameEnvironmentMapWidth,
  frameEnvironmentMapHeight as vdbFrameEnvironmentMapHeight,
} from "./frame/environment.js";
export type {
  VDBFrameEnvironmentMode,
  VDBFrameEnvironmentSettings,
  VDBFrameEnvironmentScene,
  VDBFrameEnvironmentLight,
  VDBFrameEnvironmentResult,
} from "./frame/environment.js";
export {
  vdbFrameMieTable,
  vdbFrameMieTexels,
  vdbFrameMieLutWords,
  vdbFrameMieTexelsHalf,
  vdbFrameMieTexel,
  vdbFrameMiePhase,
  vdbFrameMieRgbPhase,
  vdbFrameMieFormat,
  vdbFrameMieTableSize,
  vdbFrameMieAsymmetry,
  vdbFrameMieRgbMode,
  vdbFrameMieChannel,
  vdbFrameMieRgbChannels,
} from "./frame/mie.js";
export type { VDBFrameMieTableOptions, VDBFrameMieTableResult } from "./frame/mie.js";
export {
  vdbFrameNeuralLayout,
  vdbFrameNeuralCheck,
  vdbFrameNeuralForward,
  vdbFrameNeuralSampleLevel,
  vdbFrameNeuralUnit,
  vdbFrameNeuralInputs,
  vdbFrameNeuralLight,
  vdbFrameTileAtlasLayout,
  vdbFrameGridMipLevel,
  vdbFrameNeuralMipResolution,
} from "./frame/neural.js";
export type { VDBFrameNeuralNetwork, VDBFrameNeuralMip } from "./frame/neural.js";
export { vdbFrameNeuralWeights } from "./frame/neural-weights.js";
// K6.3: the experimental generic MLP of the core's 'mlp' multiple scattering (twins)
export {
  VDB_FRAME_MLP_MAX_LAYERS,
  VDB_FRAME_MLP_MAX_LEVELS,
  VDB_FRAME_MLP_MAX_WIDTH,
  vdbFrameMlpCheck,
  vdbFrameMlpForward,
  vdbFrameMlpFromNeural,
  vdbFrameMlpInputs,
  vdbFrameMlpMipSizes,
  vdbFrameMlpMipTexel,
  vdbFrameMlpOffsets,
  vdbFrameMlpSampleLevel,
  vdbFrameMlpShape,
  // K6.Na: the packed layout (pmndrs/denoiser's layer packing, one tap) of the core's `mlpRuntime: 'packed'`
  vdbFrameMlpPack,
  vdbFrameMlpPackedLayerVectors,
  vdbFrameMlpPackedOffsets,
  vdbFrameMlpPackedShape,
  vdbFrameMlpForwardPacked,
} from "./frame/mlp.js";
export type { VDBFrameMlp, VDBFrameMlpLayer } from "./frame/mlp.js";
// K6.Na: the network in the pmndrs neural tooling's weight container (OIDN `.tza` v2, pmndrs/denoiser `parseTZA`)
export { vdbTzaRead, vdbTzaWrite, vdbFrameMlpToTza, vdbFrameMlpFromTza } from "./frame/mlp-tza.js";
export type { VDBTzaTensor } from "./frame/mlp-tza.js";
// K6.Nc: the network in the pmndrs NRC's bias-free weight convention (pmndrs/pathtracer `@pathtracer/mlp`)
export { vdbFrameMlpToNn, vdbFrameMlpFromNn, vdbFrameMlpNnInputs, vdbFrameMlpNnWeightCount } from "./frame/mlp-nn.js";
export type { VDBFrameMlpNn } from "./frame/mlp-nn.js";
export {
  VDB_FRAME_MULTI_SCALE_MAX_LEVELS,
  VDB_FRAME_MULTI_SCALE_DEFAULT_LEVELS,
  vdbFrameMultiScaleLayout,
  vdbFrameMultiScaleMips,
  vdbFrameMultiScaleToHalf,
  vdbFrameTexture3DLinear,
  vdbFrameMultiScaleSample,
  vdbFrameMultiScaleLight,
  vdbFrameMultiScaleLevelOffset,
} from "./frame/multi-scale.js";
export type {
  VDBFrameMultiScaleEdge,
  VDBFrameMultiScaleSigma,
  VDBFrameMultiScaleMipLevel,
  VDBFrameMultiScaleLayout,
  VDBFrameTexture3D,
  VDBFrameMultiScaleMipOptions,
  VDBFrameMultiScaleMips,
  VDBFrameMultiScaleGrid,
  VDBFrameMultiScaleTextures,
} from "./frame/multi-scale.js";
export { vdbFrameMediumMeasurements, vdbFrameFitPhaseGExponent } from "./frame/medium.js";
export {
  vdbFrameSpectrumWavelengths,
  vdbFrameSpectralGroupCount,
  vdbFrameSpectralGroups,
  vdbFrameExtinctionSpectrum,
  vdbFrameMeanComponent,
  vdbFrameSpectralShadowUnit,
  vdbFrameSpectralShadowDepths,
  vdbFrameSpectralShadowStep,
  vdbFrameSpectralGridAlbedo,
  vdbFrameSpectralShadowDepth,
  vdbFrameSpectralShadowMarchDepth,
  vdbFrameSpectralShadowTransmittance,
  vdbFrameSpectralOctaveLight,
  vdbFrameSpectralShadowTotal,
  vdbFrameSpectralShadowedLight,
  vdbFrameUnpackSpectralGroup,
  vdbFramePackSpectralGroup,
  vdbFrameSpectralGridSample,
  vdbFrameSpectralGridMarch,
  vdbFrameSpectralTransmittanceVoxel,
  vdbFrameSpectralEnvironmentGridVoxel,
  vdbFrameSpectralDensityFroxel,
  vdbFrameSpectralLightingFroxel,
  vdbFrameSpectralGroupsAt,
  vdbShIrradianceAt,
  vdbFrameInjectIsotropicIrradiance,
  vdbFrameSpectralShadowMarch,
  vdbFrameSpectralInjectSlice,
  vdbFrameSpectralIntegrate,
} from "./frame/spectral.js";
export type {
  VDBFrameSpectralDepths,
  VDBFrameSpectralShadow,
  VDBFrameSpectralMarch,
  VDBFrameSpectralDensityInputs,
  VDBFrameSpectralLightingInputs,
  VDBFrameInjectMarch,
  VDBFrameInjectLightInputs,
} from "./frame/spectral.js";
export {
  frameShadowMinOpticalDepth as vdbFrameShadowMinOpticalDepth,
  frameShadowMaxOpticalDepth as vdbFrameShadowMaxOpticalDepth,
  frameMaxCastShadowSteps as vdbFrameMaxCastShadowSteps,
  frameMaxShadowMapCascades as vdbFrameMaxShadowMapCascades,
  vdbFrameShadowDepth as vdbFrameShadowCameraDepth,
  vdbFrameShadowPassRay,
  vdbFrameShadowMapTexel,
  vdbFrameShadowReceiver,
  vdbFrameShadowCascadeBreaks,
  vdbFrameShadowCascadeFar,
  vdbFrameShadowCascadeIndex,
  vdbFrameShadowCascadeUv,
  vdbFrameShadowSpotUv,
  vdbFrameShadowMapRay,
  vdbFrameCubeFaceDirection,
  vdbFrameCubeFaceUv,
  vdbFrameShadowViewCamera,
  vdbFrameShadowFitCascades,
  vdbFrameVolumeShadowLookup,
  vdbFrameVolumeShadowLookupRgb,
} from "./frame/shadows.js";
export type { VDBFrameShadowRay, VDBFrameShadowFitInputs, VDBFrameShadowCascadeCamera, VDBFrameVolumeShadowViewCpu } from "./frame/shadows.js";
export {
  frameEmissionDirections as vdbFrameEmissionDirections,
  frameEmissionRays as vdbFrameEmissionRays,
  frameEmissionMaxResolution as vdbFrameEmissionMaxResolution,
  frameEmissionMaxCascades as vdbFrameEmissionMaxCascades,
  frameEmissionAtlasSize as vdbFrameEmissionAtlasSize,
  vdbFrameEmissionDirection,
  vdbFrameBoxSegment,
  vdbFrameEmitterBounds,
  vdbFrameEmissionDirectionTexel,
  vdbFrameEmissionReduce,
  vdbFrameEmissionProbe,
  vdbFrameEmissionIrradiance,
  vdbFrameEmissionProbeGrid,
  vdbFrameEmissionFalloff,
  vdbFrameEmissionLookup,
  vdbFrameEmissionLookupWith,
} from "./frame/emission.js";
export type {
  VDBFrameEmitter,
  VDBFrameEmitterLight,
  VDBFrameEmissionGridSettings,
  VDBFrameEmissionGrid,
} from "./frame/emission.js";
export {
  vdbFrameBlackbodyRgb,
  vdbFrameBlackbodyTable,
  vdbFrameEmissionSpectrum,
  vdbFrameEmissionLutRow,
  vdbFrameEmissionLutColor,
  frameBlackbodyWavelengthStart as vdbFrameBlackbodyWavelengthStart,
  frameBlackbodyWavelengthStep as vdbFrameBlackbodyWavelengthStep,
  frameBlackbodyWavelengthCount as vdbFrameBlackbodyWavelengthCount,
  frameBlackbodyTableSize as vdbFrameBlackbodyTableSize,
  frameBlackbodyTableRows as vdbFrameBlackbodyTableRows,
} from "./frame/blackbody.js";
// K6.1: the core frame renderer's froxel view maths and froxel-stage twins.
export {
  vdbMat4Multiply,
  vdbMat4Invert,
  vdbMat4Point,
  vdbMat4Vec4,
  vdbMat4MaxScale,
  vdbFrustumPlanes,
  vdbUnitBoxBounds,
  vdbFrameClipCamera,
  vdbFrameClipVolume,
  vdbFrameFroxelRect,
  vdbFrameFroxelArea,
  vdbFroxelAreaEmpty,
  vdbFrameVolumeSlices,
  vdbFroxelAreaUnion,
  vdbFrameAtlasLayout,
  vdbFroxelHeldSize,
  vdbFrameFroxelSize,
  vdbFrameClipRect,
} from "./frame/view.js";
export type { VDBMat4, VDBFrameClipCamera, VDBFrameClippedVolume, VDBFroxelRect, VDBFroxelArea, VDBFroxelHeldSize } from "./frame/view.js";
export {
  vdbFrameGatherFroxel,
  vdbFrameMaxOpticalDepth,
  vdbFrameTemporalFroxel,
  vdbFrameScanStop,
  vdbFrameScanColumn,
  vdbFrameIntegratePrefix,
  vdbFrameSliceCoord,
} from "./frame/froxel.js";
export type { VDBFrameVolumeFroxel, VDBFrameIntegrated } from "./frame/froxel.js";
// K6.4: CPU twins of the core's spectral kernels (internal only)
export {
  vdbFrameSpectralGridDensityVoxel,
  vdbFrameSpectralTransmittanceVoxel3d,
  vdbFrameSpectralEnvironmentGridVoxel3d,
  vdbFrameSpectralStackGroups,
  vdbFrameSpectralDensitySums,
  vdbFrameSpectralEnvironmentDensityTerm,
  vdbFrameGatherFroxelSpectral,
  vdbFrameScanColumnSpectral,
  vdbFrameIntegratePrefixSpectral,
  vdbFrameCompositeSpectral,
} from "./frame/spectral-stack.js";
export type { VDBFrameSpectralGridCaster, VDBFrameSpectralGridAt, VDBFrameSpectralSums, VDBFrameSpectralDensityInputs3d, VDBFrameSpectralVolumeFroxel } from "./frame/spectral-stack.js";
// K6.R: the reference renderer's scene, random stream, CPU twin and analytic cases (internal only).
export * from "./reference/index.js";
// D36: helpers that are not public API (they were exported before 0.2): the byte length of a `.wvdb` LOD prefix (range
// requests use `wvdbLodByteRange`), the `Content-Length` of a `.nvdb` response, and the FpN leaf codec factory.
export { nvdbResponseLength } from "./nvdb-stream.js";
export { vdbMakeFpNLeafCodec } from "./nanovdb/index.js";
export { wvdbPrefixLength } from "./wvdb/read.js";
// K7.C1: the LZ4 Blosc writer under `writeNvdb({ codec: 'blosc' })` (frames byte-equal to NanoVDB v14's).
export { NVDB_BLOSC_CHUNK_BYTES, nvdbBloscPayload } from "./nanovdb/write-nvdb.js";
export { compressBlosc, shuffleBytes } from "./vdb/blosc-write.js";
export type { BloscCompressOptions } from "./vdb/blosc-write.js";
export { compressLz4Block, lz4CompressBound } from "./vdb/lz4.js";
// K7.H1: the stats accumulation A/B switch (GridStats' Welford is the default; 'sums' is the K1-K7.H path).
export { vdbSetStatsAccumulation } from "./nanovdb/stats.js";
export type { VDBStatsAccumulation } from "./nanovdb/stats.js";
