/**
 * The froxel renderer in `./three` (K6.8): a thin adapter over the core frame renderer (design §6). `VDBFrameVolume`
 * (his FrameVolume) holds the props; `VDBFrameRenderer` runs the core on three's device and composites; the volume
 * shadow receivers and the emission light read the core's maps and probes.
 */
export { VDBFrameVolume } from "./volume.js";
export type { VDBFrameVolumeProps, VDBFramePhaseSettings } from "./volume.js";
export { VDBFrameRenderer, vdbFrameRenderer } from "./renderer.js";
export type { VDBFramePassHook } from "./renderer.js";
export { getFrameMediumProps } from "./medium.js";
export type { VDBFrameMediumVolumeProps } from "./medium.js";
export { frameAngstromExponents, getFrameAngstromExponent } from "./spectrum.js";
export { VDBFrameVolumeShadowReceivers, VDBFrameVolumeShadowNode } from "./volume-shadows.js";
export type { VDBFrameShadowEntry } from "./volume-shadows.js";
export { VDBFrameEmissionLight, VDBFrameEmissionLightNode } from "./emission-light.js";
export type { VDBFrameEmissionLightProps } from "./emission-light.js";
export { VDBFrameCoreIntegrateMaterial } from "./core-integrate-material.js";
export { VDBCoreTextureBinding, vdbCoreTextureSelfTest, vdbCoreTextureStandIn, VDB_THREE_CORE_TEXTURE_SELF_TEST_LABEL } from "./core-textures.js";
export { vdbFrameCheckLightReset, VDB_FRAME_THREE_REVISION } from "./three-internals.js";
