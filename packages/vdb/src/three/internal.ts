/**
 * @pmndrs/vdb/three internals (D32): NOT a package export.
 *
 * Everything public (`./index.ts`) plus what tests and harnesses need that a user doesn't: the frame renderer's
 * composite material, core-texture stand-ins, volume shadow receivers and the three-internals hooks. Import it by
 * path (`src/three/internal.ts`); its names can change without notice.
 */

export * from "./index.js";

export { vdbAttenuatedPointSize, vdbPointDiscTexture } from "./points.js";
export {
  VDBFrameCoreIntegrateMaterial,
  VDBFrameVolumeShadowReceivers,
  VDBFrameVolumeShadowNode,
  VDBFrameEmissionLightNode,
  VDBCoreTextureBinding,
  vdbCoreTextureSelfTest,
  vdbCoreTextureStandIn,
  VDB_THREE_CORE_TEXTURE_SELF_TEST_LABEL,
  vdbFrameCheckLightReset,
  VDB_FRAME_THREE_REVISION,
  getFrameAngstromExponent as vdbFrameAngstromExponent,
  frameAngstromExponents as vdbFrameAngstromExponents,
} from "./frame/index.js";
export type { VDBFrameShadowEntry, VDBFramePhaseSettings } from "./frame/index.js";
