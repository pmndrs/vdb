/**
 * K6.8 Stage B only: the 0.1 public names the TSL deletion would remove, kept so the api-surface snapshot moves in one
 * reviewed commit (Stage C deletes this file).
 */
import { VDB_FRAME_ALL_LIGHTS, type VDBEmissionColor } from "../../core/frame/types.js";
import type { VDBFrameMultipleScatteringMode, VDBFramePhaseName, VDBFrameSettings } from "../../core/frame/settings.js";

export type { VDBFrameRedrawStats } from "../../core/frame/types.js";
export type VDBFrameRendererSettings = VDBFrameSettings;
export type VDBFramePass = string;
export type VDBFramePhase = VDBFramePhaseName;
export type VDBFrameMultipleScattering = VDBFrameMultipleScatteringMode;
export type VDBFrameBlackbodyHook = VDBEmissionColor;

export const vdbFrameLights = {
  useAmbientLights: 0b100000,
  usePointLights: 0b010000,
  useDirectionalLights: 0b001000,
  useSpotLights: 0b000100,
  useHemisphereLights: 0b000010,
  useEnvironment: 0b000001,
} as const;
export const vdbFrameAllLights = VDB_FRAME_ALL_LIGHTS;
export const vdbFrameBlackbodyEmission = (spectrum: "planck" | "soot" = "planck", exponent = 1.39): VDBEmissionColor => ({ spectrum, exponent });
