// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameSpectrum.js (the Ångström presets and getFrameAngstromExponent).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; K6.8: only the preset lookup is left (the core packs the extinction spectrum).

const toNumber = (value: unknown, fallback: number) => (Number.isFinite(Number(value)) ? Number(value) : fallback);

/** Measured Ångström exponents (his VDB-116 research; Dubovik 2002, Bodhaine 1999). */
export const frameAngstromExponents: Readonly<Record<string, number>> = {
  cloud: 0.0,
  dust: 0.3,
  smoke: 1.8,
  air: 4.09,
};

/** A preset name of {@link frameAngstromExponents} or a number. Unknown names give 0 (grey). */
export const getFrameAngstromExponent = (value: string | number | undefined) =>
  typeof value === "string" ? (frameAngstromExponents[value] ?? 0.0) : toNumber(value, 0.0);
