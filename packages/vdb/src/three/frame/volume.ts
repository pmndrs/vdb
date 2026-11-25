// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameVolume.js L8–11 (render order, all lights), L152–270 (props,
//   proxy), L339–353 (box matrix), L554–568 (getDensityScale), L624–650 (dispose).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; takes a VDBVolumeSource (dense `.wvdb`, or the sparse pool since K3.2) instead of resampling grids
// on the CPU (his L279–552: conversion, progressive upload, level sets, points, instances); one
// volume per frame volume; the mean density is the source's; K6.8: the props are the core's `VDBFrameVolumeParams`
// (the renderer copies them each update), the emission colour is core data (`VDBEmissionColor`). Divergences:
// DIVERGES.md (K3.1 rows), V23–V25.

/**
 * A volume in the froxel renderer (his `FrameVolume`, R1 §2): a `Group` whose proxy mesh draws the
 * frame. Its props are his (same names and defaults) for the ported subset; the renderer copies them into the core's
 * `VDBFrameVolumeParams` at each render.
 *
 *   const source = vdbDenseTextures(wvdbDenseVolume(readWvdb(bytes)));
 *   const volume = new VDBFrameVolume(source, { extinction: 100 });  // box: the source's (volume.bbox)
 *   scene.add(volume);                       // then renderer.render(scene, camera) as usual
 *   volume.frameRenderer.settings.multipleScattering = false;
 *   volume.setMedium("fog");                 // a measured medium: albedo, spectrum, the renderer's phase
 *
 * The phase (K5.9): the renderer's `settings.phase*` is the phase of every volume it draws. `'frame'`
 * lighting lights all volumes in one pass with it; in `'volume'` lighting (each volume marches its own
 * light) a volume's own `phase*` props, when set, override it for that volume. The source (dense or
 * sparse), its kind and its LOD are fixed at construction: make a new source and a new volume to change
 * them (`source.replace` swaps a dense frame of the same channels; the volume follows it by itself). A sparse
 * source made with `followImage` follows a `pool.replace` of its grid (a coarse-first swap), box included (G16a).
 */

import * as THREE from "three";
import { VDBError } from "../../core/errors.js";
import { VDB_FRAME_ALL_LIGHTS, type VDBEmissionColor } from "../../core/frame/types.js";
import type { VDBFramePhaseName } from "../../core/frame/settings.js";
import type { VDBThreeVolumeSource } from "../volume-source.js";
import { getFrameMediumProps, type VDBFrameMediumVolumeProps } from "./medium.js";
import { vdbFrameRenderer, type VDBFrameRenderer } from "./renderer.js";

// NOTE Draw the frame after all other transparent objects
const frameRenderOrder = 1000000;

/** Props of {@link VDBFrameVolume}: his `FrameVolume` props (names and defaults) for the ported subset. */
export interface VDBFrameVolumeProps {
  /**
   * The froxel renderer (default: the shared {@link vdbFrameRenderer}). Its `settings` apply to every
   * volume that uses it; pass your own `new VDBFrameRenderer()` (or `vdbFrameRenderer(threeRenderer)`)
   * for separate settings.
   */
  frameRenderer?: VDBFrameRenderer;
  /**
   * Box of the volume in its local space (default: the source's box: the `.wvdb` bbox or the sparse grid's world
   * box; else the unit box at the origin). Read back as {@link VDBFrameVolume.bbox}.
   */
  bbox?: THREE.Box3 | [[number, number, number], [number, number, number]];
  /** Albedo (single-scattering colour) of the medium (default white). */
  baseColor?: THREE.ColorRepresentation;
  /** Colour added to the volume's own emission, so it shows without light (default black). */
  scatterColor?: THREE.ColorRepresentation;
  /** With normalizeDensity: optical depth of the longest box side at the mean density (default 100). */
  extinction?: number;
  /** Extinction scale of each channel (renderer `settings.spectral`; with `angstromExponent`). */
  extinctionScale?: [number, number, number] | THREE.Vector3Like;
  /** Ångström exponent of the extinction spectrum: a number or a preset name (`'cloud'` 0, `'dust'`, `'smoke'`, `'air'`; default 'cloud': grey). */
  angstromExponent?: string | number;
  /**
   * `'volume'` lighting only: this volume's phase, overriding the renderer's `settings.phase` (default: unset, the
   * renderer's). `'frame'` lighting always uses the renderer's (one lighting pass for all volumes). The same for
   * `phaseG`, `phaseBackG`, `phaseBackWeight` and `phaseGExponent`, field by field.
   */
  phase?: VDBFramePhaseName | null;
  /** Forward `g` of this volume's phase in `'volume'` lighting (null: the renderer's). */
  phaseG?: number | null;
  /** Backward-lobe `g` in `'volume'` lighting (null: the renderer's). */
  phaseBackG?: number | null;
  /** Backward-lobe weight in `'volume'` lighting (null: the renderer's). */
  phaseBackWeight?: number | null;
  /** Wavelength exponent of `g` in spectral rendering (null: the renderer's). */
  phaseGExponent?: number | null;
  /** Multiplier of the density (default 1; divided by the mean density with `normalizeDensity`). */
  densityScale?: number;
  /** Divide `densityScale` by the mean density so `extinction` sets the optical depth (default true). */
  normalizeDensity?: boolean;
  /**
   * The mean density `normalizeDensity` divides by (default: unset, the source's `stats.meanNonZero`, which follows
   * `source.replace`). Set one value for a sequence so every frame has the same extinction scale: his sequences divide
   * by the mean of all frames' non-zero texels (`FrameVolume.getDensityScale`).
   */
  densityMean?: number | null;
  /** Densities at or below this read as 0 (default 0). */
  densityCutoff?: number;
  /** Scales the optical depth of the shadow march (0 turns the volume's own shadow off; default 1). */
  absorbance?: number;
  /** Opacity of the composited volume, 0 to 1 (default 1). */
  opacity?: number;
  /** Multiplier of the emission (default 1). */
  emissiveIntensity?: number;
  /** Kelvin at emission values 0 and 1. */
  temperatureRange?: [number, number];
  /**
   * Offset of the texture lookup in texture coordinates (default 0); a sparse source with a nonzero offset turns HDDA
   * skipping off.
   */
  offset3D?: THREE.Vector3Like;
  /** Texture wrapping of the lookup (default clamp to edge); wrapping turns a sparse source's HDDA skipping off. */
  wrap3D?: THREE.Wrapping;
  /** Light types bitmask (`VDB_FRAME_LIGHT_BITS`); default all (`VDB_FRAME_ALL_LIGHTS`). */
  lights?: number;
  /** Steps of the shadow march toward each light in `'volume'` lighting (default 32). */
  shadowSteps?: number;
  /** Sample the source's mip levels (default true; needs a source with a mip atlas). */
  mipmaps?: boolean;
  /**
   * The emission colour (K6.5, core data): `'white'` (default), `{ spectrum: 'planck' | 'soot', exponent }` (his
   * blackbody spectra), `{ lut }` (a 256-texel rgb row over log temperature 500–40,000 K) or `{ wgsl }` (a validated
   * WGSL hook, `fn vdbEmissionColor(temperature: f32) -> vec3f`).
   */
  emissionColor?: VDBEmissionColor;
  /**
   * HDDA empty-space skipping in the shadow march (sparse sources; K3.2). Skipped samples are exactly
   * 0, so the image does not change (within float noise); dense sources ignore it. Default false (D30: on the Apple GPU it
   * slowed the shadow march 1.07–3.31× in all 12 paired runs, +7 to +26 ms per frame; owner rule D29 was "on
   * unless the cost is large"). Timings in docs/archive/handoffs/K3-2a.md.
   */
  hdda?: boolean;
}

const toBox = (bbox: VDBFrameVolumeProps["bbox"], source: VDBThreeVolumeSource): THREE.Box3 => {
  if (bbox instanceof THREE.Box3) return bbox.clone();
  const b = bbox ?? (source.kind === "dense" ? source.volume.bbox : source.bbox);
  if (!b) return new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5));
  return new THREE.Box3(new THREE.Vector3(...b[0]), new THREE.Vector3(...b[1]));
};

/** FrameVolume.js L339–353: the volume box as a matrix of the unit box (into `target`). */
const boxMatrix = (box: THREE.Box3, target: THREE.Matrix4): THREE.Matrix4 => {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).max(new THREE.Vector3(1e-6, 1e-6, 1e-6));
  return target.compose(center, new THREE.Quaternion(), size);
};

/** The phase a volume draws with in `'volume'` lighting (its own fields where set, else the renderer's). */
export interface VDBFramePhaseSettings {
  phase: VDBFramePhaseName;
  phaseG: number;
  phaseBackG: number;
  phaseBackWeight: number;
  phaseGExponent: number;
}

/**
 * A volume drawn by the froxel renderer: a `Group` whose proxy mesh composites the volume's density and light over
 * the scene (his `FrameVolume`). Add it to a scene and render as usual; its props are live fields.
 */
export class VDBFrameVolume extends THREE.Group {
  /** Type flag. */
  readonly isVDBFrameVolume = true as const;
  /** The volume source the density (and emission) is read from. */
  readonly source: VDBThreeVolumeSource;
  /** The froxel renderer that draws this volume. */
  readonly frameRenderer: VDBFrameRenderer;
  /** The proxy mesh whose draw triggers the frame render and composites the volume. */
  readonly proxy: THREE.Mesh<THREE.InstancedBufferGeometry>;
  /** The unit box → the volume box in local space (his `boxMatrix`); follows {@link VDBFrameVolume.bbox}. */
  readonly boxMatrix: THREE.Matrix4;
  /** `matrixWorld × boxMatrix` of the last render (the core's `params.matrixWorld`). */
  readonly boxMatrixWorld = new THREE.Matrix4();
  /** The emission colour; see {@link VDBFrameVolumeProps.emissionColor}. */
  emissionColor: VDBEmissionColor | undefined;

  /** Albedo of the medium; see {@link VDBFrameVolumeProps.baseColor}. */
  baseColor: THREE.Color;
  /** Colour added to the emission; see {@link VDBFrameVolumeProps.scatterColor}. */
  scatterColor: THREE.Color;
  /** Optical depth of the longest box side at the mean density (with `normalizeDensity`). */
  extinction: number;
  /** Extinction scale of each channel (spectral rendering). */
  extinctionScale: THREE.Vector3;
  /** Ångström exponent of the extinction spectrum: a number or a preset name. */
  angstromExponent: string | number;
  /** `'volume'` lighting: phase overrides (null: the renderer's `settings`); see {@link VDBFrameVolumeProps.phase}. */
  phase: VDBFramePhaseName | null;
  /** Forward `g` override of `'volume'` lighting (null: the renderer's). */
  phaseG: number | null;
  /** Backward-lobe `g` override (null: the renderer's). */
  phaseBackG: number | null;
  /** Backward-lobe weight override (null: the renderer's). */
  phaseBackWeight: number | null;
  /** Wavelength exponent of `g` override (null: the renderer's). */
  phaseGExponent: number | null;
  /** Multiplier of the density; see {@link VDBFrameVolumeProps.densityScale}. */
  densityScale: number;
  /** Whether the density scale is divided by the mean density. */
  normalizeDensity: boolean;
  /** See {@link VDBFrameVolumeProps.densityMean}; null: the source's mean. */
  densityMean: number | null;
  /** Densities at or below this read as 0. */
  densityCutoff: number;
  /** Scales the optical depth of the shadow march. */
  absorbance: number;
  /** Opacity of the composited volume, 0 to 1. */
  opacity: number;
  /** Multiplier of the emission. */
  emissiveIntensity: number;
  /** Kelvin at emission values 0 (x) and 1 (y). */
  temperatureRange: THREE.Vector2;
  /** Offset of the texture lookup in texture coordinates. */
  offset3D: THREE.Vector3;
  /** Texture wrapping of the lookup. */
  wrap3D: THREE.Wrapping;
  /** Light types bitmask (`VDB_FRAME_LIGHT_BITS`) this volume receives. */
  lights: number;
  /** Steps of the shadow march in `'volume'` lighting. */
  shadowSteps: number;
  /** Whether the source's mip levels are sampled. */
  mipmaps: boolean;
  /** Whether the sparse source's shadow march skips empty space (no effect on dense sources). */
  hdda: boolean;
  /**
   * The volume's box in its local space: the `bbox` prop, else the source's box, re-read
   * when the source's `version` moves: a dense frame with another `.wvdb` box (a sequence frame; RV, as his WVDB sequences)
   * or a sparse source made with `followImage` after its grid's image changed (G16a, e.g. a coarse-first
   * swap), so this object is updated in place then. Do not mutate.
   */
  readonly bbox: THREE.Box3;

  /** The medium fields at construction, for `setMedium(null)`. */
  private readonly initialMedium: { baseColor: THREE.Color; extinctionScale: THREE.Vector3; angstromExponent: string | number };
  /** `source.version` the box was last read at. */
  private sourceVersion: number;
  /** G16a / RV: the box is the source's (no `bbox` prop), so it follows the source. */
  private readonly boxFromSource: boolean;
  private readonly resolvedPhase: VDBFramePhaseSettings = { phase: "henyey-greenstein", phaseG: 0.8, phaseBackG: -0.3, phaseBackWeight: 0.2, phaseGExponent: 0.0 };

  /**
   * Makes a volume over a dense or sparse source. Throws `VDB_BAD_OPTION` for a source that is neither.
   * @param source A source from `vdbDenseTextures()` or `vdbSparseSource()`.
   * @param props {@link VDBFrameVolumeProps}; all optional.
   */
  constructor(source: VDBThreeVolumeSource, props: VDBFrameVolumeProps = {}) {
    super();

    // K6.8a: `volume`, not `textures`: a dense source's three textures are made only for custom materials
    const ok = source && ((source.kind === "dense" && !!source.volume?.density) || (source.kind === "sparse" && typeof source.createNodes === "function"));
    if (!ok) {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameVolume: pass a volume source from vdbDenseTextures() or vdbSparseSource() (@pmndrs/vdb/three).");
    }
    const color = props.emissionColor as unknown;
    if (typeof color === "function") {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameVolume: emissionColor is core data since 0.2.0 ('white', { spectrum: 'planck' | 'soot', exponent }, { lut } or { wgsl }), not a TSL hook; vdbFrameBlackbodyEmission(s, e) is { spectrum: s, exponent: e }.");
    }

    const {
      frameRenderer = vdbFrameRenderer(),
      bbox,
      baseColor = 0xffffff,
      scatterColor = 0x000000,
      extinction = 100.0,
      extinctionScale = [1.0, 1.0, 1.0],
      angstromExponent = "cloud",
      phase = null,
      phaseG = null,
      phaseBackG = null,
      phaseBackWeight = null,
      phaseGExponent = null,
      densityScale = 1.0,
      normalizeDensity = true,
      densityMean = null,
      densityCutoff = 0.0,
      absorbance = 1.0,
      opacity = 1.0,
      emissiveIntensity = 1.0,
      temperatureRange = [0.0, 6500.0],
      offset3D = new THREE.Vector3(),
      wrap3D = THREE.ClampToEdgeWrapping,
      lights = VDB_FRAME_ALL_LIGHTS,
      shadowSteps = 32,
      mipmaps = true,
      emissionColor,
      hdda = false,
    } = props;

    this.source = source;
    this.frameRenderer = frameRenderer;
    this.emissionColor = emissionColor;
    this.baseColor = new THREE.Color(baseColor);
    this.scatterColor = new THREE.Color(scatterColor);
    this.extinction = extinction;
    this.extinctionScale = Array.isArray(extinctionScale) ? new THREE.Vector3().fromArray(extinctionScale) : new THREE.Vector3().copy(extinctionScale);
    this.angstromExponent = angstromExponent;
    this.phase = phase;
    this.phaseG = phaseG;
    this.phaseBackG = phaseBackG;
    this.phaseBackWeight = phaseBackWeight;
    this.phaseGExponent = phaseGExponent;
    this.densityScale = densityScale;
    this.normalizeDensity = normalizeDensity;
    this.densityMean = densityMean;
    this.densityCutoff = densityCutoff;
    this.absorbance = absorbance;
    this.opacity = opacity;
    this.emissiveIntensity = emissiveIntensity;
    this.temperatureRange = new THREE.Vector2().fromArray(temperatureRange);
    this.offset3D = new THREE.Vector3().copy(offset3D);
    this.wrap3D = wrap3D;
    this.lights = lights;
    this.shadowSteps = shadowSteps;
    this.mipmaps = mipmaps;
    this.hdda = hdda;
    this.initialMedium = { baseColor: this.baseColor.clone(), extinctionScale: this.extinctionScale.clone(), angstromExponent };
    this.sourceVersion = source.version;

    this.proxy = new THREE.Mesh(frameRenderer.createProxyGeometry(), frameRenderer.proxyMaterial());
    this.proxy.frustumCulled = false;
    this.proxy.renderOrder = frameRenderOrder;
    this.proxy.onBeforeRender = (renderer, scene, camera) => {
      frameRenderer.render(renderer as never, scene, camera, this.proxy);
    };
    this.add(this.proxy);

    // FrameVolume.js L339–353: the volume box as a matrix of the unit box.
    const box = toBox(bbox, source);
    this.bbox = box;
    this.boxFromSource = bbox === undefined;
    this.boxMatrix = boxMatrix(box, new THREE.Matrix4());
  }

  /** The mean density `normalizeDensity` divides by: `densityMean` when set, else the source's mean of the non-zero texels (his `meanDensity`). */
  get meanDensity(): number {
    return this.densityMean ?? this.source.stats.meanNonZero;
  }

  /** The phase this volume is drawn with in `'volume'` lighting: its own `phase*` fields where set, else the renderer's `settings` (a reused object). */
  getPhase(): Readonly<VDBFramePhaseSettings> {
    const s = this.frameRenderer.settings;
    const p = this.resolvedPhase;
    p.phase = this.phase ?? s.phase;
    p.phaseG = this.phaseG ?? s.phaseG;
    p.phaseBackG = this.phaseBackG ?? s.phaseBackG;
    p.phaseBackWeight = this.phaseBackWeight ?? s.phaseBackWeight;
    p.phaseGExponent = this.phaseGExponent ?? s.phaseGExponent;
    return p;
  }

  /**
   * A measured medium in one call (`'cloud'`, `'fog'`, `'ice'`, `'soot'`, `'white-smoke'`, `'dark-smoke'`, `'haze'`,
   * `'dust'`, or {@link VDBFrameMediumVolumeProps} from `vdbFrameMedium`), as his GUI's 'Medium' field: the volume's
   * `baseColor`, `extinctionScale` and `angstromExponent`; the phase on the renderer (`frameRenderer.settings.phase`,
   * `phaseG`, `phaseGExponent`: the phase of 'frame' lighting and the default of 'volume' lighting; the volume's own
   * phase overrides are cleared) and `settings.spectral` on. The renderer's settings are shared by every volume it
   * draws. `null` restores the volume's constructor colour, extinction scale and Ångström exponent (the renderer is left
   * as it is). Returns the props applied (null for `null`); an unknown name throws `VDB_BAD_OPTION`.
   */
  setMedium(medium: string | VDBFrameMediumVolumeProps | null): VDBFrameMediumVolumeProps | null {
    if (medium === null) {
      this.baseColor.copy(this.initialMedium.baseColor);
      this.extinctionScale.copy(this.initialMedium.extinctionScale);
      this.angstromExponent = this.initialMedium.angstromExponent;
      return null;
    }
    const props = typeof medium === "string" ? getFrameMediumProps(medium) : medium;
    if (!props) throw new VDBError("VDB_BAD_OPTION", `VDBFrameVolume.setMedium: unknown medium ${JSON.stringify(medium)}; use one of vdbFrameMediumNames (@pmndrs/vdb/tools).`);
    this.baseColor.copy(props.baseColor);
    this.extinctionScale.copy(props.extinctionScale);
    this.angstromExponent = props.angstromExponent;
    this.phase = this.phaseG = this.phaseBackG = this.phaseBackWeight = this.phaseGExponent = null;
    const s = this.frameRenderer.settings;
    s.phase = props.phase;
    s.phaseG = props.phaseG;
    s.phaseGExponent = props.phaseGExponent;
    s.spectral = true;
    return props;
  }

  /** The density scale the renderer draws with: `densityScale`, divided by the mean density with `normalizeDensity`. */
  getDensityScale(): number {
    const meanDensity = this.meanDensity;
    return this.normalizeDensity && meanDensity > 0.0 ? this.densityScale / meanDensity : this.densityScale;
  }

  /**
   * Re-reads the source's box (a dense frame with another `.wvdb` box, or a `followImage` sparse source after its grid's
   * image changed). The renderer calls {@link VDBFrameVolume.syncSource} before every render, so a `source.replace()`
   * needs no call; this forces it.
   */
  updateSource(): void {
    this.sourceVersion = this.source.version;
    // G16a: a sparse source's box (a followImage source re-reads it after an image change). RV: a dense frame's
    // .wvdb box (his WVDB sequences give every frame its own box; a null box, an empty frame, keeps the last one).
    const sourceBox = this.source.kind === "sparse" ? this.source.bbox : this.source.volume.bbox;
    if (this.boxFromSource && sourceBox) {
      const [min, max] = sourceBox;
      const b = this.bbox;
      if (b.min.x !== min[0] || b.min.y !== min[1] || b.min.z !== min[2] || b.max.x !== max[0] || b.max.y !== max[1] || b.max.z !== max[2]) {
        b.min.set(...min);
        b.max.set(...max);
        boxMatrix(b, this.boxMatrix);
      }
    }
  }

  /** {@link VDBFrameVolume.updateSource} if the source changed since (its `version`: a resized dense frame, a replaced pool image). Cheap; called per render. */
  syncSource(): void {
    if (this.source.version !== this.sourceVersion) this.updateSource();
  }

  /** Frees the proxy geometry. The source belongs to the caller; the renderer drops the volume when it leaves the scene. */
  dispose(): void {
    this.proxy.geometry.dispose();
  }
}
