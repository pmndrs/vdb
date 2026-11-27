/**
 * `VDBLevelSetPhysicalMaterial` (G13a): the level-set trace of `VDBLevelSetMaterial` on three r187's
 * `MeshPhysicalNodeMaterial`, so a traced surface can be glass (`transmission`, `ior`, `thickness`,
 * `attenuationColor` / `attenuationDistance`, `dispersion`), lacquered (`clearcoat`), sheen or iridescent.
 *
 *   const grids = new VDBGridNode(renderer, pool, { profile: 'march', helpers: ['levelSet'] });
 *   const glass = new VDBLevelSetPhysicalMaterial(renderer, grids, id, { transmission: 1, ior: 1.5, thickness: 0.3, roughness: 0.05 });
 *   scene.add(new Mesh(vdbProxyGeometry(pool.info(id), { pad: 1 }), glass));
 *
 * The trace, mask, depth and normal are `VDBLevelSetMaterial`'s (one setup, `vdbLevelSetTrace`); this class adds
 * three's physical lighting at the hit. Two terms of three's physical model read the proxy box rather than the
 * shading point, and are rewired here:
 * - the clearcoat normal (`clearcoatNormalNode`) is the trace's view normal;
 * - transmission (screen-space refraction) starts its refracted ray at the hit. three's `PhysicalLightingModel`
 *   reads `positionWorld`, the proxy's back-face varying, which lies on the same view ray but farther away, so the
 *   refracted sample would land too close to the pixel. {@link VDBLevelSetPhysicalLightingModel} builds the
 *   same refraction from the hit (`hitWorldNode`) instead.
 *
 * Anisotropy needs mesh tangents and is not supported on a traced surface (it shades with the box's tangents).
 *
 * The refraction below is a port of three.js r187 `src/nodes/functions/PhysicalLightingModel.js`
 * (`getVolumeTransmissionRay`, `applyIorToRoughness`, `getTransmissionSample`, `volumeAttenuation`,
 * `getIBLVolumeRefraction`). Copyright 2010-2026 three.js authors, MIT. Change: the shading position is the
 * trace's hit, and the model matrix is the identity (the proxy mesh has an identity transform, so `thickness` is
 * in world units).
 */

import * as THREE from "three";
import { MeshPhysicalNodeMaterial, PhysicalLightingModel } from "three/webgpu";
import * as TSL from "three/tsl";
import { type VDBThreeRendererLike } from "./device.js";
import { VDBGridNode, type VDBTslNode } from "./grid-node.js";
import {
  vdbApplyLevelSetTrace,
  vdbLevelSetLightingPosition,
  vdbLevelSetTrace,
  type VDBLevelSetTrace,
  type VDBLevelSetTraceParameters,
} from "./level-set-material.js";
import type { VDBUintUniform } from "./material.js";

const { EnvironmentBRDF, Fn, If, Loop, cameraPosition, cameraProjectionMatrix, cameraViewMatrix, cameraViewport, exp, float, log, log2, mix, normalWorld, refract, screenSize, textureBicubicLevel, vec2, vec3, vec4 } =
  TSL as unknown as Record<string, VDBTslNode>;
const { attenuationColor, attenuationDistance, diffuseColor, diffuseContribution, dispersion, ior, roughness, specularColorBlended, specularF90, thickness, transmission, viewportMipTexture } =
  TSL as unknown as Record<string, VDBTslNode>;

/** Parameters of {@link VDBLevelSetPhysicalMaterial}: three's physical material parameters plus the trace's. */
export interface VDBLevelSetPhysicalMaterialParameters extends THREE.MeshPhysicalMaterialParameters, VDBLevelSetTraceParameters {}

type ScalarUniform = ReturnType<typeof TSL.uniform> & { value: number };

// --- three r187's screen-space refraction, from the hit (see the file comment) ---------------------------------

/** The framebuffer copy the refraction samples (three's back-side texture: the material draws back faces). */
let backdropTexture: VDBTslNode | null = null;

function transmissionSample(fragCoord: VDBTslNode, rough: VDBTslNode, eta: VDBTslNode): VDBTslNode {
  backdropTexture ??= viewportMipTexture();
  const sample = backdropTexture.sample(fragCoord.mul(cameraViewport.zw).add(cameraViewport.xy).div(screenSize));
  // applyIorToRoughness: an ior of 1 gives no microfacet refraction, 1.5 the default amount.
  const lod = log2(cameraViewport.z).mul(rough.mul(eta.mul(2.0).sub(2.0).clamp(0.0, 1.0)));
  return textureBicubicLevel(sample, lod);
}

/** Beer's law over the transmission distance; no attenuation for a zero attenuation distance (three's rule). */
const volumeAttenuation = Fn(([distance, color, attenuationDist]: VDBTslNode[]) => {
  const out = vec3(1.0).toVar();
  If(attenuationDist.notEqual(0), () => {
    out.assign(exp(log(color).negate().div(attenuationDist).negate().mul(distance)));
  });
  return out;
});

/** Refraction exit point's screen uv (WebGPU y down) for an ior: the hit plus the refracted direction times the thickness. */
function refractionCoords(position: VDBTslNode, n: VDBTslNode, v: VDBTslNode, eta: VDBTslNode, thick: VDBTslNode): { coords: VDBTslNode; ray: VDBTslNode } {
  const ray = refract(v.negate(), n.normalize(), float(1.0).div(eta)).normalize().mul(thick);
  const ndc = cameraProjectionMatrix.mul(cameraViewMatrix.mul(vec4(position.add(ray), 1.0)));
  const uv = ndc.xy.div(ndc.w).add(1.0).div(2.0);
  return { coords: vec2(uv.x, uv.y.oneMinus()), ray };
}

/** three's `getIBLVolumeRefraction` with `position` the hit and an identity model matrix. */
function volumeRefraction(position: VDBTslNode, withDispersion: boolean): VDBTslNode {
  const v = cameraPosition.sub(position).normalize();
  const n = normalWorld;
  let transmitted: VDBTslNode;
  let transmittance: VDBTslNode;
  if (withDispersion) {
    transmitted = vec4().toVar();
    transmittance = vec3().toVar();
    const halfSpread = ior.sub(1.0).mul(dispersion.mul(0.025));
    const iors = vec3(ior.sub(halfSpread), ior, ior.add(halfSpread));
    Loop({ start: 0, end: 3 }, ({ i }: { i: VDBTslNode }) => {
      const eta = iors.element(i);
      const { coords, ray } = refractionCoords(position, n, v, eta, thickness);
      const sample = transmissionSample(coords, roughness, eta);
      transmitted.element(i).assign(sample.element(i));
      transmitted.a.addAssign(sample.a);
      transmittance.element(i).assign(diffuseContribution.element(i).mul(volumeAttenuation(ray.length(), attenuationColor, attenuationDistance).element(i)));
    });
    transmitted.a.divAssign(3.0);
  } else {
    const { coords, ray } = refractionCoords(position, n, v, ior, thickness);
    transmitted = transmissionSample(coords, roughness, ior);
    transmittance = diffuseContribution.mul(volumeAttenuation(ray.length(), attenuationColor, attenuationDistance));
  }
  const attenuated = transmittance.rgb.mul(transmitted.rgb);
  const F = vec3(EnvironmentBRDF({ dotNV: n.dot(v).clamp(), specularColor: specularColorBlended, specularF90, roughness }));
  const factor = transmittance.r.add(transmittance.g, transmittance.b).div(3.0);
  return vec4(F.oneMinus().mul(attenuated), transmitted.a.oneMinus().mul(factor).oneMinus());
}

/**
 * three's `PhysicalLightingModel` with transmission refracted from the trace's hit (see the file comment). Every
 * other term is three's: `start` runs with transmission off, then sets the backdrop three would.
 */
export class VDBLevelSetPhysicalLightingModel extends PhysicalLightingModel {
  /** The world hit the refraction starts at. */
  readonly hitWorld: VDBTslNode;

  constructor(hitWorld: VDBTslNode, flags: ConstructorParameters<typeof PhysicalLightingModel>) {
    super(...flags);
    this.hitWorld = hitWorld;
  }

  override start(builder: Parameters<PhysicalLightingModel["start"]>[0]): void {
    const self = this as unknown as { transmission: boolean; dispersion: boolean };
    const withTransmission = self.transmission;
    self.transmission = false;
    try {
      super.start(builder);
    } finally {
      self.transmission = withTransmission;
    }
    if (!withTransmission) return;
    const context = (builder as unknown as { context: { backdrop: VDBTslNode; backdropAlpha: VDBTslNode } }).context;
    context.backdrop = volumeRefraction(this.hitWorld, self.dispersion);
    context.backdropAlpha = transmission;
    diffuseColor.a.mulAssign(mix(1, context.backdrop.a, transmission));
  }
}

/**
 * A level set's surface (as {@link VDBLevelSetMaterial}) shaded by three's physical material: transmission,
 * clearcoat, sheen, iridescence, `ior`, `specularIntensity` (see the file comment).
 *
 * @typeParam TGridId What the constructor got as the id: a number (default) makes {@link gridId} a `uint`
 *   uniform; a node is kept as given.
 */
export class VDBLevelSetPhysicalMaterial<TGridId extends number | VDBTslNode = number> extends MeshPhysicalNodeMaterial implements VDBLevelSetTrace<TGridId> {
  /** Type flag: true (a level-set material, as `VDBLevelSetMaterial`). */
  readonly isVDBLevelSetMaterial = true;
  /** Type flag: true. */
  readonly isVDBLevelSetPhysicalMaterial = true;
  /** The pool node this material reads. */
  declare readonly grids: VDBGridNode;
  /** The grid id: a `uint` uniform for a numeric id (`material.gridId.value = otherId`), else the caller's node. */
  declare readonly gridId: TGridId extends number ? VDBUintUniform : VDBTslNode;
  /** Index voxels per SDF unit (live). */
  declare readonly sdfToIndex: ScalarUniform;
  /** Trilinear samples per trace (live, clamped to the compiled cap). */
  declare readonly maxSamples: ScalarUniform;
  /** Central-difference step of the normal in index voxels (live). */
  declare readonly normalStep: ScalarUniform;
  /** The entry's WGSL (for inspection and tests). */
  declare readonly traceSource: string;
  /** The packed surface (`mat4`, fragment stage). */
  declare readonly surfaceNode: VDBTslNode;
  /** `vec4(hit world position, 1)` on a hit, `vec4(0)` on a miss (fragment stage). */
  declare readonly traceNode: VDBTslNode;
  /** The hit in world space (where the refraction starts). */
  declare readonly hitWorldNode: VDBTslNode;
  /** The hit in view space. */
  declare readonly hitViewNode: VDBTslNode;
  /** True where the ray hit the surface (the material's `maskNode`). */
  declare readonly hitMaskNode: VDBTslNode;
  /** World-space unit normal at the hit. */
  declare readonly normalWorldNode: VDBTslNode;
  /** View-space unit normal at the hit (the material's `normalNode` and `clearcoatNormalNode`). */
  declare readonly normalViewNode: VDBTslNode;

  /**
   * @param renderer The initialized `WebGPURenderer` the pool and `grids` live on.
   * @param grids A `VDBGridNode` on this renderer with profile `'march'` and helpers including `'levelSet'`.
   * @param gridId The level set's pool id (a number, made a `uint` uniform) or a `uint` node.
   * @param params three's `MeshPhysicalMaterialParameters` plus the trace's ({@link VDBLevelSetPhysicalMaterialParameters}).
   * @throws VDBError `VDB_BAD_OPTION` for a `grids` this material can't use, a bad id or cap;
   *   `VDB_UNSUPPORTED_DEVICE` for an uninitialized renderer.
   */
  constructor(renderer: VDBThreeRendererLike, grids: VDBGridNode, gridId: TGridId, params: VDBLevelSetPhysicalMaterialParameters = {}) {
    const { sdfToIndex: _s, maxSamples: _m, maxSamplesCap: _c, normalStep: _n, ...physical } = params;
    super(physical);
    this.name = "VDBLevelSetPhysicalMaterial";
    Object.assign(this, vdbLevelSetTrace("VDBLevelSetPhysicalMaterial", renderer, grids, gridId, params));
    vdbApplyLevelSetTrace(this, this);
    (this as unknown as { clearcoatNormalNode: VDBTslNode }).clearcoatNormalNode = this.normalViewNode;
  }

  /** Lights read `positionView`: the shading point is the hit, not the box face. */
  override setupLighting(builder: Parameters<MeshPhysicalNodeMaterial["setupLighting"]>[0]): ReturnType<MeshPhysicalNodeMaterial["setupLighting"]> {
    vdbLevelSetLightingPosition(this);
    return super.setupLighting(builder);
  }

  /** three's physical lighting model, with transmission refracted from the hit. */
  override setupLightingModel(builder?: unknown): PhysicalLightingModel {
    void builder;
    const m = this as unknown as Record<string, boolean>;
    return new VDBLevelSetPhysicalLightingModel(this.hitWorldNode, [
      m.useClearcoat!,
      m.useSheen!,
      m.useIridescence!,
      m.useAnisotropy!,
      m.useTransmission!,
      m.useDispersion!,
      m.useRetroreflection!,
      m.useDiffuseRoughness!,
    ] as ConstructorParameters<typeof PhysicalLightingModel>);
  }
}
