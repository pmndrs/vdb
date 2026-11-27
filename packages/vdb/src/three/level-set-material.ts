/**
 * `VDBLevelSetMaterial`: a NanoVDB level set's zero surface, traced on the GPU
 * on the sparse pool (K3.7, R1 §6 Step 7), as a three r187
 * `MeshStandardNodeMaterial` (lights, shadows, depth, fog: three's own).
 *
 *   const grids = new VDBGridNode(renderer, pool, { profile: 'march', helpers: ['levelSet'] });
 *   const material = new VDBLevelSetMaterial(renderer, grids, id, { color: 0xffffff, roughness: 0.5 });
 *   scene.add(new Mesh(vdbProxyGeometry(pool.info(id), { pad: 1 }), material));
 *   // per frame: pool.update(encoder) (or pool.flush()), then grids.update(), then render.
 *
 * The trace, normal and pixel ray are the core's `vdbLevelSetSurface` / `vdbLevelSetViewRay` WGSL
 * (`core/surface/level-set-wgsl.ts`, K6.6), the same functions the raw pass `vdbCreateLevelSetPass`
 * calls; this material adds three's lighting, shadows and depth conventions around them.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/levelset/webgpu/LevelSetSurfaceMaterial.js`,
 * `LevelSetSurfaceCore.js`): the back-face proxy box (the camera can be
 * inside), perspective and orthographic rays (so shadow passes work), the mask,
 * depth, received-shadow position and the `setupLighting` override that moves
 * `positionView` to the hit. Copyright (c) mjurczyk/openvdb contributors, MIT
 * (see THIRD_PARTY_NOTICES.md). Changes: the trace is the core's `vdbLevelSetHit` on
 * the sparse tree (HDDA skip, full-resolution trilinear SDF) instead of a
 * sphere trace of a dense 8-bit texture; the ray comes from the pixel through
 * the inverse projection, not from interpolated box positions; the normal is
 * `vdbLevelSetNormal` (central differences at `normalStep` voxels through the
 * grid affine); the proxy is in world space and the grid is placed with
 * `pool.setTransform` (DIVERGES.md).
 *
 * The trace runs in the grid's index space: `o = worldToIndex(ray origin)`,
 * `d = worldToIndexDir(unit world direction)`, so `t` is in world units. The
 * ray is the pixel's (`viewportUV` through `cameraProjectionMatrixInverse`):
 * perspective from `cameraPosition` over `t >= 0`; orthographic from the pixel
 * on the camera plane along the view axis over `t` in `[-1e5, 1e5]`. The mesh
 * must have an identity transform (the pool record holds the grid's
 * placement); give it `vdbProxyGeometry(pool.info(id), { pad: 1 })`, which
 * covers the trilinear support the trace clips to.
 */

import * as THREE from "three";
import { MeshStandardNodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBError } from "../core/errors.js";
import { vdbLevelSetSurfaceWgsl } from "../core/surface/level-set-wgsl.js";
import { getDevice, type VDBThreeRendererLike } from "./device.js";
import { VDBGridNode, type VDBTslNode } from "./grid-node.js";
import type { VDBUintUniform } from "./material.js";

const { Fn, code, uint, uniform, vec4, wgslFn, positionLocal, positionView, viewportUV } = TSL as unknown as Record<string, VDBTslNode>;
const { cameraFar, cameraNear, cameraProjectionMatrix, cameraProjectionMatrixInverse, cameraViewMatrix, cameraWorldMatrix, modelViewMatrix } =
  TSL as unknown as Record<string, VDBTslNode>;
const { viewZToOrthographicDepth, viewZToPerspectiveDepth } = TSL as unknown as Record<string, VDBTslNode>;

/** Default compiled cap on the samples one fragment's trace takes (the live `maxSamples` clamps under it). */
export const VDB_LEVEL_SET_SAMPLES_CAP = 1024;

/** Parameters of {@link VDBLevelSetMaterial}: three's standard material parameters plus the trace's. */
export interface VDBLevelSetMaterialParameters extends THREE.MeshStandardMaterialParameters, VDBLevelSetTraceParameters {}

type ScalarUniform = ReturnType<typeof TSL.uniform> & { value: number };

const SURFACE = "vdbThreeLevelSetSurface";

/**
 * The material's entry: the core's pixel ray and surface (one call site: compile time, K3.7), packed into a
 * `mat4x4f` for TSL (three's `wgslFn` takes built-in types only): column 0 `(world hit, hit ? 1 : 0)`,
 * column 1 `(world normal, t)`, column 2 `(ndc depth, 0, 0, 0)`.
 */
function surfaceWgsl(cap: number): string {
  return /* wgsl */ `fn ${SURFACE}(grid_id: u32, uv: vec2f, projection_inverse: mat4x4f, camera_world: mat4x4f, view_projection: mat4x4f, orthographic: u32, sdf_to_index: f32, max_samples: f32, normal_step: f32) -> mat4x4f {
    let ray = vdbLevelSetViewRay(uv, projection_inverse, camera_world, orthographic != 0u);
    let s = vdbLevelSetSurface(grid_id, ray, view_projection, sdf_to_index, u32(clamp(max_samples, 1.0, ${cap}.0)), normal_step);
    return mat4x4f(vec4f(s.world, select(0.0, 1.0, s.hit)), vec4f(s.normal, s.t), vec4f(s.ndcDepth, 0.0, 0.0, 0.0), vec4f(0.0));
}`;
}

/** One surface `CodeNode` per `VDBGridNode` (included after its module; deduplicated within a shader). */
const surfaceCode = new WeakMap<VDBGridNode, VDBTslNode>();
function surfaceCodeOf(grids: VDBGridNode): VDBTslNode {
  let c = surfaceCode.get(grids);
  if (!c) surfaceCode.set(grids, (c = code(vdbLevelSetSurfaceWgsl(), [grids.module])));
  return c;
}

function bad(owner: string, message: string): VDBError {
  return new VDBError("VDB_BAD_OPTION", `${owner}: ${message}`);
}

/** The trace a level-set material shades: its uniforms and TSL nodes (shared by the standard and physical materials). */
export interface VDBLevelSetTrace<TGridId extends number | VDBTslNode = number> {
  readonly grids: VDBGridNode;
  readonly gridId: TGridId extends number ? VDBUintUniform : VDBTslNode;
  readonly sdfToIndex: ScalarUniform;
  readonly maxSamples: ScalarUniform;
  readonly normalStep: ScalarUniform;
  readonly traceSource: string;
  readonly surfaceNode: VDBTslNode;
  readonly traceNode: VDBTslNode;
  readonly hitWorldNode: VDBTslNode;
  readonly hitViewNode: VDBTslNode;
  readonly hitMaskNode: VDBTslNode;
  readonly normalWorldNode: VDBTslNode;
  readonly normalViewNode: VDBTslNode;
}

/** The trace parameters of {@link VDBLevelSetMaterialParameters} (also taken by `VDBLevelSetPhysicalMaterial`). */
export interface VDBLevelSetTraceParameters {
  /**
   * Index voxels per SDF unit. Default: `1 / voxelSize.x` from `grids.pool.info(id)` when the pool
   * has `info` and the id is a number, else 1. A level set stores distances in its own world units,
   * so this is 1 / its NanoVDB voxel size.
   */
  sdfToIndex?: number;
  /** Trilinear samples per fragment trace (live), clamped to `[1, maxSamplesCap]`. Default 512. */
  maxSamples?: number;
  /** Compiled cap on {@link maxSamples}. Default {@link VDB_LEVEL_SET_SAMPLES_CAP}. */
  maxSamplesCap?: number;
  /** Central-difference step of the normal, in index voxels (live). Default 1. */
  normalStep?: number;
}

/**
 * Validates the inputs and builds the trace's uniforms and nodes (G13a: one setup for both level-set materials).
 * `owner` names the class in error messages.
 */
export function vdbLevelSetTrace<TGridId extends number | VDBTslNode>(
  owner: string,
  renderer: VDBThreeRendererLike,
  grids: VDBGridNode,
  gridId: TGridId,
  params: VDBLevelSetTraceParameters,
): VDBLevelSetTrace<TGridId> {
  const { sdfToIndex, maxSamples, maxSamplesCap, normalStep } = params;
  if (!(grids instanceof VDBGridNode)) throw bad(owner, "grids must be a VDBGridNode.");
  getDevice(renderer);
  if (grids.renderer !== renderer) {
    throw bad(owner, "grids was made for another renderer: create one VDBGridNode per renderer.");
  }
  if (grids.options.profile !== "march" || !grids.options.helpers?.includes("levelSet")) {
    throw bad(owner, "grids needs profile 'march' and the 'levelSet' helper: new VDBGridNode(renderer, pool, { profile: 'march', helpers: ['levelSet'] }).");
  }
  if (typeof gridId === "number") {
    if (!Number.isInteger(gridId) || gridId < 0) throw bad(owner, `gridId must be a non-negative integer, got ${gridId}.`);
  } else if (gridId === null || typeof gridId !== "object") {
    throw bad(owner, "gridId must be a number or a uint TSL node.");
  }
  const cap = maxSamplesCap ?? VDB_LEVEL_SET_SAMPLES_CAP;
  if (!Number.isInteger(cap) || cap < 1) throw bad(owner, `maxSamplesCap must be an integer >= 1, got ${JSON.stringify(maxSamplesCap)}.`);
  let toIndex = sdfToIndex;
  if (toIndex === undefined) {
    const info = typeof gridId === "number" ? (grids.pool as { info?: (id: number) => { voxelSize: readonly number[] } }).info?.(gridId) : undefined;
    toIndex = info ? 1 / info.voxelSize[0]! : 1;
  }
  if (!(Number.isFinite(toIndex) && toIndex > 0)) throw bad(owner, `sdfToIndex must be a positive finite number, got ${toIndex}.`);

  const id = (typeof gridId === "number" ? uniform(gridId, "uint") : gridId) as VDBLevelSetTrace<TGridId>["gridId"];
  const toIndexUniform = uniform(toIndex) as ScalarUniform;
  const samples = uniform(maxSamples ?? 512) as ScalarUniform;
  const step = uniform(normalStep ?? 1) as ScalarUniform;
  const traceSource = surfaceWgsl(cap);
  const surfaceFn = wgslFn(traceSource, [surfaceCodeOf(grids)] as never);

  // The pixel's view ray (core vdbLevelSetViewRay): from the fragment's viewport uv through the inverse
  // projection, not from interpolated box positions, whose varying precision moves the ray by ~0.02 px on
  // large faces (V57). Perspective from the camera; orthographic (directional-light shadow maps) from the
  // pixel on the camera plane along the view axis.
  const surfaceNode = Fn((builder: { camera?: { isOrthographicCamera?: boolean } }) =>
    surfaceFn({
      grid_id: id,
      uv: viewportUV,
      projection_inverse: cameraProjectionMatrixInverse,
      camera_world: cameraWorldMatrix,
      view_projection: cameraProjectionMatrix.mul(cameraViewMatrix),
      orthographic: uint(builder.camera?.isOrthographicCamera === true ? 1 : 0),
      sdf_to_index: toIndexUniform,
      max_samples: samples,
      normal_step: step,
    }),
  )
    .once()()
    .toVar("vdbThreeLevelSetSurfaceOut");
  const traceNode = surfaceNode.element(0);
  const hitWorldNode = traceNode.xyz;
  const normalWorldNode = surfaceNode.element(1).xyz;
  return {
    grids,
    gridId: id,
    sdfToIndex: toIndexUniform,
    maxSamples: samples,
    normalStep: step,
    traceSource,
    surfaceNode,
    traceNode,
    hitWorldNode,
    hitMaskNode: traceNode.w.greaterThan(0.5),
    hitViewNode: cameraViewMatrix.mul(vec4(hitWorldNode, 1.0)).xyz.toVar("vdbLevelSetHitView"),
    normalWorldNode,
    normalViewNode: cameraViewMatrix.mul(vec4(normalWorldNode, 0.0)).xyz.normalize(),
  };
}

/**
 * Puts a trace on a node material: the back-face proxy, the hit mask, the vertex node (so `positionView` is a
 * variable the lighting moves to the hit), the hit's depth, the received-shadow position and the normal.
 */
export function vdbApplyLevelSetTrace(material: THREE.Material, trace: VDBLevelSetTrace<number | VDBTslNode>): void {
  const m = material as THREE.Material & Record<string, unknown>;
  m.side = THREE.BackSide;
  m.shadowSide = THREE.BackSide;
  m.maskNode = trace.hitMaskNode;
  // A vertexNode makes the fragment's positionView a variable, which setupLighting moves to the hit.
  m.vertexNode = cameraProjectionMatrix.mul(modelViewMatrix.mul(vec4(positionLocal, 1.0)));
  const hitView = trace.hitViewNode;
  m.depthNode = Fn((builder: { camera?: { isOrthographicCamera?: boolean } }) =>
    builder.camera?.isOrthographicCamera
      ? viewZToOrthographicDepth(hitView.z, cameraNear, cameraFar)
      : viewZToPerspectiveDepth(hitView.z, cameraNear, cameraFar),
  )();
  m.receivedShadowPositionNode = trace.hitWorldNode;
  m.normalNode = trace.normalViewNode;
}

/** Lights read `positionView`: the shading point is the hit, not the box face (both materials' `setupLighting`). */
export function vdbLevelSetLightingPosition(trace: VDBLevelSetTrace<number | VDBTslNode>): void {
  positionView.assign(trace.hitViewNode).toStack();
}

/**
 * A level set's surface on a world-space proxy box (see the file comment).
 * Shades as a `MeshStandardNodeMaterial`: `color`, `roughness`, `metalness`,
 * maps-free; lights and shadows read the hit point.
 *
 * - Exposes the trace as TSL nodes for custom shading or debugging:
 *   {@link hitWorldNode}, {@link hitViewNode}, {@link hitMaskNode},
 *   {@link normalWorldNode}, {@link normalViewNode} (e.g. `material.colorNode =
 *   material.normalViewNode.mul(0.5).add(0.5)` with no lights shows the normals).
 * - Perspective and orthographic cameras (directional-light shadow maps).
 *
 * @typeParam TGridId What the constructor got as the id: a number (default)
 *   makes {@link gridId} a `uint` uniform; a node is kept as given.
 */
export class VDBLevelSetMaterial<TGridId extends number | VDBTslNode = number> extends MeshStandardNodeMaterial {
  /** Type flag: true. */
  readonly isVDBLevelSetMaterial = true;
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
  /** The entry's WGSL (for inspection and tests): calls the core's `vdbLevelSetViewRay` and `vdbLevelSetSurface`. */
  declare readonly traceSource: string;
  /** The packed surface (`mat4`, fragment stage): see {@link traceNode}, {@link normalWorldNode}; column 2 `.x` is the NDC depth. */
  declare readonly surfaceNode: VDBTslNode;
  /** `vec4(hit world position, 1)` on a hit, `vec4(0)` on a miss (fragment stage). */
  declare readonly traceNode: VDBTslNode;
  /** The hit in world space. */
  declare readonly hitWorldNode: VDBTslNode;
  /** The hit in view space. */
  declare readonly hitViewNode: VDBTslNode;
  /** True where the ray hit the surface (the material's `maskNode`). */
  declare readonly hitMaskNode: VDBTslNode;
  /** World-space unit normal at the hit. */
  declare readonly normalWorldNode: VDBTslNode;
  /** View-space unit normal at the hit (the material's `normalNode`). */
  declare readonly normalViewNode: VDBTslNode;

  /**
   * @param renderer The initialized `WebGPURenderer` the pool and `grids` live on.
   * @param grids A `VDBGridNode` on this renderer with profile `'march'` and helpers including `'levelSet'`.
   * @param gridId The level set's pool id (a number, made a `uint` uniform) or a `uint` node.
   * @param params three's `MeshStandardMaterialParameters` plus the trace's ({@link VDBLevelSetMaterialParameters}).
   * @throws VDBError `VDB_BAD_OPTION` for a `grids` this material can't use, a bad id or cap;
   *   `VDB_UNSUPPORTED_DEVICE` for an uninitialized renderer.
   */
  constructor(renderer: VDBThreeRendererLike, grids: VDBGridNode, gridId: TGridId, params: VDBLevelSetMaterialParameters = {}) {
    const { sdfToIndex: _s, maxSamples: _m, maxSamplesCap: _c, normalStep: _n, ...standard } = params;
    super(standard);
    this.name = "VDBLevelSetMaterial";
    Object.assign(this, vdbLevelSetTrace("VDBLevelSetMaterial", renderer, grids, gridId, params));
    vdbApplyLevelSetTrace(this, this);
  }

  /** Lights read `positionView`: the shading point is the hit, not the box face. */
  override setupLighting(builder: Parameters<MeshStandardNodeMaterial["setupLighting"]>[0]): ReturnType<MeshStandardNodeMaterial["setupLighting"]> {
    vdbLevelSetLightingPosition(this);
    return super.setupLighting(builder);
  }
}
