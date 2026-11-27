/**
 * `VDBPoints`: the points of a parsed point data grid as round, instanced
 * sprites (K3.7, R1 §6 Step 7), three r187 `PointsNodeMaterial`.
 *
 *   const grid = parseVdb(bytes).grids.find((g) => g.valueType === 'ptdataidx32');
 *   const points = new VDBPoints(vdbPointRenderData(grid), { color: 0xffffff });
 *   scene.add(points);
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/points/webgpu/VDBPoints.js`, `points/pointsCore.js`):
 * a `Sprite` with one instance per point (WebGPU point primitives are 1 pixel),
 * the round alpha-tested disc, the attenuated size in pixels with a 1-pixel
 * minimum, `Cd` colours times the material colour, the default diameter of half
 * the mean spacing. Copyright (c) mjurczyk/openvdb contributors, MIT (see
 * THIRD_PARTY_NOTICES.md). Changes: takes the engine-free `vdbPointRenderData` (tools)
 * instead of decoding the grid itself; the pixel scale reads the render
 * target's height when one is set (DIVERGES.md).
 *
 * The sprite size and its clip-space offset are the core's WGSL (`vdbPointSpriteSize`,
 * `vdbPointSpriteVertex`, `core/surface/points.ts`, K6.6), the functions the raw pass
 * `vdbCreatePointPass` calls; the vertex node calls them through `wgslFn`. The disc map stays here.
 *
 * Positions are world space of the grid (the point list's own transform);
 * place them with the object's transform as any three object. The sprite sits
 * at the origin with the points in local space, so frustum culling is off.
 *
 * The arrays can hold more points than are drawn (`capacity` vs `pointCount`): a prefix-ordered list (tools
 * `vdbPointRenderData(grid, { order: 'prefix' })`) thins by count alone, and `.wvdb` points stream in LOD by LOD
 * (K3.7b): `VDBPoints.fromWvdbGrid(header)` sizes the arrays for the grid and `addWvdbLevel(level)` writes each
 * level a `WvdbStreamReader` decodes (only the new range is uploaded) and draws the points that have arrived.
 */

import * as THREE from "three";
import { InstancedBufferAttribute, PointsNodeMaterial, Sprite } from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBError } from "../core/errors.js";
import { vdbPointSpriteWgsl } from "../core/surface/points.js";
import type { WvdbPointsGrid, WvdbPointsGridHeader, WvdbPointsLevel } from "../tools/wvdb/index.js";
import type { VDBTslNode } from "./grid-node.js";

const { cameraProjectionMatrix, code, instancedBufferAttribute, materialColor, modelViewMatrix, positionGeometry, screenDPR, uniform, vec4, viewportSize, wgslFn } =
  TSL as unknown as Record<string, VDBTslNode>;

/**
 * The vertex entry: the core's sprite size (1-pixel minimum) times the device pixel ratio, then the
 * corner's clip position (three's `PointsNodeMaterial` sprite offset, same operation order).
 */
const SPRITE_WGSL = /* wgsl */ `fn vdbThreePointSprite(clip_centre: vec4f, corner: vec2f, pixel_scale: f32, view_depth: f32, dpr: f32, viewport_size: vec2f) -> vec4f {
    return vdbPointSpriteVertex(clip_centre, corner, vdbPointSpriteSize(pixel_scale, view_depth, 1.0) * dpr, viewport_size);
}`;
let spriteFn: VDBTslNode | null = null;

/** What `VDBPoints` draws (tools `vdbPointRenderData` returns one). */
export interface VDBPointsData {
  /** World positions of the grid, xyz per point. */
  positions: Float32Array;
  /** Linear RGB per point (multiplied by the material colour), or null. */
  colors?: Float32Array | null;
  /** Points to draw (default `positions.length / 3`); the arrays may hold more (`capacity`). */
  count?: number;
  /** Mean point spacing in world units: the default diameter is half of it. Default 1. */
  spacing?: number;
}

/** Options of {@link VDBPoints}: `PointsNodeMaterial` parameters plus the point diameter. */
export interface VDBPointsOptions extends THREE.SpriteMaterialParameters {
  /** World diameter of a point. Default `spacing * 0.5` (a gap between neighbours). */
  size?: number | null;
}

type ScalarUniform = ReturnType<typeof TSL.uniform> & { value: number };

let discTexture: THREE.DataTexture | null = null;

/** The 64x64 round alpha map every `VDBPoints` shares (alpha-tested at 0.5). */
export function vdbPointDiscTexture(): THREE.DataTexture {
  if (!discTexture) {
    const size = 64;
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = (x + 0.5) / size - 0.5;
        const dy = (y + 0.5) / size - 0.5;
        const v = Math.sqrt(dx * dx + dy * dy) <= 0.5 ? 255 : 0;
        data.set([v, v, v, 255], (y * size + x) * 4);
      }
    }
    discTexture = new THREE.DataTexture(data, size, size);
    discTexture.needsUpdate = true;
  }
  return discTexture;
}

const worldPosition = new THREE.Vector3();
const worldQuaternion = new THREE.Quaternion();
const worldScale = new THREE.Vector3();
const canvasSize = new THREE.Vector2();

/**
 * Sprite size (pixels at view depth 1, before the half-height factor) for a world diameter: his
 * `getAttenuatedPointSize`, `diameter * maxScale * zoom / tan(fov / 2)` (fov 50 for non-perspective cameras).
 */
export function vdbAttenuatedPointSize(object: THREE.Object3D, camera: THREE.Camera, diameter: number): number {
  const perspective = camera as THREE.PerspectiveCamera;
  const fov = perspective.isPerspectiveCamera ? perspective.fov : 50;
  const zoom = (camera as { zoom?: number }).zoom ?? 1;
  object.matrixWorld.decompose(worldPosition, worldQuaternion, worldScale);
  return (diameter * Math.max(worldScale.x, worldScale.y, worldScale.z) * zoom) / Math.tan(THREE.MathUtils.degToRad(fov) * 0.5);
}

interface RendererSizeLike {
  getSize(target: THREE.Vector2): THREE.Vector2;
  getRenderTarget?(): { height: number } | null;
}

/**
 * Round points of a point list as one instanced sprite (see the file comment).
 * `count` instances; the material is a `PointsNodeMaterial` (`color`, `opacity`
 * and the other sprite parameters apply; `Cd` multiplies the colour).
 */
export class VDBPoints extends Sprite {
  /** Type flag: true. */
  readonly isVDBPoints = true;
  declare material: PointsNodeMaterial;
  /** Mean spacing of the points (world units). */
  spacing: number;
  /** World diameter of a point, or null for `spacing * 0.5`. */
  size: number | null;
  /** Point diameter in pixels at view depth 1 (set before each render). */
  readonly pixelScale: ScalarUniform;
  /** The instanced position attribute (world xyz per point). */
  readonly positionAttribute: InstancedBufferAttribute;
  /** The instanced colour attribute (linear RGB per point), or null without colours. */
  readonly colorAttribute: InstancedBufferAttribute | null;
  private drawn = 0;
  /** `.wvdb` streaming: the grid's mean spacing over all its points, and the points written per LOD. */
  private wvdb: { spacing: number; written: Map<number, number> } | null = null;

  /**
   * Builds the sprite from render data (tools `vdbPointRenderData`).
   *
   * @param data World positions (xyz triples), optional colours, count and spacing.
   * @param options Sprite material parameters plus `size`, the world diameter of a point.
   */
  constructor(data: VDBPointsData, options: VDBPointsOptions = {}) {
    if (!data || !(data.positions instanceof Float32Array) || data.positions.length % 3 !== 0) {
      throw new VDBError("VDB_BAD_OPTION", "VDBPoints: data.positions must be a Float32Array of xyz triples (tools vdbPointRenderData).");
    }
    const count = data.count ?? data.positions.length / 3;
    if (!Number.isInteger(count) || count < 0 || count * 3 > data.positions.length) {
      throw new VDBError("VDB_BAD_OPTION", `VDBPoints: count ${count} does not fit ${data.positions.length / 3} positions.`);
    }
    if (data.colors && data.colors.length < count * 3) {
      throw new VDBError("VDB_BAD_OPTION", "VDBPoints: colors must hold count * 3 values.");
    }
    const { size = null, ...materialProps } = options;
    const capacity = data.positions.length / 3;
    super(
      new PointsNodeMaterial({
        alphaMap: vdbPointDiscTexture(),
        alphaTest: 0.5,
        ...materialProps,
        // sizeNode attenuates (with a 1-pixel minimum): sprites under a pixel are not drawn.
        sizeAttenuation: false,
      } as never) as never,
    );
    this.spacing = data.spacing ?? 1;
    this.size = size;
    this.pixelScale = uniform(1) as ScalarUniform;
    this.positionAttribute = new InstancedBufferAttribute(data.positions, 3);
    this.colorAttribute = data.colors ? new InstancedBufferAttribute(data.colors.length === capacity * 3 ? data.colors : data.colors.subarray(0, capacity * 3), 3) : null;
    const position = instancedBufferAttribute(this.positionAttribute);
    const view = modelViewMatrix.mul(vec4(position, 1.0));
    const m = this.material as PointsNodeMaterial & Record<string, VDBTslNode>;
    m.positionNode = position;
    spriteFn ??= wgslFn(SPRITE_WGSL, [code(vdbPointSpriteWgsl())] as never);
    // The clip position comes from the core sprite WGSL; sizeNode is unused (vertexNode replaces three's sprite offset).
    m.vertexNode = spriteFn({
      clip_centre: cameraProjectionMatrix.mul(view),
      corner: positionGeometry.xy,
      pixel_scale: this.pixelScale,
      view_depth: view.z.negate(),
      dpr: screenDPR,
      viewport_size: viewportSize,
    });
    m.colorNode = this.colorAttribute ? vec4(instancedBufferAttribute(this.colorAttribute), 1.0).mul(materialColor) : null;
    m.needsUpdate = true;
    this.pointCount = count;
    this.frustumCulled = false;
    this.onBeforeRender = (renderer, _scene, camera) => this.updatePixelScale(renderer as unknown as RendererSizeLike, camera);
  }

  /** Points drawn: the instance count, at most {@link capacity}. Set it to draw a prefix of the arrays. */
  get pointCount(): number {
    return this.drawn;
  }

  set pointCount(n: number) {
    if (!Number.isInteger(n) || n < 0 || n > this.capacity) {
      throw new VDBError("VDB_BAD_OPTION", `VDBPoints: pointCount ${n} must be an integer in [0, ${this.capacity}].`);
    }
    this.drawn = n;
    (this as unknown as { count: number }).count = n;
  }

  /** Points the arrays hold: the most {@link pointCount} can be. */
  get capacity(): number {
    return this.positionAttribute.count;
  }

  /**
   * Writes points `[first, first + n)` of the arrays (`n = positions.length / 3`) and uploads only that range at the
   * next render. Does not change {@link pointCount}.
   *
   * @param first The first point to write.
   * @param positions xyz per point.
   * @param colors rgb per point (n * 3 values), for a `VDBPoints` made with colours; null or omitted keeps them.
   */
  writePoints(first: number, positions: Float32Array, colors?: Float32Array | null): void {
    const n = positions.length / 3;
    if (!Number.isInteger(n) || !Number.isInteger(first) || first < 0 || first + n > this.capacity) {
      throw new VDBError("VDB_BAD_OPTION", `VDBPoints.writePoints: points [${first}, ${first + n}) do not fit the capacity ${this.capacity}.`);
    }
    if (colors && (!this.colorAttribute || colors.length < n * 3)) {
      throw new VDBError("VDB_BAD_OPTION", "VDBPoints.writePoints: colors need a VDBPoints made with colours and n * 3 values.");
    }
    if (n === 0) return;
    (this.positionAttribute.array as Float32Array).set(positions, first * 3);
    this.positionAttribute.addUpdateRange(first * 3, n * 3);
    this.positionAttribute.needsUpdate = true;
    if (colors && this.colorAttribute) {
      (this.colorAttribute.array as Float32Array).set(colors.subarray(0, n * 3), first * 3);
      this.colorAttribute.addUpdateRange(first * 3, n * 3);
      this.colorAttribute.needsUpdate = true;
    }
  }

  /**
   * An empty `VDBPoints` sized for a `.wvdb` points grid (K3.7b): capacity = the grid's points, colours when it has
   * them, nothing drawn. Feed it the grid's LODs with {@link addWvdbLevel} as a `WvdbStreamReader` decodes them.
   *
   * @param grid The grid's header record (`reader.header.grids[i]`, once the header arrived) or a decoded grid.
   * @param options Sprite material parameters plus `size`.
   */
  static fromWvdbGrid(grid: WvdbPointsGridHeader | WvdbPointsGrid, options: VDBPointsOptions = {}): VDBPoints {
    if (!grid || (grid as { type?: string }).type !== "points" && (grid as { kind?: string }).kind !== "points") {
      throw new VDBError("VDB_BAD_OPTION", "VDBPoints.fromWvdbGrid: grid must be a .wvdb points grid (header record or decoded grid).");
    }
    const count = grid.count;
    let spacing = grid.spacing;
    if (!(spacing !== undefined && spacing > 0)) {
      // The bbox estimate writeWvdb uses when no spacing is recorded: the box volume shared by the points.
      const b = grid.bbox;
      const volume = b ? Math.abs((b[1][0] - b[0][0]) * (b[1][1] - b[0][1]) * (b[1][2] - b[0][2])) : 0;
      spacing = count > 0 && volume > 0 ? Math.cbrt(volume / count) : 1;
    }
    const points = new VDBPoints(
      { positions: new Float32Array(count * 3), colors: grid.colors ? new Float32Array(count * 3) : null, count: 0, spacing },
      options,
    );
    points.wvdb = { spacing, written: new Map() };
    return points;
  }

  /**
   * Writes a `.wvdb` points level (a `WvdbStreamReader` event's level for this grid, complete or partial) at its
   * `firstPoint`, uploading only the points not written before, then draws every point that has arrived in order:
   * LOD k's points follow LODs 0..k-1, so the drawn prefix is an even subsample (every 8^(lods-1-k)-th point of the
   * stored order). The default diameter follows: `spacing` becomes the grid's spacing times `cbrt(capacity / pointCount)`.
   *
   * @param level A points level of the grid this `VDBPoints` was made for ({@link fromWvdbGrid}).
   */
  addWvdbLevel(level: WvdbPointsLevel): void {
    if (!this.wvdb) throw new VDBError("VDB_BAD_OPTION", "VDBPoints.addWvdbLevel: make the VDBPoints with VDBPoints.fromWvdbGrid(grid).");
    if (level?.kind !== "points") throw new VDBError("VDB_BAD_OPTION", "VDBPoints.addWvdbLevel: level must be a .wvdb points level.");
    const done = this.wvdb.written.get(level.lod) ?? 0;
    const upTo = level.pointsDecoded;
    if (upTo > done) {
      this.writePoints(level.firstPoint + done, level.positions.subarray(done * 3, upTo * 3), level.colors ? level.colors.subarray(done * 3, upTo * 3) : null);
      this.wvdb.written.set(level.lod, upTo);
    }
    // Draw the points that have arrived without a gap (the reader decodes LODs coarse first, pages in order).
    if (level.firstPoint <= this.pointCount) this.pointCount = Math.max(this.pointCount, level.firstPoint + upTo);
    this.spacing = this.wvdb.spacing * Math.cbrt(this.capacity / Math.max(1, this.pointCount));
  }

  /** The world diameter in use: `size`, or half the spacing. */
  get diameter(): number {
    return this.size ?? this.spacing * 0.5;
  }

  /**
   * Sets {@link pixelScale} for a camera and the renderer's current target (called before each render).
   *
   * @param renderer The renderer whose current target size gives the pixel height.
   * @param camera The camera whose field of view (or zoom) sets the scale.
   */
  updatePixelScale(renderer: RendererSizeLike, camera: THREE.Camera): void {
    const target = renderer.getRenderTarget?.() ?? null;
    const height = target ? target.height : renderer.getSize(canvasSize).y;
    this.pixelScale.value = vdbAttenuatedPointSize(this, camera, this.diameter) * height * 0.5;
  }

  /** Frees the material (the shared disc texture stays). */
  dispose(): void {
    this.material.dispose();
  }
}
