/**
 * The `majorant` profile's API layer (P1.3, P1.4; docs/dev/design/K1-core.md §3.2, §5): a piecewise-constant
 * bound of the sampled field along an index-space ray, for delta and ratio tracking.
 *
 * ```wgsl
 * var it = vdbMajorantInit(g, oIdx, dIdx, tMin, tMax);   // index-space ray, d unnormalized, t the caller's
 * loop { let s = vdbMajorantNext(&it); if s.y <= s.x { break; } } // s.z bounds every sample on [s.x, s.y)
 * ```
 *
 * Segments are contiguous, never of zero length, and cover the walk's box: the brick grid's cells
 * (`'brick'`), else the sampler's support box (`[min - 1, max + 1]` trilinear, `[min - 0.5, max + 0.5]`
 * nearest). Outside it the field is 0 with `outside: 'zero'` (with `'background'` it is the tree's
 * background, which the walk does not cover: use a fog grid with background 0).
 *
 * - `'brick'`: the pathtracer's DDA (`ptDdaInit` / `ptDdaNext`) over the brick cells in cell space
 *   `(p - origin) / B`; a grid without a brick range (or a 0.1 pool's range, which has no "VDBM" header)
 *   walks as `'global'`.
 * - `'global'`: one segment with the record's `valueMax` (`vdbMajorantGlobal(g)`, the scalar a VSPG or
 *   emission estimator needs).
 * - `'nodeMax'` (needs the march layer): the vendored HDDA at leaf size 8 widening over tiles; a cell's
 *   bound is the max of `vdbNodeMax` over the 2x2x2 block of cells from its corner (the 8-leaf support of
 *   a trilinear sample). Stats bound ACTIVE values: inactive leaf voxels must be <= their leaf's max (fog).
 *
 * With `outside: 'zero'` a bound below 0 becomes 0 (a tap outside the bbox reads 0). With `minima`,
 * `vdbMajorantMinimum(&it)` is the lower bound of the segment the last `vdbMajorantNext` returned: the
 * brick minimum when the range carries minima, else 0 (a minorant of a non-negative field).
 *
 * CPU twin: `tools` `majorantWalk` (internal). Axis-parallel rays: a direction component below 1e-20 in
 * magnitude never divides (WGSL may assume no infinities; CLAUDE.md Traps).
 */
import type { ApiOptions } from "./api.js";

/** The `majorant` module option: which walk the profile compiles. */
export type VDBMajorantMode = "brick" | "global" | "nodeMax";

/** `vdb_maj_clip`: slab test with the axis-parallel guard. */
const CLIP = `fn vdb_maj_clip(lo: vec3f, hi: vec3f, o: vec3f, d: vec3f, t0: f32, t1: f32) -> vec2f {
    var t = vec2f(t0, t1);
    for (var k = 0; k < 3; k++) {
        if abs(d[k]) < 1e-20 {
            if o[k] < lo[k] || o[k] >= hi[k] {
                return vec2f(t0, t0);
            }
        } else {
            let p = (lo[k] - o[k]) / d[k];
            let q = (hi[k] - o[k]) / d[k];
            t = vec2f(max(t.x, min(p, q)), min(t.y, max(p, q)));
        }
    }
    return t;
}`;

/** The majorant layer for the normalized options (`o.majorant`, `o.minima`, `o.sampler`, `o.outside`). */
export function majorantWgsl(o: ApiOptions): string {
  const zero = o.outside === "zero";
  const pad = o.sampler === "nearest" ? "0.5" : "1.0";
  const vmax = zero ? "max(bitcast<f32>(vdb_rec(g, VDB_REC_VALUE_MAX)), 0.0)" : "bitcast<f32>(vdb_rec(g, VDB_REC_VALUE_MAX))";
  const support = `vdb_maj_clip(vec3f(vdbIndexMin(g)) - vec3f(${pad}), vec3f(vdbIndexMax(g)) + vec3f(${pad}), o, d, tMin, tMax)`;
  const parts = [
    CLIP,
    `fn vdbMajorantGlobal(g: u32) -> f32 {
    if !vdb_live(g) {
        return 0.0;
    }
    return ${vmax};
}`,
  ];
  const lo = o.minima ? "\n    lo: f32," : "";

  if (o.majorant === "nodeMax") {
    parts.push(
      `struct VdbMajorantIter {
    t: f32,
    te: f32,
    o: vec3f,
    d: vec3f,
    h: VdbHdda,
    acc: VdbAccessor,${lo}
}`,
      `fn vdb_maj_bound(acc: ptr<function, VdbAccessor>, ijk: vec3i, dim: u32) -> f32 {
    var m = vdbNodeMax(acc, ijk, dim);
    for (var k = 1u; k < 8u; k++) {
        m = max(m, vdbNodeMax(acc, ijk + vec3i(vec3u(k & 1u, (k >> 1u) & 1u, k >> 2u)) * i32(dim), dim));
    }
    return ${zero ? "max(m, 0.0)" : "m"};
}`,
      `fn vdbMajorantInit(g: u32, o: vec3f, d: vec3f, tMin: f32, tMax: f32) -> VdbMajorantIter {
    var it: VdbMajorantIter;
    it.t = tMin;
    it.te = tMin;
    if !vdb_live(g) {
        return it;
    }
    let c = ${support};
    if c.x < c.y {
        it.t = c.x;
        it.te = c.y;
        it.o = o;
        it.d = d;
        var a = vdbAccessorInit(g);
        var h: VdbHdda;
        vdbHddaInit(&h, o, c.x, d, c.y, max(vdbGetDim(&a, vec3i(floor(o + d * c.x))), 8u));
        it.h = h;
        it.acc = a;
    }
    return it;
}`,
      `fn vdbMajorantNext(it: ptr<function, VdbMajorantIter>) -> vec3f {
    var h = (*it).h;
    var a = (*it).acc;
    var s = vec3f((*it).t, (*it).t, 0.0);
    for (var i = 0u; i < 16u; i++) {
        let t0 = (*it).t;
        if !(t0 < (*it).te) {
            break;
        }
        var u = 0u;
        loop {
            let w = max(vdbGetDim(&a, h.voxel), 8u);
            if w == u32(h.dim) || u >= 2u {
                break;
            }
            vdbHddaUpdate(&h, (*it).o, (*it).d, w);
            u++;
        }
        var t1 = min(min(min(h.next.x, h.next.y), min(h.next.z, h.tmax)), (*it).te);
        let v = h.voxel;
        let dim = u32(h.dim);
        if !vdbHddaStep(&h) {
            t1 = (*it).te;
        }
        if t1 > t0 {
            (*it).t = t1;
            s = vec3f(t0, t1, vdb_maj_bound(&a, v, dim));
            break;
        }
    }
    (*it).h = h;
    (*it).acc = a;${o.minima ? "\n    (*it).lo = 0.0;" : ""}
    return s;
}`,
    );
  } else {
    const brick = o.majorant === "brick";
    const fields = brick
      ? `
    mode: u32,
    cell: vec3i,
    n: vec3i,
    sgn: vec3i,
    tn: vec3f,
    dt: vec3f,
    base: u32,${o.minima ? "\n    mn: u32," : ""}`
      : "";
    parts.push(`struct VdbMajorantIter {
    t: f32,
    te: f32,
    vmax: f32,${fields}${lo}
}`);
    const brickInit = brick
      ? `
    let f = vdb_rec(g, VDB_REC_FLAGS);
    let mb = vdb_rec(g, VDB_REC_MAJORANT_OFFSET) >> 2u;
    if (f & VDB_REC_FLAG_HAS_BRICK_MAJORANT) != 0u && vdb_load(mb + VDB_MAJ_MAGIC) == VDB_MAJORANT_MAGIC {
        let s = 1.0 / f32(1u << ((f >> VDB_REC_BRICK_LOG2_SHIFT) & VDB_REC_BYTE_FIELD_MASK));
        let n = bitcast<vec3i>(vec3u(vdb_rec(g, VDB_REC_MAJORANT_DIMS), vdb_rec(g, VDB_REC_MAJORANT_DIMS + 1u), vdb_rec(g, VDB_REC_MAJORANT_DIMS + 2u)));
        let u = (o - vec3f(bitcast<vec3i>(vec3u(vdb_load(mb + VDB_MAJ_ORIGIN), vdb_load(mb + VDB_MAJ_ORIGIN + 1u), vdb_load(mb + VDB_MAJ_ORIGIN + 2u))))) * s;
        let du = d * s;
        let c = vdb_maj_clip(vec3f(0.0), vec3f(n), u, du, tMin, tMax);
        if c.x < c.y {
            let par = abs(du) < vec3f(1e-20);
            let q = select(du, vec3f(1.0), par);
            it.mode = 2u;
            it.t = c.x;
            it.te = c.y;
            it.n = n;
            it.cell = clamp(vec3i(floor(u + du * c.x)), vec3i(0), n - vec3i(1));
            it.sgn = select(vec3i(-1), vec3i(1), du > vec3f(0.0));
            it.tn = select((vec3f(it.cell + max(it.sgn, vec3i(0))) - u) / q, vec3f(1e30), par);
            it.dt = select(abs(1.0 / q), vec3f(1e30), par);
            it.base = mb + VDB_MAJORANT_HEADER_WORDS;${
              o.minima
                ? `
            if (vdb_load(mb + VDB_MAJ_FLAGS) & VDB_MAJ_FLAG_MINIMA) != 0u {
                it.mn = it.base + u32(n.x * n.y * n.z);
            }`
                : ""
            }
        }
        return it;
    }`
      : "";
    parts.push(`fn vdbMajorantInit(g: u32, o: vec3f, d: vec3f, tMin: f32, tMax: f32) -> VdbMajorantIter {
    var it: VdbMajorantIter;
    it.t = tMin;
    it.te = tMin;
    if !vdb_live(g) {
        return it;
    }
    it.vmax = ${vmax};${brickInit}
    let c = ${support};${brick ? "\n    it.mode = 1u;" : ""}
    it.t = c.x;
    it.te = c.y;
    return it;
}`);
    const global = `(*it).t = (*it).te;${o.minima ? "\n        (*it).lo = 0.0;" : ""}
        return vec3f(t0, (*it).te, (*it).vmax);`;
    parts.push(
      brick
        ? `fn vdbMajorantNext(it: ptr<function, VdbMajorantIter>) -> vec3f {
    for (var i = 0u; i < 4u; i++) {
        let t0 = (*it).t;
        if !(t0 < (*it).te) {
            break;
        }
        if (*it).mode == 1u {
            ${global.replace(/\n {8}/g, "\n            ")}
        }
        let tn = (*it).tn;
        var k = 0;
        if tn.y < tn[k] {
            k = 1;
        }
        if tn.z < tn[k] {
            k = 2;
        }
        let t1 = min(tn[k], (*it).te);
        let c = (*it).cell;
        let n = (*it).n;
        let w = u32(c.x + n.x * (c.y + n.y * c.z));
        (*it).cell[k] = c[k] + (*it).sgn[k];
        (*it).tn[k] = tn[k] + (*it).dt[k];
        if (*it).cell[k] < 0 || (*it).cell[k] >= n[k] {
            (*it).te = t1;
        }
        if t1 > t0 {
            (*it).t = t1;${
              o.minima
                ? `
            (*it).lo = 0.0;
            if (*it).mn != 0u {
                (*it).lo = bitcast<f32>(vdb_load((*it).mn + w));
            }`
                : ""
            }
            return vec3f(t0, t1, bitcast<f32>(vdb_load((*it).base + w)));
        }
    }
    return vec3f((*it).t, (*it).t, 0.0);
}`
        : `fn vdbMajorantNext(it: ptr<function, VdbMajorantIter>) -> vec3f {
    let t0 = (*it).t;
    if !(t0 < (*it).te) {
        return vec3f(t0, t0, 0.0);
    }
    ${global.replace(/\n {8}/g, "\n    ")}
}`,
    );
  }
  if (o.minima) {
    parts.push(`fn vdbMajorantMinimum(it: ptr<function, VdbMajorantIter>) -> f32 {
    return (*it).lo;
}`);
  }
  return parts.join("\n");
}
