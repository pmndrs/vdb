/**
 * The lean reader (`reader: 'lean'`, P1.5 = K1.8; docs/dev/design/K1-core.md §8): a compact root ->
 * upper -> lower -> leaf descent with literal NanoVDB offsets, a one-leaf cache and no vendored code,
 * for kernels near a compiler's size cap (pmndrs/pathtracer's media kernels sit at 46.9-50.1 KB against
 * a ~50 KB DXC class). Same public API as the vendored reader (`VdbAccessor`, `vdbAccessorInit`,
 * `vdbReadValue`, `vdbIsActive`, `vdbSample`), same values: a voxel reads what `vdbReadValue` reads
 * (CPU twin `vdbReadValueF32Cpu`), and the sampler is the `outside: 'zero'` lerp order of the vendored
 * module's API layer.
 *
 * Derived from PNanoVDB.h (OpenVDB, Apache-2.0; see NOTICE): the node layout (offsets from
 * vendor/stride-tables.json, which every NanoVDB scalar grid type shares above the leaf), the root key
 * (`pnanovdb_coord_to_key`), the child/value table lookups and the leaf decoders (Float f32, Half binary16,
 * Fp4/Fp8/Fp16 fixed-width codes, FpN codes of the leaf's own width; `code * quantum + minimum`). What it
 * leaves out of the vendored accessor: the upper/lower cache levels and the per-level typed getters.
 *
 * Every name is `vdb_ln_*` (internal) or the shared public API; none maps to a vendored declaration, so
 * the generator's closure pulls in nothing from the vendored file for these functions.
 */
import type { VDBGridTypeName } from "../types.js";
import type { ApiOptions } from "./api.js";

/** NanoVDB byte offsets the lean reader bakes in (ABI 32; equal for Float, Half and every Fp type above the leaf). */
export const LEAN_OFFSETS = {
  /** `GridData` size, where the tree starts. */
  gridSize: 672,
  /** Tree: byte offset of the root, relative to the tree (low word of the i64). */
  treeRootOffset: 24,
  rootTableSize: 24,
  rootBackground: 28,
  /** First root tile, relative to the root (`root_size`). */
  rootTiles: 64,
  rootTileSize: 32,
  rootTileChild: 8,
  rootTileState: 16,
  rootTileValue: 20,
  upperValueMask: 32,
  upperChildMask: 4128,
  upperTable: 8256,
  lowerValueMask: 32,
  lowerChildMask: 544,
  lowerTable: 1088,
  /** Child/value table entry size (`table_stride`). */
  tableStride: 8,
  leafFlags: 12,
  leafValueMask: 16,
  leafMinimum: 80,
  leafQuantum: 84,
  leafTable: 96,
} as const;

const L = LEAN_OFFSETS;

/** Grid type constant names after the rename (the vendored `PNANOVDB_GRID_TYPE_*` consts). */
const TYPE_CONST: Readonly<Record<VDBGridTypeName, string>> = {
  float: "VDB_GRID_TYPE_FLOAT",
  fp4: "VDB_GRID_TYPE_FP4",
  fp8: "VDB_GRID_TYPE_FP8",
  fp16: "VDB_GRID_TYPE_FP16",
  fpn: "VDB_GRID_TYPE_FPN",
  half: "VDB_GRID_TYPE_HALF",
};

/** log2 of the code width of a fixed-width Fp type. */
const FP_LOG_BITS: Partial<Record<VDBGridTypeName, string>> = { fp4: "2u", fp8: "3u", fp16: "4u" };

/** The leaf decode of one type, as statements ending in a `return` (inputs `b`, `l`, `n`). */
function leafDecode(t: VDBGridTypeName): string {
  if (t === "float") return `return bitcast<f32>(vdb_ln_w(b, l + ${L.leafTable}u + 4u * n));`;
  if (t === "half") return `return unpack2x16float(vdb_ln_w(b, l + ${L.leafTable}u + 2u * n) >> ((n & 1u) << 4u)).x;`;
  const lb = FP_LOG_BITS[t] ?? `(vdb_ln_w(b, l + ${L.leafFlags}u) >> 29u)`;
  return `let lb = ${lb};
    let pw = 5u - lb;
    let c = (vdb_ln_w(b, l + ${L.leafTable}u + ((n >> pw) << 2u)) >> ((n & ((1u << pw) - 1u)) << lb)) & ((1u << (1u << lb)) - 1u);
    return f32(c) * bitcast<f32>(vdb_ln_w(b, l + ${L.leafQuantum}u)) + bitcast<f32>(vdb_ln_w(b, l + ${L.leafMinimum}u));`;
}

/** Statements dispatching on `t` over the compiled types; the last type is the fallthrough (`ok` rejected the rest). */
function dispatch(types: readonly VDBGridTypeName[], body: (t: VDBGridTypeName) => string, groupKey: (t: VDBGridTypeName) => string): string {
  const groups: { key: string; types: VDBGridTypeName[] }[] = [];
  for (const t of types) {
    const k = groupKey(t);
    const g = groups.find((x) => x.key === k);
    if (g) g.types.push(t);
    else groups.push({ key: k, types: [t] });
  }
  const lines: string[] = [];
  groups.forEach((g, i) => {
    const text = body(g.types[0]!);
    if (i === groups.length - 1) {
      lines.push(`    ${text}`);
    } else {
      lines.push(`    if ${g.types.map((x) => `t == ${TYPE_CONST[x]}`).join(" || ")} {`);
      lines.push(`        ${text.replace(/\n {4}/g, "\n        ")}`);
      lines.push("    }");
    }
  });
  return lines.join("\n");
}

/**
 * The lean accessor parts of the read layer: the struct, `vdbAccessorInit`, the descent, the leaf and
 * tile decoders, `vdbReadValue` and (with `isActive`) `vdbIsActive`. `common` are the record functions
 * every reader shares (`vdb_rec`, `vdbGridCount`, `vdb_live`, ...).
 */
export function leanReadLayer(o: ApiOptions, bbox: boolean, common: readonly string[]): string {
  const extra = bbox ? "\n    lo: vec3i,\n    hi: vec3i," : "";
  const extraInit = bbox ? "\n        a.lo = vdbIndexMin(g);\n        a.hi = vdbIndexMax(g);" : "";
  const half = o.types.includes("half");
  // A tile or background slot: f32, or binary16 at a 2-byte-aligned address for Half.
  const slot = (t: VDBGridTypeName) =>
    t === "half" ? "return unpack2x16float(vdb_ln_w(b, a) >> ((a & 2u) << 3u)).x;" : "return bitcast<f32>(vdb_ln_w(b, a));";
  const parts = [
    `struct VdbAccessor {
    b: u32,
    r: u32,
    gridType: u32,
    ok: bool,
    lk: vec3i,
    la: u32,${extra}
}`,
    ...common,
    `fn vdb_ln_w(b: u32, a: u32) -> u32 {
    return vdb_load(b + (a >> 2u));
}`,
    `fn vdb_ln_bit(b: u32, m: u32, n: u32) -> u32 {
    return (vdb_ln_w(b, m + 4u * (n >> 5u)) >> (n & 31u)) & 1u;
}`,
    `fn vdbAccessorInit(g: u32) -> VdbAccessor {
    var a: VdbAccessor;
    a.gridType = vdbGridType(g);
    a.ok = vdb_type_ok(a.gridType);
    if a.ok {
        a.b = vdb_rec(g, VDB_REC_IMAGE_OFFSET) >> 2u;
        a.r = ${L.gridSize}u + vdb_ln_w(a.b, ${L.gridSize + L.treeRootOffset}u);${extraInit}
    }
    return a;
}`,
    // (address, level 0 leaf / 1 lower tile / 2 upper tile / 3 root tile / 4 background, mask word address, bit).
    `fn vdb_ln_find(b: u32, r: u32, ijk: vec3i) -> vec4u {
    let u = bitcast<vec3u>(ijk);
    let kl = (u.z >> 12u) | ((u.y >> 12u) << 21u);
    let kh = (u.y >> 23u) | ((u.x >> 12u) << 10u);
    let n = vdb_ln_w(b, r + ${L.rootTableSize}u);
    for (var i = 0u; i < n; i++) {
        let t = r + ${L.rootTiles}u + i * ${L.rootTileSize}u;
        if vdb_ln_w(b, t) == kl && vdb_ln_w(b, t + 4u) == kh {
            let c = vdb_ln_w(b, t + ${L.rootTileChild}u);
            if c == 0u && vdb_ln_w(b, t + ${L.rootTileChild + 4}u) == 0u {
                return vec4u(t + ${L.rootTileValue}u, 3u, t + ${L.rootTileState}u, 0u);
            }
            let p = r + c;
            let un = (((u.x & 4095u) >> 7u) << 10u) | (((u.y & 4095u) >> 7u) << 5u) | ((u.z & 4095u) >> 7u);
            let e = p + ${L.upperTable}u + ${L.tableStride}u * un;
            if vdb_ln_bit(b, p + ${L.upperChildMask}u, un) == 0u {
                return vec4u(e, 2u, p + ${L.upperValueMask}u, un);
            }
            let q = p + vdb_ln_w(b, e);
            let ln = (((u.x & 127u) >> 3u) << 8u) | (((u.y & 127u) >> 3u) << 4u) | ((u.z & 127u) >> 3u);
            let f = q + ${L.lowerTable}u + ${L.tableStride}u * ln;
            if vdb_ln_bit(b, q + ${L.lowerChildMask}u, ln) == 0u {
                return vec4u(f, 1u, q + ${L.lowerValueMask}u, ln);
            }
            let l = q + vdb_ln_w(b, f);
            return vec4u(l, 0u, l + ${L.leafValueMask}u, ((u.x & 7u) << 6u) | ((u.y & 7u) << 3u) | (u.z & 7u));
        }
    }
    return vec4u(r + ${L.rootBackground}u, 4u, 0u, 0u);
}`,
    `fn vdb_ln_leaf(t: u32, b: u32, l: u32, n: u32) -> f32 {
${dispatch(o.types, leafDecode, (t) => (FP_LOG_BITS[t] !== undefined || t === "fpn" ? `fp:${t}` : t))}
}`,
    `fn vdb_ln_tile(t: u32, b: u32, a: u32) -> f32 {
${half ? dispatch(o.types, slot, (t) => (t === "half" ? "half" : "f32")) : `    ${slot("float")}`}
}`,
    `fn vdbReadValue(acc: ptr<function, VdbAccessor>, ijk: vec3i) -> f32 {
    if !(*acc).ok {
        return 0.0;
    }
    let n = bitcast<u32>(((ijk.x & 7) << 6u) | ((ijk.y & 7) << 3u) | (ijk.z & 7));
    if (*acc).la == 0u || any((ijk & vec3i(-8)) != (*acc).lk) {
        let f = vdb_ln_find((*acc).b, (*acc).r, ijk);
        if f.y != 0u {
            return vdb_ln_tile((*acc).gridType, (*acc).b, f.x);
        }
        (*acc).la = f.x;
        (*acc).lk = ijk & vec3i(-8);
    }
    return vdb_ln_leaf((*acc).gridType, (*acc).b, (*acc).la, n);
}`,
  ];
  if (o.isActive) {
    parts.push(`fn vdbIsActive(acc: ptr<function, VdbAccessor>, ijk: vec3i) -> bool {
    if !(*acc).ok {
        return false;
    }
    let f = vdb_ln_find((*acc).b, (*acc).r, ijk);
    return f.y < 4u && vdb_ln_bit((*acc).b, f.z, f.w) != 0u;
}`);
  }
  return parts.join("\n");
}

/**
 * The lean `vdbSample`: taps through `vdbReadValue` (so through the one-leaf cache), the zero-mode lerp
 * order of the vendored API layer; with `outside: 'zero'` a tap outside `[lo, hi]` reads 0.
 */
export function leanSampleFunctions(o: ApiOptions, bbox: boolean): string[] {
  const guard = bbox ? "!(*acc).ok || any(b < (*acc).lo - vec3i(1)) || any(b > (*acc).hi)" : "!(*acc).ok";
  if (o.sampler === "nearest") {
    return [
      `fn vdbSample(acc: ptr<function, VdbAccessor>, xyz: vec3f) -> f32 {
    let ijk = vec3i(floor(xyz + vec3f(0.5)));
    if !(*acc).ok${bbox ? " || any(ijk < (*acc).lo) || any(ijk > (*acc).hi)" : ""} {
        return 0.0;
    }
    return vdbReadValue(acc, ijk);
}`,
    ];
  }
  return [
    `fn vdb_ln_tap(acc: ptr<function, VdbAccessor>, ijk: vec3i) -> f32 {${
      bbox
        ? `
    if any(ijk < (*acc).lo) || any(ijk > (*acc).hi) {
        return 0.0;
    }`
        : ""
    }
    return vdbReadValue(acc, ijk);
}`,
    `fn vdbSample(acc: ptr<function, VdbAccessor>, xyz: vec3f) -> f32 {
    let b = vec3i(floor(xyz));
    if ${guard} {
        return 0.0;
    }
    let u = xyz - vec3f(b);
    let x00 = mix(vdb_ln_tap(acc, b), vdb_ln_tap(acc, b + vec3i(1, 0, 0)), u.x);
    let x10 = mix(vdb_ln_tap(acc, b + vec3i(0, 1, 0)), vdb_ln_tap(acc, b + vec3i(1, 1, 0)), u.x);
    let x01 = mix(vdb_ln_tap(acc, b + vec3i(0, 0, 1)), vdb_ln_tap(acc, b + vec3i(1, 0, 1)), u.x);
    let x11 = mix(vdb_ln_tap(acc, b + vec3i(0, 1, 1)), vdb_ln_tap(acc, b + vec3i(1, 1, 1)), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}`,
  ];
}
