/**
 * @pmndrs/vdb core internals (D32): NOT a package export.
 *
 * Everything public (`./index.ts`) plus what tests, harnesses and scripts need that a user doesn't:
 * the pool's binary layout (constants, header/record codecs, the layout WGSL), the module banner, the
 * pyramid kernel's WGSL and the vendored module's URL. Import it by path (`src/core/internal.ts`, or
 * `dist/core/internal.js` from a script after `pnpm build`); its names can change without notice.
 */

export * from "./index.js";

/** Resolvable URL of the vendored WGSL module (`@pmndrs/vdb/pnanovdb.wgsl`), for Node tests. */
export const pnanovdbWgslUrl = new URL("../../vendor/pnanovdb.wgsl", import.meta.url);

export {
  VDB_HEADER_WORD,
  VDB_MAJORANT_FLAG,
  VDB_MAJORANT_HEADER_BYTES,
  VDB_MAJORANT_MAGIC,
  VDB_MAJORANT_WORD,
  VDB_POOL_ALIGN,
  VDB_POOL_HEADER_BYTES,
  VDB_POOL_MAGIC,
  VDB_POOL_RECORD_BYTES,
  VDB_POOL_RECORD_WORDS,
  VDB_POOL_TABLE_OFFSET,
  VDB_RECORD_FLAG,
  VDB_RECORD_WORD,
  decodeHeader,
  decodeRecord,
  encodeMajorantRange,
  encodeRecord,
  vdbPoolLayoutWgsl,
} from "./pool/layout.js";
export type { VDBGridRecord, VDBPoolHeader } from "./pool/layout.js";
export { VDB_MODULE_BANNER } from "./wgsl/generate.js";
// K4.22: the dense source's GPU fill of a partial `.wvdb` LOD.
export { VDBDenseFiller, vdbDenseFillFits, vdbDenseFillLayout, vdbDenseFillWgsl } from "./source/dense-fill.js";
export type { VDBDenseFillLayout } from "./source/dense-fill.js";
export { vdbDenseSourceTextures } from "./source/dense.js";
export { boxPyramidWgsl } from "./compute/index.js";
export {
  VDB_FRAME_MULTI_SCALE_PARAMS_BYTES,
  VDB_FRAME_MULTI_SCALE_SAMPLER,
  multiScaleMipsWgsl,
  vdbCreateMultiScaleMips,
  vdbFrameMultiScaleParams,
  vdbFrameMultiScaleWgsl,
} from "./compute/multi-scale.js";
export type {
  VDBFrameMultiScaleParams,
  VDBFrameMultiScaleWgslOptions,
  VDBMultiScaleFormat,
  VDBMultiScaleMips,
  VDBMultiScaleMipsOptions,
  VDBMultiScaleRecordOptions,
  VDBMultiScaleResizeOptions,
} from "./compute/multi-scale.js";
// K6.7: the sparse source's record layout (the source itself is public since K6.8).
export {
  VDB_SPARSE_RECORD_BYTES,
  VDB_SPARSE_RECORD_FIELDS,
  SPARSE_FLAG,
  packSparseRecord,
  sparseBindings,
  sparseRecordOffset,
  sparseRecordStructWgsl,
  sparseSourceWgsl,
} from "./source/sparse-wgsl.js";
// K6.1: the core frame renderer's internals: kernels, specs, uniform layouts, twins' probes (the renderer is public since K6.8).
export * from "./frame/index.js";
// K6.6: level-set surfaces and point sprites (the passes and WGSL are public since K6.8; their layouts here).
export * from "./surface/index.js";
// K6.R: the reference renderer (brute-force path tracer over a volume source; internal only).
export * from "./reference/index.js";
