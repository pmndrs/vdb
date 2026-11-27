# Changelog

All notable changes to `@pmndrs/vdb` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/) (while 0.x, minor releases may break).

## [Unreleased]

### Breaking (0.2.0, D36: naming)

- `@pmndrs/vdb` (core): no name carries a `Gpu` suffix any more; the GPU is the default and `Cpu` marks the CPU twin of a
  GPU function (`@pmndrs/vdb/tools`). `vdbGridStatsGpu` → `vdbGridStats`, `vdbDecodeToAtlasGpu` → `vdbDecodeToAtlas`,
  `VDBBoxPyramidGpu(Options)` → `VDBBoxPyramid(Options)`, `vdbSparseSourceGpu` / `VDBSparseSourceGpu(Options)` /
  `VDBSparseVolumeGpu(Options)` / `VDBSparseLodGpu` → `vdbSparseSource` / `VDBSparseSource(Options)` /
  `VDBSparseVolume(Options)` / `VDBSparseLod`, `VDBFrameRendererGpu(Options)` / `VDBFrameVolumeGpu` →
  `VDBFrameRenderer(Options)` / `VDBFrameVolume`, `vdbCreateLevelSetPassGpu` / `VDBLevelSetPassGpu(Options)` →
  `vdbCreateLevelSetPass` / `VDBLevelSetPass(Options)`, `vdbCreatePointPassGpu` / `VDBPointPassGpu(Options)` →
  `vdbCreatePointPass` / `VDBPointPass(Options)`. A core name may equal a `./three` name (the import path tells them
  apart; `./three` wraps the core object): alias when you import both. `@pmndrs/vdb/three` no longer exports the type
  `VDBFrameRendererGpu` (import `VDBFrameRenderer` from `@pmndrs/vdb`).
- `VDBFrameLoader` → `VDBSequenceFrameLoader` (core and `./three`).
- `@pmndrs/vdb/tools`: the CPU samplers and readers take the `Cpu` suffix like `vdbReadValueCpu`: `vdbSampleTrilinear` /
  `vdbSampleNearest` / `vdbSampleTrilinearVec3` → `vdbSampleTrilinearCpu` / `vdbSampleNearestCpu` /
  `vdbSampleTrilinearVec3Cpu`, `vdbReadValueF32` / `vdbReadValueVec3` → `vdbReadValueF32Cpu` / `vdbReadValueVec3Cpu`.
- `wvdbPrefixLength`, `nvdbResponseLength` and `vdbMakeFpNLeafCodec` are no longer exported from `@pmndrs/vdb/tools`
  (they are in `tools/internal`).

### Breaking (0.2.0, K6.8: `./three` is a thin adapter over the core frame renderer)

- `@pmndrs/vdb/three`: `VDBFrameRenderer` draws every frame with the core frame renderer (compute kernels on three's
  device; D34). The TSL pass materials, their shader chunks and the internal `backend` switch are deleted.
  - `settings` is the core's `VDBFrameSettings` (his names and defaults, plus `occlusion`,
    `multiScaleLevels`, `ambientOcclusion`, `ambientOcclusionPairs`, `ambientOcclusionSteps`; the multi-scale A/B
    settings `multiScaleEdge` and `multiScaleSigma` stay internal).
    `VDBFrameRendererSettings` is removed; `VDBFramePhase` → `VDBFramePhaseName`, `VDBFrameMultipleScattering` →
    `VDBFrameMultipleScatteringMode` (the core's declarations, re-exported).
  - `neuralNetwork` → `mlp` (EXPERIMENTAL: the generic small-MLP data, `VDBFrameMlp`; default our trained weights; null
    runs 'neural' / 'mlp' as 1D diffusion).
  - The default network of `'neural'` / `'mlp'` is **our own**, trained on procedural media only (K6.N; D36): scenes that
    select `'neural'` (the default mode) change a little. His weights are no longer shipped.
  - `renderFroxels()` / `holdFroxels` → `froxels` (false keeps the last froxels: the core's `update(…, { froxels })`).
  - `passHook` is `{ timestampWrites?, begin?, end? }`: timestamp writes bracket the kernels of each core update
    (`VDBFramePass` is removed).
  - `redrawStats` is the core's `VDBFrameRedrawStats` (with `volumeShadows`); `environment` / `environmentLights` (the
    TSL readback) are removed: the core extracts environment lights on the GPU.
  - New `VDBFrameRenderer.core`: the `VDBFrameRenderer` (its `outputs`, after the first render).
  - `VDBFrameVolume.emissionColor` takes `VDBEmissionColor` data (`'white'`, `{ spectrum: 'planck' | 'soot', exponent }`,
    `{ lut }`, `{ wgsl }`); `vdbFrameBlackbodyEmission(s, e)` and `VDBFrameBlackbodyHook` are removed (use
    `{ spectrum: s, exponent: e }`). A TSL hook throws `VDB_BAD_OPTION`.
  - `vdbFrameLights` / `vdbFrameAllLights` → the core's `VDB_FRAME_LIGHT_BITS` (`ambient`, `point`, `directional`, `spot`,
    `hemisphere`, `environment`) / `VDB_FRAME_ALL_LIGHTS`, re-exported.
  - `VDBFrameVolume` drops `sourceUniforms`, `volumes` and the pass-material fields; its box is `boxMatrix`.
    `vdbVolumeUniforms` / `vdbVolumeNodes` are deleted (K6.11, below).
  - `ambientOcclusion` is **on by default** (4 direction pairs; D35): ambient, hemisphere, light-probe and environment
    light is attenuated by the medium (about 35 to 148 % too bright in single scattering without it, against the K6.R
    reference). `ambientOcclusion: false` gives 0.1's unshadowed light exactly; scenes without such lights do not change.
  - `froxelMix` is deleted: overlapping volumes in a froxel always combine order-independently (the former
    `'mixture'`); his draw-order `'over'` blend is gone (D35).
  - `vdbVolumeUniforms`, `vdbVolumeNodes` and their types (`VDBVolumeNodes`, `VDBVolumeNodesOptions`, `VDBVolumeUniforms`,
    `VDBVolumeUniformParameters`, `VDBThreeSourceFactory`) are deleted with the sources' TSL half, with no deprecation
    cycle (D35): `vdbDenseTextures` / `vdbSparseSource` are handles with no GPU object. Gone from them: the dense
    source's `textures` (`VDBDenseChannelTextures`), the sparse source's `pyramid`, `pyramidBuilds`, `refreshPyramid()`,
    `nodeAverageFn()`, `skipEmptyFn()`, `createUniforms()`, `createNodes()`, `VDBSparsePyramid`,
    `VDBSparseVolumeUniforms` and the `pyramidBuild` option; a sparse source's `gpuBytes` is `{ pool }`. A custom TSL
    material uses `VDBGridNode`; custom WGSL or compute the core sources' `wgsl()`.
- `VDBSequence` moves to the core (`@pmndrs/vdb`; engine-free); `@pmndrs/vdb/three` re-exports it.

### Added

- `@pmndrs/vdb/tools` (K7.C1): `writeNvdb(images, { codec: 'blosc' })` writes NanoVDB's `Codec::BLOSC`: LZ4 Blosc
  frames (byte shuffle, 1 MiB blocks of four 256 KiB streams, a frame per 1 GiB), byte-identical to native NanoVDB v14 for
  the same image, so native tools and NVIDIA Warp read them. A pure-TypeScript LZ4 block compressor (a port of LZ4 1.9.4,
  BSD-2-Clause) backs it; no new dependency. `vdbBakeCoarseFirst({ codec: 'blosc' })` too.
- `@pmndrs/vdb/tools` (K7.H1): Half grids are sources: `vdbQuantize(half, mode)` for every mode, a new `'float'` mode
  (widens a Half grid exactly; Half → Float → Half is the identity), `vdbCollapseGrid(half)` (stays Half),
  `transformLeafValues(half, fn)` (results rounded to the nearest binary16) and core `vdbApplyValueTransform` on a Half
  pool grid. They threw `VDB_UNSUPPORTED_GRID_TYPE` before.

- `@pmndrs/vdb/tools`: `parseVdb(buffer, { select })` lists a `.vdb`'s grids without decoding them and decodes only
  the ones you pick. `VdbFile.descriptors` (`VdbGridDescriptor`: name, grid type, on-disk value type, class, compression
  flags, transform, index bounding box and an `unsupported` reason) covers every grid; with `select`, a grid the parser
  cannot read is reported there instead of thrown, and `select: () => false` is a listing pass that decompresses
  nothing. Without `select`, `parseVdb` behaves as in 0.1.
- Core (issue #2): `VDBVolumeSourceWgslOptions.stochastic` asks a volume source for
  `vdbSrcDensityStochastic(v, uvw, lod, xi)`, the level-0 density from ONE voxel picked with probability equal to its
  trilinear weight (stochastic texture filtering; its mean over uniform `xi` is `vdbSrcDensity`). A sparse source
  reads one tree value instead of eight; a dense source returns its trilinear sample. Off by default (module bytes
  unchanged).
- Core (0.2.0, K6): the frame renderer `vdbCreateFrameRenderer` (`VDBFrameRenderer`: `addVolume`, `update`, `render`,
  `prepare`, `outputs`, `mlp`) with its input and output types; `vdbSparseSource` (one source per pool);
  `vdbFrameIntegrateWgsl` / `vdbFrameIntegrateEntries` (the composite in a host pass, `{ spectral }`),
  `vdbVolumeShadowWgsl` (`vdbVolumeShadow`, `vdbVolumeShadowCube`, `vdbVolumeShadowRgb`, `vdbVolumeShadowCubeRgb`;
  `VDB_VOLUME_SHADOW_VIEW`), `vdbEmissionProbeWgsl` (`VDB_EMISSION_GRID`); the level-set and point passes
  (`vdbCreateLevelSetPass`, `vdbLevelSetSurfaceWgsl`, `vdbCreatePointPass`, `vdbPointSpriteWgsl`);
  `VDBFrameUpdateOptions.timestampWrites`.
  - The defaults and limits: `vdbFrameDefaultSettings()` (the settings object, his defaults), `vdbFrameDefaultVolumeParams()`
    (a volume's parameters), `VDB_VOLUME_SHADOW_MAX_CASCADES`, `VDB_FRAME_LIGHT_BITS` / `VDB_FRAME_ALL_LIGHTS`.
  - The renderer's input types: `VDBFrameInput`, `VDBFrameCamera`, `VDBFrameLight`, `VDBFrameEnvironment`,
    `VDBFrameOccluder`, `VDBFrameTargets`, `VDBFrameSettings`, `VDBFrameVolumeParams`, `VDBFrameEmissionLightSettings`,
    `VDBEmissionColor`, `VDBFrameMlp` / `VDBFrameMlpLayer`, `VDBFramePhaseName`, `VDBFrameMultipleScatteringMode`,
    `VDBFrameWrap`, `VDBFrameMat4`, `VDBFrameVec3`; its outputs `VDBFrameOutputs`, `VDBFrameEmissionProbes`,
    `VDBFrameUpdate`, `VDBFrameRedrawStats`, `VDBFrameTimestamps`; `VDBFrameVolume`, `VDBFrameRendererOptions`.
  - `multipleScattering: 'neural'` (the default, our trained network) and `'mlp'` run the EXPERIMENTAL generic small MLP;
    its data shape (`VDBFrameMlp`) may change before 1.0.

- NanoVDB **Half** grids (`GridType::Half` = 9, raw IEEE binary16 values; implemented in NanoVDB 13.2, OpenVDB PR 2335), end
  to end (K7.H). Core: grid type `'half'` (`VDBGridTypeName`, `VDB_GRID_TYPE_IDS.half`, in `VDB_ALL_SCALAR_TYPES`); the pool
  holds Half images (binary16 background and root min/max decoded); `vdbModuleWgsl({ types: ['half'] })` decodes them
  with `unpack2x16float` at every level and reads binary16 min/max stats. Modules without `'half'` are byte-identical to
  0.1.0's. Tools: `vdbQuantize(image, 'half')`, `VDB_HALF_LEAF_CODEC`, `vdbFloatToHalf` / `vdbHalfToFloat` /
  `vdbRoundToHalf` (NanoVDB `math/HalfLite.h` rounding), `NvdbFile` loads Half grids (`NVDB_SUPPORTED_GRID_TYPES`), and
  the CPU readers and samplers decode them. Three: `VDBGridNode` and `vdbSparseSource` take `types: ['half']`. Values and
  min/max match native NanoVDB v14.0.0 (`createNanoGrid<…, nanovdb::Half>`) bit for bit on the fixtures
  (`pnpm fixtures:bake-half`).
- Tools: `NvdbFile` and `NvdbStreamReader` / `readNvdbStream` read `.nvdb` files with the BLOSC codec: NanoVDB's
  LZ4 frames (`nanovdb_convert -b/--blosc`, now built by default in OpenVDB v14) in any chunking, and NVIDIA Warp's
  BloscLZ frames (which NanoVDB 14 reads too). Hostile-input checks run before allocating. `writeNvdb` still writes
  NONE and ZIP.
- Tools: the Blosc decoder reads BloscLZ frames (also inside `.vdb` files).
- Tools: `parseVdb` reads OpenVDB 13+ `HalfGrid`s (`Tree_half_5_4_3`) under every codec. Values widen to f32
  exactly; the grid reports `valueType: 'float'` (its `gridType` stays `Tree_half_5_4_3`), so `buildFromVdb` makes a
  NanoVDB Float grid from it.

### Changed

- Tools (K7.C1 / K7.H1): built and re-encoded grids now equal native NanoVDB byte for byte (GridData checksum and version
  aside) on every native fixture: the root table and upper nodes are in native's signed-origin order again (0.1's unsigned
  key order put grids with negative coordinates out of breadth-first order, which NanoVDB v14's `nanovdb_validate`
  rejects), and node average/standard deviation use NanoVDB GridStats' f64 Welford update (they differed by up to 4 f32 ulp;
  a node holding ±∞ now gets NaN, as native). A node whose values are all +∞ stores the type's largest finite value as
  its minimum, as native. Image bytes of such grids change; values do not.

- Tools: an `.nvdb` from another NanoVDB major version (not 32.x) throws `VDBUnsupportedError` (`VDB_UNSUPPORTED`)
  instead of `NvdbFormatError`. A `.vdb` newer than file version 225 still throws `VDBUnsupportedError`, now saying
  it was written by a newer OpenVDB.
- Tools: point data grids get OpenVDB v14's read-time checks: an attribute array of size 0, a constant stride whose
  `size × stride` passes 2^32, a total size below the point count, a zero-byte paged array and a page header of
  i32 minimum are `VdbFormatError`s.
- Tools: a raw `NanoVDB1` grid buffer whose `mGridSize` is below 736 bytes or unrepresentable is an
  `NvdbFormatError` (a 0 size used to loop over empty grids).
- Tools: for `.nvdb` files whose FileMetaData leaves `indexBBox` and `voxelCount` zero (Warp writes them so),
  `NvdbGridMetadata` takes them from the grid image.
- Three (K6.8a, K6.11): `vdbSparseSource` / `vdbDenseTextures` are handles with no GPU object; `VDBFrameVolume` reads them
  into the core's own sources (no second texel upload, no second pyramid; the TSL half is deleted, see Breaking).
- Three (K6.8a): the volume shadow receivers on meshes call the core's `vdbVolumeShadowWgsl` (one implementation),
  binding the map's `view` buffer through a storage stand-in; `VDBVolumeShadowMap.view` gains `STORAGE` usage.
- Three (K6.8b): `VDBFrameEmissionLight` lights meshes with the core's `vdbEmissionProbeWgsl` (one implementation; frames
  unchanged), binding the probe `grid` buffer through a storage stand-in; `VDBFrameEmissionProbes.grid` gains `STORAGE`
  usage. The light's `castShadow` is always `true` (it casts no shadow): three r187 rebuilds the receivers of a
  `castShadow` light when its node changes, which the light needs when the core makes a new probe stack.

### Fixed

- Three (K6.8b): changing a `VDBFrameEmissionLight`'s `resolution` mid-run left plain lit materials (`MeshLambertMaterial`,
  `MeshStandardMaterial`) bound to the retired, destroyed probe stack ("Destroyed texture … used in a submit", and a wrong
  frame after); the same class as K6.10 finding 8. Receivers now bind placeholders while the core has no probes and
  rebuild when it makes a new stack or grid.

- Three (K6.8a): a shadow light whose core map went away (hidden for a few renders, or out of the camera's layers) and
  came back left its mesh receivers bound to the retired, destroyed map texture ("Destroyed texture … used in a
  submit", and a wrong shadow; K6.10 finding 8). Receivers now bind placeholders while a light has no map, and rebuild
  when the core makes a new map. Point-light receivers no longer log `CubeTextureNode: Mapping not supported`.

- Core frame renderer (K6.10a): `settings.shadowResolution` whose transmittance stack does not tile in
  `maxTextureDimension3D` (above 1024 with one cascade, 1024 with 3–4) clamps with a one-time warning; it threw a
  `RangeError` on every update. Volume shadow maps (`castShadowMapSize`, `VDBFrameLight.shadow.size`) clamp to
  `maxTextureDimension2D` and to a texture whose zero-fill staging fits `maxBufferSize` (a point light's 6 faces count),
  warned once; a 4096 point map asked for 805 MB.
- Core (K6.10a): replaced textures and buffers (the frame renderer, the level-set and point passes, `vdbSparseSource`,
  `vdbDenseSource.replace` with a new size) are destroyed in a later task, so several updates recorded into one encoder
  before its submit no longer submit a destroyed texture. `vdbDenseSource.replace` destroyed the old textures at once.
- Tools: the unknown-codec error names the real `nanovdb_convert` flags (`-z/--zip`, `-b/--blosc`); it suggested
  `--codec=zip`, which doesn't exist.
- Three: volume shadows of spot and point lights on meshes (`frameRenderer.settings.castShadows`) read each map with
  the light state it was drawn with, not three's current shadow coordinates, so a map that waits for
  `castShadowUpdatesPerFrame` no longer slides off its shadow when the light moves (mjurczyk/openvdb 62a6e3e). The
  update budget now counts only shadow lights in the scene being rendered; a light of another scene, or one removed
  mid-frame, no longer delays the rendered scene's maps (c13f2d5) (K7.1).

### Documentation

- JSDoc on every public export of `core`, `tools` and `three`: a summary line on each export, a doc on each option,
  prop and member, and `@param` tags on functions, methods and constructors. No signature changed.
- Installation: how each bundler loads `@pmndrs/vdb/pnanovdb.wgsl`. New page on level sets (shadow-map cost) and points.
- Docs link the live examples, and the examples link back. `THIRD_PARTY_NOTICES.md` now credits three.js and `@webgpu/types`.

## [0.1.0] - 2026-11-24

First public release (0.1.0). Our earlier v1 unscoped packages were never published (the npm `webvdb` package is mjurczyk's, not ours);
`@pmndrs/vdb` is the first published name. MIT licensed. Peer: `three` r187 (optional, only for `@pmndrs/vdb/three`).
The renderer, `.wvdb` format and level-set and point code derive from
[mjurczyk/openvdb](https://github.com/mjurczyk/openvdb) (MIT); the WGSL traversal derives from
emcfarlane/webgpu-nanovdb (Apache-2.0). See `THIRD_PARTY_NOTICES.md`.

### Added

One package with three entry points plus `@pmndrs/vdb/pnanovdb.wgsl`: `@pmndrs/vdb` (engine-free WebGPU core),
`@pmndrs/vdb/tools` (pure-TS CPU tools, no GPU, no `three`) and `@pmndrs/vdb/three` (three.js adapter).

**Core (`@pmndrs/vdb`)**

- `VDBGridPool`: one storage buffer holding a grid table and any number of NanoVDB grid images (add, replace,
  remove, setTransform, grow, version and revision counters, host-buffer mode, dispose) with a versioned GPU layout.
- Generated WGSL modules (`vdbModuleWgsl`, `vdbModuleEntries`, `vdbCheckModule`) per profile (`read`, `sample`,
  `march`) for Float, Fp4, Fp8, Fp16 and FpN grids, every symbol prefixed `vdb`, slot and buffer binding modes, with
  a size budget guarded in CI. Opt-in helpers: `fastTrilinear`, `levelSet`, `cellBoundTrilinear`, `hddaCellEnd`,
  `rayClipSupport`, `nodeAverage`.
- GPU compute: `vdbGridStatsGpu`, `vdbDecodeToAtlasGpu`, the box pyramid `vdbCreateBoxPyramid`, and
  `vdbApplyValueTransform`; `vdbDenseSource` for dense volume sources.
- `VDBError` with stable `code`s.

**Tools (`@pmndrs/vdb/tools`)**

- `parseVdb`: OpenVDB `.vdb` files including Blosc and zlib compression, active tiles, vector grids, point grids
  and half-float storage.
- Build and quantize: `vdbBuildFromDense`, `buildFromVdb`, `vdbBuildFromLeavesDetailed`, `vdbQuantize` to Fp4, Fp8,
  Fp16 and FpN; `vdbResampleGrid`, `vdbCollapseGrid`, `vdbTransform`, `vdbInspect`. `vdbBuildFromDense` and
  `vdbBuildFromLeavesDetailed` take `voxelSize` as a number or `[x, y, z]` (an anisotropic, diagonal Map; G15a).
- `.nvdb`: `NvdbFile` (NONE and ZIP codecs), `NvdbStreamReader` / `readNvdbStream` (emits each grid as it
  completes), `writeNvdb` (with the ZIP codec), and coarse-first baking with `vdbBakeCoarseFirst`.
- `.wvdb` (mjurczyk/openvdb spec): `readWvdb`, `writeWvdb`, `WvdbStreamReader`, `bakeWvdb`, and converters
  (`vdbToWvdb`, `nvdbToWvdb`, `wvdbDenseToNvdb`).
- CPU reads, samplers, stats and atlas: `vdbReadValueCpu`, `vdbReadValueF32`, `vdbSampleTrilinear`,
  `vdbSampleNearest`, `vdbGridStatsCpu`, `vdbDecodeToAtlasCpu`.
- Medium presets (`vdbFrameMediumPresets`, `vdbFrameMediumFromMeasurements`) and blackbody colours from an analytic
  CIE 1931 fit (`vdbFrameBlackbodyColor`, `vdbCie1931Fit`).
- One error hierarchy: `VDBToolsError` with `VdbFormatError`, `NvdbFormatError`, `WvdbFormatError` and
  `VDBUnsupportedError`.

**three.js adapter (`@pmndrs/vdb/three`)**

- `getDevice(renderer)` and `getGPUTexture` so tools share one GPU device; `VDBGridNode` with the `vdbSample` and
  `vdbReadValue` TSL helpers.
- `VDBVolumeMaterial` (with `vdbProxyGeometry`), `VDBSequence` (frame playback on `pool.replace`), and compute
  wrappers (`vdbGridStats`, `vdbDecodeToAtlas`).
- Coarse-first streaming: `vdbStreamCoarseFirst` and `vdbCoarseFirstApply`, with `bind` to update material, mesh and
  node after each swap.
- Coarse-first into the froxel renderer without a rebuild (G16a): `vdbSparseSource(..., { followImage: true })`
  re-reads its boxes and rebuilds its pyramid after a `pool.replace` changes the grid's box (boxes, sizes and LOD levels
  are uniforms then), and a `VDBFrameVolume` whose box is the source's follows it. `VDBCoarseFirstEvent.types` lists
  the family's grid types for `new VDBGridNode(renderer, pool, { types })`, so one node, source and volume take the
  preview and the full grid with no shader or pipeline built at the swap.
- The froxel renderer, ported from mjurczyk/openvdb and re-based on the sparse pool: `VDBFrameVolume` and
  `VDBFrameRenderer` (`vdbFrameRenderer`) with `'volume'` and `'frame'` lighting, temporal accumulation, phase
  functions including Lorenz-Mie, multiple scattering (two-sided, octaves, neural), spectral media and measured
  presets (`vdbFrameMedium`), mesh shadows, and emission light (`VDBFrameEmissionLight`, blackbody).
- Volume sources with LOD: `vdbDenseSource` / `vdbDenseTextures` (dense) and `vdbSparseSource` (coarse pyramid,
  node-average LOD, optional HDDA empty-space skipping).
- `VDBLevelSetMaterial` (sphere-traced level-set surfaces lit by three's lights and shadows) and `VDBPoints`.
- Froxel renderer API (K5.9): one place for the phase (the renderer's `settings.phase*` is every volume's; a volume's
  `phase*` props are optional `'volume'`-lighting overrides, `volume.getPhase()`); `volume.setMedium(name | props |
  null)` applies a measured medium in one call (albedo, extinction spectrum, the renderer's phase, spectral on);
  `densityMean` fixes the mean `normalizeDensity` divides by (one mean for a sequence); `volume.bbox` is the resolved
  box; the volume follows a `source.replace()` by itself (`syncSource`, per draw).
- `VDBSequence<TFrame>` plays any frame type; `start({ source })` swaps dense `.wvdb` frames into a
  `vdbDenseTextures` source. A dense `replace` reads a channel the frame lacks as its absent value (emission 0, mask
  and albedo 1) and ignores a channel the source lacks. A `seek` while stopped shows the frame once decoded.
- `VDBSequence` frame range and ping-pong: `range: [start, end]` (inclusive; default all frames) limits playback, and
  `loopMode: 'once' | 'repeat' | 'pingpong'` (default `'repeat'`) holds, wraps or turns around at the range end (each
  end frame shown once per turn). `setRange()` / `setLoopMode()` change them while playing; the prefetch ring follows
  the playback order. New types `VDBSequenceRange`, `VDBSequenceLoopMode`.
- `wvdbDenseToNvdb(...).sparseSourceOptions` (`@pmndrs/vdb/tools`): the index box, dense stats and box that line a
  sparse copy of a `.wvdb` level up with the dense source, spread into `vdbSparseSource`.

### Changed

- `vdbDecodeToAtlasGpu` / `vdbDecodeToAtlas` / `vdbDecodeToAtlasCpu` sample voxel centres: cell `i` reads index
  position `min + (i + 0.5) * (max - min + 1) / n - 0.5`, so at full resolution it is voxel `min + i` (v1 sampled
  half a voxel past, and the last cell read the background). Atlases span the box `[min - 0.5, max + 0.5]` instead
  of `[min, max + 1]`.
- `VDBSequence` has one loop knob: the `loop: boolean` option and getter are gone; `loopMode: 'once' | 'repeat' |
  'pingpong'` (default `'repeat'`, as `loop: true` was; `'once'` is the old `loop: false`, holding at the range end)
  is settable while playing with `setLoopMode()`. A ping-pong keeps its direction across `stop()` / `resume()`,
  `seek()` and `setRange()` (RV).

### Fixed

- `VDBSequence` ping-pong paused on the way back resumed forward, onto a frame its prefetch ring had evicted (a
  stall and a dropped frame); it now resumes on the way back (RV).
- `VDBSequence`: a frame the clock made due, still decoding at `stop()`, was shown when its decode landed; now only a
  frame asked for (`start`, `seek`, `setRange`, `setLoopMode`) lands while stopped (RV).
- `vdbSparseSource` with `followImage`: a new image whose grid type the `VDBGridNode` does not compile read 0 silently;
  it now throws `VDB_TYPE_NOT_COMPILED` naming `VDBCoarseFirstEvent.types` as the fix (as does making a source over
  such a grid) (RV).
- `vdbSparseSource` kept a stats listener (and, with `followImage`, a layout listener) for every `VDBFrameVolume` made
  over it, for the source's lifetime (RV).
- `vdbSparseSource` with `followImage` destroyed the old pyramid textures at once on a swap (three r187's
  `texture.dispose()` destroys the GPU texture), though a pass recorded in the same render could still bind them; they
  are now destroyed once the queue's submitted work is done (RV).
- A `VDBFrameVolume` over a dense source kept the first frame's box: a `VDBSequence` frame with another `.wvdb` box was
  stretched into it. The box now follows the frame's box unless the volume was given a `bbox`, as mjurczyk's WVDB
  sequences give each frame its own box (RV).
- `wvdbDenseToNvdb` gave the grid one voxel size (the x side's), so a `.wvdb` level with non-cubic voxels (any
  default bake: each side's texel count rounds on its own) read back shifted and stretched in world space. Each axis
  now gets its own voxel size; smoke.vdb's default 128 bake measures 42.3% with `vdbResampleError` (was 45.4%; a
  `dimensions: [128, 128, 128]` bake 37.4%, was 92.4%) (G15a).
- `VDBFrameVolume.densityScale` had no effect on a sparse source: the density uniform's per-render stats refresh
  overwrote the per-draw value. A new mean from a replaced image is now applied once, when its readback lands (K5.9a).
- Mesh shadows with `multipleScattering: 'neural'` and temporal/jitter in `'frame'` lighting failed WGSL validation
  in `VDBFrameTransmittanceMaterial` ("unresolved value 'nodeUniform0'"): the network's layout function is now built
  per material build (K5.9b).

### Notes

- `vdbReadValueCpu` returns the exact f64 decode; `vdbReadValueF32` matches the GPU evaluation order. The two can
  differ by 1 ulp on Fp grids.
- Public names follow D32: library API is `VDB*` / `VDB_*` / `vdb*`; anything bound to one file format keeps its
  prefix (`Vdb*` for `.vdb`, `Nvdb*` for `.nvdb`, `Wvdb*` for `.wvdb`). Each entry point exports only the
  user-facing API.
- The `majorant` WGSL profile is not in 0.1 (`VDB_BAD_OPTION`).
- `@webgpu/types` is a dependency so the shipped declarations type-check in your project.
- The vendored `pnanovdb.wgsl` keeps its Apache-2.0 licence and NOTICE, shipped in the package, as does
  `THIRD_PARTY_NOTICES.md`.
