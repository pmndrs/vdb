# P1 (our side) handoff (2026-11-28)

P1.1–P1.6, the @pmndrs/vdb half of the pathtracer integration (PHASES P1, TASKS P1). P1.5 is K1.8 (the lean reader).
Inputs read 2026-11-28 from pmndrs/pathtracer main `06488fa` (read-only): `packages/pathtracer/src/volume/volume.ts`
(`brickMajorants` with minima, `ddaSegments`, `gridDensity`, `residualTransmittance`, `brickTrack`),
`bench/scenes/volumes/{vdb-grid,vdb-convert}.ts`, `docs/internal/design/volumes-plan.md`, its CLAUDE.md.

## What changed

**P1.1 point-sample contract.** No library change: the contract already holds. `packages/vdb/test/p1/contract.test.ts`
rebuilds the pathtracer's `vdb-smoke` dense grid from `fixtures/vdb-samples/smoke.vdb` (a port of its `convertVdbFile`,
`examples/src/harness/p1-smoke.ts`), checks it equals its generated `vdb-smoke.f32` bit for bit (when the sibling
checkout has it; `VDB_SMOKE_F32` overrides the path), transposes it (x fastest → `vdbBuildFromDense`'s x-major) and
compares `vdbSampleTrilinearCpu(…, { outside: 'zero' })` with its `gridDensity` at 10k index points (uniform over the
grid ± 2, across every face, voxel centres, half-way points): worst |Δ| = 0 (≤ 1e-6). Fail-before: half-voxel-shifted
centres and the untransposed array both differ by > 1e-2. P1.8's half on our side is in the same file: every voxel
centre of the built image is the f32 value bit for bit (414k voxels).

**P1.2 `vdbBrickMajorant`** (`tools/cpu/majorant.ts`, public in `@pmndrs/vdb/tools`; `VDBBrickMajorant` is now declared
in tools and re-exported by core, one declaration). A cell's bound is the max (and min) of every value a trilinear
sample in it can read: its `B + 1` taps per axis, from the tree's regions (`forEachTreeRegion`, new internal in
`tools/cpu/march.ts`: every leaf, every tile active or not, root-tile coverage for the background), or 0 for a tap
outside the bbox with `outside: 'zero'`. `align: 'support'` (origin `min − 1`, `ceil((max − min + 2) / B)` cells) or
`'pathtracer'` (origin −1, `floor((n + 8) / 8)` cells, its two conservative rules kept; DIVERGES V175).

**P1.3 `majorant` profile** (`core/wgsl/majorant.ts`, generator `profile: 'majorant'`, options `majorant`, `minima`):
`vdbMajorantInit(g, o, d, tMin, tMax)`, `vdbMajorantNext(&it) -> vec3f(t0, t1, bound)`, `vdbMajorantGlobal(g)`,
`vdbMajorantMinimum(&it)` with minima. `brick` is the pathtracer's DDA in cell space (zero-length corner segments
skipped, V177); a grid without bricks (or with a headerless 0.1 range) walks as `global`. `brick` / `global` are `sample`
+ the walk (no march layer, K1-core §3.2). Pool: `setMajorant` / `add({ majorant })` pack a 32-byte range header
(origin, minima flag, `"VDBM"`; `encodeMajorantRange`, constants in `core/pool/layout.ts`), layout version unchanged
(V176, owner question P1.14). `pool.info()` gains `majorantOrigin`, `majorantMinima`.

**P1.4 `nodeMax`** (`majorant: 'nodeMax'`): the march layer + an HDDA walk over the trilinear support box, leaf-sized
cells widening over tiles; bound = the 2x2x2 `vdbNodeMax` support of the cell (as `vdbCellBoundTrilinear`), ≥ 0 with
`outside: 'zero'` (V178). CPU twin `majorantWalk` (tools internal) also counts accessor lookups per segment.

**P1.5 lean reader** (`core/wgsl/lean.ts`, `reader: 'lean'`, V179, THIRD_PARTY row): one descent with literal offsets
(a test pins them to `vendor/stride-tables.json`), a one-leaf cache, active state from the same descent, Float / Half /
Fp4 / Fp8 / Fp16 / FpN decoders; no vendored function. Same public API. Profiles `read`, `sample`, `majorant`
brick/global; refuses `march`, `nodeMax` and every helper but `rayClipSupport`.

**P1.6 several grids at one binding:** no library change (D10 already). GPU test (below) and docs: `grid-pool.mdx`
"Several grids at one binding" (snippet `multi-grid`), `wgsl-module.mdx` "Majorants" (snippet `majorant-host`) and
"The lean reader", budgets table.

Also: `VDBGridNode` (./three) still refuses `profile: 'majorant'` (new message: raw WGSL hosts only); variant matrix
`scripts/lib/wgsl-variants.mjs` gains 7 majorant × 16 and 5 lean × 16 variants (appended; old ids and bytes unchanged:
the snapshot diff only adds 192 entries); `size-budgets.json` rows `majorant`, `majorant.nodeMax`, `read.lean`,
`sample.lean`, `majorant.lean`.

## Evidence (no timings: the machine was loaded, load average 28–64 at the start)

- `pnpm typecheck` clean; `pnpm docs:check` (52 blocks match, snippets type-check); full vitest 114 files, 2658 passed,
  23 skipped (`--testTimeout=120000`); `pnpm wgsl:size` matches the regenerated snapshot (464 variants).
- Browser compile (headless shell 1228, SwiftShader, `scripts/check-wgsl-variants.mjs --no-probe --only majorant|lean`):
  112/112 majorant and 112/112 lean variants, zero errors.
- CPU property (K1-core §5.4, `test/p1/majorant.test.ts`): nine fixtures × {brick + minima, global, nodeMax} ×
  {background, zero}, 1000 rays × 256 jittered samples each: 0 samples over the bound, 0 under the minima, segments
  contiguous, field 0 beyond the walk (zero mode). Fail-before: a brick grid shifted one voxel and the cell's own
  nodeMax both violate. 'pathtracer' align = its `brickMajorants` (maxima and minima) on `vdb-smoke` and on a random
  19×9×24 block touching every face; our brick walk = its `ddaSegments` (zero-length dropped) on 500 rays, > 5000
  segments, t within 1e-5.
- GPU parity (`e2e/p1-parity.spec.ts`, page `examples/src/harness/p1-parity.ts`, SwiftShader, port 5347): one pool (14
  grids incl. a dead id, grown from 4 KiB, two instance transforms), 9 modules. Every row green:

| Row | Checked | Worst |
| --- | ---: | --- |
| `contract-vendored-dense` / `-twin` (4 modules × 10k) | 40,000 | 3.3e-8 |
| `contract-lean-dense` / `-twin` (2 modules) | 20,000 | 3.3e-8 |
| `lean-read-float` / `fp8` / `fpn` / `half` | 6045 / 3224 / 4836 / 403 | exact |
| `lean-read-active` | 14,508 | exact |
| `lean-sample` / `vendored-sample` | 10,800 / 19,500 | 9.1e-8 |
| `multi-grid-{vendored,lean}-index` / `-sample` | 600 each | 9.8e-8 rel. / 7.8e-8 |
| `walk-brick+minima/zero`, `-/lean`, `walk-brick/background`, `walk-global/zero`, `walk-nodeMax/zero` | 628 rays each | all sequences identical to the twin, max \|Δt\| 0 |

  Fail-before (`?mutate=wgsl-*`, `test.describe("P1 parity fail-before")`, 9 mutations): brick axis tie-break, minima read
  from the maxima, support box one voxel short, nodeMax without its support, lean leaf index (read and active-mask sites),
  lean root key, `outside: 'zero'` one voxel late, every grid through grid 0's affine: each fails its rows.

## Module sizes (min bytes, §3.5 method; slot mode)

| Variant | Float | one Fp | Float + Fp8 + FpN |
| --- | ---: | ---: | ---: |
| `sample` (vendored, for reference) | 20,176 | | |
| `read`, lean | **3193** | 3332 (Fp8) | 3747 |
| `read`, lean, `isActive: false` | 3014 | 3153 | 3568 |
| `sample`, lean | 5236 | 5375 | 5790 |
| `sample`, lean, `outside: 'zero'` | 5393 | 5532 | 5947 |
| `majorant` brick | 22,920 | 24,205 | 25,202 |
| `majorant` brick + minima | 23,246 | 24,531 | 25,528 |
| `majorant` global | 21,155 | 22,440 | 23,437 |
| `majorant` nodeMax | 30,166 | 32,129 | 33,232 |
| `majorant` brick + minima, zero, **lean** | 8481 | 8620 | 9035 |
| `majorant` global, zero, lean | 6390 | 6529 | 6944 |

## Counters per ray (P1.4; `node --experimental-strip-types scripts/p1/majorant-counters.mjs`, 256 rays, 64 on WDAS)

Expected null collisions of delta tracking ∝ ∫bound − ∫density (any σ). Lookups: accessor queries (brick: 2 per cell
with minima).

| Grid | Mode | Segments | Lookups | ∫bound | ∫density | Null share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sphere_fog_float | brick | 18.0 | 36.0 | 64.9 | 48.6 | 0.251 |
| | global | 1 | 0 | 270 | 48.6 | 0.820 |
| | nodeMax | 18.6 | 168 | 73.4 | 48.6 | 0.338 |
| vdb-smoke (pathtracer align) | brick | 13.8 | 27.5 | 5.25 | 0.77 | 0.853 |
| | global | 1 | 0 | 60.0 | 0.77 | 0.987 |
| | nodeMax | 12.4 | 112 | 9.32 | 0.77 | 0.917 |
| wdas_cloud_quarter_fp8 | brick | 82.2 | 164 | 111 | 66.4 | 0.405 |
| | global | 1 | 0 | 408 | 66.4 | 0.837 |
| | nodeMax | 72.7 | 658 | 168 | 66.4 | 0.606 |

(Box: brick 0.205 vs nodeMax 0.181; torus 0.309 vs 0.381; Fp8/FpN equal Float to two decimals.) On counters alone
`brick` is tighter than `nodeMax` everywhere but the box and needs ~5× fewer lookups; P1.10 times them.

## What the pathtracer side (owner) now needs from us

- **P1.7 `volumes.gridSource: 'nanovdb'`:** one pool bound at its group 2 replaces the density texture AND the brick
  majorant binding (`PT_VOLUME_MAJORANT_BINDING`): `pool.add(image, { majorant: vdbBrickMajorant(image, { align:
  'pathtracer', resolution, minima: true }) })`, then `pool.update(encoder)` before building the bind group (CLAUDE.md
  trap). Module: `vdbModuleWgsl({ slot: { group: 2, firstBinding }, types, profile: 'majorant', majorant: 'brick',
  minima: true, outside: 'zero', reader: 'lean' })` (8.3 KiB Float; its own DDA and texture fetch go away). Map:
  `ptVolDensity(p)` → `vdbSample(&acc, pIdx)` (same values, P1.1); `ptDdaInit/Next` → `vdbMajorantInit/Next` (same
  cells and bounds; no zero-length segments, so a loop must not treat one as "next cell"); its minorant texture's `r` →
  `vdbMajorantMinimum(&it)`; its scalar μ (VSPG, emission) → `vdbMajorantGlobal(g)` × max σ. Rays: keep its own
  `worldToIndex` and pass index-space `o` / `d` (t unchanged), or put the medium-to-world in `pool.setTransform` and use
  `vdbWorldToIndex(g, …)`. A grid id per medium (`u32`) in its medium table; several media share the one binding (P1.6).
- **P1.8:** `vdbBuildFromDense(toXMajor(f32), resolution)` (the transpose is in `examples/src/harness/p1-smoke.ts`) gives
  the bit-equal grid; `writeNvdb` can store it as the asset. Values at every voxel centre are checked here.
- **P1.9:** one storage binding for grid data holds: the pool carries images and bricks (+ minima).
- **P1.11:** these exports ship in the next npm release (0.2.0 line, unreleased); it should import
  `vdbBrickMajorant`, `vdbBuildFromDense`, `writeNvdb` from `@pmndrs/vdb/tools` and the core from `@pmndrs/vdb`.
  Its `convertVdbFile` still assumes cubic voxels (`voxelSize[0]`).
- **P1.13:** compile the lean + majorant module inside its media kernels with DXC (sizes above; vendored `sample` would
  add ~20 KiB, the lean one ~8.3).

## Left

- P1.5a (owner): K1.8's timing A/B of the lean reader vs the vendored accessor on the reference device; `reader` stays
  opt-in until then.
- P1.14 (owner question, V176): keep the majorant range header under layout 1 (suggested) or bump the layout. No GPU
  test pins the "headerless range walks as global" path yet.
- P1.15: a GPU brick builder (only if a host rebuilds bricks per frame). P1.12 (optional B path, ours) untouched.
- `nodeMax` has no minima (returns 0); the property test covers fog only (stats bound active values).
