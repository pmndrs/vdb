# K6.10: K6 gate review (independent)

Reviewer: a separate agent. I did not build any K6 step. Tree: main `7e32f37` fast-forwarded into this worktree.
Product code is unchanged. One repro test was added in its own commit (`07bdc07`, `e2e/k6-10-repro.spec.ts`, `test.fail()`).

Machine: owner's Mac. Every e2e below ran on SwiftShader (bundled Chromium, Playwright 1.61.1).

Ports:
- 5318: the k6/k7, demo 17 and no-webgpu specs, then the compile probes.
- 5338: the demos, frames, parity and opt-in specs.
- 5337: a `git archive` of `1376cfc`, main before K6.8, with its own install.
- 5341: the repro.

Ports 5320 and 5319 were avoided: another repo's Vite server held 5320.

Logs are in `artifacts/k6-10/` (git-ignored). No timings were taken.

**Recommendation: K6 passes with follow-ups.** Every exit-gate line is met on SwiftShader. Nothing found breaks a default
render. Follow-ups:
- one reproduced crash on an out-of-range setting (`shadowResolution` > 1024);
- an unclamped volume shadow map size;
- an immediate destroy in the dense source;
- probe-guard gaps;
- stale TSL wording.

## Verdict per gate line

| Gate line | Verdict | Evidence |
|---|---|---|
| A raw-WebGPU demo renders the WDAS cloud with 'frame' lighting, shadows both ways and emission, without three | **Met** | See "Demo 17" below. |
| `./three` draws through the core | **Met** | `three/frame/renderer.ts` creates `vdbCreateFrameRenderer`, records `core.update()` into its own encoder and submits. `three/frame/` holds only the adapter and core glue. `k6-core` "K6.1c raw scene vs ./three core": 6/6 of his scenes (`K6_HIS_SCENES=1`) plus 2 CC0 scenes are identical (∞ dB, max 0/255). |
| Each kernel has a CPU twin and a fail-before mutation (≤ 2 f16 ulp or a justified bound) | **Met, with gaps** | See "Twins" below. Every kernel has a twin and a probe, and the probes pass. Some variants have no mutation, and the k6-core mutations flip a twin term, not the WGSL. |
| Frames match the TSL path (while it existed) | **Met** (reproduced) | On the `1376cfc` tree I ran `K6_CORE_AB=1 … -g "A/B core vs TSL"`: **79/79 pass**, with numbers equal to the handoffs. Step 1 is 57.15–61.42 dB, Step 3 61.47–61.90, Step 4 59.01–62.82, Step 5 57.78–62.58, Step 6 60.80–64.90, env 58.38–61.92 dB, WDAS 69.74/68.20 dB, shadow and emission on − off terms max 0–1/255. The 'outward' rows (47.09/47.91 dB, 1.5–1.9 % > 4/255) are report-only under V96. Today's core frames equal those core frames byte for byte (next row). |
| Demo goldens explained / K6.8's "98 frames byte-identical" | **Met** (reproduced) | I rendered `e2e/k6-frames.spec.ts` (`K6_FRAMES_OUT`) on the `1376cfc` archive and on this tree. A pixel diff gives **98/98 identical** (RGBA max 0). No golden or demo spec file changed during K6 (`git diff 879f263..HEAD -- e2e/*-snapshots e2e/demo-0*/1*`: only `demo-17.spec.ts` was added). Demos 02, 04–16: 32 pass. `demo-06` "nvdb … matches the golden" fails with 7,999 px (3 %, panel text). It fails identically on `1376cfc`: a Linux-only golden on macOS, pre-existing. |
| His K3 captures within tolerance; failures listed in DIVERGES | **Met** | See "His captures" below. |
| A reference renderer judges quality choices | **Met** (instrument) | `e2e/k6-reference.spec.ts` passes. Its 9 mutations patch the WGSL. The multiple-scattering default is still `'neural'` (`settings.ts:140`), the experimental MLP, pending the owner's decision (PHASES status). The K6.R1 report has the numbers. |
| Shadows both ways | **Met** | Volume → mesh: `k6-core` K6.5 shadow probes, `k6-shadows`, `k7-volume-shadow-stale`. Mesh → volume: `k6-occluders`, plus the K6.5b occluder probes in transmittance, the 'frame' gather and the 'volume' gather, dense and sparse, grey and spectral. Demo 17 checks both directions. |
| Every core kernel has CPU-twin parity | **Met** | See "Twins" below. |
| `test/boundaries.test.ts` still holds | **Met** | It passes in `pnpm test`. The only bare import in `src/core` and `src/tools` is `fflate`. |
| The TSL passes are deleted | **Met, with stale wording** | No TSL froxel pass, chunk or `backend` branch remains in `packages/vdb/src`. What remains in `three/frame` is legitimate: `emission-light.ts` and `volume-shadows.ts` are mesh receivers, `core-integrate-material.ts` is the composite seam. Leftover wording is listed under finding 6. |

**Demo 17 evidence.**
- Module graph: `examples/test/demo-08-imports.test.ts` "demo 17 import graph" walks the graph and finds no `three`, no `@pmndrs/vdb/three` and no `showcase-shell.ts`. I checked by hand that its imports are only `@pmndrs/vdb` and `@pmndrs/vdb/tools` (public index files) plus `showcase-ui`, `assets` and `webgpu-gate`. None of them reaches `internal.ts`.
- `e2e/demo-17.spec.ts`: 3/3 pass. The WDAS + icosahedron case ran because the fixtures are present. It checks 2 volumes, `lighting: 'frame'`, shadow map enabled, emitters > 0, 1 occluder, and a byte-identical repeat with 0 builds.
- Effect of turning each feature off:

  | Off | WDAS | CC0, 'frame' |
  |---|---|---|
  | volume shadows | 16,162 px brighter (+32.1 luma) | 6,466 px (+35.3) |
  | occluders | 7,313 px brighter (+19.8) | 10,200 px (+22.0) |
  | emission | 109,303 px darker (−9.9) | 108,409 px (−7.7) |

- Host composite vs core composite: within the bar (max ≤ 3).
- Without the WDAS fixture the demo uses the committed CC0 JangaFX cloud (`?cloud=cc0`, CI).

**Twins evidence.**
- Matrix: below.
- Default k6/k7 run: **159 passed, 90 skipped, 0 failed** (33.8 min, 9 workers). The 90 skipped are all opt-in: 81 `k6-frames`, 6 `K6_HIS_SCENES`, 3 `K3_FRAME_PARITY`. I ran every one of them separately; all pass.
- Compile probes: **2/2 pass** (K6.1 variants 6.8 min, K6.4 spectral variants 3.0 min).
- Nothing failed under load, so no test was seen passing only when run alone.

**His captures evidence.**
- `pnpm k3:parity` renders 101/101. `compare.mjs`: **33 PASS, 12 FAIL, 49 INFO** (INFO = his non-isolated scenes and the explosion rows; the explosion is not gating without the CIE table, D33).
- `compare.mjs` over `k6-frames`' `ours-core` gives the same 33 / 12 / 8. The handoff's "36 pass / 17 fail" counts INFO rows differently.
- The 12 FAILs are:
  - step1 ms-off-isolated;
  - step3 two-bunnies;
  - Step 4 g-low, octave-extinction-high and octaves-1;
  - Step 5 angstrom-air, dark-smoke (×2), dust, haze, soot and white-smoke.
- Every FAIL is between 0.504 % and 0.790 % of pixels over 8/255, all ≥ 47.41 dB. All are covered by V80 (Step 4 also by V137).
- Spot-check of the masks (my script, not theirs), on 5 failing scenes: **98.7–100 %** of the > 8/255 pixels lie on an edge of his image (a step > 32/255 within 2 px). Those edges are only 7.9–9.2 % of the rect. Off the edges the max diff is 6–10/255. The soot heatmap is a clean outline.
- Step 7 (`compare-step7.mjs`, run by hand because `k3:parity`'s `&&` stops at compare's exit 1):
  - surfaces IoU 0.9943 (teapot) and 0.9960 (Venus);
  - points 9802 = 9802;
  - Venus normal mean 6.57°, over the provisional 5° bar (INFO).

## Structural runs

| Command | Result |
|---|---|
| `pnpm typecheck` | exit 0 |
| `pnpm test` (root) | 101 files, 2058 passed, 22 skipped. None of the skips is K6: they need `MJURCZYK_OPENVDB_DIR`, the CIE table or native Half. |
| `pnpm build` | exit 0 |
| `pnpm examples:build && pnpm examples:smoke` | 22/22. Demo 17 is loaded from the built site, with nothing missing. |
| `node scripts/docs-snippets.mjs --check`, docs snippets `tsc` | pass (45 blocks) |

## Twins (kernel → twin → probe → mutation)

The k6-core probes compare per texel within 2 f16 ulp of max(|gpu|, |cpu|), with a floor of 2^-20. Most of them allow
0.1 % of texels over. The K6.5 and K6.2b-environment probes allow none.

| Kernel / variant | Twin (`tools`) | Probe | Mutations |
|---|---|---|---|
| `vdbFrameChunkProbeCs` (view chunks) | slice/view/noise/phase/Mie/octaves twins | K6.1 "view chunks" | coverage, mie, octaves, two-sided |
| 'volume' gather, grey + spectral (mixture/over, 1–2 groups, dense/sparse, LODs, env, occluders, AO) | `gatherTwin` / spectral | K6.1, K6.1d, K6.2b, K6.3, K6.4, K6.5b, K6.Q1 | mix, phase, extinction, fade, skip, mie, octaves, two-sided, occluder(-filter), env-*, spectrum, ao |
| grid density, transmittance, 'frame' gather, grey + spectral (partials, cascades, every MS mode incl. mlp, multi-scale, `paper` σ, occluders, AO) | `vdbFrameGridDensityVoxel`, `…TransmittanceVoxel3d`, his density + lighting twins | K6.2, K6.3, K6.4, K6.5b, K6.Q1 | grid-absorbance/lod, diffusion, slots, sky, mlp-*, ms-edge/levels/sigma, occluder(-scattered), spectral-*, ao-pairs, env-slot |
| grid mips (mlp features) | `vdbFrameMlpMipTexel` | K6.3 mlp | mlp-mip |
| resolve (temporal, spectral, rg16float prefix) | `vdbFrameScanColumn(Spectral)`, `vdbFrameTemporalFroxel` | K6.1, K6.4 | scan-stop, temporal-clamp, spectral-alpha (none for rg16float) |
| composite VS/FS | `vdbFrameIntegratePrefix`, `vdbFrameCompositeSpectral` | K6.1, K6.4 | coverage, taps, spectral-alpha |
| env area / cube / blur / reduce / env grid | `vdbFrameEnvironment*` | K6.2b | env-area, env-blur, env-seeds, env-sh, env-grid (no cube blur or cube rotation case) |
| emission direction / reduce / probe / irradiance (+ user hook) | `vdbFrameEmission*` | K6.5 | emission-direction/color/reduce/probe/irradiance |
| volume shadow map + receivers (dir/spot/point, spectral rgb) | `vdbFrameShadowMapTexel`, `vdbFrameVolumeShadowLookup(Rgb)` | K6.5, K6.4 | map-steps, map-far, receiver-far, receiver-layer, spectrum |
| `csMsLevel` / `csMsPack` (K7.3) | `vdbFrameMultiScaleMips` | `k7-multi-scale` (≤ 1 ulp) | none direct; indirect via K6.3 `ms-*` |
| pyramid (4 entries), atlas, stats | `vdbBoxPyramid`, `vdbDecodeToAtlasCpu`, `vdbGridStatsCpu` | `k3-pyramid`, `compute` | none |
| sparse source `vdbSrc*` | `vdbSparse*Cpu` | `k6-sparse` | WGSL patches: samplePos, slot, origin, blend, dilate |
| level-set pass VS/FS | `levelSetSurface`, `levelSetNdcDepth` | `k6-surface` (IoU ≥ 0.99, depth < 1e-6) | pixel-corner, ray-ortho, normal-view, depth-linear, shade-sign |
| point pass | `pointSprite*` | `k6-surface` sprite probe + raster | sprite-half/min/w (raster check has none) |
| reference `vdbRefTraceCs` | `vdbReferenceRenderCpu` + analytic | `k6-reference` (statistical) | 9 WGSL patches |

## Findings, ranked

1. **MEDIUM: `shadowResolution` above 1024 throws instead of clamping. Reproduced.**
   - Location: `packages/vdb/src/core/frame/renderer.ts:997-1010` (`#gridResolution`). Its clamp loop calls `bytes(r)` → `vdbFrameStackLayout(r, targets, maxDim)`. That function throws a `RangeError` when the stack does not tile in `maxTextureDimension3D` (`tools/frame/stack.ts:48`), so the loop never lowers R.
   - Repro (`e2e/k6-10-repro.spec.ts`): `k3-frame.html?lighting=frame&shadowRes=1100` gives "vdbFrameStackLayout: 4 sub-volumes of 1100³ do not fit 2048³". Every update throws, so the frame is lost and the doc's "clamped (warned once)" never happens.
   - Also triggered by R = 1024 with 3–4 cascades (12–16 targets > 8), and by a lower R when spectral.
   - No unit test covers the clamp.
   - Fix: count a layout that doesn't fit as too big (catch, or check `targets ≤ ⌊maxDim/r⌋³`). Turn the repro into a regression test.
2. **MEDIUM: the volume shadow map size has no limit check. Not reproduced.**
   - Location: `core/frame/shadows.ts:185` (`Math.max(1, Math.round(light.shadow.size))`) and `three/frame/volume-shadows.ts:273` (`castShadowMapSize`). Nothing clamps to `maxTextureDimension2D` or the byte size.
   - A point light at `castShadowMapSize = 4096` asks for 4096² × 6 rgba16float (805 MB; spectral doubles it). That is the same Dawn zero-fill `maxBufferSize` class K6.8 fixed for the grid stack.
   - A size > 8192 is an invalid texture, which invalidates the host's command buffer.
3. **LOW–MEDIUM: `vdbDenseSource.replace()` with a new size destroys its textures immediately.**
   - Location: `core/source/dense.ts:205-208`.
   - It bypasses the retire contract of design §4.2. A raw-core host that calls `replace` between `frame.update(encoder)` and `queue.submit` gets a destroyed-texture submit error.
   - `./three` is safe: it syncs the source before it records.
4. **LOW–MEDIUM: probe guards are weaker than the gate wording.**
   - (a) The k6-core mutations flip a twin term (`?mutate=`). That proves the comparator would see the term; it does not prove the kernel computes it. Only K6.R and K6.7 patch the WGSL.
   - (b) The K6.3, K6.4, K6.5 and K6.2b-environment probes never assert `covered` (`e2e/k6-core.spec.ts` ~222-228, ~334-339). Four of them also have no mutation:
     - two-sided sparse SH;
     - resolve rg16float;
     - SH L1 environment;
     - gather 'outward'.

     An all-zero GPU and twin pair would pass them. In this run they are not degenerate: 266–16,384 non-zero values each.
   - (c) Untested branches:
     - the non-spectral sRGB composite (compile-only);
     - the composite's orthographic and reversed-depth `vdbFrameSceneViewDepth` and its `depth32float` read;
     - the level-set pass on an orthographic GPU ray (demo 17's light depth map uses it; only a twin mutation exists);
     - the emission colour hook and LUT inside the gather kernels (compile-only; the gather twin uses white);
     - resolve under camera motion.
   - (d) The pyramid, atlas, stats and `csMs*` kernels have no direct fail-before mutation.
   - (e) Whether the rg16float alpha prefix was actually used on SwiftShader is not asserted. This is already K6.4a.
5. **LOW: the retire queue counts `update()` calls, not submits.**
   - Location: `core/frame/resources.ts:28-36`, and the same pattern in `level-set-pass.ts` and `points.ts` per `record()`.
   - A host that records three or more updates (or point-pass records after `setPoints`) into one encoder before submitting can destroy a texture or buffer an earlier recorded pass still uses.
   - This is documented as one update per submit; nothing enforces or warns.
6. **LOW: stale TSL and `backend` wording.**
   - `scripts/k6/reference-report.mjs:98-199`: about 40 rows labelled "TSL …" now render the core, because `?backend=` is ignored. The "TSL vs core" rows compare core with core, and the V80 mask code at ~586-601 is dead.
   - `core/frame/settings.ts:61-65`: `froxelMix` is documented as "internal, for the A/B against the TSL path", but it is public and in the CHANGELOG.
   - `settings.ts:15-18`: an orphaned duplicate JSDoc.
   - `tools/frame/index.ts:15-20`: still says the twins "mirror … `frame/chunks.ts`", which is deleted.
   - User-facing warnings still say "backend 'core'": `three/frame/core-occluders.ts:57,107`, `core-textures.ts:31`.
   - Harness and e2e comments: `k6-scene.ts:8`, `e2e/k6-frames.spec.ts` (`&backend=core` is passed and ignored).
7. **LOW: API and docs (D32 judgement, not test failures).**
   - The default `multipleScattering: 'neural'` runs the EXPERIMENTAL MLP. The field (`settings.ts:48`) and the `./three` re-export of `VDBFrameMlp` carry no experimental note.
   - Naming is inconsistent:
     - `vdbCreateFrameRenderer` → `VDBFrameRendererGpu` and `vdbCreateBoxPyramid` → `VDBBoxPyramidGpu` have no `Gpu` in the function name, but `vdbCreateLevelSetPassGpu` / `vdbCreatePointPassGpu` do;
     - core `VDBSparseIndexBox` vs three `VDBIndexBox`;
     - `VDBFrameLoader` (sequences) sits in the froxel `VDBFrame*` namespace.
   - Internal-looking public names: `VDB_GPU_LAYOUT_VERSION`, `VDBFroxelLayout`, `VDBBoxPyramidLayout`, `WvdbLayout`, `wvdbPrefixLength`, `nvdbResponseLength`, `vdbMakeFpNLeafCodec`, among others.
   - The 0.2.0 CHANGELOG omits some new public names: `vdbFrameDefaultSettings`, `vdbFrameDefaultVolumeParams`, `VDB_VOLUME_SHADOW_MAX_CASCADES` and the `VDBFrame*` input types.
   - `packages/vdb/README.md` is unchanged for 0.2.0: it still calls `./three` "the three.js / TSL adapter" and does not mention the core renderer.
   - `docs/site` has no 0.1 → 0.2 migration page.
8. **LOW (plausible, not reproduced): `./three` volume-shadow stand-ins.**
   - (a) `three/frame/volume-shadows.ts` ~391-396: when a light's map disappears, the entry keeps its old stand-in (enabled = 0) while the core retires the texture.
   - (b) ~411-424: on spectral → grey, `rgbBinding` keeps the stand-in of the retired rgb texture until `#free`.
   - Receivers built earlier may bind a destroyed texture until three rebuilds them.
9. **INFO: the `demo-06` golden fails on macOS.** It is Linux-only and pre-existing (same 7,999 px on `1376cfc`). This is not a K6 issue.

Checked and fine (code reading, plus a second agent's read where noted):
- Resolve/scan: one thread per column, zero outside the area, history cleared outside the rect.
- Multi-group partials: `inside` is frame-wide, so a non-last group never drops an earlier group's sums.
- Occluder slots 12–15, with placeholders and the 4-slot limit warned. The bias sign follows three and is documented.
- The spectral and sRGB composite requires `sceneColor` with a clear `VDB_BAD_OPTION`.
- The group-2 source rebuilds on `pool.version` (sparse `version = own + pool.version`).
- The level-set pass rebuilds on `pool.version`/`pool.buffer`.
- Group 0 caches only constant resources.
- The shadow bind group cache is keyed by texture.
- `VDBCoreTextureBinding.set` swaps the node's value and disposes the old stand-in in one call.
- ExternalTexture dispose never destroys the core's texture (self-test).

## What I could not verify

- **Hardware.** Everything ran on SwiftShader. The rg16float alpha prefix and hardware filter quantization are K6.4a, and no timings were taken (D25).
- **Finding 2.** Not run, because a ~805 MB texture on a shared machine was not worth the risk.
- **Finding 8.** Not reproduced, because it needs a three receiver rebuild sequence.
- **The K6.R quality conclusions.** I re-ran only the reference spec (pass), not the report (~75 min per group).
- **The K6.8 98-frame set.** It covers the scenes the A/Bs used. It does not cover settings outside them, such as large `shadowResolution` (finding 1).
- **`pnpm wgsl:variants`** (design §7.5) was not run. The two compile probes cover the frame kernel matrix.

## Follow-ups proposed (K6.10a)

1. Fix finding 1 and turn the repro into a regression test.
2. Fix finding 2: clamp the map size to the device limits and to `maxBufferSize`, warned once.
3. Fix finding 3: give the dense source the retire queue.
4. Fix finding 4b: add `covered` minimums to the K6.3, K6.4, K6.5 and K6.2b-environment probes.
5. Fix finding 4c: probes for the sRGB grey composite, ortho/reversed depth, and the ortho level-set pass.
6. Add WGSL-side mutations for a few kernels (finding 4a).
7. Clean up the stale TSL wording (finding 6) and fix `reference-report.mjs`.
8. Mark the experimental MS default and the API names (finding 7) before 0.2.0 ships.
