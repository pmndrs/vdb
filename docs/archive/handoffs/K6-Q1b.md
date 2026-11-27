# K6.Q1b / K6.11b handoff: `ambientProbe` deleted, the size-adaptive shadow grid, V128 evidence, spectral and multi-frame references, AO frames

Tasks: TASKS K6.Q1b and K6.11b. Tree: main `24895d1`. Quality only, no timings (D25; the machine ran a 12-core VM and
other agents, load 12–96). References and frames: installed Chrome headless on the owner's Mac (adapter "apple metal-3"),
from frozen copies of the worktree (`scratchpad/k6q1b-ref*`); every e2e on SwiftShader (port 5344, frozen copy 5345).
The K6.R1 report settings throughout: 160 × 90 references to 3 % noise, frames 1280 × 720 box-downsampled 8×.

## 1. `ambientProbe` deleted

The internal A/B setting (`VDBFrameInternalSettings.ambientProbe`, K6.Q1a: no gain measured) is gone with no deprecation
(D35 (5) precedent): the WGSL `vdbFrameSkyProbe` (its call sites read `vdbFrameShIrradiance`, three's irradiance, the former
default), `VdbFrameLights.sky.w` (always 0 now, the field stays), the twins' `probeMode` (`tools/frame/{ambient,grid,spectral}.ts`,
`vdbShHemisphereRadianceAt`, `vdbFrameSkyProbe`, `VDBFrameAmbientProbe`), the harness options `?probeTerm=` (k6-core) and
`?ambientProbe=` (k3-frame, k6-scene), and the five 'radiance' rows of `e2e/k6-ambient.spec.ts`. Those rows stay as
"light probe term" rows (same queries without `probeTerm`); the k6-core `probe-term` mutation now drops the probe's sky
terms from the twins, and it fails every row (585 / 4608 over). CHANGELOG (Breaking, K6.8 list) and the 0.2 migration
guide list it. Gathers shrink by 292 B of minified WGSL (the K6.Q1a growth); size snapshot rewritten.

## 2. Size-adaptive shadow grid: `settings.shadowResolutionMax`

**Setting** (public, core `VDBFrameSettings`, so `./three` too): `shadowResolutionMax`, default **0 = off** (the grid is
`shadowResolution` and every frame is byte-identical: 106 / 106 k6-frames). Above `shadowResolution`, each 'frame' update
sizes the grid from the casters (`vdbFrameAdaptiveCellSize` + `vdbFrameAdaptiveGridResolution`, `tools/frame/grid.ts`):

- each caster asks for a cell of at most 16 of its own voxels (box size / `slot.texels`) and at most 2 optical depths at its
  mean non-zero density (`mean × densityScale × extinction × absorbance / maxScale`, the grid density pass's extinction);
- the finest cell over the casters snaps down to a power of two (the cascade fitter's voxels are powers of two);
- the side is the smallest multiple of 16 whose casters' cascade uses that voxel (R − 1 ≥ extent / voxel), within
  [`shadowResolution`, `shadowResolutionMax`], then the device clamp (`vdbFrameFitGridResolution`).

`VDBFrameRenderer.shadowGridResolution` (core interface; `./three`: `frameRenderer.core?.shadowGridResolution`) reports the
side in use. Internal A/B `shadowResolutionRule: 'both' | 'optical' | 'voxels'` (default 'both').

**A finding on the way:** `vdbFrameGridCascades` picks voxel = 2^⌈log2(extent / (R − 1))⌉, so R = 64 puts only 32 cells
across a unit box and R = 128 only 64 (65 / 129 give 64 / 128). "The 64³ grid" is 32–64 cells across a cloud. R 96 and 128
gave identical numbers on most scenes below for that reason. The adaptive side is chosen to land on the voxel it wants.

**Sweep** ('frame' ms off vs the single-scattering reference; `[side]`; `volume` = 'volume' lighting ms off, which has no grid):

| group | volume | 64 | 96 | 128 | 160 | 200 | optical (½ τ) | voxels (16) | **both (chosen)** |
|---|---|---|---|---|---|---|---|---|---|
| wdas | +4.0 | +22.4 | +1.1 | +1.6 | −5.0 | −5.0 | −5.0 [144] | +1.1 [80] | **+1.1 [80]** |
| wdas-ambient | +18.9 | +45.2 | +17.7 | +17.9 | +9.7 | +9.7 | +9.7 [144] | +17.7 [80] | **+17.7 [80]** |
| wdas-probe | +27.8 | +57.8 | +27.0 | +27.2 | +18.0 | +18.0 | +18.0 [144] | +27.0 [80] | **+27.0 [80]** |
| smoke | −5.6 | −4.8 | −4.8 | −4.7 | −4.7 | −4.7 | −4.7 [112] | −4.8 [64] | **−4.8 [64]** |
| smoke-ambient | +3.5 | +4.5 | +4.5 | +4.5 | +4.5 | +4.5 | +4.5 [112] | +4.5 [64] | **+4.5 [64]** |
| smoke-hemi | +0.8 | +1.7 | +1.7 | +1.7 | +1.7 | +1.7 | +1.7 [112] | +1.7 [64] | **+1.7 [64]** |
| bunny-absorbing | −9.8 | −28.1 | −11.5 | −11.5 | −10.6 | −4.4 | −4.4 [256] | −28.1 [64] | **−11.5 [96]** |
| bunny-absorbing-ambient | +2.8 | −16.9 | +1.7 | +1.7 | +2.5 | +9.5 | +9.5 [256] | −16.9 [64] | **+1.7 [96]** |
| bunny-absorbing-hemi | −1.5 | −20.8 | −2.8 | −2.8 | −2.0 | +4.8 | +4.8 [256] | −20.8 [64] | **−2.8 [96]** |
| bunny-thin | −0.5 | +0.7 | −0.1 | −0.1 | −0.2 | −0.3 | +0.7 [64] | +0.7 [64] | **+0.7 [64]** |
| bunny-spectral | −1.3 | −4.8 | −1.2 | −1.2 | −1.0 | +0.1 | −4.8 [64] | −4.8 [64] | **−4.8 [64]** |
| bunny-spectral-hemi | +10.4 | +7.8 | +11.2 | +11.2 | +11.2 | +12.5 | +7.8 [64] | +7.8 [64] | **+7.8 [64]** |

'neural' (the default MS mode) vs the MS reference, same columns:

| group | 64 | 96 | 128 | 160 | 200 | optical | voxels | **both** |
|---|---|---|---|---|---|---|---|---|
| wdas | −31.6 | −35.1 | −32.1 | −37.8 | −35.8 | −35.0 | −31.6 | **−31.6** |
| wdas-ambient | −25.5 | −28.4 | −25.9 | −30.5 | −28.9 | −28.3 | −25.5 | **−25.5** |
| wdas-probe | −23.3 | −26.1 | −23.8 | −28.1 | −26.6 | −26.0 | −23.4 | **−23.4** |
| smoke | −2.0 | −2.0 | −10.0 | −10.0 | −9.0 | −7.5 | −2.0 | **−2.0** |
| bunny-absorbing | −5.3 | +8.2 | +8.3 | +10.0 | +9.8 | +11.4 | −5.3 | **+8.2** |
| bunny-absorbing-ambient | −2.5 | +12.7 | +12.8 | +14.1 | +15.6 | +16.9 | −2.5 | **+12.7** |
| bunny-absorbing-hemi | −3.2 | +11.5 | +11.6 | +13.0 | +13.8 | +15.3 | −3.2 | **+11.5** |
| bunny-thin | +9.1 | +3.9 | +3.8 | +5.9 | −0.8 | +9.1 | +9.1 | **+9.1** |

Reading it:

1. **Single scattering converges with ~64 real cells across the cloud** and then matches 'volume' lighting (the grid-free
   march) to a few points; past that it overshoots a little (WDAS −5 %, absorbing-ambient +9.5 % at 200).
2. **Build both, then decide.** 'optical' at half an optical depth (the first constant) asked for 112–256 (the absorbing bunny
   256: 768 MiB of grid) and moved 'neural' most. 'voxels' (16) fixed WDAS at 80 but missed the dense bunny. 'both' at 2
   optical depths gives WDAS 80 and the dense bunny 96 and leaves the other eight groups at 64 (byte-identical frames): every
   'frame' single-scattering error lands within 2 points of 'volume' lighting's.
3. **'neural' moves with the grid**, not toward the reference everywhere: its features are optical depths at h·2^k with h the
   grid voxel, so a finer grid shrinks the network's view. On the dense bunny its −3 … −5 % at 64 was two errors cancelling
   (the grid's −28 % single scattering under an MLP overshoot); with the grid fixed it shows +8 … +13 %. Smoke at 128: −2 → −10 %.
   K6.Na gets the follow-up (grid-independent reach, or train across grid sizes).
4. The ambient / probe groups keep +18 % / +27 % at any side: that is shared with 'volume' lighting and is V128 (§3).

**Cost (estimates, no timings).** Grey, one cascade: shadow grid 8 B + transmittance 4 × 8 B + environment grid 8 B per
voxel (+4 B partials with several source groups): 48 B. Side 64: 12 MiB; 80: 23 MiB (2×); 96: 40 MiB (3.4×); 128: 96 MiB.
Spectral: 8 + 8 + 3 × groups × 8 + 8 B. Work: the grid density pass is R³ source reads; the transmittance pass R³ voxels × (lights
+ 2 sky + 2 (N − 1) AO) marches of up to ~R steps (each stops at depth 6, empty voxels skip): ~R⁴, so 80 ≈ 2.4×, 96 ≈ 5×
the 64 pass. It runs only when the lights, casters or occluders change (every frame with `jitter`). The gather reads the grid
by one trilinear lookup per light: unchanged.

## 3. The phase-weighted distant light (V128): a scene needs it; not built (K6.Q1c)

New report groups `*-ambient-only` / `*-probe-only` and their `-iso` twins (distant light alone, single scattering, the
scene's dual-lobe 0.8 / −0.3 / 0.2 phase vs an isotropic phase). For a uniform ambient the isotropic reference **is** the
solid-angle mean of transmittance the shadowed ambient estimates, so the `-iso` row isolates the pair quadrature and the
difference between the two references is the phase weight. 'volume' lighting (full march), AO on, vs SS:

| scene | vs isotropic reference | vs g 0.8 reference | 'frame' 64 / 128, vs g 0.8 |
|---|---|---|---|
| WDAS, ambient | +5.4 % | **+83.7 %** | +145.6 / +89.1 % |
| WDAS, light probe | −2.5 % | **+96.9 %** | +162.4 / +101.7 % |
| smoke, ambient | −7.1 % | **+22.7 %** | +24.3 / +24.1 % |
| absorbing bunny, ambient | −12.7 % | **+59.9 %** | +34.4 / +61.6 % |

The isotropic estimate is good to −13 … +5 %; the scenes' forward phase makes the real in-scattering 1.3–2× darker (light
reaching a visible voxel from behind it has crossed the cloud). That is the remaining +18 % / +28 % of wdas-ambient /
wdas-probe with the sun, in both lightings. So the term is needed. It is not built here: it changes four kernels (grey and
spectral, both lightings), their twins and probes, and 'frame' lighting needs storage the transmittance stack does not have.

**Design for K6.Q1c.** Fit a dipole to the pair visibilities, T(ω) ≈ a + b·ω (a: the mean over the 2N sides; b = M⁻¹ Σ_k
(T(ω_k) − T(−ω_k)) ω_k / 2, M = Σ_k ω_k ω_kᵀ, constant per frame, packed with the directions). For a phase with mean cosine
g_eff (dual-lobe: (1 − w) g + w g_back) the phase-weighted visibility is a − g_eff (b · ω_cam), ω_cam the unit direction to
the camera; clamp to [0, 1] and scale every distant term by it / a. 'volume' lighting has the pairs per froxel (no storage).
'frame' lighting needs b per grid voxel: 3 more channels, a fifth target per cascade in the transmittance stack (no new
binding; +8 B per voxel; spectral: per channel, 9 values). Test: the V128 groups above (the g 0.8 rows should fall to the
`-iso` rows' level) and wdas-ambient / wdas-probe; a fail-before mutation that drops b.

## 4. Spectral reference scenes

`k6-scenes.ts` `spectral: { angstromExponent, phaseGExponent }`; scenes `bunny-spectral` (his 'dark-smoke' preset: albedo
0.84 / 0.86 / 0.88, Ångström 1.8, HG g 0.577 with exponent 0.513, extinction 20) and `bunny-spectral-hemi` (+ the K6.Q1
hemisphere light). k6-scene turns `spectral` on; the reference (`k6-reference ?case=scene`) renders each channel as a grey
medium (that channel's extinction × (λ/550)^−α and both lobes' g × (λ/550)^−exponent, clamped ±0.99, as the renderer),
each to convergence, and publishes `transmittanceRgb`; the report composites and inverts his sRGB blend with each channel's
alpha and prints per-channel bias for spectral groups (V165). No kernel change, so no new twin.

| row (vs SS / vs MS) | bunny-spectral | bunny-spectral-hemi |
|---|---|---|
| 'volume' ms off vs SS | −1.3 % (r / g / b vs MS −74 / −80 / −85 %, the reference's SS −75 / −80 / −84 %) | +10.4 % |
| 'frame' ms off vs SS (64) | −4.8 % | +7.8 % |
| 'frame' neural vs MS | −61.9 % (spectral runs it as 1D diffusion, V133) | −52.0 % |
| 'volume' ms on vs MS | −61.6 % | −51.7 % |

Per-channel single scattering and the per-channel shadowed ambient are right to a few points (the +8 … +10 % of the hemi
scene is the V128 phase term of §3, as in grey). Spectral multiple scattering is the large error: every spectral MS mode is
1D diffusion (V133), −52 … −62 % against an MS-dominated medium. Not in scope; noted for K6.Na / the owner.

## 5. Multi-frame mode and the AO ring (K6.11b owner question)

k6-scene: `?frames=N` draws N frames with `?jitter=1` / `?temporal=1`; `?average=K` reads the linear mean of the last K (the
jittered estimator's ensemble mean). Internal A/B `ambientOcclusionRing: 'turn' | 'fixed'` (k6-scene / k3-frame `?aoRing=`).
Report groups `smoke-hemi-jitter`, `bunny-absorbing-hemi-jitter` ('frame', ms off vs SS / neural vs MS):

| row | smoke-hemi | absorbing-hemi |
|---|---|---|
| static (no jitter) | +1.7 / −1.2 % | −20.8 / −3.2 % |
| jitter + temporal, 32 frames, ring turning (default) | +1.7 / −1.4 % | −20.8 / −4.2 % |
| jitter + temporal, 32 frames, ring fixed | +1.8 / −1.4 % | −20.7 / −4.1 % |
| jitter, mean of 32 frames (no temporal), ring fixed | +2.5 / −1.3 % | −17.7 / −1.2 % |
| static, AO off | −4.3 / −0.1 % | −20.7 / −3.2 % |

**Question for the owner: keep the AO ring turning per frame with `jitter` + `temporal`, or fix it?**
- Measured: at 4 pairs turning and fixed agree to 0.1 points against the reference on both scenes; the turning ring costs
  ~0.09/255 frame-to-frame shimmer after the temporal blend (K6.11) and needed `ao=0` in the K3.3 temporal gate.
- **Suggestion: fix the ring** (`'fixed'`) and delete the setting. Pros: deterministic frames under temporal, no shimmer, one
  less special case in gates, no measured quality loss. Cons: with 1–2 pairs a fixed ring keeps its directional banding
  that turning would average; hosts that drop to 1 pair for cost lose that.
- Test that settles it: the same two groups at `aoPairs=1` and `2` (`--query=aoPairs=2`): if turning beats fixed by more
  than the reference noise (~3 %) there, keep turning; else fix it.

## 6. K6.11b: AO frames in the before / after gate

`e2e/k6-frames.spec.ts` gains 8 frames: k3-frame's CC0 smoke with his hemisphere + ambient (`lights=dir-hemi`, 'volume' and
'frame'; CI) and k6-scene's `wdas-probe` ('frame', 'volume'), `smoke-hemi` ('frame'), `bunny-absorbing-hemi` ('frame',
'volume'), `bunny-absorbing-ambient` ('frame') at 640 × 360 (local fixtures). Rendered from main (`24895d1`, the new spec
copied in) with the default and with `K6_FRAMES_QUERY=ao=0`: every one of the 8 moves (max 10–69/255, 10.4–15.5 % of pixels
> 1/255), so the AO default now has a gate in each ambient-class light and both lightings.

## Gates

| gate | result |
|---|---|
| `pnpm typecheck` | 0 |
| `pnpm test` | 104 files, 2082 passed, 22 skipped |
| `pnpm docs:check` | 47 blocks match, snippets type-check |
| `e2e/k6-frames.spec.ts` vs main (before: `git archive 24895d1` + this spec, port 5347; after: this tree, 5344) | **106 / 106 identical** (max 0/255) |
| `e2e/k6-ambient.spec.ts` | 19 passed (the 5 light-probe rows: 0 over, ≤ 0.74 ulp; `probe-term` 585 / 4608 over) |
| `e2e/k6-adaptive-grid.spec.ts` (new, CC0 smoke) | 1 passed: off and max = base byte-identical (64); the default rule keeps the thin smoke at 64 (identical); at extinction 400 the side grows to 193 (SwiftShader's 256 MiB `maxBufferSize` clamp; Metal would take 256), capped 96 with max 96, frame moved ≤ 45/255 |
| `e2e/k6-core.spec.ts` subset (`-g` K6.1 gathers, every K6.2 'frame' probe, K6.3 two-sided frame / gather, K6.4 spectral gather and frame, the grey "every pipeline variant compiles"), frozen copy | **18 passed**, mutations fail. Not run: the spectral compile probe (separate 5-min run; the spectral gathers lost the same removed function as the grey ones) |
| frame size snapshot | rewritten on purpose (`K6_WRITE_SIZES=1`): gathers −292 B min |

## DIVERGES

V163 (`ambientProbe` deleted), V164 (`shadowResolutionMax`), V165 (spectral reference as three grey renders); V128 updated
with the evidence (K6.Q1c).

## Owner questions

1. **`shadowResolutionMax` default.** Suggestion: leave it 0 for 0.2 and revisit with K6.T. Pros of a default (e.g. 128): large
   and dense clouds lose a +22 % / −28 % single-scattering error at 2–3.4× the grid memory; thin or small media unchanged.
   Cons: the transmittance pass grows ~R⁴ on those scenes (every frame with `jitter`), and the 'neural' MS term moves with
   the grid (dense bunny −5 → +8 %) until K6.Na makes its features grid-independent. Test: K6.T timing of the transmittance
   pass at 64 vs 80 / 96 on WDAS and the dense bunny (static and jittered) on a quiet machine, plus the 'neural' rows after
   K6.Na.
2. **AO ring** (§5): suggestion fix it; test at 1–2 pairs.
3. **V128** (§3): build K6.Q1c? Suggestion yes ('volume' first: no storage); it is the largest single-scattering error left
   with ambient-class light (+23 … +97 % alone, +18 … +28 % beside the sun).

## Follow-ups (TASKS)

K6.Q1c (V128); K6.Na note (grid-independent MLP reach); spectral MS is 1D diffusion (−52 … −62 % on the spectral bunny).
