# K6.Q1a / K6.4a handoff: spectral shadowed ambient, the probe term A/B, ring jitter, spectral hardware checks

Tasks: TASKS K6.Q1a (shadowed ambient follow-ups) and K6.4a (spectral hardware checks). Builds on [K6-Q1.md](./K6-Q1.md)
(the grey shadowed ambient, V125–V128) and [K6-4.md](./K6-4.md) (spectral media, V129–V135). Quality only; no timings.

## 1. Spectral shadowed ambient (`ambientOcclusion` with `spectral: true`)

The spectral kernels honour `ambientOcclusion` now (the "grey-only" warning is gone). Same method as K6.Q1 (N direction
pairs, each side the mode's isotropic-light total, averaged per hemisphere), with the totals **per channel**.

- **'frame' lighting: no new target.** With the setting on, the spectral group 0's sky entry (sky up rgb, sky down rgb:
  the first 6 of its 12 values) holds the hemispherical visibilities V up, V down of each channel *instead of* the sky
  totals. That is lossless: in the grey kernels every reader of the sky totals switches to t3.zw when the setting is on
  (hemisphere, probe, 'outward' environment, ambient), so the spectral sky totals have no reader left once the
  visibilities exist. The brief's 4th target for group 0 (or stacking along z) would have cost a sub-volume of the stack
  per cascade and a sampled-texture read per lookup for nothing. Pair 0 reuses the sky marches; the other pairs are one
  extra march call site (2(N − 1) spectral grid marches per transmittance voxel, as grey).
  - Octaves: each direction's spectral octave total over an open direction's (V127), at the voxel's **grid albedo**
    (1 − absorption / extinction of the voxel's own cascade texel, as the spectral environment grid already does). The
    non-AO path uses the gather's albedo at the visible position; the transmittance pass has no such position.
  - The gather reads the entry as is when `counts.z > 0` (`select(total(sky), sky, ao)`), ambient × (V up + V down) / 2.
- **'volume' lighting:** the grey design per channel: ambient pair items in the spectral march loop (the kernel's one
  march call site), `vdbFrameSpectralShadowMarch(…, stepCount)` (`ambientOcclusionSteps`), the hemisphere lights' own
  marches and the unshadowed ambient / 'outward' probe go away when pairs run (V126 as grey).
- Twins: `tools/frame/ambient.ts` `vdbFrameSpectralAmbientPairTotals`, `vdbFrameSpectralAmbientVisibility`,
  `vdbFrameAmbientIrradiance` (accepts per-channel visibilities); `tools/frame/spectral-stack.ts`
  `vdbFrameSpectralTransmittanceVoxel3d({ ambientDirections })`, `vdbFrameSpectralEnvironmentDensityTerm(…, ambientOcclusion)`;
  `tools/frame/spectral.ts` `vdbFrameSpectralLightingFroxel({ ambientOcclusion })`; harness `volumeFroxelSpectralTwin`.
- White / grey medium: `test/tools/frame-spectral-stack.test.ts` "K6.Q1a spectral ambient visibilities": the spectral
  sky entry of a grey medium equals the grey fourth target's z, w in every channel to 12 digits (single scattering
  absorbing, diffusion absorbing, octaves white, 4 pairs on a turned ring); the pair totals and the irradiance reduce to
  grey channel by channel; a blue-heavy medium gives blue the lowest visibility and diffusion keeps V up + V down = 2 in
  every channel (white furnace).

## 2. The 'frame' probe term (`ambientProbe`, internal setting)

`VDBFrameInternalSettings.ambientProbe`: `'irradiance'` (default, his: three's cosine-weighted irradiance along ± the sky)
or `'radiance'` (π × the probe's mean radiance over each hemisphere: band 0 × 1, band 1 × ½, band 2 × 0; Funk–Hecke,
checked against a dense quadrature in `frame-ambient.test.ts`). It applies wherever the probe is read along ± the sky:
'frame' lighting (grey and spectral, with or without AO) and 'volume' lighting with AO; carried in `VdbFrameLights.sky.w`
(WGSL `vdbFrameSkyProbe`). Harness `&probeTerm=radiance` (k6-core), `?ambientProbe=radiance` (k6-scene, k3-frame).

Reference report, group `wdas-probe` (owner's Mac, installed Chrome headless, "apple metal-3"; the K6.R1 settings;
references rendered from a frozen copy, then `--reuse`). Luminance bias vs single scattering (ms off rows) / vs MS:

| row | AO off, irradiance | AO off, radiance | AO on, irradiance | AO on, radiance |
|---|---|---|---|---|
| 'volume' ms off vs SS | +189.3 % | +189.3 % (no ± sky term without AO) | +27.8 % | +27.4 % |
| 'frame' ms off vs SS | +68.5 % | +66.6 % | **+57.8 %** | +57.4 % |
| 'frame' diffusion vs MS | −51.3 % | −51.6 % | −53.0 % | −52.9 % |
| 'frame' neural vs MS | −21.6 % | −22.0 % | −23.3 % | −23.2 % |
| 'frame' octaves vs MS | −42.4 % | −42.8 % | −55.4 % | −55.4 % |

The probe term moves every row by ≤ 2 points (the reference noise is ~3 %). **The +58 % is the 64³ shadow grid, not the
probe term:**

| 'frame' ms off vs SS | `shadowResolution` 64 (default) | 128 |
|---|---|---|
| WDAS, sun only | +22.4 % | **+1.6 %** |
| WDAS + probe, AO on | +57.8 % | **+27.2 %** (= 'volume' +27.8 %) |

**Recommendation:** keep `'irradiance'` (his; no measurable gain from `'radiance'`), and delete the internal setting
when the A/B settings are pruned. The 'frame' WDAS gap is a `shadowResolution` question (the WDAS cloud is 1.6 k voxels
across a 64³ grid); a default / adaptive resolution is K6.T cost work. The remaining +27 % (both lightings, AO on) is the
isotropic hemispherical estimate of a strongly anisotropic probe term (V128-type), not the irradiance form.

## 3. Ring jitter

With `jitter` and `temporal` both on, the ambient ring turns by the frame's jitter value × its azimuth step
(`vdbFrameAmbientDirections(sky, pairs, phase)`; pair 0 stays the sky). A static frame (jitter off, every twin probe and
the k6-frames set) is unchanged. No extra cost: a jittered transmittance is redrawn every frame anyway (the directions
are in its signature). Probe: `?probe=gather&ao=4&aoJitter=5` (frame 5's jitter, jitter + temporal on) matches its twin
(0 over; the twin uses the same f32 jitter for its noise), `?mutate=ao-ring` (the unturned ring) fails (480 / 4608).
Not measured against the reference (the report pages render one static frame).

## 4. K6.4a hardware checks (installed Chrome headless, adapter "apple metal-3", texture-formats-tier1 present)

Run from a frozen copy (`scripts/k6/k6-core-hardware.mjs`, a plain Playwright `channel: 'chrome'` loop over k6-core):

| probe | result |
|---|---|
| `?probe=resolve&spectral=1&tier1=1` | alpha prefix **rg16float** (gpuGroups 2); 9520 values, 0 over, max 1.00 f16 ulp — pass |
| `… &temporal=1` | 18480 values, 0 over, max 1.01 ulp — pass |
| `… &mutate=spectral-alpha` | 2814 / 9520 over — the probe sees the alpha prefix |
| `?probe=resolve&spectral=1` (rgba16float) | 9520, 0 over, max 1.00 ulp |
| `?probe=compile&spectral=1&tier1=1` | 35 variants, 0 errors (both alpha formats of both resolves) |
| `?probe=gather&spectral=1&tier1=1` | scatter 4608: 1 over (≤ 0.1 %), max 2.17 ulp; `render()` reading the rg16float prefix: 384, 0 over, max 1.11 ulp |
| `?probe=integrate&spectral=1&tier1=1` | 1120, 0 over, max 1.00 ulp |

Nothing failed; nothing to fix. The probes' data is smooth enough for 8-bit filter weights (the resolve and composite
read texels; the gather's one outlier is within the 0.1 % allowance).

## Gates

- `e2e/k6-ambient.spec.ts` (SwiftShader, port 5321): **19 passed** (7 K6.Q1 + 12 new). New probes (2 f16 ulp, ≤ 0.1 % over):

| probe | transmittance | scatter | mutations (over / compared) |
|---|---|---|---|
| spectral frame, ss, 4 pairs | 24576, 0.50, 0 | 4608, 0.71, 0 | ao-pairs 2483, ao-grey 2410 (transmittance); ao 585 |
| spectral frame, diffusion, 7 pairs, sparse, 2 groups | 0.50, 0 | 1.67, 0 | ao-pairs 1753, ao-grey 1742; ao 250 |
| spectral frame, octaves, 3 pairs | 0.50, 0 | 0.66, 0 | ao-pairs 2316, octaves 6458 |
| spectral gather, ss, 4 pairs | — | 0.74, 0 | ao-pairs 572, ao-grey 584, ao 585 |
| spectral gather, diffusion, 4 pairs, 6 steps, sparse | — | 2.09 (abs floor), 0 | ao-pairs 238, ao-grey 253, ao 257 |
| spectral gather, octaves, 2 pairs | — | 0.73, 0 | ao 585, octaves 585 |
| probe term 'radiance': frame AO / frame no AO / spectral frame / gather / spectral gather | 0.50, 0 | ≤ 0.67, 0 | probe-term 585 / 581 / 584 / 585 / 584 |
| gather, 4 pairs, ring turned (aoJitter=5) | — | 0.72, 0 | ao-ring 480 |

- `e2e/k6-core.spec.ts`, grey and spectral compile probes run separately, and `e2e/k6-frames.spec.ts` (98 frames vs a
  baseline rendered from main): see the commit notes below.
- `pnpm typecheck` 0; `pnpm test` 0 (101 files, 2071 passed, 22 skipped); `pnpm build` 0.
- Frame size snapshot rewritten (`K6_WRITE_SIZES=1`), on purpose: spectral 'volume' gathers +1619 B (the AO items), spectral
  transmittance +1148 B, spectral 'frame' gathers +562 B, grey gathers +292 B (`vdbFrameSkyProbe` in the `sh` chunk); grey
  transmittance unchanged. Largest vs cap: spectral transmittance 19327 / 28672, spectral 'volume' gather 37769 / 57344,
  environment reduce 19679 / 24576 (unchanged).
- Defaults unchanged: `ambientOcclusion` false, `multipleScattering` 'neural', `ambientProbe` 'irradiance'.

## DIVERGES rows (proposed, from V155)

| ID | Date | Source | What we do differently | Kind | Revisit when |
|---|---|---|---|---|---|
| V155 | 2026-11-26 | our V125 (K6.Q1 grey) and his spectral sky slots (`buildSpectralNodes` group 0) | K6.Q1a: spectral media get the shadowed ambient per channel. 'frame': with `ambientOcclusion` on, the spectral group 0's sky entry holds the hemispherical visibilities of each channel instead of his sky totals (no extra target; nothing else reads the totals then); octave totals at the voxel's grid albedo. 'volume': per-channel pair items in the march loop. A grey medium equals the grey kernels (12 digits). | quality vs the reference | the default (owner) |
| V156 | 2026-11-26 | three's `getShIrradianceAt` along ± the sky (his 'frame' probe term) | K6.Q1a: internal A/B `ambientProbe: 'radiance'` (π × the hemisphere's mean radiance, band 1 × ½, band 2 dropped). Reference (WDAS + probe): ≤ 2 points on every row; default stays his irradiance. The 'frame' gap (+58 % vs 'volume' +28 %) is the 64³ grid (128³: +27 %). | A/B experiment | delete with the A/B settings |
| V157 | 2026-11-26 | (none: K6.Q1's fixed ring) | K6.Q1a: with `jitter` and `temporal`, the ambient ring turns by the frame's jitter × its azimuth step (pair 0 fixed); static frames unchanged. | temporal sampling | — |

## Follow-ups

- **`shadowResolution` for large sparse clouds** (owner / K6.T): WDAS 'frame' sun-only +22 % → +2 % and + probe +58 % → +27 %
  at 128³. A size-adaptive default (voxels across the cloud) is the candidate; cost in K6.T.
- Delete `ambientProbe` with the other A/B settings (no gain measured).
- The remaining +27 % on WDAS + probe (both lightings, AO on): an anisotropic-probe term (phase-weighted, V128) if a scene
  needs it.
- A spectral reference row (K6.R) for the per-channel AO (the report has no spectral scene yet; K6.4a's third item).
- Ring jitter vs the reference needs a temporal (multi-frame) report mode.

## Traps added to CLAUDE.md

The spectral sky entry holds visibilities with AO on (replaced the K6.Q1 grey-only trap); the 'frame' WDAS bias is the grid
resolution, not a lighting term; `?aoJitter=` probes match twins with jitter on (the IGN noise is f32-safe here).
