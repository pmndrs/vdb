# K6.Q1c handoff: the AO ring fixed (D37 (5)), the phase-weighted distant light in 'volume' lighting (D37 (6), V128)

Task: TASKS K6.Q1c. Tree: main `2b5e7b1`. Quality only, no timings (D25; the machine was loaded by other agents). References
and report frames: installed Chrome headless on the owner's Mac (adapter "apple metal-3") from a frozen copy of the worktree,
the K6.Q1b reference renders reused (`--reuse`; the reference inputs are unchanged since K6.Q1b, the report logs "(reused)");
every e2e on SwiftShader from frozen copies (ports 5349, 5352). The K6.R1 report settings throughout: 160 × 90 references to
3 % noise, frames 1280 × 720 box-downsampled 8×.

## 1. The AO ring is fixed (D37 (5))

K6.Q1a turned the ambient-occlusion ring per frame by the frame's jitter when `jitter` and `temporal` were both on (V150);
K6.11b measured turning vs fixed within 0.1 points against the reference and the owner chose the fixed ring.

- `core/frame/renderer.ts`: no ring phase; `#packLights` always packs `vdbFrameAmbientDirections(sky, pairs)`.
- `tools/frame/ambient.ts`: `vdbFrameAmbientDirections(sky, pairs)` lost its `phase` argument (the twins' turn path).
- `VDBFrameInternalSettings.ambientOcclusionRing` deleted (no deprecation, D35 (5) precedent); the harness `?aoRing=` options
  (k3-frame, k6-scene) and the report's 'ring turning' rows are gone.
- Grey and spectral alike: the directions are one uniform table (`VdbFrameLights.ambientDirections`) that every kernel reads,
  so neither the grey nor the spectral WGSL had its own turn; nothing changed in WGSL for the ring.
- k6-core: `?aoJitter=` stays (jitter + temporal on, the twins take the renderer's jitter). The probe row is now "ring fixed";
  `?mutate=ao-ring` gives the twins K6.Q1a's turned ring (a harness-only rotation, `turnedRing` in `k6-core.ts`) and fails
  (475 / 4608 over).
- Tests: the ring-phase unit test is deleted; the spectral-stack test that used a turned ring turns the fixed ring locally.
- k6-frames gains two frames that use jitter + temporal with AO (`scene-smoke-hemi-{frame,volume}-jitter-temporal`, the 8th
  frame of k6-scene's multi-frame mode): they move against the before-set by design (§5).
- Traps updated (the K6.11 temporal-gate trap and the `aoJitter` trap).

## 2. The phase-weighted distant light (V128 → V185), 'volume' lighting

**Method (K6-Q1b §3, with one change).** The 2N pair totals (the same per-direction values the isotropic visibility averages)
are fitted with a dipole, T(ω) ≈ a + b·ω (least squares over the sides ±ω_k): a is their mean and
b = M⁺ Σ_k (T(ω_k) − T(−ω_k)) ω_k / 2 with M = Σ_k ω_k ω_kᵀ. M⁺ / 2 is constant per frame: the renderer packs it
(`VdbFrameLights.ambientDipole`, 3 rows; row 0's w holds the mode), so the kernel needs one 3-vector u = (M⁺ / 2) r per froxel
and one dot product per pair. With 1–3 pairs M is singular: the pseudo-inverse (Jacobi eigen-decomposition, eigenvalues below
1e-6 of the largest dropped) fits b in the pairs' span (1 pair: along the sky only). For an HG lobe of mean cosine g,
∫ p T dω = a + g b·r (r the view ray; light from along the ray, behind the position, is forward-scattered). Every distant term
(ambient, hemisphere sky / ground, probe ± sky) is scaled by

  f = Σ_lobes w_i clamp(a + g_i b·r, 0, 1) / a   (1 where a ≤ 1e-6)

**The change: each lobe is clamped on its own.** The design's single clamp of the dual lobe's mean cosine
(g_eff = (1 − w) g + w g_back) darkens a backlit position too much: the back lobe's share is lost when the forward lobe's
estimate clamps at 0. Built both (v1: g_eff, v2: per lobe); the per-lobe form is closer on every V128 group (§3), and the unit
test "a dual lobe clamps per lobe" pins it on a step function. Spectral media: each lobe's g per channel (scaled and clamped
±0.99 as `vdbFramePhaseColor`); both Lorenz-Mie modes use the table's asymmetry 0.8526.

**Diffusion totals are not weighted** (setting 'dipole'): with multiple scattering on, the pair totals are the two-sided
isotropic diffusion, whose up / down split is a flux, not an angular radiance, and whose sum is the white-furnace invariant
(V = 1 at albedo 1). Weighting them breaks the furnace and, measured, moves every MS row 5–12 points further from the
reference ('dipole-ms', §3). So 'dipole' weights ms off and octaves; 'dipole-ms' (A/B only) weights everything.

**Setting.** `VDBFrameInternalSettings.ambientPhase: 'isotropic' (default) | 'dipole' | 'dipole-ms'`; harness `?aoPhase=`
(k6-core, k6-scene, k3-frame); report rows "volume core ms off, phase dipole" and "volume core ms on, phase dipole-ms" in every
scene group, "… AO on, phase dipole" in the V128 groups. 'frame' lighting ignores it (§4).

**What landed.**

| piece | where |
|---|---|
| WGSL | `core/frame/wgsl/chunks.ts` chunk `ambientPhase` (`vdbFrameAmbientPhaseFactor`); `wgsl/gather.ts` grey and spectral 'volume' gathers: the dipole sum in the pair items, the factor on V_up / V_down |
| uniforms | `core/frame/uniforms.ts` `VdbFrameLights.ambientDipole` (3 × vec4f, appended) |
| renderer | `#packLights` packs M⁺ / 2 and the mode when pairs run |
| twins | `tools/frame/ambient.ts` `vdbFrameAmbientDipoleMatrix`, `vdbFramePhaseLobes`, `vdbFrameAmbientPhaseFactor` (tools internals); `examples/src/harness/k6-core-twin.ts` grey and spectral gather twins |
| harness | `k6-core.ts` `&aoPhase=`, mutations `ao-phase` (the twins drop the weight), `ao-phase-ms` (the twins weight diffusion totals in mode 1) |
| tests | `test/tools/frame-ambient.test.ts` (+6): exact b for a linear T at 4–8 pairs and a + g b·r = the dense phase-weighted mean; 1–3 pairs (pseudo-inverse); clamps, a ≤ 1e-6, per channel; closer than isotropic on an off-centre homogeneous ball at g 0.8; the lobes; per-lobe clamp beats one clamp on a step |
| probes | `e2e/k6-ambient.spec.ts` (+6 rows, §5) |

## 3. Reference (luminance bias Σ Y / Σ Y_ref − 1 over the volume pixels; median relative error in brackets)

**Distant light alone (the K6.Q1b V128 groups; 'volume' ms off, AO on, vs the single-scattering reference).** The `-iso`
column is the same scene with an isotropic phase: the target level ("the g 0.8 rows should fall to the -iso rows' level").

| group | isotropic (before) | dipole, one clamp (v1) | **dipole, per lobe (v2)** | `-iso` reference row |
|---|---|---|---|---|
| WDAS, ambient | +83.7 % (127.5) | −12.5 % (40.0) | **−0.3 % (19.7)** | +5.4 % (9.8) |
| WDAS, light probe | +96.9 % (115.6) | −10.5 % (51.2) | **+3.2 % (26.7)** | −2.5 % (10.4) |
| smoke, ambient | +22.7 % (24.7) | −14.7 % (19.2) | **−11.9 % (17.8)** | −7.1 % (7.4) |
| absorbing bunny, ambient | +59.9 % (89.7) | −28.9 % (43.2) | **−15.3 % (29.6)** | −12.7 % (14.0) |

The `-iso` rows do not move with the setting (g = 0: f = 1, a check of the kernel). Per lobe, every group lands within 6 points
of its isotropic-phase row: the phase weight is no longer the error; what is left is the pair quadrature (−13 … +5 %) plus the
dipole's truncation (the per-pixel median stays 2–3× the isotropic rows': a dipole cannot hold a sharp occluder edge).

**Full scenes ('volume' lighting; ms off vs the SS reference; MS rows vs the MS reference).**

| group | ms off, isotropic | **ms off, dipole** | sun-only 'volume' ms off (K6.Q1b) | ms on (diffusion), isotropic | ms on, dipole-ms |
|---|---|---|---|---|---|
| WDAS + ambient | +18.9 % (19.0) | **+3.7 % (8.9)** | +4.0 % | −57.4 % | −64.8 % |
| WDAS + probe | +27.8 % (26.8) | **+5.0 % (10.1)** | +4.0 % | −52.5 % | −62.8 % |
| smoke + ambient | +3.5 % (10.5) | **−7.9 % (7.4)** | −5.6 % | −42.5 % | −50.2 % |
| smoke + hemisphere | +0.8 % (7.4) | **−6.8 % (5.7)** | −5.6 % | −48.0 % | −52.8 % |
| absorbing bunny + ambient | +2.8 % (10.2) | **−11.4 % (12.4)** | −9.8 % | −31.2 % | −43.6 % |
| absorbing bunny + hemisphere | −1.5 % (8.8) | **−10.5 % (11.5)** | −9.8 % | −36.8 % | −44.7 % |
| spectral bunny + hemisphere (r / g / b per channel) | +10.4 % (11.7) | **−5.1 % (6.8)** | −1.3 % | −51.7 % | −58.2 % |

Reading it:

1. **The large clouds lose their V128 error**: WDAS + ambient +18.9 → +3.7 %, + probe +27.8 → +5.0 %, each now at the sun-only
   error (+4.0 %). The medians halve or better.
2. **The small media move to their own sun-only error.** Smoke and the absorbing bunny were within 3 points before because the
   isotropic ambient's overshoot cancelled the sun's undershoot (−5.6 %, −9.8 %); with the phase weight their bias is the
   sun's (−7 … −11 %). The bias magnitude grows on these four rows (by 6–9 points) while the error of the distant term itself
   falls (the V128 rows). The median gets better on smoke, a little worse on the absorbing bunny (10.2 → 12.4, 8.8 → 11.5).
3. **'dipole-ms' is worse on every MS row** (5–12 points): diffusion's MS deficit is not a phase question; rejected. The
   setting value stays for the record (an A/B value; delete it with the A/B settings).
4. Mean |bias| over the 11 rows above that the weight touches (4 V128 + 7 scenes, ms off): 29.9 % → 7.4 %.

**Default: left 'isotropic'** (DECISIONS changes only after a clear A/B win; this one wins the term but grows the bias
magnitude of four small-media rows by masking less). See the owner question.

## 4. 'frame' lighting: a design follow-up (K6.Q1d), not built

'volume' is solid (twins, probes, the reference), and the evidence says 'frame' would gain the same: its V128 rows are
+89 … +102 % (side 128, K6.Q1b) with the same isotropic visibility. Not built here because it is a storage and layout change
of the transmittance stack (grey and spectral, twins, probes, the stack-layout traps) that should follow the owner's call on
the 'volume' default. Design:

- **What to store: b, not the weighted visibility.** The dipole b of the pair totals is view-independent, so the
  transmittance pass stores it once per grid voxel (no redraw on camera moves, unlike a phase-weighted visibility, which is why
  K6.Q1 rejected a view-dependent 'frame' form). The gather reads a = (V_up + V_down) / 2 from t3.zw as today, b from the new
  target, and applies f per froxel with its own view ray and phase (the WGSL `vdbFrameAmbientPhaseFactor` as is, given b·r).
- **Where: a fifth target per cascade** in the transmittance stack, rgba16float b.xyz (w spare), +8 B per voxel (grey 48 → 56 B;
  side 64 one cascade 12 → 14 MiB). Spectral: b per channel (9 values): the spectral group 0 has no spare channels; stack them
  along z as the spectral history does (the trap: spectral kernels already write 3 storage textures, default limit 4) or add
  one rgba16float target plus the 4th component of two others. Spectral is the harder half; grey first.
- The transmittance pass already computes each pair's two totals (pair 0 reuses the sky marches): accumulate
  (T⁺ − T⁻) ω_k and multiply by M⁺ / 2 at the end (the same uniform).
- Twins: `tools/frame/stack.ts` (`vdbFrameTransmittanceVoxel3d` returns b), `spectral-stack.ts`; probes: the k6-core frame
  probe compares the new target (`transmittanceStats` over 5 targets) and a gather row with `aoPhase=dipole`; mutation
  `ao-phase` (drop b) must fail both.
- Judge on the V128 groups' 'frame' rows and wdas-ambient / wdas-probe 'frame' rows (adaptive grid on, K6.Q1b), and the MS rows
  ('neural' reads the transmittance; the weight should leave diffusion-class totals alone, as 'dipole').

## 5. Gates

| gate | result |
|---|---|
| `pnpm typecheck` | 0 |
| `pnpm test` | 118 files, 2703 passed, 23 skipped; 7 load failures in unrelated files (timing asserts in `points-hostile`, 30 s timeouts in `compute` / `wvdb-k413`) pass on a rerun with `--testTimeout=120000` (76 / 76) |
| `pnpm docs:check` | 53 blocks match, snippets type-check |
| `pnpm wgsl:size` | 464 variants match (the VDB module is untouched) |
| frame size snapshot | rewritten on purpose (`K6_WRITE_SIZES=1`): every kernel that declares `VdbFrameLights` +29 B minified (the `ambientDipole` member); the four 'volume' gathers (grey, multi-group, spectral, spectral multi-group) +1197 B minified more (the `ambientPhase` chunk and the dipole lines) |
| `e2e/k6-ambient.spec.ts` (SwiftShader) | GATE_AMBIENT |
| `e2e/k6-frames.spec.ts` vs main (before: `git archive 2b5e7b1` + this spec, port 5349; after: frozen copy, 5352) | **106 / 108 identical** (max 0/255); the two new `scene-smoke-hemi-{frame,volume}-jitter-temporal` frames move by design (the ring no longer turns): max 5 / 4 of 255, 0.52 % / 0.56 % of pixels over 1/255 |

New k6-ambient rows (2 f16 ulp, ≤ 0.1 % over; mutations over / compared):

| probe | scatter (4608) | mutations |
|---|---|---|
| gather, 4 pairs, jitter + temporal, ring fixed | 0.72, 0 | ao-ring 475 |
| gather, dipole, single scattering, 4 pairs | 0.72, 0 | ao-phase 575, ao-pairs 583 |
| gather, dipole, 'octaves', 3 pairs, Lorenz-Mie | 0.76, 0 | ao-phase 580 |
| gather, dipole, diffusion (unweighted), 4 pairs | 0.73, 0 | ao 585, ao-phase-ms AMB_MS |
| gather, dipole-ms, diffusion, 2 pairs, sparse | 2.09 (abs floor), 0 | ao-phase 248 |
| spectral gather, dipole, single scattering, 4 pairs | 0.69, 0 | ao-phase 584, ao-grey 585 |
| spectral gather, dipole-ms, diffusion, 4 pairs | 0.67, 0 | ao-phase 584 |

(The per-lobe numbers; the v1 one-clamp kernel passed the same rows with the same maxima.)

## DIVERGES

V184 (the ring fixed; supersedes V150), V185 (the dipole phase weight, 'volume' lighting, internal); V128 updated. (Main had
V182 / V183 when this branch was cut; renumber at the merge if needed.)

## Owner questions

1. **`ambientPhase` default for 'volume' lighting.** Suggestion: make 'dipole' the default (and keep it internal until 'frame'
   has it, K6.Q1d). Pros: it removes the V128 error (distant light alone: mean |bias| 65.8 % → 7.7 % over four groups, each at
   its isotropic-phase level; WDAS + ambient / + probe +19 / +28 % → +4 / +5 %, at the sun-only error); medians improve on 5 of
   7 scenes; no cost worth naming (3 dot products per pair per froxel and one per froxel; no march). Cons: on the small media the
   whole-scene bias magnitude grows by 6–9 points (smoke +3.5 → −7.9 %, absorbing bunny +2.8 → −11.4 %), because the
   isotropic overshoot no longer hides the sun's undershoot; 'volume' lighting's default MS (diffusion) is unaffected, so only
   ms-off / octaves 'volume' scenes change. Test that settles it: the same report with the sun-only undershoot fixed (or a
   sun-free scene set): if the small-media rows then sit at their isotropic-phase level, the dipole is right and the remaining
   bias is the sun's.
2. **Build K6.Q1d ('frame')?** Suggestion: yes, grey first, after (1). Pros: 'frame' is the default lighting and has the same
   +89 … +102 % V128 error; b is view-independent (no extra redraws). Cons: +8 B per grid voxel (17 %), a fifth stack target per
   cascade, and the spectral layout question. Test: §4's probes and the V128 / WDAS 'frame' rows.

## Follow-ups (TASKS)

K6.Q1d ('frame' per-voxel dipole); the `ambientPhase` default (owner); delete 'dipole-ms' with the A/B settings.
