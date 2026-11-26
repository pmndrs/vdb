# K6.N: our own 'neural' weights (procedural media only), and the A/B against his and two-sided

Branch `worktree-agent-ab67f6b9bf53ab2ea`. Owner decision D35 (2): train our own MLP weights on procedural media, judge them on held-out
scenes, then the A/B picks the multiple-scattering default (the owner decides; this report does not change any default).

## 1. His trainer (read at `75e7345`, not run)

`scripts/neural-indirect/{media,features,trace,dataset,train,compare}.mjs` and `src/three-openvdb/frame/core/frameNeuralIndirect.js`.

- **Network.** 20 inputs → 32 (ReLU) → 32 (ReLU) → 1, output `exp(min(o, 4))` (the core's `'exp'` transform). The output is the
  scattered light (total − direct) of one light slot at one voxel of the last shadow-grid cascade, per unit of light.
- **Inputs** (`getFeatures`, `getLocalInputs` / `neuralLight` in `frameNeuralChunks.js`): for each of 6 box-mip levels k of the 64³
  grid (extinction, absorption; level k = mean of 2×2×2 of level k−1) `log(1 + σ·h·2^k)` at the voxel, at `h·2^k` toward the light and
  at `h·2^k` away from it (h = largest voxel side), then `log(1 + direct optical depth)` (the shadow march's extinction depth, march
  stopped at diffusion depth 6) and `sqrt(absorption / extinction)` of level 1. No anisotropy input: trained for HG g 0.8.
- **Target** (`trace.mjs`): the scattered fluence / irradiance at the voxel centre: 4π × mean over uniform directions of the radiance
  scattered ≥ 1 time before it arrives. Paths by delta tracking (per-cell majorants of the trilinear grid), albedo weight, Russian
  roulette below 0.1, HG sampling; at each collision next-event light with `HG(cosLight, 0.8) · T_light`, where `T_light` is
  `exp(−depth)` read from a **light-depth grid** (optical depth from every voxel centre to the box exit along the light, quarter-voxel
  steps) interpolated trilinearly. 16–128 paths per sample (stop at 15 % standard error).
- **Media** (`media.mjs`): value-noise cumulus / wisps / layer / shapes, plus **`bunny_cloud.wvdb` and `smoke.wvdb`** (openvdb.org
  derived, CC BY-SA 4.0) at random density gamma. 600 scenes × 1000 voxels, per-scene random albedo (45 % exactly 1), voxel optical
  depth (log-uniform 0.03–30), light direction, box offset. Voxels near the surface weighted up (`exp(−surface depth / 4) + 0.05`).
- **Loss / optimizer** (`train.mjs`): relative squared error `(p − t)² / (p + 0.02)²` (p in the denominator held constant for the
  gradient), batch 256, Adam (β 0.9 / 0.999), cosine learning rate 2e-3 → 5e-5, 40 epochs, He init, `b3 = log(mean target)`, input
  normalisation folded into w1 / b1 for export. Scenes with index % 10 = 0 validate.
- **In our core**: `vdbFrameMlpFromNeural(vdbFrameNeuralWeights)` is the default `renderer.mlp` (`VDBFrameMlp`: layer shapes, `'exp'`
  output, weights row-major with the biases after each W); the core's `gridMips` kernel builds the same box mips, the transmittance
  kernel evaluates the net per voxel and slot. `renderer.mlp = null` runs diffusion.

## 2. What we built (`scripts/k6/neural/`)

| file | what |
|---|---|
| `features.mjs`, `trace.mjs`, `train.mjs`, `dataset.mjs` | ported (MIT header, his commit), see the changes below |
| `media.mjs` | ours: 7 procedural kinds (cumulus, wisps, layer, shape, **blobs**, **plume**, **puffs**); no file reader in it (a test guards that) |
| `eval.mjs` | his and our weights vs the path-traced targets on held-out scenes |
| `target-check.mjs` | next-event variants of the target against each other |
| `ab-table.mjs` | the A/B tables from `reference-report.mjs` JSON |

Changes from his trainer: (1) procedural media only: his two `.wvdb` assets are replaced by `blobs` (compact sharp-edged bodies: the
"solid cloud" case), `plume` (a warped rising column: smoke) and `puffs` (sparse fine detail); (2) **the target's next-event
transmittance is ratio tracking** (`--nee=ratio`, default) instead of his interpolated light-depth grid (§3); (3) command-line seed,
holdout kind, output path; the trainer writes a TypeScript data module.

## 3. The target: ours or his? (numbers)

The target is a **local fluence at a grid voxel**, not a pixel radiance, so the K6.R camera-ray reference
(`vdbReferenceSampleCpu`) is not an oracle for it (a camera ray collects in-scattered radiance along a line; the quantity needed is the
integral over all directions at a point). The comparison is still against our own reference machinery in spirit: the point
estimator is checked against an independent quadrature (analytic single scattering in a uniform cube, 4 σ + 1 %, fail-before checked
with a 10 % phase-weight mutation), and its next-event term against unbiased ratio tracking (the same primitive as the K6.R tracer).

`node scripts/k6/neural/target-check.mjs 24 40 512` (24 scenes of the generator, 960 voxels, 512 paths each, same random stream order):

| | grid (his) / ratio − 1 | march / ratio − 1 |
|---|---|---|
| cumulus | −23.2 % | −0.4 % |
| wisps | −13.0 % | −3.5 % |
| layer | −9.5 % | −4.7 % |
| shape | −15.2 % | +0.5 % |
| blobs | +0.9 % | +1.4 % |
| plume | −12.8 % | +2.8 % |
| puffs | −12.4 % | −6.9 % |
| **all** | **−13.4 %** | **−1.0 %** |

His light-depth grid reads about 13 % **low** (more in dense, sharp-edged media): `exp` of a linearly interpolated depth over a cell
where the density ramps trilinearly overestimates the depth. The same quadrature taken from the collision itself ("march") agrees
with ratio tracking to ≈ 1 %, so the grid interpolation, not the quadrature, is the bias. A network trained on his target learns a
target that is low by that amount: his weights score −16 % bias against the ratio target on our held-out seed (§4). The ratio
target is unbiased by construction, cheaper than "march" (≈ 3 CPU-s per scene against ≈ 7), and noisier per sample (relative standard error
≈ 38 % per voxel at ≤ 128 paths; the loss averages it out, bias is what matters).

## 4. Training

Machine: Apple M5 Pro, 18 cores, 64 GB, macOS 25.6, Node 22.22.2; other agents were running on it, so the wall times are not
measurements of anything but "minutes, not hours". Deterministic: seeds are in the commands; the dataset is a function of
(scene index, `--seed`, `--nee`).

```
pnpm install --frozen-lockfile --ignore-scripts
node scripts/k6/neural/dataset.mjs 3000 12 artifacts/k6-neural/dataset-ratio.bin --nee=ratio           # seed 1: 29 min on 12 workers (1751 s), 3 M samples
node scripts/k6/neural/train.mjs artifacts/k6-neural/dataset-ratio.bin 60                                # seed 12345: 9 min on one thread (542 s), writes the TS module
node scripts/k6/neural/dataset.mjs 300 12 artifacts/k6-neural/fresh-ratio-seed2.bin --seed=2 --nee=ratio # a fresh seed: scenes the trainer never saw
node scripts/k6/neural/eval.mjs artifacts/k6-neural/fresh-ratio-seed2.bin --all
```

Data: 3000 scenes × 1000 voxels (kind weights cumulus .2, wisps .12, layer .08, shape .12, blobs .18, plume .18, puffs .12), per scene
a random albedo, voxel optical depth, light direction and box offset (his distributions). Scenes with index % 10 = 0 validate
(300 k samples). Network 20-32-32-1 (his shape, so the core needs no change), his loss and optimizer, 60 epochs (his 40: the loss
was still falling at 40; 3000 scenes instead of 600). Final training loss 0.3564 (the targets carry ≈ 25 % noise, so ≈ 0.3 is the floor
of the relative squared error; ours is within noise of it).

Fit to the path-traced fluence (relative mean absolute error / relative bias; "noise" is what the target noise alone gives):

| scenes | ours | his weights | diffusion | octaves | target noise |
|---|---|---|---|---|---|
| held-out scenes of the training seed (3000 scenes, index % 10 = 0; 300 k voxels) | 36 % / −1 % | 40 % / −16 % | 69 % / −59 % | 109 % / +13 % | 25 % |
| fresh seed 2 (300 scenes, never trained on; 300 k voxels) | 36 % / −2 % | 40 % / −16 % | 68 % / −58 % | 103 % / +3 % | 25 % |
| fresh seed 2 with a second training seed (777) | 36 % / −2 % | | | | |

Per kind on seed 2: ours −8 % (blobs) to +4 % (cumulus) bias; his −9 % to −22 %. Leave-one-kind-out (trained on the earlier
`--nee=grid` dataset of 1600 scenes, `--holdout-kind=<kind>`): blobs held out 36 % / −9 % (in-distribution 33 %), plume held out
37 % / +1 % (35 %): the network does not depend on a kind it has seen. His weights on the same grid-target data: 36 % / −4 %, so the
two trained nets are equally good fits to HIS target, and his is low only against the unbiased one (§3).

## 5. The A/B on held-out scenes

All rows are the core's frame renderer through the harness pages (`k3-frame` / `k6-scene`, `?ms=<mode>&mlp=his|pmndrs`) at 1280 × 720
(frames 8× down to 160 × 90 in linear light), against the K6.R brute-force path tracer (`core/reference`, level 0, ≤ 512 bounces,
stopped at 4 % mean relative standard error over the volume pixels, cap 200 s; three references stopped by the budget: the two Mie groups at 12.9 / 12.8 % noise and the smoke group at 4.9 %). Headless installed Chrome on the Apple M5 Pro (adapter `apple metal-3`). `ambientOcclusion` is set explicitly in every row
(`--query=ao=0` / `ao=1`); the ambient-free scenes are identical in both runs. Cells are "luminance bias / median relative luminance error"
(percent) over the volume pixels (`reference-report.mjs` semantics: Σ Y / Σ Y_ref − 1, and the median per-pixel |ΔY| / Y_ref; the
reference's own noise floor is ≈ 4 % in the median). Commands:

```
G=frame,mie,mie-rgb,env,wdas,wdas-ambient,smoke,smoke-ambient,bunny-absorbing,bunny-absorbing-ambient,bunny-thin,bunny-thin-ambient,bunny-point,smoke-hemi,wdas-probe,bunny-absorbing-hemi
node scripts/k6/reference-report.mjs --headless --port=5347 --groups=$G --rows='two-sided|neural|diffusion' --query=ao=0 --target=0.04 --budget=200
node scripts/k6/reference-report.mjs --headless --port=5347 --groups=$G --rows='two-sided|neural|diffusion' --query=ao=1 --reuse --target=0.04 --budget=200
node scripts/k6/neural/ab-table.mjs <report.json>
```

**Caveat.** His weights saw the bunny and the smoke assets in training; of the 16 groups only the three WDAS ones are new to them. Ours
saw none of these (procedural media only), so every row is held out for ours. The bunny / smoke rows therefore favour his weights, and
ours still match or beat them. The `env` group has no scene light slot (environment only): two-sided and both networks run the same path
there (identical numbers), so it says nothing about the MLP.

### ambientOcclusion ON (`--query=ao=1`)

| group | diffusion bias / median | two-sided bias / median | his bias / median | ours bias / median |
|---|---|---|---|---|
| frame | -65.7 / 60.9 | 1.9 / 28.8 | -12.1 / 23.7 | -5.9 / 24.6 |
| mie | -45.3 / 47.4 | -2.2 / 31.8 | -4.2 / 24.7 | 3.0 / 23.9 |
| mie-rgb | -45.1 / 47.4 | -1.8 / 32.9 | -3.8 / 24.1 | 3.4 / 23.6 |
| env | -4.0 / 61.8 | 14.2 / 60.9 | 14.2 / 60.9 | 14.2 / 60.9 |
| wdas | -71.3 / 71.7 | -32.0 / 31.8 | -31.6 / 32.3 | -31.5 / 33.0 |
| wdas-ambient | -58.0 / 58.2 | -25.8 / 25.7 | -25.5 / 26.9 | -25.4 / 27.5 |
| smoke | -58.0 / 40.6 | 29.9 / 27.0 | -4.2 / 19.6 | -2.0 / 18.7 |
| smoke-ambient | -42.8 / 26.6 | 21.6 / 18.0 | -2.8 / 13.1 | -1.2 / 12.9 |
| bunny-absorbing | -57.4 / 57.7 | -30.7 / 32.7 | -14.3 / 17.0 | -5.2 / 12.6 |
| bunny-absorbing-ambient | -45.2 / 44.7 | -23.3 / 25.2 | -9.9 / 14.3 | -2.5 / 11.5 |
| bunny-thin | -17.9 / 17.2 | 8.1 / 9.3 | 7.8 / 10.2 | 9.1 / 11.0 |
| bunny-thin-ambient | -10.9 / 10.5 | 4.8 / 6.1 | 4.6 / 6.6 | 5.4 / 7.0 |
| bunny-point | -30.4 / 27.0 | 30.2 / 28.2 | 19.5 / 28.1 | 23.6 / 32.7 |
| smoke-hemi | -48.3 / 31.0 | 25.2 / 21.4 | -3.0 / 15.9 | -1.2 / 15.4 |
| wdas-probe | -53.0 / 53.2 | -23.7 / 23.9 | -23.3 / 25.7 | -23.3 / 25.9 |
| bunny-absorbing-hemi | -49.7 / 49.6 | -25.9 / 27.4 | -11.3 / 14.8 | -3.2 / 11.6 |

| set | diffusion mean |bias| / mean median | two-sided mean |bias| / mean median | his mean |bias| / mean median | ours mean |bias| / mean median |
|---|---|---|---|---|
| all groups with every column | 43.9 / 44.1 | 18.8 / 27.0 | 12.0 / 22.4 | 10.0 / 22.1 |
| bunny / smoke groups (his weights saw these) | 40.1 / 40.2 | 16.9 / 26.9 | 8.6 / 21.0 | 6.1 / 20.5 |
| WDAS groups (new to his weights) | 60.8 / 61.0 | 27.2 / 27.1 | 26.8 / 28.3 | 26.7 / 28.8 |

groups: 16; |bias| smaller for ours than his in 12, his than ours in 3; two-sided smaller than both in 3.



### ambientOcclusion OFF (`--query=ao=0`)

| group | diffusion bias / median | two-sided bias / median | his bias / median | ours bias / median |
|---|---|---|---|---|
| frame | -65.7 / 60.9 | 1.9 / 28.8 | -12.1 / 23.7 | -5.9 / 24.6 |
| mie | -45.3 / 47.4 | -2.2 / 31.8 | -4.2 / 24.7 | 3.0 / 23.9 |
| mie-rgb | -45.1 / 47.4 | -1.8 / 32.9 | -3.8 / 24.1 | 3.4 / 23.6 |
| env | -9.5 / 57.2 | 14.2 / 60.9 | 14.2 / 60.9 | 14.2 / 60.9 |
| wdas | -71.3 / 71.7 | -32.0 / 31.8 | -31.6 / 32.3 | -31.5 / 33.0 |
| wdas-ambient | -58.0 / 58.2 | -25.8 / 25.7 | -25.5 / 26.9 | -25.4 / 27.5 |
| smoke | -58.0 / 40.6 | 29.9 / 27.0 | -4.2 / 19.6 | -2.0 / 18.7 |
| smoke-ambient | -42.8 / 26.6 | 21.6 / 18.1 | -2.8 / 13.1 | -1.2 / 12.8 |
| bunny-absorbing | -57.4 / 57.7 | -30.7 / 32.7 | -14.3 / 17.0 | -5.2 / 12.6 |
| bunny-absorbing-ambient | 30.5 / 28.1 | 52.3 / 50.6 | 65.8 / 64.1 | 73.2 / 72.2 |
| bunny-thin | -17.9 / 17.2 | 8.1 / 9.3 | 7.8 / 10.2 | 9.1 / 11.0 |
| bunny-thin-ambient | -10.9 / 10.5 | 4.8 / 6.1 | 4.6 / 6.6 | 5.5 / 7.0 |
| bunny-point | -30.4 / 27.0 | 30.2 / 28.2 | 19.5 / 28.1 | 23.6 / 32.7 |
| smoke-hemi | -47.2 / 29.5 | 26.2 / 22.7 | -2.0 / 16.5 | -0.1 / 15.6 |
| wdas-probe | -51.3 / 51.4 | -22.0 / 22.4 | -21.6 / 24.5 | -21.6 / 24.8 |
| bunny-absorbing-hemi | -49.7 / 49.3 | -25.9 / 27.9 | -11.3 / 14.2 | -3.2 / 12.1 |

| set | diffusion mean |bias| / mean median | two-sided mean |bias| / mean median | his mean |bias| / mean median | ours mean |bias| / mean median |
|---|---|---|---|---|
| all groups with every column | 43.2 / 42.5 | 20.6 / 28.6 | 15.3 / 25.4 | 14.3 / 25.8 |
| bunny / smoke groups (his weights saw these) | 39.3 / 38.4 | 19.2 / 29.0 | 12.8 / 24.8 | 11.5 / 25.2 |
| WDAS groups (new to his weights) | 60.2 / 60.4 | 26.6 / 26.6 | 26.2 / 27.9 | 26.2 / 28.4 |

groups: 16; |bias| smaller for ours than his in 11, his than ours in 4; two-sided smaller than both in 4.



Reading it (the owner decides; I change no default):

- **Ours vs his.** Ours has the smaller |bias| in 12 of 16 groups (ambientOcclusion on; 11 off), the same mean median error
  (22.1 % against 22.4 %), and a lower mean |bias| (10.0 % against 12.0 %; on the bunny / smoke groups, where his weights are in-sample,
  6.1 % against 8.6 %). The largest differences are on the absorbing bunny (albedo 0.5: −5.2 % against −14.3 %, median 12.6 % against
  17.0 %), which fits §3: his target is low. Ours has the smaller |bias| in the Mie rows (+3.0 % against −4.2 %, medians equal), `frame`
  (−5.9 % against −12.1 %) and the smoke rows. Ours is worse in three groups: `bunny-point` (+23.6 % against +19.5 %, median 32.7 % against
  28.1 %: the point light's 'frame' falloff, which no net models) and the thin bunny with and without ambient (+9.1 % against +7.8 %,
  +5.4 % against +4.6 %). On WDAS the two nets are equal (−31.5 % against −31.6 %).
- **Neural (either) vs two-sided.** Mean |bias| 10.0 % (ours) and 12.0 % (his) against 18.8 % (two-sided), median error 22 % against
  27 % (ambientOcclusion on). Two-sided has the smaller |bias| than both networks in 3 groups (`mie`, `mie-rgb`, `frame`, by 1–4
  points) and loses by 20–30 points on the smoke and the absorbing bunny (+30 % and −31 % against −2 % and −5 %). On the sparse WDAS cloud all
  three modes read −22 to −32 %: that is the 64³ shadow grid, not the network (K6.Q1a found the same for WDAS: `shadowResolution` 128 moved its bias by ≈ 20 points).
- **The ambient groups.** With ambientOcclusion off the absorbing bunny's ambient scene reads +66 / +73 % (neural) and +52 % (two-sided):
  the unshadowed ambient, not the MS mode (on: −2.5 % ours, −9.9 % his, −23.3 % two-sided).
- Not measured (no timings, D25): the cost of the MLP against two-sided.

## 6. Proposed DIVERGES rows (IDs from V160)

| ID | Date | Source | What we do differently | Kind | Revisit when |
|---|---|---|---|---|---|
| V160 | 2026-11-26 | his `scripts/neural-indirect` (75e7345) training set: procedural media plus `bunny_cloud.wvdb` / `smoke.wvdb` | K6.N: our trainer (`scripts/k6/neural/`) trains on procedural media only (7 kinds, `media.mjs`; his two assets replaced by `blobs`, `plume`, `puffs`); 3000 scenes × 1000 voxels, 60 epochs; our weights are `vdbFrameNeuralWeightsProcedural` beside his (`renderer.mlp = vdbFrameMlpFromNeural(…)`). | licence, data | the owner settles his weights' licence (D35 (2)); the default is chosen |
| V161 | 2026-11-26 | his `trace.mjs` next-event transmittance (a light-depth grid, trilinear in depth) | K6.N: the target uses ratio tracking with the cell majorants (`--nee=ratio`); his grid reads ≈ 13 % low in our media (`target-check.mjs`: −13.4 %, march vs ratio −1.0 %). The `grid` option stays for A/B. | fix | never (unbiased); keep `grid` for reproducing his |

## 7. THIRD_PARTY text (for `THIRD_PARTY.md`, `packages/vdb/THIRD_PARTY_NOTICES.md`)

`THIRD_PARTY.md` (new row beside the neural-weights row):

> | mjurczyk/openvdb (pinned `75e7345`, v0.26.0), K6.N neural trainer | MIT | `scripts/k6/neural/{features,trace,dataset,train}.mjs` (derived); `scripts/k6/neural/{media,eval,target-check,ab-table}.mjs`, `packages/vdb/src/tools/frame/neural-weights-procedural.ts` (ours) | The feature, march, path-tracing and training code of his `scripts/neural-indirect/` (`features.mjs`, `trace.mjs`, `dataset.mjs`, `train.mjs`), ported with procedural media only and ratio-tracking next-event targets (DIVERGES V160, V161). `neural-weights-procedural.ts` is data made by that trainer from procedural media generated by our code; it carries no data licence beyond MIT (no openvdb.org, WDAS, EmberGen, Stanford or gallery asset is read in training: `test/tools/frame-neural-train.test.ts` guards the sources). Same header, with the upstream path. |

`packages/vdb/THIRD_PARTY_NOTICES.md`, under "Neural indirect-light weights", after the existing paragraph:

> `src/tools/frame/neural-weights-procedural.ts` holds our own weights for the same network. They were trained by
> `scripts/k6/neural/` (derived from his training scripts, MIT as above) on procedural media generated by our code only; no
> third-party volume data was read. The module is ours (MIT). Select it with
> `renderer.mlp = vdbFrameMlpFromNeural(vdbFrameNeuralWeightsProcedural)` (tools internals); his weights stay the default until D35 (2) is closed.

## 8. Traps (also in CLAUDE.md)

See the K6.N lines at the end of CLAUDE.md.

## 9. What is left

- Nothing blocks the owner's decision. The A/B above supports "neural with our weights" over two-sided on these scenes (mean |bias| 10 %
  against 19 %, median error 22 % against 27 %, ambientOcclusion on), with two-sided winning three groups by a few points; his weights are
  slightly behind ours and in-sample on 13 of the 16 groups. To flip the default (not done): `multipleScattering: 'neural'` plus
  `mlp = vdbFrameMlpFromNeural(vdbFrameNeuralWeightsProcedural)` as the renderer default (`three/frame/renderer.ts` DEFAULT_MLP), K6.11 owns
  the settings defaults.
- To make our weights usable by a host without importing `internal`, the weights need a public name (D32 snapshot `-u`): not done.
- Open model questions the data could answer: a 4th input for anisotropy (g) so one network serves every phase; WDAS-like fine
  sparse media at a higher grid resolution in training (the 64³ grid limits it, §5); a wider hidden layer (the core allows 64): the fit is
  already at the target noise floor, so a better target (more paths, a variance-reduced estimator) comes first.
- The owner's question to his repo (are his weights MIT?) is no longer blocking for a release that ships ours; his stay as the
  default until this is decided.
