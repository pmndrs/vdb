# K6.Nc: 'neural' and the sibling neural tooling, second fit study ([pmndrs/vdb#3](https://github.com/pmndrs/vdb/issues/3))

Branch `worktree-agent-a7b6dd42c2638ea97`, from main `2b5e7b1`. The first study (K6.Na, [K6-Na.md](K6-Na.md)) read pmndrs/denoiser. The
owner then named two more siblings (D37 (9)): **(a)** the pmndrs/pathtracer neural radiance cache (NRC) and **(b)** the cloud library's
experimental network-rendering branches. Both were read only (`git show <ref>:<path>`, `git grep`, `git log --all`); nothing in either repo
was changed, checked out or run, except that one test here imports `@pathtracer/mlp`'s CPU oracle and WGSL generator from the
pathtracer checkout when it is there (§4).

## 0. What was read (refs)

| sibling | repo, checkout | refs read | files |
|---|---|---|---|
| (a) NRC | pmndrs/pathtracer, `/Users/dex/Developer/pmndrs-pathtracer` (working tree at detached `0e4435b`, the commit pmndrs/gi pins) | `main` = `06488fa` (2026-11-26); `0e4435b`; branch list (`origin/codex/nrc-windows-handoff`, `origin/clouds/volume-reference`, `origin/claude/eager-brown-8uj9yy`); the track-N commits (`git log --all --grep 'nrc\|neural\|mlp'`) | `packages/mlp/{README.md,package.json}`, `packages/mlp/src/{config,encoding,oracle,wgsl,wgsl-forward,engine,index}.ts`, `bench/nrc/{hash-grid,spatial-features,cache}.ts`, `docs/internal/design/{nrc,volumes-restir-nrc}.md`, `bench/results/NRC.md`, `TASKS.md` (N rows, S20), `DECISIONS.md` (D10, D28, D29), `LICENSE`. `packages/mlp` code is identical at `0e4435b` and `06488fa` (only doc links moved). |
| (a′) a consumer of (a) | pmndrs/gi, `/Users/dex/Developer/webgpu-ddgi` | `origin/main` = `8c880d9e` (2026-11-28), `3dc4176b` | `packages/cache/package.json` (`linkedSiblings`), `packages/cache/src/neural-cache/{mlp-loader,mlp-api-check,neural-tail}.ts` |
| (b) clouds | pmndrs/clouds, `/Users/dex/Developer/WebGPUClouds` | `main` `f1e9105` (`ace2838b`: the research record); `neural/bake-experiments` = `origin/neural/bake-experiments` `f4ec192a`; `claude/webgpu-neural-clouds-5e8baf`, tag `neural-halort-port-end` = `d91d9e64` | `docs/research/neural/{README,RESULTS}.md` (main), `docs/neural/{ENGINE,FOLLOWUPS,BAKE}.md`, `work/logs/{B0,N3b,N5-12}.md`, `experiments/neural/engine/runtime.ts` (branch), `LICENSE` |
| this repo | WebVDB | `git log --all` (incl. `pmndrs/k6-s2-stochastic-froxel`) | no other network experiment than K3.4 / K6.3 / K6.N / K6.Na |

Also checked: `/Users/dex/Developer/cloud-research` is a folder of PDFs and a built study (no git, no network code).

## 1. (a) The pmndrs/pathtracer neural radiance cache

### What the network does

Müller et al. 2021 as VkNRC builds it (their D29: VkNRC's design, their own WGSL). Per frame, a path vertex's outgoing radiance is
**learned online** from that frame's training paths and queried where a path stops (area-spread rule), so paths end after one or two
bounces. The cache is **per scene and per session**: it starts empty (zero output layer) and adapts within ~8 frames (NRC.md Q2). Their
result: 3.2–4.3× lower relMSE than plain path tracing per frame on Sponza; a SHaRC-like hash grid ties it on diffuse scenes (Q4).

### WGSL inference layout, and training on the GPU

`@pathtracer/mlp` (`packages/mlp`), WGSL prefix `nn`:

- **Network**: `{ inputs, width, hidden, outputs }` (inputs and width multiples of 8, one width for every hidden layer), ReLU hidden
  layers, linear last layer, **no biases** (the encoding carries constant-one inputs). Default 64 → 5 × 64 → 3 (20 672 weights),
  compact 64 → 3 × 32 → 3.
- **Inference kernels are dispatches**, not functions: `nnInfer` (one record per thread, `@workgroup_size(64)`, each layer's weights
  staged into workgroup memory between `workgroupBarrier()`s), `inferTiledWgsl` (256 threads per 48–64 records, the default) and the
  subgroup-matrix arm `wgsl-sgm` (`chromium-experimental-subgroup-matrix`, a Chrome flag; Apple 8×8×8 f16→f16 / f32→f32 tiles). The output
  is a `vec3f` (≤ 3 outputs).
- **Training on the GPU, online**: `nnPrepare` (GPU record count → indirect args, no readback), `nnForwardBackward` (per record, tiled or
  sgm), `nnWeightGradient` (per weight, or the tiled split-K pass + reduce; no float atomics), `nnAdam` (f32 master weights, Adam, EMA
  copy for inference), `nnSync`. Loss: relative L2 on luminance, `|y − t|² / (lum(y)² + 0.01)`, **3 outputs required**
  (`trainWgsl` throws otherwise).
- Correctness: an f64 CPU oracle (`oracle.ts`: forward, backward, loss, Adam, EMA) every arm is checked against (their `mlp-check`).

### Encodings

`encoding.ts`: a **triangle-wave frequency** encoding (12 octaves per axis, the paper's fast variant of sin), the **quartic one-blob**
(4 bins), and the NRC layout (position 36, direction / normal / roughness one-blobs 20, reflectances 6, two constant ones 2 = 64).
`bench/nrc/spatial-features.ts`: an optional **trainable multi-resolution grid** (6 levels 4³–128³, 2 features each, dense then hashed past
2^log2Entries, trilinear) that replaces the position frequencies. `bench/nrc/hash-grid.ts` is the non-neural competitor (a SHaRC-style
world-space hash cache). All of these encode **scene position**: they are what makes a per-scene cache, not features of the medium.

### How it binds and records

`MlpEngine` (`engine.ts`) owns ~15 buffers (master, train and use weights, EMA, moments, gradient, partials, activations, deltas, loss,
state, indirect args, params) and `layout: 'auto'` pipelines. Group 0 is the engine's (`nnWeights` storage at binding 0, `nnState`
storage at 1, params uniform at 2); the caller supplies group 1 as WGSL hooks (`nnInferLoad` / `nnInferStore` / `nnTrainLoad`) and their
bindings. `encodeInfer` / `encodeTrainStep` **record into the caller's encoder** (good), but `uploadWeights` makes and submits its own
encoder (`nn sync`). The infer kernel binds 2 storage buffers of its own plus the caller's hook buffers (typically 2 more).

### Weight format and precision

A flat `Float32Array`, layer by layer, row-major [out][in], no biases, no header (`NrcCache.loadWeights(weights)`; the shape is
configuration). f32 master weights; inference and training in f32 or f16 (`T`), f32 sums (the f16 portable arm sums each vec4 dot product
in f16 first). No file container.

### Licence and packaging

MIT, Copyright (c) 2026 Dennis Smolek (repo `LICENSE`; `packages/mlp/package.json` `"license": "MIT"`). VkNRC has **no licence** and is a
design reference only (their D29: no VkNRC text in their code); tiny-cuda-nn is BSD-3 (cited). `@pathtracer/mlp` is **private**
(`"private": true`, `exports: ./src/index.ts`: TypeScript source, no build) and the NRC itself (`NrcCache`) lives in `bench/nrc/`, not
in a package (their S20: "once its API leaves `bench/nrc` / `packages/mlp` (both private today)"). The one consumer outside the repo is
pmndrs/gi's **neural tail** (`@pmndrs/gi-cache`): `linkedSiblings` pins `@pathtracer/mlp` at `0e4435b` from a sibling clone, loads it
with a dynamic `import()` (`mlp-loader.ts`), keeps hand-written declarations for checkouts without it (`mlp-unavailable.d.ts`,
`mlp-api-check.ts`), and the published `@pmndrs/gi/radiance-cache` leaves the import external. That is the precedent if a pmndrs
library wants it at run time.

## 2. (b) The cloud library's network experiments

### What the network does

pmndrs/clouds ran a closed experiment (2026-11-23 to 25, main `ace2838b` "experiment closed, code stays on branches"): a web port of
**HaloRT** (Leonard & Westermann, arXiv 2611.12103), an **image-space** network: a WGSL feature pass makes 14 per-pixel channels
(optical depth, crossing depths, sun shadow and ~8 sky samples at one point per pixel) on the VDB density, then a **U-Net with block
attention** (1 151 649 parameters, 71 conv3×3, 22 conv1×1, 67 RMSNorms, 5 attention blocks) runs six times (sun and sky, RGB) to give
the cloud's radiance. Then a budget study (B0) of a per-pixel tiny MLP for a baked imposter, using **pmndrs/pathtracer's
`@pathtracer/mlp`** (its `mlp-timing` page) — the bake itself was not built.

### WGSL inference layout, training

`experiments/neural/engine/` (branch `claude/webgpu-neural-clouds-5e8baf`): conv / attention / graph / runtime, "pattern ported from the
Denoiser repo" (`denoiser-wgsl` at `7014384`): mat4x4-packed weights (`packWeights` / `packBias`, the layout our `mlpRuntime: 'packed'`
follows, K6.Na), fused conv + bias + RMSNorm + SiLU epilogues, **82 dispatches in one compute pass and one submit** per frame, batch of 6
passes. Output equal to ONNX Runtime's to 1.3e-6 (f32). **No training**: HaloRT's weights (MIT) via a `pt_to_tza.py` converter.

### Encodings, binding, weights, licence

- Encodings: HaloRT's own per-pixel features (depth encodings, crossings), no positional encoding.
- Binding: its own encoder and submit per run, slot buffers, its own bind groups (the denoiser runtime's pattern).
- Weights: `.tza` (denoiser's container), f32 or f16 (`shader-f16`, f16 sums per chunk), two weight sets in one buffer per layer.
- Licence: pmndrs/clouds is MIT ("Copyright (c) 2026 pmndrs contributors"), workspace `private`; the engine is experimental code on
  branches only (not on main, not published). HaloRT code / weights are MIT; the authors' web demo has no licence (kept local).

### Where the time went ("the sampling was slow")

Their own numbers (M5 Pro, headless Chrome, a loaded machine, their logs; not ours):

| what | cost |
|---|---|
| HaloRT network, 512², ONNX Runtime Web | 135–180 ms |
| HaloRT network, 512², their WGSL engine | 36 ms f32, 23 ms f16 (12.6 GMAC per pass, 78 % of it in the full-resolution 16-channel layers) |
| whole HaloRT frame, 512², f16 | 33 ms (the feature pass "small (≤ ~10 ms)") |
| their raymarch, same cloud | ≈ 1 ms |
| B0: tiny MLP 64 → 5 × 64 → 3 per pixel, `@pathtracer/mlp` portable | 1.69 / 6.33 / 24.58 ms (f16) and 1.50 / 9.04 / 39.19 ms (f32) at 128² / 256² / 512² |

Their findings (RESULTS.md): the time is **network arithmetic per pixel**, not sampling and not dispatch overhead (≈ 6.5 µs a
dispatch; "the gap is kernel speed"); a network is faster only where it replaces something expensive; even a tiny per-pixel MLP
(≥ ~1 000 multiplies plus feature reads per pixel) only ties a raymarch. Their FOLLOWUPS file lists our `'neural'` mode as family C
("network inside the march"), the one direction that attacks the light march.

**The lesson for us.** Our network is not per pixel: it runs once per **occupied** shadow-grid cell and light slot (the transmittance
kernel skips cells with no density), and only when the transmittance redraws. By multiply count (an estimate, not a timing): 1 696
multiply-adds per query × 64³ cells = 0.44 GMAC per slot per redraw, about 1/170 of one HaloRT frame. At B0's measured multiply rate
(5.4 GMAC in 24.6 ms f16, 39.2 ms f32) that would be 2–3 ms per slot per redraw for a fully occupied 64³ grid, and **8× that at
`shadowResolution` 128 (16–26 ms)**, where `jitter` redraws it every frame (the K6.Q1b class). So: keep the network on the grid (never
per froxel or pixel), keep it near 2 k multiplies, and have K6.T time 'neural' at 64³ **and** 128³ with jitter on, with the network
split from its 18 feature taps (a constant-output arm), because a feature tap may cost as much as the multiplies.

## 3. The fit with our MLP

Ours: 20 → 32 → 32 → 1 with biases, ReLU, `exp(min(o, 4))`, a WGSL **function** (`vdbFrameMlpForward`) called per thread inside the
transmittance kernel's `if (near > 0.0)` branch for each slot, f32, weights the kernel's one storage buffer (D10), on the host encoder
(D34), engine-free core with no new dependency (D20). Trained offline on CPU (`scripts/k6/neural/`, procedural media, path-traced
fluence) to generalise to any volume.

| question | (a) NRC (`@pathtracer/mlp`) | (b) clouds (HaloRT engine) |
|---|---|---|
| **Runtime dependency** | **No.** Private TS source, unpublished; dispatch-shaped kernels with the engine's own group 0 and ~15 buffers (D10: its infer kernel binds ≥ 4 storage buffers with the hooks); `uploadWeights` submits on its own (D34); `nn` symbols (D10's `vdb` prefix). Running it would split our pass in three (features to a 64³ × 20 × 4 B ≈ 21 MB buffer per slot, the network, a combine), as K6.Na §1 found for denoiser. | **No.** Image contract (conv / U-Net), its own submit, experimental branch code. Same verdict as denoiser's runtime (K6-Na §1). |
| **Share WGSL layer code** | **No.** Every kernel stages weights through workgroup memory between barriers, so it cannot be called from our per-thread function: a barrier inside our `near > 0` branch is non-uniform control flow (a WGSL error), and its layer helpers (`layerHelpers`, `forwardBody`) are not exported. Its layer math (row-major vec4 rows) is what our generic evaluator already does; staging the weights in workgroup memory would be a third evaluator whose only claim is speed (K6.T). | **No.** Conv3×3 / 1×1 and attention only; its packing is denoiser's, which our `mlpRuntime: 'packed'` already follows (V180). |
| **Share training** | **Not now.** Its trainer is online and per scene; ours must generalise across volumes, and our CPU trainer is minutes (9 min for 60 epochs; the dataset, 29 min on 12 workers, is the cost, and the GPU trainer does not make targets). To train our net its loss needs 1 output, our relative error on an `exp` output, and our weights its convention (the last we built, §4). | Nothing to share (no trainer). |
| **Encodings for K6.Nb** | **They do not address grid independence.** Frequency, one-blob and the trainable spatial grid encode scene position, which makes a per-scene cache; our features must stay physical (optical depths). Grid independence is K6-Na §7's own fix (world-unit reaches, or a side input + retrain). The one-blob is a reasonable encoding for a new scalar input (g for anisotropy): an A/B inside the K6.Nb retrain, 8 lines, no dependency. | Not applicable. |
| **Share a weight format** | **Yes, exactly, as a convention.** Its flat bias-free array holds our network with one constant-one input and one carrier unit per hidden layer (§4): 24 → 2 × 40 → 1, 2 600 weights. Their own oracle reproduces our outputs to 1e-12. | `.tza` (denoiser's): K6.Na already reads and writes it (V181). |

**Verdict: no runtime or WGSL fit with either sibling without upstream changes.** What fits is a weight convention (built, §4) and three
training ideas for K6.Nb (§6). The cloud experiment is a negative result with a useful lesson about cost (§2).

### f16: would our net survive the NRC's f16 arms? (numbers)

`scratchpad/neural-3b/f16.mjs`: our network embedded in the NRC convention, evaluated on the 40 000 inputs of a fresh dataset
(`dataset.mjs 40 6 … --seed=3 --nee=ratio`, procedural scenes the trainer never saw) with the NRC's f16 arms emulated on the CPU:

| arm | relative error of exp(o) vs f64: median / p99 / max | > 1 f16 ulp (2^-11) | > 4 ulp | > 16 ulp |
|---|---|---|---|---|
| f16 weights, inputs and activations, exact sums | 4.3e-4 / 8.7e-3 / 2.7e-2 | 45.8 % | 12.0 % | 1.4 % |
| the same plus each vec4 dot product in f16 (the portable f16 arm) | 6.2e-4 / 1.2e-2 / 4.3e-2 | 57.9 % | 18.5 % | 3.2 % |

The largest hidden activation on these inputs is 22.0 (relu6 would still clip it). f16 moves the output by about 1 % at p99: invisible
next to the network's fit error (36 % per voxel, K6-N §4) and the reference noise, but well over the core's ≤ 2 f16 ulp twin gate, so an
f16 evaluator would need a justified bound (median 1 ulp, p99 ~20 ulp). This answers half of K6.Na's Q4 (f16 storage): viable for
quality, worth building only if K6.T shows the MLP costs.

## 4. What was built: the NRC weight convention (tools internals)

The only clear fit, and it keeps D10 / D20 / D34 trivially (tools only, no renderer change, no dependency).

| piece | where | what |
|---|---|---|
| bridge | `packages/vdb/src/tools/frame/mlp-nn.ts` (re-exported from `tools/internal.ts`) | `vdbFrameMlpToNn(mlp)` → `{ shape: { inputs, width, hidden, outputs }, weights, features, ones }` in `@pathtracer/mlp`'s convention: input `features` (20) is a constant 1, the last unit of each hidden layer a carrier (weight 1 from the one / the previous carrier), each layer's biases its weights from it. `vdbFrameMlpFromNn(nn, { levels, output })` is the inverse for any network in the convention (the first layer's constant-one columns summed into its biases; then every hidden unit with no input weights folded into the next layer's biases as relu(bias) and removed). `vdbFrameMlpNnInputs`, `vdbFrameMlpNnWeightCount`. Our net: 24 → 2 × 40 → 1, 2 600 weights; ours → convention → ours is **bit-identical**. The output transform ('exp') stays an option, as in the `.tza` map. |
| tests | `packages/vdb/test/tools/frame-mlp-nn.test.ts` (7) | shape and carrier positions; the convention's f64 forward (as their oracle documents it) = ours to < 1e-12 relative on 2 000 inputs; bit-identical round trips (ours and three random shapes); a network trained in the convention (two constant ones at 22, 23, padding, no carriers) read back to < 1e-6 (one f32 rounding of the two ones' sum); data mutations (carrier dropped, biases moved to a dead unit) move > 50 % of outputs; bad inputs throw `VDBToolsError`; and, **when a pmndrs/pathtracer checkout sits beside the repo**, `@pathtracer/mlp`'s own `validateShape`, `weightCount`, oracle `forward` (< 1e-12) and `inferWgsl` (generates `NN_IN = 24u`, `NN_W = 40u`) take the network as is, and a transposed first layer fails through their oracle. |

Fail-before, on the converter's code (`scratchpad/neural-3b/mutate.sh`, each run then restored; log `mutate.log`):

| mutation of `mlp-nn.ts` | tests failed (of 7) |
|---|---|
| carrier weight 0.5 instead of 1 | 4 |
| a layer's bias written transposed (`one * nIn + j`) | 3 |
| the constant-one columns not summed into the first bias (`FromNn`) | 2 |
| a folded unit's constant without the ReLU (`bias + 0.25`) | 1 |

What this enables: a network from the NRC trainer (once it can train our output, §6) loads into our renderer with `vdbFrameMlpFromNn`,
and our network runs in their engine (any arm, including subgroup matrices) and their oracle unchanged, which is the experiment for
§5's "does sgm help our net" without touching our core.

### The issue's acceptance, for this prototype

- **CPU-twin parity with a fail-before mutation**: the bridge's twin is the sibling's own f64 oracle (< 1e-12) plus ours; four code
  mutations and two data mutations fail it (above).
- **Non-'neural' frames byte-identical in `e2e/k6-frames.spec.ts`**: not run. No file under `packages/vdb/src/core` or
  `src/three` changed and the default network is the same object, so every frame (neural or not) is unchanged by construction; the
  examples only see two more exports of `tools/internal.js`. A run would spend a before-set and an after-set on a loaded machine to show
  nothing.
- **'neural' bias ≤ 10.0 % mean |bias| on the 16 K6.N groups**: unchanged network, unchanged renderer, so K6.Na's 10.0 % stands.

`pnpm build`, `pnpm typecheck`, and `npx vitest run` of `frame-mlp-nn`, `frame-mlp-packed`, `api-surface`, `boundaries` pass (35 tests).

## 5. Recommendation

1. **No runtime dependency on `@pathtracer/mlp` or the clouds engine in 0.x** (same as D37 (9) for denoiser). Pros: D10 / D20 / D34 hold;
   no unpublished sibling at run time; a 1.7k-MAC function stays a function. Cons: no shared kernel speed-ups (sgm) for free. **Test
   that settles it:** K6.T times the transmittance pass with 'neural' against two-sided at 64³ and 128³, jitter on, Metal and one
   non-Apple GPU, with the network split from its feature taps. If the network is a small share of the pass, nothing a sibling kernel
   can win matters; if it dominates at 128³, the next step is an A/B of a split pass on `@pathtracer/mlp`'s sgm arm, fed through the
   bridge (exact), against an in-kernel f16 evaluator.
2. **Keep the bias-free bridge as a tools internal** (like the `.tza` helpers, D37 (9)) until a second network exists. Pros: exact,
   tested against the sibling's own oracle; the shared format is a convention, not a dependency. Cons: one more map to keep in step
   (the tests cover it). **Settles it:** the first network trained outside our trainer (NRC GPU trainer or other) that we want to load.
3. **Do not move training to the NRC trainer now.** Take ideas instead, each a K6.Nb A/B on `eval.mjs`'s held-out seed (§6).
4. **Keep K6.Nb's grid-independence fix ours** (K6-Na §7: world-unit reaches): the NRC's encodings are positional and cannot give it.

## 5a. Questions for the owner

1. **File the upstream proposals (§7) on pmndrs/pathtracer?** Suggestion: yes, as one low-priority issue, after K6.T. Pro: the NRC
   engine is the closest pmndrs code to a shared small-MLP runtime (pmndrs/gi already uses it); a function emitter would let us inline
   its layer code and its sgm arm. Con: none of it is needed for 0.x; asking before K6.T asks for work we may not use. Settles it: K6.T
   showing the network is a large share of the transmittance pass at 128³ with jitter.
2. **A per-scene online fine-tune of 'neural' (the NRC idea applied to us)?** Suggestion: research only, CPU first. Pro: a network
   fine-tuned on targets traced in the full-resolution medium, with inputs from the 64³ grid, could learn some of what the grid loses
   (WDAS reads −31 % in every MS mode, K6-N §5). Con: warm-up and flicker (NRC: ~8 frames; behind denoisers ~150), a GPU fluence
   estimator per cell, and an MlpEngine-style buffer set that breaks D10. Settles it: a CPU experiment with our trainer: fine-tune on
   ~1 000 WDAS voxels' targets from the full grid, evaluate on held-out WDAS voxels and on the reference report's WDAS groups (views
   and lights not used in the fine-tune); worth a GPU design only if WDAS moves by ≥ 10 points.
3. **An f16 'neural' evaluator (`shader-f16` fast path)?** Suggestion: not until K6.T. Pro: §3's numbers say quality allows it (p99
   ~1 %). Con: a justified twin bound instead of 2 ulp; no speed claim without timing. Settles it: K6.T, then a frames + reference-report
   A/B (expect bias within 0.1 point).

## 6. For K6.Nb (from the NRC, none needs the sibling's code)

- **Weight EMA in the offline trainer** (the NRC infers with an EMA of the weights, decay 0.99): average the last epochs' weights before
  export. A/B: `eval.mjs` on the fresh seed, relative error and bias, with and without; cheap (one retrain).
- **One-blob encoding of a new scalar input** (g for anisotropy, or the grid side if K6-Na §7 option 2 is chosen): 4 bins (`encoding.ts`
  `oneBlob`, the quartic kernel) against the raw scalar, in the same retrain.
- **A lower-noise target, NRC style**: the NRC trains on short suffixes ended by its own prediction (self-training: less variance,
  some bias). Our analogue: end the target's paths at the second collision with the current network's scattered light there
  (bootstrapping between training rounds). Needs care (our output is isotropic fluence, not radiance toward −ω). A/B against more paths
  per sample at equal CPU time, judged on the fresh seed's bias.
- Not from the NRC: K6-Na §7 for grid independence; this study's cost note (§2) for K6.T.

## 7. Upstream proposals (not made: both repos are read-only)

For **pmndrs/pathtracer `@pathtracer/mlp`**, in order of value to us (none blocks us):

1. **A forward-pass function emitter**: `forwardFunctionWgsl({ shape, precision, prefix, weights: { name } })` returning
   `fn <prefix>Forward(x: array<vec4<T>, N>) -> array<vec4f, M>` that reads weights straight from a storage binding the caller declares
   (no `workgroupBarrier`, so it can sit in non-uniform control flow; a staged variant only for callers that guarantee uniformity).
2. **A symbol prefix and caller-chosen bindings** for every generator (today `nn*` names and group 0 bindings 0–2 are fixed).
3. **Scalar outputs and an output transform** in forward and loss: `outputs: 1`, `output: 'linear' | 'exp'`, and a loss hook (ours is
   `(p − t)² / (sg(p) + 0.02)²` on exp(o)); today `trainWgsl` needs 3 outputs and the luminance loss.
4. **`encodeUpload(encoder, weights)`** beside `uploadWeights` (which submits its own encoder), for hosts that record everything into
   one encoder.
5. **A self-describing weight file** (or adopt `.tza` with the dense names K6.Na uses) and an optional bias flag; our bridge (§4)
   documents the constant-one convention either way.
6. **Publishing**: if more pmndrs libraries consume it, a built, published package (pmndrs/gi already pins it as a linked sibling).

For **pmndrs/clouds**: none. The experiment is closed; if it reopens on its family C (a network inside the march), our features,
trainer and weights (`scripts/k6/neural/`, MIT) are the starting point, and §2's cost note applies to them too.

## 8. Traps

In CLAUDE.md (K6.Nc lines at the end).
