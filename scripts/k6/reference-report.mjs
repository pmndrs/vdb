// K6.R / K6.R1: quality report of the froxel renderer (K6.8: the core frames through `./three`; the TSL path is deleted) and his
// K3 captures against the reference renderer (a brute-force volumetric path tracer, core/reference; design
// K6-core-renderer.md §7.4). REPORT ONLY: nothing gates on it, and no timing is measured.
//
//   node scripts/k6/reference-report.mjs                    # starts its own Vite server, Chrome on the real GPU (headed)
//   options: --groups=volume,frame,...  (default: all; see GROUP_DEFS) --w=160 --h=90 (reference size; frames render at
//            --scale× and are box-downsampled in linear light) --scale=8 (his 1280 × 720 at 160 × 90)
//            --target=0.03 (convergence stop: mean relative standard error of Y over the volume pixels)
//            --budget=300 (seconds: safety cap per reference render) --check=32 --spp=1 --maxBounces=512
//            --port=5461 --base=http://127.0.0.1:5173 (reuse a server) --headless (installed Chrome without a window;
//            check the adapter it reports) --swiftshader (bundled Chromium + SwiftShader: script debugging, never a result)
//
// K6.3a: groups v80-phase-g-low, v80-octave-extinction-high, v80-octaves-1 (his Step 4 strict-bar failures through the
// core at 640 × 360) and a 'multi-scale' σ = 2^k row (`msSigma=paper`) in every scene group. K6.10a: every row renders the
// core (the pages ignore `?backend=`); the TSL rows and the core-vs-TSL (V80) pixel columns are gone. The core-vs-TSL
// numbers of earlier runs are in the K6.R1 / K6.3 / K6.3a / K6.Q1 handoffs.
//
// Needs (local, git-ignored): fixtures/k3/bunny_cloud.wvdb, fixtures/k3-reference/75e7345/ (his captures + manifest),
// fixtures/wdas/wdas_cloud_quarter_fp8.nvdb, fixtures/vdb-samples/smoke.vdb. Groups whose fixtures are missing are skipped.
// Writes artifacts/k6-reference/<group>/*.png (reference, frames, his captures, heat maps) and report.{json,md}.
//
// Per comparison (a frame vs the reference with multiple scattering, and vs single scattering):
//   PSNR (sRGB 8-bit, whole frame), pixels > 8/255, SSIM (K6.R1: luma, 11 × 11 Gaussian; whole frame and the volume's
//   rectangle), and over the volume pixels (reference alpha > 0.02): RMSE of the linear composite, median and mean
//   relative luminance error |Y_A − Y_ref| / Y_ref (Y_ref > 1e-3), and the luminance bias Σ Y_A / Σ Y_ref − 1, Y the
//   luminance of the volume's radiance recovered from the composite by inverting his sRGB blend at the reference's
//   coverage (K6.R1; K6.R subtracted the background linearly, which goes negative where a volume is darker than it).
//   Silhouette band (K6.R1, V80): pixels within one pixel of the reference's alpha = 0.02 contour (both sides): the
//   composite's error Σ|A − R| / Σ|R − background| there (lower is better) and its RMSE. Overlap (K6.R1, V79): pixels
//   where two volumes each reach optical depth 0.1: bias and median error there. The reference's own noise is listed:
//   errors below it are not resolved.
//
// K6.Q1b: spectral scene groups (the reference renders each channel; its composite and the radiance it inverts use each
// channel's own coverage); V128 groups (distant light only, the scene phase and an isotropic phase: single scattering
// only); multi-frame rows (`frames=&jitter=1&temporal=1` or `&average=`: the jittered estimator, k6-scene); the
// size-adaptive shadow grid A/B through `--query=shadowResMax=…` or `shadowRes=…` on the scene groups.
//
// K6.Q1c: 'volume' rows with the distant light's dipole phase weight (`aoPhase=dipole`, ms off; `dipole-ms`, ms on) in every
// scene group and the V128 groups; the multi-frame rows keep the fixed AO ring (D37 (5)).
//
// K7.4 (issue #1): `--groups=k74-…` (not in the default list) measure his v0.30 defaults (dual-lobe phase, spectral on) as
// setting changes: each phase against its own reference, spectral against the grey reference.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { decodePng, encodePng } from "../k3/png.mjs";
import { diffHeatmap } from "../k3/metrics.mjs";
import { silhouetteMask, ssim } from "./metrics.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.join(ROOT, "artifacts", "k6-reference");
const CAPTURES = path.join(ROOT, "fixtures", "k3-reference", "75e7345");
const arg = (name, fallback) => {
  const hit = process.argv.find((a) => a === `--${name}` || a.startsWith(`--${name}=`));
  if (!hit) return fallback;
  return hit.includes("=") ? hit.slice(hit.indexOf("=") + 1) : true;
};
const SWIFT = arg("swiftshader", false) === true;
const W = Number(arg("w", SWIFT ? 32 : 160));
const H = Number(arg("h", SWIFT ? 18 : 90));
const SCALE = Number(arg("scale", SWIFT ? 4 : 8));
const TARGET = Number(arg("target", SWIFT ? 0.5 : 0.03));
const BUDGET = Number(arg("budget", SWIFT ? 10 : 300));
const CHECK = Number(arg("check", 32));
const SPP = Number(arg("spp", 1));
const MAX_BOUNCES = Number(arg("maxBounces", 512));
const PORT = Number(arg("port", 5461));
let BASE = arg("base", "");
// K6.Q1: --reuse keeps a group's reference renders (artifacts/k6-reference/<group>/reference-*.json, same query) across
// runs; --rows=<regex> renders only the frame rows whose label matches (A/B iterations on one setting).
const REUSE = arg("reuse", false) === true;
const ROWS = arg("rows", "") ? new RegExp(String(arg("rows", "")), "i") : null;
// K6.Q1: --query=<a=b&c=d> appends to every frame row's query (core settings A/Bs, e.g. --query=ao=1)
const EXTRA_QUERY = arg("query", "") ? `&${String(arg("query", ""))}` : "";

const need = {
  bunny: path.join(ROOT, "fixtures", "k3", "bunny_cloud.wvdb"),
  captures: path.join(CAPTURES, "manifest.json"),
  wdas: path.join(ROOT, "fixtures", "wdas", "wdas_cloud_quarter_fp8.nvdb"),
  smoke: path.join(ROOT, "fixtures", "vdb-samples", "smoke.vdb"),
};
if (!existsSync(need.bunny)) {
  console.error("needs fixtures/k3/bunny_cloud.wvdb (copy it from the main checkout)");
  process.exit(2);
}
const MANIFEST = existsSync(need.captures) ? JSON.parse(readFileSync(need.captures, "utf8")) : null;

// --- the groups ------------------------------------------------------------------------------------------------
// 'his' groups: a K3 scenario of his capture manifest (his bunny, camera, box, sun), frames from k3-frame, his captures.
// 'scene' groups (K6.R1): a scene of examples/src/harness/k6-scenes.ts, frames from k6-scene, no captures.
// Rows: { label, page, query, lighting, capture?, vsSingle? } or { label, slot: true } (a row with no frame yet).
// 'volume' lighting uses the volume's phase (k3-frame: his frame.js dual-lobe 0.8 / −0.3 / 0.2); 'frame' lighting the
// renderer's settings phase (read from the frame's `settings`; k3-frame sets 0.8 / −0.3 / 0.2 too).
const VOLUME_PHASE = [0.8, -0.3, 0.2];
// K6.3: the MS modes ('multi-scale' both edge modes, the A/B for V75; two-sided / octaves / neural)
// `captures`: his capture of a mode, when he has one (keyed by the `ms` value).
const coreMsRows = (make, base, captures = {}) =>
  [
    ["core ms two-sided", "two-sided", ""],
    ["core ms octaves", "octaves", ""],
    ["core ms neural (generic MLP)", "neural", ""],
    ["core ms multi-scale (edge zero)", "multi-scale", "&msEdge=zero"],
    ["core ms multi-scale (edge clamp, his)", "multi-scale", "&msEdge=clamp"],
    // K6.3a: the paper's wider levels (σ ≈ 2^k voxels: level k reads mip k + 1), edge zero (V75 item 5)
    ["core ms multi-scale (σ = 2^k, paper)", "multi-scale", "&msEdge=zero&msSigma=paper"],
  ].filter(([label]) => !MS_ROWS || MS_ROWS.test(label)).map(([label, ms, extra]) => make(label, `${base}&ms=${ms}${extra}`, { lighting: "frame", ...(captures[ms] && !extra ? { capture: captures[ms] } : {}) }));
// K6.Q1b: `--msRows=<regex>` keeps only the 'frame' MS mode rows whose label matches (shorter sweeps)
const MS_ROWS = arg("msRows", "") ? new RegExp(String(arg("msRows", "")), "i") : null;
const k3 = (label, query, extra = {}) => ({ label, page: "k3-frame", query, ...extra });
const k6 = (label, query, extra = {}) => ({ label, page: "k6-scene", query, ...extra });
// The renderer rows of a scene group: both lightings, every multiple-scattering mode.
const sceneRows = (scene) => [
  k6("volume core ms off", `scene=${scene}&lighting=volume&ms=0`, { lighting: "volume", vsSingle: true }),
  k6("volume core ms on", `scene=${scene}&lighting=volume&ms=1`, { lighting: "volume" }),
  // K6.Q1c (V128): the distant light's dipole phase weight ('volume' lighting; internal `ambientPhase`)
  k6("volume core ms off, phase dipole", `scene=${scene}&lighting=volume&ms=0&aoPhase=dipole`, { lighting: "volume", vsSingle: true }),
  k6("volume core ms on, phase dipole-ms", `scene=${scene}&lighting=volume&ms=1&aoPhase=dipole-ms`, { lighting: "volume" }),
  k6("frame core ms off", `scene=${scene}&lighting=frame&ms=0`, { lighting: "frame", vsSingle: true }),
  k6("frame core ms diffusion", `scene=${scene}&lighting=frame&ms=1`, { lighting: "frame" }),
  ...coreMsRows((label, query, extra) => k6(`frame ${label}`, query, extra), `scene=${scene}&lighting=frame`),
];
const GROUP_DEFS = {
  volume: {
    kind: "his",
    title: "'volume' lighting, his sun only (Step 1 isolated)",
    scenario: "step1-volume-dir-ms-on-isolated",
    lighting: "volume",
    rows: [
      k3("core ms off", "scene=step1-volume-dir-ms-off-isolated", { lighting: "volume", capture: "step1-volume-dir-ms-off-isolated", vsSingle: true }),
      k3("core ms on (diffusion)", "scene=step1-volume-dir-ms-on-isolated", { lighting: "volume", capture: "step1-volume-dir-ms-on-isolated" }),
    ],
  },
  frame: {
    kind: "his",
    title: "'frame' lighting, his sun only (Step 3 default isolated, every multipleScattering mode)",
    scenario: "step3-default-dir-isolated",
    lighting: "frame",
    rows: [
      k3("core ms off", "scene=step3-default-dir-isolated&ms=0", { lighting: "frame", vsSingle: true }),
      k3("core ms diffusion", "scene=step3-default-dir-isolated&ms=1", { lighting: "frame", capture: "step3-default-dir-isolated" }),
      ...coreMsRows(k3, "scene=step3-default-dir-isolated", {
        "two-sided": "step4-multiple-scattering-two-sided-dir-isolated",
        octaves: "step4-multiple-scattering-octaves-dir-isolated",
        neural: "step4-default-neural-dir-isolated",
      }),
    ],
  },
  two: {
    kind: "his",
    title: "two overlapping bunnies, 'frame' lighting (density sums: order-independent, V90)",
    scenario: "step3-two-bunnies-dir-isolated",
    lighting: "frame",
    rows: [
      k3("core diffusion", "scene=step3-two-bunnies-dir-isolated", { lighting: "frame", capture: "step3-two-bunnies-dir-isolated" }),
      k3("core ms off", "scene=step3-two-bunnies-dir-isolated&ms=0", { lighting: "frame", vsSingle: true }),
    ],
  },
  // K6.8a (V79): two DIFFERENT media overlapping (one bunny white, the other a red absorbing or a dark medium), each
  // colour on either bunny (the record order of 'over' is the scene order: the second bunny goes over the first in a
  // froxel). `refExtra` gives the reference page the same albedos (linear).
  ...Object.fromEntries(
    [
      ["mix-red2", "albedo2=0.9,0.3,0.2", "1,1,1;0.9,0.3,0.2", "the second bunny red, absorbing"],
      ["mix-red1", "albedo1=0.9,0.3,0.2", "0.9,0.3,0.2;1,1,1", "the first bunny red, absorbing"],
      ["mix-dark2", "albedo2=0.1,0.1,0.1", "1,1,1;0.1,0.1,0.1", "the second bunny dark (albedo 0.1)"],
      ["mix-dark1", "albedo1=0.1,0.1,0.1", "0.1,0.1,0.1;1,1,1", "the first bunny dark (albedo 0.1)"],
    ].map(([key, q, albedo, what]) => [
      key,
      {
        kind: "his",
        title: `V79 (K6.8a): two overlapping bunnies, ${what}, 'volume' lighting (the order-independent mixture; K6.11 deleted 'over')`,
        scenario: "step3-two-bunnies-dir-isolated",
        lighting: "volume",
        refExtra: `&albedo=${albedo}`,
        rows: ["0", "1"].map((ms) =>
          k3(`core ms ${ms === "0" ? "off" : "on"}`, `scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=${ms}&${q}`, { lighting: "volume", ...(ms === "0" ? { vsSingle: true } : {}) }),
        ),
      },
    ]),
  ),
  mix: {
    kind: "his",
    title: "V79: two overlapping bunnies, 'volume' lighting (the order-independent mixture; K6.11 deleted 'over')",
    scenario: "step3-two-bunnies-dir-isolated",
    lighting: "volume",
    rows: [
      k3("core ms off", "scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=0", { lighting: "volume", vsSingle: true }),
      k3("core ms on", "scene=step3-two-bunnies-dir-isolated&lighting=volume&ms=1", { lighting: "volume" }),
    ],
  },
  silhouette: {
    kind: "his",
    title: "V80: silhouettes, 'volume' lighting ms off, reference single scattering at 4× the pixels (frames 2× down)",
    scenario: "step1-volume-dir-ms-off-isolated",
    lighting: "volume",
    size: [640, 360],
    scale: 2,
    singleOnly: true,
    rows: [
      k3("core ms off (prefix)", "scene=step1-volume-dir-ms-off-isolated", { lighting: "volume", capture: "step1-volume-dir-ms-off-isolated", vsSingle: true }),
    ],
  },
  mie: {
    kind: "his",
    title: "Lorenz-Mie 'frame' phase (his table), sun only",
    scenario: "step4-frame-phase-lorenz-mie-dir-isolated",
    lighting: "frame",
    mie: "mono",
    rows: [
      k3("core ms neural", "scene=step4-frame-phase-lorenz-mie-dir-isolated", { lighting: "frame", capture: "step4-frame-phase-lorenz-mie-dir-isolated" }),
      k3("core ms two-sided", "scene=step4-frame-phase-lorenz-mie-dir-isolated&ms=two-sided", { lighting: "frame" }),
      k3("core ms diffusion", "scene=step4-frame-phase-lorenz-mie-dir-isolated&ms=1", { lighting: "frame" }),
      k3("core ms off", "scene=step4-frame-phase-lorenz-mie-dir-isolated&ms=0", { lighting: "frame", vsSingle: true }),
    ],
  },
  "mie-rgb": {
    kind: "his",
    title: "Mie RGB 'frame' phase (his tables), sun only",
    scenario: "step4-frame-phase-lorenz-mie-rgb-dir-isolated",
    lighting: "frame",
    mie: "rgb",
    rows: [
      k3("core ms neural", "scene=step4-frame-phase-lorenz-mie-rgb-dir-isolated", { lighting: "frame", capture: "step4-frame-phase-lorenz-mie-rgb-dir-isolated" }),
      k3("core ms two-sided", "scene=step4-frame-phase-lorenz-mie-rgb-dir-isolated&ms=two-sided", { lighting: "frame" }),
      k3("core ms diffusion", "scene=step4-frame-phase-lorenz-mie-rgb-dir-isolated&ms=1", { lighting: "frame" }),
      k3("core ms off", "scene=step4-frame-phase-lorenz-mie-rgb-dir-isolated&ms=0", { lighting: "frame", vsSingle: true }),
    ],
  },
  env: {
    kind: "his",
    title: "V96: his sky-midday environment only, 'frame' lighting: 'outward' irradiance (SH L2) and 'directional'",
    scenario: "step3-environment-sky-midday-env-isolated",
    lighting: "frame",
    env: "/fixtures/k3/sky-midday-HDR.jpg",
    rows: [
      k3("core outward, ms off", "scene=step3-environment-sky-midday-env-isolated&envMode=outward&ms=0", { lighting: "frame", vsSingle: true }),
      k3("core outward, ms diffusion", "scene=step3-environment-sky-midday-env-isolated&envMode=outward&ms=1", { lighting: "frame" }),
      k3("core directional, ms diffusion", "scene=step3-environment-sky-midday-env-isolated&envMode=directional&ms=1", { lighting: "frame", capture: "step3-environment-sky-midday-env-isolated" }),
      k3("core directional, ms two-sided", "scene=step3-environment-sky-midday-env-isolated&envMode=directional&ms=two-sided", { lighting: "frame" }),
      k3("core directional, ms neural", "scene=step3-environment-sky-midday-env-isolated&envMode=directional&ms=neural", { lighting: "frame" }),
    ],
  },
  // K6.3a: the three Step 4 scenes the core fails his strict bar on, at 640 × 360 (frames 2× down) so the 2.5 × 1.4 px
  // froxel footprint is resolved.
  ...Object.fromEntries(
    [
      ["v80-phase-g-low", "step4-lighting-volume-phase-g-low-dir-isolated", "volume", [-0.5, -0.3, 0.2]],
      ["v80-octave-extinction-high", "step4-octave-extinction-high-dir-isolated", "frame"],
      ["v80-octaves-1", "step4-octaves-1-dir-isolated", "frame"],
    ].map(([key, scenario, lighting, phase]) => [
      key,
      {
        kind: "his",
        title: `K6.3a / V80: ${scenario} (his strict-bar failure through the core), 640 × 360`,
        scenario,
        lighting,
        ...(phase ? { phase } : {}),
        size: [640, 360],
        scale: 2,
        rows: [k3("core (prefix)", `scene=${scenario}`, { lighting, capture: scenario })],
      },
    ]),
  ),
  ...Object.fromEntries(
    [
      ["wdas", "WDAS cloud (sparse, Fp8), sun", "wdas"],
      ["wdas-ambient", "WDAS cloud, sun + ambient", "wdas"],
      ["smoke", "openvdb.org smoke (sparse), sun", "smoke"],
      ["smoke-ambient", "smoke, sun + ambient", "smoke"],
      ["bunny-absorbing", "absorbing bunny (albedo 0.5), sun", "bunny"],
      ["bunny-absorbing-ambient", "absorbing bunny, sun + ambient", "bunny"],
      ["bunny-thin", "thin bunny (extinction 5), sun", "bunny"],
      ["bunny-thin-ambient", "thin bunny, sun + ambient", "bunny"],
      ["bunny-point", "bunny, point light (falloff per lighting mode)", "bunny", true],
      // K6.Q1: the other distant lights (hemisphere, light probe)
      ["smoke-hemi", "smoke, sun + hemisphere light", "smoke"],
      ["wdas-probe", "WDAS cloud, sun + light probe", "wdas"],
      ["bunny-absorbing-hemi", "absorbing bunny, sun + hemisphere light", "bunny"],
      // K6.Q1b: a spectral medium (his 'dark-smoke' preset on the bunny); the reference renders each channel
      ["bunny-spectral", "spectral 'dark-smoke' bunny (extinction 20), sun", "bunny"],
      ["bunny-spectral-hemi", "spectral 'dark-smoke' bunny, sun + hemisphere light (per-channel shadowed ambient)", "bunny"],
    ].map(([scene, title, asset, falloff]) => [scene, { kind: "scene", title, scene, asset, falloff: !!falloff, spectral: scene.includes("spectral"), rows: sceneRows(scene) }]),
  ),
  // K6.Q1b (V128 evidence): distant light alone, single scattering only. In the isotropic groups the reference is the
  // solid-angle mean of transmittance the shadowed ambient estimates; the scene-phase group adds the phase weighting.
  ...Object.fromEntries(
    ["wdas-ambient-only", "smoke-ambient-only", "bunny-absorbing-ambient-only", "wdas-probe-only"].flatMap((base) =>
      ["", "-iso"].map((suffix) => {
        const scene = `${base}${suffix}`;
        return [
          scene,
          {
            kind: "scene",
            title: `V128: ${scene} (single scattering${suffix ? ", isotropic phase" : ", dual-lobe 0.8 / −0.3 / 0.2"})`,
            scene,
            asset: base.startsWith("wdas") ? "wdas" : base.startsWith("smoke") ? "smoke" : "bunny",
            singleOnly: true,
            rows: [
              k6("volume core ms off, AO on", `scene=${scene}&lighting=volume&ms=0&ao=1`, { lighting: "volume", vsSingle: true }),
              // K6.Q1c: the dipole phase weight (the g 0.8 rows should fall to the -iso rows' level)
              k6("volume core ms off, AO on, phase dipole", `scene=${scene}&lighting=volume&ms=0&ao=1&aoPhase=dipole`, { lighting: "volume", vsSingle: true }),
              k6("volume core ms off, AO off", `scene=${scene}&lighting=volume&ms=0&ao=0`, { lighting: "volume", vsSingle: true }),
              k6("frame core ms off, AO on", `scene=${scene}&lighting=frame&ms=0&ao=1`, { lighting: "frame", vsSingle: true }),
              k6("frame core ms off, AO on, shadowRes 128", `scene=${scene}&lighting=frame&ms=0&ao=1&shadowRes=128`, { lighting: "frame", vsSingle: true }),
            ],
          },
        ];
      }),
    ),
  ),
  // K6.Q1b / K6.11b: the jittered estimator (multi-frame mode) on hemisphere scenes: one static frame, 32 frames of
  // jitter + temporal (the AO ring turning, then fixed), and the mean of 32 jittered frames without temporal
  ...Object.fromEntries(
    [
      ["smoke-hemi", "smoke"],
      ["bunny-absorbing-hemi", "bunny"],
    ].map(([scene, asset]) => [
      `${scene}-jitter`,
      {
        kind: "scene",
        title: `K6.11b: ${scene}, 'frame' neural, static vs jittered (multi-frame mode)`,
        scene,
        asset,
        rows: [
          ["static (1 frame of 4)", ""],
          // K6.Q1c (D37 (5)): the ring is fixed (K6.11b's 'turn' / 'fixed' A/B setting is deleted)
          ["jitter + temporal, 32 frames", "&frames=32&jitter=1&temporal=1"],
          ["jitter, mean of 32 frames (no temporal)", "&frames=32&jitter=1&average=32"],
          ["static, AO off", "&ao=0"],
        ].flatMap(([label, q]) => [
          k6(`frame neural ms off, ${label}`, `scene=${scene}&lighting=frame&ms=0${q}`, { lighting: "frame", vsSingle: true }),
          k6(`frame neural, ${label}`, `scene=${scene}&lighting=frame&ms=neural${q}`, { lighting: "frame" }),
        ]),
      },
    ]),
  ),
  // K7.4 (issue #1): his v0.30 defaults as setting changes in our renderer. Not in the default group list (`--groups=k74-…`).
  // Phase: his bunny in 'frame' lighting with Henyey-Greenstein g 0.8 (his v0.26 default, ours) and with 'dual-lobe'
  // 0.8 / −0.3 / 0.2 (his v0.30 default), each against its own reference (the reference reads the phase from the frame's
  // settings); the same in 'volume' lighting; and the report scenes in both phases (`-hg` scenes, k6-scenes.ts).
  // Spectral: `&spectral=1` on every grey row (the medium stays grey, so the reference is the same; 'neural' then runs as
  // 1D diffusion, V133, as his `isNeural`).
  // His demos already set the renderer phase to dual-lobe 0.8 / −0.3 / 0.2 at v0.26 (utils/phase.js; k3-frame does too), so
  // his step3-default capture is dual-lobe; the HG group overrides the phase with `&phase=henyey-greenstein`.
  ...Object.fromEntries(
    [
      ["k74-frame-hg", "Henyey-Greenstein g 0.8 (our default)", "&phase=henyey-greenstein", {}],
      ["k74-frame-dual-lobe", "dual-lobe 0.8 / −0.3 / 0.2 (his v0.30 default, his demos' phase)", "", { neural: "step4-default-neural-dir-isolated", diffusion: "step3-default-dir-isolated" }],
    ].map(([key, what, q, captures]) => [
      key,
      {
        kind: "his",
        title: `K7.4: his bunny, 'frame' lighting, sun, ${what}`,
        scenario: "step3-default-dir-isolated",
        lighting: "frame",
        rows: [
          k3("ms neural", `scene=step3-default-dir-isolated&ms=neural${q}`, { lighting: "frame", ...(captures.neural ? { capture: captures.neural } : {}) }),
          k3("ms neural + spectral", `scene=step3-default-dir-isolated&ms=neural&spectral=1${q}`, { lighting: "frame" }),
          k3("ms diffusion", `scene=step3-default-dir-isolated&ms=1${q}`, { lighting: "frame", ...(captures.diffusion ? { capture: captures.diffusion } : {}) }),
          k3("ms off", `scene=step3-default-dir-isolated&ms=0${q}`, { lighting: "frame", vsSingle: true }),
        ],
      },
    ]),
  ),
  ...Object.fromEntries(
    [
      ["k74-volume-hg", "Henyey-Greenstein g 0.8", [0.8, 0, 0], "&phase=henyey-greenstein"],
      ["k74-volume-dual-lobe", "dual-lobe 0.8 / −0.3 / 0.2 (his frame.js volume phase)", VOLUME_PHASE, ""],
    ].map(([key, what, phase, q]) => [
      key,
      {
        kind: "his",
        title: `K7.4: his bunny, 'volume' lighting, sun, ${what}`,
        scenario: "step1-volume-dir-ms-on-isolated",
        lighting: "volume",
        phase,
        rows: [
          k3("ms on", `scene=step1-volume-dir-ms-on-isolated${q}`, { lighting: "volume" }),
          k3("ms on + spectral", `scene=step1-volume-dir-ms-on-isolated${q}&spectral=1`, { lighting: "volume" }),
          k3("ms off", `scene=step1-volume-dir-ms-off-isolated${q}`, { lighting: "volume", vsSingle: true }),
        ],
      },
    ]),
  ),
  ...Object.fromEntries(
    ["wdas", "wdas-ambient", "smoke-hemi", "bunny-absorbing", "bunny-thin"].flatMap((base) =>
      [base, `${base}-hg`].map((scene) => [
        `k74-${scene}`,
        {
          kind: "scene",
          title: `K7.4: ${scene} (${scene.endsWith("-hg") ? "Henyey-Greenstein g 0.8" : "dual-lobe 0.8 / −0.3 / 0.2"}), spectral off / on`,
          scene,
          asset: base.startsWith("wdas") ? "wdas" : base.startsWith("smoke") ? "smoke" : "bunny",
          rows: [
            k6("frame ms neural (default)", `scene=${scene}&lighting=frame&ms=neural`, { lighting: "frame" }),
            k6("frame ms neural + spectral", `scene=${scene}&lighting=frame&ms=neural&spectral=1`, { lighting: "frame" }),
            k6("frame ms diffusion", `scene=${scene}&lighting=frame&ms=1`, { lighting: "frame" }),
            k6("frame ms off", `scene=${scene}&lighting=frame&ms=0`, { lighting: "frame", vsSingle: true }),
            k6("volume ms on", `scene=${scene}&lighting=volume&ms=1`, { lighting: "volume" }),
            k6("volume ms on + spectral", `scene=${scene}&lighting=volume&ms=1&spectral=1`, { lighting: "volume" }),
          ],
        },
      ]),
    ),
  ),
};
// K7.4: the k74-* groups run only when named
const GROUPS = String(arg("groups", Object.keys(GROUP_DEFS).filter((k) => !k.startsWith("k74-")).join(","))).split(",");

// --- colour ------------------------------------------------------------------------------------------------------
const eotf = (x) => (x < 0.04045 ? x * 0.0773993808 : Math.pow(x * 0.9478672986 + 0.0521327014, 2.4));
const oetf = (x) => (x < 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 0.41666) - 0.055);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const BG_SRGB = [0x59 / 255, 0x8e / 255, 0xff / 255];
const BG = BG_SRGB.map(eotf);
const lum = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** His srgbBlending composite (tools vdbFrameSrgbBlend, clamp on): linear out. K6.Q1b: `alpha` may be per channel. */
function srgbBlend(bg, radiance, alpha) {
  return [0, 1, 2].map((c) => {
    const a = Array.isArray(alpha) ? alpha[c] : alpha;
    if (!(a > 1e-6)) return bg[c];
    const b = clamp01(oetf(bg[c]));
    const v = clamp01(oetf(radiance[c] / a));
    return eotf(b + (v - b) * a);
  });
}

/** Linear float RGB image → RGBA8 sRGB. */
function toRgba8(img) {
  const data = new Uint8Array(img.width * img.height * 4);
  for (let i = 0; i < img.width * img.height; i++) {
    for (let c = 0; c < 3; c++) data[i * 4 + c] = Math.round(clamp01(oetf(Math.max(img.rgb[i * 3 + c], 0))) * 255);
    data[i * 4 + 3] = 255;
  }
  return { width: img.width, height: img.height, data };
}

/** RGBA8 sRGB (w × h) → linear RGB, box-downsampled by `k` in linear light. */
function fromRgba8(rgba, w, h, k) {
  const ow = Math.floor(w / k);
  const oh = Math.floor(h / k);
  const rgb = new Float64Array(ow * oh * 3);
  for (let y = 0; y < oh * k; y++) for (let x = 0; x < ow * k; x++) {
    const o = (Math.floor(y / k) * ow + Math.floor(x / k)) * 3;
    for (let c = 0; c < 3; c++) rgb[o + c] += eotf(rgba[(y * w + x) * 4 + c] / 255) / (k * k);
  }
  return { width: ow, height: oh, rgb };
}

/** The reference's mean (L, T) → composited linear RGB (his sRGB blend over the background), alpha, noise, masks. */
function referenceImage(r) {
  const mean = new Float32Array(Buffer.from(r.mean, "base64").buffer.slice(0));
  const n = r.width * r.height;
  const rgb = new Float64Array(n * 3);
  const alpha = new Float64Array(n);
  // K6.Q1b: a spectral reference has each channel's transmittance (its composite and unblend use them)
  const tRgb = r.transmittanceRgb ? new Float32Array(Buffer.from(r.transmittanceRgb, "base64").buffer.slice(0)) : null;
  const alphaRgb = tRgb ? new Float64Array(n * 3) : null;
  for (let i = 0; i < n; i++) {
    const L = [mean[i * 4], mean[i * 4 + 1], mean[i * 4 + 2]];
    const a = clamp01(1 - mean[i * 4 + 3]);
    alpha[i] = a;
    if (alphaRgb) {
      const ac = [0, 1, 2].map((c) => clamp01(1 - tRgb[i * 3 + c]));
      alphaRgb.set(ac, i * 3);
      rgb.set(srgbBlend(BG, L, ac), i * 3);
    } else rgb.set(a > 1e-6 ? srgbBlend(BG, L, a) : BG, i * 3);
  }
  const overlap = r.overlap ? new Uint8Array(Buffer.from(r.overlap, "base64")) : null;
  const twoOrMore = overlap ? overlap.map((b) => (b & (b - 1) ? 1 : 0)) : null;
  return { width: r.width, height: r.height, rgb, alpha, alphaRgb, noise: r.noise?.mean ?? 0, noiseP90: r.noise?.p90 ?? 0, volumePixels: r.noise?.pixels ?? 0, silhouette: silhouetteMask(alpha, r.width, r.height), overlap: twoOrMore };
}

/** His sRGB blend inverted: the radiance L (linear, ≤ α per channel) whose composite over the background at α is pixel i. K6.Q1b: α per channel. */
function unblend(rgb, i, alpha) {
  return [0, 1, 2].map((c) => {
    const a = Array.isArray(alpha) ? alpha[c] : alpha;
    if (!(a > 1e-6)) return 0;
    const b = clamp01(oetf(BG[c]));
    const v = clamp01((clamp01(oetf(Math.max(rgb[i * 3 + c], 0))) - b * (1 - a)) / a);
    return a * eotf(v);
  });
}

/** Pixel rectangle of the volume (reference alpha > 0.02) grown by `pad`, as a mask. */
function rectMask(ref, pad = 6) {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (let y = 0; y < ref.height; y++) for (let x = 0; x < ref.width; x++) {
    if (ref.alpha[y * ref.width + x] > 0.02) {
      x0 = Math.min(x0, x);
      y0 = Math.min(y0, y);
      x1 = Math.max(x1, x);
      y1 = Math.max(y1, y);
    }
  }
  const m = new Uint8Array(ref.width * ref.height);
  for (let y = Math.max(0, y0 - pad); y <= Math.min(ref.height - 1, y1 + pad); y++) for (let x = Math.max(0, x0 - pad); x <= Math.min(ref.width - 1, x1 + pad); x++) m[y * ref.width + x] = 1;
  return m;
}

/** Luminance errors of `a` vs `ref` over the pixels where `inMask(i)`: bias, median and mean relative, RMSE linear. */
function regionStats(a, ref, inMask) {
  let sq = 0;
  let count = 0;
  let absErr = 0;
  let contrast = 0;
  let yA = 0;
  let yR = 0;
  // K6.Q1b: per-channel radiance sums (the chromatic bias of spectral media)
  const cA = [0, 0, 0];
  const cR = [0, 0, 0];
  const rels = [];
  for (let i = 0; i < ref.width * ref.height; i++) {
    if (!inMask(i)) continue;
    count++;
    for (let c = 0; c < 3; c++) {
      sq += (a.rgb[i * 3 + c] - ref.rgb[i * 3 + c]) ** 2;
      absErr += Math.abs(a.rgb[i * 3 + c] - ref.rgb[i * 3 + c]);
      contrast += Math.abs(ref.rgb[i * 3 + c] - BG[c]);
    }
    if (!(ref.alpha[i] > 0.02)) continue;
    // Luminance of the volume's radiance L: his sRGB blend inverted at the reference's coverage (K6.R1; the linear
    // "composite minus (1 − α)·background" of K6.R goes negative for volumes darker than the background).
    const at = ref.alphaRgb ? [ref.alphaRgb[i * 3], ref.alphaRgb[i * 3 + 1], ref.alphaRgb[i * 3 + 2]] : ref.alpha[i];
    const la = unblend(a.rgb, i, at);
    const lr = unblend(ref.rgb, i, at);
    for (let c = 0; c < 3; c++) {
      cA[c] += la[c];
      cR[c] += lr[c];
    }
    const ya = lum(...la);
    const yr = lum(...lr);
    yA += ya;
    yR += yr;
    if (yr > 1e-3) rels.push(Math.abs(ya - yr) / yr);
  }
  rels.sort((x, y) => x - y);
  if (!count) return { pixels: 0, rmseLinear: null, medianRelLum: null, meanRelLum: null, lumBias: null, relError: null };
  return {
    pixels: count,
    // composite error relative to the reference's contrast against the background (defined on both sides of an edge)
    relError: contrast > 0 ? absErr / contrast : null,
    rmseLinear: count ? Math.sqrt(sq / (count * 3)) : 0,
    medianRelLum: rels.length ? rels[Math.floor(rels.length / 2)] : 0,
    meanRelLum: rels.length ? rels.reduce((s, v) => s + v, 0) / rels.length : 0,
    lumBias: yR > 0 ? yA / yR - 1 : null,
    rgbBias: cR.map((r, c) => (r > 0 ? cA[c] / r - 1 : null)),
  };
}

function compare(a, ref) {
  if (a.width !== ref.width || a.height !== ref.height) throw new Error(`size ${a.width}x${a.height} vs ${ref.width}x${ref.height}`);
  const n = ref.width * ref.height;
  const A8 = toRgba8(a);
  const R8 = toRgba8(ref);
  let mse8 = 0;
  let over8 = 0;
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let c = 0; c < 3; c++) {
      const d = A8.data[i * 4 + c] - R8.data[i * 4 + c];
      mse8 += d * d;
      m = Math.max(m, Math.abs(d));
    }
    if (m > 8) over8++;
  }
  mse8 /= n * 3;
  const vol = regionStats(a, ref, (i) => ref.alpha[i] > 0.02);
  const sil = regionStats(a, ref, (i) => ref.silhouette[i] === 1);
  const ovl = ref.overlap ? regionStats(a, ref, (i) => ref.overlap[i] === 1) : null;
  return {
    psnr: mse8 === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse8),
    over8: over8 / n,
    ssim: ssim(A8, R8).ssim,
    ssimRect: ssim(A8, R8, rectMask(ref)).ssim,
    volumePixels: vol.pixels,
    rmseLinear: vol.rmseLinear,
    meanRelLum: vol.meanRelLum,
    medianRelLum: vol.medianRelLum,
    lumBias: vol.lumBias,
    rgbBias: vol.rgbBias,
    silhouette: { pixels: sil.pixels, relError: sil.relError, rmseLinear: sil.rmseLinear, lumBias: sil.lumBias },
    overlap: ovl && { pixels: ovl.pixels, lumBias: ovl.lumBias, medianRelLum: ovl.medianRelLum, rmseLinear: ovl.rmseLinear },
  };
}

// --- server and browser ------------------------------------------------------------------------------------------
async function startServer() {
  if (BASE) return null;
  BASE = `http://127.0.0.1:${PORT}`;
  const child = spawn("pnpm", ["--filter", "@webvdb/examples", "dev", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"], { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] });
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(BASE)).ok) return child;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  child.kill();
  throw new Error(`Vite did not start on ${BASE}`);
}

async function runPage(browser, page, query, state) {
  const p = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  await p.goto(`${BASE}/src/harness/${page}.html?${query}`);
  await p.waitForFunction((s) => window[s]?.done === true, state, { timeout: 3_600_000, polling: 1000 });
  const r = await p.evaluate((s) => window[s], state);
  await p.close();
  if (r.error) throw new Error(`${page}?${query}: ${String(r.error).split("\n")[0]}`);
  return { ...r, pageErrors: errors };
}

const STATE = { "k3-frame": "__VDB_K3_FRAME__", "k6-scene": "__VDB_K6_SCENE__" };
const f = (x, d = 2) => (x === null || x === undefined ? "—" : Number.isFinite(x) ? x.toFixed(d) : String(x));
const pct = (x, d = 1) => (x === null || x === undefined ? "—" : `${f(x * 100, d)} %`);

function groupUsable(g) {
  if (g.kind === "his") return MANIFEST && MANIFEST.scenarios.some((s) => s.name === g.scenario) ? null : `scenario ${g.scenario} not in the capture manifest`;
  if (g.asset === "wdas" && !existsSync(need.wdas)) return "fixtures/wdas/wdas_cloud_quarter_fp8.nvdb missing";
  if (g.asset === "smoke" && !existsSync(need.smoke)) return "fixtures/vdb-samples/smoke.vdb missing";
  return null;
}

const server = await startServer();
const browser = await chromium.launch({
  ...(SWIFT ? {} : { channel: "chrome" }),
  // Headed: macOS Chrome picks the hardware adapter only with a window server session (scripts/k3/time-hardware.mjs).
  headless: SWIFT || arg("headless", false) === true,
  args: ["--enable-unsafe-webgpu", ...(SWIFT ? ["--enable-features=Vulkan", "--enable-unsafe-swiftshader"] : [])],
});
const report = { at: new Date().toISOString(), browser: browser.version(), size: [W, H], scale: SCALE, target: TARGET, budgetSeconds: BUDGET, maxBounces: MAX_BOUNCES, groups: [] };
mkdirSync(OUT, { recursive: true });
try {
  for (const key of GROUPS) {
    const g = GROUP_DEFS[key];
    if (!g) throw new Error(`unknown group ${key}`);
    const skip = groupUsable(g);
    if (skip) {
      console.log(`== ${key}: skipped (${skip})`);
      continue;
    }
    const [gw, gh] = g.size ?? [W, H];
    const scale = g.scale ?? SCALE;
    const dir = path.join(OUT, key);
    mkdirSync(dir, { recursive: true });
    console.log(`== ${key}: ${g.title}`);
    // Frames first ('frame' lighting: their settings give the his-group reference its phase).
    const frames = [];
    for (const t of g.rows) {
      if (ROWS && !ROWS.test(t.label)) continue;
      if (t.slot) {
        frames.push({ ...t });
        continue;
      }
      process.stdout.write(`  ${t.label} … `);
      const r = await runPage(browser, t.page, `${t.query}${EXTRA_QUERY}&w=${gw * scale}&h=${gh * scale}`, STATE[t.page]);
      const img = fromRgba8(new Uint8Array(Buffer.from(r.rgba, "base64")), r.width, r.height, scale);
      frames.push({ ...t, img, settings: r.settings, adapter: r.adapter, backend: r.backend, shadowGridResolution: r.shadowGridResolution });
      console.log(`${r.width}x${r.height} ${r.backend ?? ""}${r.shadowGridResolution ? ` grid ${r.shadowGridResolution}` : ""} (${r.adapter})`);
    }
    // References: per lighting mapping when the scene has point lights, else one; multiple and single scattering.
    const lightings = g.kind === "scene" && g.falloff ? ["volume", "frame"] : [g.lighting ?? "frame"];
    const refs = {};
    for (const lighting of lightings) {
      let query;
      if (g.kind === "his") {
        const s = frames.find((t) => t.settings && t.lighting === "frame")?.settings;
        const phase = g.phase ?? (g.lighting === "volume" ? VOLUME_PHASE : s?.phase === "dual-lobe" ? [s.phaseG, s.phaseBackG, s.phaseBackWeight] : [s?.phaseG ?? 0.8, 0, 0]);
        if (g.lighting === "frame" && s && !["henyey-greenstein", "dual-lobe", "isotropic", "lorenz-mie", "lorenz-mie-rgb"].includes(s.phase)) console.warn(`  phase ${s.phase}: the reference uses HG g = ${phase[0]}`);
        query = `case=report&scenario=${g.scenario}&lighting=${lighting}&phase=${phase.join(",")}${g.mie ? `&mie=${g.mie}` : ""}${g.env ? `&env=${g.env}` : ""}${g.refExtra ?? ""}`;
      } else query = `case=scene&scene=${g.scene}&lighting=${lighting}`;
      const common = `&w=${gw}&h=${gh}&spp=${SPP}&target=${TARGET}&check=${CHECK}&budget=${BUDGET}`;
      const entry = {};
      for (const [kind, bounces] of g.singleOnly ? [["single", 1]] : [["full", MAX_BOUNCES], ["single", 1]]) {
        process.stdout.write(`  reference ${lighting} ${kind === "full" ? `(multiple scattering, ≤ ${MAX_BOUNCES} bounces)` : "(single scattering)"} … `);
        const refQuery = `${query}${common}&maxBounces=${bounces}`;
        const cacheFile = path.join(dir, `reference-${lighting}-${kind}.json`);
        const cached = REUSE && existsSync(cacheFile) ? JSON.parse(readFileSync(cacheFile, "utf8")) : null;
        const r = cached?.query === refQuery ? { ...cached.result, reused: true } : await runPage(browser, "k6-reference", refQuery, "__VDB_K6_REFERENCE__");
        if (!r.reused) writeFileSync(cacheFile, JSON.stringify({ query: refQuery, result: { report: r.report, adapter: r.adapter } }));
        const img = referenceImage(r.report);
        entry[kind] = { img, report: { ...r.report, mean: undefined, stderr: undefined, overlap: undefined, transmittanceRgb: undefined }, adapter: r.adapter };
        console.log(`${r.reused ? "(reused) " : ""}${r.report.batches} batches in ${r.report.seconds.toFixed(0)} s, noise ${(img.noise * 100).toFixed(2)} % (p90 ${(img.noiseP90 * 100).toFixed(1)} %, ${r.report.stop}) (${r.adapter})`);
        writeFileSync(path.join(dir, `reference-${lighting}-${kind}.png`), encodePng(toRgba8(img)));
      }
      refs[lighting] = entry;
    }
    const rows = [];
    const add = (label, img, kind, lighting, extra = {}) => {
      const ref = refs[lightings.includes(lighting) ? lighting : lightings[0]];
      const file = label.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
      writeFileSync(path.join(dir, `${file}.png`), encodePng(toRgba8(img)));
      const vsFull = ref.full ? compare(img, ref.full.img) : null;
      const vsSingle = compare(img, ref.single.img);
      writeFileSync(path.join(dir, `${file}.err.png`), encodePng(diffHeatmap(toRgba8(img), toRgba8((ref.full ?? ref.single).img))));
      rows.push({ label, kind, lighting, vsFull, vsSingle, ...extra });
    };
    for (const t of frames) {
      if (t.slot) {
        rows.push({ label: t.label, kind: "slot", vsFull: null, vsSingle: null });
        continue;
      }
      add(t.label, t.img, "core", t.lighting, { vsSingleExpected: !!t.vsSingle, ...(t.shadowGridResolution ? { shadowGridResolution: t.shadowGridResolution } : {}) });
      if (t.capture && existsSync(path.join(CAPTURES, `${t.capture}.png`))) {
        const png = decodePng(readFileSync(path.join(CAPTURES, `${t.capture}.png`)));
        add(`his ${t.capture}`, fromRgba8(png.data, png.width, png.height, Math.round(png.width / gw)), "his", t.lighting, { vsSingleExpected: !!t.vsSingle });
      }
    }
    for (const l of lightings) if (refs[l].full) add(`reference single scattering (${l})`, refs[l].single.img, "reference", l);
    const group = {
      key,
      kind: g.kind,
      title: g.title,
      scenario: g.scenario ?? g.scene,
      size: [gw, gh],
      scale,
      references: Object.fromEntries(Object.entries(refs).map(([l, e]) => [l, { full: e.full?.report ?? null, single: e.single.report, noise: { full: e.full?.img.noise ?? null, single: e.single.img.noise } }])),
      rows,
    };
    report.groups.push(group);
    group.md = markdown(group);
    for (const r of rows) {
      if (!r.vsSingle) {
        console.log(`  ${r.label.padEnd(56)} (slot)`);
        continue;
      }
      const v = r.vsFull ?? r.vsSingle;
      console.log(`  ${r.label.padEnd(56)}${r.shadowGridResolution ? ` [grid ${r.shadowGridResolution}]` : ""} vs ${r.vsFull ? "MS" : "SS"}: bias ${pct(v.lumBias)}  median ${pct(v.medianRelLum)}  SSIM ${f(v.ssimRect, 3)}  sil. err ${pct(v.silhouette.relError)} | vs SS: bias ${pct(r.vsSingle.lumBias)}  sil. err ${pct(r.vsSingle.silhouette.relError)}${g.spectral ? ` | rgb bias ${(r.vsFull ?? r.vsSingle).rgbBias.map((x) => pct(x)).join(" / ")}` : ""}`);
    }
  }
} finally {
  await browser.close();
  server?.kill();
}

function markdown(g) {
  const md = [];
  const refLine = Object.entries(g.references)
    .map(([l, r]) => `${l} mapping: ${r.full ? `multiple scattering ${r.full.batches} × ${SPP} spp (noise ${pct(r.noise.full, 2)}${r.full.converged ? "" : `, stopped by ${r.full.stop}`}), ` : ""}single ${r.single.batches} × ${SPP} spp (noise ${pct(r.noise.single, 2)})`)
    .join("; ");
  md.push(`## ${g.key}: ${g.title}`, "", `\`${g.scenario}\`, reference ${g.size[0]}×${g.size[1]} (frames ${g.scale}× down): ${refLine}.`, "");
  md.push("| Image | vs MS: lum. bias | median rel. err | SSIM (rect) | PSNR dB | silhouette err | vs SS: lum. bias | median rel. err | silhouette err |" + (g.rows.some((r) => r.vsSingle?.overlap) ? " overlap bias (MS / SS) |" : ""), "|---|---|---|---|---|---|---|---|---|" + (g.rows.some((r) => r.vsSingle?.overlap) ? "---|" : ""));
  for (const r of g.rows) {
    if (!r.vsSingle) {
      md.push(`| ${r.label} | — | — | — | — | — | — | — | — |`);
      continue;
    }
    const a = r.vsFull;
    const s = r.vsSingle;
    const ovl = s.overlap ? ` ${a?.overlap ? pct(a.overlap.lumBias) : "—"} / ${pct(s.overlap.lumBias)} |` : "";
    md.push(`| ${r.label}${r.shadowGridResolution ? ` (grid ${r.shadowGridResolution})` : ""} | ${a ? pct(a.lumBias) : "—"} | ${a ? pct(a.medianRelLum) : "—"} | ${a ? f(a.ssimRect, 3) : "—"} | ${a ? f(a.psnr) : "—"} | ${a ? pct(a.silhouette.relError) : "—"} | ${pct(s.lumBias)} | ${pct(s.medianRelLum)} | ${pct(s.silhouette.relError)} |${ovl}`);
  }
  md.push("");
  return md;
}

const header = [
  "# K6.R / K6.R1 reference report",
  "",
  `${report.at}, Chrome ${report.browser}${SWIFT ? " (SwiftShader: script debugging only, not a result)" : ""}. Report only: nothing gates on these numbers; no timings.`,
  "Reference: brute-force volumetric path tracing (core/reference) at level 0, box-filtered pixels, per-cell majorants; stopped at",
  `a mean relative standard error of Y of ${pct(TARGET)} over the volume pixels (cap ${BUDGET} s). Frames at ${SCALE}× the size (his 1280 × 720 at`,
  "160 × 90), box-downsampled in linear light; his captures likewise. All composited with his sRGB blend over 0x598eff.",
  "Columns: luminance bias Σ Y / Σ Y_ref − 1 over the volume pixels (the energy error), median relative luminance error",
  "per pixel (floor ≈ the reference noise), SSIM of the volume rectangle, PSNR of the whole frame, the silhouette band's",
  "error Σ|A − R| / Σ|R − background| (V80), and vs single scattering the same (rows marked ms off compare there).",
  "",
];
// A partial run (--groups) keeps the other groups of the previous report.
const previous = existsSync(path.join(OUT, "report.json")) ? JSON.parse(readFileSync(path.join(OUT, "report.json"), "utf8")) : null;
const kept = (previous?.groups ?? []).filter((g) => !report.groups.some((n) => n.key === g.key) && GROUP_DEFS[g.key]);
report.groups = [...kept, ...report.groups].sort((a, b) => Object.keys(GROUP_DEFS).indexOf(a.key) - Object.keys(GROUP_DEFS).indexOf(b.key));
writeFileSync(path.join(OUT, "report.json"), JSON.stringify(report, null, 2));
writeFileSync(path.join(OUT, "report.md"), [...header, ...report.groups.flatMap((g) => [...(g.md ?? []), ...(kept.includes(g) ? [`(from the run of ${previous.at})`, ""] : [])])].join("\n"));
console.log(`wrote ${path.relative(ROOT, path.join(OUT, "report.md"))}`);
