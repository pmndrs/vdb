import { expect, test } from "@playwright/test";

// P1 (pathtracer integration, our side): GPU parity of the P1.1 point-sample contract, the P1.3/P1.4 majorant walks,
// the P1.5 lean reader and P1.6 several grids at one binding, against the CPU twins. The page compares and publishes
// per-row counts; see examples/src/harness/p1-parity.ts for the tolerances.
const PAGE_URL = "/src/harness/p1-parity.html";
const PROBES = ["/fixtures/primitives/box_fog_float.nvdb", "/fixtures/vdb-samples/smoke.vdb"];

interface Row {
  total: number;
  failed: number;
  maxDelta: number;
}
interface P1Parity {
  done: boolean;
  error?: string;
  missing?: string[];
  modules: { id: string; bytes: number; compileErrors: string[] }[];
  rows: Record<string, Row>;
  failures: string[];
  info: Record<string, unknown>;
  validationError?: string;
}

// Rows every run must have exercised, so a silently skipped check fails the gate.
const REQUIRED = [
  "contract-vendored-twin",
  "contract-vendored-dense",
  "contract-lean-twin",
  "contract-lean-dense",
  "vendored-read-float",
  "lean-read-float",
  "lean-read-fp8",
  "lean-read-fpn",
  "lean-read-half",
  "lean-read-active",
  "lean-sample",
  "vendored-sample",
  "multi-grid-vendored-index",
  "multi-grid-vendored-sample",
  "multi-grid-vendored-sees-all-three",
  "multi-grid-lean-sample",
  "walk-brick+minima/zero",
  "walk-global/zero",
  "walk-nodeMax/zero",
  "walk-brick+minima/zero/lean",
  "walk-brick/background",
];

async function load(page: import("@playwright/test").Page, query: string, consoleErrors: string[]): Promise<P1Parity> {
  for (const url of PROBES) {
    const probe = await page.request.get(url);
    test.skip(!probe.ok(), `fixtures missing (${url} -> ${probe.status()})`);
  }
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __P1_PARITY__?: { done: boolean } }).__P1_PARITY__?.done === true, undefined, { timeout: 540_000 });
  return (await page.evaluate(() => (window as { __P1_PARITY__?: unknown }).__P1_PARITY__)) as P1Parity;
}

// Fail-before: each WGSL mutation (examples/src/harness/p1-parity.ts WGSL_PATCHES) must fail the rows of the entry it breaks.
const MUTATIONS: Record<string, string[]> = {
  "wgsl-brick-axis": ["walk-brick+minima/zero", "walk-brick+minima/zero/lean", "walk-brick/background"],
  "wgsl-brick-minima": ["walk-brick+minima/zero", "walk-brick+minima/zero/lean"],
  "wgsl-support-pad": ["walk-global/zero", "walk-nodeMax/zero"],
  "wgsl-nodemax-support": ["walk-nodeMax/zero"],
  "wgsl-lean-leaf-index": ["lean-read-float", "lean-read-fp8", "lean-read-fpn", "lean-sample"],
  "wgsl-lean-active-index": ["lean-read-active"],
  "wgsl-lean-root-key": ["lean-read-float"],
  "wgsl-contract-zero": ["contract-vendored-dense", "contract-lean-dense"],
  "wgsl-multi-grid": ["multi-grid-vendored-index", "multi-grid-lean-index"],
};

test.describe("P1 parity fail-before", () => {
  for (const [mutate, rows] of Object.entries(MUTATIONS)) {
    test(`${mutate} fails ${rows.join(", ")}`, async ({ page }) => {
      test.setTimeout(600_000);
      const r = await load(page, `?mutate=${mutate}`, []);
      const detail = JSON.stringify({ error: r.error, rows: r.rows }, null, 2);
      expect(r.error, detail).toBeUndefined();
      for (const row of rows) expect(r.rows[row]?.failed ?? 0, `${mutate}: ${row} must fail\n${detail}`).toBeGreaterThan(0);
    });
  }
});

test("P1 parity: contract, majorant walks, lean reader, several grids at one binding", async ({ page }) => {
  test.setTimeout(600_000);
  const consoleErrors: string[] = [];
  const r = await load(page, "", consoleErrors);
  const detail = JSON.stringify({ error: r.error, missing: r.missing, validationError: r.validationError, consoleErrors, failures: r.failures.slice(0, 20), info: r.info }, null, 2);

  expect(r.error, detail).toBeUndefined();
  expect(r.missing, detail).toBeUndefined();
  expect(r.validationError, detail).toBeUndefined();
  expect(r.modules.flatMap((m) => m.compileErrors), detail).toEqual([]);
  for (const name of REQUIRED) expect(r.rows[name]?.total ?? 0, `${name} ran\n${detail}`).toBeGreaterThan(0);
  // The 10k contract points run once per vendored module with outside 'zero' and Float (four of them), once per lean one (two).
  expect(r.rows["contract-vendored-dense"]!.total % 10_000, detail).toBe(0);
  expect(r.rows["contract-lean-dense"]!.total, detail).toBe(20_000);
  console.log(
    Object.entries(r.rows)
      .map(([k, s]) => `${k}: ${s.total - s.failed}/${s.total} max ${s.maxDelta.toExponential(2)}`)
      .join("\n"),
  );
  const failed = Object.entries(r.rows).filter(([, s]) => s.failed > 0);
  expect(failed, detail).toEqual([]);
  expect(consoleErrors, detail).toEqual([]);
});
