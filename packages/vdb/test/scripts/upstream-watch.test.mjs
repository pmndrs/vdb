import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  compareGithubCommits,
  compareGithubReleases,
  compareGithubTags,
  compareNpm,
  compareSemver,
  digestInBody,
  digestOf,
  main,
  planIssue,
  renderIssueBody,
  validatePins,
  watchRequest,
} from "../../../../scripts/upstream-watch.mjs";

// K7.6: the upstream watch's comparisons and issue plan over API responses captured on 2026-11-28 (trimmed to the
// fields the script reads). No network: main() runs against a fake JSON client.

const fixture = (name) =>
  JSON.parse(readFileSync(new URL(`./fixtures/upstream-watch/${name}.json`, import.meta.url), "utf8"));
const pinsUrl = new URL("../../../../.github/upstream-pins.json", import.meta.url);
const committed = JSON.parse(readFileSync(pinsUrl, "utf8"));
// The fixtures were captured on 2026-11-28 with the mjurczyk pins at v0.29.0 / 0.29.0. K7.4 (2026-11-29) reviewed his
// releases and moved the committed pins past the fixtures, so the comparisons run against the capture-time pins (main()
// reads them from a temp pins file); the committed file is checked on its own below.
const CAPTURE_PINS = { "mjurczyk-tag": "v0.29.0", "mjurczyk-npm-openvdb": "0.29.0", "mjurczyk-npm-three-openvdb": "0.29.0", "mjurczyk-npm-webvdb": "0.29.0" };
const pins = { ...committed, watches: committed.watches.map((w) => (w.id in CAPTURE_PINS ? { ...w, pin: CAPTURE_PINS[w.id] } : w)) };
const pinsFile = join(mkdtempSync(join(tmpdir(), "upstream-watch-")), "pins.json");
writeFileSync(pinsFile, JSON.stringify(pins));
const watch = (id) => pins.watches.find((w) => w.id === id);

const releases = fixture("openvdb-releases"); // v14.0.0, v13.2.0, v13.1.0, …
const tags = fixture("mjurczyk-tags"); // v0.30.0 … v0.25.0
const npm = fixture("npm-three-openvdb"); // 0.0.0-stage, 0.5.0 … 0.30.0, latest 0.30.0
const commits = fixture("emcfarlane-commits"); // 265e8d8 (HEAD), 80153e3, …

describe("K7.6 upstream watch: versions", () => {
  it("orders semver with and without v, pre-releases below releases", () => {
    expect(compareSemver("v0.30.0", "0.29.0")).toBeGreaterThan(0);
    expect(compareSemver("v14.0.0", "v13.2.0")).toBeGreaterThan(0);
    expect(compareSemver("0.10.0", "0.9.0")).toBeGreaterThan(0);
    expect(compareSemver("1.0.0-rc.1", "1.0.0")).toBeLessThan(0);
    expect(compareSemver("v1.2.3", "1.2.3")).toBe(0);
  });
});

describe("K7.6 upstream watch: comparisons", () => {
  it("OpenVDB releases: at the pin, nothing moved", () => {
    const r = compareGithubReleases(watch("openvdb-release"), releases);
    expect(r).toMatchObject({ pin: "v14.0.0", latest: "v14.0.0", moved: false, newer: [] });
  });

  it("OpenVDB releases: newer releases after an older pin, newest first, drafts and pre-releases skipped", () => {
    const extra = [
      { tag_name: "v15.0.0", published_at: "2027-04-01T00:00:00Z", draft: true, prerelease: false },
      { tag_name: "v14.1.0-rc1", published_at: "2027-03-01T00:00:00Z", draft: false, prerelease: true },
      // A patch on an older line, published after the pin: still news.
      { tag_name: "v13.2.1", published_at: "2027-02-01T00:00:00Z", draft: false, prerelease: false, html_url: "u" },
    ];
    const r = compareGithubReleases({ ...watch("openvdb-release"), pin: "v13.1.0" }, [...extra, ...releases]);
    expect(r.moved).toBe(true);
    expect(r.newer.map((n) => n.name)).toEqual(["v13.2.1", "v14.0.0", "v13.2.0"]);
    expect(r.latest).toBe("v13.2.1");
  });

  it("OpenVDB releases: a pin off the page falls back to version order and says so", () => {
    const r = compareGithubReleases({ ...watch("openvdb-release"), pin: "v9.0.0" }, releases);
    expect(r.moved).toBe(true);
    expect(r.newer.length).toBe(releases.length);
    expect(r.note).toMatch(/not among the latest/);
  });

  it("mjurczyk tags: v0.30.0 is past the v0.29.0 pin; non-version tags are ignored", () => {
    const r = compareGithubTags(watch("mjurczyk-tag"), [{ name: "latest", commit: { sha: "x" } }, ...tags]);
    expect(r).toMatchObject({ pin: "v0.29.0", latest: "v0.30.0", moved: true });
    expect(r.newer).toEqual([
      expect.objectContaining({ name: "v0.30.0", detail: tags.find((t) => t.name === "v0.30.0").commit.sha.slice(0, 7) }),
    ]);
    expect(compareGithubTags({ ...watch("mjurczyk-tag"), pin: "v0.30.0" }, tags).moved).toBe(false);
    expect(compareGithubTags({ ...watch("mjurczyk-tag"), pin: "v0.27.0" }, tags).newer.map((n) => n.name)).toEqual([
      "v0.30.0",
      "v0.29.0",
      "v0.28.0",
    ]);
  });

  it("npm: versions above the pin (numeric order, pre-releases skipped), and latest", () => {
    const w = watch("mjurczyk-npm-three-openvdb");
    expect(compareNpm(w, npm)).toMatchObject({ latest: "0.30.0", moved: true, newer: [expect.objectContaining({ name: "0.30.0" })] });
    expect(compareNpm({ ...w, pin: "0.30.0" }, npm).moved).toBe(false);
    expect(compareNpm({ ...w, pin: "0.9.0" }, npm).newer.map((n) => n.name).slice(-2)).toEqual(["0.11.0", "0.10.0"]);
    const next = { ...npm, versions: { ...npm.versions, "0.31.0-beta.1": {} } };
    expect(compareNpm({ ...w, pin: "0.30.0" }, next).moved).toBe(false);
    expect(compareNpm({ ...w, pin: "0.31.0" }, npm).note).toMatch(/below the pin/);
  });

  it("emcfarlane commits: HEAD at the pin is unchanged; commits above it are listed newest first", () => {
    const w = watch("emcfarlane-commit");
    expect(compareGithubCommits(w, commits)).toMatchObject({ moved: false, newer: [], latest: w.pin });
    const r = compareGithubCommits({ ...w, pin: commits[2].sha.slice(0, 7) }, commits);
    expect(r.moved).toBe(true);
    expect(r.newer.map((n) => n.name)).toEqual([commits[0].sha.slice(0, 7), commits[1].sha.slice(0, 7)]);
    expect(r.newer[0].detail).toBe(commits[0].commit.message.split("\n")[0]);
    const gone = compareGithubCommits({ ...w, pin: "0".repeat(40) }, commits);
    expect(gone).toMatchObject({ moved: true });
    expect(gone.newer.length).toBe(commits.length);
    expect(gone.note).toMatch(/force-push/);
  });

  it("the committed pins are valid and every watch has a request", () => {
    expect(validatePins(committed).length).toBeGreaterThanOrEqual(6);
    for (const w of committed.watches) expect(watchRequest(w).url).toMatch(/^https:\/\//);
    expect(Object.keys(CAPTURE_PINS).every((id) => committed.watches.some((w) => w.id === id))).toBe(true);
    expect(() => validatePins({ ...pins, watches: [...pins.watches, pins.watches[0]] })).toThrow(/duplicate/);
    expect(() => validatePins({ ...pins, watches: [{ ...pins.watches[0], pin: "main" }] })).toThrow(/not a version/);
  });
});

describe("K7.6 upstream watch: issue plan", () => {
  const moved = [compareGithubTags(watch("mjurczyk-tag"), tags), compareGithubReleases(watch("openvdb-release"), releases)];
  const quiet = [compareGithubReleases(watch("openvdb-release"), releases)];

  it("the digest follows what moved, not dates, and round-trips through the body", () => {
    const body = renderIssueBody(moved, { checkedAt: "2026-11-28" });
    expect(digestInBody(body)).toBe(digestOf(moved));
    expect(digestInBody(renderIssueBody(moved, { checkedAt: "2027-01-01" }))).toBe(digestOf(moved));
    expect(body).toContain("v0.30.0");
    expect(body).not.toContain("### OpenVDB"); // unmoved watches stay out of the body
    const more = [compareGithubTags({ ...watch("mjurczyk-tag"), pin: "v0.28.0" }, tags)];
    expect(digestOf(more)).not.toBe(digestOf(moved));
  });

  it("create, leave alone, update, close", () => {
    const body = renderIssueBody(moved, {});
    expect(planIssue(moved, null).action).toBe("create");
    expect(planIssue(moved, { number: 7, body }).action).toBe("none");
    const older = renderIssueBody([compareGithubTags({ ...watch("mjurczyk-tag"), pin: "v0.28.0" }, tags)], {});
    expect(planIssue(moved, { number: 7, body: older })).toMatchObject({ action: "update" });
    expect(planIssue(quiet, null).action).toBe("none");
    expect(planIssue(quiet, { number: 7, body })).toMatchObject({ action: "close" });
    // A failed watch never closes the issue.
    expect(planIssue([...quiet, { ...quiet[0], id: "x", error: "HTTP 403" }], { number: 7, body }).action).toBe("none");
  });
});

describe("K7.6 upstream watch: main() against a fake client", () => {
  const responses = (overrides = {}) => ({
    "AcademySoftwareFoundation/openvdb/releases": releases,
    "mjurczyk/openvdb/tags": tags,
    "registry.npmjs.org/openvdb": npm,
    "registry.npmjs.org/three-openvdb": npm,
    "registry.npmjs.org/webvdb": npm,
    "emcfarlane/webgpu-nanovdb/commits": commits,
    "pmndrs/vdb/issues?state=open": [],
    ...overrides,
  });
  const fake = (table) => {
    const calls = [];
    const request = async (url, opts = {}) => {
      calls.push({ url, method: opts.method ?? "GET", body: opts.body });
      if (opts.method && opts.method !== "GET") return { html_url: "https://github.com/pmndrs/vdb/issues/99" };
      const key = Object.keys(table).find((k) => url.includes(k));
      if (!key) throw new Error(`unexpected request ${url}`);
      if (table[key] instanceof Error) throw table[key];
      return table[key];
    };
    return { request, calls };
  };
  const run = (argv, table, env = { GITHUB_TOKEN: "t" }) => {
    const { request, calls } = fake(table);
    const log = [];
    return main([...argv, "--pins", pinsFile], { request, env, log: (s) => log.push(s), now: () => new Date("2026-11-28T00:00:00Z") }).then(
      (code) => ({ code, calls, log: log.join("\n") }),
    );
  };
  const writes = (calls) => calls.filter((c) => c.method !== "GET");

  it("--dry-run reads only, needs no token, and prints the plan", async () => {
    const { code, calls, log } = await run(["--dry-run"], responses(), {});
    expect(code).toBe(0);
    expect(writes(calls)).toEqual([]);
    expect(log).toMatch(/mjurczyk-tag\s+v0\.29\.0\s+v0\.30\.0\s+MOVED/);
    expect(log).toMatch(/dry run: would create/);
  });

  it("without --dry-run it needs a token", async () => {
    await expect(run([], responses(), {})).rejects.toThrow(/GITHUB_TOKEN/);
  });

  it("opens one issue with the stable title when something moved", async () => {
    const { code, calls } = await run([], responses());
    expect(code).toBe(0);
    const w = writes(calls);
    expect(w).toHaveLength(1);
    expect(w[0]).toMatchObject({ method: "POST", url: "https://api.github.com/repos/pmndrs/vdb/issues" });
    expect(w[0].body.title).toBe(pins.issue.title);
  });

  it("updates the open issue (body + comment) when the moved set changed, and skips pull requests", async () => {
    const issues = [
      { number: 3, title: pins.issue.title, pull_request: {}, body: "" },
      { number: 5, title: pins.issue.title, body: "<!-- upstream-watch digest=0000 -->", html_url: "i5" },
    ];
    const { calls } = await run([], responses({ "pmndrs/vdb/issues?state=open": issues }));
    expect(writes(calls).map((c) => `${c.method} ${c.url.replace("https://api.github.com/repos/pmndrs/vdb", "")}`)).toEqual([
      "PATCH /issues/5",
      "POST /issues/5/comments",
    ]);
  });

  it("closes the open issue when every pin matches again; a failed watch exits 1 and keeps it open", async () => {
    const atPins = {
      "mjurczyk/openvdb/tags": tags.filter((t) => t.name !== "v0.30.0"),
      "registry.npmjs.org/openvdb": { ...npm, "dist-tags": { latest: "0.29.0" }, versions: { "0.29.0": {} } },
      "registry.npmjs.org/three-openvdb": { "dist-tags": { latest: "0.29.0" }, versions: { "0.29.0": {} } },
      "registry.npmjs.org/webvdb": { "dist-tags": { latest: "0.29.0" }, versions: { "0.29.0": {} } },
      "pmndrs/vdb/issues?state=open": [{ number: 5, title: pins.issue.title, body: "", html_url: "i5" }],
    };
    const closed = await run([], responses(atPins));
    expect(closed.code).toBe(0);
    expect(writes(closed.calls).map((c) => c.method + " " + JSON.stringify(c.body))).toEqual([
      expect.stringMatching(/^POST .*matches its pin/),
      'PATCH {"state":"closed","state_reason":"completed"}',
    ]);
    const failed = await run([], responses({ ...atPins, "emcfarlane/webgpu-nanovdb/commits": new Error("HTTP 403 rate limit") }));
    expect(failed.code).toBe(1);
    expect(writes(failed.calls)).toEqual([]);
    expect(failed.log).toMatch(/emcfarlane-commit .*ERROR HTTP 403/);
  });
});
