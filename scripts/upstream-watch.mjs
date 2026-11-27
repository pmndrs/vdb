#!/usr/bin/env node
// K7.6 upstream watch. Compares the pins in .github/upstream-pins.json with upstream (GitHub releases, tags and
// commits; npm versions) and keeps ONE open issue on pmndrs/vdb with a stable title in step with what moved:
// create it, update it (body, plus a comment when the moved set changed), or close it once every pin matches again.
//
//   node scripts/upstream-watch.mjs --dry-run       # fetch and compare, print the report and the issue action; no writes
//   GITHUB_TOKEN=… node scripts/upstream-watch.mjs   # what .github/workflows/upstream-watch.yml runs
//
// Options: --pins <path> (default .github/upstream-pins.json), --repo <owner/name> (default: GITHUB_REPOSITORY, else
// the pins file's issue.repo). Reads use GITHUB_TOKEN / GH_TOKEN when set (rate limit), else go unauthenticated.
// Exit 1 when a watch could not be checked (the issue is still kept in step with the watches that could).
// The comparison and planning functions are pure and exported for test/scripts/upstream-watch.test.mjs.

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_PINS = path.join(ROOT, ".github/upstream-pins.json");
const GITHUB_API = "https://api.github.com";
const NPM_REGISTRY = "https://registry.npmjs.org";
const USER_AGENT = "pmndrs-vdb-upstream-watch";
const MARKER = "upstream-watch";
/** How many commits / releases / tags one check reads (one page). */
export const PAGE = 30;

// ---------------------------------------------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------------------------------------------

/** `v1.2.3` / `1.2.3` / `1.2.3-rc.1` → { major, minor, patch, pre } or null (other tag names). */
export function parseSemver(text) {
  const m = /^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/.exec(String(text).trim());
  if (!m) return null;
  return { major: Number(m[1]), minor: Number(m[2]), patch: Number(m[3]), pre: m[4] ?? "" };
}

/** Semver order (a pre-release sorts below its release; pre-release identifiers compare as strings). */
export function compareSemver(a, b) {
  const x = typeof a === "string" ? parseSemver(a) : a;
  const y = typeof b === "string" ? parseSemver(b) : b;
  if (!x || !y) throw new Error(`not a version: ${!x ? a : b}`);
  for (const k of ["major", "minor", "patch"]) if (x[k] !== y[k]) return x[k] - y[k];
  if (x.pre === y.pre) return 0;
  if (!x.pre) return 1;
  if (!y.pre) return -1;
  return x.pre < y.pre ? -1 : 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Comparisons (pure: a watch from the pins file + the API response → a result)
// ---------------------------------------------------------------------------------------------------------------

/**
 * @typedef {{ name: string, date?: string, url?: string, detail?: string }} UpstreamItem
 * @typedef {{ id: string, name: string, kind: string, source: string, pin: string, latest: string | null,
 *   moved: boolean, newer: UpstreamItem[], note?: string, review?: string, error?: string }} WatchResult
 */

function base(watch, source) {
  return { id: watch.id, name: watch.name ?? watch.id, kind: watch.kind, source, pin: watch.pin, review: watch.review };
}

/** GitHub `GET /repos/{repo}/releases`: releases published after the pinned one (drafts and pre-releases skipped). */
export function compareGithubReleases(watch, releases) {
  const out = base(watch, `https://github.com/${watch.repo}/releases`);
  const list = releases
    .filter((r) => !r.draft && (watch.includePrereleases || !r.prerelease))
    .map((r) => ({ name: r.tag_name, date: r.published_at ?? r.created_at, url: r.html_url }))
    .sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  const pinned = list.find((r) => r.name === watch.pin);
  let newer;
  let note;
  if (pinned) {
    newer = list.filter((r) => r.name !== watch.pin && r.date > pinned.date);
  } else {
    // The pin fell off the page (or was deleted): fall back to version order.
    note = `pin ${watch.pin} is not among the latest ${releases.length} releases; compared by version`;
    newer = list.filter((r) => parseSemver(r.name) && parseSemver(watch.pin) && compareSemver(r.name, watch.pin) > 0);
  }
  return { ...out, latest: list[0]?.name ?? null, moved: newer.length > 0, newer, ...(note ? { note } : {}) };
}

/** GitHub `GET /repos/{repo}/tags`: version tags above the pin (other tag names ignored). */
export function compareGithubTags(watch, tags) {
  const out = base(watch, `https://github.com/${watch.repo}/tags`);
  const versions = tags
    .filter((t) => parseSemver(t.name) && (watch.includePrereleases || !parseSemver(t.name).pre))
    .sort((a, b) => compareSemver(b.name, a.name));
  const newer = versions
    .filter((t) => compareSemver(t.name, watch.pin) > 0)
    .map((t) => ({
      name: t.name,
      detail: t.commit?.sha ? t.commit.sha.slice(0, 7) : undefined,
      url: `https://github.com/${watch.repo}/releases/tag/${encodeURIComponent(t.name)}`,
    }));
  return { ...out, latest: versions[0]?.name ?? null, moved: newer.length > 0, newer };
}

/** npm abbreviated packument (`GET /{package}`, install-v1): versions above the pin, and `latest`. */
export function compareNpm(watch, packument) {
  const out = base(watch, `https://www.npmjs.com/package/${watch.package}`);
  const latest = packument["dist-tags"]?.latest ?? null;
  const newer = Object.keys(packument.versions ?? {})
    .filter((v) => parseSemver(v) && (watch.includePrereleases || !parseSemver(v).pre))
    .filter((v) => compareSemver(v, watch.pin) > 0)
    .sort((a, b) => compareSemver(b, a))
    .map((v) => ({ name: v, url: `https://www.npmjs.com/package/${watch.package}/v/${v}` }));
  let note;
  if (latest && parseSemver(latest) && compareSemver(latest, watch.pin) < 0) {
    note = `dist-tag latest (${latest}) is below the pin`;
  }
  return { ...out, latest, moved: newer.length > 0, newer, ...(note ? { note } : {}) };
}

/** GitHub `GET /repos/{repo}/commits?sha={branch}` (newest first): commits after the pinned one. */
export function compareGithubCommits(watch, commits) {
  const out = base(watch, `https://github.com/${watch.repo}/commits/${watch.branch ?? ""}`);
  const at = commits.findIndex((c) => c.sha === watch.pin || (watch.pin.length >= 7 && c.sha.startsWith(watch.pin)));
  const take = at === -1 ? commits : commits.slice(0, at);
  const newer = take.map((c) => ({
    name: c.sha.slice(0, 7),
    date: c.commit?.committer?.date ?? c.commit?.author?.date,
    detail: (c.commit?.message ?? "").split("\n")[0],
    url: c.html_url,
  }));
  const note =
    at === -1
      ? `pin ${watch.pin.slice(0, 7)} is not in the latest ${commits.length} commits of ${watch.branch ?? "the default branch"} (more commits, or a force-push)`
      : undefined;
  return {
    ...out,
    latest: commits[0]?.sha ?? null,
    moved: at !== 0 && commits.length > 0,
    newer,
    ...(note ? { note } : {}),
  };
}

const COMPARE = {
  "github-release": compareGithubReleases,
  "github-tag": compareGithubTags,
  npm: compareNpm,
  "github-commit": compareGithubCommits,
};

/** A watch's request (the URL whose JSON the matching compare function takes). */
export function watchRequest(watch) {
  switch (watch.kind) {
    case "github-release":
      return { url: `${GITHUB_API}/repos/${watch.repo}/releases?per_page=${PAGE}`, github: true };
    case "github-tag":
      return { url: `${GITHUB_API}/repos/${watch.repo}/tags?per_page=100`, github: true };
    case "npm":
      return {
        url: `${NPM_REGISTRY}/${watch.package.replace("/", "%2F")}`,
        headers: { accept: "application/vnd.npm.install-v1+json" },
      };
    case "github-commit": {
      const sha = watch.branch ? `&sha=${encodeURIComponent(watch.branch)}` : "";
      return { url: `${GITHUB_API}/repos/${watch.repo}/commits?per_page=${PAGE}${sha}`, github: true };
    }
    default:
      throw new Error(`unknown watch kind '${watch.kind}' (${watch.id})`);
  }
}

/** Compare one watch with its fetched JSON. */
export function compareWatch(watch, json) {
  const fn = COMPARE[watch.kind];
  if (!fn) throw new Error(`unknown watch kind '${watch.kind}' (${watch.id})`);
  return fn(watch, json);
}

/** Validate the pins file; returns its watches. */
export function validatePins(pins) {
  if (!pins || !Array.isArray(pins.watches)) throw new Error("pins: 'watches' must be an array");
  if (!pins.issue?.title || !pins.issue?.repo) throw new Error("pins: 'issue.repo' and 'issue.title' are required");
  const ids = new Set();
  for (const w of pins.watches) {
    if (!w.id || ids.has(w.id)) throw new Error(`pins: missing or duplicate id '${w.id}'`);
    ids.add(w.id);
    if (!COMPARE[w.kind]) throw new Error(`pins: unknown kind '${w.kind}' (${w.id})`);
    if (typeof w.pin !== "string" || !w.pin) throw new Error(`pins: '${w.id}' needs a pin`);
    if (w.kind === "npm" ? !w.package : !w.repo) throw new Error(`pins: '${w.id}' needs a ${w.kind === "npm" ? "package" : "repo"}`);
    if ((w.kind === "github-release" || w.kind === "github-tag" || w.kind === "npm") && !parseSemver(w.pin)) {
      throw new Error(`pins: '${w.id}' pin '${w.pin}' is not a version`);
    }
  }
  return pins.watches;
}

// ---------------------------------------------------------------------------------------------------------------
// Report and issue plan
// ---------------------------------------------------------------------------------------------------------------

/** A stable digest of what moved (ids and new item names only; no dates), kept in the issue body. */
export function digestOf(results) {
  const moved = results
    .filter((r) => r.moved)
    .map((r) => [r.id, r.newer.map((n) => n.name)])
    .sort((a, b) => (a[0] < b[0] ? -1 : 1));
  return createHash("sha256").update(JSON.stringify(moved)).digest("hex").slice(0, 16);
}

/** The digest an issue body carries, or null. */
export function digestInBody(body) {
  const m = new RegExp(`<!-- ${MARKER} digest=([0-9a-f]+) -->`).exec(body ?? "");
  return m ? m[1] : null;
}

function itemLine(item) {
  const label = item.url ? `[${item.name}](${item.url})` : `\`${item.name}\``;
  const extra = [item.date ? item.date.slice(0, 10) : "", item.detail ?? ""].filter(Boolean).join(", ");
  return `- ${label}${extra ? ` (${extra.replace(/\|/g, "\\|")})` : ""}`;
}

/** The issue body for the moved watches. */
export function renderIssueBody(results, { checkedAt, runUrl, pinsPath = ".github/upstream-pins.json" } = {}) {
  const moved = results.filter((r) => r.moved);
  const checked = [checkedAt ? `checked ${checkedAt}` : "", runUrl ? `[run](${runUrl})` : ""].filter(Boolean).join(", ");
  const lines = [
    `<!-- ${MARKER} digest=${digestOf(results)} -->`,
    `Upstream moved past the pins in \`${pinsPath}\`${checked ? ` (${checked})` : ""}.`,
    "",
  ];
  for (const r of moved) {
    const short = (s) => (r.kind === "github-commit" && s ? s.slice(0, 7) : s);
    lines.push(`### ${r.name}`, "", `Pinned \`${short(r.pin)}\`, latest \`${short(r.latest)}\` · [source](${r.source})`, "");
    const shown = r.newer.slice(0, 20);
    lines.push(...shown.map(itemLine));
    if (r.newer.length > shown.length) lines.push(`- … ${r.newer.length - shown.length} more`);
    if (r.note) lines.push("", `Note: ${r.note}.`);
    if (r.review) lines.push("", `Review: ${r.review}`);
    lines.push("");
  }
  lines.push(
    "After the review, bump each pin (and its `reviewed` date) in the pins file; the next run closes this issue once",
    "every pin matches. Written by `scripts/upstream-watch.mjs` (K7.6).",
  );
  return `${lines.join("\n")}\n`;
}

/** A plain-text table of every watch, for the log. */
export function renderSummary(results) {
  const rows = results.map((r) => {
    const short = (s) => (r.kind === "github-commit" && s ? s.slice(0, 7) : (s ?? "?"));
    const status = r.error ? `ERROR ${r.error}` : r.moved ? `MOVED (${r.newer.length} new)` : "ok";
    return [r.id, short(r.pin), short(r.latest), status];
  });
  const head = ["watch", "pin", "latest", "status"];
  const width = head.map((h, i) => Math.max(h.length, ...rows.map((row) => String(row[i]).length)));
  const fmt = (row) => row.map((c, i) => String(c).padEnd(width[i])).join("  ").trimEnd();
  const out = [fmt(head), ...rows.map(fmt)];
  for (const r of results) {
    if (r.moved) for (const n of r.newer.slice(0, 10)) out.push(`  ${r.id}: ${itemLine(n).slice(2)}`);
    if (r.note) out.push(`  ${r.id}: note: ${r.note}`);
  }
  return out.join("\n");
}

/**
 * What to do with the issue. `existing` is the open issue with the title, or null.
 * @returns {{ action: 'none' | 'create' | 'update' | 'close', comment?: string, reason: string }}
 */
export function planIssue(results, existing) {
  const moved = results.filter((r) => r.moved);
  const errors = results.filter((r) => r.error);
  if (moved.length === 0) {
    if (!existing) return { action: "none", reason: "every pin matches upstream; no open issue" };
    if (errors.length > 0) return { action: "none", reason: "no movement seen, but some watches failed: issue left open" };
    return {
      action: "close",
      comment: "Every watched upstream matches its pin in `.github/upstream-pins.json` again. Closing.",
      reason: `every pin matches upstream: close #${existing.number}`,
    };
  }
  if (!existing) return { action: "create", reason: `${moved.length} watch(es) moved: open a new issue` };
  const digest = digestOf(results);
  if (digestInBody(existing.body) === digest) {
    return { action: "none", reason: `#${existing.number} already lists these changes` };
  }
  const newOnes = moved.map((r) => `- ${r.name}: ${r.newer.map((n) => `\`${n.name}\``).slice(0, 10).join(", ")}`);
  return {
    action: "update",
    comment: `The watched upstreams moved again; the description is updated.\n\n${newOnes.join("\n")}`,
    reason: `the moved set changed: update #${existing.number}`,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Network (injectable for tests)
// ---------------------------------------------------------------------------------------------------------------

/** A JSON HTTP client over global fetch. */
export function fetchClient({ token } = {}) {
  return async function request(url, { method = "GET", headers = {}, body, github = url.startsWith(GITHUB_API) } = {}) {
    const h = { "user-agent": USER_AGENT, ...headers };
    if (github) {
      h.accept ??= "application/vnd.github+json";
      h["x-github-api-version"] = "2022-11-28";
      if (token) h.authorization = `Bearer ${token}`;
    }
    if (body !== undefined) h["content-type"] = "application/json";
    const res = await fetch(url, { method, headers: h, body: body === undefined ? undefined : JSON.stringify(body) });
    const text = await res.text();
    if (!res.ok) {
      let message = text.slice(0, 200);
      try {
        message = JSON.parse(text).message ?? message;
      } catch {}
      throw new Error(`${method} ${url}: HTTP ${res.status} ${message}`);
    }
    return text ? JSON.parse(text) : null;
  };
}

/** Fetch and compare every watch; a failed watch becomes a result with `error`. */
export async function checkWatches(watches, request) {
  return Promise.all(
    watches.map(async (w) => {
      try {
        const req = watchRequest(w);
        return compareWatch(w, await request(req.url, { headers: req.headers, github: req.github }));
      } catch (e) {
        return { ...base(w, ""), latest: null, moved: false, newer: [], error: e instanceof Error ? e.message : String(e) };
      }
    }),
  );
}

/** The open issue (not a pull request) with exactly this title, or null. */
export async function findIssue(request, repo, title) {
  for (let page = 1; page <= 10; page++) {
    const list = await request(`${GITHUB_API}/repos/${repo}/issues?state=open&per_page=100&page=${page}`);
    const hit = list.find((i) => !i.pull_request && i.title === title);
    if (hit) return { number: hit.number, body: hit.body ?? "", url: hit.html_url };
    if (list.length < 100) return null;
  }
  return null;
}

/** Carry out a plan (never called in --dry-run). */
export async function applyPlan(request, repo, title, plan, existing, body) {
  const api = `${GITHUB_API}/repos/${repo}/issues`;
  if (plan.action === "create") {
    const made = await request(api, { method: "POST", body: { title, body } });
    return made.html_url;
  }
  if (plan.action === "update") {
    await request(`${api}/${existing.number}`, { method: "PATCH", body: { body } });
    await request(`${api}/${existing.number}/comments`, { method: "POST", body: { body: plan.comment } });
    return existing.url;
  }
  if (plan.action === "close") {
    await request(`${api}/${existing.number}/comments`, { method: "POST", body: { body: plan.comment } });
    await request(`${api}/${existing.number}`, { method: "PATCH", body: { state: "closed", state_reason: "completed" } });
    return existing.url;
  }
  return existing?.url ?? null;
}

// ---------------------------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------------------------

export function parseArgs(argv) {
  const opts = { dryRun: false, pins: DEFAULT_PINS, repo: undefined };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dry-run") opts.dryRun = true;
    else if (a === "--pins") opts.pins = path.resolve(argv[++i]);
    else if (a.startsWith("--pins=")) opts.pins = path.resolve(a.slice(7));
    else if (a === "--repo") opts.repo = argv[++i];
    else if (a.startsWith("--repo=")) opts.repo = a.slice(7);
    else if (a === "--help" || a === "-h") opts.help = true;
    else throw new Error(`unknown argument '${a}'`);
  }
  return opts;
}

/** Run the watch. Everything external is injectable: `request` (JSON client), `env`, `log`, `now`. */
export async function main(argv, { env = process.env, request, log = console.log, now = () => new Date() } = {}) {
  const opts = parseArgs(argv);
  if (opts.help) {
    log("usage: node scripts/upstream-watch.mjs [--dry-run] [--pins <path>] [--repo <owner/name>]");
    return 0;
  }
  const pins = JSON.parse(await readFile(opts.pins, "utf8"));
  const watches = validatePins(pins);
  const token = env.GITHUB_TOKEN || env.GH_TOKEN || "";
  if (!opts.dryRun && !token) throw new Error("GITHUB_TOKEN (or GH_TOKEN) is required to write the issue; use --dry-run to only read");
  const http = request ?? fetchClient({ token });
  const repo = opts.repo ?? env.GITHUB_REPOSITORY ?? pins.issue.repo;
  const title = pins.issue.title;

  const results = await checkWatches(watches, http);
  log(renderSummary(results));

  const runUrl =
    env.GITHUB_SERVER_URL && env.GITHUB_REPOSITORY && env.GITHUB_RUN_ID
      ? `${env.GITHUB_SERVER_URL}/${env.GITHUB_REPOSITORY}/actions/runs/${env.GITHUB_RUN_ID}`
      : undefined;
  const body = renderIssueBody(results, { checkedAt: now().toISOString().slice(0, 10), runUrl });

  let existing = null;
  let lookupError;
  try {
    existing = await findIssue(http, repo, title);
  } catch (e) {
    lookupError = e instanceof Error ? e.message : String(e);
  }
  if (lookupError && !opts.dryRun) throw new Error(`issue lookup failed: ${lookupError}`);
  const plan = planIssue(results, existing);
  const failed = results.filter((r) => r.error).length;

  log("");
  log(`issue "${title}" on ${repo}: ${existing ? `#${existing.number} open` : lookupError ? `lookup failed (${lookupError})` : "none open"}`);
  if (opts.dryRun) {
    log(`dry run: would ${plan.action} (${plan.reason})`);
    if (plan.action === "create" || plan.action === "update") log(`\n--- issue body ---\n${body}--- end ---`);
    if (plan.comment) log(`\n--- comment ---\n${plan.comment}\n--- end ---`);
  } else {
    const url = await applyPlan(http, repo, title, plan, existing, body);
    log(`${plan.action}: ${plan.reason}${url ? ` (${url})` : ""}`);
  }
  if (failed > 0) {
    log(`\n${failed} watch(es) could not be checked`);
    return 1;
  }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (e) => {
      console.error(e instanceof Error ? e.message : e);
      process.exit(2);
    },
  );
}
