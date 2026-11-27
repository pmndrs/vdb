import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// THIRD_PARTY audit (2026-11-28): K6.8 and K6.11 deleted ported files that THIRD_PARTY.md and the shipped
// THIRD_PARTY_NOTICES.md still listed. These checks keep both lists in step with the source headers:
//   - every path in THIRD_PARTY.md's "Where it lives" column and in the notice's file lists exists;
//   - every source file whose header says it derives from mjurczyk/openvdb is in the shipped notice.

const pkgRoot = fileURLToPath(new URL("../", import.meta.url));
const repoRoot = path.resolve(pkgRoot, "../..");
const notices = readFileSync(path.join(pkgRoot, "THIRD_PARTY_NOTICES.md"), "utf8");
/** THIRD_PARTY.md's "Where it lives" column (the third cell of each table row). */
const whereItLives = readFileSync(path.join(repoRoot, "THIRD_PARTY.md"), "utf8")
  .split("\n")
  .filter((l) => l.startsWith("| ") && !l.startsWith("| Component"))
  .map((l) => l.split(" | ")[2] ?? "")
  .join("\n");
/** The shipped notice's file lists (its `- \`src/…\`` items). */
const noticeItems = notices
  .split("\n")
  .filter((l) => l.startsWith("- `src/"))
  .join("\n");

/** `a/{b,c}.ts` → `a/b.ts`, `a/c.ts`. */
function expand(span: string): string[] {
  const m = /\{([^{}]+)\}/.exec(span);
  if (!m) return [span];
  return m[1]!.split(",").flatMap((p) => expand(span.slice(0, m.index) + p + span.slice(m.index + m[0].length)));
}

/** Back-ticked paths that start with `prefix`, brace lists expanded. */
function pathsIn(text: string, prefix: string): string[] {
  return [...text.matchAll(/`([^`\s]+)`/g)].flatMap((m) => expand(m[1]!)).filter((p) => p.startsWith(prefix));
}

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    return statSync(full).isDirectory() ? walk(full) : /\.ts$/.test(name) ? [full] : [];
  });
}

describe("third-party lists", () => {
  it("every package path in THIRD_PARTY.md and THIRD_PARTY_NOTICES.md exists", () => {
    const missing = [
      ...pathsIn(whereItLives, "packages/").filter((p) => !existsSync(path.join(repoRoot, p))),
      ...pathsIn(whereItLives, "scripts/").filter((p) => !existsSync(path.join(repoRoot, p))),
      ...pathsIn(noticeItems, "src/").filter((p) => !existsSync(path.join(pkgRoot, p))),
    ];
    expect(missing).toEqual([]);
  });

  it("every source file derived from mjurczyk/openvdb is listed in the shipped notice", () => {
    const listed = new Set(pathsIn(noticeItems, "src/"));
    const derived = walk(path.join(pkgRoot, "src"))
      // The header's attribution ("Portions derived from mjurczyk/openvdb (MIT)", mie-table's "a port of …"), with
      // comment line breaks folded. neural-weights.ts (ours, D36) names his trainer differently and has its own section.
      .filter((f) => {
        const head = readFileSync(f, "utf8").slice(0, 4000).replace(/\s*\n\s*(\*|\/\/)?\s*/g, " ");
        return /(derived from|a port of) mjurczyk\/openvdb \(MIT\)/.test(head);
      })
      .map((f) => path.relative(pkgRoot, f).split(path.sep).join("/"));
    expect(derived.length).toBeGreaterThan(40);
    expect(derived.filter((f) => !listed.has(f))).toEqual([]);
  });
});
