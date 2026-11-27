// Runs the native OpenVDB / NanoVDB v14.0.0 tools (K7.7) either from the local cache that
// scripts/native/build-native-v14.sh builds (no Docker) or from the docker/fixture-bake image.
//
//   const tools = await nativeTools({ mode: "auto" | "local" | "docker" });
//   tools.run("nanovdb_print", ["-l", "/abs/path/in/repo.nvdb"]);   // -> { status, stdout, stderr }
//
// Paths are absolute host paths under the repo root; the Docker runner mounts the root at /repo and
// rewrites them. Programs: nanovdb_print, nanovdb_convert, nanovdb_validate, vdb_print, normalize_vdb,
// vdb_probe, nvdb_probe, gen_nvdb_blosc, gen_vdb_half, bake_half.

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const BUILD_SCRIPT = path.join(ROOT, "scripts", "native", "build-native-v14.sh");
export const IMAGE = process.env.FIXTURE_BAKE_IMAGE ?? "webvdb-fixture-bake";
/** `sh -c` script that runs "$0" "$@" with stderr merged into stdout (in order). */
const MERGED = '"$0" "$@" 2>&1';

function localPrefix() {
  const cache = process.env.PMNDRS_VDB_NATIVE ?? path.join(process.env.HOME ?? "", ".cache", "pmndrs-vdb-native");
  return path.join(cache, "v14.0.0", "prefix");
}

function dockerCommand() {
  for (const command of [process.env.DOCKER_BIN, "docker", "/Applications/Docker.app/Contents/Resources/bin/docker"].filter(Boolean)) {
    if (spawnSync(command, ["info"], { stdio: "ignore" }).status === 0) return command;
  }
  return null;
}

function dockerImageExists(docker) {
  return spawnSync(docker, ["image", "inspect", IMAGE], { stdio: "ignore" }).status === 0;
}

/** Local runner: (re)builds the cache first (idempotent; rebuilds our programs when their sources changed). */
function localTools() {
  const build = spawnSync("sh", [BUILD_SCRIPT], { stdio: "inherit", cwd: ROOT });
  if (build.status !== 0) throw new Error("scripts/native/build-native-v14.sh failed");
  const bin = path.join(localPrefix(), "bin");
  return {
    mode: "local",
    bin,
    run(tool, args, { merged = false, ...options } = {}) {
      const [cmd, argv] = merged ? ["sh", ["-c", MERGED, path.join(bin, tool), ...args]] : [path.join(bin, tool), args];
      const result = spawnSync(cmd, argv, { cwd: ROOT, encoding: "utf8", maxBuffer: 1 << 28, stdio: ["ignore", "pipe", "pipe"], ...options });
      if (result.error) throw result.error;
      return result;
    },
  };
}

function dockerTools(docker) {
  const toContainer = (arg) =>
    typeof arg === "string" && (arg === ROOT || arg.startsWith(ROOT + path.sep))
      ? `/repo/${path.relative(ROOT, arg).split(path.sep).join("/")}`
      : arg;
  return {
    mode: "docker",
    bin: null,
    run(tool, args, { merged = false, ...options } = {}) {
      const command = merged ? ["sh", "-c", MERGED, tool, ...args.map(toContainer)] : [tool, ...args.map(toContainer)];
      const result = spawnSync(docker, ["run", "--rm", "-v", `${ROOT}:/repo`, "-w", "/repo", IMAGE, ...command], {
        cwd: ROOT,
        encoding: "utf8",
        maxBuffer: 1 << 28,
        stdio: ["ignore", "pipe", "pipe"],
        ...options,
      });
      if (result.error) throw result.error;
      // Messages name container paths; map them back for the log.
      for (const key of ["stdout", "stderr"]) {
        if (typeof result[key] === "string") result[key] = result[key].split("/repo/").join(`${ROOT}/`);
      }
      return result;
    },
  };
}

/**
 * Picks the runner. `auto`: the local cache when it has the tools, else the Docker image when a
 * daemon runs and the image exists, else a fresh local build.
 */
export function nativeTools({ mode = process.env.NATIVE_MODE ?? "auto" } = {}) {
  if (mode === "local") return localTools();
  if (mode === "docker") {
    const docker = dockerCommand();
    if (!docker) throw new Error("--native=docker: no running Docker daemon");
    if (!dockerImageExists(docker))
      throw new Error(`--native=docker: image ${IMAGE} missing; build it with \`docker build -t ${IMAGE} --build-context generators=packages/vdb/test/tools/fixtures/native-v14/generators docker/fixture-bake\``);
    return dockerTools(docker);
  }
  if (existsSync(path.join(localPrefix(), "bin", "nanovdb_convert"))) return localTools();
  const docker = dockerCommand();
  if (docker && dockerImageExists(docker)) return dockerTools(docker);
  return localTools();
}

/** Runs a tool and throws with its output when it fails. */
export function mustRun(tools, tool, args, options) {
  const result = tools.run(tool, args, options);
  if (result.status !== 0) {
    throw new Error(`${tool} ${args.join(" ")} failed (${result.status}):\n${result.stdout ?? ""}${result.stderr ?? ""}`);
  }
  return result;
}
