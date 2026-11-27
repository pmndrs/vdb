# Native probe sidecars (K7.7: K4.10, K4.15, K4.17)

Reference values that native OpenVDB / NanoVDB **v14.0.0** reads, written by our native programs
(`docker/fixture-bake/vdb_probe.cpp`, `nvdb_probe.cpp`, MIT) and checked by `test/tools/native-probes.test.ts`.
They hold measurements only (probe values, counts, boxes, stats, hashes), never grid or point arrays.

| File | Source | Program |
|---|---|---|
| `bunny_cloud.density.json` | `fixtures/vdb-samples/bunny_cloud.vdb` (Stanford derivative, not fetched; see `fixtures/ATTRIBUTION.md`) | `vdb_probe float` |
| `explosion.v.json` | `explosion.vdb` grid `v` (openvdb.org, CC BY-SA 4.0) | `vdb_probe vec3` |
| `waterfall_points.json` | `waterfall_points.vdb` (openvdb.org, CC BY-SA 4.0) | `vdb_probe points` |
| `coarse-first.json` | our `coarse-first-source.mjs` file (codec NONE and ZIP) | `nvdb_probe`, `nanovdb_validate`, `nanovdb_convert` |

The tests for the first three skip when the git-ignored sample is absent; the coarse-first test always runs (it
rebuilds the file and checks its SHA-256 first).

## Regenerating

```sh
sh scripts/native/build-native-v14.sh              # once: ~/.cache/pmndrs-vdb-native (no Docker)
pnpm build && pnpm parity:native --update          # rewrites these sidecars; without --update it compares
```

`--native=docker` runs the same programs in the `docker/fixture-bake` image. Every sidecar is deterministic: a
rerun writes the same bytes.
