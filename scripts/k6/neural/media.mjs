// K6.N: training media of the 'neural' / 'mlp' multiple-scattering network. PROCEDURAL ONLY: every medium is a density
// function in [0, 1] over a box, made from a seed by this file. No openvdb.org, WDAS, EmberGen, Stanford or gallery
// asset is read here, and none may be added (the trained weights must carry no data licence, D35 (2)).
// The kinds follow the shapes the renderer meets: billowy clouds, wisps, layers, plain shapes, and (K6.N) two kinds
// that stand for the compact sharp-edged "bunny-like" and the rising "smoke-like" media without using their data.
// The density functions are ours (MIT). buildMediumGrid at the end is his scripts/neural-indirect/media.mjs
// buildMediumGrid (mjurczyk/openvdb contributors, MIT, 75e7345), re-written: the shadow density pass's voxels.
import { createRandom, fitGrid } from './features.mjs';

// A hashed lattice, smoothstep-interpolated. Range [-1, 1].
const lattice = (x, y, z, seed) => {
  let h = (Math.imul(x, 0x27d4eb2d) ^ Math.imul(y, 0x165667b1) ^ Math.imul(z, 0x9e3779b1) ^ Math.imul(seed + 1, 0x85ebca6b)) | 0;

  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);

  return ((h ^ (h >>> 15)) >>> 0) / 2147483648 - 1;
};

const fade = (t) => t * t * (3 - 2 * t);
const lerp = (a, b, t) => a + (b - a) * t;

const noise3 = (x, y, z, seed) => {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = fade(x - ix);
  const fy = fade(y - iy);
  const fz = fade(z - iz);
  const plane = (dz) =>
    lerp(
      lerp(lattice(ix, iy, iz + dz, seed), lattice(ix + 1, iy, iz + dz, seed), fx),
      lerp(lattice(ix, iy + 1, iz + dz, seed), lattice(ix + 1, iy + 1, iz + dz, seed), fx),
      fy,
    );

  return lerp(plane(0), plane(1), fz);
};

// Fractal sum of noise3, about [-0.9, 0.9].
const fractal = (x, y, z, seed, octaves = 4, lacunarity = 2.07, gain = 0.5) => {
  let sum = 0;
  let amplitude = 0.5;
  let frequency = 1;

  for (let i = 0; i < octaves; i++) {
    sum += amplitude * noise3(x * frequency, y * frequency, z * frequency, seed + 31 * i);
    amplitude *= gain;
    frequency *= lacunarity;
  }

  return sum;
};

const clamp01 = (value) => Math.min(Math.max(value, 0), 1);

// Soft spheres eroded by noise: billowy cumulus.
const makeCumulus = (random, seed) => {
  const count = 4 + Math.floor(random() * 20);
  const spheres = Array.from({ length: count }, () => ({
    center: [random() * 1.2 - 0.6, (random() * 0.6 - 0.2) * 0.8, random() * 1.2 - 0.6],
    radius: 0.12 + random() * 0.25,
  }));
  const frequency = 3 + random() * 6;
  const erosion = 0.2 + random() * 0.5;
  const softness = 0.05 + random() * 0.2;

  return {
    boxMin: [-1, -0.6, -1],
    boxMax: [1, 0.8, 1],
    density: (x, y, z) => {
      let field = -Infinity;

      for (const { center, radius } of spheres) {
        field = Math.max(field, 1 - Math.hypot(x - center[0], y - center[1], z - center[2]) / radius);
      }

      return clamp01((field + fractal(x * frequency, y * frequency, z * frequency, seed) * erosion) / softness);
    },
  };
};

// Thresholded noise in a soft box: wisps and patchy fog.
const makeWisps = (random, seed) => {
  const frequency = 2 + random() * 5;
  const threshold = random() * 0.25;
  const gain = 2 + random() * 6;

  return {
    boxMin: [-1, -1, -1],
    boxMax: [1, 1, 1],
    density: (x, y, z) => {
      const edge = clamp01((1 - Math.max(Math.abs(x), Math.abs(y), Math.abs(z))) * 4);

      return clamp01((fractal(x * frequency, y * frequency, z * frequency, seed, 5) - threshold) * gain) * edge;
    },
  };
};

// A flat layer with a noisy top and bottom: stratus.
const makeLayer = (random, seed) => {
  const thickness = 0.05 + random() * 0.35;
  const frequency = 2 + random() * 4;
  const roughness = random() * 0.15;

  return {
    boxMin: [-1, -0.5, -1],
    boxMax: [1, 0.5, 1],
    density: (x, y, z) => {
      const top = thickness + fractal(x * frequency, 0, z * frequency, seed) * roughness;
      const bottom = -thickness + fractal(x * frequency, 5, z * frequency, seed + 3) * roughness;

      return clamp01(Math.min(top - y, y - bottom) * 30) * clamp01((1 - Math.max(Math.abs(x), Math.abs(z))) * 10);
    },
  };
};

// A sphere or a box, uniform or noisy: the plain cases (and the analytic ones).
const makeShape = (random, seed) => {
  const isSphere = random() < 0.5;
  const noise = random() < 0.5 ? 0 : random() * 0.8;
  const frequency = 3 + random() * 5;

  return {
    boxMin: [-1, -1, -1],
    boxMax: [1, 1, 1],
    density: (x, y, z) => {
      const inside = isSphere
        ? clamp01((1 - Math.hypot(x, y, z)) * 20)
        : clamp01((1 - Math.max(Math.abs(x), Math.abs(y), Math.abs(z))) * 20);

      return inside * clamp01(1 - noise * (0.5 + 0.5 * fractal(x * frequency, y * frequency, z * frequency, seed)));
    },
  };
};

// Compact sharp-edged bodies: a few stretched ellipsoids joined smoothly, a dense core and a rough skin, with narrow
// features (limbs). Stands for a "solid-looking" cloud such as a statue-shaped volume, with no data from one.
const makeBlobs = (random, seed) => {
  const count = 2 + Math.floor(random() * 6);
  const bodies = Array.from({ length: count }, (_, i) => ({
    center: [(random() - 0.5) * (i === 0 ? 0.2 : 1.1), (random() - 0.5) * (i === 0 ? 0.2 : 1.1), (random() - 0.5) * (i === 0 ? 0.2 : 1.1)],
    radii: [0.15 + random() * (i === 0 ? 0.5 : 0.25), 0.1 + random() * (i === 0 ? 0.5 : 0.25), 0.15 + random() * (i === 0 ? 0.5 : 0.25)],
  }));
  const sharpness = 6 + random() * 24;
  const skin = random() * 0.25;
  const frequency = 4 + random() * 9;
  const coreFalloff = random() * 2;

  return {
    boxMin: [-1, -1, -1],
    boxMax: [1, 1, 1],
    density: (x, y, z) => {
      let field = 0;

      for (const { center, radii } of bodies) {
        const d = Math.hypot((x - center[0]) / radii[0], (y - center[1]) / radii[1], (z - center[2]) / radii[2]);

        field = Math.max(field, 1 - d);
      }

      const rough = field + skin * fractal(x * frequency, y * frequency, z * frequency, seed, 3);

      return clamp01(rough * sharpness) * (1 - coreFalloff * 0.25 * clamp01(1 - field * 2) ** 2);
    },
  };
};

// A rising column bent by a warped noise field: smoke and flame-like plumes (high contrast, thin sheets).
const makePlume = (random, seed) => {
  const frequency = 1.5 + random() * 3;
  const warp = 0.2 + random() * 0.5;
  const width = 0.15 + random() * 0.3;
  const gain = 1.5 + random() * 4;
  const taper = random();

  return {
    boxMin: [-1, -1, -1],
    boxMax: [1, 1, 1],
    density: (x, y, z) => {
      const wx = x + warp * fractal(x * frequency, y * frequency, z * frequency, seed, 3);
      const wz = z + warp * fractal(x * frequency, y * frequency, z * frequency, seed + 101, 3);
      const radius = width * (1 + taper * (y + 1) * 0.8);
      const core = 1 - Math.hypot(wx, wz) / radius;
      const detail = fractal(wx * frequency * 3, y * frequency * 3, wz * frequency * 3, seed + 7, 4);
      const ends = clamp01((1 - Math.abs(y)) * 5);

      return clamp01((core + detail * 0.6) * gain) * ends;
    },
  };
};

// Many small puffs on a coarse cloud: a sparse, high-detail medium with real empty space (fine structure at a
// few voxels, the case the box mips of the features are for).
const makePuffs = (random, seed) => {
  const frequency = 5 + random() * 8;
  const threshold = 0.1 + random() * 0.4;
  const coarse = 1 + random() * 2;
  const gain = 3 + random() * 8;

  return {
    boxMin: [-1, -1, -1],
    boxMax: [1, 1, 1],
    density: (x, y, z) => {
      const envelope = fractal(x * coarse, y * coarse, z * coarse, seed + 5, 2) + 0.55 - Math.hypot(x, y * 1.3, z) * 0.6;
      const fine = Math.abs(fractal(x * frequency, y * frequency, z * frequency, seed, 3)) * 2;

      return clamp01((envelope * 1.5 - fine * threshold) * gain) * clamp01((1 - Math.max(Math.abs(x), Math.abs(y), Math.abs(z))) * 6);
    },
  };
};

export const mediumKinds = ['cumulus', 'wisps', 'layer', 'shape', 'blobs', 'plume', 'puffs'];

/** A medium of a kind, from a seed. */
export const createMedium = (kind, seed) => {
  const random = createRandom(seed * 7919 + 13);

  switch (kind) {
    case 'cumulus':
      return makeCumulus(random, seed);
    case 'wisps':
      return makeWisps(random, seed);
    case 'layer':
      return makeLayer(random, seed);
    case 'shape':
      return makeShape(random, seed);
    case 'blobs':
      return makeBlobs(random, seed);
    case 'plume':
      return makePlume(random, seed);
    case 'puffs':
      return makePuffs(random, seed);
    default:
      throw new Error(`Unknown medium ${kind}`);
  }
};

// The world shadow grid of a medium, as the shadow density pass writes it: extinction and absorption at the voxel
// centres, each the mean of 2 x 2 x 2 density samples. voxelDepth: the largest optical depth of one voxel (grid.unit),
// albedo: one value for the whole medium.
export const buildMediumGrid = (medium, { voxelDepth, albedo, offset = [0, 0, 0] }) => {
  const boxMin = medium.boxMin.map((value, axis) => value + offset[axis]);
  const boxMax = medium.boxMax.map((value, axis) => value + offset[axis]);
  const grid = fitGrid(boxMin, boxMax);
  const { resolution, voxel, min } = grid;
  const density = new Float32Array(resolution ** 3);
  let maxDensity = 0;

  for (let z = 0; z < resolution; z++) {
    for (let y = 0; y < resolution; y++) {
      for (let x = 0; x < resolution; x++) {
        let sum = 0;

        for (let i = 0; i < 8; i++) {
          const px = min[0] + (x + 0.25 + 0.5 * (i & 1)) * voxel[0];
          const py = min[1] + (y + 0.25 + 0.5 * ((i >> 1) & 1)) * voxel[1];
          const pz = min[2] + (z + 0.25 + 0.5 * ((i >> 2) & 1)) * voxel[2];
          const inside =
            px >= boxMin[0] && px <= boxMax[0] && py >= boxMin[1] && py <= boxMax[1] && pz >= boxMin[2] && pz <= boxMax[2];

          sum += inside ? medium.density(px - offset[0], py - offset[1], pz - offset[2]) : 0;
        }

        density[x + (y + z * resolution) * resolution] = sum / 8;
        maxDensity = Math.max(maxDensity, sum / 8);
      }
    }
  }

  const scale = maxDensity > 0 ? voxelDepth / (maxDensity * grid.unit) : 0;
  const data = new Float32Array(resolution ** 3 * 2);

  for (let i = 0; i < density.length; i++) {
    data[i * 2] = density[i] * scale;
    data[i * 2 + 1] = density[i] * scale * (1 - albedo);
  }

  return { grid, data };
};
