// K6.N: the A/B table of the 'neural' weights from reference-report.mjs reports (report.json): per scene group, the
// luminance bias and the median relative luminance error against the path-traced reference (multiple scattering) of
// two-sided, his weights and ours (and diffusion for context), then the means of |bias| and of the median error over the
// groups. Our code (MIT). History (D36): it reads K6.N-era reports, whose rows named the weights ("his" / "pmndrs"); since
// K6.11a reference-report.mjs has one "neural" row (ours): in a new report it lands in the "his" column and "ours" is empty.
// Run: node scripts/k6/neural/ab-table.mjs <report.json> [<report.json> ...]   (one table per file)
import { readFileSync } from 'node:fs';

// His weights saw the bunny and the smoke assets (his training set); only the WDAS cloud is new to them.
const seenByHis = (key) => !/^wdas/.test(key);
const modes = [
  ['diffusion', (l) => /ms diffusion/.test(l) && !/^his /.test(l)],
  ['two-sided', (l) => /two-sided/.test(l) && !/^his /.test(l)],
  ['his', (l) => /neural/.test(l) && !/pmndrs/.test(l) && !/^his /.test(l)],
  ['ours', (l) => /neural.*pmndrs/.test(l)],
];
const pct = (v) => `${(100 * v).toFixed(1)}`;

for (const file of process.argv.slice(2)) {
  const report = JSON.parse(readFileSync(file, 'utf8'));
  const lines = [`### ${file}`, '', '| group | ' + modes.map(([m]) => `${m} bias / median`).join(' | ') + ' |', '|---|' + modes.map(() => '---|').join('')];
  const sums = Object.fromEntries(modes.map(([m]) => [m, { bias: 0, median: 0, n: 0, biasSeen: 0, medianSeen: 0, nSeen: 0, biasNew: 0, medianNew: 0, nNew: 0 }]));
  const wins = { ours: 0, his: 0, twoSided: 0, groups: 0 };

  for (const group of report.groups) {
    const cells = [];
    let complete = true;
    const found = {};

    for (const [mode, test] of modes) {
      const row = group.rows.find((r) => r.kind === 'core' && test(r.label) && r.vsFull);

      if (!row) {
        cells.push('-');
        if (mode !== 'diffusion') {
          complete = false;
        }
        continue;
      }

      found[mode] = row.vsFull;
      cells.push(`${pct(row.vsFull.lumBias)} / ${pct(row.vsFull.medianRelLum)}`);
    }

    lines.push(`| ${group.key} | ${cells.join(' | ')} |`);

    if (complete) {
      wins.groups++;
      wins.ours += Math.abs(found.ours.lumBias) < Math.abs(found.his.lumBias) ? 1 : 0;
      wins.his += Math.abs(found.his.lumBias) < Math.abs(found.ours.lumBias) ? 1 : 0;
      wins.twoSided += Math.abs(found['two-sided'].lumBias) < Math.min(Math.abs(found.ours.lumBias), Math.abs(found.his.lumBias)) ? 1 : 0;

      for (const [mode] of modes) {
        const s = sums[mode];
        const v = found[mode];

        if (!v) {
          continue;
        }

        s.bias += Math.abs(v.lumBias);
        s.median += v.medianRelLum;
        s.n++;

        if (seenByHis(group.key)) {
          s.biasSeen += Math.abs(v.lumBias);
          s.medianSeen += v.medianRelLum;
          s.nSeen++;
        } else {
          s.biasNew += Math.abs(v.lumBias);
          s.medianNew += v.medianRelLum;
          s.nNew++;
        }
      }
    }
  }

  lines.push('', '| set | ' + modes.map(([m]) => `${m} mean |bias| / mean median`).join(' | ') + ' |', '|---|' + modes.map(() => '---|').join(''));

  const cell = (s, b, m, n) => (s[n] ? `${pct(s[b] / s[n])} / ${pct(s[m] / s[n])}` : '-');

  lines.push(`| all groups with every column | ${modes.map(([m]) => cell(sums[m], 'bias', 'median', 'n')).join(' | ')} |`);
  lines.push(`| bunny / smoke groups (his weights saw these) | ${modes.map(([m]) => cell(sums[m], 'biasSeen', 'medianSeen', 'nSeen')).join(' | ')} |`);
  lines.push(`| WDAS groups (new to his weights) | ${modes.map(([m]) => cell(sums[m], 'biasNew', 'medianNew', 'nNew')).join(' | ')} |`);
  lines.push('', `groups: ${wins.groups}; |bias| smaller for ours than his in ${wins.ours}, his than ours in ${wins.his}; two-sided smaller than both in ${wins.twoSided}.`, '');
  console.log(lines.join('\n'));
}
