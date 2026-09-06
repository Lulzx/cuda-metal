#!/usr/bin/env python3
"""Compare two compiler-produced metallibs with identical inputs and one runner."""
import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--runner', type=pathlib.Path, required=True)
p.add_argument('--before', type=pathlib.Path, required=True)
p.add_argument('--after', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--size', type=int, default=4096)
p.add_argument('--iterations', type=int, default=10)
p.add_argument('--variants', nargs='+', default=['block64_8x8', 'block64_32', 'registers'])
p.add_argument('--max-drift', type=float, default=0.10,
               help='fail if a variant/mode round mean drifts more than this fraction across rounds')
a = p.parse_args()
if not 1 <= a.size <= 8192 or not 1 <= a.iterations <= 1000 or a.max_drift <= 0:
    p.error('size must be 1..8192, iterations 1..1000, max-drift positive')
for path in (a.runner, a.before, a.after):
    if not path.is_file():
        p.error(f'missing artifact: {path}')
a.output.mkdir(parents=True, exist_ok=True)
metadata = dict(size=a.size, iterations=a.iterations, rounds=3,
                artifacts={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (a.runner, a.before, a.after)})
(a.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
rows = []
for repeat in range(3):
    split = repeat % len(a.variants)
    for name in a.variants[split:] + a.variants[:split]:
        order = ['before', 'after'] if repeat % 2 == 0 else ['after', 'before']
        for mode in order:
            library = getattr(a, mode)
            command = [str(a.runner.resolve()), str(library.resolve()),
                       *([str(a.size)] * 3), str(a.iterations), name]
            run = subprocess.run(command, capture_output=True, text=True)
            (a.output / f'{repeat + 1}-{name}-{mode}.log').write_text(run.stdout + run.stderr)
            lines = run.stdout.splitlines()
            if run.returncode or not any(line.startswith(f'check,{name},') and line.endswith('bad=0') for line in lines):
                raise SystemExit(f'FAIL {mode}/{name}: {run.stdout}\n{run.stderr}')
            result = [line for line in lines if line.startswith(f'result,{name},')]
            if len(result) != 1:
                raise SystemExit(f'FAIL {mode}/{name}: expected one timing row')
            fields = result[0].split(',')
            rows.append(dict(round=repeat + 1, name=name, mode=mode, mean_ms=float(fields[6]),
                             min_ms=float(fields[8]), gflops=float(fields[10])))
            (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(mode, result[0], flush=True)
# Sustained GPU load throttles laptops mid-run; a throttled round makes the
# median-of-means compare different clock states, not different compilers.
# Each variant/mode must stay within --max-drift of itself across rounds.
drift_failures = []
summary = ['| Kernel | Before ms | After ms | Speedup | Before min | After min | Drift |',
           '|---|---:|---:|---:|---:|---:|---:|']
for name in a.variants:
    stats = {}
    for mode in ('before', 'after'):
        means = [row['mean_ms'] for row in rows if row['name'] == name and row['mode'] == mode]
        mins = [row['min_ms'] for row in rows if row['name'] == name and row['mode'] == mode]
        stats[mode] = (statistics.median(means), min(mins), max(means) / min(means) - 1)
    before, after = stats['before'], stats['after']
    drift = max(before[2], after[2])
    if drift > a.max_drift:
        drift_failures.append(f'{name}: {100 * drift:.1f}% round-to-round drift')
    summary.append(f'| {name} | {before[0]:.3f} | {after[0]:.3f} | {before[0] / after[0]:.2f}x '
                   f'| {before[1]:.3f} | {after[1]:.3f} | {100 * drift:.1f}% |')
report = '\n'.join(summary) + '\n'
if drift_failures:
    report += '\nFAIL: unstable measurement (throttling or contention): ' + '; '.join(drift_failures) + '\n'
(a.output / 'summary.md').write_text(report)
print(report)
if drift_failures:
    raise SystemExit(1)
