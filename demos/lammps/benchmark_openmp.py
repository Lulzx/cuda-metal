"""Warm LJ timing: CuMetal Apple GPU against Kokkos/OpenMP on every CPU core.

Runs stock bench/in.lj (optionally replicated with -var x/y/z) on:
  * the Kokkos/OpenMP CPU build at several thread counts, both neighbour modes;
  * the CuMetal Kokkos/CUDA build on the Apple GPU, both neighbour modes;
  * optionally the KOKKOS_PREC=double GPU build under each CUMETAL_FP64_MODE,
    next to a double-precision Kokkos/OpenMP CPU build;
  * optionally an independent LAMMPS binary (the official macOS DMG).
Configurations are interleaved per repeat; the first repeat is a discarded
warmup that also fills the GPU pipeline cache. Times are LAMMPS "Loop time",
which excludes setup and JIT. The 1-minute load average is recorded per run.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get('CUMETAL_LAMMPS_DIR', '/tmp/cumetal-lammps-30Sep2026'))
REV = '8de817dd79bfe4525d5d39246a212d833e6dee07'
NEWTON = {'full': 'off', 'half': 'on'}


def thermo_rows(text):
    rows, active = {}, False
    for line in text.splitlines():
        fields = line.split()
        if fields[:2] == ['Step', 'Temp']:
            active = True
        elif line.startswith('Loop time'):
            active = False
        elif active and len(fields) == 6:
            rows[int(fields[0])] = [float(v) for v in fields[1:]]
    return rows


def configs(args):
    omp = SRC / 'build-cumetal-openmp' / 'lmp'
    gpu = SRC / 'build-cumetal-cuda' / 'lmp'
    for exe in (omp, gpu):
        if not exe.is_file():
            sys.exit(f'missing {exe}; build with scripts/build_lammps_cumetal.sh')
    out = []
    for mode in ('full', 'half'):
        kk = ['-sf', 'kk', '-pk', 'kokkos', 'neigh', mode, 'newton', NEWTON[mode]]
        for t in args.threads:
            out.append(dict(name=f'omp-t{t}-{mode}', device='cpu', precision='single', mode=mode,
                            threads=t, cmd=[str(omp), '-k', 'on', 't', str(t)] + kk))
        out.append(dict(name=f'gpu-{mode}', device='gpu', precision='single', mode=mode,
                        threads=None, cmd=[str(gpu), '-k', 'on', 'g', '1'] + kk))
        if args.fp64_modes:
            omp_double = SRC / 'build-cumetal-openmp-double' / 'lmp'
            gpu_double = SRC / 'build-cumetal-cuda-double' / 'lmp'
            for exe in (omp_double, gpu_double):
                if not exe.is_file():
                    sys.exit(f'missing {exe}; build with scripts/build_lammps_cumetal.sh')
            t = max(args.threads)
            out.append(dict(name=f'omp-double-t{t}-{mode}', device='cpu', precision='double',
                            mode=mode, threads=t,
                            cmd=[str(omp_double), '-k', 'on', 't', str(t)] + kk))
            for fp64 in args.fp64_modes:
                out.append(dict(name=f'gpu-double-{fp64}-{mode}', device='gpu', precision='double',
                                mode=mode, threads=None, env=dict(CUMETAL_FP64_MODE=fp64),
                                cmd=[str(gpu_double), '-k', 'on', 'g', '1'] + kk))
    if args.dmg_lmp:
        t = str(max(args.threads))
        out.append(dict(name=f'dmg-kk-t{t}-half', device='cpu', precision='double', mode='half',
                        threads=int(t), cmd=[args.dmg_lmp, '-k', 'on', 't', t, '-sf', 'kk',
                                             '-pk', 'kokkos', 'neigh', 'half', 'newton', 'on']))
        out.append(dict(name=f'dmg-omp-t{t}', device='cpu', precision='double', mode='half',
                        threads=int(t), cmd=[args.dmg_lmp, '-sf', 'omp', '-pk', 'omp', t]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--sizes', default='1,2', help='in.lj replication factors (x=y=z)')
    p.add_argument('--threads', default='1,4,8,12')
    p.add_argument('--dmg-lmp', help='independent lmp binary (official macOS DMG)')
    p.add_argument('--fp64-modes', default='',
                   help='also time the double-precision builds, e.g. ieee64,wide48')
    p.add_argument('--only', default='', help='comma-separated name prefixes to keep')
    args = p.parse_args()
    args.threads = [int(t) for t in args.threads.split(',')]
    args.fp64_modes = [m for m in args.fp64_modes.split(',') if m]
    sizes = [int(s) for s in args.sizes.split(',')]
    if subprocess.check_output(['git', '-C', str(SRC), 'rev-parse', 'HEAD']).decode().strip() != REV:
        sys.exit('unpinned LAMMPS checkout')

    env = dict(os.environ, DYLD_LIBRARY_PATH=str(ROOT / 'build'),
               CUMETAL_PTX_BACKEND='cumetal-ir', CUMETAL_USE_METAL_DEVICE_ADDRESSES='1',
               CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0', CUMETAL_TRACE_GPU='0',
               CUMETAL_DEBUG_REGISTRATION='0',
               CUMETAL_CACHE_DIR=str(ROOT / 'demos/lammps/out/cache-benchmark'))
    out = Path(tempfile.mkdtemp(prefix='benchmark-openmp.', dir=ROOT / 'demos/lammps/out'))
    print('output:', out, flush=True)
    plan = configs(args)
    if args.only:
        plan = [c for c in plan if c['name'].startswith(tuple(args.only.split(',')))]
    rows = []
    for size in sizes:
        for repeat in range(args.repeats + 1):
            # Rotate the order so no configuration always runs first or last.
            k = repeat % len(plan)
            for c in plan[k:] + plan[:k]:
                name = f'{c["name"]}-x{size}-r{repeat}'
                log, clean = out / f'{name}.log', out / f'{name}.lammps.log'
                cmd = c['cmd'] + ['-var', 'x', str(size), '-var', 'y', str(size), '-var', 'z', str(size),
                                  '-in', str(SRC / 'bench/in.lj'), '-log', str(clean)]
                load = os.getloadavg()[0]
                start = time.monotonic()
                with log.open('w') as stream:
                    rc = subprocess.run(cmd, cwd=out, env=dict(env, **c.get('env', {})), stdout=stream,
                                        stderr=subprocess.STDOUT).returncode
                wall = time.monotonic() - start
                text = clean.read_text() if clean.exists() else ''
                loop = re.search(r'Loop time of ([0-9.eE+-]+)', text)
                atoms = re.search(r'with (\d+) atoms', text)
                row = dict(c, size=size, repeat=repeat, warmup=repeat == 0, rc=rc,
                           load_before=load, wall_seconds=wall,
                           loop_seconds=float(loop[1]) if loop else None,
                           atoms=int(atoms[1]) if atoms else None, thermo=thermo_rows(text))
                del row['cmd']
                rows.append(row)
                print(f'{name:28} rc={rc} loop={row["loop_seconds"]} load={load:.1f}', flush=True)
                if rc or loop is None:
                    print(f'  FAILED, log: {log}', flush=True)

    summary = {}
    for size in sizes:
        timed = [r for r in rows if r['size'] == size and not r['warmup'] and r['loop_seconds']]
        med = {}
        for name in dict.fromkeys(r['name'] for r in timed):
            vals = [r['loop_seconds'] for r in timed if r['name'] == name]
            med[name] = dict(median=statistics.median(vals), min=min(vals), max=max(vals), n=len(vals))
        cpu = {n: m for n, m in med.items() if n.startswith('omp-t')}
        gpu = {n: m for n, m in med.items() if n.startswith('gpu-') and '-double-' not in n}
        best_cpu = min(cpu, key=lambda n: cpu[n]['median']) if cpu else None
        best_gpu = min(gpu, key=lambda n: gpu[n]['median']) if gpu else None
        atoms = next((r['atoms'] for r in timed if r['atoms']), None)
        cpu_double = {n: m for n, m in med.items() if n.startswith('omp-double-')}
        best_cpu_double = min(cpu_double, key=lambda n: cpu_double[n]['median']) if cpu_double else None
        double = {n: cpu_double[best_cpu_double]['median'] / m['median']
                  for n, m in med.items() if n.startswith('gpu-double-')} if best_cpu_double else {}
        summary[f'x{size}'] = dict(
            atoms=atoms, medians=med, best_cpu=best_cpu, best_gpu=best_gpu,
            gpu_speedup_vs_best_cpu=(cpu[best_cpu]['median'] / gpu[best_gpu]['median'])
            if best_cpu and best_gpu else None,
            best_cpu_double=best_cpu_double, gpu_double_speedup_vs_best_cpu_double=double)
    result = dict(
        cumetal_revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(),
        lammps_revision=REV, hardware=subprocess.check_output(
            ['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip(),
        cores=dict(performance=int(subprocess.check_output(['sysctl', '-n', 'hw.perflevel0.physicalcpu'])),
                   efficiency=int(subprocess.check_output(['sysctl', '-n', 'hw.perflevel1.physicalcpu']))),
        timing='LAMMPS Loop time; warm; excludes setup/JIT; first repeat discarded',
        repeats=args.repeats, rows=rows, summary=summary)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for key, s in summary.items():
        print(f'\n{key}: {s["atoms"]} atoms')
        for n, m in sorted(s['medians'].items(), key=lambda kv: kv[1]['median']):
            print(f'  {n:18} {m["median"]:8.3f} s  (min {m["min"]:.3f}, max {m["max"]:.3f}, n={m["n"]})')
        if s['gpu_speedup_vs_best_cpu']:
            print(f'  GPU {s["best_gpu"]} vs best CPU {s["best_cpu"]}: {s["gpu_speedup_vs_best_cpu"]:.2f}x')
        for n, x in s['gpu_double_speedup_vs_best_cpu_double'].items():
            print(f'  {n} vs {s["best_cpu_double"]}: {x:.2f}x')
    print('\nresults:', out / 'results.json')


if __name__ == '__main__':
    main()
