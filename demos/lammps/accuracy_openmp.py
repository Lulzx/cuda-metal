"""Single-precision accuracy of the CuMetal GPU run, judged like-for-like.

Reference hierarchy, per neighbour mode:
  * omp-single: Kokkos/OpenMP CPU, KOKKOS_PREC=single (the like-for-like reference);
  * serial-double: Kokkos Serial CPU, KOKKOS_PREC=double (ground truth);
  * gpu-single: CuMetal Kokkos/CUDA on the Apple GPU, KOKKOS_PREC=single;
  * gpu-double-<mode>: the KOKKOS_PREC=double GPU build under CUMETAL_FP64_MODE
    (full neighbour lists only: half lists need double atomics, which CuMetal
    refuses), judged against serial-double with the same short/long checks.

Two workloads:
  short  stock bench/in.lj, 100 steps. Thermo at steps 0 and 100 is compared
         pointwise. Gate: |gpu - omp-single| scaled <= 5e-4 (the existing
         demo tolerance), and the GPU's error against serial-double is reported
         next to omp-single's own error against serial-double.
  long   inputs/in.lj-nve-long, 10,000 NVE steps. Trajectories are chaotic,
         so only conserved/statistical quantities are compared: relative total
         energy drift max|E(t)-E(0)|/|E(0)|, and second-half mean temperature
         and pressure. Gate (fixed before measurement): GPU drift <= max(2x
         omp-single drift, 1e-4).
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_openmp import NEWTON, REV, ROOT, SRC, thermo_rows  # noqa: E402

TOLERANCE = 5e-4
FIELDS = ['temperature', 'pair_energy', 'molecular_energy', 'total_energy', 'pressure']
LONG_INPUT = ROOT / 'demos/lammps/inputs/in.lj-nve-long'


def scaled(a, b):
    return max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(a, b))


def main():
    steps = os.environ.get('CUMETAL_LAMMPS_LONG_STEPS', '10000')
    builds = {
        'omp-single': (SRC / 'build-cumetal-openmp/lmp', ['-k', 'on', 't', '8'], {}),
        'serial-double': (SRC / 'build-cumetal-reference-double/lmp', ['-k', 'on'], {}),
        'gpu-single': (SRC / 'build-cumetal-cuda/lmp', ['-k', 'on', 'g', '1'], {}),
    }
    fp64_modes = [m for m in os.environ.get('CUMETAL_LAMMPS_FP64_MODES', 'ieee64,wide48').split(',') if m]
    for mode in fp64_modes:
        builds[f'gpu-double-{mode}'] = (SRC / 'build-cumetal-cuda-double/lmp',
                                        ['-k', 'on', 'g', '1'], dict(CUMETAL_FP64_MODE=mode))
    for exe, _, _ in builds.values():
        if not exe.is_file():
            sys.exit(f'missing {exe}; build with scripts/build_lammps_cumetal.sh')
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(ROOT / 'build'),
               CUMETAL_PTX_BACKEND='cumetal-ir', CUMETAL_USE_METAL_DEVICE_ADDRESSES='1',
               CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0', CUMETAL_TRACE_GPU='0',
               CUMETAL_CACHE_DIR=str(ROOT / 'demos/lammps/out/cache-benchmark'))
    out = Path(tempfile.mkdtemp(prefix='accuracy-openmp.', dir=ROOT / 'demos/lammps/out'))
    print('output:', out, flush=True)

    runs = {}
    for workload, inp, extra in (('short', SRC / 'bench/in.lj', []),
                                 ('long', LONG_INPUT, ['-var', 'steps', steps])):
        for mode in ('full', 'half'):
            for name, (exe, prefix, extra_env) in builds.items():
                if name.startswith('gpu-double-') and mode == 'half':
                    continue
                tag = f'{workload}-{mode}-{name}'
                clean = out / f'{tag}.lammps.log'
                cmd = [str(exe)] + prefix + ['-sf', 'kk', '-pk', 'kokkos', 'neigh', mode,
                                             'newton', NEWTON[mode], '-in', str(inp),
                                             '-log', str(clean)] + extra
                with (out / f'{tag}.log').open('w') as stream:
                    rc = subprocess.run(cmd, cwd=out, env=dict(env, **extra_env), stdout=stream,
                                        stderr=subprocess.STDOUT).returncode
                text = clean.read_text() if clean.exists() else ''
                if rc or 'ERROR' in text:
                    sys.exit(f'{tag} failed (rc={rc}); see {out / (tag + ".log")}')
                precision = 'double' if 'double' in name else 'single'
                if f'using {precision} precision' not in text:
                    sys.exit(f'{tag}: expected {precision} precision build')
                runs[tag] = thermo_rows(text)
                print(f'{tag:32} {len(runs[tag])} thermo rows', flush=True)

    short = {}
    for mode in ('full', 'half'):
        g, o, d = (runs[f'short-{mode}-{n}'] for n in ('gpu-single', 'omp-single', 'serial-double'))
        short[mode] = dict(
            gpu_vs_omp_single=max(scaled(g[s], o[s]) for s in (0, 100)),
            gpu_vs_serial_double=max(scaled(g[s], d[s]) for s in (0, 100)),
            omp_single_vs_serial_double=max(scaled(o[s], d[s]) for s in (0, 100)),
            fields={f'{s}:{f}': dict(gpu=a, omp_single=b, serial_double=c)
                    for s in (0, 100) for f, a, b, c in zip(FIELDS, g[s], o[s], d[s])})
        for name in builds:
            if name.startswith('gpu-double-') and f'short-{mode}-{name}' in runs:
                gd = runs[f'short-{mode}-{name}']
                short[mode][f'{name}_vs_serial_double'] = max(scaled(gd[s], d[s]) for s in (0, 100))

    long = {}
    for mode in ('full', 'half'):
        long[mode] = {}
        for name in builds:
            if f'long-{mode}-{name}' not in runs:
                continue
            rows = runs[f'long-{mode}-{name}']
            ordered = [rows[k] for k in sorted(rows)]
            e0 = ordered[0][3]
            half = ordered[len(ordered) // 2:]
            long[mode][name] = dict(
                steps=max(rows), samples=len(rows),
                relative_energy_drift=max(abs(r[3] - e0) for r in ordered) / abs(e0),
                final_relative_energy_change=(ordered[-1][3] - e0) / abs(e0),
                mean_temperature_second_half=statistics.fmean(r[0] for r in half),
                mean_pressure_second_half=statistics.fmean(r[4] for r in half))

    short_ok = all(s['gpu_vs_omp_single'] <= TOLERANCE for s in short.values())
    long_ok = all(m['gpu-single']['relative_energy_drift']
                  <= max(2 * m['omp-single']['relative_energy_drift'], 1e-4) for m in long.values())
    # Double: the same fixed bounds, against serial-double.
    short_ok &= all(v <= TOLERANCE for s in short.values()
                    for k, v in s.items() if k.startswith('gpu-double-'))
    long_ok &= all(v['relative_energy_drift']
                   <= max(2 * m['serial-double']['relative_energy_drift'], 1e-4)
                   for m in long.values() for n, v in m.items() if n.startswith('gpu-double-'))
    result = dict(status='pass' if short_ok and long_ok else 'fail', lammps_revision=REV,
                  cumetal_revision=subprocess.check_output(
                      ['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(),
                  tolerance=TOLERANCE, short=short, long=long, long_steps=int(steps))
    (out / 'accuracy.json').write_text(json.dumps(result, indent=2) + '\n')
    for mode, s in short.items():
        print(f'short {mode}: gpu-vs-omp {s["gpu_vs_omp_single"]:.2e}  gpu-vs-double '
              f'{s["gpu_vs_serial_double"]:.2e}  omp-vs-double {s["omp_single_vs_serial_double"]:.2e}')
        for k, v in s.items():
            if k.startswith('gpu-double-'):
                print(f'short {mode}: {k} {v:.2e}')
    for mode, m in long.items():
        for name, v in m.items():
            print(f'long {mode} {name:14} drift {v["relative_energy_drift"]:.2e}  '
                  f'<T> {v["mean_temperature_second_half"]:.5f}  <P> {v["mean_pressure_second_half"]:.5f}')
    print('status:', result['status'], '\nresults:', out / 'accuracy.json')
    sys.exit(0 if result['status'] == 'pass' else 1)


if __name__ == '__main__':
    main()
