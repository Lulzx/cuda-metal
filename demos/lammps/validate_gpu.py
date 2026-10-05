"""Strict stock LJ GPU completion, provenance, and CPU-double comparison."""
import json
from pathlib import Path
import sys
from validate_reference import read_reference
from characterize_reference import initial_fcc_pair_energy

FIELDS = ['temperature', 'pair_energy', 'molecular_energy', 'total_energy', 'pressure']
TOLERANCE = 5e-4


def validate(gpu, reference):
    comparisons = []
    thermo = {}
    for mode in ('full', 'half'):
        log = (gpu / f'{mode}.log').read_text()
        traces = [line for line in log.splitlines() if line.startswith('CUMETAL_PROVENANCE event=kernel_launch')]
        if not traces or any('device=apple_gpu' not in line or 'launch_success=true' not in line
                             or 'source=approximate_stub' in line or 'source=cpu' in line for line in traces):
            raise ValueError(f'{mode}: missing or invalid GPU launch provenance')
        if not any('compile_cache_hit=false' in line for line in traces):
            raise ValueError(f'{mode}: no cold compile observed')
        # LAMMPS writes a clean log independently of diagnostic stderr traces.
        actual = read_reference(gpu / f'{mode}.lammps.log', mode)
        expected = read_reference(reference / f'{mode}.log', mode, 'double')
        thermo[mode] = actual
        for step in (0,100):
            for field, a, b in zip(FIELDS, actual[step], expected[step]):
                comparisons.append({'mode':mode,'step':step,'field':field,'gpu':a,'cpu_double':b,
                                    'scaled_error':abs(a-b)/max(1,abs(b))})
    maximum = max(row['scaled_error'] for row in comparisons)
    cross_mode = max(abs(a-b)/max(1,abs(a)) for step in (0,100)
                     for a,b in zip(thermo['full'][step], thermo['half'][step]))
    analytic, _ = initial_fcc_pair_energy()
    initial = max(abs(thermo[mode][0][1]-analytic)/abs(analytic) for mode in thermo)
    passed = max(maximum,cross_mode,initial) <= TOLERANCE
    result = {'status':'pass' if passed else 'fail', 'atoms':32000,'steps':100,
              'lammps_revision':'8de817dd79bfe4525d5d39246a212d833e6dee07',
              'backend':'cumetal-ir','tolerance':TOLERANCE,'comparisons':comparisons,
              'max_scaled_cpu_double_error':maximum,'max_scaled_mode_difference':cross_mode,
              'analytic_initial_pair_energy':analytic,'max_scaled_analytic_initial_error':initial}
    (gpu / 'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    if not passed:
        raise ValueError(f'GPU numerical comparison failed: {maximum=}, {cross_mode=}, {initial=}')
    return result


if __name__ == '__main__':
    print(json.dumps(validate(Path(sys.argv[1]), Path(sys.argv[2])),indent=2))
