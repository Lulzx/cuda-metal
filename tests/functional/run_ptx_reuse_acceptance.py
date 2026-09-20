#!/usr/bin/env python3
"""Complete #76 ReLU acceptance matrix, with no backend-specific omissions.

The default stage only translates. The validation owner explicitly requests
``--stage numerical`` to run the 12 cells serially on Apple GPU. Every rejected,
timed-out, unverified, or untested required cell makes this command fail.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time

from ptx_test_support import run_integer_case
from run_ptx_register_reuse import relu_cases


ABI = ['CUMETAL_ABI_V2', 'kernel clamp_relu', 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
BACKENDS = ('legacy', 'cumetal-ir')
EXPECTED_CASES = ('original', 'original joined', 'copied diamond',
                  'reordered diamond', 'renamed diamond', 'bounded loop')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_and_reference():
    # Binary32 exactly represents every input and expected result. This reference
    # is independent of the PTX conversion, pointer, branch, and loop machinery.
    values = [(lane - 32) / 4 for lane in range(65)]
    bits = lambda value: struct.unpack('<I', struct.pack('<f', value))[0]
    return [bits(value) for value in values], [bits(max(value, 0.0)) for value in values]


def fixtures():
    cases = relu_cases()
    if tuple(label for label, _ in cases) != EXPECTED_CASES:
        raise AssertionError('required #76 fixture denominator changed')
    if len({source for _, source in cases}) != len(cases):
        raise AssertionError('ReLU fixtures must exercise distinct control flow')
    return cases


def run_command(command, folder, timeout):
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (folder / 'stdout.log').open('w') as out, (folder / 'stderr.log').open('w') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return dict(command=command, exit_code=process.returncode, timed_out=timed_out,
                elapsed_seconds=round(time.monotonic() - started, 6),
                stdout_sha256=digest(folder / 'stdout.log'),
                stderr_sha256=digest(folder / 'stderr.log'))


def worker(build, backend, index, artifacts_dir):
    label, source = fixtures()[index]
    values, expected = inputs_and_reference()
    run_integer_case(build, source, values, expected,
                     f'{backend} {label} complete ReLU acceptance', entry='clamp_relu',
                     word_bits=32, output_words=1, backend=backend, abi_lines=ABI,
                     artifacts_dir=artifacts_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path, help='directory containing cumetalc and libcumetal.dylib')
    parser.add_argument('--stage', choices=('translate', 'numerical'), default='translate')
    parser.add_argument('--output', type=Path, help='retained evidence directory (required outside worker mode)')
    parser.add_argument('--timeout', type=float, default=120, help='per compiler or numerical process limit, seconds')
    parser.add_argument('--worker', nargs=2, metavar=('BACKEND', 'FIXTURE_INDEX'), help=argparse.SUPPRESS)
    parser.add_argument('--worker-artifacts', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    build = args.build.resolve()
    if args.worker:
        backend, index = args.worker
        if backend not in BACKENDS:
            parser.error('invalid worker backend')
        worker(build, backend, int(index), args.worker_artifacts)
        return 0
    if not args.output:
        parser.error('--output is required')
    if not 0 < args.timeout <= 120:
        parser.error('--timeout must be in (0,120]')
    compiler = build / 'cumetalc'
    if not compiler.is_file():
        parser.error(f'compiler not found: {compiler}')
    runtime = build / 'libcumetal.dylib'
    if args.stage == 'numerical' and not runtime.is_file():
        parser.error(f'runtime not found: {runtime}')
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    values, expected = inputs_and_reference()
    identity = dict(schema=1, stage=args.stage, required_cells=12,
                    compiler=str(compiler), compiler_sha256=digest(compiler),
                    runtime=str(runtime) if runtime.is_file() else None,
                    runtime_sha256=digest(runtime) if runtime.is_file() else None,
                    script_sha256=digest(Path(__file__).resolve()),
                    fixture_generator_sha256=digest(Path(__file__).with_name('run_ptx_register_reuse.py')),
                    inputs=values, expected=expected, abi=ABI,
                    input_count=65, guard_words_each_side=16,
                    planned_launch_threads=128, workload_specializations=False)
    (destination / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    records = []
    for index, (label, source) in enumerate(fixtures()):
        for backend in BACKENDS:
            folder = destination / re.sub(r'[^a-z0-9]+', '-', label) / backend
            folder.mkdir(parents=True, exist_ok=True)
            ptx, msl = folder / 'input.ptx', folder / 'output.metal'
            ptx.write_text(source)
            command = [str(compiler), str(ptx), '--backend=' + backend, '--ptx-strict',
                       '--entry', 'clamp_relu', '--emit=msl', '--overwrite', '-o', str(msl)]
            compiled = run_command(command, folder / 'translation', args.timeout)
            record = dict(fixture=label, backend=backend, ptx_sha256=digest(ptx),
                          translation=compiled, numerical=None, status='TRANSLATION_FAILED')
            if compiled['timed_out']:
                record['status'] = 'TRANSLATION_TIMEOUT'
            elif compiled['exit_code'] == 0:
                sidecar = Path(str(msl) + '.cumetal-abi')
                actual = sidecar.read_text().splitlines() if sidecar.is_file() else None
                record.update(msl_sha256=digest(msl) if msl.is_file() else None,
                              abi=actual, abi_matches=actual == ABI)
                if not msl.is_file() or actual != ABI:
                    record['status'] = 'OUTPUT_OR_ABI_MISMATCH'
                elif args.stage == 'translate':
                    record['status'] = 'MSL_EMITTED_NUMERICAL_NOT_RUN'
                else:
                    # Separate processes keep errors/timeouts in one cell from
                    # suppressing the remaining backend/fixture denominator.
                    command = [sys.executable, str(Path(__file__).resolve()), str(build),
                               '--worker', backend, str(index),
                               '--worker-artifacts', str(folder / 'executed-artifacts')]
                    measured = run_command(command, folder / 'numerical', args.timeout)
                    log = (folder / 'numerical/stdout.log').read_text() + (folder / 'numerical/stderr.log').read_text()
                    launches = [line for line in log.splitlines() if
                                'CUMETAL_PROVENANCE event=kernel_launch' in line and
                                'kernel="clamp_relu"' in line]
                    provenance = any(all(part in line for part in (
                        'source=generic_ptx ', 'provenance=generic_ptx_lowering ',
                        'semantic_quality=exact ', 'device=apple_gpu ',
                        'launch_success=true ')) for line in launches)
                    executed = folder / 'executed-artifacts'
                    measured['executed_artifacts_sha256'] = {
                        path.name: digest(path) for path in sorted(executed.glob('*')) if path.is_file()}
                    measured['executed_input_matches'] = (executed / 'test.ptx').is_file() and (
                        digest(executed / 'test.ptx') == record['ptx_sha256'])
                    measured.update(gpu_launches=launches, gpu_provenance_matches=provenance,
                                    value_guard_input_checks_passed='NUMERICAL_PASS ' in log)
                    record['numerical'] = measured
                    record['status'] = 'NUMERICAL_TIMEOUT' if measured['timed_out'] else (
                        'NUMERICAL_PASS' if measured['exit_code'] == 0 and provenance and
                        measured['executed_input_matches'] and
                        measured['value_guard_input_checks_passed'] else 'NUMERICAL_FAILED')
            (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
            records.append(record)
            print(record['status'], backend, label, flush=True)
            (destination / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    wanted = 'NUMERICAL_PASS' if args.stage == 'numerical' else 'MSL_EMITTED_NUMERICAL_NOT_RUN'
    complete = len(records) == 12 and all(record['status'] == wanted for record in records)
    summary = dict(stage=args.stage, required_cells=12, recorded_cells=len(records),
                   statuses=dict(Counter(record['status'] for record in records)),
                   stage_pass=complete, full_numerical_acceptance=complete and args.stage == 'numerical')
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, sort_keys=True))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
