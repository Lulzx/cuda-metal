#!/usr/bin/env python3
"""Verify legacy register storage widths and assemble the emitted LLVM IR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


FUNCTIONAL = Path(__file__).resolve().parents[1] / 'functional'
sys.dont_write_bytecode = True
sys.path.insert(0, str(FUNCTIONAL))
from run_ptx_register_reuse import relu_cases


CASE_NAMES = ('original', 'original joined', 'copied diamond',
              'reordered diamond', 'renamed diamond', 'bounded loop')
TIMEOUT_SECONDS = 30


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def executable(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f'required executable is missing or not executable: {path}')
    return path


def command_result(command):
    started = time.monotonic()
    result = {'command': list(map(str, command)), 'timeout_seconds': TIMEOUT_SECONDS}
    environment = dict(os.environ, CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0')
    try:
        child = subprocess.run(result['command'], capture_output=True, text=True,
                               timeout=TIMEOUT_SECONDS, env=environment)
        result.update(exit_code=child.returncode, timed_out=False,
                      stdout=child.stdout, stderr=child.stderr,
                      passed=child.returncode == 0)
    except subprocess.TimeoutExpired as error:
        def text(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        result.update(exit_code=None, timed_out=True, stdout=text(error.stdout),
                      stderr=text(error.stderr), passed=False)
    except OSError as error:
        result.update(exit_code=None, timed_out=False, stdout='', stderr=str(error), passed=False)
    result['seconds'] = time.monotonic() - started
    return result


def assemble(llvm_as, source):
    output = source.with_suffix('.bc')
    result = command_result([llvm_as, source, '-o', output])
    result['llvm_sha256'] = sha256(source)
    result['bitcode_sha256'] = sha256(output) if output.is_file() else None
    result['passed'] = result['passed'] and result['bitcode_sha256'] is not None
    return result


def renamed_address_widths(llvm_ir):
    matches = re.findall(
        r'^\s*%cm_reg_address([0-3])_[0-9]+\s*=\s*alloca\s+i([0-9]+)\b',
        llvm_ir, re.MULTILINE)
    widths = {str(index): [] for index in range(4)}
    for register, bits in matches:
        widths[register].append(int(bits))
    # Each address, including the two loaded pointer parameters, must have
    # exactly one i64 storage slot. Narrowing a parameter into i32 fails here.
    return {'address_slot_widths': widths,
            'passed': all(found == [64] for found in widths.values())}


def report_case(label, result):
    print(f"{'PASS' if result['passed'] else 'FAIL'} {label}", flush=True)
    if not result['passed']:
        for stage in ('translation', 'assembly', 'execution'):
            if stage in result and result[stage]['stderr']:
                print(result[stage]['stderr'], end='', file=sys.stderr)
        if 'register_widths' in result:
            print(json.dumps(result['register_widths'], sort_keys=True), file=sys.stderr)


def run_suite(args, evidence):
    fixtures = relu_cases()
    if tuple(label for label, _ in fixtures) != CASE_NAMES:
        raise ValueError('expected exactly the six existing ReLU fixtures')
    report = {
        'schema': 1, 'stage': 'legacy_llvm_assembly', 'timeout_seconds': TIMEOUT_SECONDS,
        'compiler': str(args.compiler), 'compiler_sha256': sha256(args.compiler),
        'llvm_as': str(args.llvm_as), 'llvm_as_sha256': sha256(args.llvm_as),
        'script_sha256': sha256(Path(__file__).resolve()),
        'fixture_generator_sha256': sha256(FUNCTIONAL / 'run_ptx_register_reuse.py'),
        'fixture_sha256': sha256(FUNCTIONAL / 'reference/ptx_register_reuse.ptx'),
        'required_cases': len(CASE_NAMES), 'gpu_run': False, 'cases': [],
    }
    for label, source in fixtures:
        directory = evidence / label.replace(' ', '-')
        directory.mkdir()
        ptx, llvm_ir = directory / 'input.ptx', directory / 'output.ll'
        ptx.write_bytes(source.encode())
        result = {'case': label, 'ptx_sha256': sha256(ptx)}
        result['translation'] = command_result([
            args.compiler, ptx, '--backend=legacy', '--ptx-strict',
            '--entry', 'clamp_relu', '--emit=llvm', '-o', llvm_ir])
        result['passed'] = result['translation']['passed'] and llvm_ir.is_file()
        result['llvm_sha256'] = sha256(llvm_ir) if llvm_ir.is_file() else None
        if result['passed']:
            result['assembly'] = assemble(args.llvm_as, llvm_ir)
            result['passed'] = result['assembly']['passed']
            if label == 'renamed diamond':
                result['register_widths'] = renamed_address_widths(llvm_ir.read_text())
                result['passed'] = result['passed'] and result['register_widths']['passed']
        report['cases'].append(result)
        report_case(label, result)

    if args.unit_binary is not None:
        unit_directory = evidence / 'unit-ir'
        unit = {'binary': str(args.unit_binary), 'binary_sha256': sha256(args.unit_binary)}
        unit['execution'] = command_result([args.unit_binary, unit_directory])
        unit['assemblies'] = []
        for source in sorted(unit_directory.rglob('*.ll')):
            result = {'source': str(source), 'assembly': assemble(args.llvm_as, source)}
            result['passed'] = result['assembly']['passed']
            unit['assemblies'].append(result)
            report_case('unit IR ' + source.name, result)
        unit['passed'] = (unit['execution']['passed'] and bool(unit['assemblies'])
                          and all(result['passed'] for result in unit['assemblies']))
        report['unit_binary'] = unit
        report_case('register-width unit binary and emitted IR', unit)

    report['passed_cases'] = sum(result['passed'] for result in report['cases'])
    report['passed'] = (report['passed_cases'] == len(CASE_NAMES)
                        and report.get('unit_binary', {'passed': True})['passed'])
    (evidence / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Legacy LLVM register widths: {report['passed_cases']}/{len(CASE_NAMES)} ReLU cases passed")
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=executable)
    parser.add_argument('llvm_as', type=executable)
    parser.add_argument('--output', type=Path,
                        help='preserve artifacts and results.json in a new or empty directory')
    parser.add_argument('--unit-binary', type=executable,
                        help='run the C++ register-width test and assemble every .ll it emits')
    args = parser.parse_args()
    try:
        if args.output is not None:
            evidence = args.output.expanduser().resolve()
            evidence.mkdir(parents=True, exist_ok=True)
            if any(evidence.iterdir()):
                parser.error('--output must be a new or empty directory')
            return run_suite(args, evidence)
        with tempfile.TemporaryDirectory(prefix='cumetal-legacy-register-width-') as work:
            return run_suite(args, Path(work))
    except (OSError, ValueError) as error:
        parser.exit(2, f'ERROR: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
