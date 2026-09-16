#!/usr/bin/env python3
"""Exact PTX f16/f32-to-integer exceptional-value and rounding acceptance.

24 fixtures: {f16, f32, f32-ftz} x four rounding modes x {direct, join}.
Each checks all six s/u16/32/64 destination formats in b64 register storage.
The oracle decodes IEEE bits and rounds rational numbers using integer arithmetic.
Contract: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


MASK64 = (1 << 64) - 1
ENTRY = 'float_integer_probe'
DESTINATIONS = ('s16', 'u16', 's32', 'u32', 's64', 'u64')
ROUNDING = ('rni', 'rzi', 'rmi', 'rpi')
SOURCES = ('f16', 'f32', 'f32-ftz')
FORMS = ('direct', 'join')
CASE_NAMES = tuple(f'{source}-{mode}-{form}' for source in SOURCES
                   for mode in ROUNDING for form in FORMS)
ABI = ['CUMETAL_ABI_V2', 'kernel ' + ENTRY, 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']


def specification(label):
    if label not in CASE_NAMES:
        raise ValueError('unknown float-to-integer fixture: ' + label)
    source, mode, form = label.rsplit('-', 2)
    return int(source[1:3]), source.endswith('-ftz'), mode, form


def float_layout(width):
    return {16: (10, 5, 15), 32: (23, 8, 127)}[width]


def oracle(raw, source_width, destination, mode, flush_subnormal=False):
    """Return the complete b64 result, including destination sign extension."""
    fraction_bits, exponent_bits, bias = float_layout(source_width)
    negative = bool(raw & (1 << (source_width - 1)))
    fraction = raw & ((1 << fraction_bits) - 1)
    exponent = (raw >> fraction_bits) & ((1 << exponent_bits) - 1)
    width, signed = int(destination[1:]), destination.startswith('s')
    minimum = -(1 << (width - 1)) if signed else 0
    maximum = (1 << (width - int(signed))) - 1
    if exponent == (1 << exponent_bits) - 1:
        if fraction:
            return (1 << 63) if width == 64 else 0
        return (minimum if negative else maximum) & MASK64
    if exponent == 0 and flush_subnormal:
        if source_width != 32:
            raise ValueError('PTX .ftz here applies only to f32 sources')
        return 0
    significand = fraction if exponent == 0 else (1 << fraction_bits) | fraction
    power = (1 - bias if exponent == 0 else exponent - bias) - fraction_bits
    if power >= 0:
        magnitude, remainder, denominator = significand << power, 0, 1
    else:
        denominator = 1 << -power
        magnitude, remainder = divmod(significand, denominator)
    if mode == 'rni':
        magnitude += (2 * remainder > denominator or
                      (2 * remainder == denominator and magnitude & 1))
    elif mode == 'rmi':
        magnitude += negative and remainder != 0
    elif mode == 'rpi':
        magnitude += not negative and remainder != 0
    elif mode != 'rzi':
        raise ValueError('unknown rounding mode')
    integer = -magnitude if negative else magnitude
    return min(max(integer, minimum), maximum) & MASK64


def source_patterns(width):
    """65 distinct raw values; representable threshold neighbors are exact bits."""
    fraction_bits, exponent_bits, bias = float_layout(width)
    sign = 1 << (width - 1)
    infinity = ((1 << exponent_bits) - 1) << fraction_bits
    minimum_normal = 1 << fraction_bits
    patterns = []

    def both(magnitude):
        patterns.extend((magnitude, magnitude | sign))

    both(0)
    both(infinity)
    both(infinity | (1 << (fraction_bits - 1)) | 1)  # quiet NaN, both signs
    both(infinity | 1)  # signaling NaN, both signs
    both(1)
    both(minimum_normal - 1)
    both(minimum_normal)
    half = (bias - 1) << fraction_bits
    for raw in (half - 1, half, half + 1):
        both(raw)
    both((bias << fraction_bits) | (1 << (fraction_bits - 1)))  # 1.5
    both(((bias + 1) << fraction_bits) | (1 << (fraction_bits - 2)))  # 2.5
    both(bias << fraction_bits)  # 1
    both((bias << fraction_bits) | (1 << (fraction_bits - 2)))  # 1.25
    patterns.append(((bias + 1) << fraction_bits) | (3 << (fraction_bits - 3)))  # 2.75
    # All signed lower/upper and unsigned upper boundaries for 16/32/64.
    # Half cannot represent powers above 2^15; its finite extrema and infinities
    # cover the reachable domain for the remaining destination boundaries.
    for power in (15, 16, 31, 32, 63, 64):
        exponent = bias + power
        if exponent < (1 << exponent_bits) - 1:
            center = exponent << fraction_bits
            for raw in (center - 1, center, center + 1):
                both(raw)
    if width == 16:
        both(infinity - 1)
        both(infinity - 2)
        step = 0
        while len(patterns) < 65:
            raw = (0x123 + 977 * step) & ((1 << width) - 1)
            step += 1
            if raw not in patterns:
                patterns.append(raw)
    if len(patterns) != 65 or len(set(patterns)) != 65:
        raise AssertionError('float-to-integer input denominator changed')
    return patterns


def inputs_and_expected(label):
    width, ftz, mode, _ = specification(label)
    values, expected = [], []
    source_mask = (1 << width) - 1
    for lane, raw in enumerate(source_patterns(width)):
        # Poison high container bits. The instruction consumes only f16/f32.
        values.extend(((0xA5A55A5AA5A55A5A & (MASK64 ^ source_mask)) | raw, lane & 1))
        expected.extend(oracle(raw, width, destination, mode, ftz)
                        for destination in DESTINATIONS)
    return values, expected


def fixture_source(label):
    width, ftz, mode, form = specification(label)
    opcode = f'cvt.{mode}' + ('.ftz' if ftz else '')
    lines = ['.version 9.0', '.target sm_80', '.address_size 64',
             f'.visible .entry {ENTRY}(', '    .param .u64 .ptr .global input,',
             '    .param .u64 .ptr .global output,', '    .param .u32 count', ') {',
             '    .reg .b64 %input, %output, %input_offset, %input_address, %source;',
             '    .reg .b64 %selector, %output_offset, %output_address;',
             '    .reg .b32 %count, %lane, %block, %threads;',
             '    .reg .pred %done, %even;']
    for index in range(len(DESTINATIONS)):
        lines.append(f'    .reg .b64 %converted{index}, %copied{index};')
    lines.extend(('    ld.param.u64 %input, [input];',
                  '    ld.param.u64 %output, [output];', '    ld.param.u32 %count, [count];',
                  '    mov.u32 %lane, %tid.x;', '    mov.u32 %block, %ctaid.x;',
                  '    mov.u32 %threads, %ntid.x;',
                  '    mad.lo.u32 %lane, %block, %threads, %lane;',
                  '    setp.ge.u32 %done, %lane, %count;', '    @%done bra DONE;',
                  '    mul.wide.u32 %input_offset, %lane, 16;',
                  '    add.u64 %input_address, %input, %input_offset;',
                  '    ld.global.b64 %source, [%input_address];',
                  '    ld.global.u64 %selector, [%input_address+8];'))
    if form == 'join':
        lines.extend(('    setp.eq.u64 %even, %selector, 0;', '    @%even bra EVEN;'))
        for index, destination in enumerate(DESTINATIONS):
            lines.extend((f'    {opcode}.{destination}.f{width} %copied{index}, %source;',
                          f'    mov.b64 %converted{index}, %copied{index};'))
        lines.extend(('    bra JOINED;', 'EVEN:'))
    for index, destination in enumerate(DESTINATIONS):
        lines.append(f'    {opcode}.{destination}.f{width} %converted{index}, %source;')
    if form == 'join':
        lines.append('JOINED:')
    lines.extend(('    mul.wide.u32 %output_offset, %lane, 48;',
                  '    add.u64 %output_address, %output, %output_offset;'))
    for index in range(len(DESTINATIONS)):
        lines.append(f'    st.global.b64 [%output_address+{8 * index}], %converted{index};')
    return '\n'.join(lines + ['DONE:', '    ret;', '}']) + '\n'


def fixtures():
    return [(label, fixture_source(label)) for label in CASE_NAMES]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command, timeout):
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True,
                             env=dict(os.environ, CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0'))
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = child.communicate(timeout=5)
    return dict(command=command, exit_code=child.returncode, timed_out=timed_out,
                seconds=time.monotonic() - started, stdout=stdout, stderr=stderr)


def worker(build, label, artifacts):
    from ptx_test_support import run_integer_case
    values, expected = inputs_and_expected(label)
    run_integer_case(build, fixture_source(label), values, expected,
                     'float integer: ' + label, entry=ENTRY, input_words=2,
                     output_words=len(DESTINATIONS), abi_lines=ABI, artifacts_dir=artifacts)


def run_suite(args, evidence):
    compiler = args.build / 'cumetalc'
    records = []
    selected = [(label, source) for label, source in fixtures()
                if args.case is None or label == args.case]
    for label, source in selected:
        folder = evidence / label
        folder.mkdir()
        ptx, msl = folder / 'input.ptx', folder / 'output.metal'
        ptx.write_text(source)
        values, expected = inputs_and_expected(label)
        record = dict(case=label, ptx_sha256=digest(ptx), inputs=values, expected=expected,
                      input_count=65, output_formats=DESTINATIONS, output_words=6)
        if args.stage == 'translate':
            measured = run_command([str(compiler), str(ptx), '--backend=cumetal-ir',
                                    '--ptx-strict', '--entry', ENTRY, '--emit=msl', '-o', str(msl)], 30)
            abi = Path(str(msl) + '.cumetal-abi')
            good = (not measured['timed_out'] and measured['exit_code'] == 0 and
                    msl.is_file() and abi.is_file() and abi.read_text().splitlines() == ABI)
            record['status'] = 'MSL_EMITTED_NUMERICAL_NOT_RUN' if good else 'TRANSLATION_FAILED'
            record['msl_sha256'] = digest(msl) if msl.is_file() else None
        else:
            artifacts = folder / 'executed-artifacts'
            measured = run_command([sys.executable, str(Path(__file__).resolve()), str(args.build),
                                    '--case', label, '--gpu-child', '--artifacts', str(artifacts)], 90)
            launches = [line for line in measured['stderr'].splitlines()
                        if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
            provenance = len(launches) == 1 and all(token in launches[0] for token in (
                f'kernel="{ENTRY}"', 'device=apple_gpu ', 'launch_success=true ',
                'source=generic_ptx ', 'provenance=generic_ptx_lowering ', 'semantic_quality=exact '))
            numerical = f'NUMERICAL_PASS float integer: {label}: 65 inputs, output values, guards'
            executed = artifacts / 'test.ptx'
            same_input = executed.is_file() and digest(executed) == record['ptx_sha256']
            good = (not measured['timed_out'] and measured['exit_code'] == 0 and provenance and
                    same_input and numerical in measured['stdout'].splitlines())
            record.update(gpu_launches=launches, gpu_provenance_matches=provenance,
                          executed_input_matches=same_input,
                          executed_artifacts_sha256={path.name: digest(path)
                              for path in sorted(artifacts.glob('*')) if path.is_file()})
            record['status'] = 'NUMERICAL_PASS' if good else 'NUMERICAL_FAILED'
        record.update(passed=good, process=measured)
        records.append(record)
        (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(record['status'], label, flush=True)
        if not good:
            print(measured['stderr'], end='', file=sys.stderr)
    report = dict(schema=1, stage=args.stage, compiler=str(compiler), compiler_sha256=digest(compiler),
                  script_sha256=digest(Path(__file__).resolve()),
                  helper_sha256=digest(Path(__file__).with_name('ptx_test_support.py')),
                  runtime_sha256=digest(args.build / 'libcumetal.dylib') if args.stage == 'numerical' else None,
                  required_cases=len(selected), full_denominator=len(CASE_NAMES),
                  full_conversion_contracts=len(CASE_NAMES) * len(DESTINATIONS),
                  statuses=dict(Counter(item['status'] for item in records)),
                  gpu_run=args.stage == 'numerical', passed=all(item['passed'] for item in records),
                  cases=records)
    (evidence / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


def main():
    if not __debug__:
        raise RuntimeError('This test requires Python assertions for ABI and guard checks')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--stage', choices=('numerical', 'translate'), default='numerical')
    parser.add_argument('--case', choices=CASE_NAMES)
    parser.add_argument('--output', type=Path, help='new or empty evidence directory')
    parser.add_argument('--gpu-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--artifacts', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.build = args.build.resolve()
    if not (args.build / 'cumetalc').is_file():
        parser.error('build directory must contain cumetalc')
    if args.stage == 'numerical' and not (args.build / 'libcumetal.dylib').is_file():
        parser.error('numerical stage requires libcumetal.dylib')
    if args.gpu_child:
        if args.case is None:
            parser.error('--gpu-child requires --case')
        worker(args.build, args.case, args.artifacts)
        return 0
    if args.output is not None:
        evidence = args.output.resolve()
        evidence.mkdir(parents=True, exist_ok=True)
        if any(evidence.iterdir()):
            parser.error('--output must be a new or empty directory')
        return run_suite(args, evidence)
    with tempfile.TemporaryDirectory(prefix='cumetal-float-integer-') as work:
        return run_suite(args, Path(work))


if __name__ == '__main__':
    raise SystemExit(main())
