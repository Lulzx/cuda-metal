#!/usr/bin/env python3
"""Check PTX conversion formats stored in wider declared integer registers."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time


MASK64 = (1 << 64) - 1
POISON = 0x55667788A5A55A5A
FORMS = ('direct', 'copy', 'join')
INTEGER_SOURCES = ('s32', 'u32', 's64', 'u64')
INTEGER_CONTRACTS = (('s16', 32), ('u16', 32), ('s16', 64),
                     ('u16', 64), ('s32', 64), ('u32', 64))
CASE_NAMES = tuple(f'{source}-{form}' for source in INTEGER_SOURCES for form in FORMS) + tuple(
    f'f32-s64-{form}' for form in FORMS) + tuple(
    f'f32-roundtrip-{form}' for form in FORMS) + tuple(
    f'f16-source-{form}' for form in FORMS) + tuple(
    f'f32-store-u32-{form}' for form in FORMS)
ENTRY = 'cvt_storage_probe'
ABI = ['CUMETAL_ABI_V2', 'kernel ' + ENTRY, 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']


def specification(label):
    if label not in CASE_NAMES:
        raise ValueError('unknown conversion fixture: ' + label)
    if label.startswith('f16-source-'):
        return 'u32', label.rsplit('-', 1)[1], (('b32', 32),)
    if label.startswith('f32-roundtrip-'):
        return 's64', label.rsplit('-', 1)[1], (('f32', 64),)
    if label.startswith('f32-store-u32-'):
        return 'u32', label.rsplit('-', 1)[1], (('f32', 64),)
    floating = label.startswith('f32-')
    source, form = label.removeprefix('f32-').split('-')
    return source, form, (('f32', 64),) if floating else INTEGER_CONTRACTS


def roundtrip_format(label):
    if label.startswith('f32-roundtrip-'):
        return 'f32'
    if label.startswith('f16-source-'):
        return 'f16'
    return None


def output_words(label):
    return len(specification(label)[2]) + (roundtrip_format(label) is not None)


def conversion(source, destination, register):
    if destination == 'b32':
        return f'    mov.b32 %{register}, %source;'
    rounding = '.rn' if destination == 'f32' else ''
    return f'    cvt{rounding}.{destination}.{source} %{register}, %source;'


def fixture_source(label):
    source, form, contracts = specification(label)
    source_bits = int(source[1:])
    words = output_words(label)
    lines = ['.version 7.0', '.target sm_80', '.address_size 64',
             f'.visible .entry {ENTRY}(',
             '    .param .u64 .ptr .global input,',
             '    .param .u64 .ptr .global output,',
             '    .param .u32 count', ') {',
             f'    .local .align 8 .b8 cells[{8 * words}];',
             '    .reg .b64 %input, %output, %input_offset, %input_address;',
             '    .reg .b64 %selector, %choice, %scratch, %output_offset, %output_address;',
             '    .reg .b32 %count, %lane, %block, %threads;',
             '    .reg .pred %done, %even;',
             f'    .reg .b{source_bits} %source;']
    for index, (_, storage) in enumerate(contracts):
        lines.append(f'    .reg .b{storage} %converted{index}, %copied{index};')
    for index in range(words):
        lines.append(f'    .reg .b64 %read{index};')
    if roundtrip_format(label):
        lines.append('    .reg .b32 %roundtrip;')
    lines.extend(('    ld.param.u64 %input, [input];',
                  '    ld.param.u64 %output, [output];',
                  '    ld.param.u32 %count, [count];',
                  '    mov.u32 %lane, %tid.x;',
                  '    mov.u32 %block, %ctaid.x;',
                  '    mov.u32 %threads, %ntid.x;',
                  '    mad.lo.u32 %lane, %block, %threads, %lane;',
                  '    setp.ge.u32 %done, %lane, %count;',
                  '    @%done bra DONE;',
                  '    mul.wide.u32 %input_offset, %lane, 16;',
                  '    add.u64 %input_address, %input, %input_offset;',
                  f'    ld.global.b{source_bits} %source, [%input_address];',
                  '    ld.global.u64 %selector, [%input_address+8];',
                  '    mov.u64 %scratch, cells;'))
    for index in range(words):
        lines.append(f'    st.local.u64 [%scratch+{8 * index}], {POISON};')
    if form == 'join':
        lines.extend(('    and.b64 %choice, %selector, 1;',
                      '    setp.eq.u64 %even, %choice, 0;',
                      '    @%even bra EVEN;'))
        for index, (destination, storage) in enumerate(contracts):
            lines.extend((conversion(source, destination, f'copied{index}'),
                          f'    mov.b{storage} %converted{index}, %copied{index};'))
        lines.extend(('    bra JOINED;', 'EVEN:'))
    for index, (destination, storage) in enumerate(contracts):
        lines.append(conversion(source, destination, f'converted{index}'))
        if form == 'copy':
            lines.append(f'    mov.b{storage} %copied{index}, %converted{index};')
    if form == 'join':
        lines.append('JOINED:')
    for index, (_, storage) in enumerate(contracts):
        register = f'copied{index}' if form == 'copy' else f'converted{index}'
        lines.append(f'    st.local.b{storage} [%scratch+{8 * index}], %{register};')
    if roundtrip_format(label):
        register = 'copied0' if form == 'copy' else 'converted0'
        lines.extend((f'    cvt.rzi.s32.{roundtrip_format(label)} %roundtrip, %{register};',
                      '    st.local.b32 [%scratch+8], %roundtrip;'))
    lines.extend((f'    mul.wide.u32 %output_offset, %lane, {8 * words};',
                  '    add.u64 %output_address, %output, %output_offset;'))
    for index in range(words):
        if label.startswith('f32-store-u32-'):
            register = 'copied0' if form == 'copy' else 'converted0'
            # The second store must leave the trailing four initialized bytes
            # unchanged, even though its source register contains 64 bits.
            lines.extend((f'    st.global.u64 [%output_address], {POISON};',
                          f'    st.global.f32 [%output_address], %{register};'))
        else:
            lines.extend((f'    ld.local.u64 %read{index}, [%scratch+{8 * index}];',
                          f'    st.global.u64 [%output_address+{8 * index}], %read{index};'))
    lines.extend(('DONE:', '    ret;', '}'))
    return '\n'.join(lines) + '\n'


def fixtures():
    """Pure construction: 12 integer cases and 12 float storage/source cases."""
    return [(label, fixture_source(label)) for label in CASE_NAMES]


def inputs_and_expected(label):
    source, _, contracts = specification(label)
    if label.startswith('f16-source-'):
        half_values = (-1024.0, -511.0, -7.0, -2.0, -1.5, -1.0, -0.5,
                       -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 7.0, 511.0, 1024.0)
        seeds, expected = [], []
        for lane in range(65):
            packed = struct.pack('<e', half_values[lane % len(half_values)])
            raw_half = struct.unpack('<H', packed)[0]
            # These bits must be discarded before the binary16 bitcast.
            seed = ((0xA5A5 if lane % 2 == 0 else 0x5A5A) << 16) | raw_half
            seeds.append(seed)
            integer = int(struct.unpack('<e', packed)[0])
            expected.extend(((POISON & 0xFFFFFFFF00000000) | seed,
                             (POISON & 0xFFFFFFFF00000000) | (integer & 0xFFFFFFFF)))
    elif label.startswith('f32-roundtrip-'):
        boundaries = (0, 1, -1, 32767, -32768, (1 << 24) - 1, 1 << 24,
                      (1 << 24) + 1, (1 << 24) + 3, -((1 << 24) + 1),
                      -((1 << 24) + 3), (1 << 30) - 1, -((1 << 30) - 1))
        seeds, expected = [], []
        for lane in range(65):
            value = boundaries[lane % len(boundaries)]
            packed = struct.pack('<f', float(value))
            raw_float = struct.unpack('<I', packed)[0]
            integer = int(struct.unpack('<f', packed)[0])
            seeds.append(value & MASK64)
            expected.extend((raw_float, (POISON & 0xFFFFFFFF00000000) |
                             (integer & 0xFFFFFFFF)))
    elif label.startswith('f32-store-u32-'):
        boundaries = (0, 1, 32767, 32768, 65535, (1 << 24) - 1, 1 << 24,
                      (1 << 24) + 1, (1 << 24) + 3, (1 << 31) - 1,
                      1 << 31, (1 << 32) - 2, (1 << 32) - 1)
        seeds, expected = [], []
        for lane in range(65):
            value = boundaries[lane % len(boundaries)]
            seeds.append(0xA5A55A5A00000000 | value)
            raw_float = struct.unpack('<I', struct.pack('<f', float(value)))[0]
            expected.append((POISON & 0xFFFFFFFF00000000) | raw_float)
    elif label.startswith('f32-'):
        boundaries = (0, 1, -1, 32767, -32768, (1 << 24) - 1, 1 << 24,
                      (1 << 24) + 1, (1 << 24) + 3, -((1 << 24) + 1),
                      -((1 << 24) + 3), (1 << 31) - 1, -(1 << 31),
                      (1 << 31) + 1, -((1 << 31) + 1), (1 << 52) - 1,
                      -((1 << 52) - 1), 1 << 52, -(1 << 52))
        signed_inputs = [boundaries[lane % len(boundaries)] for lane in range(65)]
        # Every integer is exactly representable in binary64. Packing to f32
        # supplies an independent nearest-even reference, including negative
        # results whose sign bit must remain bit31 rather than extend to bit63.
        expected = [struct.unpack('<I', struct.pack('<f', float(value)))[0]
                    for value in signed_inputs]
        seeds = [value & MASK64 for value in signed_inputs]
    else:
        boundaries = (0, 1, 0x7fff, 0x8000, 0x8001, 0xffff, 0x10000,
                      0x7fffffff, 0x80000000, 0x80000001, 0xffffffff,
                      0x100000000, (1 << 63) - 1, 1 << 63, MASK64 - 1, MASK64)
        upper_poison = (0, 0xA5A55A5A00000000, 0xFFFF000000000000,
                        0x7EADBEEF00000000)
        seeds = [(boundaries[lane % len(boundaries)] ^
                  upper_poison[(lane // len(boundaries)) % len(upper_poison)])
                 for lane in range(65)]
        expected = []
        for seed in seeds:
            source_value = seed & ((1 << int(source[1:])) - 1)
            for destination, storage in contracts:
                width = int(destination[1:])
                chopped = source_value & ((1 << width) - 1)
                if destination[0] == 's' and chopped & (1 << (width - 1)):
                    chopped -= 1 << width
                stored = chopped & ((1 << storage) - 1)
                if storage == 32:
                    stored |= POISON & 0xFFFFFFFF00000000
                expected.append(stored)
    values = [word for lane, seed in enumerate(seeds) for word in (seed, lane & 1)]
    return values, expected


def negative_fixtures():
    joined = fixture_source('u64-join')
    missing = (conversion('u64', 's16', 'copied0') + '\n' +
               '    mov.b32 %converted0, %copied0;\n')
    if joined.count(missing) != 1:
        raise AssertionError('undefined-edge fixture replacement is no longer unique')
    undefined_source = fixture_source('u64-direct').replace(
        '    ld.global.b64 %source, [%input_address];\n', '')
    incompatible = joined.replace(conversion('u64', 's16', 'copied2'),
                                  '    cvt.s16.u64 %copied2, 1;')
    incompatible = incompatible.replace(conversion('u64', 's16', 'converted2'),
                                        '    mov.b64 %converted2, %input;')
    cases = [('undefined incoming conversion', joined.replace(missing, ''),
             'undefined on an incoming edge'),
            ('undefined conversion source', undefined_source, 'used before definition'),
            ('incompatible conversion join', incompatible,
             'pointer branch argument requires a pointer or proven null')]
    for destination in ('s16', 'u16'):
        source = fixture_source('s32-direct').replace(
            f'cvt.{destination}.s32', f'cvt.sat.{destination}.s32')
        cases.append((f'unsupported sat {destination}', source,
                      'unsupported saturating PTX conversion modifier'))
    source = fixture_source('f32-s64-direct').replace('cvt.rn.f32.s64', 'cvt.rn.sat.f32.s64')
    cases.append(('unsupported sat f32', source, 'unsupported saturating PTX conversion modifier'))
    for rounding in ('rz', 'rm', 'rp'):
        for storage in (32, 64):
            source = fixture_source('f32-s64-direct').replace('cvt.rn.f32.s64', f'cvt.{rounding}.f32.s64')
            if storage == 32:
                source = source.replace('.reg .b64 %converted0, %copied0;', '.reg .b32 %converted0, %copied0;')
                source = source.replace('st.local.b64 [%scratch+0], %converted0;',
                                        'st.local.b32 [%scratch+0], %converted0;')
            cases.append((f'unsupported {rounding} f32 b{storage}', source,
                          'unsupported directed integer-to-f32 conversion rounding'))
    return cases


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
                     'conversion storage: ' + label, entry=ENTRY, input_words=2,
                     output_words=output_words(label), abi_lines=ABI,
                     artifacts_dir=artifacts)


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
        record = dict(case=label, ptx_sha256=digest(ptx), inputs=values,
                      expected=expected, input_count=65,
                      output_words=output_words(label))
        if args.stage == 'translate':
            measured = run_command([str(compiler), str(ptx), '--backend=cumetal-ir',
                                    '--ptx-strict', '--entry', ENTRY, '--emit=msl',
                                    '-o', str(msl)], 30)
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
            numerical = f'NUMERICAL_PASS conversion storage: {label}: 65 inputs, output values, guards'
            executed_ptx = artifacts / 'test.ptx'
            same_input = executed_ptx.is_file() and digest(executed_ptx) == record['ptx_sha256']
            good = (not measured['timed_out'] and measured['exit_code'] == 0 and provenance
                    and same_input and numerical in measured['stdout'].splitlines())
            record.update(gpu_launches=launches, gpu_provenance_matches=provenance,
                          executed_input_matches=same_input,
                          executed_artifacts_sha256={path.name: digest(path)
                              for path in sorted(artifacts.glob('*')) if path.is_file()})
            record['status'] = 'NUMERICAL_PASS' if good else 'NUMERICAL_FAILED'
        record['passed'] = good
        record['process'] = measured
        records.append(record)
        (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(record['status'], label, flush=True)
        if not good:
            print(measured['stderr'], end='', file=sys.stderr)
    for label, source, diagnostic in negative_fixtures():
        folder = evidence / label.replace(' ', '-')
        folder.mkdir()
        ptx, msl = folder / 'input.ptx', folder / 'output.metal'
        ptx.write_text(source)
        measured = run_command([str(compiler), str(ptx), '--backend=cumetal-ir',
                                '--ptx-strict', '--entry', ENTRY, '--emit=msl', '-o', str(msl)], 30)
        good = (not measured['timed_out'] and measured['exit_code'] != 0 and
                diagnostic in measured['stderr'] and not msl.exists() and
                not Path(str(msl) + '.cumetal-abi').exists())
        record = dict(case=label, ptx_sha256=digest(ptx), diagnostic=diagnostic,
                      process=measured, passed=good,
                      status='EXPECTED_REJECTION' if good else 'NEGATIVE_FAILED')
        records.append(record)
        (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(record['status'], label, flush=True)
        if not good:
            print(measured['stderr'], end='', file=sys.stderr)
    report = dict(schema=1, stage=args.stage, compiler=str(compiler), compiler_sha256=digest(compiler),
                  script_sha256=digest(Path(__file__).resolve()),
                  runtime_sha256=digest(args.build / 'libcumetal.dylib') if args.stage == 'numerical' else None,
                  required_positive_cases=len(selected), required_negative_cases=len(negative_fixtures()),
                  full_positive_denominator=len(CASE_NAMES), statuses=dict(Counter(item['status'] for item in records)),
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
    parser.add_argument('--output', type=Path, help='new or empty directory for retained evidence')
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
    with tempfile.TemporaryDirectory(prefix='cumetal-cvt-storage-') as work:
        return run_suite(args, Path(work))


if __name__ == '__main__':
    raise SystemExit(main())
