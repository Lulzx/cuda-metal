#!/usr/bin/env python3
"""Numerical scalar-zero guards without inventing an absent payload.

Sixteen fixtures cover literal/dynamic 32/64-bit markers and private-pointer
markers, copies, signed/unsigned eq/ne, commuted operands and reordered blocks.
Four also select between a default and the conditional scalar payload, including
a terminal block without a later conditional branch.
Each executes 65 independent inputs and checks three output words: an observable
pre-guard write, a scalar payload and a value read through a payload pointer.
Unsafe variants are compiler-only rejection controls and are never launched.
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
ENTRY = 'scalar_zero_guard'
PREFIX_XOR = 0x6B8B4567327B23C6
POINTER_XOR = 0xC0DEC0DE12345678
ABSENT_SCALAR = 0xD15EA5ED5CA1AB1E
ABSENT_POINTER = 0xF00DFACE0BADCAFE
ABI = ['CUMETAL_ABI_V2', 'kernel ' + ENTRY, 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']


def specifications():
    cases = []
    for width in (32, 64):
        for kind, relation, signed, copied, reversed_layout in (
                ('literal', 'eq', False, False, False),
                ('literal', 'ne', True, True, True),
                ('dynamic', 'eq', True, True, False),
                ('dynamic', 'ne', False, False, True)):
            cases.append(dict(kind=kind, width=width, relation=relation,
                              signed=signed, copied=copied, reverse=reversed_layout,
                              commuted=reversed_layout))
    for relation, signed, reversed_layout in (
            ('eq', False, False), ('ne', True, True),
            ('eq', True, True), ('ne', False, False)):
        cases.append(dict(kind='private', width=64, relation=relation,
                          signed=signed, copied=True, reverse=reversed_layout,
                          commuted=reversed_layout))
    for kind, width, relation, signed, reverse, terminal in (
            ('literal', 32, 'eq', False, False, False),
            ('dynamic', 32, 'ne', True, True, False),
            ('literal', 64, 'eq', True, True, False),
            ('dynamic', 64, 'ne', False, False, True)):
        cases.append(dict(kind=kind, width=width, relation=relation, signed=signed,
                          copied=True, reverse=reverse, commuted=reverse,
                          select=True, terminal=terminal))
    for case in cases:
        fmt = ('s' if case['signed'] else 'u') + str(case['width'])
        case['name'] = '-'.join((case['kind'], fmt, case['relation'],
                                'copy' if case['copied'] else 'direct',
                                'reordered' if case['reverse'] else 'forward'))
        if case.get('select'):
            case['name'] = ('terminal-select-' if case['terminal'] else 'select-') + case['name']
    return cases


CASES = specifications()
CASE_NAMES = tuple(case['name'] for case in CASES)
NEGATIVES = ('overwritten-marker', 'predicated-overwrite', 'unguarded-payload')


def negative_cases(case):
    return NEGATIVES + (('swapped-select-arms', 'overwritten-select-predicate')
                        if case.get('select') else ())


def test_inputs():
    words = [0, 1, MASK64, 1 << 63, (1 << 63) - 1,
             0x0123456789ABCDEF, 0xFEDCBA9876543210]
    markers = [0, 1, 1 << 31, 1 << 32, 1 << 63, MASK64, 0x8000000100000000]
    values = []
    for lane in range(65):
        # Bits other than bit0 do not determine presence. The two cycles are
        # coprime, so present lanes include both zero and nonzero markers.
        selector = (lane * 0x9E3779B97F4A7C15) & MASK64
        word = words[lane] if lane < len(words) else (lane * 0xD1342543DE82EF95) & MASK64
        values.extend((selector, word, markers[lane % len(markers)]))
    return values


def expected_values(case, values):
    expected = []
    marker_mask = (1 << case['width']) - 1
    for offset in range(0, len(values), 3):
        selector, word, marker = values[offset:offset + 3]
        selected = bool(selector & 1)
        if case['kind'] == 'dynamic':
            selected = selected and bool(marker & marker_mask)
        scalar = (word + 17) & MASK64 if selected else ABSENT_SCALAR
        if case.get('select'):
            scalar = ((word + 17) & marker_mask) if selected else 8
        pointer = word ^ POINTER_XOR if selected else ABSENT_POINTER
        if case.get('terminal'):
            pointer = ABSENT_POINTER
        expected.extend((word ^ PREFIX_XOR, scalar, pointer))
    return expected


def fixture_source(case, negative=None):
    width = case['width']
    fmt = ('s' if case['signed'] else 'u') + str(width)
    lines = ['.version 7.1', '.target sm_80', '.address_size 64',
             f'.visible .entry {ENTRY}(.param .u64 .ptr .global input,',
             ' .param .u64 .ptr .global output, .param .u32 count) {',
             ' .local .align 8 .b8 scratch[8];',
             ' .reg .b64 %input, %output, %offset, %selector, %choice, %word, %dynamic;',
             ' .reg .b64 %payload, %payload_address, %loaded, %answer, %prefix, %selected_result;',
             ' .reg .b64 %local, %generic, %private_address;',
             f' .reg .b{width} %zero, %marker, %copied, %converted, %narrow_payload, %selected;',
             ' .reg .b32 %lane, %block, %threads, %count;',
             ' .reg .pred %done, %absent, %guard;',
             ' ld.param.u64 %input, [input];', ' ld.param.u64 %output, [output];',
             ' ld.param.u32 %count, [count];', ' mov.u32 %lane, %tid.x;',
             ' mov.u32 %block, %ctaid.x;', ' mov.u32 %threads, %ntid.x;',
             ' mad.lo.u32 %lane, %block, %threads, %lane;',
             ' setp.ge.u32 %done, %lane, %count;', ' @%done bra DONE;',
             ' mul.wide.u32 %offset, %lane, 24;',
             ' add.u64 %input, %input, %offset;', ' add.u64 %output, %output, %offset;',
             ' ld.global.u64 %selector, [%input];',
             ' ld.global.u64 %word, [%input+8];',
             ' ld.global.u64 %dynamic, [%input+16];',
             f' st.global.u64 [%output], {PREFIX_XOR};',
             ' and.b64 %choice, %selector, 1;', ' setp.eq.u64 %absent, %choice, 0;',
             f' mov.u{width} %zero, 0;', f' mov.b{width} %marker, %zero;',
             ' @%absent bra JOIN;']
    present = ['PRESENT:', ' add.u64 %payload, %word, 17;',
               ' add.u64 %payload_address, %input, 8;']
    if case.get('select'):
        present.append(f' cvt.u{width}.u64 %narrow_payload, %payload;')
    if case['kind'] == 'literal':
        present.append(f' mov.u{width} %marker, 1;')
    elif case['kind'] == 'dynamic':
        present.extend((f' cvt.u{width}.u64 %converted, %dynamic;',
                        f' mov.b{width} %marker, %converted;'))
    elif case['kind'] == 'private':
        present.extend((' mov.u64 %local, scratch;', ' cvta.local.u64 %generic, %local;',
                        ' st.local.u64 [%generic], %word;', ' mov.b64 %marker, %generic;'))
    else:
        raise ValueError(case['kind'])
    present.append(' bra JOIN;')
    if case['reverse']:
        lines.append(' bra PRESENT;')
    else:
        lines.extend(present)
    lines.append('JOIN:')
    # These writes are observable on both paths, including the absent path that
    # skips the payload. Specialization must retain them exactly once.
    lines.extend((' ld.global.u64 %prefix, [%output];',
                  ' xor.b64 %prefix, %prefix, %word;',
                  ' st.global.u64 [%output], %prefix;',
                  f' st.global.u64 [%output+8], {ABSENT_SCALAR};',
                  f' st.global.u64 [%output+16], {ABSENT_POINTER};'))
    if negative == 'overwritten-marker':
        lines.append(f' mov.u{width} %marker, 1;')
    elif negative == 'predicated-overwrite':
        lines.append(f' @%absent mov.u{width} %marker, 1;')
    elif negative == 'unguarded-payload':
        lines.append(' st.global.u64 [%output+8], %payload;')
    elif negative not in (None, 'swapped-select-arms', 'overwritten-select-predicate'):
        raise ValueError(negative)
    compared = '%marker'
    if case['copied']:
        lines.append(f' mov.b{width} %copied, %marker;')
        compared = '%copied'
    operands = f'0, {compared}' if case['commuted'] else f'{compared}, 0'
    lines.append(f' setp.{case["relation"]}.{fmt} %guard, {operands};')
    if case.get('select'):
        if negative == 'overwritten-select-predicate':
            lines.append(' mov.pred %guard, ' + ('0;' if case['relation'] == 'eq' else '1;'))
        arms = ['8', '%narrow_payload'] if case['relation'] == 'eq' else ['%narrow_payload', '8']
        if negative == 'swapped-select-arms':
            arms.reverse()
        lines.extend((f' selp.b{width} %selected, {arms[0]}, {arms[1]}, %guard;',
                      f' cvt.u64.u{width} %selected_result, %selected;',
                      ' st.global.u64 [%output+8], %selected_result;'))
    if not case.get('terminal'):
        lines.append(' @' + ('!' if case['relation'] == 'ne' else '') + '%guard bra DONE;')
        if case['kind'] == 'private':
            lines.extend((f' cvta.to.local.u64 %private_address, {compared};',
                          ' ld.local.u64 %loaded, [%private_address];'))
        else:
            lines.append(' ld.global.u64 %loaded, [%payload_address];')
        lines.append(f' xor.b64 %answer, %loaded, {POINTER_XOR};')
        if not case.get('select'):
            lines.append(' st.global.u64 [%output+8], %payload;')
        lines.append(' st.global.u64 [%output+16], %answer;')
    lines.append(' bra DONE;')
    if case['reverse']:
        lines.extend(present)
    return '\n'.join(lines + ['DONE:', ' ret;', '}']) + '\n'


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


def worker(build, case, artifacts):
    from ptx_test_support import run_integer_case
    values = test_inputs()
    run_integer_case(build, fixture_source(case), values, expected_values(case, values),
                     'scalar zero guard: ' + case['name'], entry=ENTRY,
                     input_words=3, output_words=3, abi_lines=ABI, artifacts_dir=artifacts)


def run_suite(args, evidence):
    compiler = args.build / 'cumetalc'
    records = []
    selected = [case for case in CASES if args.case is None or case['name'] == args.case]
    for case in selected:
        folder = evidence / case['name']
        folder.mkdir()
        ptx, msl = folder / 'input.ptx', folder / 'output.metal'
        ptx.write_text(fixture_source(case))
        values = test_inputs()
        record = dict(case=case['name'], ptx_sha256=digest(ptx), inputs=values,
                      expected=expected_values(case, values), input_count=65, output_words=3)
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
                                    '--case', case['name'], '--gpu-child', '--artifacts', str(artifacts)], 90)
            launches = [line for line in measured['stderr'].splitlines()
                        if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
            provenance = len(launches) == 1 and all(token in launches[0] for token in (
                f'kernel="{ENTRY}"', 'device=apple_gpu ', 'launch_success=true ',
                'source=generic_ptx ', 'provenance=generic_ptx_lowering ', 'semantic_quality=exact '))
            executed = artifacts / 'test.ptx'
            same_input = executed.is_file() and digest(executed) == record['ptx_sha256']
            numerical = f'NUMERICAL_PASS scalar zero guard: {case["name"]}: 65 inputs, output values, guards'
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
        print(record['status'], case['name'], flush=True)
        if not good:
            print(measured['stderr'], end='', file=sys.stderr)

    # Invalid controls are compiled only. Never execute an undefined payload or
    # a pointer whose defining path was not taken.
    for case in selected:
        for negative in negative_cases(case):
            name = case['name'] + '--' + negative
            folder = evidence / name
            folder.mkdir()
            ptx, msl = folder / 'input.ptx', folder / 'output.metal'
            ptx.write_text(fixture_source(case, negative))
            measured = run_command([str(compiler), str(ptx), '--backend=cumetal-ir',
                                    '--ptx-strict', '--entry', ENTRY, '--emit=msl', '-o', str(msl)], 30)
            definedness_error = ('PTX register ' in measured['stderr'] and
                                 any(message in measured['stderr'] for message in (
                                     ' is undefined ', ' is used before definition ')))
            good = (not measured['timed_out'] and measured['exit_code'] != 0 and
                    definedness_error and not msl.exists() and
                    not Path(str(msl) + '.cumetal-abi').exists())
            record = dict(case=name, ptx_sha256=digest(ptx), passed=good, process=measured,
                          status='EXPECTED_REJECTION' if good else 'REJECTION_FAILED')
            records.append(record)
            (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
            print(record['status'], name, flush=True)
    report = dict(schema=1, stage=args.stage, compiler=str(compiler), compiler_sha256=digest(compiler),
                  script_sha256=digest(Path(__file__).resolve()),
                  helper_sha256=digest(Path(__file__).with_name('ptx_test_support.py')),
                  runtime_sha256=digest(args.build / 'libcumetal.dylib') if args.stage == 'numerical' else None,
                  required_positive_cases=len(selected), full_positive_denominator=len(CASES),
                  required_rejections=sum(len(negative_cases(case)) for case in selected),
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
        worker(args.build, next(case for case in CASES if case['name'] == args.case), args.artifacts)
        return 0
    if args.output is not None:
        evidence = args.output.resolve()
        evidence.mkdir(parents=True, exist_ok=True)
        if any(evidence.iterdir()):
            parser.error('--output must be a new or empty directory')
        return run_suite(args, evidence)
    with tempfile.TemporaryDirectory(prefix='cumetal-scalar-zero-guards-') as work:
        return run_suite(args, Path(work))


if __name__ == '__main__':
    raise SystemExit(main())
