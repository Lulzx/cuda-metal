#!/usr/bin/env python3
"""Unsigned narrowing may discard an otherwise undefined packed high lane.

Twelve numerical cases: u16/u32 x natural/b64 storage x straight/independent/loop.
Each checks 65 lanes against integer-only references, including a separate fully
defined pack with an observable full-width use. Compile-only rejection controls
cover undefined reads and intentionally unsupported normalization shapes.
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


MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
EFFECT_SEED = 0x6B8B4567327B23C6
ENTRY = 'narrow_pack_probe'
ABI = ['CUMETAL_ABI_V2', 'kernel ' + ENTRY, 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
CASES = tuple(dict(format=width, storage=width if storage == 'natural' else 64,
                   layout=layout, name=f'u{width}-{storage}-{layout}')
              for width in (16, 32) for storage in ('natural', 'wide')
              for layout in ('straight', 'independent', 'loop'))
CASE_NAMES = tuple(case['name'] for case in CASES)
NEGATIVES = ('observed-high', 'full-width-use', 'second-narrow-use', 'undefined-low',
             'predicated-pack', 'predicated-consumer', 'overwritten-low',
             'predicated-low-overwrite', 'call-return-low', 'call-return-packed',
             'intervening-call')


def test_inputs():
    boundaries = [0, 1, 0x7F, 0x80, 0xFF, 0x100, 0x7FFF, 0x8000, 0xFFFF,
                  0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    high_boundaries = [0, MASK32, 1, 0x80000000, 0x7FFFFFFF, 0xDEADBEEF]
    values = []
    for lane in range(65):
        low = boundaries[lane] if lane < len(boundaries) else (lane * 0x9E3779B9) & MASK32
        high = (high_boundaries[lane] if lane < len(high_boundaries)
                else (lane * 0x85EBCA6B + 0x12345678) & MASK32)
        # Only the low32 of each input word is loaded. Poison surrounding bytes
        # independently; input readback checks them and both buffer guards.
        values.extend((0xA55A699600000000 | low, 0x5AA5966900000000 | high))
    return values


def expected_values(case, values):
    expected = []
    mask = (1 << case['format']) - 1
    for offset in range(0, len(values), 2):
        low, high = (word & MASK32 for word in values[offset:offset + 2])
        expected.extend((low & mask, (low >> 8) & 0xFF, low & mask,
                         low | (high << 32),
                         EFFECT_SEED ^ low ^ (3 if case['layout'] == 'loop' else 0)))
    return expected


def fixture_source(case, negative=None):
    fmt, storage = case['format'], case['storage']
    helpers = ''
    if negative == 'call-return-low':
        helpers = '''.func (.param .b32 result) change_low() {
 .reg .b32 %value;
 mov.b32 %value, 99;
 st.param.b32 [result], %value;
 ret;
}
'''
    elif negative == 'call-return-packed':
        helpers = '''.func (.param .b64 result) change_pack() {
 .reg .b64 %value;
 mov.b64 %value, 99;
 st.param.b64 [result], %value;
 ret;
}
'''
    elif negative == 'intervening-call':
        helpers = '.func helper() { ret; }\n'
    lines = ['.version 7.1', '.target sm_80', '.address_size 64', helpers,
             f'.visible .entry {ENTRY}(.param .u64 .ptr .global input,',
             ' .param .u64 .ptr .global output, .param .u32 count) {',
             ' .reg .b64 %input, %output, %input_offset, %output_offset;',
             ' .reg .b64 %packed, %reference, %answer, %reference_answer, %byte;',
             ' .reg .b64 %low_wide, %counter_wide, %effect;',
             ' .reg .b32 %low, %high, %undefined_high, %extracted_low, %extracted_high;',
             ' .reg .b32 %lane, %block, %threads, %count, %counter, %parity, %extra;',
             f' .reg .b{storage} %converted, %reference_converted;',
             ' .reg .pred %done, %again, %guard;',
             ' .param .b32 return32;', ' .param .b64 return64;',
             ' ld.param.u64 %input, [input];', ' ld.param.u64 %output, [output];',
             ' ld.param.u32 %count, [count];', ' mov.u32 %lane, %tid.x;',
             ' mov.u32 %block, %ctaid.x;', ' mov.u32 %threads, %ntid.x;',
             ' mad.lo.u32 %lane, %block, %threads, %lane;',
             ' setp.ge.u32 %done, %lane, %count;', ' @%done bra DONE;',
             ' mul.wide.u32 %input_offset, %lane, 16;',
             ' mul.wide.u32 %output_offset, %lane, 40;',
             ' add.u64 %input, %input, %input_offset;',
             ' add.u64 %output, %output, %output_offset;']
    if negative != 'undefined-low':
        lines.append(' ld.global.u32 %low, [%input];')
    lines.extend((' ld.global.u32 %high, [%input+8];',
                  ' cvt.u64.u32 %low_wide, %low;',
                  ' mov.u32 %counter, 0;', ' and.b32 %parity, %lane, 1;',
                  ' setp.eq.u32 %guard, %parity, 0;'))
    if case['layout'] == 'loop':
        lines.extend(('LOOP:', ' add.u32 %counter, %counter, 1;',
                      ' setp.lt.u32 %again, %counter, 3;', ' @%again bra LOOP;'))
    lines.append(' cvt.u64.u32 %counter_wide, %counter;')
    if case['layout'] == 'independent':
        lines.append(f' st.global.u64 [%output+32], {EFFECT_SEED};')
    pack = 'mov.b64 %packed, {%low, %undefined_high};'
    lines.append(' ' + ('@%guard ' if negative == 'predicated-pack' else '') + pack)
    if case['layout'] == 'independent':
        # An observable read/modify/write between the pair must be retained,
        # not moved across the pack or duplicated by a rewrite.
        lines.extend((' ld.global.u64 %effect, [%output+32];',
                      ' xor.b64 %effect, %effect, %low_wide;',
                      ' xor.b64 %effect, %effect, %counter_wide;',
                      ' st.global.u64 [%output+32], %effect;'))
    if negative == 'overwritten-low':
        lines.append(' add.u32 %low, %low, 1;')
    elif negative == 'predicated-low-overwrite':
        lines.append(' @%guard mov.u32 %low, 99;')
    elif negative == 'call-return-low':
        lines.extend((' call.uni (return32), change_low, ();', ' ld.param.b32 %low, [return32];'))
    elif negative == 'call-return-packed':
        lines.extend((' call.uni (return64), change_pack, ();', ' ld.param.b64 %packed, [return64];'))
    elif negative == 'intervening-call':
        lines.append(' call.uni helper, ();')
    lines.append(' ' + ('@%guard ' if negative == 'predicated-consumer' else '') +
                 f'cvt.u{fmt}.u64 %converted, %packed;')
    if negative == 'observed-high':
        lines.extend((' mov.b64 {%extracted_low, %extracted_high}, %packed;',
                      ' st.global.u32 [%output+32], %extracted_high;'))
    elif negative == 'full-width-use':
        lines.append(' st.global.u64 [%output+32], %packed;')
    elif negative == 'second-narrow-use':
        lines.extend((' cvt.u32.u64 %extra, %packed;', ' st.global.u32 [%output+32], %extra;'))
    elif negative is not None and negative not in NEGATIVES:
        raise ValueError(negative)
    if storage == 64:
        # Observe the entire declared container, including the upper bits that
        # the conversion must zero-extend. Re-narrowing here would hide a bug.
        lines.append(' mov.b64 %answer, %converted;')
    else:
        lines.append(f' cvt.u64.u{fmt} %answer, %converted;')
    lines.extend((' shr.u64 %byte, %answer, 8;', ' and.b64 %byte, %byte, 255;',
                  ' st.global.u64 [%output], %answer;', ' st.global.u64 [%output+8], %byte;',
                  ' mov.b64 %reference, {%low, %high};',
                  f' cvt.u{fmt}.u64 %reference_converted, %reference;'))
    lines.append(' mov.b64 %reference_answer, %reference_converted;' if storage == 64
                 else f' cvt.u64.u{fmt} %reference_answer, %reference_converted;')
    lines.extend((' st.global.u64 [%output+16], %reference_answer;',
                  ' st.global.u64 [%output+24], %reference;'))
    if case['layout'] != 'independent':
        lines.extend((f' xor.b64 %effect, %low_wide, {EFFECT_SEED};',
                      ' xor.b64 %effect, %effect, %counter_wide;',
                      ' st.global.u64 [%output+32], %effect;'))
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
                     'narrow pack: ' + case['name'], entry=ENTRY, input_words=2,
                     output_words=5, abi_lines=ABI, artifacts_dir=artifacts)


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
                      expected=expected_values(case, values), input_count=65, output_words=5)
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
            numerical = f'NUMERICAL_PASS narrow pack: {case["name"]}: 65 inputs, output values, guards'
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

    # Check each format/storage contract. These controls never run on the GPU.
    negative_bases = [case for case in selected if case['layout'] == 'straight']
    for case in negative_bases:
        for negative in NEGATIVES:
            name = case['name'] + '--' + negative
            folder = evidence / name
            folder.mkdir()
            ptx, msl = folder / 'input.ptx', folder / 'output.metal'
            ptx.write_text(fixture_source(case, negative))
            measured = run_command([str(compiler), str(ptx), '--backend=cumetal-ir',
                                    '--ptx-strict', '--entry', ENTRY, '--emit=msl', '-o', str(msl)], 30)
            definedness = ('PTX register ' in measured['stderr'] and any(message in measured['stderr']
                           for message in (' is undefined ', ' is used before definition ')))
            good = (not measured['timed_out'] and measured['exit_code'] != 0 and definedness and
                    not msl.exists() and not Path(str(msl) + '.cumetal-abi').exists())
            record = dict(case=name, ptx_sha256=digest(ptx), passed=good, process=measured,
                          status='EXPECTED_REJECTION' if good else 'REJECTION_FAILED')
            records.append(record)
            (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
            print(record['status'], name, flush=True)
            if not good:
                print(measured['stderr'], end='', file=sys.stderr)
    report = dict(schema=1, stage=args.stage, compiler=str(compiler), compiler_sha256=digest(compiler),
                  script_sha256=digest(Path(__file__).resolve()),
                  helper_sha256=digest(Path(__file__).with_name('ptx_test_support.py')),
                  runtime_sha256=digest(args.build / 'libcumetal.dylib') if args.stage == 'numerical' else None,
                  required_positive_cases=len(selected), full_positive_denominator=len(CASES),
                  required_rejections=len(negative_bases) * len(NEGATIVES),
                  full_rejection_denominator=4 * len(NEGATIVES),
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
    with tempfile.TemporaryDirectory(prefix='cumetal-narrow-pack-') as work:
        return run_suite(args, Path(work))


if __name__ == '__main__':
    raise SystemExit(main())
