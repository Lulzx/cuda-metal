#!/usr/bin/env python3
"""Saved pointer cells survive bounded, disjoint device-helper writes.

Five configurations check 65 runtime lanes, exact pointee and byte-write results,
ABI, buffer guards and Apple-GPU provenance. Refusals compile only; --stage
translate never loads a module or launches a kernel.
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


MASK = (1 << 64) - 1
PRIVATE_XOR = 0x0123456789ABCDEF
GUARD_XOR = 0xFEDCBA9876543210
GLOBAL_XOR = 0x6B8B4567327B23C6
SCALAR_XOR = 0xA55A96695AA56996
ENTRY = 'call_write_footprints'
ABI = ['CUMETAL_ABI_V2', 'kernel ' + ENTRY, 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
CASES = ('same-allocation', 'separate-allocation', 'nested-helper',
         'read-only-control', 'two-pointer-arguments')
NEGATIVES = ('overlap', 'partial', 'unknown-offset', 'unknown-call', 'recursive-call',
             'missing-argument', 'guarded-argument', 'ambiguous-argument', 'escaped-cell')


def inputs():
    boundaries = (0, 1, 255, 256, (1 << 32) - 1, 1 << 32,
                  (1 << 63) - 1, 1 << 63, MASK - 1, MASK)
    return [word for lane in range(65) for word in (
        boundaries[lane] if lane < len(boundaries) else (lane * 0x9E3779B97F4A7C15) & MASK,
        (0x89ABCDEF01234567 ^ (lane + 17) * 0x85EBCA6B27D4EB2F) & MASK,
        boundaries[lane] if lane < len(boundaries) else (lane * 0x0102030405060749 + 7) & MASK)]


def expected(case, values):
    result = []
    for a, b, tag in zip(values[::3], values[1::3], values[2::3]):
        byte = tag & 255
        pointee = a ^ PRIVATE_XOR if case == 'separate-allocation' else a
        local = b if case == 'read-only-control' else (b & (MASK ^ 255)) | byte
        global_word = b ^ GLOBAL_XOR
        if case == 'two-pointer-arguments':
            global_word = (global_word & (MASK ^ (255 << 24))) | ((byte ^ 90) << 24)
        observed = b & 255 if case == 'read-only-control' else byte
        # Full words make a widened byte store observable. The guard and scalar
        # control are derived independently from the immutable input values.
        result.extend((pointee, local, a ^ GUARD_XOR, global_word,
                       observed, tag ^ a ^ SCALAR_XOR))
    return result


def helpers(case, negative):
    if negative == 'unknown-call':
        return ''
    if case == 'read-only-control':
        return '''.func (.param .b32 result) read_byte(.param .b64 address) {
 .reg .b64 %raw, %pointer;
 .reg .b32 %value;
 ld.param.b64 %raw, [address];
 cvta.to.local.u64 %pointer, %raw;
 ld.u8 %value, [%pointer];
 st.param.b32 [result], %value;
 ret;
}
'''
    if case == 'two-pointer-arguments':
        return '''.func write_two(.param .b64 local_address, .param .b64 global_address,
 .param .b32 value) {
 .reg .b64 %local_raw, %global_raw, %local, %global;
 .reg .b32 %value, %alternate;
 ld.param.b64 %local_raw, [local_address];
 ld.param.b64 %global_raw, [global_address];
 cvta.to.local.u64 %local, %local_raw;
 cvta.to.global.u64 %global, %global_raw;
 ld.param.b32 %value, [value];
 xor.b32 %alternate, %value, 90;
 st.u8 [%local], %value;
 st.u8 [%global+3], %alternate;
 ret;
}
'''
    # Keep the public reducer's generic parameter/dereference shape in the
    # same-allocation case. Other writers make their formal space explicit.
    pointer_load = (' ld.param.b64 %pointer, [address];' if case == 'same-allocation' else
                    ' ld.param.b64 %raw, [address];\n cvta.to.local.u64 %pointer, %raw;')
    recursive = negative == 'recursive-call'
    recursive_slots = ' .param .b64 recurse_address;\n .param .b32 recurse_value;' if recursive else ''
    source = f'''.func write_byte(.param .b64 address, .param .b32 value) {{
 .reg .b64 %raw, %pointer;
 .reg .b32 %value;
{recursive_slots}
{pointer_load}
 ld.param.b32 %value, [value];
 st.u8 [%pointer], %value;
'''
    if recursive:
        source += ''' st.param.b64 [recurse_address], %pointer;
 st.param.b32 [recurse_value], %value;
 call.uni write_byte, (recurse_address, recurse_value);
 st.u8 [%pointer], %value;
'''
    source += ' ret;\n}\n'
    if case == 'nested-helper':
        source += '''.func nested_write(.param .b64 address, .param .b32 value) {
 .reg .b64 %raw, %pointer;
 .reg .b32 %value;
 .param .b64 leaf_address;
 .param .b32 leaf_value;
 ld.param.b64 %raw, [address];
 cvta.to.local.u64 %pointer, %raw;
 add.u64 %pointer, %pointer, 16;
 ld.param.b32 %value, [value];
 st.param.b64 [leaf_address], %pointer;
 st.param.b32 [leaf_value], %value;
 call.uni write_byte, (leaf_address, leaf_value);
 ret;
}
'''
    return source


def fixture(case, negative=None):
    if case not in CASES or negative is not None and negative not in NEGATIVES:
        raise ValueError('unknown call-write fixture')
    if negative is not None and case != 'same-allocation':
        raise ValueError('refusals use the same-allocation fixture')
    lines = ['.version 7.1', '.target sm_80', '.address_size 64', helpers(case, negative),
             f'.visible .entry {ENTRY}(.param .u64 .ptr .global input,',
             ' .param .u64 .ptr .global output, .param .u32 count) {',
             ' .local .align 16 .b8 depot[80];', ' .local .align 16 .b8 other[24];',
             ' .reg .b64 %input, %output, %offset, %a, %b, %tag, %control;',
             ' .reg .b64 %base, %cell, %target, %other, %payload, %private_value;',
             ' .reg .b64 %actual, %bad_actual, %unknown, %global_target, %global_initial;',
             ' .reg .b64 %saved, %pointee, %local_word, %guard, %observed;',
             ' .reg .b32 %lane, %block, %threads, %count, %parity, %byte, %answer;',
             ' .reg .pred %done, %choice;',
             ' .param .b64 argument;', ' .param .b64 global_argument;',
             ' .param .b32 value_argument;', ' .param .b32 return_value;',
             ' ld.param.u64 %input, [input];', ' ld.param.u64 %output, [output];',
             ' ld.param.u32 %count, [count];', ' mov.u32 %lane, %tid.x;',
             ' mov.u32 %block, %ctaid.x;', ' mov.u32 %threads, %ntid.x;',
             ' mad.lo.u32 %lane, %block, %threads, %lane;',
             ' setp.ge.u32 %done, %lane, %count;', ' @%done bra DONE;',
             ' mul.wide.u32 %offset, %lane, 24;', ' add.u64 %input, %input, %offset;',
             ' mul.wide.u32 %offset, %lane, 48;', ' add.u64 %output, %output, %offset;',
             ' ld.global.u64 %a, [%input];', ' ld.global.u64 %b, [%input+8];',
             ' ld.global.u64 %tag, [%input+16];', ' cvt.u32.u64 %byte, %tag;',
             ' and.b32 %byte, %byte, 255;', ' and.b32 %parity, %lane, 1;',
             ' setp.eq.u32 %choice, %parity, 0;',
             ' mov.u64 %base, depot;', ' add.u64 %cell, %base, 16;',
             ' mov.u64 %other, other;',
             (' add.u64 %target, %other, 8;' if case == 'separate-allocation' else ' add.u64 %target, %base, 24;'),
             ' add.u64 %payload, %base, 48;',
             f' xor.b64 %private_value, %a, {PRIVATE_XOR};',
             ' st.local.u64 [%payload], %private_value;', ' st.local.u64 [%target], %b;',
             f' xor.b64 %guard, %a, {GUARD_XOR};', ' st.local.u64 [%base+8], %guard;',
             ' st.local.b64 [%cell], ' + ('%payload;' if case == 'separate-allocation' else '%input;'),
             ' add.u64 %global_target, %output, 24;',
             f' xor.b64 %global_initial, %b, {GLOBAL_XOR};',
             ' st.global.u64 [%global_target], %global_initial;',
             (' add.u64 %actual, %base, 8;' if case == 'nested-helper' else ' mov.b64 %actual, %target;')]
    if negative == 'overlap':
        lines.append(' add.u64 %actual, %base, 16;')
    elif negative == 'partial':
        lines.append(' add.u64 %actual, %base, 23;')
    elif negative == 'unknown-offset':
        lines.extend((' and.b64 %unknown, %tag, 7;', ' add.u64 %actual, %cell, %unknown;'))
    if negative == 'ambiguous-argument':
        lines.extend((' add.u64 %bad_actual, %base, 23;', ' @%choice bra ARGUMENT_A;',
                      ' st.param.b64 [argument], %bad_actual;', ' bra ARGUMENT_DONE;',
                      'ARGUMENT_A:', ' st.param.b64 [argument], %actual;', 'ARGUMENT_DONE:'))
    elif negative != 'missing-argument':
        guard = '@%choice ' if negative == 'guarded-argument' else ''
        # Passing %cell directly retains the original escaped-cell discovery
        # regression, independent of recomputing an equal address from %base.
        actual = '%cell' if negative == 'escaped-cell' else '%actual'
        lines.append(f' {guard}st.param.b64 [argument], {actual};')
    lines.append(' st.param.b32 [value_argument], %byte;')
    if case == 'read-only-control':
        lines.extend((' call.uni (return_value), read_byte, (argument);',
                      ' ld.param.b32 %answer, [return_value];', ' cvt.u64.u32 %observed, %answer;'))
    elif case == 'two-pointer-arguments':
        lines.extend((' st.param.b64 [global_argument], %global_target;',
                      ' call.uni write_two, (argument, global_argument, value_argument);',
                      ' cvt.u64.u32 %observed, %byte;'))
    else:
        callee = 'missing_write' if negative == 'unknown-call' else 'nested_write' if case == 'nested-helper' else 'write_byte'
        lines.extend((f' call.uni {callee}, (argument, value_argument);', ' cvt.u64.u32 %observed, %byte;'))
    lines.extend((' ld.local.b64 %saved, [%cell];', ' ld.u64 %pointee, [%saved];',
                  ' ld.local.u64 %local_word, [%target];', ' ld.local.u64 %guard, [%base+8];',
                  ' st.global.u64 [%output], %pointee;', ' st.global.u64 [%output+8], %local_word;',
                  ' st.global.u64 [%output+16], %guard;', ' st.global.u64 [%output+32], %observed;',
                  ' xor.b64 %control, %tag, %a;', f' xor.b64 %control, %control, {SCALAR_XOR};',
                  ' st.global.u64 [%output+40], %control;', 'DONE:', ' ret;', '}'))
    return '\n'.join(lines) + '\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(command, timeout):
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True,
                             env=dict(os.environ, CUMETAL_TRACE_GPU='1', CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0'))
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate(timeout=5)
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return dict(command=command, seconds=time.monotonic() - started, exit_code=child.returncode,
                timed_out=timed_out, stdout=stdout, stderr=stderr)


def worker(build, case, artifacts):
    from ptx_test_support import run_integer_case
    values = inputs()
    run_integer_case(build, fixture(case), values, expected(case, values), case,
                     entry=ENTRY, input_words=3, output_words=6, abi_lines=ABI, artifacts_dir=artifacts)


def rejection_diagnostic(kind, stderr):
    choices = ['pointer memory proof', 'call write effect proof', 'private helper pointer field proof',
               'pointer call argument', 'operand type', 'incompatible pointer']
    if kind == 'unknown-call':
        choices.extend(('has no typed PTX definition', 'unknown direct callee'))
    elif kind == 'recursive-call':
        choices.extend(('recursive PTX device-call', 'recursive call'))
    elif kind in ('missing-argument', 'guarded-argument', 'ambiguous-argument'):
        choices.extend(('call parameter slot', 'call argument', 'parameter staging',
                        'predicated st.param', 'guarded st.param', 'ambiguous st.param'))
    return any(text in stderr for text in choices)


def suite(args, evidence):
    compiler = args.build / 'cumetalc'
    selected = (args.case,) if args.case else CASES
    records = []
    for case, negative in [(case, None) for case in selected] + [('same-allocation', name) for name in NEGATIVES]:
        name = 'reject-' + negative if negative else case
        stage = 'rejection' if negative else args.stage
        folder = evidence / name
        folder.mkdir()
        ptx, msl = folder / 'test.ptx', folder / 'test.metal'
        source = fixture(case, negative)
        ptx.write_text(source)
        command = ([sys.executable, str(Path(__file__).resolve()), str(args.build), '--case', case,
                    '--gpu-child', '--artifacts', str(folder)] if stage == 'numerical' else
                   [str(compiler), str(ptx), '--backend=cumetal-ir', '--ptx-strict', '--entry', ENTRY,
                    '--emit=msl', '-o', str(msl)])
        measured = process(command, 90 if stage == 'numerical' else 45)
        launches = [line for line in measured['stderr'].splitlines() if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        abi_path = Path(str(msl) + '.cumetal-abi')
        good = not measured['timed_out'] and ptx.read_text() == source
        if negative:
            good = (good and measured['exit_code'] not in (0, None) and rejection_diagnostic(negative, measured['stderr']) and
                    not launches and not msl.exists() and not abi_path.exists())
        else:
            good = (good and measured['exit_code'] == 0 and msl.is_file() and abi_path.is_file() and
                    abi_path.read_text().splitlines() == ABI)
            if stage == 'numerical':
                good = good and f'NUMERICAL_PASS {case}: 65 inputs, output values, guards' in measured['stdout'].splitlines()
                good = good and len(launches) == 1 and all(token in launches[0] for token in (
                    'kernel="' + ENTRY + '"', 'device=apple_gpu ', 'launch_success=true ',
                    'source=generic_ptx ', 'provenance=generic_ptx_lowering ', 'semantic_quality=exact '))
            else:
                good = good and not launches
        values = inputs()
        record = dict(case=name, passed=bool(good), stage=stage, ptx_sha256=digest(ptx),
                      msl_sha256=digest(msl) if msl.is_file() else None,
                      inputs=None if negative else values, expected=None if negative else expected(case, values),
                      process=measured)
        records.append(record)
        (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print('PASS' if good else 'FAIL', stage, name, flush=True)
        if not good:
            print(measured['stderr'], end='', file=sys.stderr)
    report = dict(schema=1, stage=args.stage, compiler_sha256=digest(compiler),
                  runtime_sha256=digest(args.build / 'libcumetal.dylib') if args.stage == 'numerical' else None,
                  script_sha256=digest(Path(__file__).resolve()),
                  helper_sha256=digest(Path(__file__).with_name('ptx_test_support.py')),
                  required_positive_cases=len(selected), full_positive_denominator=len(CASES),
                  required_rejections=len(NEGATIVES),
                  counts=dict(Counter(('pass' if row['passed'] else 'fail') + '-' + row['stage'] for row in records)),
                  passed=all(row['passed'] for row in records), cases=records)
    (evidence / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


def main():
    if not __debug__:
        raise RuntimeError('This test requires Python assertions for ABI and guard checks')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--stage', choices=('numerical', 'translate'), default='numerical')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--output', type=Path, help='new or empty evidence directory')
    parser.add_argument('--gpu-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--artifacts', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.build = args.build.resolve()
    if args.gpu_child:
        if args.case is None:
            parser.error('--gpu-child requires --case')
        worker(args.build, args.case, args.artifacts)
        return 0
    if not (args.build / 'cumetalc').is_file():
        parser.error('build directory must contain cumetalc')
    if args.stage == 'numerical' and not (args.build / 'libcumetal.dylib').is_file():
        parser.error('numerical stage requires libcumetal.dylib')
    if args.output:
        evidence = args.output.resolve()
        evidence.mkdir(parents=True, exist_ok=True)
        if any(evidence.iterdir()):
            parser.error('--output must be a new or empty directory')
        return suite(args, evidence)
    with tempfile.TemporaryDirectory(prefix='cumetal-call-write-footprints-') as work:
        return suite(args, Path(work))


if __name__ == '__main__':
    raise SystemExit(main())
