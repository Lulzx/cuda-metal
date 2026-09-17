#!/usr/bin/env python3
"""Bounded pointer fields retain their address space across helper calls.

The retained issue-140 fixture and focused controls run 65 lanes with exact CPU
references, ABI checks, buffer guards and Apple-GPU provenance. Ambiguous memory
contents are translation-only refusals; --stage translate never loads a module.
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
ALTERNATE_XOR = 0xFEDCBA9876543210
SCALAR_XOR = 0xA55A96695AA56996
CONSTANTS = (0, 0x123456789ABCDEF0, 0x8000000000000001, MASK)
ENTRY = 'helper_record_fields'
CASES = (
    'retained-private-field', 'private-field', 'device-field', 'constant-field',
    'copied-offset-record', 'two-private-records', 'two-device-records',
    'direct-private-pointer', 'inline-private-field', 'zero-length-private-field',
    'ordered-helper-scratch', 'commuted-helper-scratch',
)
NEGATIVES = ('partial-overwrite', 'unknown-alias', 'missing-field',
             'conflicting-field', 'cross-space-calls')

# Byte-for-byte copy of issue-130-zero-guards/solana-numerical/nix-gpu/
# private-field/fixture.ptx. Its original failure is tracked by issue 140.
RETAINED_SHA256 = '7b3b4582f8f8a95fce381caeb3a8954573e508cc0166de3d3a01a1abdb458073'
RETAINED_PTX = '''.version 7.1
.target sm_80
.address_size 64

.func (.param .b32 result) read_record(.param .b64 request) {
 .reg .b64 %rd<5>;
 .reg .b32 %r0;
 .reg .pred %p0;
 ld.param.b64 %rd0, [request];
 cvta.to.local.u64 %rd1, %rd0;
 ld.local.b64 %rd2, [%rd1];
 ld.local.b64 %rd3, [%rd1+8];
 mov.u32 %r0, 0;
 setp.eq.u64 %p0, %rd3, 0;
 @%p0 bra DONE;
 ld.b8 %r0, [%rd2];
DONE:
 st.param.b32 [result], %r0;
 ret;
}

.visible .entry pointer_record_probe(.param .u64 .ptr .global input,
                                    .param .u64 .ptr .global output, .param .u32 count) {
 .local .align 16 .b8 depot[32];
 .reg .b64 %rd<10>;
 .reg .b32 %r<6>;
 .reg .pred %guard;
 .param .b64 argument;
 .param .b32 answer;


 mov.u32 %r1, %ctaid.x;
 mov.u32 %r2, %ntid.x;
 mov.u32 %r3, %tid.x;
 mad.lo.u32 %r1, %r1, %r2, %r3;
 ld.param.u32 %r4, [count];
 setp.ge.u32 %guard, %r1, %r4;
 @%guard bra RETURN;
 mul.wide.u32 %rd6, %r1, 4;
 ld.param.u64 %rd7, [input];
 add.u64 %rd7, %rd7, %rd6;
 ld.global.u32 %r0, [%rd7];
 mov.b64 %rd0, depot;
 cvta.local.u64 %rd1, %rd0;
 add.u64 %rd2, %rd1, 16;
 st.local.b8 [%rd0+16], %r0;
 st.local.b64 [%rd0], %rd2;
 st.local.b64 [%rd0+8], 1;
 st.param.b64 [argument], %rd1;
 call.uni (answer), read_record, (argument);
 ld.param.b32 %r0, [answer];

 ld.param.b64 %rd4, [output];
 add.u64 %rd4, %rd4, %rd6;
 st.global.b32 [%rd4], %r0;
RETURN:
 ret;
}
'''


def entry(case):
    return 'pointer_record_probe' if case == CASES[0] else ENTRY


def abi(case):
    return ['CUMETAL_ABI_V2', 'kernel ' + entry(case), 'shared 0',
            'arg buffer 8', 'arg buffer 8', 'arg bytes 4']


def inputs(case):
    if case == CASES[0]:
        return [97, 0, 1, 127, 128, 254, 255] + [(19 + 73 * i) & 255 for i in range(58)]
    boundaries = (0, 1, 255, 256, (1 << 32) - 1, 1 << 32,
                  (1 << 63) - 1, 1 << 63, MASK - 1, MASK)
    return [word for lane in range(65) for word in (
        boundaries[lane] if lane < len(boundaries) else (lane * 0x9E3779B97F4A7C15) & MASK,
        (0x6B8B4567327B23C6 ^ (lane + 17) * 0x85EBCA6B27D4EB2F) & MASK,
        boundaries[lane % len(boundaries)])]


def expected(case, values):
    if case == CASES[0]:
        return list(values)
    result = []
    for a, b, length in zip(values[::3], values[1::3], values[2::3]):
        first = (a if case in ('device-field', 'two-device-records') else
                 CONSTANTS[length & 3] if case == 'constant-field' else a ^ PRIVATE_XOR)
        if length == 0 or case == 'zero-length-private-field':
            first = 0
        if case in ('ordered-helper-scratch', 'commuted-helper-scratch'):
            first ^= 90
        second = b ^ ALTERNATE_XOR if case == 'two-private-records' else b
        # These controls are computed from immutable inputs, independently of
        # the helper's field reloads and returned value.
        result.extend((first, second, length ^ SCALAR_XOR, a))
    return result


def fixture(case, negative=None):
    if case == CASES[0]:
        assert negative is None
        assert hashlib.sha256(RETAINED_PTX.encode()).hexdigest() == RETAINED_SHA256
        return RETAINED_PTX
    if case not in CASES or negative is not None and negative not in NEGATIVES:
        raise ValueError('unknown helper record fixture')
    direct = case == 'direct-private-pointer'
    inline = case == 'inline-private-field'
    two = case.startswith('two-') or negative == 'cross-space-calls'
    kind = 'device' if case in ('device-field', 'two-device-records') else 'constant' if case == 'constant-field' else 'private'
    helper_loads = (' ld.param.b64 %payload, [request];\n ld.param.b64 %length, [length];' if direct else
                    ' ld.param.b64 %raw, [request];\n cvta.to.local.u64 %record, %raw;\n'
                    ' ld.local.b64 %payload, [%record+8];\n ld.local.b64 %length, [%record+16];')
    helper = '' if inline else f'''.func (.param .b64 result) read_record(
 .param .b64 request{', .param .b64 length' if direct else ''}) {{
 .reg .b64 %raw, %record, %payload, %length, %value;
 .reg .pred %empty;
{helper_loads}
 mov.u64 %value, 0;
 setp.eq.u64 %empty, %length, 0;
 @%empty bra HELPER_DONE;
 ld.u64 %value, [%payload];
HELPER_DONE:
 st.param.b64 [result], %value;
 ret;
}}
'''
    if case in ('ordered-helper-scratch', 'commuted-helper-scratch'):
        helper = helper.replace(' .reg .pred %empty;', ''' .reg .pred %empty;
 .local .align 16 .b8 scratch[16];
 .reg .b64 %scratch_base, %scratch_offset, %scratch_address, %scratch_value;''')
        operands = ('%scratch_offset, %scratch_base' if case == 'commuted-helper-scratch'
                    else '%scratch_base, %scratch_offset')
        # Same helper-owned array and byte index, with both legal add orders.
        # The store is observed in the return value and cannot be dropped.
        helper = helper.replace(' ld.local.b64 %payload, [%record+8];', f''' ld.local.b64 %length, [%record+16];
 mov.u64 %scratch_base, scratch;
 and.b64 %scratch_offset, %length, 7;
 add.u64 %scratch_address, {operands};
 st.local.u8 [%scratch_address], 90;
 ld.local.u8 %scratch_value, [%scratch_address];
 ld.local.b64 %payload, [%record+8];''')
        helper = helper.replace('HELPER_DONE:\n', 'HELPER_DONE:\n xor.b64 %value, %value, %scratch_value;\n')
    constant_bytes = ','.join(str(byte) for value in CONSTANTS for byte in value.to_bytes(8, 'little'))
    lines = ['.version 7.1', '.target sm_80', '.address_size 64',
             f'.const .align 8 .b8 table[32] = {{{constant_bytes}}};', helper,
             f'.visible .entry {ENTRY}(.param .u64 .ptr .global input,',
             ' .param .u64 .ptr .global output, .param .u32 count) {',
             ' .local .align 16 .b8 record_a[64];',
             ' .local .align 16 .b8 record_b[32];',
             ' .local .align 16 .b8 payloads[16];',
             ' .reg .b64 %input, %output, %offset, %a, %b, %length, %control;',
             ' .reg .b64 %base, %record, %copy, %other_record, %other_copy;',
             ' .reg .b64 %private, %private_alt, %private_value, %private_alt_value;',
             ' .reg .b64 %device_alt, %table, %selection, %constant, %unknown;',
             ' .reg .b64 %answer, %second, %loaded, %loaded_length;',
             ' .reg .b32 %lane, %block, %threads, %count, %parity, %byte;',
             ' .reg .pred %done, %choice, %empty;',
             ' .param .b64 argument;', ' .param .b64 argument_length;',
             ' .param .b64 result;',
             ' ld.param.u64 %input, [input];', ' ld.param.u64 %output, [output];',
             ' ld.param.u32 %count, [count];', ' mov.u32 %lane, %tid.x;',
             ' mov.u32 %block, %ctaid.x;', ' mov.u32 %threads, %ntid.x;',
             ' mad.lo.u32 %lane, %block, %threads, %lane;',
             ' setp.ge.u32 %done, %lane, %count;', ' @%done bra DONE;',
             ' mul.wide.u32 %offset, %lane, 24;', ' add.u64 %input, %input, %offset;',
             ' mul.wide.u32 %offset, %lane, 32;', ' add.u64 %output, %output, %offset;',
             ' ld.global.u64 %a, [%input];', ' ld.global.u64 %b, [%input+8];',
             ' ld.global.u64 %length, [%input+16];', ' cvt.u32.u64 %byte, %a;',
             ' and.b32 %parity, %lane, 1;', ' setp.eq.u32 %choice, %parity, 0;',
             ' mov.u64 %base, record_a;',
             (' add.u64 %record, %base, 16;' if case == 'copied-offset-record' else ' mov.b64 %record, %base;'),
             ' cvta.local.u64 %copy, %record;',
             ' mov.u64 %other_record, record_b;', ' cvta.local.u64 %other_copy, %other_record;',
             ' mov.u64 %private, payloads;', ' add.u64 %private_alt, %private, 8;',
             f' xor.b64 %private_value, %a, {PRIVATE_XOR};',
             f' xor.b64 %private_alt_value, %b, {ALTERNATE_XOR};',
             ' st.local.u64 [%private], %private_value;', ' st.local.u64 [%private_alt], %private_alt_value;',
             ' add.u64 %device_alt, %input, 8;', ' mov.u64 %table, table;',
             ' and.b64 %selection, %length, 3;', ' shl.b64 %selection, %selection, 3;',
             ' add.u64 %constant, %table, %selection;',
             ' st.local.u64 [%record], %b;',
             ' st.local.u64 [%record+16], ' + ('0;' if case == 'zero-length-private-field' else '%length;'),
             ' mov.b64 %second, %b;']
    pointer = {'private': '%private', 'device': '%input', 'constant': '%constant'}[kind]
    if negative != 'missing-field':
        lines.append(f' st.local.b64 [%record+8], {pointer};')
    if negative == 'partial-overwrite':
        lines.append(' st.local.u32 [%record+12], %byte;')
    elif negative == 'unknown-alias':
        lines.extend((' cvt.u64.u32 %unknown, %lane;', ' and.b64 %unknown, %unknown, 7;',
                      ' add.u64 %unknown, %unknown, 8;', ' add.u64 %unknown, %copy, %unknown;',
                      ' st.local.u8 [%unknown], %byte;'))
    elif negative == 'conflicting-field':
        lines.extend((' @%choice bra CALL;', ' st.local.b64 [%record+8], %input;', 'CALL:'))
    if two:
        second_pointer = '%private_alt' if kind == 'private' and negative != 'cross-space-calls' else '%device_alt'
        lines.extend((' st.local.u64 [%other_record], %a;',
                      f' st.local.b64 [%other_record+8], {second_pointer};',
                      ' st.local.u64 [%other_record+16], 1;'))
    if inline:
        lines.extend((' ld.local.b64 %loaded, [%copy+8];',
                      ' ld.local.b64 %loaded_length, [%copy+16];',
                      ' mov.u64 %answer, 0;', ' setp.eq.u64 %empty, %loaded_length, 0;',
                      ' @%empty bra STORE;', ' ld.u64 %answer, [%loaded];', 'STORE:'))
    else:
        lines.append(' st.param.b64 [argument], ' + ('%private;' if direct else '%copy;'))
        if direct:
            lines.append(' st.param.b64 [argument_length], %length;')
        lines.extend((' call.uni (result), read_record, (argument' + (', argument_length);' if direct else ');'),
                      ' ld.param.b64 %answer, [result];'))
        if two:
            lines.extend((' st.param.b64 [argument], %other_copy;',
                          ' call.uni (result), read_record, (argument);',
                          ' ld.param.b64 %second, [result];'))
    lines.extend((' st.global.u64 [%output], %answer;', ' st.global.u64 [%output+8], %second;',
                  f' xor.b64 %control, %length, {SCALAR_XOR};',
                  ' st.global.u64 [%output+16], %control;', ' st.global.u64 [%output+24], %a;',
                  'DONE:', ' ret;', '}'))
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
    values = inputs(case)
    retained = case == CASES[0]
    run_integer_case(build, fixture(case), values, expected(case, values), case,
                     entry=entry(case), word_bits=32 if retained else 64,
                     input_words=1 if retained else 3, output_words=1 if retained else 4,
                     abi_lines=abi(case), artifacts_dir=artifacts)


def suite(args, evidence):
    compiler = args.build / 'cumetalc'
    selected = (args.case,) if args.case else CASES
    records = []
    for case, negative in [(case, None) for case in selected] + [('private-field', name) for name in NEGATIVES]:
        name = 'reject-' + negative if negative else case
        stage = 'rejection' if negative else args.stage
        folder = evidence / name
        folder.mkdir()
        ptx, msl = folder / 'test.ptx', folder / 'test.metal'
        source = fixture(case, negative)
        ptx.write_text(source)
        command = ([sys.executable, str(Path(__file__).resolve()), str(args.build), '--case', case,
                    '--gpu-child', '--artifacts', str(folder)] if stage == 'numerical' else
                   [str(compiler), str(ptx), '--backend=cumetal-ir', '--ptx-strict', '--entry', entry(case),
                    '--emit=msl', '-o', str(msl)])
        measured = process(command, 90 if stage == 'numerical' else 45)
        launches = [line for line in measured['stderr'].splitlines() if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        abi_path = Path(str(msl) + '.cumetal-abi')
        good = not measured['timed_out'] and ptx.read_text() == source
        if negative:
            diagnostic = any(text in measured['stderr'] for text in (
                'private helper pointer field proof', 'pointer memory proof',
                'operand type', 'incompatible pointer', 'conflicting pointer'))
            good = (good and measured['exit_code'] not in (0, None) and diagnostic and
                    not launches and not msl.exists() and not abi_path.exists())
        else:
            good = (good and measured['exit_code'] == 0 and msl.is_file() and abi_path.is_file() and
                    abi_path.read_text().splitlines() == abi(case))
            if stage == 'numerical':
                good = good and f'NUMERICAL_PASS {case}: 65 inputs, output values, guards' in measured['stdout'].splitlines()
                good = good and len(launches) == 1 and all(token in launches[0] for token in (
                    'kernel="' + entry(case) + '"', 'device=apple_gpu ', 'launch_success=true ',
                    'source=generic_ptx ', 'provenance=generic_ptx_lowering ', 'semantic_quality=exact '))
            else:
                good = good and not launches
        values = inputs(case)
        record = dict(case=name, passed=bool(good), stage=stage, ptx_sha256=digest(ptx),
                      msl_sha256=digest(msl) if msl.is_file() else None,
                      inputs=None if negative else values,
                      expected=None if negative else expected(case, values),
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
                  retained_fixture_sha256=RETAINED_SHA256,
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
    with tempfile.TemporaryDirectory(prefix='cumetal-helper-record-fields-') as work:
        return suite(args, Path(work))


if __name__ == '__main__':
    raise SystemExit(main())
