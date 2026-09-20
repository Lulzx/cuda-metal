#!/usr/bin/env python3
"""Empty local slices preserve sentinel bits while their dereference is skipped.

Thirteen 65-lane configurations use exact CPU references, ABI checks, guarded buffers
and Apple-GPU provenance. Hazardous controls only translate and must refuse.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


MASK = (1 << 64) - 1
ENTRY = 'empty_slice_guards'
CASES = ('direct-eq0', 'min8-eq0', 'minunknown-lt64', 'concrete-empty', 'pruned-loop', 'bounded-copy', 'captured-offset', 'aligned-base', 'zero-offset-call',
         'unsigned-lt-zero', 'unsigned-ge-zero', 'zero-le-unsigned', 'zero-gt-unsigned')
NEGATIVES = ('nonzero-length', 'overwritten-length', 'missing-initializer',
             'predicated-initializer', 'backedge-only-zero', 'partial-write',
             'overlapping-write', 'unknown-helper-clobber', 'unguarded-sentinel',
             'signed-min-negative', 'ranged-overlap', 'ranged-unbounded', 'ranged-out-of-bounds', 'ranged-unaligned-or', 'ranged-high-bit-or', 'staged-overlap')
ABI = ['CUMETAL_ABI_V2', 'kernel ' + ENTRY, 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']


def inputs():
    boundaries = (0, 1, 255, 256, (1 << 32) - 1, 1 << 32,
                  (1 << 63) - 1, 1 << 63, MASK - 1, MASK)
    return [word for lane in range(65) for word in (
        (0x0123456789ABCDEF ^ (lane + 7) * 0x9E3779B97F4A7C15) & MASK,
        boundaries[lane] if lane < len(boundaries) else (lane * 0x0102030405060709) & MASK)]


def expected(case, values):
    # Concrete pointers have process-dependent address bits; that control
    # observes the reloaded zero length instead of publishing the address.
    return [word for value, length in zip(values[::2], values[1::2])
            for word in ((47 if case == 'zero-offset-call' else 47 * min(length, 32) if case == 'captured-offset' else
                          47 * length if case in ('bounded-copy', 'aligned-base') and length <= 32 else 0),
                         0 if case == 'concrete-empty' else 1, value)]


def fixture(case, negative=None):
    if case not in CASES or negative is not None and negative not in NEGATIVES:
        raise ValueError('unknown empty-slice fixture')
    if negative is not None and case != 'direct-eq0':
        raise ValueError('refusals use direct-eq0')
    helper = '''.func clobber(.param .b64 address, .param .b64 offset) {
 .reg .b64 %raw, %pointer, %offset;
 ld.param.b64 %raw, [address];
 cvta.to.local.u64 %pointer, %raw;
 ld.param.b64 %offset, [offset];
 add.u64 %pointer, %pointer, %offset;
 st.local.u8 [%pointer], 1;
 ret;
}
''' if negative == 'unknown-helper-clobber' else ''
    if case == 'zero-offset-call' or negative == 'staged-overlap':
        helper = """.func write_byte(.param .b64 address) {
 .reg .b64 %raw, %pointer;
 ld.param.b64 %raw, [address];
 cvta.to.local.u64 %pointer, %raw;
 st.local.u8 [%pointer], 47;
 ret;
}
"""
    lines = ['.version 7.1', '.target sm_80', '.address_size 64', helper,
             f'.visible .entry {ENTRY}(.param .u64 .ptr .global input,',
             ' .param .u64 .ptr .global output, .param .u32 count) {',
             ' .local .align 16 .b8 record[16];',
             ' .reg .b64 %record, %input, %output, %offset, %data, %unknown;',
             ' .reg .b64 %stored, %zero, %pointer, %length, %extent, %result;',
             ' .reg .b32 %lane, %block, %threads, %count, %value;',
             ' .reg .pred %outside, %empty, %maybe;',
             ' .param .b64 clobber_record;', ' .param .b64 clobber_offset;',
             ' ld.param.u64 %input, [input];', ' ld.param.u64 %output, [output];',
             ' ld.param.u32 %count, [count];', ' mov.u32 %lane, %tid.x;',
             ' mov.u32 %block, %ctaid.x;', ' mov.u32 %threads, %ntid.x;',
             ' mad.lo.u32 %lane, %block, %threads, %lane;',
             ' setp.ge.u32 %outside, %lane, %count;', ' @%outside bra DONE;',
             ' mul.wide.u32 %offset, %lane, 16;', ' add.u64 %input, %input, %offset;',
             ' mul.wide.u32 %offset, %lane, 24;', ' add.u64 %output, %output, %offset;',
             ' ld.global.u64 %data, [%input];', ' ld.global.u64 %unknown, [%input+8];',
             ' mov.u64 %record, record;', ' mov.u32 %value, 0;']
    if negative == 'backedge-only-zero':
        lines.extend((' st.local.u64 [%record], 1;', 'READ_SLICE:'))
    elif negative == 'predicated-initializer':
        lines.extend((' setp.ne.u64 %maybe, %unknown, 0;',
                      ' @%maybe st.local.v2.b64 [%record], {1, 0};'))
    elif negative != 'missing-initializer':
        if case.startswith('min'):
            lines.extend((' mov.u64 %stored, 1;', ' mov.u64 %zero, 0;',
                          ' st.local.v2.u64 [%record], {%stored, %zero};'))
        else:
            payload = '%input' if case == 'concrete-empty' else '1'
            length = '1' if negative == 'nonzero-length' else '0'
            lines.append(f' st.local.v2.b64 [%record], {{{payload}, {length}}};')
    if negative == 'overwritten-length':
        lines.append(' st.local.u64 [%record+8], 1;')
    elif negative == 'partial-write':
        lines.append(' st.local.u8 [%record+15], 1;')
    elif negative == 'overlapping-write':
        lines.append(' st.local.u64 [%record+4], 4294967296;')
    elif negative == 'unknown-helper-clobber':
        lines.extend((' st.param.b64 [clobber_record], %record;',
                      ' st.param.b64 [clobber_offset], %unknown;',
                      ' call.uni clobber, (clobber_record, clobber_offset);'))
    if case in ('bounded-copy', 'captured-offset', 'aligned-base') or negative and negative.startswith('ranged-'):
        # Keep pointer/length at 96/104 and copy bytes in the same allocation.
        lines = [line.replace('record[16]', 'record[192]') for line in lines]
        pos = lines.index(' mov.u64 %record, record;') + 1
        lines.insert(pos, ' add.u64 %record, %record, 96;')
        lines.extend((' .reg .b64 %scratch, %index, %write, %sum_address;',
                      ' .reg .b32 %byte;', ' .reg .pred %limit, %again;',
                      ' mov.u64 %scratch, record;',
                      ' st.local.v4.u64 [%scratch], {0, 0, 0, 0};'))
        if case == 'aligned-base' or negative in ('ranged-unaligned-or', 'ranged-high-bit-or'):
            lines.append(' st.local.u64 [%scratch+32], 0;')
            lines.append(' or.b64 %scratch, %scratch, ' + ('16;' if negative == 'ranged-high-bit-or' else '1;'))
            if negative == 'ranged-unaligned-or':
                lines = [line.replace('.align 16', '.align 1') for line in lines]
        if negative == 'ranged-overlap':
            lines.append(' add.u64 %scratch, %scratch, 104;')
        elif negative == 'ranged-out-of-bounds':
            lines.append(' add.u64 %scratch, %scratch, 184;')
        lines.extend((' setp.gt.u64 %limit, %unknown, 32;',))
        if negative != 'ranged-unbounded' and case != 'captured-offset':
            lines.append(' @%limit bra COPY_DONE;')
        lines.extend((' setp.eq.u64 %limit, %unknown, 0;', ' @%limit bra COPY_DONE;',
                      ' mov.u64 %index, 0;', 'COPY_BYTES:',
                      ' add.u64 %write, %scratch, %index;'))
        if case == 'captured-offset':
            # A later guard can refine a captured offset only while its SSA
            # dependencies still denote the same dynamic values.
            lines.extend((' setp.ge.u64 %limit, %index, 32;', ' @%limit bra COPY_DONE;'))
        lines.extend((' st.local.u8 [%write], 47;', ' add.u64 %index, %index, 1;',
                      ' setp.lt.u64 %again, %index, %unknown;', ' @%again bra COPY_BYTES;',
                      'COPY_DONE:', ' mov.u64 %index, 0;', 'SUM_BYTES:',
                      ' add.u64 %sum_address, %scratch, %index;',
                      ' ld.local.u8 %byte, [%sum_address];', ' add.u32 %value, %value, %byte;',
                      ' add.u64 %index, %index, 1;', ' setp.lt.u64 %again, %index, 32;',
                      ' @%again bra SUM_BYTES;'))
    if case == 'zero-offset-call' or negative == 'staged-overlap':
        lines = [line.replace('record[16]', 'record[32]') for line in lines]
        offset = 8 if negative == 'staged-overlap' else 16
        lines.extend((' .reg .b64 %actual;', f' add.u64 %actual, %record, {offset};',
                      ' st.param.b64 [clobber_record+0], %actual;',
                      ' call.uni write_byte, (clobber_record);',
                      ' ld.local.u8 %value, [%actual];'))
    suffix = 'u64' if case.startswith('min') else 'b64'
    lines.append(f' ld.local.v2.{suffix} {{%pointer, %length}}, [%record];')
    if negative == 'signed-min-negative':
        lines.append(' min.s64 %extent, %length, -1;')
    elif case == 'min8-eq0':
        lines.append(' min.u64 %extent, %length, 8;')
    elif case == 'minunknown-lt64':
        lines.append(' min.u64 %extent, %length, %unknown;')
    else:
        lines.append(' mov.b64 %extent, %length;')
    if case == 'pruned-loop':
        lines.extend((' setp.eq.u64 %empty, %length, 0;', ' @%empty bra AFTER_LOOP;',
                      'CONFLICTING_LOOP:', ' sub.u64 %length, %length, 1;',
                      ' setp.ne.u64 %maybe, %length, 0;', ' @%maybe bra CONFLICTING_LOOP;',
                      'AFTER_LOOP:', ' mov.b64 %extent, %length;'))
    lines.append(' st.global.u64 [%output+8], ' + ('%length;' if case == 'concrete-empty' else '%pointer;'))
    comparisons = {
        'unsigned-lt-zero': 'setp.lt.u64 %empty, %unknown, %extent;',
        'unsigned-ge-zero': 'setp.ge.u64 %empty, %unknown, %extent;',
        'zero-le-unsigned': 'setp.le.u64 %empty, %extent, %unknown;',
        'zero-gt-unsigned': 'setp.gt.u64 %empty, %extent, %unknown;',
    }
    lines.append(' ' + comparisons[case] if case in comparisons else
                 ' setp.lt.u64 %empty, %extent, 64;' if case == 'minunknown-lt64' else
                 ' setp.eq.u64 %empty, %extent, 0;')
    if negative != 'unguarded-sentinel':
        lines.append(' @!%empty bra EMPTY;' if case in ('unsigned-lt-zero', 'zero-gt-unsigned') else
                     ' @%empty bra EMPTY;')
    lines.append(' ld.u8 %value, [%pointer];')
    if negative == 'backedge-only-zero':
        lines.extend((' st.local.u64 [%record+8], 0;', ' bra READ_SLICE;'))
    lines.extend(('EMPTY:', ' cvt.u64.u32 %result, %value;', ' st.global.u64 [%output], %result;',
                  ' st.global.u64 [%output+16], %data;', 'DONE:', ' ret;', '}'))
    return '\n'.join(lines) + '\n'


def worker(build, case, artifacts):
    from ptx_test_support import run_integer_case
    values = inputs()
    run_integer_case(build, fixture(case), values, expected(case, values), case,
                     entry=ENTRY, input_words=2, output_words=3, abi_lines=ABI, artifacts_dir=artifacts)


def process(command, timeout):
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
    return dict(command=command, exit_code=child.returncode, timed_out=timed_out, stdout=stdout, stderr=stderr)


def suite(args, evidence):
    selected = (args.case,) if args.case else CASES
    records = []
    for case, negative in [(case, None) for case in selected] + [('direct-eq0', name) for name in NEGATIVES]:
        name = 'reject-' + negative if negative else case
        stage = 'rejection' if negative else args.stage
        folder = evidence / name
        folder.mkdir()
        ptx, msl = folder / 'test.ptx', folder / 'test.metal'
        source = fixture(case, negative)
        ptx.write_text(source)
        command = ([sys.executable, str(Path(__file__).resolve()), str(args.build), '--case', case,
                    '--gpu-child', '--artifacts', str(folder)] if stage == 'numerical' else
                   [str(args.build / 'cumetalc'), str(ptx), '--backend=cumetal-ir', '--ptx-strict',
                    '--entry', ENTRY, '--emit=msl', '-o', str(msl)])
        measured = process(command, 90 if stage == 'numerical' else 45)
        launches = [line for line in measured['stderr'].splitlines() if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        abi_path = Path(str(msl) + '.cumetal-abi')
        good = not measured['timed_out'] and ptx.read_text() == source
        if negative:
            diagnostic = any(text in measured['stderr'] for text in (
                'pointer memory proof', 'pointer field proof', 'operand type', 'incompatible pointer'))
            good = (good and measured['exit_code'] not in (0, None) and diagnostic and
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
        record = dict(case=name, stage=stage, passed=bool(good),
                      ptx_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      expected=None if negative else expected(case, inputs()), process=measured)
        records.append(record)
        (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print('PASS' if good else 'FAIL', stage, name, flush=True)
        if not good:
            print(measured['stderr'], end='', file=sys.stderr)
    report = dict(stage=args.stage, full_positive_denominator=len(CASES),
                  required_positive_cases=len(selected), required_rejections=len(NEGATIVES),
                  passed=all(record['passed'] for record in records), cases=records)
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
    with tempfile.TemporaryDirectory(prefix='cumetal-empty-slice-guards-') as work:
        return suite(args, Path(work))


if __name__ == '__main__':
    raise SystemExit(main())
