#!/usr/bin/env python3
"""Same-base cancellation computes modular scalar counts on the Apple GPU."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


CASES = ('straight', 'commuted', 'negated', 'selected', 'loop', 'branch',
         'scalar-overwrite', 'scalar-control')
MASK = (1 << 64) - 1


def run_case(build, case):
    from ptx_test_support import run_integer_case

    source = (Path(__file__).parent / 'reference/ptx_address_cancellation.ptx').read_text()
    boundary_lengths = [0, 1, 30, 31, 32, 63, (1 << 63) - 1, 1 << 63, MASK - 1, MASK]
    pairs = [(lane % 32, boundary_lengths[lane % len(boundary_lengths)])
             for lane in range(65)]
    if case in ('straight', 'commuted', 'negated', 'selected',
                'scalar-overwrite', 'scalar-control'):
        # These forms never dereference the cursor. Its cancelled arithmetic
        # must preserve all 64 bits, including wraparound in the byte offset.
        boundary_offsets = [0, 1, 31, 32, (1 << 63) - 1, 1 << 63, MASK - 1, MASK]
        pairs = [(boundary_offsets[lane % len(boundary_offsets)], length)
                 for lane, (_, length) in enumerate(pairs)]
    offsets = [offset for offset, _ in pairs]
    if case == 'commuted':
        source = source.replace('add.u64 %rd7, %rd6, %rd4;', 'add.u64 %rd7, %rd4, %rd6;')
        source = source.replace('add.s64 %rd11, %rd6, %rd10;', 'add.s64 %rd11, %rd10, %rd6;')
    elif case == 'negated':
        source = source.replace('sub.s64 %rd9, %rd8, %rd7;', '''
    neg.s64 %rd9, %rd7;
    add.s64 %rd9, %rd9, %rd8;
''')
    elif case == 'selected':
        source = source.replace('add.u64 %rd7, %rd6, %rd4;', '''
    add.u64 %rd7, %rd6, %rd4;
    add.u64 %rd16, %rd7, 3;
    and.b32 %r4, %r1, 1;
    setp.eq.u32 %p1, %r4, 0;
    selp.u64 %rd7, %rd16, %rd7, %p1;
''')
        offsets = [offset + (3 if lane % 2 == 0 else 0)
                   for lane, (offset, _) in enumerate(pairs)]
    elif case == 'loop':
        source = source.replace('add.u64 %rd7, %rd6, %rd4;', '''
    mov.u64 %rd7, %rd6;
    mov.u64 %rd15, 0;
    setp.eq.u64 %p1, %rd4, 0;
    @%p1 bra filled;
fill:
    st.local.u8 [%rd7], 42;
    add.u64 %rd7, %rd7, 1;
    add.u64 %rd15, %rd15, 1;
    setp.lt.u64 %p2, %rd15, %rd4;
    @%p2 bra fill;
filled:
''')
    elif case == 'branch':
        source = source.replace('add.u64 %rd7, %rd6, %rd4;', '''
    and.b32 %r4, %r1, 1;
    setp.eq.u32 %p1, %r4, 0;
    @%p1 bra even;
    add.u64 %rd7, %rd6, %rd4;
    bra joined;
even:
    add.u64 %rd16, %rd4, 3;
    add.u64 %rd7, %rd6, %rd16;
joined:
''')
        offsets = [offset + (3 if lane % 2 == 0 else 0)
                   for lane, (offset, _) in enumerate(pairs)]
    elif case == 'scalar-overwrite':
        source = source.replace('mov.u64 %rd8, 1;', 'mov.u64 %rd7, %rd4;\n    mov.u64 %rd8, 1;')
        source = source.replace('add.s64 %rd11, %rd6, %rd10;', 'mov.u64 %rd11, %rd10;')
    elif case == 'scalar-control':
        begin = source.index('    mov.u64 %rd8, 1;')
        end = source.index('    mul.wide.u32 %rd13', begin)
        source = source[:begin] + '''    sub.u64 %rd10, 31, %rd4;
    sub.u64 %rd12, %rd10, %rd5;
''' + source[end:]
    expected = [(31 - offset - length) & MASK
                for offset, (_, length) in zip(offsets, pairs)]
    run_integer_case(build, source, [value for pair in pairs for value in pair],
                     expected, 'same-base cancellation: ' + case,
                     entry='cancellation_probe', input_words=2, output_words=1,
                     abi_lines=['CUMETAL_ABI_V2', 'kernel cancellation_probe', 'shared 0',
                                'arg buffer 8', 'arg buffer 8', 'arg bytes 4'])


def main():
    if not __debug__:
        raise RuntimeError('This test requires Python assertions for ABI and guard checks')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--gpu-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu_child:
        if args.case is None:
            parser.error('--gpu-child requires --case')
        run_case(args.build.resolve(), args.case)
        return 0
    failed = []
    for case in (args.case,) if args.case else CASES:
        child = subprocess.Popen(
            [sys.executable, __file__, str(args.build.resolve()), '--case', case, '--gpu-child'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = child.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            stdout, stderr = child.communicate(timeout=5)
        finally:
            # A timed-out compiler child belongs to this same isolated group.
            # Never kill shared Apple compiler services or other test runners.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        print('CASE ' + case, flush=True)
        print(stdout, end='')
        print(stderr, end='', file=sys.stderr)
        launches = [line for line in stderr.splitlines()
                    if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        if timed_out or child.returncode or len(launches) != 1 or not all(
            token in launches[0] for token in ('device=apple_gpu', 'launch_success=true',
                                              'provenance=generic_ptx_lowering')):
            failed.append(case)
    if failed:
        print('FAILED cases: ' + ', '.join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
