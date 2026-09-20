#!/usr/bin/env python3
"""Aligned local-address ORs preserve byte loads and stores on the Apple GPU."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


CASES = ('direct', 'copy', 'selected', 'join', 'loop')
MASK = (1 << 64) - 1


def case_source(case):
    source = (Path(__file__).parent / 'reference/ptx_address_alignment.ptx').read_text()
    definition = '    mov.u64 %rd7, %rd6;'
    if case == 'direct':
        return source
    definitions = {
        'copy': '''    add.u64 %rd8, %rd6, 16;
    mov.u64 %rd9, %rd8;
    mov.u64 %rd7, %rd9;''',
        'selected': '''    add.u64 %rd8, %rd6, 16;
    add.u64 %rd9, %rd6, 48;
    and.b64 %rd14, %rd5, 1;
    setp.eq.u64 %p2, %rd14, 0;
    selp.u64 %rd7, %rd8, %rd9, %p2;''',
        'join': '''    and.b64 %rd14, %rd5, 1;
    setp.eq.u64 %p2, %rd14, 0;
    @%p2 bra even;
    add.u64 %rd7, %rd6, 64;
    bra joined;
even:
    add.u64 %rd7, %rd6, 32;
joined:''',
        'loop': '''    mov.u64 %rd7, %rd6;
    mov.u32 %r8, 0;
    cvt.u32.u64 %r9, %rd5;
    setp.eq.u32 %p2, %r9, 0;
    @%p2 bra advanced;
advance:
    or.b64 %rd7, %rd7, 1;
    add.u64 %rd7, %rd7, 15;
    add.u32 %r8, %r8, 1;
    setp.lt.u32 %p3, %r8, %r9;
    @%p3 bra advance;
advanced:''',
    }
    if source.count(definition) != 1:
        raise AssertionError('alignment fixture must contain exactly one address definition')
    return source.replace(definition, definitions[case])


def case_inputs_and_expected(case):
    seeds = (0, 1, 127, 128, 255, 256, (1 << 32) - 1,
             1 << 32, (1 << 63) - 1, 1 << 63, MASK - 1, MASK)
    pairs = [(seeds[lane % len(seeds)], lane % 8) for lane in range(65)]
    expected = []
    for seed, selector in pairs:
        offset = {
            'direct': 0,
            'copy': 16,
            'selected': 16 if selector % 2 == 0 else 48,
            'join': 32 if selector % 2 == 0 else 64,
            'loop': 16 * selector,
        }[case]
        byte1 = (seed + 17 * (offset + 1)) & 255
        byte4 = (seed + 17 * (offset + 4)) & 255
        expected.extend((byte1, byte1, byte4, byte4,
                         (seed ^ 0x5a) & 255, (seed ^ 0xc3) & 255,
                         (seed + 0x71) & 255, (seed + 0xb6) & 255))
    return [value for pair in pairs for value in pair], expected


def run_case(build, case):
    from ptx_test_support import run_integer_case

    values, expected = case_inputs_and_expected(case)
    run_integer_case(build, case_source(case), values, expected,
                     'aligned local address OR: ' + case,
                     entry='alignment_probe', input_words=2, output_words=8,
                     abi_lines=['CUMETAL_ABI_V2', 'kernel alignment_probe', 'shared 0',
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
            # Stop only this child's isolated process group, including compilers.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        print('CASE ' + case, flush=True)
        print(stdout, end='')
        print(stderr, end='', file=sys.stderr)
        launches = [line for line in stderr.splitlines()
                    if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        numerical = f'NUMERICAL_PASS aligned local address OR: {case}: 65 inputs, output values, guards'
        if timed_out or child.returncode or numerical not in stdout.splitlines() or len(launches) != 1 or not all(
            token in launches[0] for token in ('device=apple_gpu', 'launch_success=true',
                                              'provenance=generic_ptx_lowering')):
            failed.append(case)
    if failed:
        print('FAILED cases: ' + ', '.join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
