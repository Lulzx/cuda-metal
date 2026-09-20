#!/usr/bin/env python3
"""Device-pointer fields retain their space for explicit and generic loads."""
import argparse
from pathlib import Path
import subprocess
import sys


GPU_CASES = ('explicit-global-u64', 'generic-u64', 'generic-u8', 'generic-store-u64')
NEGATIVE_CASES = ('reject-reused', 'reject-predicated', 'reject-narrow')
CASES = GPU_CASES + NEGATIVE_CASES


def run_case(build, case):
    from ptx_test_support import expect_compile_failure, run_integer_case

    source = (Path(__file__).parent / 'reference/ptx_pointer_record_fields.ptx').read_text()
    if case in ('explicit-global-u64', 'generic-store-u64'):
        values = [value * 0x0101010101010101 for value in range(65)]
        expected = [values[lane if lane % 2 == 0 else lane + 1] for lane in range(65)]
        label = 'explicit global loads through private record fields'
        if case == 'generic-store-u64':
            source = source.replace('st.global.u64 [%rd3], %rd4;',
                                    'st.u64 [%rd3], %rd4;')
            label = 'generic store through private record field with implicit entry ABI'
        run_integer_case(build, source, values, expected,
                         label, output_words=1,
                         # The legacy unannotated entry transports exact device
                         # address bits through scalar arguments. Keep its ABI;
                         # the explicit-pointer controls below use buffers.
                         abi_lines=['CUMETAL_ABI_V2', 'kernel integer_probe', 'shared 0',
                                    'arg bytes 8', 'arg bytes 8', 'arg bytes 4'])
    elif case in ('generic-u64', 'generic-u8'):
        source = (Path(__file__).parent /
                  'reference/ptx_generic_pointer_record_fields.ptx').read_text()
        # Each lane owns two input words: both branches stay inside the payload,
        # including lane 64. Nonuniform bytes distinguish byte-load offsets.
        mask = (1 << 64) - 1
        pairs = [((0xfedcba9876543210 + lane * 0x01020304050607) & mask,
                  (0x89abcdef01234567 ^ (lane * 0x070503010b0d0f11)) & mask)
                 for lane in range(65)]
        selected = [pair[lane % 2] for lane, pair in enumerate(pairs)]
        if case == 'generic-u8':
            source = source.replace(
                'ld.u64 %rd4, [%rd2+8];',
                'ld.u8 %r0, [%rd2+11];\n    cvt.u64.u32 %rd4, %r0;').replace(
                'ld.u64 %rd4, [%rd2];',
                'ld.u8 %r0, [%rd2+3];\n    cvt.u64.u32 %rd4, %r0;')
            selected = [(value >> 24) & 0xff for value in selected]
        expected = [(value + 0x0102030405060708 + lane * 17 + 5) & mask
                    for lane, value in enumerate(selected)]
        run_integer_case(build, source, [word for pair in pairs for word in pair],
                         expected, case + ' through offset private record with scalar neighbors',
                         entry='offset_record_probe', input_words=2, output_words=1,
                         abi_lines=['CUMETAL_ABI_V2', 'kernel offset_record_probe', 'shared 0',
                                    'arg buffer 8', 'arg buffer 8', 'arg bytes 4'])
    else:
        definitions = {
            'reject-reused': 'ld.local.u64 %rd2, [%rd1];\n    mov.u64 %rd2, 0;',
            'reject-predicated': 'mov.pred %p0, 1;\n    @%p0 ld.local.u64 %rd2, [%rd1];',
            'reject-narrow': 'ld.local.u32 %rd2, [%rd1];',
        }
        invalid = source.replace('ld.local.u64 %rd2, [%rd1];', definitions[case])
        expect_compile_failure(build, invalid, 'integer_probe', 'does not match')
        print('REJECTED ' + case.removeprefix('reject-') + ' pointer-field definition')


def main():
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

    # Isolate every case so one Metal compile failure does not prevent coverage
    # of the other load width or of the unchanged explicit-global control.
    failed = []
    for case in (args.case,) if args.case else CASES:
        result = subprocess.run(
            [sys.executable, __file__, str(args.build.resolve()), '--case', case, '--gpu-child'],
            capture_output=True, text=True)
        print(f'CASE {case}', flush=True)
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
        launches = [line for line in result.stderr.splitlines()
                    if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        expected_launches = 1 if case in GPU_CASES else 0
        provenance_ok = len(launches) == expected_launches and all(
            'device=apple_gpu' in line and 'launch_success=true' in line and
            'provenance=generic_ptx_lowering' in line for line in launches)
        if result.returncode or not provenance_ok:
            failed.append(case)
            print(f'FAILED {case}: exit={result.returncode}, launches={launches}', file=sys.stderr)
    if failed:
        print('FAILED cases: ' + ', '.join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
