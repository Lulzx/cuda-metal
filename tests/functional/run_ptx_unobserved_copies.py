#!/usr/bin/env python3
"""Discarded retry payload copies must not create an initial SSA value."""
from pathlib import Path
import subprocess
import sys

if '--gpu-child' not in sys.argv:
    result = subprocess.run([sys.executable, __file__, *sys.argv[1:], '--gpu-child'],
                            capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    launches = [line for line in result.stderr.splitlines()
                if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
    assert len(launches) == 13, launches
    assert all('device=apple_gpu' in line and 'launch_success=true' in line and
               'provenance=generic_ptx_lowering' in line for line in launches), launches
    raise SystemExit(0)

from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_unobserved_copies.ptx').read_text()
for width in (16, 32, 64):
    case = source
    if width == 16:
        case = case.replace('.reg .b32 %r<9>;', '.reg .b32 %r<9>;\n.reg .b16 %rs<2>;')
        case = case.replace('%r6', '%rs0').replace('%r7', '%rs1')
        case = case.replace('mov.b32', 'mov.b16').replace('ld.global.u32', 'ld.global.u16')
        case = case.replace('st.global.u32 [%rd4], %rs1;',
                            'cvt.u32.u16 %r6, %rs1;\nst.global.u32 [%rd4], %r6;')
    if width == 64:
        case = case.replace('.reg .b64 %rd<5>;', '.reg .b64 %rd<7>;')
        case = case.replace('%r6', '%rd5').replace('%r7', '%rd6')
        case = case.replace('mov.b32', 'mov.b64').replace('ld.global.u32', 'ld.global.u64')
        case = case.replace('st.global.u32', 'st.global.u64').replace('%r0, 4;', '%r0, 8;')
    values = [17 + x * 23 + ((1 << 40) if width == 64 else 0) for x in range(65)]
    for mixed in (False, True):
        variant = case
        expected = values
        if mixed:
            variant = variant.replace('mov.u32 %r8, 0;',
                'mov.u32 %r8, 0;\nand.b32 %r1, %r0, 3;\nadd.u32 %r1, %r1, 1;')
            variant = variant.replace('setp.eq.u32 %p1, %r8, 1;',
                                      'setp.le.u32 %p1, %r8, %r1;')
            # Keep the retry counter in a separate output word, so losing a
            # volatile store is a numerical failure rather than an invisible edit.
            variant = variant.replace('add.u64 %rd4, %rd1, %rd2;',
                'add.u64 %rd4, %rd1, %rd2;\nadd.u64 %rd4, %rd4, %rd2;')
            if width == 64:
                variant = variant.replace('.reg .b64 %rd<7>;', '.reg .b64 %rd<8>;')
                store = 'cvt.u64.u32 %rd7, %r8;\nst.volatile.global.u64 [%rd4+8], %rd7;'
            else:
                store = 'st.volatile.global.u32 [%rd4+4], %r8;'
            variant = variant.replace('mov.pred %p2, 1;', 'mov.pred %p2, 1;\n' + store)
            expected = [word for lane, value in enumerate(values) for word in (value, lane % 4 + 1)]
        run_integer_case(build, variant, values, expected, f'b{width} mixed={mixed}',
                         word_bits=64 if width == 64 else 32, output_words=2 if mixed else 1)
        straight = variant.replace('JOIN:\n',
            'JOIN:\n' + 'mov.b32 %r3, %r0;\n' * 48 + 'AFTER_COPIES:\n')
        run_integer_case(build, straight, values, expected, f'b{width} mixed={mixed} straight',
                         word_bits=64 if width == 64 else 32, output_words=2 if mixed else 1)

# Copies consumed before and after intervening overwrites remain distinct.
chain = source[:source.index('LOOP:')] + '''LOOP:
ld.global.u32 %r6, [%rd3];
mov.b32 %r7, %r6;
mov.b32 %r1, %r7;
mov.u32 %r7, 19;
mov.b32 %r8, %r7;
mov.b32 %r7, %r1;
add.u32 %r7, %r7, %r8;
st.global.u32 [%rd4], %r7;
DONE:
ret;
}
'''
values = [17 + lane * 23 for lane in range(65)]
run_integer_case(build, chain, values, [value + 19 for value in values],
                 'live copies across overwrites', word_bits=32, output_words=1)

for label, operation in (
    ('store', 'st.global.u32 [%rd4], %r7;'),
    ('address', 'ld.global.u32 %r1, [%r7];'),
    ('branch', 'setp.eq.u32 %p0, %r7, 0;\n@%p0 bra DONE;'),
):
    variant = source.replace('mov.pred %p2, 1;', 'mov.pred %p2, 1;\n' + operation)
    expect_compile_failure(build, variant, 'integer_probe', 'PTX register')
    print(f'REJECTED undefined {label}')
