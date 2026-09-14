#!/usr/bin/env python3
"""A commuted address addition must not turn its runtime length into a pointer."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case, expect_compile_failure
from run_ptx_integer_widths import TEMPLATE

source = TEMPLATE.replace('.param .u64 input', '.param .u64 .ptr input')
source = source.replace('.param .u64 output', '.param .u64 .ptr output')
source = source.replace('.param .u32 count', '.param .u64 count')
source = source.replace('ld.param.u32 %r1, [count];',
                        'ld.param.u64 %rd9, [count];\ncvt.u32.u64 %r1, %rd9;')
source = source.replace('add.u64 %rd5, %rd1, %rd3;', 'add.u64 %rd5, %rd9, %rd1;')
source = source.replace('BODY', 'sub.u64 %rd7, 31, %rd9;\nld.global.u8 %r5, [%rd5];\ncvt.u64.u32 %rd8, %r5;')
abi = ['CUMETAL_ABI_V2', 'kernel integer_probe', 'shared 0', 'arg buffer 8', 'arg buffer 8', 'arg bytes 8']
build = Path(sys.argv[1]).resolve()
for count in (1, 7, 15, 31, 32, 63, 257):
    values = [((i * 0x0102030405060708) ^ 0xabcdef9876543210) & ((1 << 64)-1) for i in range(count)]
    byte = b''.join(v.to_bytes(8, 'little') for v in values)[count]
    expected = [(31 - count) & ((1 << 64)-1), byte] * count
    for commuted in (False, True):
        ptx = source if commuted else source.replace('%rd5, %rd9, %rd1', '%rd5, %rd1, %rd9')
        run_integer_case(build, ptx, values, expected, f'commuted={commuted}, length={count}', abi_lines=abi)
for expression in ('sub.u64 %rd7, 31, %rd1;', 'sub.u64 %rd7, %rd9, %rd1;'):
    expect_compile_failure(build, source.replace('sub.u64 %rd7, 31, %rd9;', expression),
                           'integer_probe', 'pointer subtraction')
