#!/usr/bin/env python3
"""A tuple pack retains both halves through loop edges and output stores."""
from pathlib import Path
import random
import sys
from ptx_test_support import run_integer_case, expect_compile_failure
from run_ptx_integer_widths import TEMPLATE

body = """
ld.global.u64 %rd7, [%rd5];
ld.global.u32 %r5, [%rd5+8];
cvt.u64.u32 %rd8, %r5;
setp.eq.u32 %p2, %r5, 0;
@%p2 bra STORE;
LOOP:
mov.b64 {%r6, %r7}, %rd7;
add.u32 %r6, %r6, 1;
mov.b64 %rd7, {%r6, %r7};
sub.u32 %r5, %r5, 1;
setp.ne.u32 %p2, %r5, 0;
@%p2 bra LOOP;
STORE:
"""
source = TEMPLATE.replace('mul.wide.u32 %rd3, %r2, 8;', 'mul.wide.u32 %rd3, %r2, 16;')
source = source.replace('.reg .pred %p1;', '.reg .pred %p<3>;').replace('BODY', body)
source = source.replace('.param .u64 input', '.param .u64 .ptr input')
source = source.replace('.param .u64 output', '.param .u64 .ptr output')
rng = random.Random(64032)
states = [0, 1, 0xffffffff, 1 << 32, 1 << 63, (1 << 64)-1, 0x0123456789abcdef]
states += [rng.getrandbits(64) for _ in range(32)]
values, expected = [], []
for state in states:
    for count in (0, 1, 7, 15, 31, 32, 63, 1024):
        values += [state, count]
        expected += [(state & ~0xffffffff) | ((state + count) & 0xffffffff), count]
build = Path(sys.argv[1]).resolve()
run_integer_case(build, source, values, expected, '64-bit packs through runtime loops', input_words=2)
for pack in ('mov.b64 %rd7, {%r6, %r7, %r8};', '@%p2 mov.b64 %rd7, {%r6, %r7};'):
    expect_compile_failure(build, source.replace('mov.b64 %rd7, {%r6, %r7};', pack), 'integer_probe', '')
