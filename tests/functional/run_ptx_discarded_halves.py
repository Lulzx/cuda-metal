#!/usr/bin/env python3
"""Unused tuple lanes do not require values; the observed lane is exact."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case
from run_ptx_integer_widths import TEMPLATE

body = """
ld.global.u32 %r5, [%rd5];
mov.b64 %rd9, {PACK};
add.u32 %r8, %r5, 3;
mov.b64 {EXTRACT}, %rd9;
cvt.u64.u32 %rd7, %r7;
cvt.u64.u32 %rd8, %r8;
"""
values = list(range(65536)) + [0x7fffffff, 0x80000000, 0xfffffffd, 0xfffffffe, 0xffffffff]
expected = [word for value in values for word in (value, (value + 3) & 0xffffffff)]
for high in (False, True):
    for discarded in ('_', '%r9'):
        pack = '%r6, %r5' if high else '%r5, %r6'
        extract = f'{discarded}, %r7' if high else f'%r7, {discarded}'
        source = TEMPLATE.replace('.reg .b32 %r<9>;', '.reg .b32 %r<10>;')
        source = source.replace('BODY', body.replace('PACK', pack).replace('EXTRACT', extract))
        run_integer_case(Path(sys.argv[1]).resolve(), source, values, expected,
                         f'observed {"high" if high else "low"} half, discarded {discarded}')
