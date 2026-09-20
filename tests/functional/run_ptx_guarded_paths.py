#!/usr/bin/env python3
"""Numerical guarded definitions; proof-negative cases live in the host test."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case

build = Path(sys.argv[1]).resolve()
case = sys.argv[2]
values = [i | ((i ^ 65535) << 16) for i in range(65536)]
values += [0, 0xffffffff, 0x80000000, 0x00008000, 0x1234abcd] + list(range(16))
expected = []
for value in values:
    if case == 'guarded_load':
        expected.append((value + 1) & 0xffffffff if value & 1 and value >= 8 else 99)
    else:
        indices = range(value & 7, 4) if case == 'bounded_self_select' else range(value & 7)
        expected.append(sum(i for i in indices if i & 1))
fixture = Path(__file__).parent / 'reference' / f'ptx_{case}.ptx'
run_integer_case(build, fixture.read_text(), values, expected, case,
                 entry='guarded_load' if case == 'guarded_load' else 'guarded_select',
                 word_bits=32, output_words=1)
