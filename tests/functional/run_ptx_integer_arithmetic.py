#!/usr/bin/env python3
"""Numerical 64-bit Metal integer operations with independent Python oracles."""
from pathlib import Path
import sys
from ptx_test_support import expect_compile_failure, run_integer_case

REFERENCE = Path(__file__).parent / 'reference'

def integer_minmax(build):
    values = [0, 1, 63, 64, 65, (1 << 31), (1 << 32), (1 << 63)-1,
              1 << 63, (1 << 63)+1, (1 << 64)-2, (1 << 64)-1]
    source = values
    expected = []
    for value in values:
        signed = value if value < (1 << 63) else value - (1 << 64)
        mask = (1 << 64)-1
        expected.extend((min(value, 64), max(value, 64), min(signed, -1) & mask, max(signed, -1) & mask))
    run_integer_case(build, (REFERENCE / 'ptx_integer_minmax.ptx').read_text(),
                     source, expected, 'integer_minmax', entry='integer_minmax',
                     input_words=1, output_words=4)


def mul_hi_64(build):
    mask = (1 << 64)-1
    boundary = [0, 1, 2, (1 << 32)-1, 1 << 32, (1 << 32)+1,
                (1 << 63)-1, 1 << 63, (1 << 63)+1, mask-1, mask]
    pairs = [(a, b) for a in boundary for b in boundary]
    state = 0x123456789abcdef
    for _ in range(4096):
        state = (state * 6364136223846793005 + 1) & mask
        a = state
        state = (state * 6364136223846793005 + 1) & mask
        pairs.append((a, state))
    source = [v for pair in pairs for v in pair]
    expected = []
    for a, b in pairs:
        sa = a if a < (1 << 63) else a - (1 << 64)
        sb = b if b < (1 << 63) else b - (1 << 64)
        expected.extend(((a*b) >> 64, (a*7544311872078572213) >> 64,
                         ((sa*sb) >> 64) & mask, ((sa*-1) >> 64) & mask))
    run_integer_case(build, (REFERENCE / 'ptx_mul_hi_64.ptx').read_text(),
                     source, expected, 'mul_hi_64', entry='mul_hi_64',
                     input_words=2, output_words=4)
    invalid = (REFERENCE / 'ptx_mul_hi_64.ptx').read_text().replace(
        'mul.hi.u64 %rd6, %rd5, %rd12;', 'mul.hi.u64 %rd6, %rd5, %r1;')
    expect_compile_failure(build, invalid, 'mul_hi_64', 'matching')


if __name__ == '__main__':
    cases = dict(integer_minmax=integer_minmax, mul_hi_64=mul_hi_64)
    cases[sys.argv[2]](Path(sys.argv[1]).resolve())
