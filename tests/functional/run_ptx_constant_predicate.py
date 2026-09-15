#!/usr/bin/env python3
"""Constant predicate flags guard conditionally defined values across CFG joins."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_constant_predicate.ptx').read_text()
values = [1234567] * 65
expected = [99] + values[1:]
for label, case in (
    ('inverted predicate', source),
    ('direct inverse branch', source.replace('not.pred %p2, %p1;\n@%p2 bra USE;', '@!%p1 bra USE;')),
    ('true selects definition', source.replace('mov.pred %p1, -1;', 'mov.pred %p1, 0;')
     .replace('ld.global.u32 %r1, [%rd0];\nmov.pred %p1, 0;',
              'ld.global.u32 %r1, [%rd0];\nmov.pred %p1, 1;')
     .replace('not.pred %p2, %p1;', 'mov.pred %p2, %p1;')),
):
    run_integer_case(build, case, values, expected, label,
                     entry='constant_guard', word_bits=32, output_words=1)
# The true flag must survive an intervening loop on the no-definition path.
loop = source.replace('@%p0 bra JOIN;', '@%p0 bra WAIT;')
loop = loop.replace('JOIN:\n', 'bra JOIN;\nWAIT:\nmov.u32 %r6, 0;\nLOOP:\n'
                    'add.u32 %r6, %r6, 1;\nsetp.lt.u32 %p3, %r6, 3;\n'
                    '@%p3 bra LOOP;\nJOIN:\n')
run_integer_case(build, loop, values, expected, 'constant flag across a loop',
                 entry='constant_guard', word_bits=32, output_words=1)
for case in (
    source.replace('mov.pred %p1, -1;', 'mov.pred %p1, 0;'),
    source.replace('JOIN:\n', 'JOIN:\n@%p0 mov.pred %p1, 0;\n'),
    source.replace('JOIN:\n', 'JOIN:\nsetp.ne.u32 %p1, %r0, 0;\n'),
):
    expect_compile_failure(build, case, 'constant_guard', 'undefined')
