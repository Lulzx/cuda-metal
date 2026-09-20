#!/usr/bin/env python3
"""A branch establishes its predicate even when its producer is not analyzed."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_repeated_predicate.ptx').read_text()
floating = source.replace('.reg .b16 %rs1;', '.reg .f32 %f1;')
floating = floating.replace('cvt.u16.u32 %rs1, %r0;\nsetp.eq.b16 %p0, %rs1, 0;',
                            'cvt.rn.f32.u32 %f1, %r0;\nsetp.eq.f32 %p0, %f1, 0f00000000;')
for label, case in (
    ('16-bit predicate', source),
    ('inverted alias', source.replace('@!%p0 bra USE;', 'not.pred %p2, %p0;\n@%p2 bra USE;')),
    ('floating-point predicate', floating),
):
    run_integer_case(build, case, [1234567] * 65, [99] + [1234567] * 64,
                     label, entry='constant_guard', word_bits=32, output_words=1)
for case in (
    source.replace('JOIN:\n', 'JOIN:\nnot.pred %p0, %p0;\n'),
    source.replace('JOIN:\n', 'JOIN:\nst.global.u32 [%rd3], %r1;\n'),
    source.replace('@!%p0 bra USE;', '@!%p3 bra USE;'),
):
    expect_compile_failure(build, case, 'constant_guard', 'PTX register')
