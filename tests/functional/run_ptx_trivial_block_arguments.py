#!/usr/bin/env python3
"""Loop invariants can be shared while changing values still reach each lane."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_trivial_block_arguments.ptx').read_text()
values = [(i * 17) & 255 for i in range(65)]
base = [max(1, value & 31) for value in values]
run_integer_case(build, source, values, base, 'invariant pointer and loop bound',
                 entry='invariant_loop', word_bits=32, output_words=1)
diamond = source.replace('st.global.u32 [%rd4], %r2;', '''
and.b32 %r6, %r3, 1;
setp.eq.u32 %p1, %r6, 0;
@%p1 bra RIGHT;
add.u32 %r2, %r2, 7;
bra JOIN;
RIGHT:
add.u32 %r2, %r2, 11;
JOIN:
st.global.u32 [%rd4], %r2;''')
run_integer_case(build, diamond, values,
                 [value + (11 if lane % 2 == 0 else 7) for lane, value in enumerate(base)],
                 'different join values remain distinct', entry='invariant_loop',
                 word_bits=32, output_words=1)
nested = source.replace('add.u32 %r2, %r2, 1;', '''
mov.u32 %r6, 0;
INNER:
add.u32 %r6, %r6, 1;
setp.lt.u32 %p1, %r6, 3;
@%p1 bra INNER;
add.u32 %r2, %r2, 1;''')
run_integer_case(build, nested, values, base, 'nested loop invariant aliases',
                 entry='invariant_loop', word_bits=32, output_words=1)
expect_compile_failure(build, source.replace('mov.u32 %r2, 0;', ''),
                       'invariant_loop', 'PTX register')
