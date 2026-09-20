#!/usr/bin/env python3
"""A literal-zero SSA value remains null when passed to a pointer block argument."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_null_edges.ptx').read_text()
values = [1234567] * 65
expected = [99] + values[1:]
run_integer_case(build, source, values, expected, 'null pointer branch edge',
                 entry='null_edge', word_bits=32, output_words=1)
expect_compile_failure(build, source.replace('mov.u64 %rd2, 0;', 'mov.u64 %rd2, 7;'),
                       'null_edge', 'pointer branch argument requires a pointer or proven null')

# The production RSA/P-256 failures use thread-local pointers rather than device
# pointers. Exercise both address spaces with the same independently checked data.
local = source.replace('.reg .pred', '.local .align 4 .b8 slot[4];\n.reg .pred', 1)
local = local.replace('setp.eq.u32 %p0, %r0, 0;',
                      'mov.u64 %rd6, slot;\nld.global.u32 %r3, [%rd0];\n'
                      'st.local.u32 [%rd6], %r3;\nsetp.eq.u32 %p0, %r0, 0;')
local = local.replace('mov.u64 %rd2, %rd0;', 'mov.u64 %rd2, %rd6;')
local = local.replace('ld.global.u32 %r1, [%rd2];', 'ld.local.u32 %r1, [%rd2];')
run_integer_case(build, local, values, expected, 'null thread-local pointer edge',
                 entry='null_edge', word_bits=32, output_words=1)

# Two entries into the A/B cycle require dispatcher emission. The input controls
# whether the cycle is taken; compare both the null and non-null exits.
dispatch = source.replace('mov.u64 %rd2, 0;\n@%p0 bra JOIN;\nmov.u64 %rd2, %rd0;', '''
mov.u64 %rd2, 0;
ld.global.u32 %r3, [%rd0];
@%p0 bra A;
B:
mov.u64 %rd2, %rd0;
add.u32 %r3, %r3, 1;
setp.ge.u32 %p2, %r3, 2;
@%p2 bra JOIN;
A:
add.u32 %r3, %r3, 1;
setp.lt.u32 %p3, %r3, 2;
@%p3 bra B;''')
for initial in (0, 1):
    run_integer_case(build, dispatch, [initial] * 65,
                     ([99] if initial else [0]) + [initial] * 64,
                     f'null pointer dispatcher edge initial={initial}',
                     entry='null_edge', word_bits=32, output_words=1)
