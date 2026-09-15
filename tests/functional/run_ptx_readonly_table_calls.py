#!/usr/bin/env python3
"""Relocated immutable tables can be read through proven direct helpers."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_readonly_table_calls.ptx').read_text()
values = [i * 17 for i in range(65)]
table = [42, 17, 99, 5]
run_integer_case(build, source, values, [table[x & 3] for x in values],
                 'read-only table helper', entry='table_probe', word_bits=32, output_words=1)
wrapper = '''
.func (.param .u32 answer) forward_table(.param .u64 pointer, .param .u32 choice) {
.reg .b64 %rd1;
.reg .b32 %r<3>;
.param .u64 forwarded;
.param .u32 index;
.param .u32 value;
ld.param.u64 %rd1, [pointer];
ld.param.u32 %r1, [choice];
st.param.u64 [forwarded], %rd1;
st.param.u32 [index], %r1;
call.uni (value), read_table, (forwarded, index);
ld.param.u32 %r2, [value];
st.param.u32 [answer], %r2;
ret;
}
'''
chained = source.replace('.visible .entry', wrapper + '\n.visible .entry')
chained = chained.replace('call.uni (result), read_table,', 'call.uni (result), forward_table,')
run_integer_case(build, chained, values, [table[x & 3] for x in values],
                 'forwarded read-only table', entry='table_probe', word_bits=32, output_words=1)
twice = source.replace('st.global.u32 [%rd4], %r2;', '''
add.u32 %r1, %r1, 1;
and.b32 %r1, %r1, 3;
st.param.u64 [argument], %rd5;
st.param.u32 [index], %r1;
call.uni (result), read_table, (argument, index);
ld.param.u32 %r6, [result];
add.u32 %r2, %r2, %r6;
st.global.u32 [%rd4], %r2;''')
run_integer_case(build, twice, values, [table[x & 3] + table[(x + 1) & 3] for x in values],
                 'reused argument slots', entry='table_probe', word_bits=32, output_words=1)
for label, insertion in (
    ('write through argument', 'st.global.u32 [%rd3], 7;'),
    ('pointer escape', 'st.global.u64 [%rd3], %rd1;'),
    ('atomic mutation', 'atom.global.add.u32 %r2, [%rd3], 1;'),
    ('reduction mutation', 'red.global.add.u32 [%rd3], 1;'),
):
    invalid = source.replace('ld.global.u32 %r2, [%rd3];', insertion + '\nld.global.u32 %r2, [%rd3];')
    expect_compile_failure(build, invalid, 'table_probe', 'relocated table address')
    print('REJECTED', label)
    expect_compile_failure(build, chained.replace('ld.global.u32 %r2, [%rd3];',
                           insertion + '\nld.global.u32 %r2, [%rd3];'),
                           'table_probe', 'relocated table address')
for replacement in (
    'st.param.u64 [argument], %rd5;\nbra CALL;\nCALL:',
    '@%p1 st.param.u64 [argument], %rd5;',
    'st.param.u64 [argument], %rd5;\nst.param.u64 [argument], 0;',
    'st.param.u64 [argument], %rd5;\n' + 'mov.u32 %r7, 0;\n' * 65,
):
    expect_compile_failure(build, source.replace('st.param.u64 [argument], %rd5;', replacement),
                           'table_probe', 'relocated table address')
escaped_after_call = source.replace('ld.param.u32 %r2, [result];',
    'ld.param.u64 %rd5, [argument];\nst.global.u32 [%rd5], 7;\nld.param.u32 %r2, [result];')
expect_compile_failure(build, escaped_after_call, 'table_probe', 'relocated table address')
