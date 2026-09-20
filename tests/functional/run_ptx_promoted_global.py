#!/usr/bin/env python3
"""PTX globals proven read-only keep their physical storage through conversions."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case, expect_compile_failure
from run_ptx_integer_widths import TEMPLATE

table = [i ^ 0x5a for i in range(64)]
header = '.global .align 8 .b8 private$table[64] = {' + ','.join(map(str, table)) + '};\n'
helper = """
.func (.param .b32 result) load_byte(.param .b64 address) {
.reg .b64 %rd<3>;
.reg .b32 %r1;
ld.param.b64 %rd1, [address];
cvta.to.global.u64 %rd2, %rd1;
ld.global.u8 %r1, [%rd2];
st.param.b32 [result], %r1;
ret;
}
"""
body = """
ld.global.u32 %r5, [%rd5];
and.b32 %r5, %r5, 63;
cvt.u64.u32 %rd9, %r5;
mov.b64 %rd10, private$table;
cvta.global.u64 %rd11, %rd10;
cvta.to.global.u64 %rd12, %rd11;
add.u64 %rd13, %rd12, %rd9;
LOAD
cvt.u64.u32 %rd7, %r6;
cvt.u64.u32 %rd8, %r7;
"""
loads = ['ld.global.u8 %r6, [%rd13];\nld.global.u8 %r7, [%rd5];', """
.param .b64 address;
.param .b32 answer;
st.param.b64 [address], %rd13;
call.uni (answer), load_byte, (address);
ld.param.b32 %r6, [answer];
st.param.b64 [address], %rd5;
call.uni (answer), load_byte, (address);
ld.param.b32 %r7, [answer];
"""]
values = list(range(256)) + [0x80000000, 0xffffffff, 0x123456789abcdef0]
expected = [word for value in values for word in (table[value & 63], value & 255)]
build = Path(sys.argv[1]).resolve()
for index, load in enumerate(loads):
    source = TEMPLATE.replace('.reg .b64 %rd<10>;', '.reg .b64 %rd<14>;')
    source = source.replace('.visible .entry', header + helper + '.visible .entry')
    source = source.replace('BODY', body.replace('LOAD', load))
    run_integer_case(build, source, values, expected, f'promoted globals, helper={bool(index)}')
    expect_compile_failure(build, source.replace('.global .align 8', '.const .align 8'), 'integer_probe', '')
    expect_compile_failure(build, source.replace('cvta.to.global.u64 %rd12', 'cvta.to.local.u64 %rd12'), 'integer_probe', '')
    # A write removes eligibility for constant promotion. It must not make a
    # global pointer convertible into local storage.
    mutable = source.replace('cvta.global.u64 %rd11, %rd10;',
                             'st.global.u8 [%rd10], 7;\ncvta.to.local.u64 %rd11, %rd10;')
    expect_compile_failure(build, mutable, 'integer_probe', '')
