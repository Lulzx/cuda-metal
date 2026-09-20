#!/usr/bin/env python3
"""An indexed helper recovers its pointer parameter from its own cvta.to.global.

Rust-CUDA compiles `#[inline(never)] fn read(bytes: &[u8], i: usize) -> u8` into
a helper whose base pointer, length and index all arrive as plain `.b64`
parameters. The only pointer evidence inside the helper is its own
`cvta.to.global`, so backward recovery has to accept that as proof; the base
must not be guessed from 64-bit width alone.
"""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case, expect_compile_failure
from run_ptx_integer_widths import TEMPLATE

table = [(i * 37 + 11) & 0xFF for i in range(64)]
header = ('.global .align 8 .b8 lookup_table[64] = {' +
          ','.join(map(str, table)) + '};\n')

# The bounds branch and its trap must survive the retyping, so keep them in the
# helper exactly as the producer emits them.
helper = """
.func (.param .b32 result) read_byte(.param .b64 address, .param .b64 length, .param .b64 index) {
.reg .b64 %rd<6>;
.reg .b32 %r1;
.reg .pred %p1;
ld.param.b64 %rd1, [address];
ld.param.b64 %rd3, [length];
ld.param.b64 %rd2, [index];
setp.ge.u64 %p1, %rd2, %rd3;
@%p1 bra TRAP;
cvta.to.global.u64 %rd4, %rd1;
add.s64 %rd5, %rd4, %rd2;
ld.global.u8 %r1, [%rd5];
st.param.b32 [result], %r1;
ret;
TRAP:
trap;
}
"""

# Two reads through the same helper: one from a promoted module table and one
# from the caller's own device buffer. Both indices are masked into range, so
# the trap edge stays unreachable and the results are exact.
body = """
ld.global.u32 %r5, [%rd5];
and.b32 %r5, %r5, 63;
cvt.u64.u32 %rd9, %r5;
mov.b64 %rd10, lookup_table;
cvta.global.u64 %rd11, %rd10;
.param .b64 address;
.param .b64 length;
.param .b64 index;
.param .b32 answer;
mov.u64 %rd12, 64;
st.param.b64 [address], %rd11;
st.param.b64 [length], %rd12;
st.param.b64 [index], %rd9;
call.uni (answer), read_byte, (address, length, index);
ld.param.b32 %r6, [answer];
ld.global.u32 %r7, [%rd5];
and.b32 %r7, %r7, 7;
cvt.u64.u32 %rd13, %r7;
mov.u64 %rd14, 8;
st.param.b64 [address], %rd5;
st.param.b64 [length], %rd14;
st.param.b64 [index], %rd13;
call.uni (answer), read_byte, (address, length, index);
ld.param.b32 %r8, [answer];
cvt.u64.u32 %rd7, %r6;
cvt.u64.u32 %rd8, %r8;
"""

values = list(range(256)) + [0x80000000, 0xffffffff, 0x123456789abcdef0]
expected = []
for value in values:
    expected.append(table[value & 63])
    expected.append((value >> (8 * (value & 7))) & 0xFF)

build = Path(sys.argv[1]).resolve()
source = TEMPLATE.replace('.reg .b64 %rd<10>;', '.reg .b64 %rd<15>;')
source = source.replace('.reg .b32 %r<9>;', '.reg .b32 %r<10>;')
source = source.replace('.visible .entry', header + helper + '.visible .entry')
source = source.replace('BODY', body)
run_integer_case(build, source, values, expected, 'indexed helper reads')

# A helper's pointer parameter is still recovered, not guessed: converting the
# same address to a different space inside the helper must stay a refusal.
expect_compile_failure(
    build, source.replace('cvta.to.global.u64 %rd4, %rd1;',
                          'cvta.to.local.u64 %rd4, %rd1;'),
    'integer_probe', '')
