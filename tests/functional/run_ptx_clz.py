#!/usr/bin/env python3
"""PTX clz b32/b64: exact counts, u32 results and preserved predicated values."""
from pathlib import Path
import sys

from ptx_test_support import expect_compile_failure, run_integer_case


PTX = '''.version 7.1
.target sm_80
.address_size 64
.visible .entry clz_probe(.param .u64 .ptr .global input,
                         .param .u64 .ptr .global output, .param .u32 count) {
 .reg .b64 %input, %output, %offset, %bits, %reused;
 .reg .b32 %lane, %block, %threads, %count, %parity, %low;
 .reg .b32 %count64, %count32, %narrow, %predicated, %predicated32, %joined, %zero32, %zero64;
 .reg .pred %done, %choose;
 ld.param.u64 %input, [input];
 ld.param.u64 %output, [output];
 ld.param.u32 %count, [count];
 mov.u32 %lane, %tid.x;
 mov.u32 %block, %ctaid.x;
 mov.u32 %threads, %ntid.x;
 mad.lo.u32 %lane, %block, %threads, %lane;
 setp.ge.u32 %done, %lane, %count;
 @%done bra DONE;
 mul.wide.u32 %offset, %lane, 8;
 add.u64 %input, %input, %offset;
 mul.wide.u32 %offset, %lane, 36;
 add.u64 %output, %output, %offset;
 ld.global.b64 %bits, [%input];
 ld.global.b32 %low, [%input];
 clz.b64 %count64, %bits;
 clz.b32 %count32, %low;
 clz.b32 %narrow, %bits;
 and.b32 %parity, %lane, 1;
 setp.ne.u32 %choose, %parity, 0;
 mov.u32 %predicated, 77;
 mov.u32 %predicated32, 79;
 bra GUARD;
GUARD:
 @%choose clz.b64 %predicated, %bits;
 @!%choose clz.b32 %predicated32, %low;
 @%choose bra ODD;
 clz.b64 %joined, %bits;
 bra JOIN;
ODD:
 mov.u32 %joined, 19;
JOIN:
 mov.b64 %reused, %bits;
 clz.b64 %reused, %reused;
 clz.b32 %zero32, 0;
 clz.b64 %zero64, 0;
 st.global.u32 [%output], %count64;
 st.global.u32 [%output+4], %count32;
 st.global.u32 [%output+8], %narrow;
 st.global.u32 [%output+12], %predicated;
 st.global.u32 [%output+16], %joined;
 st.global.u32 [%output+20], %reused;
 st.global.u32 [%output+24], %zero32;
 st.global.u32 [%output+28], %zero64;
 st.global.u32 [%output+32], %predicated32;
DONE:
 ret;
}
'''


def main():
    build = Path(sys.argv[1]).resolve()
    mask32 = (1 << 32) - 1
    values = [0, (1 << 64) - 1, mask32, 0x100000001, 0xffffffff00000000]
    values += [1 << bit for bit in range(64)]
    values += [0x0000000180000000, 0x8000000000000001, 0x0000000100000001,
               0x7fffffffffffffff, 0xffff, 0xfedcba9876543210, 0xaaaaaaaa,
               0x5555555555555555]
    expected = []
    for lane, value in enumerate(values):
        count64, count32 = 64 - value.bit_length(), 32 - (value & mask32).bit_length()
        expected.extend((count64, count32, count32, count64 if lane & 1 else 77,
                         19 if lane & 1 else count64, count64, 32, 64,
                         79 if lane & 1 else count32))
    for opcode in ('clz.b16', 'clz.u64', 'clz.b64.extra'):
        expect_compile_failure(build, PTX.replace('clz.b64 %count64', opcode + ' %count64'),
                               'clz_probe', 'clz')
    run_integer_case(build, PTX,
                     [word for value in values for word in (value & mask32, value >> 32)],
                     expected, '77 clz inputs: both widths, every one-hot, zero, guards and branches',
                     entry='clz_probe', word_bits=32, input_words=2, output_words=9,
                     abi_lines=['CUMETAL_ABI_V2', 'kernel clz_probe', 'shared 0',
                                'arg buffer 8', 'arg buffer 8', 'arg bytes 4'])


if __name__ == '__main__':
    main()
