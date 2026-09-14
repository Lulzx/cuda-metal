#!/usr/bin/env python3
"""Read an immutable table through both supported symbolic pointer encodings."""
from pathlib import Path
import re
import sys
from ptx_test_support import run_integer_case

PTX = r'''
.version 7.1
.target sm_80
.address_size 64
.global .align 8 .u8 alias[8] = {0XFF(table),0XFF00(table),0XFF0000(table),0XFF000000(table),0XFF00000000(table),0XFF0000000000(table),0XFF000000000000(table),0XFF00000000000000(table)};
.global .align 4 .b8 table[16] = {42,17,99,5,254,1,128,63,9,8,7,6,255,0,127,128};
.visible .entry relocated_table(.param .u64 input, .param .u64 output, .param .u32 count) {
 .reg .b64 %rd<9>;
 .reg .b32 %r<8>;
 .reg .pred %p1;
 ld.param.u64 %rd1, [input];
 ld.param.u64 %rd2, [output];
 ld.param.u32 %r1, [count];
 mov.u32 %r2, %ctaid.x;
 mov.u32 %r3, %ntid.x;
 mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4;
 setp.ge.u32 %p1, %r5, %r1;
 @%p1 bra DONE;
 mul.wide.u32 %rd3, %r5, 4;
 add.u64 %rd4, %rd1, %rd3;
 add.u64 %rd5, %rd2, %rd3;
 ld.global.u32 %r6, [%rd4];
 ld.global.nc.u64 %rd6, [alias];
 mul.wide.u32 %rd7, %r6, 4;
 add.u64 %rd8, %rd6, %rd7;
 ld.global.u32 %r7, [%rd8];
 st.global.u32 [%rd5], %r7;
DONE:
 ret;
}
'''

def main():
    ptx = PTX
    if sys.argv[2] == 'typed':
        ptx = re.sub(r'\.global \.align 8 \.u8 alias\[8\] = \{[^}]+\};',
                     '.global .align 8 .u64 alias[1] = {table};', ptx)
        ptx = ptx.replace('ld.global.nc.u64', 'ld.global.nc.b64')
    table_bytes = bytes([42, 17, 99, 5, 254, 1, 128, 63, 9, 8, 7, 6, 255, 0, 127, 128])
    table = [int.from_bytes(table_bytes[i:i+4], 'little') for i in range(0, 16, 4)]
    values = [(i * 7 + 1) % len(table) for i in range(257)]
    run_integer_case(Path(sys.argv[1]).resolve(), ptx, values,
                     [table[i] for i in values], 'immutable table '+sys.argv[2],
                     entry='relocated_table', word_bits=32, outputs_per_input=1)


if __name__ == '__main__':
    main()
