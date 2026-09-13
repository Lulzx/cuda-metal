#!/usr/bin/env python3
"""Integer source-width semantics for negative-spelled widening immediates."""
from pathlib import Path
import random
import sys
from run_ptx_scalar_tail_call import run_case

TEMPLATE = '''
.version 7.1
.target sm_80
.address_size 64
.visible .entry tail_probe(.param .u64 input, .param .u64 output, .param .u32 count) {
.reg .b64 %rd<10>;
.reg .b32 %r<9>;
.reg .b16 %rs1;
.reg .pred %p1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.param.u32 %r1, [count];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.u32 %r2, %r2, %r3, %r4;
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra DONE;
mul.wide.u32 %rd3, %r2, 8;
mul.wide.u32 %rd4, %r2, 16;
add.u64 %rd5, %rd1, %rd3;
add.u64 %rd6, %rd2, %rd4;
BODY
st.global.u64 [%rd6], %rd7;
st.global.u64 [%rd6+8], %rd8;
DONE:
ret;
}
'''


def main():
    build = Path(sys.argv[1]).resolve()
    rng = random.Random(5832)
    for width, coefficients in [(32, [-1925330167, -1, -2147483648]), (16, [-12345, -1, -32768])]:
        mask = (1 << width) - 1
        values = [0, 1, 2, (1 << (width-1))-1, 1 << (width-1), mask-1, mask]
        values += [rng.getrandbits(width) for _ in range(256)]
        for coefficient in coefficients:
            if width == 32:
                body = f'''ld.global.u32 %r5, [%rd5];
mul.wide.u32 %rd7, %r5, {coefficient};
mul.wide.s32 %rd8, %r5, {coefficient};'''
            else:
                body = f'''ld.global.u16 %rs1, [%rd5];
mul.wide.u16 %r5, %rs1, {coefficient};
mul.wide.s16 %r6, %rs1, {coefficient};
cvt.u64.u32 %rd7, %r5;
cvt.u64.u32 %rd8, %r6;'''
            expected = []
            for value in values:
                signed = value if value < 1 << (width-1) else value - (1 << width)
                expected += [value * (coefficient & mask), (signed * coefficient) & ((1 << (2*width))-1)]
            run_case(build, TEMPLATE.replace('BODY', body), values, expected,
                     f'u{width}/s{width} widening immediate {coefficient}')


if __name__ == '__main__':
    main()
