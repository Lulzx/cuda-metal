#!/usr/bin/env python3
"""Expanded helper calls preserve aggregate returns, local pointers and loops."""
from pathlib import Path
import random
import sys
from run_ptx_scalar_tail_call import run_case

PTX = '''
.version 7.1
.target sm_80
.address_size 64
.func (.param .align 8 .b8 result[16]) pair(.param .u64 input, .param .u64 local_ptr) {
.reg .b64 %rd<6>;
.reg .pred %p1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [local_ptr];
cvta.to.local.u64 %rd2, %rd2;
setp.eq.u64 %p1, %rd1, -1;
@%p1 bra FAIL;
ld.local.u64 %rd3, [%rd2];
add.u64 %rd4, %rd1, 5;
add.u64 %rd5, %rd3, 1;
st.local.u64 [%rd2], %rd5;
st.param.u64 [result], %rd4;
st.param.u64 [result+8], %rd5;
ret;
FAIL:
trap;
}
.visible .entry tail_probe(.param .u64 input, .param .u64 output, .param .u32 count) {
.local .align 8 .b8 storage[8];
.reg .b64 %rd<12>;
.reg .b32 %r<7>;
.reg .pred %p1;
.param .u64 x;
.param .u64 p;
.param .align 8 .b8 answer[16];
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
ld.global.u64 %rd7, [%rd5];
mov.u64 %rd8, storage;
st.local.u64 [%rd8], %rd7;
mov.u32 %r5, 0;
LOOP:
st.param.u64 [x], %rd7;
st.param.u64 [p], %rd8;
call.uni (answer), pair, (x, p);
ld.param.u64 %rd7, [answer];
ld.param.u64 %rd9, [answer+8];
add.u32 %r5, %r5, 1;
setp.lt.u32 %p1, %r5, 2;
@%p1 bra LOOP;
// Same helper at a distinct call site: its value/block IDs must be disjoint.
st.param.u64 [x], %rd7;
st.param.u64 [p], %rd8;
call.uni (answer), pair, (x, p);
ld.param.u64 %rd7, [answer];
ld.param.u64 %rd9, [answer+8];
st.global.u64 [%rd6], %rd7;
st.global.u64 [%rd6+8], %rd9;
DONE:
ret;
}
'''

if __name__ == '__main__':
    rng = random.Random(47238)
    values = [0, 1, 255, 1 << 32, 1 << 59] + [rng.getrandbits(60) for _ in range(256)]
    expected = [word for value in values for word in (value + 15, value + 3)]
    run_case(Path(sys.argv[1]).resolve(), PTX, values, expected,
             'trap-capable aggregate helper returns, local pointer side effects, repeated and loop calls')
