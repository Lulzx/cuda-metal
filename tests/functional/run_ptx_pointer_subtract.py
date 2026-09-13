#!/usr/bin/env python3
"""Pointer-minus-offset keeps local/device provenance through loops."""
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from run_ptx_scalar_tail_call import run_case

PTX = '''
.version 7.1
.target sm_80
.address_size 64
.visible .entry tail_probe(.param .u64 input, .param .u64 output, .param .u32 count) {
.local .align 8 .b8 local_bytes[8];
.reg .b64 %rd<20>;
.reg .b32 %r<8>;
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
ld.global.u64 %rd7, [%rd5];
mov.u64 %rd8, local_bytes;
st.local.u64 [%rd8], %rd7;
add.u64 %rd9, %rd8, 7;
mov.u64 %rd10, 0;
mov.u64 %rd11, 0;
LOOP:
sub.u64 %rd12, %rd9, %rd10;
ld.local.u8 %r5, [%rd12];
cvt.u64.u32 %rd13, %r5;
shl.b64 %rd14, %rd10, 3;
shl.b64 %rd15, %rd13, %rd14;
or.b64 %rd11, %rd11, %rd15;
add.u64 %rd10, %rd10, 1;
setp.lt.u64 %p1, %rd10, 8;
@%p1 bra LOOP;
add.u64 %rd16, %rd5, 8;
sub.u64 %rd17, %rd16, %rd10;
ld.global.u64 %rd18, [%rd17];
st.global.u64 [%rd6], %rd11;
st.global.u64 [%rd6+8], %rd18;
DONE:
ret;
}
'''


def main():
    build = Path(sys.argv[1]).resolve()
    rng = random.Random(19058)
    values = [0, 1, (1 << 64)-1, 0x0123456789abcdef, 1 << 63]
    values += [rng.getrandbits(64) for _ in range(256)]
    expected = [word for value in values for word in (int.from_bytes(value.to_bytes(8, 'little'), 'big'), value)]
    run_case(build, PTX, values, expected, 'local byte reversal and device pointer subtraction')
    with tempfile.TemporaryDirectory(prefix='pointer-sub-negative-') as work:
        source, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
        for invalid in ('sub.u64 %rd12, %rd9, %rd8;', 'sub.u64 %rd12, %rd10, %rd9;',
                        'sub.u32 %rd12, %rd9, %rd10;'):
            source.write_text(PTX.replace('sub.u64 %rd12, %rd9, %rd10;', invalid))
            result = subprocess.run([str(build / 'cumetalc'), str(source), '--backend=cumetal-ir',
                                     '--ptx-strict', '--entry', 'tail_probe', '--emit=msl',
                                     '--overwrite', '-o', str(msl)], capture_output=True, text=True)
            assert result.returncode != 0 and 'pointer subtraction' in result.stderr, result.stderr
    print('NEGATIVE_PASS pointer difference, integer-minus-pointer and narrow pointer arithmetic rejected')


if __name__ == '__main__':
    main()
