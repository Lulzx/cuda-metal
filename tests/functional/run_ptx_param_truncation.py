#!/usr/bin/env python3
"""Parameter stores truncate wide registers before argument/return ABI handling."""
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from run_ptx_scalar_tail_call import run_case
from run_ptx_widening import TEMPLATE


def fixture(width):
    helpers = f'''
.func (.param .b{width} result) arg_path(.param .b{width} input) {{
.reg .b32 %r1;
ld.param.u{width} %r1, [input];
st.param.b{width} [result], %r1;
ret;
}}
.func (.param .b{width} result) ret_path(.param .b64 input) {{
.reg .b64 %rd1;
ld.param.b64 %rd1, [input];
// Explicit scalar use avoids legacy unannotated-64-bit pointer inference.
xor.b64 %rd1, %rd1, 0;
st.param.b{width} [result], %rd1;
ret;
}}
'''
    body = f'''
.param .b{width} narrow;
.param .b64 wide;
.param .b{width} answer;
ld.global.u64 %rd9, [%rd5];
st.param.b{width} [narrow], %rd9;
call.uni (answer), arg_path, (narrow);
ld.param.u{width} %r5, [answer];
cvt.u64.u32 %rd7, %r5;
st.param.b64 [wide], %rd9;
call.uni (answer), ret_path, (wide);
ld.param.u{width} %r6, [answer];
cvt.u64.u32 %rd8, %r6;
'''
    return TEMPLATE.replace('.visible .entry', helpers + '\n.visible .entry').replace('BODY', body)


def main():
    build = Path(sys.argv[1]).resolve()
    rng = random.Random(6432)
    values = [0, 1, 255, 256, 65535, 65536, (1 << 32)-1, 1 << 32,
              1 << 63, (1 << 64)-1, 0xdeadbeef01234567]
    values += [rng.getrandbits(64) for _ in range(256)]
    for width in (32, 16, 8):
        mask = (1 << width)-1
        run_case(build, fixture(width), values,
                 [word for value in values for word in (value & mask, value & mask)],
                 f'64-to-{width} parameter argument and return stores')
    with tempfile.TemporaryDirectory(prefix='param-width-negative-') as work:
        source, msl = Path(work)/'bad.ptx', Path(work)/'bad.metal'
        invalid = fixture(32).replace('arg_path(.param .b32 input)',
                                      'arg_path(.param .b64 input)')
        invalid = invalid.replace('ld.param.u32 %r1, [input];',
                                  '.reg .b64 %rd0;\nld.param.u64 %rd0, [input];\ncvt.u32.u64 %r1, %rd0;')
        source.write_text(invalid)
        result = subprocess.run([str(build/'cumetalc'), str(source), '--backend=cumetal-ir',
                                 '--ptx-strict', '--entry', 'tail_probe', '--emit=msl',
                                 '-o', str(msl)], capture_output=True, text=True)
        assert result.returncode != 0 and 'does not fit its declared argument type' in result.stderr, result.stderr
    print('NEGATIVE_PASS mismatched parameter width rejected before Metal compilation')


if __name__ == '__main__':
    main()
