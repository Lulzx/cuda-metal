#!/usr/bin/env python3
"""PTX instruction widths are distinct from register widths and literal spelling."""
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from ptx_test_support import run_u64_case

TEMPLATE = '''
.version 7.1
.target sm_80
.address_size 64
.visible .entry integer_probe(.param .u64 input, .param .u64 output, .param .u32 count) {
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

def widening(build):
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
            run_u64_case(build, TEMPLATE.replace('BODY', body), values, expected,
                     f'u{width}/s{width} widening immediate {coefficient}')


def conversions(build):
    # A byte load zero-extends into a .b16 register; cvt.s16.s8 must then
    # interpret the byte's sign, including before a subsequent absolute value.
    body = '''
ld.global.b8 %rs1, [%rd5];
cvt.s16.s8 %rs2, %rs1;
abs.s16 %rs3, %rs2;
cvt.u64.u16 %rd7, %rs3;
cvt.u16.u8 %rs4, %rs1;
cvt.u64.u16 %rd8, %rs4;
'''
    template = TEMPLATE.replace('.reg .b16 %rs1;', '.reg .b16 %rs<5>;')
    values = list(range(256))
    run_u64_case(build, template.replace('BODY', body), values,
             [x for v in values for x in (abs(v if v < 128 else v-256), v)],
             'all byte encodings through signed conversion and abs')

    # Upper register bits must not affect the instruction's source value.
    rng = random.Random(8163264)
    mask64 = (1 << 64)-1
    for width, container, register in [(8, 16, '%rs1'), (8, 32, '%r5'),
                                       (16, 32, '%r5'), (32, 64, '%rd9')]:
        mask = (1 << width)-1
        values = [0, 1, mask >> 1, (mask >> 1)+1, mask,
                  1 << width, (1 << container)-1]
        values += [rng.getrandbits(container) for _ in range(256)]
        body = f'''ld.global.b{container} {register}, [%rd5];
cvt.s64.s{width} %rd7, {register};
cvt.u64.u{width} %rd8, {register};'''
        expected = []
        for value in values:
            low = value & mask
            signed = low if low < 1 << (width-1) else low-(1 << width)
            expected += [signed & mask64, low]
        run_u64_case(build, template.replace('BODY', body), values, expected,
                 f'signed/unsigned {width}-bit cvt in {container}-bit register')


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


def parameters(build):
    rng = random.Random(6432)
    values = [0, 1, 255, 256, 65535, 65536, (1 << 32)-1, 1 << 32,
              1 << 63, (1 << 64)-1, 0xdeadbeef01234567]
    values += [rng.getrandbits(64) for _ in range(256)]
    for width in (32, 16, 8):
        mask = (1 << width)-1
        run_u64_case(build, fixture(width), values,
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
                                 '--ptx-strict', '--entry', 'integer_probe', '--emit=msl',
                                 '-o', str(msl)], capture_output=True, text=True)
        assert result.returncode != 0 and 'does not fit its declared argument type' in result.stderr, result.stderr
    print('NEGATIVE_PASS mismatched parameter width rejected before Metal compilation')


def stores(build):
    # Poison bytes adjacent to each narrow store detect accidental wide stores.
    poison = 0xa5a5a5a5
    values = [0, 1, 255, 256, 65535, 65536, (1 << 32)-1,
              1 << 32, (1 << 64)-1, 0xdeadbeef01234567]
    for width in (8, 16, 32):
        mask = (1 << width)-1
        body = f"""ld.global.u64 %rd9, [%rd5];
xor.b64 %rd10, %rd9, -1;
// Scalar and vector stores must preserve bytes outside their memory widths.
st.global.b{width} [%rd6], %rd9;
st.global.v2.b{width} [%rd6+8], {{%rd9, %rd10}};"""
        ptx = TEMPLATE.replace('.reg .b64 %rd<10>;', '.reg .b64 %rd<11>;').replace('BODY', body).replace(
            'st.global.u64 [%rd6], %rd7;\nst.global.u64 [%rd6+8], %rd8;', '')
        expected = []
        for value in values:
            expected += [(poison & ~mask) | (value & mask),
                         (poison & ~((1 << (2*width))-1)) |
                         (value & mask) | ((~value & mask) << width)]
        run_u64_case(build, ptx, values, expected, f'narrow {width}-bit stores')


def loads(build):
    for width, container in [(8, 16), (8, 32), (8, 64), (16, 32), (16, 64), (32, 64)]:
        mask = (1 << width)-1
        values = [0, 1, mask >> 1, (mask >> 1)+1, mask]
        prefix = {16: '%rs', 32: '%r', 64: '%rd'}[container]
        first, second = prefix+'5', prefix+'6'
        # Use distinct destinations from the pointer registers.
        if container == 64:
            first, second = '%rd10', '%rd11'
        body = f"""ld.global.s{width} {first}, [%rd5];
ld.global.u{width} {second}, [%rd5];
cvt.u64.u{container} %rd7, {first};
cvt.u64.u{container} %rd8, {second};"""
        ptx = TEMPLATE.replace('.reg .b16 %rs1;', '.reg .b16 %rs<7>;').replace(
            '.reg .b64 %rd<10>;', '.reg .b64 %rd<12>;').replace('BODY', body)
        expected = []
        for value in values:
            signed = value if value < 1 << (width-1) else value-(1 << width)
            expected += [signed & ((1 << container)-1), value]
        run_u64_case(build, ptx, values, expected, f'signed/unsigned {width}-bit loads into {container}')


if __name__ == '__main__':
    cases = dict(widening=widening, conversions=conversions, parameters=parameters,
                 stores=stores, loads=loads)
    cases[sys.argv[2]](Path(sys.argv[1]).resolve())
