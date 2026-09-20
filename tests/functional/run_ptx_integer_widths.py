#!/usr/bin/env python3
"""PTX instruction widths are distinct from register widths and literal spelling."""
from pathlib import Path
import math
import random
import struct
import subprocess
import sys
import tempfile
from ptx_test_support import run_integer_case

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
            run_integer_case(build, TEMPLATE.replace('BODY', body), values, expected,
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
    run_integer_case(build, template.replace('BODY', body), values,
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
        run_integer_case(build, template.replace('BODY', body), values, expected,
                 f'signed/unsigned {width}-bit cvt in {container}-bit register')

    conversion_paths(build)


def type_path_cases():
    """Frozen-seed numerical fixtures, also usable for baseline compiler replay.

    Expected results use Python integers and closed-form expressions rather than
    an implementation of the importer's type rules. Large values are arithmetic
    operands only; address offsets are derived solely from the bounded lane ID.
    """
    rng = random.Random(760032)
    mask32, mask64 = (1 << 32) - 1, (1 << 64) - 1
    values = [0, 1, 64, 0x7fffffff, 0x80000000, mask32, 1 << 32,
              (1 << 32) + 1, 1 << 63, mask64, 0xdeadbeef01234567]
    values += [rng.getrandbits(64) for _ in range(65 - len(values))]
    template = TEMPLATE.replace('.reg .b64 %rd<10>;', '.reg .b64 %rd<12>;')
    template = template.replace('.reg .pred %p1;', '.reg .pred %p<4>;')

    def diamond(prefix, even, odd, suffix, reverse=False):
        arms = [('EVEN_VALUE', even), ('ODD_VALUE', odd)]
        if reverse:
            arms.reverse()
        body = prefix + '''
and.b32 %r6, %r2, 1;
setp.eq.u32 %p2, %r6, 0;
@%p2 bra EVEN_VALUE;
bra ODD_VALUE;
'''
        for label, instructions in arms:
            body += label + ':\n' + instructions + '\nbra VALUE_READY;\n'
        return body + 'VALUE_READY:\n' + suffix

    def variants(label, prefix, even, odd, suffix, expected):
        for reverse in (False, True):
            source = template.replace('BODY', diamond(prefix, even, odd, suffix, reverse))
            yield label + (' reversed blocks' if reverse else ''), source, values, expected
        renamed = source
        for old, new in (('%rd', '%word'), ('%rs', '%half'),
                         ('%r', '%lane'), ('%p', '%condition')):
            renamed = renamed.replace(old, new)
        yield label + ' renamed registers', renamed, values, expected

    # Reproduce a cvt-derived operand joining with another i64 definition before
    # mul.hi, including nonzero high product bits and poisoned upper input bits.
    multiplier = 0xfedcba9876543211
    expected = [word for value in values for word in
                (value & mask32, ((value & mask32) * multiplier) >> 64)]
    yield from variants(
        'conversion/copy join into mul.hi.u64',
        'ld.global.u64 %rd9, [%rd5];\ncvt.u32.u64 %r5, %rd9;\n',
        'cvt.u64.u32 %rd10, %r5;\nmov.b64 %rd7, %rd10;',
        f'and.b64 %rd7, %rd9, {mask32};',
        f'mul.hi.u64 %rd8, %rd7, {multiplier};', expected)

    # A scalar join is copied, used for pointer subtraction, then the same
    # register is reused as a pointer and finally as a scalar. The address
    # subtract/add round trip must reload this lane, never a neighbour or guard.
    expected = [word for value in values for word in (value & mask32, value)]
    yield from variants(
        'joined scalar/pointer/scalar round trip', '',
        'cvt.u64.u32 %rd7, %r2;\nshl.b64 %rd7, %rd7, 3;',
        'mul.wide.u32 %rd7, %r2, 8;',
        '''mov.b64 %rd10, %rd7;
sub.u64 %rd11, %rd5, %rd10;
add.u64 %rd7, %rd11, %rd7;
ld.global.u64 %rd8, [%rd7];
cvt.u32.u64 %r5, %rd8;
cvt.u64.u32 %rd7, %r5;''', expected)

    # Exercise every input with zero, one, two and three iterations. Each
    # iteration narrows after incrementing and then re-extends with the stated
    # signedness; wrapping across the sign boundary is observable in the result.
    loop_values = [value for value in values for _ in range(4)]
    for width in (8, 16, 32):
        mask = (1 << width) - 1
        expected = []
        for value in values:
            for count in range(4):
                low = (value + count) & mask
                signed = low if low < 1 << (width - 1) else low - (1 << width)
                expected += [signed & mask64, low]
        body = f'''ld.global.u64 %rd9, [%rd5];
cvt.s64.s{width} %rd7, %rd9;
cvt.u64.u{width} %rd8, %rd9;
and.b32 %r5, %r2, 3;
mov.u32 %r6, 0;
CONVERSION_LOOP:
setp.ge.u32 %p2, %r6, %r5;
@%p2 bra CONVERSION_DONE;
add.u64 %rd7, %rd7, 1;
add.u64 %rd8, %rd8, 1;
cvt.s64.s{width} %rd7, %rd7;
cvt.u64.u{width} %rd8, %rd8;
add.u32 %r6, %r6, 1;
bra CONVERSION_LOOP;
CONVERSION_DONE:
'''
        yield (f'signed/unsigned {width}-bit conversion loop',
               template.replace('BODY', body), loop_values, expected)

    yield from float_path_cases()


def float_path_cases():
    """Exact binary32 conversion bits across joins and integer/float reuse.

    struct.pack supplies an independent round-to-nearest/even binary32 reference.
    Reverse cases start from explicit binary32 encodings and Python's rounding
    operations. All operands and rounded results remain finite and in range;
    unsigned cases use nonnegative inputs.
    Every boundary appears on both branch arms, plus an odd final lane to retain
    a partial launch. No FP64 operation is present in the generated PTX.
    """
    mask32 = (1 << 32) - 1
    template = TEMPLATE.replace('.reg .pred %p1;',
                                '.reg .pred %p<3>;\n.reg .f32 %f<2>;')

    def f32_bits(value):
        return struct.unpack('<I', struct.pack('<f', value))[0]

    def variants(label, prefix, even, odd, suffix, values, expected):
        for reverse in (False, True):
            arms = [('EVEN_CONVERSION', even), ('ODD_CONVERSION', odd)]
            if reverse:
                arms.reverse()
            body = prefix + '''
and.b32 %r6, %r2, 1;
setp.eq.u32 %p2, %r6, 0;
@%p2 bra EVEN_CONVERSION;
bra ODD_CONVERSION;
'''
            for label_name, instructions in arms:
                body += label_name + ':\n' + instructions + '\nbra CONVERSION_READY;\n'
            body += 'CONVERSION_READY:\n' + suffix
            yield (label + (' reversed blocks' if reverse else ''),
                   template.replace('BODY', body), values, expected)

    # 2^24+1 and 2^24+3 are halfway cases with different even-neighbour choices.
    # Around 2^31/2^32, binary32 spacing is much larger than one: retain the
    # original integer separately so numeric conversion cannot masquerade as a
    # bit reinterpretation or discard the original register assignment.
    integer_boundaries = {
        'u32': [0, 1, 64, (1 << 24) - 1, 1 << 24, (1 << 24) + 1,
                (1 << 24) + 2, (1 << 24) + 3, 0x7fffff80, 0x7fffffff,
                0x80000000, 0x80000001, 0xffffff00, 0xffffff7f, 0xffffff80, mask32],
        's32': [-(1 << 31), -(1 << 31) + 1, -(1 << 24) - 3,
                -(1 << 24) - 1, -(1 << 24), -(1 << 24) + 1, -65, -1,
                0, 1, 64, (1 << 24) - 1, (1 << 24) + 1, (1 << 24) + 3,
                0x7fffff80, 0x7fffffff],
    }
    for source_type, boundaries in integer_boundaries.items():
        numbers = [number for number in boundaries for _ in range(2)] + [boundaries[-1]]
        values = [number & mask32 for number in numbers]
        expected = [word for number in numbers
                    for word in (f32_bits(number), number & mask32)]
        yield from variants(
            f'{source_type}-to-f32 nearest-even copied join and register reuse',
            f'ld.global.u32 %r5, [%rd5];\ncvt.rn.f32.{source_type} %r7, %r5;\n',
            'mov.f32 %f1, %r7;',
            'mov.b32 %r8, %r7;\nmov.f32 %f1, %r8;',
            '''mov.b32 %r8, %f1;
cvt.u64.u32 %rd7, %r8;
mov.u32 %r7, %r5;
cvt.u64.u32 %rd8, %r7;''', values, expected)

    # Signed zero, fractions on either side of integer/halfway boundaries, ties
    # with both even-neighbour choices, and large exactly representable inputs.
    # 0x4affffff is 8388607.5, the last positive half below the integer-only
    # binary32 range; 0x4f7fffff is the largest binary32 below 2^32.
    signed_encodings = [0x00000000, 0x80000000, 0x3effffff, 0x3f000000,
                        0x3f000001, 0x3f7fffff, 0x3fc00000, 0x3fffffff,
                        0x40200000, 0x40600000, 0xbf000000, 0xbf000001,
                        0xbf7fffff, 0xbfc00000, 0xbfffffff, 0xc0200000,
                        0xc0600000, 0x477fffc0, 0xc77fffc0, 0x4affffff,
                        0xcaffffff, 0x4b7fffff, 0xcb7fffff, 0x4b800000,
                        0xcb800000, 0x4effffff, 0xcf000000]
    unsigned_encodings = [bits for bits in signed_encodings if not bits >> 31]
    unsigned_encodings += [0x4f000000, 0x4f000001, 0x4f7fffff]
    modes = [('rni', 'nearest-even', round), ('rzi', 'truncation', int),
             ('rmi', 'floor', math.floor), ('rpi', 'ceil', math.ceil)]
    for destination, encodings, minimum, maximum in (
            ('s32', signed_encodings, -(1 << 31), (1 << 31) - 1),
            ('u32', unsigned_encodings, 0, mask32)):
        values = [bits for bits in encodings for _ in range(2)] + [encodings[-1]]
        for modifier, mode_name, reference in modes:
            expected = []
            for bits in values:
                number = struct.unpack('<f', struct.pack('<I', bits))[0]
                rounded = reference(number)
                assert math.isfinite(number) and minimum <= number <= maximum
                assert minimum <= rounded <= maximum
                expected += [rounded & mask32, bits]
            yield from variants(
                f'f32-to-{destination} {mode_name} copied join and register reuse',
                f'ld.global.f32 %f0, [%rd5];\ncvt.{modifier}.{destination}.f32 %r7, %f0;\n',
                'mov.u32 %r8, %r7;',
                'mov.b32 %r6, %r7;\nmov.u32 %r8, %r6;',
                '''cvt.u64.u32 %rd7, %r8;
mov.f32 %r7, %f0;
mov.b32 %r5, %r7;
cvt.u64.u32 %rd8, %r5;''', values, expected)


def conversion_paths(build, case_generator=type_path_cases):
    abi = ['CUMETAL_ABI_V2', 'kernel integer_probe', 'shared 0',
           'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
    for label, source, values, expected in case_generator():
        run_integer_case(build, source, values, expected, label, abi_lines=abi)


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
call.uni (answer),
arg_path,
(narrow);
ld.param.u{width} %r5, [answer];
cvt.u64.u32 %rd7, %r5;
st.param.b64 [wide], %rd9;
call.uni (answer),
ret_path,
(wide);
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
        run_integer_case(build, fixture(width), values,
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
        run_integer_case(build, ptx, values, expected, f'narrow {width}-bit stores')


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
        run_integer_case(build, ptx, values, expected, f'signed/unsigned {width}-bit loads into {container}')


if __name__ == '__main__':
    cases = dict(widening=widening, conversions=conversions, parameters=parameters,
                 stores=stores, loads=loads, type_paths=conversion_paths,
                 float_paths=lambda build: conversion_paths(build, float_path_cases))
    cases[sys.argv[2]](Path(sys.argv[1]).resolve())
