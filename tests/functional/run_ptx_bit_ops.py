#!/usr/bin/env python3
"""Independent integer-reference checks for PTX permutations, fields, and tuples."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case

REFERENCE = Path(__file__).parent / 'reference'

def bit_permutation(build):
    count = 65536  # Every selector, both byte signs, and all wrapped shift distances.
    source = [0] * (count * 4)
    expected = []
    pairs = [(0x807f01ff, 0xfe028000), (0x12345678, 0x9abcdef0), (0, 0xffffffff)]
    shifts = [0, 1, 7, 16, 31, 32, 33, 63, 64, 0xffffffff]
    for i in range(count):
        a, b = pairs[i % len(pairs)]
        shift = shifts[i % len(shifts)]
        selector = i | 0xabcd0000  # High selector bits must be ignored.
        source[i*4:i*4+4] = a, b, selector, shift
        packed = a | (b << 32)
        left = ((packed << (shift & 31)) >> 32) & 0xffffffff
        right = (packed >> (shift & 31)) & 0xffffffff
        permuted = 0
        for lane in range(4):
            nibble = (selector >> (lane * 4)) & 15
            byte = (packed >> ((nibble & 7) * 8)) & 255
            if nibble & 8:
                byte = 255 if byte & 128 else 0
            permuted |= byte << (lane * 8)
        expected.extend((left, right, permuted, ((a & 255) << 24) | 0xff0000 | (b & 65535)))
    run_integer_case(build, (REFERENCE / 'ptx_bit_permutation.ptx').read_text(),
                     source, expected, 'bit_permutation', entry='bit_permutation',
                     word_bits=32, input_words=4, output_words=4)
    constant_bit_permutation(build)


def constant_bit_permutation(build):
    selectors = [(hex(nibble << (lane * 4)), nibble << (lane * 4))
                 for lane in range(4) for nibble in range(16)]
    selectors += [(hex(value), value) for value in
                  (0x0123, 0x7771, 0x7772, 0x7773, 0x7770, 0x3340, 0x5410, 0x7600,
                   0xabcd5410, 0xfedc, 0x89ab)]
    selectors += [('21520', 21520), ('052020', 0o52020),
                  ('0b0101010000010000U', 0x5410), ('0XFFFF5410u', 0xffff5410),
                  ('+21520', 21520), ('-1', -1), ('-0xABEF', -0xabef),
                  ('18446744073709551615U', (1 << 64) - 1)]
    words = len(selectors) + 3
    kernel = '''.version 7.1
.target sm_80
.address_size 64
.visible .entry immediate_permutation(.param .u64 input, .param .u64 output,
                                     .param .u32 count) {
 .reg .b64 %input, %output, %inptr, %outptr, %inoffset, %outoffset, %wide;
 .reg .b32 %a, %b, %answer, %n, %index, %block, %size, %thread;
 .reg .pred %done, %odd;
 ld.param.u64 %input, [input];
 ld.param.u64 %output, [output];
 ld.param.u32 %n, [count];
 mov.u32 %block, %ctaid.x;
 mov.u32 %size, %ntid.x;
 mov.u32 %thread, %tid.x;
 mad.lo.u32 %index, %block, %size, %thread;
 setp.ge.u32 %done, %index, %n;
 @%done bra DONE;
 mul.wide.u32 %inoffset, %index, 8;
 add.u64 %inptr, %input, %inoffset;
'''
    kernel += f' mul.wide.u32 %outoffset, %index, {words * 4};\n'
    kernel += ''' add.u64 %outptr, %output, %outoffset;
 ld.global.u32 %a, [%inptr];
 ld.global.u32 %b, [%inptr+4];
'''
    for slot, (spelling, _) in enumerate(selectors):
        kernel += (f' prmt.b32 %answer, %a, %b, {spelling};\n'
                   f' st.global.b32 [%outptr+{slot * 4}], %answer;\n')
    extra = len(selectors) * 4
    kernel += f''' cvt.u64.u32 %wide, %a;
 or.b64 %wide, %wide, 18446744069414584320U;
 prmt.b32 %answer, %wide, %b, 0x7543;
 st.global.b32 [%outptr+{extra}], %answer;
 prmt.b32 %answer, -1, 0x80000000U, 0xfedc;
 st.global.b32 [%outptr+{extra + 4}], %answer;
 and.b32 %thread, %index, 1;
 setp.ne.u32 %odd, %thread, 0;
 mov.b32 %answer, 0xdeadbeefU;
 @%odd bra PERMUTE;
 bra SAVE;
PERMUTE:
 prmt.b32 %answer, %a, %b, 0x0123;
SAVE:
 st.global.b32 [%outptr+{extra + 8}], %answer;
DONE:
 ret;
}}
'''

    # Byte indexing/sign checks are independent of the shift-and-mask lowering.
    def reference(a, b, selector):
        source_bytes = a.to_bytes(4, 'little') + b.to_bytes(4, 'little')
        result = bytearray()
        for position in range(4):
            nibble = (selector >> (position * 4)) & 15
            byte = source_bytes[nibble & 7]
            result.append((255 if byte >= 128 else 0) if nibble & 8 else byte)
        return int.from_bytes(result, 'little')

    values, expected = [], []
    for index in range(65):
        a = (0x807f01ff ^ (index * 0x9e3779b9)) & 0xffffffff
        b = (0xfe028000 + index * 0x6a09e667) & 0xffffffff
        values.extend((a, b))
        expected.extend(reference(a, b, value) for _, value in selectors)
        expected.extend((reference(a, b, 0x7543), reference(0xffffffff, 0x80000000, 0xfedc),
                         reference(a, b, 0x0123) if index & 1 else 0xdeadbeef))
    run_integer_case(build, kernel, values, expected, 'immediate bit permutation',
                     entry='immediate_permutation', word_bits=32, input_words=2,
                     output_words=words)


def bit_insert(build):
    # Independent bit-by-bit oracle following PTX's loop semantics.
    def insert(a, b, pos, length, width):
        pos, length = pos & 255, length & 255
        for bit in range(min(length, max(0, width - pos))):
            mask = 1 << (pos + bit)
            b = (b | mask) if ((a >> bit) & 1) else (b & ~mask)
        return b
    pairs = [(0, 0xffffffff), (0xffffffff, 0), (0x96a55a69, 0x1234abcd)]
    count = 65536 * len(pairs)
    source = [0] * (count * 4)
    expected = []
    for i in range(count):
        a, b = pairs[i // 65536]
        pos = ((i >> 8) & 255) | 0xffffff00
        length = (i & 255) | 0x12340000
        source[i*4:i*4+4] = a, b, pos, length
        wide = insert(a | (b << 32), b | (a << 32), pos, length, 64)
        expected.extend((insert(a, b, pos, length, 32), insert(a, b, 3, 13, 32),
                         wide & 0xffffffff, wide >> 32))
    run_integer_case(build, (REFERENCE / 'ptx_bit_insert.ptx').read_text(),
                     source, expected, 'bit_insert', entry='bit_insert',
                     word_bits=32, input_words=4, output_words=4)


def tuple_move(build):
    values = [i | ((i ^ 65535) << 16) for i in range(65536)]
    values += [0, 0xffffffff, 0x80000000, 0x00008000, 0x1234abcd]
    count = len(values)
    source = values
    expected = []
    for value in values:
        low, high = value & 65535, value >> 16
        expected.extend((value, low, high, high | (low << 16)))
    run_integer_case(build, (REFERENCE / 'ptx_tuple_move.ptx').read_text(),
                     source, expected, 'tuple_move', entry='tuple_move',
                     word_bits=32, input_words=1, output_words=4)


if __name__ == '__main__':
    cases = dict(bit_permutation=bit_permutation, bit_insert=bit_insert, tuple_move=tuple_move)
    cases[sys.argv[2]](Path(sys.argv[1]).resolve())
