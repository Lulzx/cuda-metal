#!/usr/bin/env python3
"""Independent integer-reference checks for PTX permutations, fields, and tuples."""
from pathlib import Path
import sys
from ptx_test_support import expect_compile_failure, run_integer_case

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
    constant_funnel_shift(build)


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


KERNEL_HEAD = '''.version 7.1
.target sm_80
.address_size 64
.visible .entry {entry}(.param .u64 input, .param .u64 output, .param .u32 count) {{
 .reg .b64 %input, %output, %inptr, %outptr, %inoffset, %outoffset, %wa, %wb, %wide;
 .reg .b32 %a, %b, %answer, %n, %index, %block, %size, %thread;
 .reg .pred %done;
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
 mul.wide.u32 %outoffset, %index, {stride};
 add.u64 %outptr, %output, %outoffset;
 ld.global.u32 %a, [%inptr];
 ld.global.u32 %b, [%inptr+4];
 mov.b64 %wa, {{%a, %b}};
 mov.b64 %wb, {{%b, %a}};
'''
KERNEL_TAIL = '''DONE:
 ret;
}
'''


def pair_values():
    values = []
    for index in range(65):
        values.append(((0x807f01ff ^ (index * 0x9e3779b9)) & 0xffffffff,
                       (0xfe028000 + index * 0x6a09e667) & 0xffffffff))
    values += [(0, 0xffffffff), (0xffffffff, 0), (0x80000000, 1)]
    return values


def constant_funnel_shift(build):
    # Constant counts take a two-shift lowering; the register form is covered above.
    counts = list(range(34)) + [63, 64, 0xffffffff]
    # (label, a operand, b operand): literal inputs must stay unsigned 32-bit.
    sources = [('%a', '%b'), ('-1', '%b'), ('%a', '-2'), ('0x80000000', '%a')]
    cases = [(left, count, pair) for pair in sources for count in counts for left in (True, False)]
    stride = len(cases) * 4
    kernel = KERNEL_HEAD.format(entry='constant_funnel_shift', stride=stride)
    for slot, (left, count, (sa, sb)) in enumerate(cases):
        kernel += (f' shf.{"l" if left else "r"}.wrap.b32 %answer, {sa}, {sb}, {count};\n'
                   f' st.global.b32 [%outptr+{slot * 4}], %answer;\n')
    kernel += KERNEL_TAIL
    literal = {'-1': 0xffffffff, '-2': 0xfffffffe, '0x80000000': 0x80000000}
    values, expected = [], []
    for a, b in pair_values():
        values.extend((a, b))
        for left, count, (sa, sb) in cases:
            x = a if sa == '%a' else b if sa == '%b' else literal[sa]
            y = a if sb == '%a' else b if sb == '%b' else literal[sb]
            packed = x | (y << 32)
            n = count & 31
            expected.append(((packed << n) >> 32) & 0xffffffff if left else (packed >> n) & 0xffffffff)
    run_integer_case(build, kernel, values, expected, 'constant funnel shift',
                     entry='constant_funnel_shift', word_bits=32, input_words=2,
                     output_words=len(cases))


def constant_bit_insert(build):
    def insert(a, b, pos, length, width):
        pos, length = pos & 255, length & 255
        for bit in range(min(length, max(0, width - pos))):
            mask = 1 << (pos + bit)
            b = (b | mask) if ((a >> bit) & 1) else (b & ~mask)
        return b
    fields = [(0, 0), (0, 1), (0, 8), (3, 13), (8, 16), (16, 16), (24, 8), (24, 16), (31, 1),
              (31, 2), (0, 32), (0, 33), (1, 32), (32, 1), (33, 4), (40, 8), (48, 16),
              (56, 8), (63, 1), (63, 9), (0, 64), (0, 65), (64, 1), (0, 255), (255, 1),
              (256 + 8, 256 + 4)]  # Only the low eight bits of each count matter.
    # (width, a operand, b operand, python a, python b)
    cases = []
    for pos, length in fields:
        cases.append((32, '%a', '%b', pos, length))
        cases.append((64, '%wa', '%wb', pos, length))
    cases += [(32, '-1', '%b', 4, 8), (32, '%a', '-1', 4, 8), (64, '-1', '%wb', 20, 24),
              (64, '%wa', '-1', 40, 16)]
    stride = len(cases) * 8
    kernel = KERNEL_HEAD.format(entry='constant_bit_insert', stride=stride)
    for slot, (width, sa, sb, pos, length) in enumerate(cases):
        if width == 32:
            kernel += (f' bfi.b32 %answer, {sa}, {sb}, {pos}, {length};\n'
                       f' cvt.u64.u32 %wide, %answer;\n')
        else:
            kernel += f' bfi.b64 %wide, {sa}, {sb}, {pos}, {length};\n'
        kernel += f' st.global.b64 [%outptr+{slot * 8}], %wide;\n'
    kernel += KERNEL_TAIL
    values, expected = [], []
    for a, b in pair_values():
        values.extend((a, b))
        wa, wb = a | (b << 32), b | (a << 32)
        for width, sa, sb, pos, length in cases:
            full = (1 << width) - 1
            x = {'%a': a, '%b': b, '%wa': wa, '%wb': wb}.get(sa, full)
            y = {'%a': a, '%b': b, '%wa': wa, '%wb': wb}.get(sb, full)
            result = insert(x, y, pos, length, width)
            expected.extend((result & 0xffffffff, result >> 32))
    run_integer_case(build, kernel, values, expected, 'constant bit insert',
                     entry='constant_bit_insert', word_bits=32, input_words=2,
                     output_words=len(cases) * 2)


def logic3(build):
    # Every truth table, plus literal inputs and bit reversal, against Python.
    tables = list(range(256))
    literal_cases = [('-1', '%b', '%a', 0xe8), ('%a', '0x0f0f0f0f', '%b', 0xca),
                     ('%a', '%b', '0', 0x96)]
    slots = len(tables) + len(literal_cases) + 1 + 2
    kernel = KERNEL_HEAD.format(entry='logic3', stride=slots * 4)
    kernel = kernel.replace('.reg .b32 %a, %b,', '.reg .b32 %c, %a, %b,')
    kernel += ' xor.b32 %c, %a, 0x5a5a00ff;\n'
    for slot, table in enumerate(tables):
        kernel += (f' lop3.b32 %answer, %a, %b, %c, {table};\n'
                   f' st.global.b32 [%outptr+{slot * 4}], %answer;\n')
    slot = len(tables)
    for x, y, z, table in literal_cases:
        kernel += (f' lop3.b32 %answer, {x}, {y}, {z}, {hex(table)};\n'
                   f' st.global.b32 [%outptr+{slot * 4}], %answer;\n')
        slot += 1
    kernel += (f' brev.b32 %answer, %a;\n st.global.b32 [%outptr+{slot * 4}], %answer;\n'
               f' brev.b64 %wide, %wa;\n st.global.b64 [%outptr+{slot * 4 + 4}], %wide;\n')
    kernel += KERNEL_TAIL

    def lop3(x, y, z, table):
        result = 0
        for bit in range(32):
            index = (((x >> bit) & 1) << 2) | (((y >> bit) & 1) << 1) | ((z >> bit) & 1)
            result |= ((table >> index) & 1) << bit
        return result

    def reverse(value, width):
        return int(format(value, f'0{width}b')[::-1], 2)

    literal = {'-1': 0xffffffff, '0x0f0f0f0f': 0x0f0f0f0f, '0': 0}
    values, expected = [], []
    for a, b in pair_values():
        values.extend((a, b))
        c = a ^ 0x5a5a00ff
        registers = {'%a': a, '%b': b}
        expected.extend(lop3(a, b, c, table) for table in tables)
        for x, y, z, table in literal_cases:
            pick = lambda s: registers.get(s, literal.get(s))
            expected.append(lop3(pick(x), pick(y), pick(z), table))
        wide = reverse(a | (b << 32), 64)
        expected.extend((reverse(a, 32), wide & 0xffffffff, wide >> 32))
    run_integer_case(build, kernel, values, expected, 'logic3 and brev', entry='logic3',
                     word_bits=32, input_words=2, output_words=slots)
    # PTX requires a constant table; a register or predicate-output form refuses.
    for bad in (' lop3.b32 %answer, %a, %b, %c, %n;\n',
                ' lop3.or.b32 %answer, %a, %b, %c, 0x96;\n',
                ' brev.b16 %answer, %a;\n'):
        source = (KERNEL_HEAD.format(entry='bad', stride=4).replace('.reg .b32 %a,', '.reg .b32 %c, %a,')
                  + ' mov.b32 %c, %a;\n' + bad + KERNEL_TAIL)
        expect_compile_failure(build, source, 'bad', 'typed PTX')


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
    constant_bit_insert(build)


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
    cases = dict(bit_permutation=bit_permutation, bit_insert=bit_insert, tuple_move=tuple_move,
                 logic3=logic3)
    cases[sys.argv[2]](Path(sys.argv[1]).resolve())
