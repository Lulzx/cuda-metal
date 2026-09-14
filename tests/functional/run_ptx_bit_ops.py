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
