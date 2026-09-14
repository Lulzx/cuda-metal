#!/usr/bin/env python3
"""Tail-call and aggregate ABI checks with independent integer oracles."""
from pathlib import Path
import random
import sys
from ptx_test_support import expect_compile_failure, run_integer_case

REFERENCE = Path(__file__).parent / 'reference'


def run_tail_case(build, source, values, expected, label, input_words=1):
    run_integer_case(build, source, values, expected, label, entry='tail_probe',
                     input_words=input_words)

def scalar_cases(build):
    values = list(range(64)) + [255, 1024]
    run_tail_case(build, (REFERENCE / 'ptx_scalar_tail_call.ptx').read_text(), values,
             [word for _ in values for word in (0, 81985529216486895)],
             'tail count: zero through 1024 iterations')

    # Nonrecursive immediate return stores; read via both b64 and independent
    # b32 fields so complementary packing/unpacking mistakes cannot cancel.
    template = (REFERENCE / 'ptx_scalar_tail_call.ptx').read_text()
    start, entry_start = template.index('.func'), template.index('.visible .entry')
    for first, second in [(0, 1), (-1, -9223372036854775808),
                          (81985529216486895, -81985529216486896)]:
        helper = ('.func (.param .align 8 .b8 retval[16]) tail_count() {\n'
                  f'st.param.b64 [retval], {first};\nst.param.b64 [retval+8], {second};\nret;\n}}\n')
        ptx = template[:start] + helper + template[entry_start:]
        ptx = ptx.replace('st.param.b64 [arg], %rd7;', '')
        ptx = ptx.replace('call.uni (result), tail_count, (arg);', 'call.uni (result), tail_count, ();')
        expected = [first & ((1 << 64) - 1), second & ((1 << 64) - 1)]
        run_tail_case(build, ptx, [0], expected, f'immediate aggregate {first}, {second}')
        vector_ptx = ptx.replace('.align 8 .b8', '.align 16 .b8').replace(
            f'st.param.b64 [retval], {first};\nst.param.b64 [retval+8], {second};',
            f'st.param.v2.b64 [retval], {{{first}, {second}}};')
        vector_ptx = vector_ptx.replace(
            'ld.param.b64 %rd8, [result];\nld.param.b64 %rd9, [result+8];',
            'ld.param.v2.b64 {%rd8, %rd9}, [result];')
        run_tail_case(build, vector_ptx, [0], expected, f'vector parameter lanes {first}, {second}')

        ptx = ptx.replace('.reg .b32 %r<5>;', '.reg .b32 %r<9>;')
        begin = ptx.index('ld.param.b64 %rd8, [result];')
        end = ptx.index('DONE:', begin)
        ptx = ptx[:begin] + ''.join(
            f'ld.param.b32 %r{5+i}, [result+{4*i}];\n'
            f'st.global.b32 [%rd6+{4*i}], %r{5+i};\n' for i in range(4)) + ptx[end:]
        run_tail_case(build, ptx, [0], expected, f'independent u32 fields {first}, {second}')

    # Independent SplitMix64 seed expansion oracle, modulo 2**64.
    mask = (1 << 64) - 1
    def mix(value):
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
        return value ^ (value >> 31)
    rng = random.Random(7219)
    seeds = [0, 1, mask, 1 << 63, 0x61c8864680b583eb]
    seeds += [rng.getrandbits(64) for _ in range(256)]
    expected = [mix((seed + step * 0x9e3779b97f4a7c15) & mask)
                for seed in seeds for step in (1, 2)]
    run_tail_case(build, (REFERENCE / 'ptx_xoroshiro_seed_tail_call.ptx').read_text(),
             seeds, expected, 'original LLVM 7 xoroshiro seed expansion')


def local_cases(build):
    ptx = (Path(__file__).parent / 'reference/ptx_local_buffer_tail_call.ptx').read_text()
    rng = random.Random(1907)
    pairs = [(0, 0), (1, 0), (0, 1), ((1 << 64)-1, (1 << 64)-1)]
    pairs += [(1 << bit, 0) for bit in range(64)]
    pairs += [(0, 1 << bit) for bit in range(64)]
    pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(256)]
    # Independent seed_from_u64(0) oracle, rather than copying PTX constants.
    mask = (1 << 64) - 1
    def splitmix(value):
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
        return value ^ (value >> 31)
    fallback = tuple(splitmix((i * 0x9e3779b97f4a7c15) & mask) for i in (1, 2))
    expected = [word for pair in pairs for word in (fallback if pair == (0, 0) else pair)]
    run_tail_case(build, ptx, [word for pair in pairs for word in pair], expected,
             'local frame preserves input and zero fallback', input_words=2)

    # Exact source fragments ensure these cases change the intended proof edge.
    mutations = [
        ('[%rd2+15]', '[%rd2+14]'),  # incomplete input consumption
        ('[%rd2+15]', '[%rd2+16]'),  # out-of-bounds input
        ('st.local.v2.b64', 'st.local.b64'),  # incomplete replacement
        ('[%rd49]', '[%rd49+8]'),  # replacement must cover the same full frame
        ('add.u64 \t%rd48, %SP, 0;', 'add.u64 \t%rd48, %SP, 8;'),
        ('or.b64 \t%rd51, %rd23, %rd12;', 'or.b64 \t%rd51, %SP, %rd12;'),  # escaping address bits
        ('st.local.v2.b64', '@%p1 st.local.v2.b64'),
        ('ld.param.v2.b64 \t{%rd50, %rd51}, [retval0];',
         'ld.param.v2.b64 \t{%rd50, %rd51}, [retval0];\nadd.u64 %rd50, %rd50, 1;'),
        ('ld.param.v2.b64 \t{%rd50, %rd51}, [retval0];',
         'ld.param.v2.b64 \t{%rd50, %rd51}, [retval0];\nld.local.b8 %rd3, [%rd2];'),
    ]
    for old, new in mutations:
        assert old in ptx, old
        expect_compile_failure(build, ptx.replace(old, new, 1), 'tail_probe',
                               'recursive PTX device-call cycle')

if __name__ == '__main__':
    build = Path(sys.argv[1]).resolve()
    (local_cases if sys.argv[2] == 'local' else scalar_cases)(build)
