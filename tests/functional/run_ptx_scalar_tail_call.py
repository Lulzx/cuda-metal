#!/usr/bin/env python3
"""Numerical Apple-GPU regression for scalar tail recursion normalization."""
import ctypes as c
import os
import random
from pathlib import Path
import subprocess
import sys
import tempfile

REFERENCE = Path(__file__).parent / 'reference'

def run_case(build, ptx_source, values, expected, label):
    os.environ['CUMETAL_TRACE_GPU'] = '1'
    os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
    lib = c.CDLL(str(build / 'libcumetal.dylib'))
    def api(name, types, *args):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, c.c_int
        result = fn(*args)
        if result:
            raise RuntimeError(f'{name} failed: {result}')
    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
    count = len(values)
    source = (u64 * count)(*values)
    result = (u64 * (len(expected) + 16))(*([0xa5a5a5a5] * (len(expected) + 16)))
    context, module, function = ptr(), ptr(), ptr()
    allocations = []
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-scalar-tail-') as work:
            ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
            ptx.write_text(ptx_source)
            subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
                            '--ptx-strict', '--entry', 'tail_probe', '--emit=msl', '-o', str(msl)], check=True)
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'tail_probe')
            for data in (source, result):
                allocation = u64()
                api('cuMemAlloc', [c.POINTER(u64), c.c_size_t], c.byref(allocation), c.sizeof(data))
                allocations.append(allocation)
                api('cuMemcpyHtoD', [u64, ptr, c.c_size_t], allocation, c.cast(data, ptr), c.sizeof(data))
            count_storage = u64(count)  # Current classifier reads eight bytes for scalars.
            args = (ptr * 4)(*[c.cast(c.pointer(x), ptr) for x in (*allocations, count_storage)], None)
            api('cuLaunchKernel', [ptr] + [u32]*7 + [ptr, c.POINTER(ptr), ptr],
                function, count // 64 + 1, 1, 1, 64, 1, 1, 0, None, args, None)
            api('cuCtxSynchronize', [])
            api('cuMemcpyDtoH', [ptr, u64, c.c_size_t], c.cast(result, ptr), allocations[1], c.sizeof(result))
            for i, value in enumerate(expected):
                if result[i] != value:
                    raise RuntimeError(f'word {i}: got {result[i]:08x}, expected {value:08x}')
            assert list(result)[len(expected):] == [0xa5a5a5a5] * 16, 'tail guard overwritten'
            print(f'NUMERICAL_PASS {label}: {count} inputs, aggregate return, guards')
    finally:
        for allocation in allocations:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)

def main():
    build = Path(sys.argv[1]).resolve()
    values = list(range(64)) + [255, 1024]
    run_case(build, (REFERENCE / 'ptx_scalar_tail_call.ptx').read_text(), values,
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
        run_case(build, ptx, [0], expected, f'immediate aggregate {first}, {second}')
        ptx = ptx.replace('.reg .b32 %r<5>;', '.reg .b32 %r<9>;')
        begin = ptx.index('ld.param.b64 %rd8, [result];')
        end = ptx.index('DONE:', begin)
        ptx = ptx[:begin] + ''.join(
            f'ld.param.b32 %r{5+i}, [result+{4*i}];\n'
            f'st.global.b32 [%rd6+{4*i}], %r{5+i};\n' for i in range(4)) + ptx[end:]
        run_case(build, ptx, [0], expected, f'independent u32 fields {first}, {second}')

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
    run_case(build, (REFERENCE / 'ptx_xoroshiro_seed_tail_call.ptx').read_text(),
             seeds, expected, 'original LLVM 7 xoroshiro seed expansion')


if __name__ == '__main__':
    main()
