#!/usr/bin/env python3
"""Numerical Apple-GPU regression for 64-bit high-half signed and unsigned multiplication."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PTX = (Path(__file__).parent / 'reference/ptx_mul_hi_64.ptx').read_text()

def main():
    build = Path(sys.argv[1]).resolve()
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
    mask = (1 << 64)-1
    boundary = [0, 1, 2, (1 << 32)-1, 1 << 32, (1 << 32)+1,
                (1 << 63)-1, 1 << 63, (1 << 63)+1, mask-1, mask]
    pairs = [(a, b) for a in boundary for b in boundary]
    state = 0x123456789abcdef
    for _ in range(4096):
        state = (state * 6364136223846793005 + 1) & mask
        a = state
        state = (state * 6364136223846793005 + 1) & mask
        pairs.append((a, state))
    count = len(pairs)
    source = (u64 * (2 * count))(*[v for pair in pairs for v in pair])
    expected = []
    for a, b in pairs:
        sa = a if a < (1 << 63) else a - (1 << 64)
        sb = b if b < (1 << 63) else b - (1 << 64)
        expected.extend(((a*b) >> 64, (a*7544311872078572213) >> 64,
                         ((sa*sb) >> 64) & mask, ((sa*-1) >> 64) & mask))
    result = (u64 * (len(expected) + 16))(*([0xa5a5a5a5] * (len(expected) + 16)))
    context, module, function = ptr(), ptr(), ptr()
    allocations = []
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-mul-hi-64-') as work:
            ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
            ptx.write_text(PTX)
            subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
                            '--ptx-strict', '--entry', 'mul_hi_64', '--emit=msl', '-o', str(msl)], check=True)
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'mul_hi_64')
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
            print('NUMERICAL_PASS mul_hi_64: 4217 input pairs; unsigned/signed high products and immediate operands, guards')
    finally:
        for allocation in allocations:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)

if __name__ == '__main__':
    main()
