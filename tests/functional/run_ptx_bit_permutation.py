#!/usr/bin/env python3
"""Numerical Apple-GPU regression for typed shf/prmt and vector-store literals."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PTX = r'''
.version 7.0
.target sm_80
.address_size 64
.visible .entry bit_permutation(.param .u64 input, .param .u64 output, .param .u32 count) {
    .reg .b64 %rd<8>;
    .reg .b32 %r<14>;
    .reg .pred %p1;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [count];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd3, %r5, 16;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    ld.global.v4.b32 {%r6, %r7, %r8, %r9}, [%rd4];
    shf.l.wrap.b32 %r10, %r6, %r7, %r9;
    shf.r.wrap.b32 %r11, %r6, %r7, %r9;
    prmt.b32 %r12, %r6, %r7, %r8;
    st.global.v4.b32 [%rd5], {%r10, %r11, %r12, -1};
    st.global.b8 [%rd5+15], %r6;
    st.global.b16 [%rd5+12], %r7;
DONE:
    ret;
}
'''

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
    count = 65536  # Every selector, both byte signs, and all wrapped shift distances.
    source = (u32 * (count * 4))()
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
    result = (u32 * (count * 4 + 16))(*([0xa5a5a5a5] * (count * 4 + 16)))
    context, module, function = ptr(), ptr(), ptr()
    allocations = []
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-bit-permutation-') as work:
            ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
            ptx.write_text(PTX)
            subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
                            '--ptx-strict', '--entry', 'bit_permutation', '--emit=msl', '-o', str(msl)], check=True)
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'bit_permutation')
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
            print('NUMERICAL_PASS bit_permutation: 65536 selectors, shift boundaries, mixed stores, guards')
    finally:
        for allocation in allocations:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)

if __name__ == '__main__':
    main()
