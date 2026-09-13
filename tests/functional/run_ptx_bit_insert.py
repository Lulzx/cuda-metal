#!/usr/bin/env python3
"""Numerical Apple-GPU regression for PTX bfi.b32/b64."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PTX = r'''
.version 7.1
.target sm_80
.address_size 64
.visible .entry bit_insert(.param .u64 input, .param .u64 output, .param .u32 count) {
 .reg .b64 %rd<9>;
 .reg .b32 %r<12>;
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
 ld.global.v4.u32 {%r6, %r7, %r8, %r9}, [%rd4];
 bfi.b32 %r10, %r6, %r7, %r8, %r9;
 bfi.b32 %r11, %r6, %r7, 3, 13;
 mov.b64 %rd6, {%r6, %r7};
 mov.b64 %rd7, {%r7, %r6};
 bfi.b64 %rd8, %rd6, %rd7, %r8, %r9;
 st.global.u32 [%rd5], %r10;
 st.global.u32 [%rd5+4], %r11;
 st.global.u64 [%rd5+8], %rd8;
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
    # Independent bit-by-bit oracle following PTX's loop semantics.
    def insert(a, b, pos, length, width):
        pos, length = pos & 255, length & 255
        for bit in range(min(length, max(0, width - pos))):
            mask = 1 << (pos + bit)
            b = (b | mask) if ((a >> bit) & 1) else (b & ~mask)
        return b
    pairs = [(0, 0xffffffff), (0xffffffff, 0), (0x96a55a69, 0x1234abcd)]
    count = 65536 * len(pairs)
    source = (u32 * (count * 4))()
    expected = []
    for i in range(count):
        a, b = pairs[i // 65536]
        pos = ((i >> 8) & 255) | 0xffffff00
        length = (i & 255) | 0x12340000
        source[i*4:i*4+4] = a, b, pos, length
        wide = insert(a | (b << 32), b | (a << 32), pos, length, 64)
        expected.extend((insert(a, b, pos, length, 32), insert(a, b, 3, 13, 32),
                         wide & 0xffffffff, wide >> 32))
    result = (u32 * (count * 4 + 16))(*([0xa5a5a5a5] * (count * 4 + 16)))
    context, module, function = ptr(), ptr(), ptr()
    allocations = []
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-bit-insert-') as work:
            ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
            ptx.write_text(PTX)
            subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
                            '--ptx-strict', '--entry', 'bit_insert', '--emit=msl', '-o', str(msl)], check=True)
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'bit_insert')
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
            print('NUMERICAL_PASS bit_insert: all 65536 position/length pairs x 3 patterns; b32/b64 and immediate insertion; guards')
    finally:
        for allocation in allocations:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)

if __name__ == '__main__':
    main()
