#!/usr/bin/env python3
"""Numerical Apple-GPU regression for two-halfword mov.b32 tuples."""
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
.visible .entry tuple_move(.param .u64 input, .param .u64 output, .param .u32 count) {
 .reg .b64 %rd<7>;
 .reg .b32 %r<12>;
 .reg .b16 %rs<7>;
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
 mul.wide.u32 %rd3, %r5, 4;
 add.u64 %rd4, %rd1, %rd3;
 mul.wide.u32 %rd6, %r5, 16;
 add.u64 %rd5, %rd2, %rd6;
 ld.global.u32 %r6, [%rd4];
 cvt.u16.u32 %rs1, %r6;
 shr.u32 %r7, %r6, 16;
 cvt.u16.u32 %rs2, %r7;
 mov.b32 %r8, {%rs1, %rs2};
 st.global.u32 [%rd5], %r8;
 mov.b32 {%rs3, %rs4}, %r6;
 cvt.u32.u16 %r9, %rs3;
 cvt.u32.u16 %r10, %rs4;
 st.global.u32 [%rd5+4], %r9;
 st.global.u32 [%rd5+8], %r10;
 mov.b32 {_, %rs5}, %r6;
 mov.b32 {%rs6, _}, %r6;
 mov.b32 %r11, {%rs5, %rs6};
 st.global.u32 [%rd5+12], %r11;
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
    values = [i | ((i ^ 65535) << 16) for i in range(65536)]
    values += [0, 0xffffffff, 0x80000000, 0x00008000, 0x1234abcd]
    count = len(values)
    source = (u32 * count)(*values)
    expected = []
    for value in values:
        low, high = value & 65535, value >> 16
        expected.extend((value, low, high, high | (low << 16)))
    result = (u32 * (len(expected) + 16))(*([0xa5a5a5a5] * (len(expected) + 16)))
    context, module, function = ptr(), ptr(), ptr()
    allocations = []
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-tuple-move-') as work:
            ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
            ptx.write_text(PTX)
            subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
                            '--ptx-strict', '--entry', 'tuple_move', '--emit=msl', '-o', str(msl)], check=True)
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'tuple_move')
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
            print('NUMERICAL_PASS tuple_move: 65541 inputs; packing, independent unpacking, both sink lanes, guards')
    finally:
        for allocation in allocations:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)

if __name__ == '__main__':
    main()
