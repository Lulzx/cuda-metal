#!/usr/bin/env python3
"""Numerical regression for narrow integer loads returning through 32-bit slots."""
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

.func (.param .b32 returned) narrow_u8(.param .b32 input) {
 .local .align 2 .b8 scratch[2];
 .reg .b64 %rd1;
 .reg .b16 %rs1;
 .reg .b32 %r1;
 mov.u64 %rd1, scratch;
 ld.param.u8 %rs1, [input];
 st.local.u8 [%rd1], %rs1;
 ld.volatile.local.u8 %r1, [%rd1];
 st.param.b32 [returned], %r1;
 ret;
}

.func (.param .b32 returned) narrow_b8(.param .b32 input) {
 .local .align 2 .b8 scratch[2];
 .reg .b64 %rd1;
 .reg .b16 %rs1;
 .reg .b32 %r1;
 mov.u64 %rd1, scratch;
 ld.param.u8 %rs1, [input];
 st.local.u8 [%rd1], %rs1;
 ld.volatile.local.b8 %r1, [%rd1];
 st.param.b32 [returned], %r1;
 ret;
}

.func (.param .b32 returned) narrow_s8(.param .b32 input) {
 .local .align 2 .b8 scratch[2];
 .reg .b64 %rd1;
 .reg .b16 %rs1;
 .reg .b32 %r1;
 mov.u64 %rd1, scratch;
 ld.param.u8 %rs1, [input];
 st.local.u8 [%rd1], %rs1;
 ld.volatile.local.s8 %r1, [%rd1];
 st.param.b32 [returned], %r1;
 ret;
}

.func (.param .b32 returned) narrow_u16(.param .b32 input) {
 .local .align 2 .b8 scratch[2];
 .reg .b64 %rd1;
 .reg .b16 %rs1;
 .reg .b32 %r1;
 mov.u64 %rd1, scratch;
 ld.param.u16 %rs1, [input];
 st.local.u16 [%rd1], %rs1;
 ld.volatile.local.u16 %r1, [%rd1];
 st.param.b32 [returned], %r1;
 ret;
}

.func (.param .b32 returned) narrow_b16(.param .b32 input) {
 .local .align 2 .b8 scratch[2];
 .reg .b64 %rd1;
 .reg .b16 %rs1;
 .reg .b32 %r1;
 mov.u64 %rd1, scratch;
 ld.param.u16 %rs1, [input];
 st.local.u16 [%rd1], %rs1;
 ld.volatile.local.b16 %r1, [%rd1];
 st.param.b32 [returned], %r1;
 ret;
}

.func (.param .b32 returned) narrow_s16(.param .b32 input) {
 .local .align 2 .b8 scratch[2];
 .reg .b64 %rd1;
 .reg .b16 %rs1;
 .reg .b32 %r1;
 mov.u64 %rd1, scratch;
 ld.param.u16 %rs1, [input];
 st.local.u16 [%rd1], %rs1;
 ld.volatile.local.s16 %r1, [%rd1];
 st.param.b32 [returned], %r1;
 ret;
}

.visible .entry narrow_loads(.param .u64 input, .param .u64 output, .param .u32 count) {
 .reg .b64 %rd<8>;
 .reg .b32 %r<8>;
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
 ld.global.u32 %r6, [%rd4];
 mul.wide.u32 %rd5, %r5, 24;
 add.u64 %rd6, %rd2, %rd5;
 {
 .param .b32 arg;
 .param .b32 result;
 st.param.b32 [arg], %r6;
 call.uni (result), narrow_u8, (arg);
 ld.param.b32 %r7, [result];
 st.global.u32 [%rd6+0], %r7;
 }
 {
 .param .b32 arg;
 .param .b32 result;
 st.param.b32 [arg], %r6;
 call.uni (result), narrow_b8, (arg);
 ld.param.b32 %r7, [result];
 st.global.u32 [%rd6+4], %r7;
 }
 {
 .param .b32 arg;
 .param .b32 result;
 st.param.b32 [arg], %r6;
 call.uni (result), narrow_s8, (arg);
 ld.param.b32 %r7, [result];
 st.global.u32 [%rd6+8], %r7;
 }
 {
 .param .b32 arg;
 .param .b32 result;
 st.param.b32 [arg], %r6;
 call.uni (result), narrow_u16, (arg);
 ld.param.b32 %r7, [result];
 st.global.u32 [%rd6+12], %r7;
 }
 {
 .param .b32 arg;
 .param .b32 result;
 st.param.b32 [arg], %r6;
 call.uni (result), narrow_b16, (arg);
 ld.param.b32 %r7, [result];
 st.global.u32 [%rd6+16], %r7;
 }
 {
 .param .b32 arg;
 .param .b32 result;
 st.param.b32 [arg], %r6;
 call.uni (result), narrow_s16, (arg);
 ld.param.b32 %r7, [result];
 st.global.u32 [%rd6+20], %r7;
 }
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
    count = 65536
    source = (u32 * count)(*[i | 0xa5a50000 for i in range(count)])
    expected = []
    for i in range(count):
        byte = i & 255
        signed_byte = byte if byte < 128 else byte - 256
        signed_half = i if i < 32768 else i - 65536
        expected.extend((byte, byte, signed_byte & 0xffffffff, i, i, signed_half & 0xffffffff))
    result = (u32 * (len(expected) + 16))(*([0xa5a5a5a5] * (len(expected) + 16)))
    context, module, function = ptr(), ptr(), ptr()
    allocations = []
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-narrow-loads-') as work:
            ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
            ptx.write_text(PTX)
            subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
                            '--ptx-strict', '--entry', 'narrow_loads', '--emit=msl', '-o', str(msl)], check=True)
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'narrow_loads')
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
            print('NUMERICAL_PASS narrow_loads: 65536 inputs x 6 helper return paths; unsigned/bit zero extension, signed extension, guards')
    finally:
        for allocation in allocations:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)

if __name__ == '__main__':
    main()
