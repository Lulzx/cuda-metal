#!/usr/bin/env python3
"""Numerical Apple-GPU regression for a byte-encoded PTX pointer initializer."""
import ctypes as c
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

PTX = r'''
.version 7.1
.target sm_80
.address_size 64
.global .align 8 .u8 alias[8] = {0XFF(table),0XFF00(table),0XFF0000(table),0XFF000000(table),0XFF00000000(table),0XFF0000000000(table),0XFF000000000000(table),0XFF00000000000000(table)};
.global .align 4 .b8 table[16] = {42,17,99,5,254,1,128,63,9,8,7,6,255,0,127,128};
.visible .entry relocated_table(.param .u64 input, .param .u64 output, .param .u32 count) {
 .reg .b64 %rd<9>;
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
 add.u64 %rd5, %rd2, %rd3;
 ld.global.u32 %r6, [%rd4];
 ld.global.nc.u64 %rd6, [alias];
 mul.wide.u32 %rd7, %r6, 4;
 add.u64 %rd8, %rd6, %rd7;
 ld.global.u32 %r7, [%rd8];
 st.global.u32 [%rd5], %r7;
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
    ptx_source = PTX
    table_bytes = bytes([42,17,99,5,254,1,128,63,9,8,7,6,255,0,127,128])
    if len(sys.argv) > 2 and sys.argv[2] == '--typed':
        ptx_source = re.sub(r'\.global \.align 8 \.u8 alias\[8\] = \{[^}]+\};',
                            '.global .align 8 .u64 alias[1] = {table};', ptx_source)
        ptx_source = ptx_source.replace('ld.global.nc.u64', 'ld.global.nc.b64')
    elif len(sys.argv) > 2:
        # Keep the supplied module byte-for-byte as a prefix; append only a
        # diagnostic entry that reads its real table through its real alias.
        original = Path(sys.argv[2]).read_text()
        match = re.search(r"\.global \.align 8 \.b8 (\w*ED25519_BASEPOINT_TABLE_INNER_DOC_HIDDEN)\[30720\] = \{([^}]+)\};", original)
        if not match:
            raise RuntimeError('expected full-miner Ed25519 table declaration')
        table_name = match[1]
        table_bytes = bytes(int(x.strip()) for x in match[2].split(','))
        if len(table_bytes) > 30720:
            raise RuntimeError('table initializer exceeds declared size')
        table_bytes = table_bytes.ljust(30720, b'\0')
        alias = re.search(r"\.global \.align 8 \.u8 (\w*23ED25519_BASEPOINT_TABLE)\[8\]", original)
        if not alias:
            raise RuntimeError('expected full-miner Ed25519 pointer declaration')
        entry = PTX[PTX.index('.visible .entry'):].replace('[alias]', '[' + alias[1] + ']')
        ptx_source = original + '\n' + entry
    table = [int.from_bytes(table_bytes[i:i+4], 'little') for i in range(0, len(table_bytes), 4)]
    count = max(257, len(table) + 1)
    source = (u32 * count)(*[(i * 7 + 1) % len(table) for i in range(count)])
    expected = [table[i] for i in source]
    result = (u32 * (count + 16))(*([0xa5a5a5a5] * (count + 16)))
    context, module, function = ptr(), ptr(), ptr()
    allocations = []
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-relocated-table-') as work:
            ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
            ptx.write_text(ptx_source)
            subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
                            '--ptx-strict', '--entry', 'relocated_table', '--emit=msl', '-o', str(msl)], check=True)
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'relocated_table')
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
            print(f'NUMERICAL_PASS relocated_table: {count} runtime-selected reads from {len(table_bytes)} table bytes through relocated pointer; guards intact')
    finally:
        for allocation in allocations:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)

if __name__ == '__main__':
    main()
