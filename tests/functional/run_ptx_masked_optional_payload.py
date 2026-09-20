#!/usr/bin/env python3
"""Validate masked optional payloads without inventing absent values."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ptx_test_support import driver_api, expect_compile_failure


build = Path(sys.argv[1]).resolve()
reference = Path(__file__).parent / 'reference'
option128 = (reference / 'ptx_masked_optional_payload.ptx').read_text()
loop = (reference / 'ptx_masked_optional_loop.ptx').read_text()
os.environ['CUMETAL_TRACE_GPU'] = '1'
os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
api = driver_api(build)
ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
context = ptr()
api('cuInit', [u32], 0)
api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)


def run(source, entry, cases, output_words, label):
    """Compile once, then run scalar-argument cases with guarded output."""
    module, function, allocation = ptr(), ptr(), u64()
    guard = [0xa5a5a5a5] * 8
    host_type = u32 * (len(guard) * 2 + output_words)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-masked-payload-') as work:
            ptx = Path(work) / 'test.ptx'
            msl = Path(work) / 'test.metal'
            ptx.write_text(source)
            compiled = subprocess.run(
                [str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
                 '--ptx-strict', '--entry', entry, '--emit=msl', '-o', str(msl)],
                capture_output=True, text=True)
            if compiled.returncode:
                raise RuntimeError(f'{label}: PTX compilation failed\n{compiled.stderr}')
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p],
                c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p],
                c.byref(function), module, entry.encode())
            api('cuMemAlloc', [c.POINTER(u64), c.c_size_t],
                c.byref(allocation), c.sizeof(host_type))
            for arguments, expected in cases:
                host = host_type(*(guard + [0] * output_words + guard))
                api('cuMemcpyHtoD', [u64, ptr, c.c_size_t],
                    allocation, c.cast(host, ptr), c.sizeof(host))
                storage = [u64(allocation.value + len(guard) * c.sizeof(u32))]
                storage.extend(u64(value) for value in arguments)
                params = (ptr * (len(storage) + 1))(
                    *[c.cast(c.pointer(value), ptr) for value in storage], None)
                api('cuLaunchKernel', [ptr] + [u32] * 7 + [ptr, c.POINTER(ptr), ptr],
                    function, 1, 1, 1, 1, 1, 1, 0, None, params, None)
                api('cuCtxSynchronize', [])
                api('cuMemcpyDtoH', [ptr, u64, c.c_size_t],
                    c.cast(host, ptr), allocation, c.sizeof(host))
                actual = list(host)
                assert actual[:len(guard)] == guard and actual[-len(guard):] == guard
                assert actual[len(guard):len(guard) + output_words] == expected, (
                    label, arguments, actual, expected)
            print(f'NUMERICAL_PASS {label}: {len(cases)} cases, outputs, guards')
    finally:
        if allocation.value:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)


try:
    cases = [([0, 17], [0, 2]), ([1, 42], [1, 2]), ([1, 41], [0, 2])]
    run(option128, 'option128', cases, 2, '128-byte last mismatch')
    for index, name in [(0, 'first'), (64, 'middle')]:
        moved = option128.replace('cvt.u16.u32 %rs127, %r1;', 'mov.u16 %rs127, 42;')
        moved = moved.replace(f'mov.u16 %rs{index}, 42;',
                              f'cvt.u16.u32 %rs{index}, %r1;')
        run(moved, 'option128', cases, 2, f'128-byte {name} mismatch')

    reversed_and = option128.replace(
        'and.pred %p4, %p0, %p3;', 'and.pred %p4, %p3, %p0;')
    run(reversed_and, 'option128', cases, 2, 'commuted absorbing AND')

    through_call = option128.replace(
        '.visible .entry', '.func helper() { ret; }\n.visible .entry', 1)
    through_call = through_call.replace('JOIN:\n', 'JOIN:\ncall.uni helper, ();\n', 1)
    run(through_call, 'option128', cases, 2, 'mask across direct call')
    run(loop, 'masked_optional_loop', [([], [2])], 1,
        'present-absent-present loop')

    overwritten = option128.replace(
        'ld.param.u32 %r3, [last];',
        'ld.param.u32 %r3, [last];\nsetp.eq.u32 %p0, %r3, 42;')
    expect_compile_failure(build, overwritten, 'option128', 'undefined')
    escaped = option128.replace(
        'ld.param.u32 %r3, [last];',
        'ld.param.u32 %r3, [last];\n'
        'cvt.u32.u16 %r5, %rs0;\nst.global.u32 [%rd0+8], %r5;')
    expect_compile_failure(build, escaped, 'option128', 'undefined')
    print('NEGATIVE_PASS overwritten mask and escaped payload remain rejected')
finally:
    api('cuCtxDestroy', [ptr], context)
