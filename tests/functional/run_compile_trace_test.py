#!/usr/bin/env python3
"""Real runtime trace stages, cache reuse, failures, and verified GPU output."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compile_trace_support import begin_stages, parse_spans

SOURCE = '''#include <metal_stdlib>
using namespace metal;
kernel void trace_probe(device uint* output [[buffer(0)]],
                        uint lane [[thread_position_in_grid]]) {
    output[lane] = lane * 3u + 7u;
}
'''


def child(runtime, mode):
    library = c.CDLL(str(runtime))
    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64

    def api(name, types, *arguments, expected=0):
        function = getattr(library, name)
        function.argtypes, function.restype = types, c.c_int
        result = function(*arguments)
        assert result == expected, (name, result, expected)
        return result

    library.cuInit.argtypes, library.cuInit.restype = [u32], c.c_int
    status = library.cuInit(0)
    if status == 100:
        print('SKIP: no supported Metal device')
        return 77
    assert status == 0, ('cuInit', status)
    context, module, function, allocation = ptr(), ptr(), ptr(), u64()
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-trace-runtime-') as work:
            source = Path(work) / 'probe.metal'
            source.write_text('not valid Metal source' if mode == 'invalid' else SOURCE)
            Path(str(source) + '.cumetal-abi').write_text(
                'CUMETAL_ABI_V1\nkernel trace_probe\narg buffer 8\nshared 0\n')
            name = b'absent_kernel' if mode == 'missing' else b'trace_probe'
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(source))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p],
                c.byref(function), module, name)
            maximum = c.c_int()
            # This property query uses the existing preparation path without dispatch.
            # cuModuleLoad/cuModuleGetFunction alone are not compilation evidence.
            api('cuFuncGetAttribute', [c.POINTER(c.c_int), c.c_int, ptr],
                c.byref(maximum), 0, function, expected=0 if mode == 'success' else 1)
            if mode != 'success':
                print('EXPECTED_FAILURE', mode, 'CUDA_ERROR_INVALID_VALUE')
                return 0
            assert maximum.value >= 32
            api('cuFuncGetAttribute', [c.POINTER(c.c_int), c.c_int, ptr],
                c.byref(maximum), 0, function)
            guard = 0xa5a5a5a5
            words = (u32 * 64)(*([guard] * 64))
            api('cuMemAlloc', [c.POINTER(u64), c.c_size_t], c.byref(allocation), c.sizeof(words))
            api('cuMemcpyHtoD', [u64, ptr, c.c_size_t], allocation, c.cast(words, ptr), c.sizeof(words))
            argument = u64(allocation.value + 16 * c.sizeof(u32))
            arguments = (ptr * 2)(c.cast(c.pointer(argument), ptr), None)
            api('cuLaunchKernel', [ptr] + [u32] * 7 + [ptr, c.POINTER(ptr), ptr],
                function, 1, 1, 1, 32, 1, 1, 0, None, arguments, None)
            api('cuCtxSynchronize', [])
            api('cuMemcpyDtoH', [ptr, u64, c.c_size_t], c.cast(words, ptr), allocation, c.sizeof(words))
            assert list(words) == [guard] * 16 + [i * 3 + 7 for i in range(32)] + [guard] * 16
            print('NUMERICAL_PASS: 32 GPU values and surrounding guards')
    finally:
        if allocation.value:
            api('cuMemFree', [u64], allocation)
        if module.value:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[2] == '--child':
        return child(Path(sys.argv[1]).resolve(), sys.argv[3])
    runtime = Path(sys.argv[1]).resolve()
    expected = {
        'success': ['metal_new_library_with_source', 'metal_new_function',
                    'metal_new_compute_pipeline'],
        'invalid': ['metal_new_library_with_source'],
        'missing': ['metal_new_library_with_source', 'metal_new_function'],
    }
    for mode, stages in expected.items():
        for setting in (None, '1'):
            env = os.environ.copy()
            env.pop('CUMETAL_TRACE_COMPILE', None)
            env['CUMETAL_TRACE_GPU'] = '1'
            env['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
            if setting is not None:
                env['CUMETAL_TRACE_COMPILE'] = setting
            result = subprocess.run(
                [sys.executable, __file__, str(runtime), '--child', mode],
                env=env, capture_output=True, text=True, timeout=60)
            print(result.stdout, end='')
            print(result.stderr, end='', file=sys.stderr)
            if result.returncode == 77:
                return 77
            assert result.returncode == 0, (mode, result.stderr)
            records = parse_spans(result.stderr)
            assert begin_stages(records) == (stages if setting == '1' else []), result.stderr
            if records and mode != 'invalid':
                assert int(records[0]['input_bytes']) == len(SOURCE.encode())
            if mode == 'success':
                assert 'NUMERICAL_PASS' in result.stdout
                assert any(line.startswith('CUMETAL_PROVENANCE ') and
                           'device=apple_gpu' in line and 'launch_success=true' in line
                           for line in result.stderr.splitlines()), result.stderr
    print('PASS: runtime trace stages, cached preparation, errors, and GPU values')
    return 0


if __name__ == '__main__':
    sys.exit(main())
