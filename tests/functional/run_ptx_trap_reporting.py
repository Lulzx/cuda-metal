#!/usr/bin/env python3
"""Taken, untaken and divergent traps retain per-launch status until completion."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    build = Path(sys.argv[1]).resolve()
    os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
    # Exercise non-tracing path where regular kernels would otherwise batch.
    if '--trace' in sys.argv: os.environ['CUMETAL_TRACE_GPU'] = '1'
    else: os.environ.pop('CUMETAL_TRACE_GPU', None)
    lib = c.CDLL(str(build / 'libcumetal.dylib'))
    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
    def api(name, types, *args, expected=0):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, c.c_int
        status = fn(*args)
        if status != expected:
            raise RuntimeError(f'{name}: {status}, expected {expected}')
    context, module, function = ptr(), ptr(), ptr()
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    fixture = 'ptx_trap_calls.ptx' if '--calls' in sys.argv else 'ptx_trap_reporting.ptx'
    source = (Path(__file__).parent / 'reference' / fixture).read_text()
    with tempfile.TemporaryDirectory(prefix='cumetal-trap-') as work:
        ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
        ptx.write_text(source)
        subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir', '--ptx-strict',
                        '--entry', 'trap_probe', '--emit=msl', '-o', str(msl)], check=True)
        if '--calls' in sys.argv:
            generated = msl.read_text()
            assert 'finite(' in generated and 'finite_leaf(' in generated
            assert 'spin(' not in generated and 'store_then_trap(' not in generated
        subprocess.run([str(build / 'tests/functional/cumetal_trap_binding_test'), str(msl)], check=True)
        api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
        api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'trap_probe')
        cases = []
        for mode in (0, 1, 2, 3, 4, 0):
            host, stream, device = ptr(), ptr(), u64()
            api('cuMemHostAlloc', [c.POINTER(ptr), c.c_size_t, u32], c.byref(host), 80*4, 2)
            api('cuMemHostGetDevicePointer', [c.POINTER(u64), ptr, u32], c.byref(device), host, 0)
            words = (u32 * 80).from_address(host.value)
            for i in range(80): words[i] = 0xa5a5a5a5
            api('cuStreamCreate', [c.POINTER(ptr), u32], c.byref(stream), 1)
            cases.append((mode, host, stream, device, words))
        events = []
        for mode, host, stream, device, words in cases:
            event = ptr()
            api('cuEventCreate', [c.POINTER(ptr), u32], c.byref(event), 0)
            events.append(event)
        for (mode, host, stream, device, words), event in zip(cases, events):
            mode_arg = u64(mode)
            args = (ptr * 3)(c.cast(c.pointer(device), ptr), c.cast(c.pointer(mode_arg), ptr), None)
            api('cuLaunchKernel', [ptr]+[u32]*7+[ptr,c.POINTER(ptr),ptr],
                function, 1,1,1,1 if mode == 4 else 64,1,1,0,stream,args,None)
            api('cuEventRecord', [ptr, ptr], event, stream)
        for (mode, host, stream, device, words), event in zip(cases, events):
            error = 719 if mode else 0  # CUDA_ERROR_LAUNCH_FAILED
            api('cuEventSynchronize', [ptr], event, expected=error)
            api('cuStreamSynchronize', [ptr], stream, expected=error)
            api('cuStreamQuery', [ptr], stream, expected=error)
            for i in range(64):
                allowed = (1,) if mode == 0 else (0xa5a5a5a5,)
                if mode == 2 and i % 2 == 0: allowed = (1, 0xa5a5a5a5)
                if mode == 4 and i == 0: allowed = (2,)
                assert words[i] in allowed, (mode, i, words[i])
            assert list(words)[64:] == [0xa5a5a5a5]*16
        print('TRAP_PASS: untaken/all/divergent/spinning peers, concurrent independent streams, event/stream completion, repeated errors, outputs and guards')
        # Faulted streams retain their error; resource APIs may report it too.
        # This isolated process owns all handles and exits after validation.


if __name__ == '__main__':
    main()
