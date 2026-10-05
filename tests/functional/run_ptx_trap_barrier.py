#!/usr/bin/env python3
"""A trap reached through a helper call inside a kernel that synchronizes.

Barriers cannot wait on a lane that has left, so a taken trap must report
CUDA_ERROR_LAUNCH_FAILED without hanging the block, and the untaken path
must still synchronize exactly. LAMMPS' fix wall/* kernels have this shape
and were refused outright before.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from ptx_test_support import driver_api

GUARD = 0xa5a5a5a5


def main():
    build = Path(sys.argv[1]).resolve()
    os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
    api = driver_api(build)
    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
    context, module, function = ptr(), ptr(), ptr()
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    source = Path(__file__).parent / 'reference' / 'ptx_trap_barrier.ptx'
    with tempfile.TemporaryDirectory(prefix='cumetal-trap-barrier-') as work:
        msl = Path(work) / 'test.metal'
        subprocess.run([str(build / 'cumetalc'), str(source), '--backend=cumetal-ir', '--ptx-strict',
                        '--entry', 'trap_probe', '--emit=msl', '-o', str(msl)], check=True)
        generated = msl.read_text()
        assert 'threadgroup_barrier' in generated and 'check__cm_trap_guarded(' in generated
        api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
        api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'trap_probe')
        for mode in (0, 1, 2, 3, 0):
            host, stream, device = ptr(), ptr(), u64()
            api('cuMemHostAlloc', [c.POINTER(ptr), c.c_size_t, u32], c.byref(host), 80 * 4, 2)
            api('cuMemHostGetDevicePointer', [c.POINTER(u64), ptr, u32], c.byref(device), host, 0)
            words = (u32 * 80).from_address(host.value)
            for i in range(80): words[i] = GUARD
            api('cuStreamCreate', [c.POINTER(ptr), u32], c.byref(stream), 1)
            mode_arg = u64(mode)
            args = (ptr * 3)(c.cast(c.pointer(device), ptr), c.cast(c.pointer(mode_arg), ptr), None)
            api('cuLaunchKernel', [ptr] + [u32] * 7 + [ptr, c.POINTER(ptr), ptr],
                function, 1, 1, 1, 64, 1, 1, 0, stream, args, None)
            api('cuStreamSynchronize', [ptr], stream, expected=719 if mode else 0)
            for i in range(64):
                # A trap aborts the launch, so survivors' outputs are
                # unspecified (a cancelled peer may never have written the
                # shared slot a survivor reads). Only the untaken path is
                # exact, and when every lane traps nothing is written.
                if mode == 0: assert words[i] == i + 1, (mode, i, hex(words[i]))
                if mode == 1: assert words[i] == GUARD, (mode, i, hex(words[i]))
            assert list(words)[64:] == [GUARD] * 16, mode
    print('TRAP_BARRIER_PASS: exact untaken path; all, divergent and single-lane traps report without hanging the block')


if __name__ == '__main__':
    main()
