#!/usr/bin/env python3
"""Generic addresses from LAMMPS' Kokkos pair styles, run on the GPU.

space_join / space_select: one register carries a private pointer on some lanes and a device
pointer on others into a displaced generic load. The typed importer used to
refuse the join (and typed the select as its last operand's space); a select
now becomes a branch, and the join block gets one copy per address space.

cell_displacement: a raw device address reloaded from a stack cell and read
with a displacement. The importer relabelled the integer as a pointer for the
displaced form and failed IR verification.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from ptx_test_support import driver_api, expect_compile_failure

THREADS = 64
ROW = 16          # words per thread in space_join's device input
ITERATIONS = 5    # cell_displacement rows


class Record(c.Structure):
    _fields_ = [('data', c.c_uint64), ('count', c.c_uint32), ('pad', c.c_uint32)]


def main():
    build = Path(sys.argv[1]).resolve()
    os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
    api = driver_api(build)
    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
    context = ptr()
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    source = Path(__file__).parent / 'reference' / 'ptx_generic_address_joins.ptx'
    # A loop-header join cannot be split without a second loop entry.
    expect_compile_failure(build, source.read_text(), 'loop_header_join',
                           "cannot split PTX block '$L__BB3_1' by incoming pointer address space")

    def buffer(words):
        host, device = ptr(), u64()
        api('cuMemHostAlloc', [c.POINTER(ptr), c.c_size_t, u32], c.byref(host), words * 4, 2)
        api('cuMemHostGetDevicePointer', [c.POINTER(u64), ptr, u32], c.byref(device), host, 0)
        return (u32 * words).from_address(host.value), device

    def launch(function, args):
        # The MSL has no ABI sidecar, so the driver counts arguments up to a NULL.
        array = (ptr * (len(args) + 1))(*[c.cast(c.pointer(a), ptr) for a in args], None)
        api('cuLaunchKernel', [ptr] + [u32] * 7 + [ptr, c.POINTER(ptr), ptr],
            function, 1, 1, 1, THREADS, 1, 1, 0, None, array, None)
        api('cuCtxSynchronize', [])

    with tempfile.TemporaryDirectory(prefix='cumetal-generic-joins-') as work:
        def kernel(entry):
            msl = Path(work) / f'{entry}.metal'
            subprocess.run([str(build / 'cumetalc'), str(source), '--backend=cumetal-ir', '--ptx-strict',
                            '--entry', entry, '--emit=msl', '-o', str(msl)], check=True)
            module, function = ptr(), ptr()
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module,
                entry.encode())
            return function

        cell_displacement = kernel('cell_displacement')

        source_words, source_device = buffer(THREADS * ROW)
        for i in range(THREADS * ROW): source_words[i] = 0x5000 + i
        out, out_device = buffer(THREADS)
        # 0: device only; 64: private only; 13: lanes diverge within a SIMD group.
        for entry in ('space_join', 'space_select'):
            function = kernel(entry)
            for split in (0, 13, THREADS):
                for i in range(THREADS): out[i] = 0
                launch(function, [out_device, source_device, u32(split)])
                for t in range(THREADS):
                    index = (t & 7) + 1
                    expected = 1000 + index + t if t < split else 0x5000 + t * ROW + index
                    assert out[t] == expected, (entry, split, t, out[t], expected)

        rows, rows_device = buffer(THREADS * ITERATIONS)
        for i in range(THREADS * ITERATIONS): rows[i] = i * 3 + 1
        for i in range(THREADS): out[i] = 0
        record = Record(rows_device.value, ITERATIONS, 0)
        launch(cell_displacement, [out_device, record])
        for t in range(THREADS):
            expected = sum(rows[t + THREADS * j] for j in range(ITERATIONS))
            assert out[t] == expected, ('cell_displacement', t, out[t], expected)
    print('GENERIC_ADDRESS_JOINS_PASS: private/device joins split per space; displaced stack-cell addresses load exactly')


if __name__ == '__main__':
    main()
