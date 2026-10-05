#!/usr/bin/env python3
"""A rotated loop with a terminal second exit, then a barrier.

Kokkos reductions have this shape (the second exit is the Kokkos::abort
path). Lanes leave the loop after different trip counts, so the barrier must
hold the early ones until every lane has written its shared slot. A per-lane
CFG dispatcher cannot do that: Metal releases a barrier reached by lanes
whose peers are still iterating, and the neighbour read sees a stale slot.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from ptx_test_support import driver_api, expect_compile_failure

GUARD = 0xa5a5a5a5


def main():
    build = Path(sys.argv[1]).resolve()
    os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
    api = driver_api(build)
    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
    context, module, function = ptr(), ptr(), ptr()
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    source = Path(__file__).parent / 'reference' / 'ptx_loop_terminal_exit_barrier.ptx'
    # A barrier on the second exit makes it rejoin-relevant: the loop then has
    # no single structured exit, and the per-lane dispatcher must be refused
    # rather than emitted around a barrier.
    text = source.read_text()
    synchronized_exit = text.replace('FAIL:\n', 'FAIL:\nbar.sync 0;\n')
    assert synchronized_exit != text
    expect_compile_failure(build, synchronized_exit, 'loop_exit_barrier',
                           'CFG dispatcher cannot carry barriers')
    with tempfile.TemporaryDirectory(prefix='cumetal-loop-exit-barrier-') as work:
        msl = Path(work) / 'test.metal'
        subprocess.run([str(build / 'cumetalc'), str(source), '--backend=cumetal-ir', '--ptx-strict',
                        '--entry', 'loop_exit_barrier', '--emit=msl', '-o', str(msl)], check=True)
        generated = msl.read_text()
        assert 'threadgroup_barrier' in generated and 'cm_block_state' not in generated
        api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
        api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module,
            b'loop_exit_barrier')
        acc = [t * (t + 1) // 2 for t in range(64)]
        # limit 1 << 20 never takes the second exit; limit 100 sends lanes
        # whose running sum passes it down the terminal path.
        for limit in (1 << 20, 100, 1 << 20):
            host, device = ptr(), u64()
            api('cuMemHostAlloc', [c.POINTER(ptr), c.c_size_t, u32], c.byref(host), 80 * 4, 2)
            api('cuMemHostGetDevicePointer', [c.POINTER(u64), ptr, u32], c.byref(device), host, 0)
            words = (u32 * 80).from_address(host.value)
            for i in range(80): words[i] = GUARD
            limit_arg = u32(limit)
            args = (ptr * 3)(c.cast(c.pointer(device), ptr), c.cast(c.pointer(limit_arg), ptr), None)
            api('cuLaunchKernel', [ptr] + [u32] * 7 + [ptr, c.POINTER(ptr), ptr],
                function, 1, 1, 1, 64, 1, 1, 0, None, args, None)
            api('cuCtxSynchronize', [], expected=0)
            for t in range(64):
                failed = any(acc[s] > limit for s in range(t + 1))
                if failed:
                    assert words[t] == 0xffffffff, (limit, t, hex(words[t]))
                elif limit == 1 << 20:
                    assert words[t] == acc[(t + 1) % 64] + acc[((t ^ 1) + 1) % 64], (limit, t, words[t])
            assert list(words)[64:] == [GUARD] * 16, limit
    print('LOOP_EXIT_BARRIER_PASS: rotated loop with a terminal exit stays structured across the barrier; '
          'a synchronizing second exit is refused')


if __name__ == '__main__':
    main()
