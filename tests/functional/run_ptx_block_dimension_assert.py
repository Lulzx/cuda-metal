"""Uniform block assertions preserve barriers and reject unsupported shapes."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from ptx_test_support import driver_api, expect_compile_failure

build = Path(sys.argv[1]).resolve()
os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
source = (Path(__file__).parent / 'reference/ptx_block_dimension_assert.ptx').read_text()
# Lane-dependent guards and malformed assertions must not acquire an assumption.
for invalid in (source.replace('%ntid.y', '%tid.y'),
                source.replace('(a,b,c,d,e);', '(a,b,c,d);'),
                source.replace('add.s32 %r2, %r1, -1;', 'mov.u32 %r1, 6;\nadd.s32 %r2, %r1, -1;'),
                source.replace('and.b32 %r3, %r1, %r2;', 'or.b32 %r3, %r1, %r2;')):
    diagnostic = '__assertfail' if '(a,b,c,d);' in invalid else 'barriers, collectives'
    expect_compile_failure(build, invalid, 'block_assert', diagnostic)
api = driver_api(build)
ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
ctx, module, function = ptr(), ptr(), ptr()
api('cuInit', [u32], 0)
api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(ctx), 0, 0)
with tempfile.TemporaryDirectory(prefix='cumetal-block-assert-') as work:
    ptx, msl = Path(work)/'test.ptx', Path(work)/'test.metal'
    ptx.write_text(source)
    subprocess.run([str(build/'cumetalc'), str(ptx), '--backend=cumetal-ir', '--ptx-strict',
                    '--entry', 'block_assert', '--emit=msl', '-o', str(msl)], check=True)
    api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
    api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'block_assert')
    # Distinct streams prove one bad shape cannot cancel a valid block at a barrier.
    cases = []
    for count in (32, 6, 64, 24, 8):
        host, stream, device = ptr(), ptr(), u64()
        api('cuMemHostAlloc', [c.POINTER(ptr), c.c_size_t, u32], c.byref(host), 80*4, 2)
        api('cuMemHostGetDevicePointer', [c.POINTER(u64), ptr, u32], c.byref(device), host, 0)
        words = (u32*80).from_address(host.value)
        for i in range(80): words[i] = 0xa5a5a5a5
        api('cuStreamCreate', [c.POINTER(ptr), u32], c.byref(stream), 1)
        args = (ptr*2)(c.cast(c.pointer(device),ptr), None)
        api('cuLaunchKernel', [ptr]+[u32]*7+[ptr,c.POINTER(ptr),ptr], function, 2,1,1,1,count,1,0,stream,args,None)
        cases.append((count,stream,words))
    for count,stream,words in cases:
        valid = count & (count-1) == 0
        api('cuStreamSynchronize', [ptr], stream, expected=0 if valid else 719)
        assert list(words) == ([count]*count+[0xa5a5a5a5]*(80-count) if valid else [0xa5a5a5a5]*80)
print('BLOCK_ASSERT_PASS: nested helper, valid shapes with barriers, invalid shapes and concurrent independent streams')
