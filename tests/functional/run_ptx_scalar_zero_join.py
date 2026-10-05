#!/usr/bin/env python3
"""A float loop joins a b32 zero seed before a warp shuffle."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from ptx_test_support import driver_api

build = Path(sys.argv[1]).resolve()
os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
source = r'''
.version 7.1
.target sm_80
.address_size 64
.visible .entry scalar_zero_join(.param .u64 output) {
.reg .b64 %rd<4>;
.reg .b32 %r<8>;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
mov.u32 %r1, %tid.x;
mov.b32 %r2, 0;
setp.eq.u32 %p1, %r1, 0;
@%p1 bra JOIN;
mov.u32 %r3, 0;
LOOP:
add.f32 %r2, %r2, 0f3F800000;
add.u32 %r3, %r3, 1;
setp.lt.u32 %p1, %r3, 3;
@%p1 bra LOOP;
JOIN:
shfl.sync.idx.b32 %r4, %r2, 1, 31, -1;
mov.u32 %r5, %ctaid.x;
mad.lo.u32 %r6, %r5, 32, %r1;
mul.wide.u32 %rd2, %r6, 8;
add.u64 %rd3, %rd1, %rd2;
st.global.b32 [%rd3], %r2;
st.global.b32 [%rd3+4], %r4;
ret;
}
'''
api = driver_api(build)
ptr,u32,u64 = c.c_void_p,c.c_uint32,c.c_uint64
ctx,module,function,host = ptr(),ptr(),ptr(),ptr()
device=u64()
api('cuInit',[u32],0)
api('cuCtxCreate',[c.POINTER(ptr),u32,c.c_int],c.byref(ctx),0,0)
with tempfile.TemporaryDirectory(prefix='cumetal-scalar-zero-') as work:
    ptx,msl=Path(work)/'join.ptx',Path(work)/'join.metal'
    ptx.write_text(source)
    subprocess.run([str(build/'cumetalc'),str(ptx),'--backend=cumetal-ir','--ptx-strict','--entry','scalar_zero_join','--emit=msl','-o',str(msl)],check=True)
    api('cuModuleLoad',[c.POINTER(ptr),c.c_char_p],c.byref(module),os.fsencode(msl))
    api('cuModuleGetFunction',[c.POINTER(ptr),ptr,c.c_char_p],c.byref(function),module,b'scalar_zero_join')
    api('cuMemHostAlloc',[c.POINTER(ptr),c.c_size_t,u32],c.byref(host),256*4,2)
    api('cuMemHostGetDevicePointer',[c.POINTER(u64),ptr,u32],c.byref(device),host,0)
    words=(u32*256).from_address(host.value)
    for i in range(256):words[i]=0xa5a5a5a5
    args=(ptr*2)(c.cast(c.pointer(device),ptr),None)
    api('cuLaunchKernel',[ptr]+[u32]*7+[ptr,c.POINTER(ptr),ptr],function,4,1,1,32,1,1,0,None,args,None)
    api('cuCtxSynchronize',[])
    for i in range(128):
        assert words[2*i] == (0 if i%32==0 else 0x40400000),(i,hex(words[2*i]))
        assert words[2*i+1] == 0x40400000,(i,hex(words[2*i+1]))
print('SCALAR_ZERO_JOIN_PASS: float loop, zero branch and shuffled bit patterns on Apple GPU')
# ld.param must sign/zero extend its access width into a wider destination.
with tempfile.TemporaryDirectory(prefix='cumetal-param-extension-') as work:
    ptx,msl=Path(work)/'extend.ptx',Path(work)/'extend.metal'
    ptx.write_text('''
.version 7.1
.target sm_80
.address_size 64
.visible .entry param_extension(.param .align 4 .b8 capture[8], .param .u64 output) {
.reg .b64 %rd<4>;
ld.param.s32 %rd1, [capture];
ld.param.u32 %rd2, [capture+4];
ld.param.u64 %rd3, [output];
st.global.u64 [%rd3], %rd1;
st.global.u64 [%rd3+8], %rd2;
ret;
}
''')
    subprocess.run([str(build/'cumetalc'),str(ptx),'--backend=cumetal-ir','--ptx-strict','--entry','param_extension','--emit=msl','-o',str(msl)],check=True)
    api('cuModuleLoad',[c.POINTER(ptr),c.c_char_p],c.byref(module),os.fsencode(msl))
    api('cuModuleGetFunction',[c.POINTER(ptr),ptr,c.c_char_p],c.byref(function),module,b'param_extension')
    capture=(u32*2)(0xfffffffd,0xfffffffd)
    args=(ptr*3)(c.cast(capture,ptr),c.cast(c.pointer(device),ptr),None)
    api('cuLaunchKernel',[ptr]+[u32]*7+[ptr,c.POINTER(ptr),ptr],function,1,1,1,1,1,1,0,None,args,None)
    api('cuCtxSynchronize',[])
    widened=(u64*2).from_address(host.value)
    assert list(widened)==[0xfffffffffffffffd,0xfffffffd],list(widened)
print('PARAM_EXTENSION_PASS: signed/unsigned parameter loads preserve destination register bits')
# PTX red has no destination; both address and payload are sources.
with tempfile.TemporaryDirectory(prefix='cumetal-atomic-red-') as work:
    ptx,msl=Path(work)/'red.ptx',Path(work)/'red.metal'
    ptx.write_text('''
.version 8.8
.target sm_80
.address_size 64
.visible .entry atomic_red(.param .u64 output) {
.reg .b64 %rd1;
.reg .b32 %r1;
.reg .f32 %f1;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
mov.f32 %f1, 0f3FA00000;
red.add.global.relaxed.gpu.f32 [%rd1], %f1;
mov.u32 %r1, %tid.x;
and.b32 %r1, %r1, 1;
setp.eq.u32 %p1, %r1, 0;
@!%p1 bra SKIP;
red.add.global.relaxed.gpu.f32 [%rd1+4], %f1;
SKIP:
red.global.add.u32 [%rd1+8], 1;
ret;
}
''')
    subprocess.run([str(build/'cumetalc'),str(ptx),'--backend=cumetal-ir','--ptx-strict','--entry','atomic_red','--emit=msl','-o',str(msl)],check=True)
    api('cuModuleLoad',[c.POINTER(ptr),c.c_char_p],c.byref(module),os.fsencode(msl))
    api('cuModuleGetFunction',[c.POINTER(ptr),ptr,c.c_char_p],c.byref(function),module,b'atomic_red')
    for i in range(3): words[i]=0
    args=(ptr*2)(c.cast(c.pointer(device),ptr),None)
    api('cuLaunchKernel',[ptr]+[u32]*7+[ptr,c.POINTER(ptr),ptr],function,8,1,1,64,1,1,0,None,args,None)
    api('cuCtxSynchronize',[])
    sums=(c.c_float*2).from_address(host.value)
    assert list(sums)==[640.0,320.0],list(sums)
    assert words[2]==512,words[2]
print('ATOMIC_RED_PASS: contended float/integer and branch-guarded reductions on Apple GPU')
