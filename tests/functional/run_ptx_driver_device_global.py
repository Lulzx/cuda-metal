#!/usr/bin/env python3
"""A promoted device global is module-owned storage on the driver-API path.

Registration owns this storage through the host shadow registered by
`__cudaRegisterVar`. A driver-API `cuModuleLoad` has no registration, so the
module itself has to own it. Before the ABI sidecar described the binding, the
hidden Metal buffer was simply left unpopulated: the kernel read zeros instead
of the source initializer and its writes did not survive the launch.
"""
import ctypes as c
import subprocess
import sys
import tempfile
from pathlib import Path

PTX = """
.version 7.0
.target sm_80
.address_size 64

.global .align 4 .b8 private_counter[4] = {5, 0, 0, 0};
.visible .global .align 4 .u32 visible_counter = 100;

.visible .entry bump_private(.param .u64 output)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<6>;
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.u32 %r1, [private_counter];
    add.u32 %r2, %r1, 7;
    st.global.u32 [private_counter], %r2;
    ld.global.u32 %r3, [visible_counter];
    add.u32 %r4, %r3, %r2;
    st.global.u32 [visible_counter], %r4;
    st.global.u32 [%rd2], %r4;
    ret;
}

"""

ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64


def main() -> int:
    build = Path(sys.argv[1]).resolve()
    cumetalc = build / 'cumetalc'
    if not cumetalc.exists():
        print('SKIP: cumetalc is not built')
        return 77

    work = tempfile.TemporaryDirectory(prefix='cumetal-driver-global-')
    root = Path(work.name)
    source = root / 'globals.ptx'
    source.write_text(PTX)
    metallib = root / 'globals.metallib'
    built = subprocess.run(
        [str(cumetalc), str(source), '--backend=cumetal-ir', '--emit=metallib',
         '--overwrite', '-o', str(metallib)],
        capture_output=True, text=True)
    if built.returncode != 0 or not metallib.exists():
        print(f'SKIP: metallib build unavailable: {built.stderr.strip()[:400]}')
        return 77

    sidecar = (metallib.parent / (metallib.name + '.cumetal-abi')).read_text()
    for expected in ('global private_counter 4 4 05000000',
                     'global visible_counter 4 4 64000000'):
        if expected not in sidecar:
            raise AssertionError(
                f'sidecar does not describe the hidden storage: {expected!r} '
                f'missing from\n{sidecar}')

    lib = c.CDLL(str(build / 'libcumetal.dylib'))

    def api(name, types, *args, expected=0):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, c.c_int
        status = fn(*args)
        if status != expected:
            raise RuntimeError(f'{name} -> {status}, expected {expected}')

    ctx, module = ptr(), ptr()
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(ctx), 0, 0)
    api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module),
        str(metallib).encode())

    functions = {}
    for name in ('bump_private',):
        handle = ptr()
        api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p],
            c.byref(handle), module, name.encode())
        functions[name] = handle

    out = u64()
    api('cuMemAlloc', [c.POINTER(u64), c.c_size_t], c.byref(out), 4)
    host = (u32 * 1)(0xa5a5a5a5)

    def launch(name):
        argument = u64(out.value)
        params = (c.c_void_p * 1)(c.cast(c.byref(argument), c.c_void_p))
        api('cuLaunchKernel',
            [ptr, u32, u32, u32, u32, u32, u32, u32, ptr,
             c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)],
            functions[name], 1, 1, 1, 1, 1, 1, 0, None, params, None)
        api('cuCtxSynchronize', [])
        api('cuMemcpyDtoH', [ptr, u64, c.c_size_t], c.cast(host, ptr), out, 4)
        return host[0]

    # Both globals start at their source initializers and keep what the GPU
    # wrote: 100 + (5 + 7) = 112, then 112 + (12 + 7) = 131. Reading zeros or
    # losing the writes -- the behaviour before the sidecar described this
    # storage -- gives 7 then 7.
    observed = [launch('bump_private'), launch('bump_private')]
    if observed != [112, 131]:
        raise AssertionError(
            f'device globals on the driver path: got {observed}, '
            'expected [112, 131]')

    # The storage is addressable from the host, as CUDA promises.
    address, size = u64(), c.c_size_t()
    api('cuModuleGetGlobal', [c.POINTER(u64), c.POINTER(c.c_size_t), ptr, c.c_char_p],
        c.byref(address), c.byref(size), module, b'visible_counter')
    if size.value != 4 or address.value == 0:
        raise AssertionError(f'cuModuleGetGlobal returned {address.value}/{size.value}')
    api('cuMemcpyDtoH', [ptr, u64, c.c_size_t], c.cast(host, ptr), address, 4)
    if host[0] != 131:
        raise AssertionError(f'cuModuleGetGlobal storage reads {host[0]}, expected 131')

    # A host write through that address is what the next launch reads.
    seed = (u32 * 1)(900)
    api('cuMemcpyHtoD', [u64, ptr, c.c_size_t], address, c.cast(seed, ptr), 4)
    if launch('bump_private') != 926:
        raise AssertionError(
            'a host write through cuModuleGetGlobal was not observed by the GPU')

    # An unrecorded name is still NOT_FOUND rather than a fabricated address.
    api('cuModuleGetGlobal', [c.POINTER(u64), c.POINTER(c.c_size_t), ptr, c.c_char_p],
        c.byref(address), c.byref(size), module, b'no_such_symbol', expected=500)

    api('cuMemFree', [u64], out)
    api('cuModuleUnload', [ptr], module)
    print('DRIVER_DEVICE_GLOBAL_OK 112 131 926')
    return 0


if __name__ == '__main__':
    sys.exit(main())
