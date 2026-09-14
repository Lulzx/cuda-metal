"""Shared launch/cleanup support; numerical expectations belong to each test."""
import ctypes as c
import os
from pathlib import Path
import subprocess
import tempfile


def driver_api(build):
    """Keep the loaded driver alive and check exact CUDA status codes."""
    lib = c.CDLL(str(build / 'libcumetal.dylib'))
    def api(name, types, *args, expected=0):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, c.c_int
        status = fn(*args)
        if status != expected:
            raise RuntimeError(f'{name}: {status}, expected {expected}')
    return api


def expect_compile_failure(build, source, entry, diagnostic):
    """Check a rejected PTX form without depending on a Metal compiler or GPU."""
    with tempfile.TemporaryDirectory(prefix='cumetal-ptx-negative-') as work:
        ptx, msl = Path(work) / 'test.ptx', Path(work) / 'test.metal'
        ptx.write_text(source)
        result = subprocess.run(
            [str(build / 'cumetalc'), str(ptx), '--backend=cumetal-ir',
             '--ptx-strict', '--entry', entry, '--emit=msl', '-o', str(msl)],
            capture_output=True, text=True)
        if result.returncode == 0 or diagnostic not in result.stderr:
            raise AssertionError(f'expected {diagnostic!r}: {result.stderr}')


def run_integer_case(build, ptx_source, values, expected, label, entry="integer_probe",
                     word_bits=64, input_words=1, output_words=2,
                     backend="cumetal-ir", emit="msl", abi_lines=None, argument_order=(0, 1, 2)):
    """Run a two-buffer/count kernel with integer inputs and expected outputs."""
    os.environ['CUMETAL_TRACE_GPU'] = '1'
    os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'] = '0'
    api = driver_api(build)
    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
    if input_words < 1 or len(values) % input_words:
        raise ValueError('input count does not match the kernel layout')
    count = len(values) // input_words
    if not count or len(expected) != output_words * count:
        raise ValueError('output count does not match the kernel layout')
    word_type = {32: u32, 64: u64}[word_bits]
    guard = [0xa5a5a5a5] * 16
    source_words = guard + list(values) + guard
    source = (word_type * len(source_words))(*source_words)
    result = (word_type * (len(expected) + 32))(*(guard + [0xa5a5a5a5] * len(expected) + guard))
    context, module, function = ptr(), ptr(), ptr()
    allocations = []
    api('cuInit', [u32], 0)
    api('cuCtxCreate', [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        with tempfile.TemporaryDirectory(prefix='cumetal-ptx-') as work:
            ptx, msl = Path(work) / 'test.ptx', Path(work) / ('test.metallib' if emit == 'metallib' else 'test.metal')
            ptx.write_text(ptx_source)
            compiled = subprocess.run(
                [str(build / 'cumetalc'), str(ptx), '--backend=' + backend,
                 '--ptx-strict', '--entry', entry, '--emit=' + emit, '-o', str(msl)],
                capture_output=True, text=True)
            if compiled.returncode:
                raise RuntimeError(f'{label}: PTX compilation failed\n{compiled.stderr}')
            if abi_lines is not None:
                actual = Path(str(msl) + '.cumetal-abi').read_text().splitlines()
                assert actual == abi_lines, (label, actual, abi_lines)
            api('cuModuleLoad', [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(msl))
            api('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, entry.encode())
            for data in (source, result):
                allocation = u64()
                api('cuMemAlloc', [c.POINTER(u64), c.c_size_t], c.byref(allocation), c.sizeof(data))
                allocations.append(allocation)
                api('cuMemcpyHtoD', [u64, ptr, c.c_size_t], allocation, c.cast(data, ptr), c.sizeof(data))
            count_storage = u64(count)  # Current classifier reads eight bytes for scalars.
            arguments = (*(u64(allocation.value + 16 * c.sizeof(word_type)) for allocation in allocations),
                         count_storage)
            args = (ptr * 4)(*[c.cast(c.pointer(arguments[i]), ptr) for i in argument_order], None)
            api('cuLaunchKernel', [ptr] + [u32]*7 + [ptr, c.POINTER(ptr), ptr],
                function, (count + 63) // 64, 1, 1, 64, 1, 1, 0, None, args, None)
            api('cuCtxSynchronize', [])
            api('cuMemcpyDtoH', [ptr, u64, c.c_size_t], c.cast(result, ptr), allocations[1], c.sizeof(result))
            for i, value in enumerate(expected):
                if result[i + 16] != value:
                    raise RuntimeError(f'word {i}: got {result[i + 16]:08x}, expected {value:08x}')
            assert list(result)[:16] == guard and list(result)[-16:] == guard, 'output guard overwritten'
            readback = type(source)()
            api('cuMemcpyDtoH', [ptr, u64, c.c_size_t], c.cast(readback, ptr), allocations[0], c.sizeof(readback))
            assert list(readback) == source_words, 'input or input guards changed'
            print(f'NUMERICAL_PASS {label}: {count} inputs, output values, guards')
    finally:
        for allocation in allocations:
            api('cuMemFree', [u64], allocation)
        if module:
            api('cuModuleUnload', [ptr], module)
        api('cuCtxDestroy', [ptr], context)
