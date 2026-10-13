#!/usr/bin/env python3
"""The documented sample must reject NaN and an unwritten final GPU element."""
import ctypes as c
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    compiler, sample, work, runtime = (Path(arg).resolve() for arg in sys.argv[1:])
    if sys.platform != 'darwin' or shutil.which('xcrun') is None:
        print('SKIP: Apple Metal toolchain unavailable')
        return 77
    for tool in ('metal', 'metallib'):
        if subprocess.run(['xcrun', '--find', tool], capture_output=True).returncode:
            print(f'SKIP: xcrun {tool} unavailable')
            return 77
    library = c.CDLL(str(runtime))
    library.cuInit.argtypes, library.cuInit.restype = [c.c_uint], c.c_int
    status = library.cuInit(0)
    if status == 100:
        print('SKIP: no supported Metal device')
        return 77
    assert status == 0, ('cuInit', status)

    source = sample.read_text()
    assignment, guard = 'c[i] = a[i] + b[i];', 'if (i < n) {'
    assert source.count(assignment) == 1, 'sample assignment anchor changed'
    assert source.count(guard) == 1, 'sample bounds anchor changed'
    variants = (
        ('nan', source.replace(assignment, 'c[i] = __int_as_float(0x7fc00000);'), 0),
        ('unwritten_tail', source.replace(guard, 'if (i < n - 1) {'), (1 << 14) + 16),
    )
    for name, mutated_source, mismatch_index in variants:
        case = work / name
        case.mkdir(parents=True, exist_ok=True)
        cuda, executable = case / 'vectorAdd.cu', case / 'vectorAdd'
        cuda.write_text(mutated_source)
        executable.unlink(missing_ok=True)
        env = dict(os.environ, CUMETAL_TRACE_GPU='1',
                   CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0',
                   CUMETAL_ENABLE_LLMC_CPU_EMULATION='0',
                   CUMETAL_DISABLE_LLMC_EMULATION='1')
        built = subprocess.run([str(compiler), str(cuda), '-o', str(executable)],
                               env=env, capture_output=True, text=True, timeout=60)
        (case / 'build.log').write_text(built.stdout + built.stderr)
        assert built.returncode == 0, (name, 'build failed', built.stdout, built.stderr)
        with tempfile.TemporaryDirectory(prefix='runtime-cache-', dir=case) as cache:
            env['CUMETAL_CACHE_DIR'] = cache
            result = subprocess.run([str(executable)], env=env,
                                    capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        (case / 'run.log').write_text(output)
        print(output, end='')
        assert any(line.startswith('CUMETAL_PROVENANCE event=kernel_launch ') and
                   'device=apple_gpu' in line and 'launch_success=true' in line
                   for line in result.stderr.splitlines()), (name, 'no completed Apple GPU launch')
        diagnostic = f'FAIL: mismatch at {mismatch_index} ('
        assert result.returncode != 0 and diagnostic in output, (
            name, 'sample accepted corrupt output or failed for another reason',
            result.returncode, output)
        assert 'PASS: samples/vectorAdd' not in output, (name, 'sample falsely reported success')
        print(f'PASS: vectorAdd rejected {name} after a completed Apple GPU launch')
    return 0


if __name__ == '__main__':
    sys.exit(main())
