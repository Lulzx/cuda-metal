#!/usr/bin/env python3
"""Device-pointer fields loaded from private records retain pointer type."""
from pathlib import Path
import subprocess
import sys

if '--gpu-child' not in sys.argv:
    result = subprocess.run([sys.executable, __file__, *sys.argv[1:], '--gpu-child'],
                            capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    launches = [line for line in result.stderr.splitlines()
                if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
    assert len(launches) == 1, launches
    assert all('device=apple_gpu' in line and 'launch_success=true' in line and
               'provenance=generic_ptx_lowering' in line for line in launches), launches
    raise SystemExit(0)

from ptx_test_support import expect_compile_failure, run_integer_case

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_pointer_record_fields.ptx').read_text()
values = [value * 0x0101010101010101 for value in range(65)]
expected = [values[lane if lane % 2 == 0 else lane + 1] for lane in range(65)]
run_integer_case(build, source, values, expected,
                 'device pointers loaded from private record fields', output_words=1)

reused = source.replace(
    'ld.local.u64 %rd2, [%rd1];',
    'ld.local.u64 %rd2, [%rd1];\n    mov.u64 %rd2, 0;')
expect_compile_failure(build, reused, 'integer_probe', 'does not match')

predicated = source.replace(
    'ld.local.u64 %rd2, [%rd1];',
    'mov.pred %p0, 1;\n    @%p0 ld.local.u64 %rd2, [%rd1];')
expect_compile_failure(build, predicated, 'integer_probe', 'does not match')

narrow = source.replace('ld.local.u64 %rd2, [%rd1];',
                        'ld.local.u32 %rd2, [%rd1];')
expect_compile_failure(build, narrow, 'integer_probe', 'does not match')
print('REJECTED reused, predicated, and non-64-bit pointer-field definitions')
