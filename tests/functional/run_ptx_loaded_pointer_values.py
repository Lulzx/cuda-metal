#!/usr/bin/env python3
"""Pointers loaded from homogeneous local tables retain their address space."""
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
    assert len(launches) == 5, launches
    assert all('device=apple_gpu' in line and 'launch_success=true' in line and
               'provenance=generic_ptx_lowering' in line for line in launches), launches
    raise SystemExit(0)

from ptx_test_support import expect_compile_failure, run_integer_case

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_loaded_pointer_values.ptx').read_text()
values = list(range(65))

run_integer_case(build, source, values,
                 [29 if value % 2 == 0 else 31 for value in values],
                 'vector-store scalar-load pointer table with byte offset', output_words=1)

zero_offset = source.replace('ld.u64 %rd13, [%rd12+8];',
                             'ld.u64 %rd13, [%rd12];')
run_integer_case(build, zero_offset, values,
                 [17 if value % 2 == 0 else 23 for value in values],
                 'loaded pointer without byte offset', output_words=1)

scalar_stores = source.replace(
    'st.local.v2.u64 [%rd7], {%rd8, %rd9};',
    'st.local.u64 [%rd7], %rd8;\nst.local.u64 [%rd7+8], %rd9;')
run_integer_case(build, scalar_stores, values,
                 [29 if value % 2 == 0 else 31 for value in values],
                 'scalar-store pointer table', output_words=1)

vector_load = source.replace(
    'ld.local.u64 %rd12, [%rd11];\n    ld.u64 %rd13, [%rd12+8];',
    'ld.local.v2.u64 {%rd12, %rd10}, [%rd7];\n    ld.u64 %rd13, [%rd10];')
run_integer_case(build, vector_load, values, [23] * len(values),
                 'vector-load pointer table', output_words=1)

run_integer_case(build, source, values, [value for _ in values for value in (208, 212)],
                 'mixed stack depot pointer suballocation with bounded loop',
                 entry='suballocation_probe', output_words=2)

for label, invalid in [
    ('mixed private and device pointers', source.replace(
        'st.local.v2.u64 [%rd7], {%rd8, %rd9};',
        'st.local.v2.u64 [%rd7], {%rd8, %rd0};')),
    ('mixed pointer and integer values', source.replace(
        'st.local.v2.u64 [%rd7], {%rd8, %rd9};',
        'st.local.v2.u64 [%rd7], {%rd8, %rd5};')),
    ('reused pointer source register', source.replace(
        'st.local.v2.u64 [%rd7], {%rd8, %rd9};',
        'mov.u64 %rd8, 7;\nst.local.v2.u64 [%rd7], {%rd8, %rd9};')),
    ('escaped local pointer table', source.replace(
        'st.local.v2.u64 [%rd7], {%rd8, %rd9};',
        'st.global.u64 [%rd1], %rd7;\nst.local.v2.u64 [%rd7], {%rd8, %rd9};')),
]:
    expect_compile_failure(build, invalid, 'integer_probe', 'operand type')
    print('REJECTED ' + label)
