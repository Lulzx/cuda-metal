#!/usr/bin/env python3
"""A typed nonzero PTX pointer literal survives a conditional select."""
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
source = (Path(__file__).parent / 'reference/ptx_pointer_literal_select.ptx').read_text()
values = list(range(65))
run_integer_case(build, source, values,
                 [1 if value % 2 == 0 else value for value in values],
                 'device pointer or Rust empty-slice sentinel', output_words=1)

mixed = source.replace('.reg .b64 %rd<8>;',
                       '.local .align 8 .b8 slot[8];\n    .reg .b64 %rd<9>;')
mixed = mixed.replace('selp.b64 %rd6, 1, %rd3, %p1;',
                      'mov.u64 %rd8, slot;\n    selp.b64 %rd6, %rd8, %rd3, %p1;')
expect_compile_failure(build, mixed, 'integer_probe',
                       'directional pointer flow reaches a conflicting concrete address space')
print('REJECTED mixed private/device pointer select')
