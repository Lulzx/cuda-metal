#!/usr/bin/env python3
"""Parameter loads are bound by CFG dominance rather than source block order."""
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
source = (Path(__file__).parent / 'reference/ptx_out_of_order_parameters.ptx').read_text()
values = [value * 0x0101010101010101 for value in range(65)]
run_integer_case(build, source, values, values,
                 'CFG-dominating parameter load after textual use', output_words=1)

undefined = source.replace(
    'bra LOAD_PARAMETER;',
    'setp.eq.u32 %p2, %r0, 0;\n    @%p2 bra READ;\n    bra LOAD_PARAMETER;')
expect_compile_failure(build, undefined, 'integer_probe', 'undefined on an incoming edge')
print('REJECTED parameter value on a path not dominated by its load')
