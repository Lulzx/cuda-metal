#!/usr/bin/env python3
"""A former pointer cell contains scalar bits after a generic overwrite."""
from pathlib import Path
import sys

from ptx_test_support import run_integer_case


mask = (1 << 64) - 1
values = [0, 1, 17, (1 << 32) - 1, 1 << 32, (1 << 63), mask]
values += [(lane * 0x9E3779B97F4A7C15) & mask for lane in range(58)]
expected = [word for value in values for word in ((value * 5) & mask, 7)]
source = (Path(__file__).parent / 'reference/ptx_reused_local_cells.ptx').read_text()
build = Path(sys.argv[1]).resolve()
for scalar in (False, True):
    current = source
    if scalar:
        current = current.replace('ld.local.v2.u64 {%x, %y}, [%cell];',
                                  'ld.local.u64 %x, [%cell];\n ld.local.u64 %y, [%cell+8];')
    run_integer_case(build, current, values, expected,
                     'reused local scalar cells, scalar loads=' + str(scalar),
                     entry='probe', output_words=2)
