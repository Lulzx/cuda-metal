#!/usr/bin/env python3
"""Repeated stores must not turn their scalar value into pointer provenance."""
from pathlib import Path
import sys
from ptx_test_support import run_integer_case
from run_ptx_integer_widths import TEMPLATE

source = TEMPLATE.replace('.param .u32 count', '.param .u64 count')
source = source.replace('ld.param.u32 %r1, [count];',
                        'ld.param.u64 %rd9, [count];\ncvt.u32.u64 %r1, %rd9;')
source = source.replace('BODY', 'mul.lo.u64 %rd7, %rd9, 3;\nmov.u64 %rd8, %rd9;')
count = 263
run_integer_case(Path(sys.argv[1]).resolve(), source, list(range(count)),
                 [3 * count, count] * count, 'repeated stores of a runtime scalar parameter')
