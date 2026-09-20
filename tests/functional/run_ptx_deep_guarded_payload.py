#!/usr/bin/env python3
"""Carry an optional-payload predicate through more than eight CFG blocks."""
from pathlib import Path
import sys
from ptx_test_support import expect_compile_failure, run_integer_case

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_deep_guarded_payload.ptx').read_text()
values = list(range(65))
expected = [99] + [7] * 64
run_integer_case(build, source, values, expected, 'deep guarded payload',
                 entry='deep_guarded_payload', word_bits=32, output_words=1)

reachable = source.replace('MERGE:\n', 'MERGE:\n    mov.pred %p0, 1;\n')
expect_compile_failure(build, reachable, 'deep_guarded_payload', 'undefined')
