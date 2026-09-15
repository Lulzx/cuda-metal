#!/usr/bin/env python3
"""Dead predicate constants do not consume the guarded-path fact budget."""
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
    assert len(launches) == 2, launches
    assert all('device=apple_gpu' in line and 'launch_success=true' in line and
               'provenance=generic_ptx_lowering' in line for line in launches), launches
    raise SystemExit(0)

from ptx_test_support import expect_compile_failure, run_integer_case

build = Path(sys.argv[1]).resolve()
base = (Path(__file__).parent / 'reference/ptx_caller_predicate.ptx').read_text()
values = [17 + lane * 23 for lane in range(65)]


def predicates(count, live):
    source = base.replace('.reg .pred %p<4>;', f'.reg .pred %p<{count + 4}>;')
    definitions = ''.join(f'mov.pred %p{index}, 0;\n' for index in range(4, count + 4))
    source = source.replace('LOOP:\n', definitions + 'LOOP:\n', 1)
    if live:
        uses = ''.join(f'or.pred %p4, %p4, %p{index};\n'
                       for index in range(5, count + 4))
        source = source.replace('st.global.u32 [%rd4], %r7;',
                                uses + 'st.global.u32 [%rd4], %r7;')
    return source


run_integer_case(build, predicates(256, False), values, values,
                 '256 dead predicate constants', word_bits=32, output_words=1)
run_integer_case(build, predicates(126, True), values, values,
                 '128 live predicate facts', word_bits=32, output_words=1)
expect_compile_failure(build, predicates(127, True), 'integer_probe', 'PTX register')
print('REJECTED 129 live predicate facts')
