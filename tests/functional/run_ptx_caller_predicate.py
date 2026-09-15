#!/usr/bin/env python3
"""Caller-local predicates survive direct helpers; unknown state does not."""
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
    launches = [line for line in result.stderr.splitlines() if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
    assert len(launches) == 4, launches
    assert all('device=apple_gpu' in line and 'launch_success=true' in line and
               'provenance=generic_ptx_lowering' in line for line in launches), launches
    raise SystemExit(0)

from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_caller_predicate.ptx').read_text()
values = [17 + x * 23 for x in range(65)]
for label, case in (
    ('range-local predicate', source),
    ('individually declared predicate', source.replace('.reg .pred %p<4>;', '.reg .pred %p0, %p1, %p2, %p3;')),
    ('callee-local names do not clobber caller', source.replace('.func noop() { ret; }',
       '.func noop() {\n.reg .pred %p3;\nmov.pred %p3, 0;\nret;\n}')),
):
    run_integer_case(build, case, values, values, label, word_bits=32, output_words=1)
effect = source.replace('.func noop() { ret; }', """.func noop(.param .b64 output) {
.reg .b64 %rd0;
.reg .pred %p3;
ld.param.u64 %rd0, [output];
mov.pred %p3, 0;
st.global.u32 [%rd0], 23;
ret;
}""")
effect = effect.replace('call.uni noop, ();', """.param .b64 address;
st.param.b64 [address], %rd4;
call.uni noop, (address);""")
effect = effect.replace('st.global.u32 [%rd4], %r7;',
                        'ld.global.u32 %r1, [%rd4];\nadd.u32 %r7, %r7, %r1;\nst.global.u32 [%rd4], %r7;')
run_integer_case(build, effect, values, [x + 23 for x in values],
                 'helper store remains observable', word_bits=32, output_words=1)
for label, case in (
    ('undeclared predicate', source.replace('.reg .pred %p<4>;', '.reg .pred %p<3>;')),
    ('wrong declaration type', source.replace('.reg .pred %p<4>;', '.reg .pred %p<3>;\n.reg .b32 %p3;')),
    ('predicate overwritten', source.replace('mov.pred %p2, %p3;', 'mov.pred %p3, 0;\nmov.pred %p2, %p3;')),
    ('observable undefined payload', source.replace('call.uni noop, ();', 'st.global.u32 [%rd4], %r7;\ncall.uni noop, ();')),
):
    expect_compile_failure(build, case, 'integer_probe', 'PTX register')
    print('REJECTED ' + label)
