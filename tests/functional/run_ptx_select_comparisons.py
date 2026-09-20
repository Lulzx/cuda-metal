#!/usr/bin/env python3
"""Compare-derived predicate aliases must protect self-select old inputs."""
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
    assert len(launches) == 19, launches
    assert all('device=apple_gpu' in line and 'launch_success=true' in line and
               'provenance=generic_ptx_lowering' in line for line in launches), launches
    raise SystemExit(0)

from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_select_comparisons.ptx').read_text()
values = [17 + x * 23 for x in range(65)]
for width in (16, 32, 64):
    for kind in ('b', 'u'):
        base = source
        counter = '%r8'
        if width == 16:
            counter = '%rs0'
            base = base.replace('.reg .pred %p<3>;', '.reg .pred %p<3>;\n.reg .b16 %rs0;')
            base = base.replace('add.u32 %r8, %r8, 1;', 'add.u32 %r8, %r8, 1;\ncvt.u16.u32 %rs0, %r8;')
        if width == 64:
            counter = '%rd5'
            base = base.replace('.reg .b64 %rd<5>;', '.reg .b64 %rd<6>;')
            base = base.replace('add.u32 %r8, %r8, 1;', 'add.u32 %r8, %r8, 1;\ncvt.u64.u32 %rd5, %r8;')
        base = base.replace('setp.eq.u32 %p1, %r8, 1;', f'setp.eq.{kind}{width} %p1, {counter}, 1;')
        base = base.replace('setp.ne.u32 %p2, %r8, 1;', f'setp.ne.{kind}{width} %p2, {counter}, 1;')
        for form, case in (
            ('complement', base),
            ('same', base.replace(f'setp.eq.{kind}{width} %p1', f'setp.ne.{kind}{width} %p1')
                         .replace('@%p1 bra LOOP;', '@!%p1 bra LOOP;')),
            ('swapped', base.replace(f'%p2, {counter}, 1;', f'%p2, 1, {counter};')),
        ):
            run_integer_case(build, case, values, values, f'{kind}{width} {form}', word_bits=32, output_words=1)
mixed = source.replace('mov.u32 %r8, 0;', 'mov.u32 %r8, 0;\nand.b32 %r1, %r0, 3;\nadd.u32 %r1, %r1, 1;')
mixed = mixed.replace('setp.eq.u32 %p1, %r8, 1;', 'setp.ne.u32 %p1, %r8, %r1;')
mixed = mixed.replace('setp.ne.u32 %p2, %r8, 1;', 'setp.eq.u32 %p2, %r8, %r1;')
run_integer_case(build, mixed, values, values, 'mixed per-lane retry counts', word_bits=32, output_words=1)
for label, case in (
    ('operand write', source.replace('setp.ne.u32 %p2', 'mov.u32 %r8, 42;\nsetp.ne.u32 %p2')),
    ('predicate overwrite', source.replace('selp.b32', 'not.pred %p1, %p1;\nselp.b32')),
    ('different literal', source.replace('%p2, %r8, 1;', '%p2, %r8, 2;')),
    ('different type', source.replace('setp.ne.u32', 'setp.ne.b32')),
    ('observed old input', source.replace('@%p1 bra LOOP;', 'st.global.u32 [%rd4], %r7;\n@%p1 bra LOOP;')),
    ('call', source.replace('.visible .entry', '.func noop() { ret; }\n.visible .entry')
                   .replace('setp.ne.u32', 'call.uni noop, ();\nsetp.ne.u32')),
):
    expect_compile_failure(build, case, 'integer_probe', 'PTX register')
    print('REJECTED ' + label)
