#!/usr/bin/env python3
"""Recomputed unsigned comparisons must retain the load's path constraint."""
from pathlib import Path
import sys
import subprocess

# Capture the driver's native stderr so a successful numerical run must also
# prove every variant actually dispatched through generic PTX on the Apple GPU.
if '--gpu-child' not in sys.argv:
    result = subprocess.run([sys.executable, __file__, *sys.argv[1:], '--gpu-child'],
                            capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    launches = [line for line in result.stderr.splitlines()
                if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
    assert len(launches) == 24, launches
    assert all('device=apple_gpu' in line and 'launch_success=true' in line and
               'provenance=generic_ptx_lowering' in line for line in launches), launches
    raise SystemExit(0)
from ptx_test_support import run_integer_case, expect_compile_failure

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_repeated_relation.ptx').read_text()
for width in (32, 64):
    base = source
    if width == 64:
        base = base.replace('.reg .b64 %rd<6>;', '.reg .b64 %rd<6>;\n.reg .b64 %v<3>;')
        base = base.replace('%r6', '%v0').replace('%r7', '%v1').replace('%r1', '%v2')
        base = base.replace('ld.global.u32', 'ld.global.u64').replace('st.global.u32', 'st.global.u64')
        base = base.replace('[%rd4+4]', '[%rd4+8]')
        base = base.replace('%r0, 8;', '%r0, 16;').replace('%r0, 4;', '%r0, 8;')
        base = base.replace('setp.ge.u32 %p0', 'setp.ge.u64 %p0').replace('setp.ge.u32 %p1', 'setp.ge.u64 %p1')
    left, right = ('%r6', '%r7') if width == 32 else ('%v0', '%v1')
    cases = [(0, 0), (0, 1), (1, 0), (1 << (width - 1), 0),
             (0, 1 << (width - 1)), ((1 << width) - 1, (1 << width) - 1),
             ((1 << width) - 2, (1 << width) - 1), ((1 << width) - 1, 7)]
    pairs = (cases * 9)[:65]
    for relation, compare, complement, swapped in (
        ('ge', lambda a, b: a >= b, 'lt', 'le'),
        ('lt', lambda a, b: a < b, 'ge', 'gt'),
        ('gt', lambda a, b: a > b, 'le', 'lt'),
        ('le', lambda a, b: a <= b, 'gt', 'ge'),
    ):
        case = base
        for predicate in ('%p0', '%p1'):
            case = case.replace(f'setp.ge.u{width} {predicate}', f'setp.{relation}.u{width} {predicate}')
        for form, variant in (
            ('same', case),
            ('complement', case.replace('setp.' + relation + '.u' + str(width) + ' %p1',
                                       'setp.' + complement + '.u' + str(width) + ' %p1').replace('@!%p1 bra USE;', '@%p1 bra USE;')),
            ('swapped', case.replace('%p1, ' + left + ', ' + right + ';', '%p1, ' + right + ', ' + left + ';').replace(
                'setp.' + relation + '.u' + str(width) + ' %p1', 'setp.' + swapped + '.u' + str(width) + ' %p1')),
        ):
            run_integer_case(build, variant, [x for pair in pairs for x in pair],
                             [99 if compare(a, b) else a for a, b in pairs],
                             f'u{width} {relation} {form}', entry='guarded_relation',
                             word_bits=width, input_words=2, output_words=1)

for index, case in enumerate((
    source.replace('JOIN:\n', 'JOIN:\nmov.u32 %r6, 0;\n'),
    source.replace('.visible .entry', '.func unknown_effect() { ret; }\n.visible .entry').replace(
        'JOIN:\n', 'JOIN:\ncall.uni unknown_effect, ();\n'),
    source.replace('JOIN:\n', 'JOIN:\nmov.u32 %r7, 0;\n'),
    source.replace('JOIN:\n', 'JOIN:\n@%p0 mov.u32 %r7, 0;\n'),
    source.replace('JOIN:\n', 'JOIN:\nst.global.u32 [%rd5], %r1;\n'),
    source.replace('%p1, %r6, %r7;', '%p1, %r6, %r2;'),
    source.replace('setp.ge.u32 %p1', 'setp.ge.u64 %p1'),
    source.replace('setp.ge.u32 %p1', 'setp.ge.s32 %p1'),
    source.replace('setp.ge.u32 %p1', 'setp.eq.u32 %p1'),
    source.replace('setp.ge.u32 %p1', 'setp.le.u32 %p1'),
)):
    expect_compile_failure(build, case, 'guarded_relation', 'PTX register')
    print(f'REJECTED invalid comparison path {index}')
