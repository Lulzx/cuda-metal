#!/usr/bin/env python3
"""Typed launch metadata must describe the emitted buffer/scalar signature."""
from pathlib import Path
import subprocess
import sys
import tempfile
from ptx_test_support import run_integer_case
from run_ptx_integer_widths import TEMPLATE

build = Path(sys.argv[1]).resolve()
emit = sys.argv[2] if len(sys.argv) > 2 else 'msl'
if emit == 'metallib':
    for tool in ('metal', 'metallib'):
        check = subprocess.run(['xcrun', '--sdk', 'macosx', tool, '--version'], capture_output=True)
        if check.returncode:
            print('SKIP: public offline Metal toolchain unavailable')
            sys.exit(77)

values = [0, 1, 0x7fffffff, 0x80000000, 0xffffffff] + list(range(257))
expected = [v for value in values for v in (value, value)]
base = TEMPLATE.replace('BODY', 'ld.global.u32 %r5, [%rd5];\nmov.u32 %r6, %r5;')
base = base.replace('mul.wide.u32 %rd3, %r2, 8;', 'mul.wide.u32 %rd3, %r2, 4;')
base = base.replace('mul.wide.u32 %rd4, %r2, 16;', 'mul.wide.u32 %rd4, %r2, 8;')
base = base.replace('st.global.u64 [%rd6], %rd7;', 'st.global.u32 [%rd6], %r5;')
base = base.replace('st.global.u64 [%rd6+8], %rd8;', 'st.global.u32 [%rd6+4], %r6;')
abi = ['CUMETAL_ABI_V2', 'kernel integer_probe', 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
# Both existing backends retain their directly inferred two-buffer/count ABI.
for backend in ('legacy', 'cumetal-ir'):
    run_integer_case(build, base, values, expected, backend + ' direct ABI',
                     backend=backend, emit=emit, abi_lines=abi, word_bits=32)

# Both addresses combine an unannotated pointer with a scalar-derived offset.
# Parser provenance cannot choose between them. The typed importer recovers
# pointer arguments from dereferences; the sidecar must agree with its signature.
source = base.replace('.reg .b64 %rd<10>;', '.reg .b64 %rd<11>;')
source = source.replace('add.u64 %rd5, %rd1, %rd3;',
    'mul.wide.u32 %rd10, %r1, 0;\nadd.u64 %rd3, %rd3, %rd10;\nadd.u64 %rd5, %rd1, %rd3;')
source = source.replace('add.u64 %rd6, %rd2, %rd4;',
    'add.u64 %rd4, %rd4, %rd10;\nadd.u64 %rd6, %rd2, %rd4;')
run_integer_case(build, source, values, expected, 'recovered pointer ABI', emit=emit, abi_lines=abi, word_bits=32)

# A real 64-bit scalar stays bytes even when its arithmetic is only subtraction.
source = base.replace('.param .u32 count', '.param .u64 count')
source = source.replace('.param .u64 input', '.param .u64 .ptr input')
source = source.replace('ld.param.u32 %r1, [count];',
    'ld.param.u64 %rd9, [count];\ncvt.u32.u64 %r1, %rd9;')
source = source.replace('ld.global.u32 %r5, [%rd5];', 'sub.u64 %rd7, 31, %rd9;\ncvt.u32.u64 %r5, %rd7;')
scalar = (31 - len(values)) & ((1 << 32)-1)
for backend in ('cumetal-ir',):
    run_integer_case(build, source, values, [scalar, scalar] * len(values),
                     backend + ' scalar64 ABI', backend=backend, emit=emit,
                     abi_lines=abi[:-1] + ['arg bytes 8'], word_bits=32)

# The sidecar and runtime retain source argument order, including a leading scalar.
reordered = base.replace('.param .u64 input, .param .u64 output, .param .u32 count',
                         '.param .u32 count, .param .u64 input, .param .u64 output')
run_integer_case(build, reordered, values, expected, 'scalar-first argument order',
                 emit=emit, word_bits=32, argument_order=(2, 0, 1),
                 abi_lines=abi[:3] + ['arg bytes 4', 'arg buffer 8', 'arg buffer 8'])

# Metadata checks supplement numerical tests with unused scalars, static memory,
# and a hidden module-global buffer. The hidden buffer is not a user launch
# argument, so it must not appear in the `arg` list -- but it is described
# separately, because a driver-API caller has no registration to own that
# storage and would otherwise leave the binding unpopulated.
def check_metadata(source, expected_abi, backend="cumetal-ir"):
    with tempfile.TemporaryDirectory(prefix='cumetal-abi-layout-') as work:
        ptx = Path(work) / 'input.ptx'
        output = Path(work) / ('output.metallib' if emit == 'metallib' else 'output.metal')
        ptx.write_text(source)
        subprocess.run([str(build / 'cumetalc'), str(ptx), '--backend=' + backend,
                        '--ptx-strict', '--entry', 'integer_probe', '--emit=' + emit,
                        '-o', str(output)], check=True, capture_output=True)
        actual = Path(str(output) + '.cumetal-abi').read_text().splitlines()
        assert actual == expected_abi, (actual, expected_abi)

unused = base.replace('.param .u32 count)', '.param .u32 count, .param .u64 unused)')
for backend in ('legacy', 'cumetal-ir'):
    check_metadata(unused, abi + ['arg bytes 8'], backend)
shared = base.replace('.reg .b64 %rd<10>;', '.shared .align 16 .b8 scratch[32];\n.reg .b64 %rd<10>;')
shared = shared.replace('DONE:\nret;', 'DONE:\nmov.u64 %rd9, scratch;\nst.shared.u32 [%rd9], 7;\nret;')
check_metadata(shared, abi[:2] + ['shared 32'] + abi[3:])
hidden = base.replace('.visible .entry', '.global .align 4 .u32 hidden;\n.visible .entry')
hidden = hidden.replace('DONE:\nret;', 'DONE:\nmov.u64 %rd9, hidden;\nst.global.u32 [%rd9], 7;\nret;')
check_metadata(hidden, abi + ['global hidden 4 4 -'])

# An initialized device global carries its source bytes, so a module-owned
# allocation can start at the right value instead of zero.
seeded = base.replace('.visible .entry', '.global .align 4 .u32 seeded = 5;\n.visible .entry')
seeded = seeded.replace('DONE:\nret;', 'DONE:\nmov.u64 %rd9, seeded;\nst.global.u32 [%rd9], 7;\nret;')
check_metadata(seeded, abi + ['global seeded 4 4 05000000'])
print('PASS: ordered arguments, unused scalar, static shared memory, hidden buffer metadata')
