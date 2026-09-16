#!/usr/bin/env python3
"""Numerical local pointer-cell refinement, with reused slot addresses."""
import argparse
from pathlib import Path
import subprocess
import sys

from ptx_test_support import run_integer_case


CONSTANTS = [0, 0x0123456789abcdef, 0x8000000000000001, 0xffffffffffffffff]


def fixture(kind):
    source = '''.version 7.1
.target sm_80
.address_size 64
.global .align 8 .b8 literal_data[32] = {0,0,0,0,0,0,0,0,239,205,171,137,103,69,35,1,1,0,0,0,0,0,0,128,255,255,255,255,255,255,255,255};
.visible .entry integer_probe(.param .u64 .ptr .global input,
                             .param .u64 .ptr .global output, .param .u32 count) {
 .local .align 16 .b8 depot[176];
 .reg .b64 %base, %local, %cell, %slot, %stored, %loaded, %next, %index;
 .reg .b64 %input, %output, %offset, %word, %answer, %selection;
 .reg .b32 %lane, %block, %threads, %count, %scratch;
 .reg .pred %done, %again;
 ld.param.u64 %input, [input];
 ld.param.u64 %output, [output];
 ld.param.u32 %count, [count];
 mov.u32 %lane, %tid.x;
 mov.u32 %block, %ctaid.x;
 mov.u32 %threads, %ntid.x;
 mad.lo.u32 %lane, %block, %threads, %lane;
 setp.ge.u32 %done, %lane, %count;
 @%done bra DONE;
 mul.wide.u32 %offset, %lane, 8;
 add.u64 %input, %input, %offset;
 add.u64 %output, %output, %offset;
 ld.global.u64 %word, [%input];
 mov.b64 %local, depot;
 cvta.local.u64 %base, %local;
 st.local.u64 [%base], 0;
 add.u64 %cell, %base, 64;
 cvta.to.local.u64 %slot, %cell;
'''
    if kind == 'constant':
        source += ''' mov.u64 %stored, literal_data;
 cvta.global.u64 %stored, %stored;
 and.b64 %selection, %word, 3;
 shl.b64 %selection, %selection, 3;
 add.u64 %stored, %stored, %selection;
'''
    elif kind == 'device':
        source += ' mov.b64 %stored, %input;\n'
    elif kind == 'private':
        source += ' add.u64 %stored, %base, 16;\n st.local.u64 [%stored], %word;\n'
    else:
        raise ValueError(kind)
    return source + ''' st.local.u64 [%slot], %stored;
 ld.local.u64 %loaded, [%slot];
 ld.u64 %answer, [%loaded];
 st.local.u64 [%slot], 0;
 st.local.u64 [%slot+8], 0;
 st.local.u64 [%slot+16], 0;
 st.local.u64 [%slot+24], 0;
 st.local.u64 [%slot+32], 0;
 st.local.u64 [%slot+40], 0;
 st.local.u64 [%slot+48], 0;
 st.local.u64 [%slot+56], 0;
 or.b64 %next, %cell, 4;
 mov.u64 %index, 0;
LOOP:
 ld.u8 %scratch, [%cell];
 add.u64 %index, %index, 1;
 setp.lt.u64 %again, %index, 16;
 selp.b64 %cell, %next, %cell, %again;
 add.u64 %next, %next, 4;
 @%again bra LOOP;
 st.global.u64 [%output], %answer;
DONE:
 ret;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--gpu-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.gpu_child:
        result = subprocess.run([sys.executable, __file__, *sys.argv[1:], '--gpu-child'],
                                capture_output=True, text=True, timeout=120)
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
        if result.returncode:
            return result.returncode
        launches = [line for line in result.stderr.splitlines()
                    if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        assert len(launches) == 3, launches
        assert all('device=apple_gpu' in line and 'launch_success=true' in line and
                   'provenance=generic_ptx_lowering' in line and 'semantic_quality=exact' in line
                   for line in launches), launches
        return 0

    values = [(i * 0x9e3779b97f4a7c15) & ((1 << 64) - 1) for i in range(65)]
    values[:4] = CONSTANTS
    abi = ['CUMETAL_ABI_V2', 'kernel integer_probe', 'shared 0',
           'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
    for kind in ('constant', 'device', 'private'):
        expected = [CONSTANTS[value & 3] for value in values] if kind == 'constant' else values
        run_integer_case(args.build.resolve(), fixture(kind), values, expected,
                         f'{kind} pointer payload survives later scalar reuse of its slot address',
                         output_words=1, abi_lines=abi,
                         artifacts_dir=args.artifacts / kind if args.artifacts else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
