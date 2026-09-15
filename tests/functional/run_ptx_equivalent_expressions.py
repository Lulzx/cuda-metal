#!/usr/bin/env python3
"""Equivalent pure expressions may guard a value; stale expressions may not."""
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
    assert len(launches) == 9, launches
    assert all('device=apple_gpu' in line and 'launch_success=true' in line and
               'provenance=generic_ptx_lowering' in line for line in launches), launches
    raise SystemExit(0)

from ptx_test_support import run_integer_case, expect_compile_failure
build=Path(sys.argv[1]).resolve()
source=(Path(__file__).parent/'reference/ptx_equivalent_expressions.ptx').read_text()
pairs=([(0,0),(0,4),(1,1),(0xfffffffd,1),(0xfffffffe,2),(0xffffffff,1),(0x80000000,0x80000004)]*10)[:65]
run_integer_case(build,source,[v for p in pairs for v in p],
                 [99 if ((a+3)&0xffffffff)>=b else a for a,b in pairs],
                 'repeated wrapping add.u32',entry='guarded_relation',word_bits=32,input_words=2,output_words=1)
convert=source.replace('.reg .b64 %rd<6>;', '.reg .b64 %rd<9>;')
convert=convert.replace('add.u32 %r8, %r6, 3;', 'cvt.u64.u32 %rd6, %r6;\ncvt.u64.u32 %rd8, %r7;')
convert=convert.replace('add.u32 %r9, %r6, 3;', 'cvt.u64.u32 %rd7, %r6;')
convert=convert.replace('setp.ge.u32 %p0, %r8, %r7;', 'setp.ge.u64 %p0, %rd6, %rd8;')
convert=convert.replace('setp.ge.u32 %p1, %r9, %r7;', 'setp.ge.u64 %p1, %rd7, %rd8;')
run_integer_case(build,convert,[v for p in pairs for v in p], [99 if a>=b else a for a,b in pairs],
                 'repeated cvt.u64.u32',entry='guarded_relation',word_bits=32,input_words=2,output_words=1)
def split_predecessors(count):
    blocks=''.join(f'bra CONVERT_CHECK_{index};\nCONVERT_CHECK_{index}:\n'
                   for index in range(count))
    return convert.replace('setp.ge.u64 %p0, %rd6, %rd8;',
                           blocks+'setp.ge.u64 %p0, %rd6, %rd8;')
predecessor_convert=split_predecessors(1)
run_integer_case(build,predecessor_convert,[v for p in pairs for v in p],
                 [99 if a>=b else a for a,b in pairs],
                 'predecessor cvt.u64.u32',entry='guarded_relation',word_bits=32,input_words=2,output_words=1)
run_integer_case(build,split_predecessors(8),[v for p in pairs for v in p],
                 [99 if a>=b else a for a,b in pairs],
                 '8-block predecessor boundary',entry='guarded_relation',word_bits=32,input_words=2,output_words=1)
wide=source.replace('.reg .b64 %rd<6>;', '.reg .b64 %rd<11>;')
for a,b in [('%r6','%rd6'),('%r7','%rd7'),('%r8','%rd8'),('%r9','%rd9'),('%r1','%rd10')]:
 wide=wide.replace(a,b)
wide=wide.replace('ld.global.u32','ld.global.u64').replace('st.global.u32','st.global.u64')
wide=wide.replace('[%rd4+4]','[%rd4+8]').replace('%r0, 8;', '%r0, 16;').replace('%r0, 4;', '%r0, 8;')
wide=wide.replace('add.u32 %rd','add.s64 %rd').replace('setp.ge.u32 %p0','setp.eq.b64 %p0').replace('setp.ge.u32 %p1','setp.eq.b64 %p1')
pairs64=([(0,3),(1,0),(0xffffffffffffffff,2),(0xfffffffffffffffe,0),(0x8000000000000000,0x8000000000000003)]*13)
run_integer_case(build,wide,[v for p in pairs64 for v in p],
                 [99 if ((a+3)&0xffffffffffffffff)==b else a for a,b in pairs64],
                 'repeated add.s64 equality',entry='guarded_relation',word_bits=64,input_words=2,output_words=1)
signed32=source.replace('add.u32 %r8', 'add.s32 %r8').replace('add.u32 %r9', 'add.s32 %r9')
run_integer_case(build,signed32,[v for p in pairs for v in p],
                 [99 if ((a+3)&0xffffffff)>=b else a for a,b in pairs],
                 'add.s32 wrapping',entry='guarded_relation',word_bits=32,input_words=2,output_words=1)
unsigned64=wide.replace('add.s64 %rd', 'add.u64 %rd').replace('setp.eq.b64','setp.ge.u64')
run_integer_case(build,unsigned64,[v for p in pairs64 for v in p],
                 [99 if ((a+3)&0xffffffffffffffff)>=b else a for a,b in pairs64],
                 'add.u64 ordering',entry='guarded_relation',word_bits=64,input_words=2,output_words=1)
register_add=source.replace('add.u32 %r8, %r6, 3;', 'add.u32 %r8, %r6, %r2;').replace('add.u32 %r9, %r6, 3;', 'add.u32 %r9, %r6, %r2;')
run_integer_case(build,register_add,[v for p in pairs for v in p],
                 [99 if ((a+65)&0xffffffff)>=b else a for a,b in pairs],
                 'two-register add',entry='guarded_relation',word_bits=32,input_words=2,output_words=1)
def padded(count):
    case=source.replace('.reg .b32 %r<10>;', '.reg .b32 %r<200>;')
    padding=''.join(f'add.u32 %r{100+i}, %r6, {10+i};\n' for i in range(count))
    return case.replace('setp.ge.u32 %p0', padding+'setp.ge.u32 %p0')
run_integer_case(build,padded(63),[v for p in pairs for v in p],
                 [99 if ((a+3)&0xffffffff)>=b else a for a,b in pairs],
                 '64-expression capacity boundary',entry='guarded_relation',word_bits=32,input_words=2,output_words=1)
for label,case in [
 ('input overwrite',source.replace('JOIN:\n','JOIN:\nmov.u32 %r6, 0;\n')),
 ('predicated input overwrite',source.replace('JOIN:\n','JOIN:\n@%p0 mov.u32 %r6, 0;\n')),
 ('different constant',source.replace('add.u32 %r9, %r6, 3;','add.u32 %r9, %r6, 4;')),
 ('different opcode',source.replace('add.u32 %r9','sub.u32 %r9')),
 ('observable undefined load',source.replace('JOIN:\n','JOIN:\nst.global.u32 [%rd5], %r1;\n')),
 ('capacity exhausted',padded(64)),
 ('comparison input overwrite',source.replace('JOIN:\n','JOIN:\nmov.u32 %r7, 0;\n')),
 ('expression result overwritten',source.replace('JOIN:\n','JOIN:\nmov.u32 %r8, 0;\n')),
 ('predicated first expression',source.replace('add.u32 %r8', '@%p3 add.u32 %r8')),
 ('predicated second expression',source.replace('add.u32 %r9', '@%p0 add.u32 %r9')),
 ('in-place source expression',source.replace('add.u32 %r8, %r6, 3;', 'add.u32 %r6, %r6, 3;').replace('%p0, %r8, %r7;', '%p0, %r6, %r7;')),
 ('call',source.replace('.visible .entry','.func noop() { ret; }\n.visible .entry').replace('JOIN:\n','JOIN:\ncall.uni noop, ();\n')),
 ('predecessor input overwrite',predecessor_convert.replace('CONVERT_CHECK_0:\n', 'CONVERT_CHECK_0:\nmov.u32 %r6, 0;\n')),
 ('predecessor merge',predecessor_convert.replace('cvt.u64.u32 %rd6, %r6;',
    'setp.eq.u32 %p3, %r6, 0;\n@%p3 bra CONVERT_CHECK_0;\ncvt.u64.u32 %rd6, %r6;')),
 ('predecessor depth exhausted',split_predecessors(9)),
]:
 expect_compile_failure(build,case,'guarded_relation','PTX register')
 print('REJECTED '+label)
