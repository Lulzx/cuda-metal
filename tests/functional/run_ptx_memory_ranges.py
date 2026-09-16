#!/usr/bin/env python3
"""Numerical local-pointer-cell checks after bounded byte writes."""
from pathlib import Path
import random
import sys
from ptx_test_support import expect_compile_failure, run_integer_case


def fixture(body):
    return r'''.version 7.0
.target sm_80
.address_size 64
.visible .entry integer_probe(.param .u64 .ptr .global input,
 .param .u64 .ptr .global output, .param .u32 count) {
 .local .align 16 .b8 scratch[192];
 .reg .b64 %rd<18>;
 .reg .b32 %r<6>;
 .reg .pred %p<6>;
 ld.param.u64 %rd0,[input];
 ld.param.u64 %rd1,[output];
 ld.param.u32 %r0,[count];
 mov.u32 %r1,%ctaid.x;
 mov.u32 %r2,%ntid.x;
 mov.u32 %r3,%tid.x;
 mad.lo.u32 %r4,%r1,%r2,%r3;
 setp.ge.u32 %p0,%r4,%r0;
 @%p0 bra DONE;
 mul.wide.u32 %rd2,%r4,8;
 add.u64 %rd0,%rd0,%rd2;
 add.u64 %rd1,%rd1,%rd2;
 ld.global.u64 %rd3,[%rd0];
 mov.u64 %rd4,scratch;
 add.u64 %rd5,%rd4,96;
 st.local.u64 [%rd5],%rd0;
''' + body + r'''
LOAD:
 ld.local.u64 %rd8,[%rd5];
 ld.global.u64 %rd9,[%rd8];
 st.global.u64 [%rd1],%rd9;
DONE:
 ret;
}
'''


SINGLE = '''
 setp.lt.u64 %p1,%rd3,33;
 @!%p1 bra LOAD;
 add.u64 %rd6,%rd4,%rd3;
 st.local.u8 [%rd6],47;
'''
LOOP = '''
 setp.gt.u64 %p1,%rd3,32;
 @%p1 bra LOAD;
 setp.eq.u64 %p2,%rd3,0;
 @%p2 bra LOAD;
 mov.u64 %rd7,0;
LOOP:
 add.u64 %rd6,%rd4,%rd7;
 st.local.u8 [%rd6],47;
 add.u64 %rd7,%rd7,1;
 setp.lt.u64 %p3,%rd7,%rd3;
 @%p3 bra LOOP;
'''


PREFIX = '''
 add.u64 %rd10,%rd4,128;
 st.local.u64 [%rd10],10;
 ld.local.u64 %rd11,[%rd10];
 add.u64 %rd12,%rd4,32;
 add.u64 %rd13,%rd12,%rd11;
PREFIX_LOOP:
 st.local.u8 [%rd12],47;
 add.u64 %rd12,%rd12,1;
 setp.ne.u64 %p3,%rd12,%rd13;
 @%p3 bra PREFIX_LOOP;
'''


def reversal_fixture():
    """The Solana reverse-index shape, with a separate pointer cell at 224.

    Each result contains the preserved input length and all 64 array bytes, so
    the CPU reference checks the reversal as well as the pointer-cell proof.
    """
    return r'''.version 7.0
.target sm_80
.address_size 64
.visible .entry integer_probe(.param .u64 .ptr .global input,
 .param .u64 .ptr .global output, .param .u32 count) {
 .local .align 16 .b8 scratch[272];
 .reg .b64 %rd<24>;
 .reg .b32 %r<6>;
 .reg .b16 %rs<2>;
 .reg .pred %p<12>;
 ld.param.u64 %rd0,[input];
 ld.param.u64 %rd1,[output];
 ld.param.u32 %r0,[count];
 mov.u32 %r1,%ctaid.x;
 mov.u32 %r2,%ntid.x;
 mov.u32 %r3,%tid.x;
 mad.lo.u32 %r4,%r1,%r2,%r3;
 setp.ge.u32 %p0,%r4,%r0;
 @%p0 bra DONE;
 mul.wide.u32 %rd2,%r4,8;
 add.u64 %rd0,%rd0,%rd2;
 mul.wide.u32 %rd2,%r4,520;
 add.u64 %rd1,%rd1,%rd2;
 ld.global.u64 %rd3,[%rd0];
 mov.u64 %rd4,scratch;
 add.u64 %rd5,%rd4,224;
 add.u64 %rd6,%rd4,64;
 st.local.u64 [%rd5],%rd0;
 mov.u64 %rd7,0;
INITIALIZE:
 add.u64 %rd8,%rd6,%rd7;
 cvt.u16.u64 %rs0,%rd7;
 st.local.u8 [%rd8],%rs0;
 add.u64 %rd7,%rd7,1;
 setp.lt.u64 %p1,%rd7,64;
 @%p1 bra INITIALIZE;
 setp.gt.u64 %p2,%rd3,64;
 @%p2 bra LOAD;
REVERSE:
 shr.u64 %rd9,%rd3,1;
 setp.eq.u64 %p3,%rd9,0;
 @%p3 bra LOAD;
 mov.u64 %rd10,0;
REVERSE_LOOP:
 setp.lt.u64 %p4,%rd10,%rd9;
 @%p4 bra REVERSE_INDEX;
 bra BAD_INDEX;
REVERSE_INDEX:
 not.b64 %rd11,%rd10;
 add.s64 %rd21,%rd9,%rd11;
 setp.lt.u64 %p5,%rd21,%rd9;
 @%p5 bra SWAP;
 bra BAD_INDEX;
SWAP:
 add.u64 %rd13,%rd6,%rd10;
 add.s64 %rd12,%rd3,%rd11;
 add.u64 %rd14,%rd6,%rd12;
 ld.local.u8 %rs0,[%rd13];
 ld.local.u8 %rs1,[%rd14];
 st.local.u8 [%rd13],%rs1;
 st.local.u8 [%rd14],%rs0;
 add.u64 %rd10,%rd10,1;
 setp.lt.u64 %p6,%rd10,%rd9;
 @%p6 bra REVERSE_LOOP;
LOAD:
 ld.local.u64 %rd15,[%rd5];
 ld.global.u64 %rd16,[%rd15];
 st.global.u64 [%rd1],%rd16;
 mov.u64 %rd17,0;
COPY_RESULT:
 add.u64 %rd20,%rd6,%rd17;
 ld.local.u8 %rs0,[%rd20];
 cvt.u64.u16 %rd21,%rs0;
 shl.b64 %rd18,%rd17,3;
 add.u64 %rd19,%rd1,%rd18;
 st.global.u64 [%rd19+8],%rd21;
 add.u64 %rd17,%rd17,1;
 setp.lt.u64 %p7,%rd17,64;
 @%p7 bra COPY_RESULT;
DONE:
 ret;
BAD_INDEX:
 trap;
}
'''


def reversal_inputs_and_expected():
    values = list(range(60)) + [63, 64, 65, 161, (1 << 64) - 1]
    expected = []
    for length in values:
        array = list(range(64))
        if length <= 64:
            array[:length] = reversed(array[:length])
        expected.extend([length] + array)
    return values, expected


def reversal_negative_fixtures():
    source = reversal_fixture()
    guard = ' @%p2 bra LOAD;\n'
    return [
        ('reversal missing length guard', source.replace(guard, '')),
        # 161 passes neither the old guard nor the intended array bounds. If
        # substituted after checking the old SSA value, reverse[0] is offset160
        # from array base64: it overwrites the pointer cell at byte224.
        ('reversal overwritten length', source.replace(
            'REVERSE:\n', 'REVERSE:\n mov.u64 %rd3,161;\n')),
        ('reversal guard bypass', source.replace(
            ' setp.gt.u64 %p2,%rd3,64;\n',
            ' setp.eq.u64 %p8,%rd3,161;\n @%p8 bra REVERSE;\n'
            ' setp.gt.u64 %p2,%rd3,64;\n')),
    ]


def pointer_iterator_fixture():
    source = reversal_fixture()
    start, end = source.index('REVERSE:\n'), source.index('LOAD:\n')
    return source[:start] + '''ITERATOR:
 setp.eq.u64 %p3,%rd3,0;
 @%p3 bra LOAD;
 mov.u64 %rd10,%rd6;
 add.u64 %rd11,%rd6,1;
 add.u64 %rd12,%rd6,%rd3;
POINTER_LOOP:
 st.local.u8 [%rd10],47;
 setp.eq.u64 %p4,%rd11,%rd12;
 setp.ne.u64 %p5,%rd11,%rd12;
 add.u64 %rd13,%rd11,1;
 mov.u64 %rd10,%rd11;
 selp.b64 %rd11,%rd11,%rd13,%p4;
 @%p5 bra POINTER_LOOP;
''' + source[end:]


def scalar_dynamic_start_fixture():
    source = reversal_fixture()
    start, end = source.index('REVERSE:\n'), source.index('LOAD:\n')
    return source[:start] + '''SCALAR_START:
 setp.eq.u64 %p3,%rd3,64;
 @%p3 bra LOAD;
 mov.u64 %rd10,%rd3;
SCALAR_LOOP:
 add.u64 %rd11,%rd6,%rd10;
 st.local.u8 [%rd11],47;
 add.u64 %rd10,%rd10,1;
 setp.lt.u64 %p4,%rd10,64;
 @%p4 bra SCALAR_LOOP;
''' + source[end:]


def main(build):
    values=list(range(34))+[95,96,97,127,128,(1<<63),(1<<64)-1]
    rng=random.Random(76)
    values += [rng.getrandbits(64) for _ in range(65-len(values))]
    for name,body in [('guard',SINGLE),('bounded loop',LOOP),
                      ('initialized pointer loop',PREFIX),
                      ('four-byte pointer loop',PREFIX.replace('],10;','],12;').replace('%rd12,1;','%rd12,4;')),
                      ('conditional early exit',PREFIX.replace('setp.ne.u64 %p3', 'setp.eq.u64 %p4,%rd3,0;\n @%p4 bra LOAD;\n setp.ne.u64 %p3')),
                      ('copied guard',SINGLE.replace('setp.lt.u64 %p1,%rd3,33;',
                         'mov.u64 %rd10,%rd3;\n setp.lt.u64 %p1,%rd10,33;'))]:
        run_integer_case(build,fixture(body),values,values,name,output_words=1)
    reversal_values, reversal_expected = reversal_inputs_and_expected()
    run_integer_case(build, reversal_fixture(), reversal_values, reversal_expected,
                     'bounded array reversal and pointer cell', output_words=65)
    iterator_expected = []
    for length in reversal_values:
        array = list(range(64))
        if length <= 64:
            array[:length] = [47] * length
        iterator_expected.extend([length] + array)
    run_integer_case(build, pointer_iterator_fixture(), reversal_values, iterator_expected,
                     'bounded pointer iterator and pointer cell', output_words=65)
    joined_iterator = pointer_iterator_fixture().replace('ITERATOR:\n', '''ITERATOR:
 and.b64 %rd21,%rd3,1;
 setp.eq.u64 %p8,%rd21,0;
 @%p8 bra COUNT_READY;
 mov.u64 %rd3,1;
COUNT_READY:
''')
    joined_expected = []
    for length in reversal_values:
        array = list(range(64))
        if length <= 64:
            effective = 1 if length & 1 else length
            array[:effective] = [47] * effective
        joined_expected.extend([length] + array)
    run_integer_case(build, joined_iterator, reversal_values, joined_expected,
                     'joined length pointer iterator', output_words=65)
    dynamic_start_expected = []
    for start in reversal_values:
        array = list(range(64))
        if start <= 64:
            array[start:] = [47] * (64 - start)
        dynamic_start_expected.extend([start] + array)
    run_integer_case(build, scalar_dynamic_start_fixture(), reversal_values, dynamic_start_expected,
                     'bounded dynamic scalar loop start', output_words=65)
    for name,body in [
        ('unguarded',SINGLE.replace('@!%p1 bra LOAD;','')),
        ('overwritten',SINGLE.replace('add.u64 %rd6', 'mov.u64 %rd3,96;\n add.u64 %rd6')),
        ('overlap',SINGLE.replace('33','98')),
        ('guard bypass',SINGLE.replace('setp.lt.u64', 'setp.eq.u64 %p4,%rd3,96;\n @%p4 bra WRITE;\n setp.lt.u64').replace('add.u64 %rd6','WRITE:\n add.u64 %rd6')),
        ('complemented second result',SINGLE.replace('%p1,%rd3','%p2|%p1,%rd3')),
        ('unbounded loop',LOOP.replace('@%p1 bra LOAD;','')),
        ('pointer loop overlap',PREFIX.replace('],10;','],80;')),
        ('pointer loop unknown length',PREFIX.replace('],10;','],%rd3;')),
        ('overwritten scalar length',PREFIX.replace('ld.local.u64 %rd11', 'st.local.u8 [%rd10],%rd3;\n ld.local.u64 %rd11')),
    ]:
        expect_compile_failure(build,fixture(body),'integer_probe','pointer memory proof')
        print('NEGATIVE_PASS',name)
    for name, source in reversal_negative_fixtures():
        expect_compile_failure(build, source, 'integer_probe', 'pointer memory proof')
        print('NEGATIVE_PASS', name)
    for name, source in [
        ('iterator missing bound', pointer_iterator_fixture().replace(' @%p2 bra LOAD;\n', '')),
        ('iterator zero-length bypass', pointer_iterator_fixture().replace(' @%p3 bra LOAD;\n', '')),
        ('iterator skips end', pointer_iterator_fixture().replace('add.u64 %rd13,%rd11,1;',
                                                                 'add.u64 %rd13,%rd11,2;')),
        ('scalar dynamic start missing bound', scalar_dynamic_start_fixture().replace(' @%p2 bra LOAD;\n', '')),
    ]:
        expect_compile_failure(build, source, 'integer_probe', 'pointer memory proof')
        print('NEGATIVE_PASS', name)

if __name__=='__main__': main(Path(sys.argv[1]).resolve())
