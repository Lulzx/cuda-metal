#!/usr/bin/env python3
"""Numerical local-pointer-cell checks after bounded byte writes."""
from pathlib import Path
import os
import random
import signal
import subprocess
import sys
import tempfile
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


def sentinel_iterator_fixture(literal_end=False):
    """A zero selected at the end controls whether the cursor loops again."""
    source = pointer_iterator_fixture().replace(
        ' setp.ne.u64 %p5,%rd11,%rd12;\n',
        ' selp.b64 %rd22,0,%rd11,%p4;\n setp.ne.u64 %p5,%rd22,0;\n')
    if literal_end:
        source = source.replace(' add.u64 %rd12,%rd6,%rd3;\n',
                                ' add.u64 %rd12,%rd6,64;\n')
    return source


def endpoint_fixture(copied_pointer=False):
    """A saved pointer at 256 survives writes admitted by both scalar guards.

    Each lane returns the two original input words through the saved pointer,
    the byte actually read back from scratch, and whether the write executed.
    The no-write path returns 511, outside the possible byte-result range.
    """
    pointer_use = ' mov.u64 %rd12,%rd9;\n'
    if copied_pointer:
        pointer_use = '''
 and.b64 %rd15,%rd3,1;
 setp.ne.u64 %p3,%rd15,0;
 @%p3 bra COPY_RIGHT;
 mov.u64 %rd12,%rd9;
 bra POINTER_READY;
COPY_RIGHT:
 mov.u64 %rd13,%rd9;
 mov.u64 %rd12,%rd13;
POINTER_READY:
'''
    return r'''.version 7.0
.target sm_80
.address_size 64
.visible .entry integer_probe(.param .u64 .ptr .global input,
 .param .u64 .ptr .global output, .param .u32 count) {
 .local .align 16 .b8 scratch[272];
 .reg .b64 %rd<18>;
 .reg .b32 %r<6>;
 .reg .b16 %rs<2>;
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
 mul.wide.u32 %rd2,%r4,16;
 add.u64 %rd0,%rd0,%rd2;
 mul.wide.u32 %rd2,%r4,32;
 add.u64 %rd1,%rd1,%rd2;
 ld.global.u64 %rd3,[%rd0];
 ld.global.u64 %rd4,[%rd0+8];
 mov.u64 %rd5,scratch;
 add.u64 %rd6,%rd5,256;
 st.local.u64 [%rd6],%rd0;
 mov.u64 %rd10,511;
 mov.u64 %rd11,0;
 setp.gt.u64 %p1,%rd3,255;
 @%p1 bra LOAD;
 setp.eq.s64 %p2,%rd3,255;
 @%p2 bra LOAD;
WRITE:
 add.u64 %rd7,%rd5,%rd3;
 cvt.u16.u64 %rs0,%rd4;
 st.local.u8 [%rd7+1],%rs0;
 ld.local.u8 %rs1,[%rd7+1];
 cvt.u64.u16 %rd10,%rs1;
 mov.u64 %rd11,1;
LOAD:
 ld.local.u64 %rd9,[%rd6];
''' + pointer_use + r'''
 ld.global.u64 %rd14,[%rd12];
 ld.global.u64 %rd16,[%rd12+8];
 st.global.u64 [%rd1],%rd14;
 st.global.u64 [%rd1+8],%rd16;
 st.global.u64 [%rd1+16],%rd10;
 st.global.u64 [%rd1+24],%rd11;
DONE:
 ret;
}
'''


def affine_endpoint_fixture(compound=False):
    """The compared and written offsets are different modulo64 siblings."""
    source = endpoint_fixture()
    source = source.replace(''' setp.gt.u64 %p1,%rd3,255;
 @%p1 bra LOAD;
 setp.eq.s64 %p2,%rd3,255;
 @%p2 bra LOAD;
WRITE:
 add.u64 %rd7,%rd5,%rd3;
''', ''' add.u64 %rd17,%rd3,2;
 setp.gt.u64 %p1,%rd17,254;
 @%p1 bra LOAD;
WRITE:
 add.u64 %rd15,%rd3,3;
 add.u64 %rd7,%rd5,%rd15;
''')
    source = source.replace('[%rd7+1]', '[%rd7]')
    if compound:
        source = source.replace(' setp.gt.u64 %p1,%rd17,254;\n', ''' setp.gt.u64 %p1,%rd17,254;
 and.b64 %rd8,%rd4,1;
 setp.ne.u64 %p4,%rd8,0;
 or.pred %p1,%p1,%p4;
''')
    return source


def endpoint_inputs_and_expected(affine=False, compound=False):
    maximum = (1 << 64) - 1
    indices = [0, 1, 2, 127, 128, 253, 254, 255, 256, 257, 279, 280,
               (1 << 32) - 1, 1 << 32, (1 << 63) - 1, 1 << 63,
               maximum - 1, maximum]
    while len(indices) < 65:
        lane = len(indices)
        indices.append((lane * 37) % 255 if lane % 3 == 0 else
                       255 if lane % 3 == 1 else (1 << 63) + lane)
    values, expected = [], []
    for lane, index in enumerate(indices):
        payload = ((lane + 1) * 0x9e3779b97f4a7c15 ^ 0x6b8b4567327b23c6) & maximum
        if lane < 5:
            payload = (payload & ~255) | [0, 1, 127, 128, 255][lane]
        written = ((index + 2) & maximum) <= 254 if affine else index < 255
        if compound:
            written = written and not (payload & 1)
        values.extend((index, payload))
        expected.extend((index, payload, payload & 255 if written else 511, int(written)))
    return values, expected


def endpoint_worker(build, copied_pointer, artifacts, affine=False, compound=False):
    values, expected = endpoint_inputs_and_expected(affine, compound)
    source = affine_endpoint_fixture(compound) if affine else endpoint_fixture(copied_pointer)
    label = 'compound affine sibling guard' if compound else 'affine sibling endpoint guard' if affine else (
        'endpoint guard with ' + ('copied pointer' if copied_pointer else 'direct pointer'))
    run_integer_case(build, source, values, expected, label,
                     input_words=2, output_words=4,
                     abi_lines=['CUMETAL_ABI_V2', 'kernel integer_probe', 'shared 0',
                                'arg buffer 8', 'arg buffer 8', 'arg bytes 4'],
                     artifacts_dir=artifacts)


def run_endpoint_case(build, copied_pointer=False, affine=False, compound=False):
    # Keep the normal fixture harness, while checking actual launch provenance
    # from an isolated child. Neither rejected variants nor input references
    # invoke the GPU. Retained source must equal the source selected here.
    with tempfile.TemporaryDirectory(prefix='cumetal-endpoint-memory-') as directory:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(build), '--endpoint-worker',
             'compound' if compound else 'affine' if affine else 'copied' if copied_pointer else 'direct', directory],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
            env=dict(os.environ, CUMETAL_TRACE_GPU='1', CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0'))
        try:
            stdout, stderr = child.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.communicate(timeout=5)
            raise AssertionError('endpoint memory fixture exceeded 90 seconds')
        print(stdout, end='')
        print(stderr, end='', file=sys.stderr)
        launches = [line for line in stderr.splitlines()
                    if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        assert child.returncode == 0, 'endpoint memory fixture failed'
        assert len(launches) == 1 and all(token in launches[0] for token in (
            'kernel="integer_probe"', 'device=apple_gpu ', 'launch_success=true ',
            'source=generic_ptx ', 'provenance=generic_ptx_lowering ', 'semantic_quality=exact '
        )), 'endpoint memory fixture lacks exact generic Apple GPU provenance'
        assert '65 inputs, output values, guards' in stdout
        source = affine_endpoint_fixture(compound) if affine else endpoint_fixture(copied_pointer)
        assert (Path(directory) / 'test.ptx').read_text() == source


def endpoint_negative_fixtures():
    source = endpoint_fixture()
    return [
        ('endpoint missing upper bound', source.replace(' @%p1 bra LOAD;\n', '')),
        ('endpoint missing exclusion', source.replace(' @%p2 bra LOAD;\n', '')),
        ('endpoint guard bypass', source.replace(
            ' setp.gt.u64 %p1,%rd3,255;\n',
            ' setp.eq.u64 %p4,%rd3,255;\n @%p4 bra WRITE;\n'
            ' setp.gt.u64 %p1,%rd3,255;\n')),
        ('endpoint overwritten index', source.replace('WRITE:\n', 'WRITE:\n mov.u64 %rd3,255;\n')),
        ('endpoint overlapping displacement', source.replace('[%rd7+1]', '[%rd7+2]')),
        ('endpoint overwritten predicate', source.replace(
            ' @%p2 bra LOAD;\n', ' setp.eq.u64 %p2,%rd4,0;\n @%p2 bra LOAD;\n')),
    ]


def affine_endpoint_negative_fixtures():
    source = affine_endpoint_fixture()
    return [
        ('affine endpoint guard bypass', source.replace(
            ' add.u64 %rd17,%rd3,2;\n',
            ' setp.eq.u64 %p4,%rd3,253;\n @%p4 bra WRITE;\n add.u64 %rd17,%rd3,2;\n')),
        ('affine endpoint foreign base', source.replace(' add.u64 %rd15,%rd3,3;\n',
                                                       ' add.u64 %rd15,%rd4,3;\n')),
        ('affine endpoint overlapping offset', source.replace(' add.u64 %rd15,%rd3,3;\n',
                                                             ' add.u64 %rd15,%rd3,4;\n')),
    ]


def guarded_origins_fixture(copied=False):
    """Two bounded pointer origins carried through nested loops (#76)."""
    body = '''
 mov.u64 %rd14,0;
INITIALIZE:
 add.u64 %rd15,%rd4,%rd14;
 st.local.u8 [%rd15],0;
 add.u64 %rd14,%rd14,1;
 setp.lt.u64 %p5,%rd14,64;
 @%p5 bra INITIALIZE;
 shr.u64 %rd11,%rd3,32;
 setp.gt.u64 %p1,%rd11,32;
 @%p1 bra LOAD;
 add.u64 %rd10,%rd4,%rd11;
 mov.u32 %r5,0;
OUTER:
 and.b64 %rd12,%rd3,255;
 setp.gt.u64 %p2,%rd12,32;
 @%p2 bra LOAD;
 setp.eq.u64 %p2,%rd12,0;
 @%p2 bra NEXT;
 mov.u64 %rd7,0;
INNER:
 add.u64 %rd6,%rd10,%rd7;
 st.local.u8 [%rd6],47;
 add.u64 %rd7,%rd7,1;
 setp.lt.u64 %p3,%rd7,%rd12;
 @%p3 bra INNER;
NEXT:
 add.u32 %r5,%r5,1;
 setp.ge.u32 %p4,%r5,2;
 @%p4 bra LOAD;
 shr.u64 %rd13,%rd3,8;
 and.b64 %rd11,%rd13,255;
 setp.gt.u64 %p1,%rd11,32;
 @%p1 bra LOAD;
 add.u64 %rd10,%rd4,%rd11;
 bra OUTER;
'''
    if copied:
        body = body.replace('INNER:\n', '''INNER:
 mov.u64 %rd17,%rd10;
 setp.eq.u32 %p5,%r5,0;
 @%p5 bra COPIED;
 mov.u64 %rd10,%rd17;
COPIED:
''')
    source = fixture(body).replace(' add.u64 %rd1,%rd1,%rd2;',
        ' mul.wide.u32 %rd2,%r4,520;\n add.u64 %rd1,%rd1,%rd2;')
    return source.replace(' st.global.u64 [%rd1],%rd9;', ''' st.global.u64 [%rd1],%rd9;
 mov.u64 %rd14,0;
OUTPUT:
 add.u64 %rd15,%rd4,%rd14;
 ld.local.u8 %r5,[%rd15];
 cvt.u64.u32 %rd16,%r5;
 mul.lo.u64 %rd15,%rd14,8;
 add.u64 %rd15,%rd1,%rd15;
 st.global.u64 [%rd15+8],%rd16;
 add.u64 %rd14,%rd14,1;
 setp.lt.u64 %p5,%rd14,64;
 @%p5 bra OUTPUT;''')


def guarded_origins_inputs():
    triples = [(a, b, n) for a, b in ((0, 32), (32, 0), (1, 31), (32, 32),
                                    (33, 0), (0, 33), (96, 0), (0, 96))
               for n in (0, 1, 31, 32, 33)]
    triples += [(i % 33, (i * 13) % 33, (i * 7) % 33) for i in range(25)]
    values, expected = [], []
    for initial, replacement, length in triples:
        value = (initial << 32) | (replacement << 8) | length
        array = [0] * 64
        if initial <= 32 and length <= 32:
            array[initial:initial + length] = [47] * length
            if replacement <= 32:
                array[replacement:replacement + length] = [47] * length
        values.append(value)
        expected.extend([value] + array)
    return values, expected


def guarded_origins_negatives():
    source = guarded_origins_fixture()
    guard = ' setp.gt.u64 %p1,%rd11,32;\n @%p1 bra LOAD;\n'
    creation = ' add.u64 %rd10,%rd4,%rd11;'
    yield 'initial origin unbounded', source.replace(guard, '', 1)
    index = source.rfind(guard)
    yield 'replacement origin unbounded', source[:index] + source[index:].replace(guard, '', 1)
    yield 'origin guard bypass', source.replace(guard, guard.replace(' @%p1 bra LOAD;\n', ''), 1)
    yield 'overlapping origin', source.replace(creation, ' add.u64 %rd10,%rd4,96;', 1)
    index = source.rfind(creation)
    yield 'overlapping replacement', source[:index] + source[index:].replace(
        creation, ' add.u64 %rd10,%rd4,96;', 1)
    yield 'arithmetic recurrence', source[:index] + source[index:].replace(
        creation, ' add.u64 %rd10,%rd10,64;', 1)
    yield 'overlapping displacement', source.replace('st.local.u8 [%rd6],47;',
                                                    'st.local.u8 [%rd6+96],47;')
    yield 'overlapping width', source.replace('st.local.u8 [%rd6],47;',
        'st.local.v2.u64 [%rd6+32],{47,47};')
    # A guard on a newer loop value must not bound the old value captured by
    # the remembered pointer. Rejected before dispatch for arbitrary inputs.
    yield 'changed captured scalar', fixture('''
 mov.u64 %rd11,%rd3;
 mov.u64 %rd10,%rd4;
 mov.u32 %r5,0;
REMEMBER:
 setp.eq.u32 %p1,%r5,0;
 @!%p1 bra WRITE;
 add.u64 %rd10,%rd4,%rd11;
 mov.u64 %rd11,0;
 mov.u32 %r5,1;
 bra REMEMBER;
WRITE:
 setp.gt.u64 %p2,%rd11,32;
 @%p2 bra LOAD;
 st.local.u8 [%rd10],47;
''')


def main(build):
    values=list(range(34))+[95,96,97,127,128,(1<<63),(1<<64)-1]
    rng=random.Random(76)
    values += [rng.getrandbits(64) for _ in range(65-len(values))]
    for name,body in [('guard',SINGLE),('bounded loop',LOOP),
                      ('guard after pointer creation', '''
 add.u64 %rd6,%rd4,%rd3;
 setp.lt.u64 %p1,%rd3,33;
 @!%p1 bra LOAD;
 st.local.u8 [%rd6],47;
'''),
                      ('late guard tightens captured mask', '''
 and.b64 %rd10,%rd3,255;
 add.u64 %rd6,%rd4,%rd10;
 setp.lt.u64 %p1,%rd10,33;
 @!%p1 bra LOAD;
 st.local.u8 [%rd6],47;
'''),
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
    run_integer_case(build, sentinel_iterator_fixture(), reversal_values, iterator_expected,
                     'zero-sentinel pointer iterator', output_words=65)
    literal_iterator_expected = []
    for length in reversal_values:
        array = [47] * 64 if 0 < length <= 64 else list(range(64))
        literal_iterator_expected.extend([length] + array)
    run_integer_case(build, sentinel_iterator_fixture(True), reversal_values, literal_iterator_expected,
                     'literal-end zero-sentinel pointer iterator', output_words=65)
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
    for copied_pointer in (False, True):
        run_endpoint_case(build, copied_pointer)
    run_endpoint_case(build, affine=True)
    run_endpoint_case(build, affine=True, compound=True)
    origin_values, origin_expected = guarded_origins_inputs()
    for copied in (False, True):
        run_integer_case(build, guarded_origins_fixture(copied), origin_values, origin_expected,
                         'bounded pointer origins' + (' with copies' if copied else ''), output_words=65)
    for name, source in guarded_origins_negatives():
        expect_compile_failure(build, source, 'integer_probe', 'pointer memory proof')
        print('NEGATIVE_PASS', name)
    conditional = fixture('''
 add.u64 %rd6,%rd4,128;
 st.local.u64 [%rd4+128],0;
 st.local.u64 [%rd4+136],0;
 mov.u64 %rd7,1;
CONDITIONAL_LOOP:
 xor.b64 %rd10,%rd3,%rd7;
 st.local.u64 [%rd6],%rd10;
 shl.b64 %rd11,%rd7,3;
 add.u64 %rd6,%rd4,%rd11;
 add.u64 %rd6,%rd6,128;
 setp.lt.u64 %p3,%rd7,2;
 selp.u64 %rd12,1,0,%p3;
 add.u64 %rd7,%rd7,%rd12;
 setp.ne.u64 %p4,%rd3,0;
 and.pred %p5,%p4,%p3;
 @%p5 bra CONDITIONAL_LOOP;
''').replace(' add.u64 %rd1,%rd1,%rd2;',
              ' mul.wide.u32 %rd17,%r4,24;\n add.u64 %rd1,%rd1,%rd17;').replace(
                  ' st.global.u64 [%rd1],%rd9;',
                  ' st.global.u64 [%rd1],%rd9;\n ld.local.u64 %rd10,[%rd4+128];\n'
                  ' ld.local.u64 %rd11,[%rd4+136];\n st.global.u64 [%rd1+8],%rd10;\n'
                  ' st.global.u64 [%rd1+16],%rd11;')
    conditional_expected = [word for value in values for word in
                            (value, value ^ 1, (value ^ 2) if value else 0)]
    for inverted in (False, True):
        source = conditional
        if inverted:
            source = source.replace('setp.lt.u64 %p3', 'setp.ge.u64 %p3').replace(
                'selp.u64 %rd12,1,0,%p3;', 'selp.u64 %rd12,0,1,%p3;').replace(
                'and.pred %p5,%p4,%p3;', 'not.pred %p2,%p3;\n and.pred %p5,%p4,%p2;')
        run_integer_case(build, source, values, conditional_expected,
                         'conditional induction increment' + (' inverted' if inverted else ''), output_words=3)
    overlap = conditional.replace('%rd4,128;', '%rd4,0;').replace(
        '%rd6,%rd6,128;', '%rd6,%rd6,0;').replace('%rd7,2;', '%rd7,16;')
    expect_compile_failure(build, overlap, 'integer_probe', 'pointer memory proof')
    print('NEGATIVE_PASS conditional induction overlaps pointer cell')
    shifted = fixture('''
 setp.ge.u64 %p1,%rd3,16;
 @%p1 bra LOAD;
 shl.b64 %rd10,%rd3,3;
 add.u64 %rd11,%rd4,128;
 add.u64 %rd6,%rd11,%rd10;
 st.local.u64 [%rd6],47;
''').replace('scratch[192]', 'scratch[320]')
    run_integer_case(build, shifted, values, values, 'bounded shifted element offset', output_words=1)
    for name, source in [
        ('shifted offset unbounded', shifted.replace(' @%p1 bra LOAD;\n', '')),
        ('shifted offset overlaps cell', shifted.replace('%rd4,128;', '%rd4,0;')),
        ('shifted offset may overflow', shifted.replace('%rd3,3;', '%rd3,63;')),
    ]:
        expect_compile_failure(build, source, 'integer_probe', 'pointer memory proof')
        print('NEGATIVE_PASS', name)
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
    for name, source in endpoint_negative_fixtures():
        expect_compile_failure(build, source, 'integer_probe', 'pointer memory proof')
        print('NEGATIVE_PASS', name)
    for name, source in affine_endpoint_negative_fixtures():
        expect_compile_failure(build, source, 'integer_probe', 'pointer memory proof')
        print('NEGATIVE_PASS', name)
    for name, source in [
        ('iterator nonzero end sentinel', sentinel_iterator_fixture().replace(
            'selp.b64 %rd22,0,%rd11,%p4;', 'selp.b64 %rd22,1,%rd11,%p4;')),
        ('iterator sentinel overwritten', sentinel_iterator_fixture().replace(
            ' setp.ne.u64 %p5,%rd22,0;', ' mov.u64 %rd22,1;\n setp.ne.u64 %p5,%rd22,0;')),
        ('iterator sentinel bypassed backedge guard', sentinel_iterator_fixture().replace(
            ' @%p5 bra POINTER_LOOP;',
            ' setp.eq.u64 %p8,%rd3,3;\n @%p8 bra POINTER_LOOP;\n @%p5 bra POINTER_LOOP;')),
        ('iterator sentinel skips end', sentinel_iterator_fixture().replace(
            'add.u64 %rd13,%rd11,1;', 'add.u64 %rd13,%rd11,2;')),
        ('iterator missing bound', pointer_iterator_fixture().replace(' @%p2 bra LOAD;\n', '')),
        ('iterator zero-length bypass', pointer_iterator_fixture().replace(' @%p3 bra LOAD;\n', '')),
        ('iterator skips end', pointer_iterator_fixture().replace('add.u64 %rd13,%rd11,1;',
                                                                 'add.u64 %rd13,%rd11,2;')),
        ('scalar dynamic start missing bound', scalar_dynamic_start_fixture().replace(' @%p2 bra LOAD;\n', '')),
    ]:
        expect_compile_failure(build, source, 'integer_probe', 'pointer memory proof')
        print('NEGATIVE_PASS', name)

if __name__ == '__main__':
    build = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2 and sys.argv[2] == '--endpoint-worker':
        endpoint_worker(build, sys.argv[3] == 'copied', Path(sys.argv[4]),
                        sys.argv[3] in ('affine', 'compound'), sys.argv[3] == 'compound')
    else:
        main(build)
