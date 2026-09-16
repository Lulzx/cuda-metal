#!/usr/bin/env python3
"""Reused scalar/pointer registers retain their types across joins and loops."""
from pathlib import Path
import struct
import sys
from ptx_test_support import run_integer_case

def relu_cases():
    """Pure fixture construction also permits baseline translation-only replay."""
    original = (Path(__file__).parent / 'reference/ptx_register_reuse.ptx').read_text()

    # The original #76 reducer skips each shift for index zero. The skipped
    # value is still the correct byte offset, so both paths must retain i64
    # before the joined register is reused as an address.
    joined = original
    for register, label in (('%rd2', 'INPUT_ADDRESS'), ('%rd3', 'OUTPUT_ADDRESS')):
        shift = f'    shl.b64     {register}, {register}, 2;'
        joined = joined.replace(shift, f'''    setp.eq.u32 %p1, %r4, 0;
    @%p1 bra {label};
{shift}
{label}:''')

    offset = '''    cvt.u64.u32 %rd2, %r4;
    shl.b64     %rd2, %rd2, 2;'''
    output_offset = '''    cvt.u64.u32 %rd3, %r4;
    shl.b64     %rd3, %rd3, 2;
'''
    prefix = '''    and.b32 %r5, %r4, 1;
    setp.eq.u32 %p1, %r5, 0;
    @%p1 bra EVEN_OFFSET;
    bra ODD_OFFSET;
'''
    even = '''EVEN_OFFSET:
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd2, %rd2, 2;
    bra OFFSET_READY;
'''
    odd = '''ODD_OFFSET:
    mul.wide.u32 %rd2, %r4, 4;
    bra OFFSET_READY;
'''
    suffix = '''OFFSET_READY:
    mov.b64 %rd3, %rd2;'''
    copied = original.replace(offset, prefix + even + odd + suffix).replace(output_offset, '')
    reordered = original.replace(offset, prefix + odd + even + suffix).replace(output_offset, '')

    # Each lane performs zero, one, two or three loop iterations. Undo the
    # increments before addressing memory; all accesses remain in the 65-item
    # input even though the loop carries an independently defined i64 value.
    loop = '''    and.b32 %r5, %r4, 3;
    mov.u32 %r6, 0;
    cvt.u64.u32 %rd2, %r4;
OFFSET_LOOP:
    setp.ge.u32 %p1, %r6, %r5;
    @%p1 bra OFFSET_DONE;
    add.u64 %rd2, %rd2, 1;
    add.u32 %r6, %r6, 1;
    bra OFFSET_LOOP;
OFFSET_DONE:
    cvt.u64.u32 %rd3, %r5;
    sub.u64 %rd2, %rd2, %rd3;
    shl.b64 %rd2, %rd2, 2;
    mov.b64 %rd3, %rd2;'''
    looped = original.replace(offset, loop).replace(output_offset, '')

    # Declarations, not conventional %rd/%r spellings, supply storage widths.
    renamed = reordered
    for old, new in (('%rd', '%address'), ('%r', '%index'),
                     ('%f', '%sample'), ('%p', '%guard')):
        renamed = renamed.replace(old, new)
    return [('original', original), ('original joined', joined),
            ('copied diamond', copied), ('reordered diamond', reordered),
            ('renamed diamond', renamed), ('bounded loop', looped)]


def main(build):
    values = [float(i - 32) / 4 for i in range(65)]

    def bits(value):
        return struct.unpack('<I', struct.pack('<f', value))[0]

    abi = ['CUMETAL_ABI_V2', 'kernel clamp_relu', 'shared 0',
           'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
    for label, source in relu_cases():
        # The pinned legacy backend emits MSL for the original straight-line
        # shape, but rejects these CFG forms before Metal compilation. Keep its
        # original numerical regression and report the missing CFG gate without
        # preventing the typed importer regressions from running.
        backends = ('legacy', 'cumetal-ir') if label == 'original' else ('cumetal-ir',)
        if label != 'original':
            print(f'NOT_TESTED legacy {label}: baseline backend cannot emit MSL for '
                  'this CFG fixture; legacy numerical acceptance remains open')
        for backend in backends:
            run_integer_case(build, source, list(map(bits, values)),
                             [bits(max(value, 0.0)) for value in values],
                             f'{backend} {label} scalar/pointer reuse', entry='clamp_relu',
                             word_bits=32, output_words=1, backend=backend, abi_lines=abi)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
