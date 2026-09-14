#!/usr/bin/env python3
"""cvt interprets the instruction's source width, including wider registers."""
from pathlib import Path
import random
import sys
from run_ptx_scalar_tail_call import run_case
from run_ptx_widening import TEMPLATE


def main():
    build = Path(sys.argv[1]).resolve()
    # Match k256: ld.b8 zero-extends into a .b16 register, then cvt.s16.s8
    # must sign-extend the low byte before abs/table selection.
    body = '''
ld.global.b8 %rs1, [%rd5];
cvt.s16.s8 %rs2, %rs1;
abs.s16 %rs3, %rs2;
cvt.u64.u16 %rd7, %rs3;
cvt.u16.u8 %rs4, %rs1;
cvt.u64.u16 %rd8, %rs4;
'''
    template = TEMPLATE.replace('.reg .b16 %rs1;', '.reg .b16 %rs<5>;')
    values = list(range(256))
    run_case(build, template.replace('BODY', body), values,
             [x for v in values for x in (abs(v if v < 128 else v-256), v)],
             'all byte encodings through signed conversion and abs')

    # Upper register bits must not affect the instruction's source value.
    rng = random.Random(8163264)
    mask64 = (1 << 64)-1
    for width, container, register in [(8, 16, '%rs1'), (8, 32, '%r5'),
                                       (16, 32, '%r5'), (32, 64, '%rd9')]:
        mask = (1 << width)-1
        values = [0, 1, mask >> 1, (mask >> 1)+1, mask,
                  1 << width, (1 << container)-1]
        values += [rng.getrandbits(container) for _ in range(256)]
        body = f'''ld.global.b{container} {register}, [%rd5];
cvt.s64.s{width} %rd7, {register};
cvt.u64.u{width} %rd8, {register};'''
        expected = []
        for value in values:
            low = value & mask
            signed = low if low < 1 << (width-1) else low-(1 << width)
            expected += [signed & mask64, low]
        run_case(build, template.replace('BODY', body), values, expected,
                 f'signed/unsigned {width}-bit cvt in {container}-bit register')


if __name__ == '__main__':
    main()
