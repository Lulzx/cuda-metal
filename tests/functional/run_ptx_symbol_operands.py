#!/usr/bin/env python3
"""Rust-CUDA module-global shapes that once produced MSL Metal rejected (#219).

vanity-miner-rs reported, against September typed-backend builds:
  * `use of undeclared identifier 'private$39'`: a writable module private
    global referenced only from device helpers;
  * `as_type<device uchar*>` applied to a `ulong`: an address carried as an
    integer through memory;
  * `ulong([private$em])`: raw PTX symbol expressions left in address operands.

This reduced module has all three in one call graph and runs it on the GPU
with exact references and buffer guards, for a read-only `private$em` (which
becomes a Metal `constant` array) and a written one (a hidden device buffer
threaded through the helpers).
"""
import os
from pathlib import Path
import sys

from ptx_test_support import run_integer_case

EM = list(range(1, 33))  # private$em initializer
WRITTEN = {9, 10}         # bytes the writable variant stores to

TEMPLATE = r'''.version 8.0
.target sm_80
.address_size 64

.global .align 8 .b8 private$39[512];
.global .align 1 .b8 private$em[32] = {@EM@};

// A helper-only reference to a writable private global: each lane writes its
// own word and reads it back.
.func (.param .b32 result) helper_state(.param .b32 lane, .param .b32 value)
{
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<4>;
	ld.param.b32 	%r1, [lane];
	ld.param.b32 	%r2, [value];
	mov.u64 	%rd1, private$39;
	mul.wide.u32 	%rd2, %r1, 4;
	add.s64 	%rd3, %rd1, %rd2;
	st.global.u32 	[%rd3], %r2;
	ld.global.u32 	%r3, [%rd3];
	st.param.b32 	[result], %r3;
	ret;
}

// Symbol expressions in address operands: direct, offset, cvta'd generic and
// computed from the symbol.
.func (.param .b32 result) helper_bytes(.param .b32 index)
{
	.reg .b16 	%rs<6>;
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<6>;
	ld.param.b32 	%r1, [index];
	ld.global.u8 	%rs1, [private$em];
	ld.global.u8 	%rs2, [private$em+7];
@WRITE_DIRECT@
	cvta.global.u64 	%rd1, private$em;
	ld.u8 	%rs3, [%rd1+3];
@WRITE_GENERIC@
	cvt.u64.u32 	%rd2, %r1;
	mov.u64 	%rd3, private$em;
	add.s64 	%rd4, %rd3, %rd2;
	ld.global.u8 	%rs4, [%rd4];
	cvt.u32.u16 	%r2, %rs1;
	cvt.u32.u16 	%r3, %rs2;
	cvt.u32.u16 	%r4, %rs3;
	cvt.u32.u16 	%r5, %rs4;
	add.s32 	%r6, %r2, %r3;
	add.s32 	%r6, %r6, %r4;
	add.s32 	%r7, %r6, %r5;
	st.param.b32 	[result], %r7;
	ret;
}

.visible .entry integer_probe(.param .u64 .ptr .global .align 4 input,
                              .param .u64 .ptr .global .align 4 output, .param .u64 count)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<16>;
	.param .b32 	lane_argument;
	.param .b32 	value_argument;
	.param .b32 	index_argument;
	.param .b32 	state_answer;
	.param .b32 	bytes_answer;
	ld.param.u64 	%rd1, [input];
	ld.param.u64 	%rd2, [output];
	ld.param.u64 	%rd3, [count];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	cvt.u64.u32 	%rd4, %r4;
	setp.ge.u64 	%p1, %rd4, %rd3;
	@%p1 bra 	$L__done;
	cvta.to.global.u64 	%rd5, %rd1;
	cvta.to.global.u64 	%rd6, %rd2;
	shl.b64 	%rd7, %rd4, 2;
	add.s64 	%rd8, %rd5, %rd7;
	ld.global.u32 	%r5, [%rd8];
	st.param.b32 	[index_argument], %r5;
	call.uni 	(bytes_answer), helper_bytes, (index_argument);
	ld.param.b32 	%r6, [bytes_answer];
	st.param.b32 	[lane_argument], %r4;
	st.param.b32 	[value_argument], %r6;
	call.uni 	(state_answer), helper_state, (lane_argument, value_argument);
	ld.param.b32 	%r7, [state_answer];
	// An address carried as an integer through memory: the lane stores its
	// input element's address in its private$39 slot pair and loads it back.
	mov.u64 	%rd9, private$39;
	add.s64 	%rd10, %rd9, 256;
	shl.b64 	%rd11, %rd4, 3;
	add.s64 	%rd12, %rd10, %rd11;
	st.global.u64 	[%rd12], %rd8;
	ld.global.u64 	%rd13, [%rd12];
	ld.global.u32 	%r8, [%rd13];
	shl.b64 	%rd14, %rd4, 3;
	add.s64 	%rd15, %rd6, %rd14;
	st.global.u32 	[%rd15], %r7;
	st.global.u32 	[%rd15+4], %r8;
$L__done:
	ret;
}
'''


def module(writable):
    text = TEMPLATE.replace('@EM@', ', '.join(str(value) for value in EM))
    text = text.replace('@WRITE_DIRECT@', '\tst.global.u8 \t[private$em+9], %rs1;' if writable else '')
    text = text.replace('@WRITE_GENERIC@', '\tst.u8 \t[%rd1+10], %rs2;' if writable else '')
    return text


def main():
    build = Path(sys.argv[1]).resolve()
    if not (build / 'libcumetal.dylib').exists():
        print('SKIP: libcumetal.dylib is not built')
        return 77
    os.environ.setdefault('CUMETAL_PTX_BACKEND', 'cumetal-ir')
    indices = [i for i in range(32) if i not in WRITTEN] * 2  # 60 lanes, below the 64-word slots
    expected = []
    for index in indices:
        expected += [EM[0] + EM[7] + EM[3] + EM[index], index]
    for writable in (False, True):
        run_integer_case(build, module(writable), indices, expected,
                         'writable private$em' if writable else 'read-only private$em',
                         word_bits=32, input_words=1, output_words=2)
    print('PASS: Rust-CUDA private global, symbol-operand and integer-address shapes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
