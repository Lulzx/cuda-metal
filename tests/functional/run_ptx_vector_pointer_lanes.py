#!/usr/bin/env python3
"""Fixed vector lanes retain independent pointer spaces and scalar payloads.

Numerical fixtures use 65 lanes, generic dereferences, immutable CPU references,
ABI checks and guarded buffers. Hazardous memory-proof controls compile only.
Dynamic scalar loads from heterogeneous tables remain a separate refusal suite.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


MASK = (1 << 64) - 1
PRIVATE_XOR = 0x0123456789ABCDEF
ALTERNATE_XOR = 0xFEDCBA9876543210
SCALAR_XOR = 0xA55A96695AA56996
CONSTANTS = (0, 0x123456789ABCDEF0, 0x8000000000000001, MASK)
ENTRY = 'vector_pointer_lanes'
ABI = ['CUMETAL_ABI_V2', 'kernel ' + ENTRY, 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
CASES = tuple(
    dict(name=f'v2-{kind}-lane{lane}', kinds=(kind, 'scalar') if lane == 0 else ('scalar', kind),
         vector_store=lane == 1, layout='straight')
    for kind in ('private', 'device', 'constant') for lane in (0, 1)
) + tuple(
    dict(name=f'v4-rotation{rotation}', kinds=(('private', 'scalar', 'device', 'constant') * 2)[rotation:rotation + 4],
         vector_store=rotation % 2 == 0, layout='straight')
    for rotation in range(4)
) + tuple(
    dict(name='v4-' + layout, kinds=('private', 'scalar', 'device', 'constant'),
         vector_store=False, layout=layout)
    for layout in ('joined', 'reordered')
) + (
    dict(name='v4-offset32', kinds=('private', 'scalar', 'device', 'constant'),
         vector_store=True, layout='offset32'),
    dict(name='v2-private-cursor', kinds=('private', 'scalar'),
         vector_store=False, layout='cursor'),
)
CASE_NAMES = tuple(case['name'] for case in CASES)
NEGATIVE_KINDS = ('missing-store', 'predicated-store', 'overlap-byte',
                  'scalar-source', 'conflicting-space', 'helper-write')


def inputs():
    boundaries = (0, 1, 255, 256, (1 << 32) - 1, 1 << 32,
                  (1 << 63) - 1, 1 << 63, MASK - 1, MASK)
    values = []
    for lane in range(65):
        a = boundaries[lane] if lane < len(boundaries) else (lane * 0x9E3779B97F4A7C15) & MASK
        b = (0x6B8B4567327B23C6 ^ (lane + 17) * 0x85EBCA6B27D4EB2F) & MASK
        length = boundaries[lane % len(boundaries)] if lane < 30 else (lane * 0x0102030405060709) & MASK
        values.extend((a, b, length))
    return values


def expected(case, values):
    result = []
    for lane in range(65):
        a, b, length = values[3 * lane:3 * lane + 3]
        alternate = case['layout'] == 'joined' and lane % 2 == 1
        observed = {
            'private': b ^ ALTERNATE_XOR if alternate else a ^ PRIVATE_XOR,
            'device': b if alternate else a,
            'constant': CONSTANTS[(length & 3) ^ int(alternate)],
            'scalar': length ^ SCALAR_XOR if alternate else length,
        }
        if case['layout'] == 'cursor':
            observed['private'] = ((a ^ PRIVATE_XOR) + (b ^ ALTERNATE_XOR)) & MASK
        result.extend(observed[kind] for kind in case['kinds'])
        # Independent runtime controls are deliberately not recomputed from
        # loaded vector values: corrupting a scalar lane cannot mask itself.
        result.extend((length ^ SCALAR_XOR, a))
    return result


def fixture(case, negative=None, bad_lane=None):
    width = len(case['kinds'])
    if negative is not None:
        if negative not in NEGATIVE_KINDS or bad_lane not in range(width) or case['kinds'][bad_lane] == 'scalar':
            raise ValueError('negative must select a pointer lane')
    helper = '''.func overwrite_record(.param .b64 address, .param .b32 value) {
 .reg .b64 %address;
 .reg .b32 %value;
 ld.param.b64 %address, [address];
 ld.param.b32 %value, [value];
 st.u8 [%address], %value;
 ret;
}
''' if negative == 'helper-write' else ''
    constant_bytes = ','.join(str(byte) for value in CONSTANTS for byte in value.to_bytes(8, 'little'))
    lines = ['.version 7.1', '.target sm_80', '.address_size 64',
             f'.const .align 8 .b8 table[32] = {{{constant_bytes}}};', helper,
             f'.visible .entry {ENTRY}(.param .u64 .ptr .global input,',
             ' .param .u64 .ptr .global output, .param .u32 count) {',
             ' .local .align 32 .b8 depot[128];',
             ' .reg .b64 %input, %output, %offset, %a, %b, %length, %scalar_alt;',
             ' .reg .b64 %base, %cell, %cell_copy, %private, %private_alt, %device_alt;',
             ' .reg .b64 %constant, %constant_alt, %table, %selection, %selection_alt;',
             ' .reg .b64 %private_value, %private_alt_value, %control, %call_cell;',
             ' .reg .b64 %loaded<4>, %copy<4>, %answer<4>;',
             ' .reg .b64 %cursor, %cursor_sum, %cursor_value;',
             ' .reg .b32 %cursor_step;', ' .reg .pred %cursor_more;',
             ' .reg .b32 %lane, %block, %threads, %count, %parity, %byte;',
             ' .reg .pred %done, %choice;',
             ' .param .b64 call_record;', ' .param .b32 call_byte;',
             ' ld.param.u64 %input, [input];', ' ld.param.u64 %output, [output];',
             ' ld.param.u32 %count, [count];', ' mov.u32 %lane, %tid.x;',
             ' mov.u32 %block, %ctaid.x;', ' mov.u32 %threads, %ntid.x;',
             ' mad.lo.u32 %lane, %block, %threads, %lane;',
             ' setp.ge.u32 %done, %lane, %count;', ' @%done bra DONE;',
             ' mul.wide.u32 %offset, %lane, 24;', ' add.u64 %input, %input, %offset;',
             f' mul.wide.u32 %offset, %lane, {8 * (width + 2)};', ' add.u64 %output, %output, %offset;',
             ' ld.global.u64 %a, [%input];', ' ld.global.u64 %b, [%input+8];',
             ' ld.global.u64 %length, [%input+16];', ' cvt.u32.u64 %byte, %a;',
             ' and.b32 %parity, %lane, 1;', ' setp.eq.u32 %choice, %parity, 0;',
             ' mov.u64 %base, depot;',
             (' add.u64 %cell, %base, 32;' if case['layout'] == 'offset32' else ' mov.b64 %cell, %base;'),
             ' mov.u64 %cell_copy, %cell;',
             ' add.u64 %private, %base, 64;', ' add.u64 %private_alt, %base, 72;',
             f' xor.b64 %private_value, %a, {PRIVATE_XOR};',
             f' xor.b64 %private_alt_value, %b, {ALTERNATE_XOR};',
             ' st.local.u64 [%private], %private_value;', ' st.local.u64 [%private_alt], %private_alt_value;',
             ' add.u64 %device_alt, %input, 8;', ' mov.u64 %table, table;',
             ' and.b64 %selection, %length, 3;', ' shl.b64 %selection, %selection, 3;',
             ' xor.b64 %selection_alt, %selection, 8;',
             ' add.u64 %constant, %table, %selection;', ' add.u64 %constant_alt, %table, %selection_alt;',
             f' xor.b64 %scalar_alt, %length, {SCALAR_XOR};']

    def stores(alternate=False):
        names = {'private': '%private_alt' if alternate else '%private',
                 'device': '%device_alt' if alternate else '%input',
                 'constant': '%constant_alt' if alternate else '%constant',
                 'scalar': '%scalar_alt' if alternate else '%length'}
        operands = [names[kind] for kind in case['kinds']]
        if case['vector_store'] and negative is None:
            return [f' st.local.v{width}.b64 [%cell_copy], {{{", ".join(operands)}}};']
        result = []
        for lane, value in enumerate(operands):
            if lane == bad_lane and negative == 'missing-store':
                continue
            if lane == bad_lane and negative == 'scalar-source':
                value = '%length'
            predicate = '@%choice ' if lane == bad_lane and negative == 'predicated-store' else ''
            result.append(f' {predicate}st.local.b64 [%cell_copy+{lane * 8}], {value};')
        return result

    def load_and_consume():
        result = [f' ld.local.v{width}.b64 {{{", ".join(f"%loaded{i}" for i in range(width))}}}, [%cell_copy];']
        if case['layout'] == 'joined':
            result.append(' @%choice bra COPY_EVEN;')
            result.extend(f' mov.b64 %copy{i}, %loaded{i};' for i in range(width))
            result.extend((' bra CONSUME;', 'COPY_EVEN:'))
            result.extend(f' mov.u64 %copy{i}, %loaded{i};' for i in range(width))
            result.append('CONSUME:')
        else:
            result.extend(f' mov.b64 %copy{i}, %loaded{i};' for i in range(width))
        for lane, kind in enumerate(case['kinds']):
            if case['layout'] == 'cursor' and kind == 'private':
                result.extend((f' mov.b64 %cursor, %copy{lane};',
                               ' mov.u64 %cursor_sum, 0;', ' mov.u32 %cursor_step, 0;',
                               'CURSOR_LOOP:', ' ld.u64 %cursor_value, [%cursor];',
                               ' add.u64 %cursor_sum, %cursor_sum, %cursor_value;',
                               ' add.u64 %cursor, %cursor, 8;',
                               ' add.u32 %cursor_step, %cursor_step, 1;',
                               ' setp.lt.u32 %cursor_more, %cursor_step, 2;',
                               ' @%cursor_more bra CURSOR_LOOP;',
                               f' mov.b64 %answer{lane}, %cursor_sum;'))
            else:
                result.append(f' mov.b64 %answer{lane}, %copy{lane};' if kind == 'scalar' else
                              f' ld.u64 %answer{lane}, [%copy{lane}];')
            result.append(f' st.global.u64 [%output+{lane * 8}], %answer{lane};')
        result.extend((f' xor.b64 %control, %length, {SCALAR_XOR};',
                       f' st.global.u64 [%output+{width * 8}], %control;',
                       f' st.global.u64 [%output+{(width + 1) * 8}], %a;', ' ret;'))
        return result

    if case['layout'] == 'reordered':
        lines.extend((' bra STORES;', 'LOAD_RECORD:'))
        lines.extend(load_and_consume())
        lines.append('STORES:')
        lines.extend(stores())
        lines.append(' bra LOAD_RECORD;')
    else:
        if case['layout'] == 'joined':
            lines.append(' @%choice bra STORE_EVEN;')
            lines.extend(stores(True))
            lines.extend((' bra LOAD_RECORD;', 'STORE_EVEN:'))
        lines.extend(stores())
        if negative == 'overlap-byte':
            lines.append(f' st.local.u8 [%cell_copy+{bad_lane * 8 + 7}], %byte;')
        elif negative == 'conflicting-space':
            other = '%input' if case['kinds'][bad_lane] != 'device' else '%private'
            lines.extend((' @%choice bra LOAD_RECORD;',
                          f' st.local.b64 [%cell_copy+{bad_lane * 8}], {other};'))
        elif negative == 'helper-write':
            lines.extend((f' add.u64 %call_cell, %cell_copy, {bad_lane * 8};',
                          ' st.param.b64 [call_record], %call_cell;',
                          ' st.param.b32 [call_byte], %byte;',
                          ' call.uni overwrite_record, (call_record, call_byte);'))
        lines.append('LOAD_RECORD:')
        lines.extend(load_and_consume())
    lines.extend(('DONE:', ' ret;', '}'))
    return '\n'.join(lines) + '\n'


def rejection_cases():
    # Cover every vector lane position, including fields after a scalar lane.
    bases = ((CASES[6], 0), (CASES[7], 1), (CASES[6], 2), (CASES[6], 3))
    return tuple((f'lane{lane}-{kind}', fixture(case, kind, lane))
                 for case, lane in bases for kind in NEGATIVE_KINDS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(command, timeout):
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True,
                             env=dict(os.environ, CUMETAL_TRACE_GPU='1', CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0'))
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate(timeout=5)
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return dict(command=command, seconds=time.monotonic() - started, exit_code=child.returncode,
                timed_out=timed_out, stdout=stdout, stderr=stderr)


def worker(build, case, artifacts):
    from ptx_test_support import run_integer_case
    values = inputs()
    run_integer_case(build, fixture(case), values, expected(case, values), case['name'],
                     entry=ENTRY, input_words=3, output_words=len(case['kinds']) + 2,
                     abi_lines=ABI, artifacts_dir=artifacts)


def suite(args, evidence):
    compiler = args.build / 'cumetalc'
    selected = [case for case in CASES if args.case is None or case['name'] == args.case]
    records = []
    for case in selected:
        folder = evidence / case['name']
        folder.mkdir()
        ptx, msl = folder / 'test.ptx', folder / 'test.metal'
        source = fixture(case)
        ptx.write_text(source)
        command = ([sys.executable, str(Path(__file__).resolve()), str(args.build), '--case', case['name'],
                    '--gpu-child', '--artifacts', str(folder)] if args.stage == 'numerical' else
                   [str(compiler), str(ptx), '--backend=cumetal-ir', '--ptx-strict', '--entry', ENTRY,
                    '--emit=msl', '-o', str(msl)])
        measured = process(command, 90)
        launches = [line for line in measured['stderr'].splitlines() if 'CUMETAL_PROVENANCE event=kernel_launch' in line]
        good = not measured['timed_out'] and measured['exit_code'] == 0 and ptx.read_text() == source
        abi_path = Path(str(msl) + '.cumetal-abi')
        good = good and msl.is_file() and abi_path.is_file() and abi_path.read_text().splitlines() == ABI
        if args.stage == 'numerical':
            good = good and f"NUMERICAL_PASS {case['name']}: 65 inputs, output values, guards" in measured['stdout'].splitlines()
            good = good and len(launches) == 1 and all(token in launches[0] for token in (
                'kernel="' + ENTRY + '"', 'device=apple_gpu ', 'launch_success=true ',
                'source=generic_ptx ', 'provenance=generic_ptx_lowering ', 'semantic_quality=exact '))
        else:
            good = good and not launches
        record = dict(case=case['name'], passed=good, stage=args.stage, ptx_sha256=digest(ptx),
                      msl_sha256=digest(msl) if msl.is_file() else None, process=measured)
        records.append(record)
        (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print('PASS' if good else 'FAIL', args.stage, case['name'], flush=True)
        if not good:
            print(measured['stderr'], end='', file=sys.stderr)

    for name, source in rejection_cases():
        folder = evidence / ('reject-' + name)
        folder.mkdir()
        ptx, msl = folder / 'test.ptx', folder / 'test.metal'
        ptx.write_text(source)
        measured = process([str(compiler), str(ptx), '--backend=cumetal-ir', '--ptx-strict',
                            '--entry', ENTRY, '--emit=msl', '-o', str(msl)], 45)
        # Accept only pointer/proof diagnostics, not an unrelated syntax error
        # or unsupported instruction. Retain full text to audit the actual stage.
        diagnostic = any(text in measured['stderr'] for text in (
            'pointer memory proof', 'operand type', 'incompatible pointer', 'conflicting pointer'))
        good = (not measured['timed_out'] and measured['exit_code'] not in (0, None) and diagnostic and
                'CUMETAL_PROVENANCE event=kernel_launch' not in measured['stderr'] and
                not msl.exists() and not Path(str(msl) + '.cumetal-abi').exists())
        record = dict(case=name, passed=good, stage='rejection', ptx_sha256=digest(ptx), process=measured)
        records.append(record)
        (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print('PASS' if good else 'FAIL', 'rejection', name, flush=True)
        if not good:
            print(measured['stderr'], end='', file=sys.stderr)
    report = dict(schema=1, stage=args.stage, compiler_sha256=digest(compiler),
                  runtime_sha256=digest(args.build / 'libcumetal.dylib') if args.stage == 'numerical' else None,
                  script_sha256=digest(Path(__file__).resolve()),
                  helper_sha256=digest(Path(__file__).with_name('ptx_test_support.py')),
                  required_positive_cases=len(selected), full_positive_denominator=len(CASES),
                  required_rejections=len(rejection_cases()),
                  counts=dict(Counter(('pass' if row['passed'] else 'fail') + '-' + row['stage'] for row in records)),
                  passed=all(row['passed'] for row in records), cases=records)
    (evidence / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


def main():
    if not __debug__:
        raise RuntimeError('This test requires Python assertions for ABI and guard checks')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--stage', choices=('numerical', 'translate'), default='numerical')
    parser.add_argument('--case', choices=CASE_NAMES)
    parser.add_argument('--output', type=Path, help='new or empty evidence directory')
    parser.add_argument('--gpu-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--artifacts', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.build = args.build.resolve()
    if args.gpu_child:
        if args.case is None:
            parser.error('--gpu-child requires --case')
        worker(args.build, next(case for case in CASES if case['name'] == args.case), args.artifacts)
        return 0
    if not (args.build / 'cumetalc').is_file():
        parser.error('build directory must contain cumetalc')
    if args.stage == 'numerical' and not (args.build / 'libcumetal.dylib').is_file():
        parser.error('numerical stage requires libcumetal.dylib')
    if args.output:
        evidence = args.output.resolve()
        evidence.mkdir(parents=True, exist_ok=True)
        if any(evidence.iterdir()):
            parser.error('--output must be a new or empty directory')
        return suite(args, evidence)
    with tempfile.TemporaryDirectory(prefix='cumetal-vector-pointer-lanes-') as work:
        return suite(args, Path(work))


if __name__ == '__main__':
    raise SystemExit(main())
