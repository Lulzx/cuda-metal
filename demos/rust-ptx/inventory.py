#!/usr/bin/env python3
"""Inventory unchanged PTX entry compilation; does not load or execute GPU code."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ptx', type=Path)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout <= 0:
        parser.error('jobs and timeout must be positive')
    ptx, compiler, out = args.ptx.resolve(strict=True), args.compiler.resolve(strict=True), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = ptx.read_bytes()
    entries = re.findall(r'^\s*(?:\.visible\s+)?\.entry\s+(\w+)\s*\(', data.decode(), re.M)
    if not entries or len(entries) != len(set(entries)):
        parser.error('PTX must contain a nonempty, unique entry inventory')
    report = {
        'evidence': 'strict typed PTX-to-MSL compilation only; no Metal compilation or GPU execution',
        'ptx_sha256': hashlib.sha256(data).hexdigest(),
        'compiler_sha256': hashlib.sha256(compiler.read_bytes()).hexdigest(),
        'entry_count': len(entries), 'timeout_seconds': args.timeout,
        'results': [],
    }
    env = dict(os.environ, CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0')

    def compile_entry(entry):
        output = out / f'{entry}.metal'
        sidecar = Path(str(output) + '.cumetal-abi')
        # A previous run's output cannot turn a failed attempt into a pass.
        output.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        command = [str(compiler), str(ptx), '--backend=cumetal-ir', '--ptx-strict',
                   '--overwrite', '--entry', entry, '--emit=msl', '-o', str(output)]
        started = time.monotonic()
        returncode = None
        try:
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
            log, returncode = result.stdout, result.returncode
            status = 'msl_pass' if returncode == 0 and output.is_file() and sidecar.is_file() else 'compile_failure'
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or b''
            log = captured.decode(errors='replace') if isinstance(captured, bytes) else captured
            log += f'\nTIMEOUT after {args.timeout} seconds\n'
            status = 'timeout'
        (out / f'{entry}.log').write_text(log)
        diagnostic = ''
        if status != 'msl_pass':
            diagnostic = next((line for line in log.splitlines() if 'failed:' in line),
                              next((line for line in log.splitlines() if line.strip()), 'no diagnostic'))
            diagnostic = re.sub(r'^cumetalc failed:\s*', '', diagnostic)
            diagnostic = re.sub(r'\bline \d+:\s*', '', diagnostic)
        return {'entry': entry, 'status': status, 'returncode': returncode,
                'seconds': round(time.monotonic() - started, 3), 'diagnostic': diagnostic}

    def save():
        report['results'].sort(key=lambda item: item['entry'])
        report['completed'] = len(report['results'])
        report['counts'] = dict(Counter(item['status'] for item in report['results']))
        report['failure_groups'] = dict(Counter(item['diagnostic'] for item in report['results'] if item['status'] != 'msl_pass'))
        temporary = out / 'inventory.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'inventory.json')

    save()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(compile_entry, entry) for entry in entries]
        for future in as_completed(futures):
            item = future.result()
            report['results'].append(item)
            save()
            print(f"{report['completed']}/{len(entries)} {item['status']}: {item['entry']} {item['diagnostic']}", flush=True)
    print(json.dumps(report['counts'], sort_keys=True))


if __name__ == '__main__':
    main()
