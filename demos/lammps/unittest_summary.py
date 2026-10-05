"""Summarize LAMMPS force-style unit tests from `ctest -V` logs.

ctest reports a yaml test as Passed even when every gtest case inside it was
skipped, so this reads the per-case gtest lines instead. Each line in a -V log
is prefixed with its ctest number; "Start N: Name" maps numbers to names.

usage: unittest_summary.py <ctest -V log> [<second log> ...] [--json out.json]
Each log contributes one column, labelled by its parent directory name.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

START = re.compile(r'^\s*Start\s+(\d+):\s+(\S+)')
CASE = re.compile(r'^(\d+): \[\s+(OK|FAILED|SKIPPED)\s+\]\s+(\w+)\.(\w+)(?: \(|$)')
TIMEOUT = re.compile(r'^\s*\d+/\d+ Test\s+#(\d+):\s+(\S+)\s+\.+\*\*\*(Timeout|Failed|Exception)')


def parse(path):
    names, cases, crashed = {}, {}, {}
    for line in Path(path).read_text(errors='replace').splitlines():
        if m := START.match(line):
            names[m[1]] = m[2]
        elif m := CASE.match(line):
            cases[(m[1], m[4])] = m[2]
        elif m := TIMEOUT.match(line):
            crashed[m[1]] = m[3]
    out = {}
    for (num, case), status in cases.items():
        out.setdefault(names.get(num, num), {})[case] = status
    # A yaml test that died before printing a case result still counts.
    for num, kind in crashed.items():
        entry = out.setdefault(names.get(num, num), {})
        if not any(s == 'FAILED' for s in entry.values()):
            entry['<process>'] = kind.upper()
    return out


def main():
    args = sys.argv[1:]
    json_out = None
    if '--json' in args:
        i = args.index('--json')
        json_out = args[i + 1]
        del args[i:i + 2]
    columns = {Path(p).parent.name: parse(p) for p in args}
    summary = {}
    for label, results in columns.items():
        tally = Counter(status for cases in results.values() for status in cases.values())
        failed = sorted(f'{t}.{c}' for t, cases in results.items() for c, s in cases.items()
                        if s not in ('OK', 'SKIPPED'))
        ran = sorted(t for t, cases in results.items() if any(s != 'SKIPPED' for s in cases.values()))
        summary[label] = dict(cases=dict(tally), yaml_tests_with_a_run_case=len(ran), failures=failed)
        print(f'{label}: {dict(tally)}; {len(ran)} yaml tests ran at least one case')
        for f in failed:
            print(f'  {f}')
    if json_out:
        Path(json_out).write_text(json.dumps(dict(summary=summary, results=columns), indent=2) + '\n')


if __name__ == '__main__':
    main()
