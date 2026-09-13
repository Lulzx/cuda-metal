#!/usr/bin/env python3
"""Exercise inventory accounting without requiring the Metal SDK or a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / 'demos/rust-ptx/inventory.py'


class InventoryTest(unittest.TestCase):
    def test_results_and_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / 'compiler'
            compiler.write_text(f'#!{sys.executable}\n' + '''
import pathlib, sys, time
name = sys.argv[sys.argv.index('--entry') + 1]
output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])
if name == 'slow':
    time.sleep(10)
if name == 'bad':
    print("cumetalc failed: line 7: unsupported opcode 'fake'")
    sys.exit(1)
output.write_text('generated MSL')
if name != 'missing_abi':
    pathlib.Path(str(output) + '.cumetal-abi').write_text('ABI')
''')
            compiler.chmod(0o755)
            ptx = root / 'input.ptx'
            ptx.write_text('\n'.join(f'.visible .entry {name}() {{ ret; }}'
                                     for name in ['good', 'bad', 'missing_abi', 'slow']))
            out = root / 'out'
            out.mkdir()
            (out / 'missing_abi.metal.cumetal-abi').write_text('stale ABI')
            subprocess.run([sys.executable, str(INVENTORY), str(ptx), '--compiler', str(compiler),
                            '--out', str(out), '--jobs', '1', '--timeout', '2'],
                           check=True, capture_output=True, text=True)
            report = json.loads((out / 'inventory.json').read_text())
            self.assertEqual(report['completed'], 4)
            self.assertEqual(report['counts'], {'msl_pass': 1, 'compile_failure': 2, 'timeout': 1})
            results = {row['entry']: row for row in report['results']}
            self.assertEqual(results['bad']['diagnostic'], "unsupported opcode 'fake'")
            self.assertEqual(results['slow']['returncode'], None)
            self.assertFalse((out / 'missing_abi.metal.cumetal-abi').exists())

    def test_duplicate_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ptx = root / 'input.ptx'
            ptx.write_text('.entry duplicate() {}\n.entry duplicate() {}\n')
            result = subprocess.run([sys.executable, str(INVENTORY), str(ptx),
                                     '--compiler', sys.executable, '--out', str(root / 'out')],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unique entry inventory', result.stderr)


if __name__ == '__main__':
    unittest.main()
