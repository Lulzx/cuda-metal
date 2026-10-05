"""Negative coverage for the benchmark-output checker."""
from pathlib import Path
import tempfile
import unittest

from validate_reference import read_reference


LOG = """LAMMPS (30 Sep 2026)
KOKKOS mode with Kokkos version 5.2.1 is enabled
using single precision
Created 32000 atoms
attributes: full, newton off, kokkos_device
Step Temp E_pair E_mol TotEng Press
0 1.44 -6.773 0 -4.613 -5.019
100 0.757 -5.758 0 -4.622 0.207
Loop time of 1 on 1 procs for 100 steps with 32000 atoms
"""


class ReferenceOutputTest(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "full.log"
            path.write_text(text)
            return read_reference(path, "full")

    def test_complete(self):
        self.assertEqual(set(self.read(LOG)), {0, 100})

    def test_reject_truncated(self):
        with self.assertRaises(ValueError):
            self.read(LOG.split("100 0.757")[0])

    def test_reject_nonfinite(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read(LOG.replace("0.757", value))

    def test_reject_duplicate_step(self):
        with self.assertRaises(ValueError):
            self.read(LOG.replace("100 0.757", "0 0.757"))

    def test_reject_wrong_mode(self):
        with self.assertRaises(ValueError):
            self.read(LOG.replace("attributes: full", "attributes: half"))

    def test_reject_wrong_atom_count(self):
        with self.assertRaises(ValueError):
            self.read(LOG.replace("32000", "0"))


if __name__ == "__main__":
    unittest.main()
