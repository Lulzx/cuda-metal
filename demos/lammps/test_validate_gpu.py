"""Completion, numerical and GPU provenance gates must fail independently."""
import json
from pathlib import Path
import tempfile
import unittest
from validate_gpu import validate
from characterize_reference import initial_fcc_pair_energy

class GpuValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.gpu, self.cpu = Path(self.tmp.name)/'gpu', Path(self.tmp.name)/'cpu'
        self.gpu.mkdir(); self.cpu.mkdir()
        pair, _ = initial_fcc_pair_energy()
        for mode in ('full','half'):
            log = f'''LAMMPS (30 Sep 2026)
Kokkos version 5.2.1
using single precision
Created 32000 atoms
attributes: {mode}, newton {'off' if mode == 'full' else 'on'}, kokkos_device
Step Temp E_pair E_mol TotEng Press
0 1.44 {pair} 0 -4.6134356 -5.019
100 .757 -5.758 0 -4.622 .207
Loop time of 1 on 1 procs for 100 steps with 32000 atoms
'''
            (self.gpu/f'{mode}.lammps.log').write_text(log)
            (self.cpu/f'{mode}.log').write_text(log.replace('single precision','double precision'))
            (self.gpu/f'{mode}.log').write_text('CUMETAL_PROVENANCE event=kernel_launch source=generic_ptx device=apple_gpu launch_success=true compile_cache_hit=false\n')
    def test_complete(self):
        self.assertEqual(validate(self.gpu,self.cpu)['status'],'pass')
    def test_provenance(self):
        path=self.gpu/'full.log'; original=path.read_text()
        for invalid in ('', original.replace('apple_gpu','cpu'), original.replace('true','false'),
                        original.replace('generic_ptx','approximate_stub'),
                        original.replace('compile_cache_hit=false','compile_cache_hit=true')):
            with self.subTest(log=invalid), self.assertRaises(ValueError):
                path.write_text(invalid); validate(self.gpu,self.cpu)
    def test_completion(self):
        path=self.gpu/'half.lammps.log';path.write_text(path.read_text().split('100 .757')[0])
        with self.assertRaises(ValueError):validate(self.gpu,self.cpu)
    def test_cpu_difference(self):
        path=self.gpu/'full.lammps.log';path.write_text(path.read_text().replace('100 .757','100 .9'))
        with self.assertRaises(ValueError):validate(self.gpu,self.cpu)
        self.assertEqual(json.loads((self.gpu/'validation.json').read_text())['status'],'fail')
    def test_analytic_difference(self):
        for mode in ('full','half'):
            for path in (self.gpu/f'{mode}.lammps.log',self.cpu/f'{mode}.log'):
                text=path.read_text(); pair,_=initial_fcc_pair_energy()
                path.write_text(text.replace(str(pair),'-6.7'))
        with self.assertRaises(ValueError):validate(self.gpu,self.cpu)

if __name__=='__main__':unittest.main()
