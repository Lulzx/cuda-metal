#!/usr/bin/env python3
"""Independent initial FCC lattice LJ sum for stock bench/in.lj."""
import itertools
import json
import math
from pathlib import Path
import sys


def initial_fcc_pair_energy(density=0.8442, cutoff=2.5):
    spacing = (4 / density) ** (1 / 3)
    radius = math.ceil(cutoff / spacing) + 1
    basis = ((0,0,0), (0,.5,.5), (.5,0,.5), (.5,.5,0))
    terms = []
    for cell in itertools.product(range(-radius, radius+1), repeat=3):
        for offset in basis:
            r2 = math.fsum(((cell[i]+offset[i])*spacing)**2 for i in range(3))
            if 0 < r2 < cutoff*cutoff:
                inv6 = r2**-3
                terms.append(4*(inv6*inv6-inv6))
    return math.fsum(terms)/2, len(terms)


if __name__ == '__main__':
    energy, neighbours = initial_fcc_pair_energy()
    result = {'density':.8442, 'cutoff':2.5, 'neighbours':neighbours,
              'analytic_initial_pair_energy':energy,
              'scope':'initial infinite FCC lattice only; not a trajectory validation'}
    if len(sys.argv) > 1:
        reference = json.loads(Path(sys.argv[1]).read_text())
        result['reference'] = reference
    print(json.dumps(result, indent=2))
