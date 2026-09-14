#!/usr/bin/env python3
"""Verify saved diagnostic snapshots with CPU big integers; does not run a GPU."""
import json
import sys
from pathlib import Path

P = 2**256 - 2**32 - 977
N = int('fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141', 16)
G = (int('79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798', 16),
     int('483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8', 16))
BETA = int('7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee', 16)
LAMBDA = int('5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72', 16)
G1 = int('3086d221a7d46bcde86c90e49284eb153daa8a1471e8ca7fe893209a45dbb031', 16)
G2 = int('e4437ed6010e88286f547fa90abfe4c4221208ac9df506c61571b4ae8ac47f71', 16)
MINUS_B1 = int('e4437ed6010e88286f547fa90abfe4c3', 16)
B2 = int('3086d221a7d46bcde86c90e49284eb15', 16)


def add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    x, y = a
    u, v = b
    if x == u and (y + v) % P == 0:
        return None
    slope = ((3*x*x) * pow(2*y, -1, P) if a == b
             else (v-y) * pow(u-x, -1, P)) % P
    z = (slope*slope-x-u) % P
    return z, (slope*(x-z)-y) % P


def affine(limbs):
    x, y, z = [sum(limbs[j+i] << (52*i) for i in range(5)) % P
               for j in (0, 5, 10)]
    return None if z == 0 else (x*pow(z, -1, P) % P, y*pow(z, -1, P) % P)


def main(path):
    data = json.loads(Path(path).read_text())
    k = int(data['scalar'], 16)
    c1 = (k*G1 + (1 << 383)) >> 384
    c2 = (k*G2 + (1 << 383)) >> 384
    r2 = (c1*MINUS_B1-c2*B2) % N
    r1 = (k-r2*LAMBDA) % N
    assert [int(data[name], 16) for name in ('gpu_r1', 'gpu_r2')] == [r1, r2]
    signs = [int(r > N//2) for r in (r1, r2)]
    magnitudes = [min(r, N-r) for r in (r1, r2)]
    assert data['gpu_signs'] == signs
    assert [int(v, 16) for v in data['gpu_magnitudes']] == magnitudes
    for i, magnitude in enumerate(magnitudes):
        digits = [(magnitude >> (4*j)) & 15 for j in range(33)]
        for j in range(32):
            carry = (digits[j]+8) >> 4
            digits[j] -= carry << 4
            digits[j+1] += carry
        assert data['gpu_digits'][i] == digits, f'digits {i}'
    limbs = data['gpu_table_limbs']
    assert len(limbs) == 240
    for t, x in enumerate((G[0], G[0]*BETA % P)):
        base = (x, (-G[1] if signs[t] else G[1]) % P)
        expected = None
        for j in range(8):
            expected = add(expected, base)
            offset = (t*8+j)*15
            assert affine(limbs[offset:offset+15]) == expected, f'table {t}, point {j+1}'
    print('CPU_REFERENCE_PASS: decomposition, signs, magnitudes, 66 digits, 16 table points')


if __name__ == '__main__':
    main(sys.argv[1])
