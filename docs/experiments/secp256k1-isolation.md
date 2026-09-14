# LLVM 19 secp256k1 numerical isolation

The initial isolation below records the failure at `0b3081e`. The follow-up
found and fixed incorrect source-width handling in `cvt.s16.s8`. The original
LLVM 19 compressed primitive now passes on M5, with guards intact. See the investigation
at the end of this page and [fix 12](miner-fix-backlog.md#fix-12-cvt-interprets-the-instructions-source-width).

Tested on Apple M5 at compiler commit `0b3081e`, using the unchanged LLVM 19
PTX from run `34778991430`. No manual MSL changes or workload specializations.
[Machine-readable results and hashes](secp256k1-isolation-0b3081e.json).

| Slot | Self-test | Result |
| --- | --- | --- |
| 93 | Affine generator encoding | Pass |
| 78 | Projective generator to affine and encoding | Pass |
| 79 | Double generator, convert to affine, encode | Pass |
| 80 | Scalar one serialization round-trip | Pass |
| 74 | Derive public key for scalar 1 | Pass |
| 75 | Derive public key for scalar 2 | Pass |
| 4 | Original compressed public-key fixture | **Fail: 0 instead of 1** |

Every launch preserves the other 117 result slots and all 16 guard words.
The two successful derivations take about 18 ms on the GPU; the failing original
also takes about 18 ms. Host-side Metal compilation takes substantially longer.
The original failure was reproduced after the six smaller checks passed.

An independent Python integer-arithmetic double-and-add calculation, using
secp256k1's field modulus and generator, agrees with the original expected key:
`039163ab449d4b90de13ce60b504bfc27a4aed378c1f8338686156b91445637c8d`.
This checks the fixture, not CuMetal's arithmetic.

## What this narrows down

The simple encoding, doubling and scalar-serialization fixtures work, and full
derivation works for scalars 1 and 2. The failure is therefore scalar-dependent.
These results do **not** prove field arithmetic, point addition or conversion
correct for all inputs. Compiler constant folding can also limit fixed-fixture
coverage. No faulty instruction has been identified yet.

The local dependency sources clarify the path: `elliptic-curve` 0.13.8
`PublicKey::from_secret_scalar` computes `generator() * scalar`, which reaches
k256 0.13.4's generic multiplication/`lincomb` implementation. It does not call
the separate `MulByGenerator` precomputed-generator-table method. Generic
multiplication splits the scalar into two components, builds local point tables,
forms signed radix-16 digits, then selects and accumulates points.

Next diagnostic boundaries, in order:

1. Compare the original scalar's two decomposed components and signs with a CPU
   reference. Include modular reduction and component negation.
2. Compare signed radix-16 digits and selected points, especially negative digits.
3. Locate the first accumulator difference across point additions/doublings.
4. Reduce the first divergence to a runtime-input regression, fix the lowering,
   and rerun the original unchanged PTX fixture before claiming a fix.

Any diagnostic instrumentation must be explicitly separate from the unchanged
PTX integration run. LLVM 7's timeout remains a separate unresolved observation.
This is a targeted batch, not a new full-suite compatibility count.

## Reproduce

For each entry in the JSON ledger, compile from the full original module:

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$MSL"
python3 demos/rust-ptx/run_self_test.py "$MSL" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Local generated modules and compile/GPU logs are under `/tmp/secp-isolate`.
The scalar round-trip result was captured directly in the tool transcript;
the ledger records its result and GPU duration. The original generated module
is `/tmp/param-trunc-secp-llvm19.metal`.

## Follow-up: correct decomposition and tables, incorrect signed conversion

Separate diagnostic copies of the generated MSL export intermediate values and
return at the selected boundary. These are diagnostic observations, not numerical
passes of the original self-test. The PTX and original MSL are unchanged.
[Captured values, table limbs and snapshot locations](secp256k1-intermediates-0b3081e.json).

The GPU agrees with independent CPU integer arithmetic at these boundaries:

- Both full 256-bit scalar components after decomposition.
- Both sign bits (1) and the two positive 128-bit magnitudes after negation.
- All 66 signed radix-16 digits, including negative digits and the final carries.
- All 16 table points: multiples 1 through 8 of the signed generator and its
  signed endomorphism. The CPU decodes the GPU's five-limb projective coordinates
  and compares their affine values with independent curve addition.

The failure is downstream of those boundaries. At original PTX lines 639901–639903,
the selector loads an unsigned byte into a 16-bit register, sign-extends its low
byte with `cvt.s16.s8`, then takes `abs.s16`. The old MSL instead copies the
zero-extended 16-bit value. A digit of -7 thus remains 249 rather than becoming
-7, and its absolute value cannot match any table index 1 through 8. The second
selector has the same form at lines 649520–649521.

A standalone GPU regression reproduced the loss of sign: input byte `0x81`
returned absolute value 129 instead of 127. The importer now truncates a wider
integer register to the source width declared by `cvt` before applying signed
or unsigned conversion. Generated Metal includes `ushort(char(byte))` rather
than the old 16-bit identity conversion. This fixes instruction semantics without
changing cryptographic code, tables, scalar values or PTX artifacts.

Recheck the saved snapshots with the CPU reference (no GPU required):

```sh
python3 demos/rust-ptx/check_secp256k1_intermediates.py \
  docs/experiments/secp256k1-intermediates-0b3081e.json
```

After the fix, the unchanged original PTX was recompiled to
`/tmp/secp-cvt-fixed.metal` and passed the normal `run_self_test.py` check:
slot 4 = 1, other 117 slots and 16 guards intact. This full-kernel result is
separate from the instrumented diagnostic snapshots above. The compiler/GPU
regression suite passes all 33 focused tests.
