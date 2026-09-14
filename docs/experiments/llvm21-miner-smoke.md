# LLVM 21 miner: first CuMetal checks

The unchanged full PTX downloaded as `vanity-miner-aarch64-llvm21/output-llvm21.ptx`
contains all 123 entries. Its SHA-256 is
`2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.
Producer: CUDA 13.3.33, NVVM 23.0.0, PTX 9.3, `sm_100`; 25,161,784 bytes.

Four entries were checked on CuMetal `798dc62`, with workload specializations
disabled. [Logs and binary hashes](llvm21-miner-smoke.json).

| Entry suffix (`kernel_self_test_`) | Result |
| --- | --- |
| `stub` | Launch/copy probe passes on M5; 117 other slots and 16 guards intact |
| `primitive_sha512` | Slot 1 is 1; 117 other slots and 16 guards intact |
| `primitive_ed25519` | Import fails: `%rd22022` undefined on incoming edge to `$L__BB95_1` |
| `primitive_secp256k1_compressed` | Import fails: `%rd34990` undefined on incoming edge to `$L__BB127_1` |

SHA-512 is heavily constant-folded (272-byte PTX entry); this pass is not evidence
of general runtime-input hashing correctness. Neither elliptic-curve check reached
Metal compilation or GPU execution. The other 115 numerical self-tests and all
four mining kernels were not attempted in this batch. Historical LLVM 7/19 totals
are unchanged.

The probe emits about 7 KiB of Metal source, substantially smaller than the previous
full-module MSL output. However, importing the full PTX still takes substantial CPU
time: the three timed imports took approximately 69–85 seconds. GPU execution of
the passing entries took approximately 1.5–1.7 microseconds (trace duration only).

## Next blocker

Ed25519 contains this loop sequence:

```ptx
$L__BB95_1:
    mov.b64 %rd3, %rd22021;
    setp.gt.u64 %p1, %rd3, 63;
    min.u64 %rd117, %rd3, 63;
    add.s64 %rd22021, %rd117, 1;
    @%p1 bra $L__BB95_3;
    and.b64 %rd118, %rd3, 1;
    setp.ne.b64 %p2, %rd118, 0;
    selp.b64 %rd22022, %rd3, %rd22022, %p2;
    not.pred %p3, %p2;
    @%p3 bra $L__BB95_1;
```

The register is subsequently consumed in `$L__BB95_13`. Investigate whether all
paths to those consumers establish the selected value, including the loop's
upper-bound exit. The current self-select normalization does not accept this
shape. Do not initialize the register arbitrarily or relax general definedness
checks without a control-flow proof and negative regression cases.

## Reproduce

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$OUT.metal"
python3 demos/rust-ptx/run_self_test.py "$OUT.metal" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Use `.metal` for source loading: the runtime treats `.msl` as a binary library.
The source and ABI sidecar are generated automatically. Neither PTX nor generated
Metal source was edited for these checks. GPU runs need access to the Metal device;
sandbox-only `cuInit` failures were retried with device access and are not counted
as compiler or numerical failures. Local outputs: `/tmp/miner-llvm21-cumetal`.
