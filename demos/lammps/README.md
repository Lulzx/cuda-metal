# LAMMPS / Kokkos GPU demo

**Status: numerical GPU gate passed on 2026-10-04.** The unmodified CUDA
application compiles, links, and completes the stock 32,000-atom, 100-step
Lennard-Jones benchmark on Apple M4 Pro in both neighbour modes. The fixed
`5e-4` CPU-double, cross-mode, and analytic initial-energy checks pass.
Performance comparisons remain open.

The current target is unmodified LAMMPS `stable_30Sep2026`, commit
`8de817dd79bfe4525d5d39246a212d833e6dee07`, with its bundled Kokkos 5.2.1.
Nothing from LAMMPS or NVIDIA's toolkit is vendored. The build uses CuMetal's
clean-room toolkit, one process, C++20, `KOKKOS_PREC=single`, CUDA architecture
`sm_80`, and no relocatable device code. Source registration is enabled;
the `libcuda.dylib` binary alias is not required.

The [recorded result](results/2026-10-04-m4-pro.json) reports maximum scaled
CPU-double error `5.91e-6`, cross-mode difference `1.27e-6`, and analytic initial
pair-energy error `1.26e-7`. Full/Newton-off records 826 successful Apple-GPU
launches; half/Newton-on records 1,534. Workload specializations are disabled.
FP64 uses reduced-precision software emulation, identified in the trace.
Raw logs are retained locally in `out/gpu.CHtxEp`, with CPU-double reference
`out/reference.aKdfsP` and earlier failed attempts in separate directories.

The measured simulation loops took 4.67 seconds (full) and 6.77 seconds (half),
but regression runs were concurrent, so these are completion receipts rather
than performance comparisons. Cold half-neighbour pipeline compilation took
minutes; total wall time was 9 minutes 14 seconds versus 18 seconds for full.
All 396 CuMetal CTest gates passed with zero skips across the final partitioned
runs, including the Clang 21/22/23 matrix, 83 enrolled CUDA samples, and typed
PTX/native-AOT numerical corpora. The demo validator's 11 tests also pass.

## Reproduce

Build CuMetal first, then run:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build build -j6
mkdir -p demos/lammps/out

# Isolates header diagnostics without a full build.
bash scripts/build_lammps_cumetal.sh --probe > demos/lammps/out/probe.log 2>&1

# Full application build.
bash scripts/build_lammps_cumetal.sh --gpu > demos/lammps/out/gpu-build.log 2>&1

# Builds the Serial CPU version, runs stock bench/in.lj in both neighbour modes,
# and records a numerical comparison. The recorded agreement check fails.
bash demos/lammps/run_reference.sh

# Characterized double-precision CPU reference for the numerical GPU gate.
bash demos/lammps/run_reference.sh double

# Runs both GPU neighbour modes with independent fresh caches.
bash demos/lammps/run_gpu.sh
# Use the output directories printed by the preceding commands.
python3 demos/lammps/validate_gpu.py <gpu-output-dir> <double-reference-output-dir>
```

Sources default to `/tmp/cumetal-lammps-30Sep2026`. Set
`CUMETAL_LAMMPS_DIR`, `CUMETAL_BUILD_DIR`, `CUMETAL_CLANG`, or `CUMETAL_JOBS`
to override local paths/toolchain/parallelism; checkout and build paths must
be absolute. The script checks the exact commit and refuses tracked source
changes. `--cpu` builds only Serial; `--compare` builds Serial then CUDA.

Image/compressed-output support and Kokkos dynamic profiling-library loading
are disabled in both configurations. Their CMake dependencies otherwise put
the macOS SDK's C headers ahead of Homebrew libc++, breaking compilation.
`nvcc_wrapper` also needs separate `-Xlinker -rpath -Xlinker <path>` arguments
to avoid `nvcc_wrapper` comma quoting. The shim also splits comma-separated
`-Xlinker` arguments before passing them to Clang.

## Original experiment result: 2026-10-04

Apple M4 Pro, Homebrew Clang 23.1.0, CMake 4.4.3, CuMetal Release with source
registration enabled and binary shim disabled:

| Check | Result |
| --- | --- |
| CPU LAMMPS/Kokkos Serial build | Compiles and links |
| Stock CPU `bench/in.lj`, full/Newton-off neighbours | Completes 32,000 atoms, 100 steps |
| Stock CPU `bench/in.lj`, half/Newton-on neighbours | Completes 32,000 atoms, 100 steps |
| CPU mode agreement at `5e-4` scaled tolerance | **Fails**: maximum difference `0.000578169` |
| CUDA configuration | Succeeds |
| CUDA header probe | **Fails**: 70 errors, 55 distinct diagnostics |
| CUDA LAMMPS build | **Fails** in Kokkos core |
| Metallib, numerical Apple-GPU gate, performance | Not reached |
| CuMetal baseline `ctest --test-dir build -LE slow` | 376/376 pass |

The CPU disagreement is largest in initial total energy: full neighbours report
`-4.6104156`, half neighbours `-4.6130812`. FP32 reduction order is a possible
explanation, not a confirmed root cause. The tolerance was not widened after
the failure. Reference logs and `reference.json` preserve the failed result;
it must be resolved or characterized before selecting GPU acceptance limits.

### Original CUDA blockers

The compile-only `kokkos_header_probe.cpp` includes stock `Kokkos_Core.hpp`.
`--probe` takes the real Kokkos core compiler flags from CMake's compilation
database and reports every diagnostic with `-ferror-limit=0`.

- Half/bfloat16 conversion helpers: double/integer to half/bfloat, and
  half/bfloat to integer with round-toward-zero semantics.
- Half/bfloat math helpers, including `hexp`, `hlog`, `hsqrt`, `hceil`,
  `hfloor`, `htrunc`, `hrint`, `__hisnan`, `hrsqrt`, and `hrcp`. Some bfloat
  calls resolve to half overloads and then produce ambiguous conversions.
- CUDA fatal-error enum names: hardware stack, illegal instruction, misaligned
  address, invalid address space, and invalid PC.
- `cudaDeviceProp.reservedSharedMemPerBlock`.
- `cudaGraphAddDependencies`, `cudaGraphAddEmptyNode`, and
  `cudaGraphAddChildGraphNode` declarations/behavior.
- Three-argument `cudaMemcpyToSymbol` calls on pointer-valued device symbols
  in Desul's atomic-lock arrays. Simply passing the pointer value to the C
  API would not preserve the symbol's identity.

These were confirmed header blockers, not an exhaustive list of defects.
Current fixes preserve symbol identity and transactional graph dependencies;
child graphs still return `cudaErrorNotSupported`. CUDA-source Thrust policy
sorting uses GPU merge kernels; CUDA-policy sort-by-key still refuses.
Typed address-space
predicates, half/bfloat helpers, CUDA version macros, and integral math promotion
have focused tests. The compatibility GPU regression and seven related tests
pass; these do not establish full LAMMPS numerical correctness. The typed
backend also preserves private/shared allocation alignment, PTX byte offsets,
and clamped shifts. PTX warp barriers retain SIMD-group scope; the pinned
Kokkos prefix scan passes exact integer checks at 2,048 and 32,000 elements.
The build increases Clang's inlining threshold to keep
Kokkos capture records in their kernel rather than crossing an opaque private
helper boundary. Proven terminal power-of-two block-dimension assertions
become checked launch constraints, with an entry guard before any barrier;
invalid shapes report `cudaErrorLaunchFailure`. Five-argument device assertions
without barriers, collectives, or printf
report a per-launch failure and cancel spinning peers. General assertions
in synchronized kernels, assertion text, and exact `cudaErrorAssert`
reporting remain unsupported.
The runtime supplies a documented virtual register cost
for Kokkos launch-size calculations.

The typed backend preserves scalar zero values across integer/float bit-container
joins, widens signed and unsigned parameter loads correctly, and recovers
captured device-pointer fields through private spills. Indirect vector parameter
loads remain subject to the existing provenance and bounds checks. Full-CTA
`bar.red.popc` uses a barrier-protected atomic count; named and partial-CTA
variants remain unsupported. Ordered and unordered FP64 comparisons have an
exhaustive 49-pair, 14-predicate GPU regression. Natural loops with multiple
exits save their exit arguments and emit each continuation once, avoiding
duplicated neighbour-kernel tails. Nested-loop and collective GPU tests cover
these changes independently of the stock simulation.
Destination-free PTX `red` instructions reuse the typed atomic operations and
discard the old value. Float/integer addition has a contended GPU regression;
compare-and-swap reductions and unsupported memory ordering are refused.

### First attempted release

The original `stable_22Jul2025_update2` attempt is retained in local `out/`
logs. Configure initially failed on the rpath spelling, then succeeded.
Kokkos 4.6.2 still failed on missing CUDA declarations. That release predates
`KOKKOS_PREC`; forcing `LMP_PRECISION=1` also made the CPU build fail because
`PairZBLKokkos::init_one` returns `float` while its base method returns `double`.
The newer pin avoids that upstream precision-build defect; it does not close
CuMetal's CUDA-header gaps.

## Numerical GPU acceptance gate

The numerical gate requires the unmodified stock `bench/in.lj` workload to
complete 32,000 atoms and 100 steps in both full/Newton-off and half/Newton-on
modes. It checks temperature, pair and molecular energy, total energy, and
pressure at steps 0 and 100 against the double-precision Serial reference,
plus cross-mode agreement and initial pair energy against the independent
FCC lattice sum. The fixed scaled tolerance is `5e-4`. The validator requires
successful Apple-GPU launch provenance and at least one cold compile per mode.
`run_gpu.sh` selects `CUMETAL_PTX_BACKEND=cumetal-ir`, disables workload
specializations, and enables Metal device addresses. FP64 operations use
CuMetal's reduced-precision emulation and are identified as such in the trace.

Successful host compilation alone would still be weaker than production
metallib compilation and numerical GPU execution.
