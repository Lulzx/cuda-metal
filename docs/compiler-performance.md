# Compiler timing and optimization

## Compilation stage tracing

Set `CUMETAL_TRACE_COMPILE=1` to write wall-clock spans to stderr for typed
PTX import, Metal legalization, MSL generation, and the runtime calls that
create a Metal source library, function, and compute pipeline:

```sh
CUMETAL_TRACE_COMPILE=1 ./build/cumetalc input.ptx --backend=cumetal-ir \
  --emit=msl -o output.metal 2>compile.log
```

For runtime compilation, set the same variable on the application loading
`libcumetal`. Only the exact value `1` enables tracing. Tracing is otherwise
silent and does not scan source text for diagnostic byte counts.

Each span flushes a begin record before entering the stage, then emits elapsed
milliseconds when its scope exits. An interrupted process can leave an open
span, identifying the call it entered. Match records by process ID and span ID,
including when threads interleave the same stage. Span IDs do not associate
nested stages with a particular concurrent top-level compilation. Input sizes
describe PTX or UTF-8 MSL bytes; zero means the stage has no recorded input size.
Source text, arguments, filenames, and kernel names are not logged.

```text
CUMETAL_COMPILE event=begin pid=123 span=1 stage=ptx_import input_bytes=456
CUMETAL_COMPILE event=end pid=123 span=1 stage=ptx_import elapsed_ms=1.234
```

An end record means the call returned or its scope unwound, **not that it
succeeded**. Inspect the operation's error/exit status separately. Preparation
does not prove a kernel was dispatched or that its result is correct. Set
`CUMETAL_TRACE_GPU=1` for the separate execution/provenance records and verify
numerical results. Compilation spans use CPU wall time, not GPU duration.

Tracing covers the typed PTX-to-MSL and NVVM-to-MSL paths (`ptx_import`,
`nvvm_import`, `metal_legalization`, `msl_generation`), Apple's offline compiler
steps (`air_opt`, `metal_compile`, `air_link`, `metallib_package`), `cumetalc`'s
external steps, and runtime source/pipeline creation. Legacy PTX lowering,
precompiled-metallib loads, cache hits, source reads, and lock waiting have no
spans. Runtime cache hits
therefore produce no additional creation spans. The total PTX span includes its
nested stages and intervening work; do not add it to its children. Apple's own
compiler caches can affect timings even when a creation call is recorded.
Tracing measures these stages; it does not fix slow compilation or compiler
allocation failures.

Focused checks are `unit_compile_trace` (host-only pairing, opt-in, interruption,
and successful/failed PTX compilation) and `functional_compile_trace` (real Metal
preparation, cache reuse, errors, GPU values, and guards).

## Per-phase timing for people

`CUMETAL_TRACE_COMPILE` is for tools. For a readable report of where a slow
compile spends its time, pass `--compile-timing` to `cumetalc` or set
`CUMETAL_COMPILE_TIMING=1` for `cumetalc` or for an application that JIT-compiles
kernels through `libcumetal`:

```text
$ cumetalc saxpy.cu -o saxpy --compile-timing
cumetal timing: start cuda_frontend
cumetal timing: done  cuda_frontend 124.5 ms
cumetal timing: start nvvm_import (2.8 KiB input)
...
cumetal timing: Metal source for saxpy.cu: 3.2 KiB; largest functions: _Z5saxpyfPfS_ (585 B)
cumetal timing: start metal_compile [saxpy.cu] (3.2 KiB input)
cumetal timing: done  metal_compile [saxpy.cu] 39.7 ms
...
cumetal timing: summary (wall clock; phases can nest or overlap, do not add)
cumetal timing:   cuda_frontend                           1x        124.5 ms
...
```

Each phase prints a `start` line before it runs, so a compile that stalls for
minutes shows which phase it is in. Before Apple's compiler sees generated
Metal source, the report gives the source size and its five largest functions,
which is usually enough to find the kernel that blew up. The process prints a
per-phase total at exit; phases nest (a `cumetalc` executable build runs the
device compile as a child, whose lines are passed through), so totals do not
add up to the wall time. Only the exact value `1` enables timing; `-v` keeps
its nvcc meaning. Unlike the machine-readable trace, timing lines name the
source file or kernel.

Phases: `cuda_frontend`, `llvm_opt`, `nvvm_import`, `ptx_import`,
`metal_legalization`, `msl_generation`, `air_opt`, `metal_compile`,
`metal_compile_support`, `air_link`, `metallib_package`, and for executables
`device_compile`, `host_frontend`, `host_lower`, `metallib_embed`,
`registration_compile`, `object_merge`, `host_link`. At runtime:
`metal_new_library_with_source`, `metal_new_function`,
`metal_new_compute_pipeline`.

## Failing fast on very large kernels

Apple's compile time grows with source size: a 24 MB generated MSL source took
300–600 s to become a library (#115, #133). Set `CUMETAL_MSL_SIZE_LIMIT` to a
byte count with an optional `K`, `M` or `G` suffix (binary units) to refuse
generated Metal source above that size before Apple's compiler runs:

```text
$ CUMETAL_MSL_SIZE_LIMIT=8M cumetalc huge.cu -o huge
cumetalc failed: generated Metal source for huge.cu is 24.1 MiB (25270519 bytes),
over CUMETAL_MSL_SIZE_LIMIT=8M (8388608 bytes); largest functions: ...
```

The limit applies to `cumetalc` and to runtime compilation (registration JIT,
driver-loaded modules, `cuModuleLoad` of `.metal` source); at runtime the
message goes to stderr and the kernel fails to load with the usual CUDA error.
A malformed value is an error rather than being ignored. The limit is unset by
default. Without it, CuMetal still warns once per process when a generated
source reaches 16 MiB. The limit covers generated MSL only: the legacy
PTX-to-LLVM path hands LLVM IR to `xcrun metal` and is not size-checked.

`functional_compile_timing` and `unit_compile_timing` cover both features.

## Bounded private-array optimization

CuMetal can improve generated GPU code without changing a CUDA kernel or
substituting MPS. The NVVM importer now selectively unrolls tiny loops over
private arrays and runs LLVM scalar replacement, exposing individual values
that Metal can keep in registers.

This addresses a specific weakness in the conservative CUDA-to-LLVM pipeline.
It does not enable Clang's general `-O2` pipeline, change FP32 precision, retile
matmul automatically, or select a CPU/library fallback. PTX import is unchanged.

## Selection policy

The pass is deliberately bounded:

- A function must contain a static private array of at least **256 bytes**.
  Smaller arrays remain delegated to Apple's optimizer.
- A loop must explicitly request `unroll.enable` or `unroll.full`, have private
  memory accesses, and have a provable constant trip count of **2–9 header
  executions**. A pre-tested eight-iteration loop visits its header nine times.
- Only innermost loops are selected, with estimated expanded instruction count
  at most **1,024**. At most three rounds expose nested array loops.
- Calls (including warp operations and barriers), atomics, fences, and volatile
  memory accesses exclude a loop. Dynamic bounds and explicit `unroll.disable`
  or `unroll.count` hints are not selected by this pass.
- Unrelated hints are suppressed only during the scoped LLVM pass and restored
  on surviving loops. Partial/runtime unrolling and peeling are disabled.
- LLVM module verification runs after normalization, before typed import.

The optimization runs in the NVVM importer, so it applies to accepted LLVM/NVVM
inputs as well as CUDA source compiled through that path. It is separate from
the pre-existing small aggregate-shuffle correctness normalization.

## Address handling prerequisite

LLVM unrolling can fold shared-array addresses into constant GEP expressions.
The importer previously allowed some of those expressions to reach generated
MSL as LLVM text, causing a Metal compile error. It now lowers the shared symbol
plus byte displacement as a typed threadgroup pointer, preserving its address
space across a generic pointer cast. Embedded read-only constants retain their
constant-space handling.

## Performance scope

The experiment uses the **unchanged** `block64_8x8` kernel in
[the matmul demo](../demos/matmul/README.md): a 64x64 output tile with 64 FP32
accumulators per thread. It is a deliberately difficult alternative to the
already faster `block64_32` configuration. Improving this kernel is not an
improvement to the previous demo's best throughput, nor evidence of MPS parity.

The broad experiment that unrolled every hint was rejected: it expanded the
LLVM listing from roughly 5,400 to 61,000 lines and regressed multiple variants.
The final pass targets large private arrays instead of all hinted loops.

Reproduce an interleaved comparison using two artifacts compiled from the same
CUDA source, before and after the compiler change:

```bash
python3 tools/matmul_bench/compare_compilers.py \
  --runner build-release-noshim/matmul-bench/matmul_bench \
  --before /path/to/before.metallib --after /path/to/after.metallib \
  --output /tmp/cumetal-compiler-comparison
```

The script runs three rounds of ten timed iterations after two warmups, rotates
kernel order, alternates before/after order, checks every output, and saves
artifact hashes, logs, results, and a median-of-round-means summary with each
mode's per-round minimum. Timings include synchronized runtime launch/wait
overhead. The before artifact must be saved before rebuilding the compiler.

The script fails when any variant/mode round mean drifts more than `--max-drift`
(default 10%) from its own other rounds. Sustained load throttles a laptop GPU
within a couple of minutes; an unguarded first attempt here reported a
spurious 0.68x for `registers` because two of three rounds ran at roughly half
clock in both modes. Keep other GPU workloads idle and let the machine cool
before rerunning after a drift failure.

## Tests

`unit_nvvm_private_array` checks array scalarization, explicit opt-out, small
arrays, dynamic/large loops, volatile memory, barriers, and folded shared-array
addresses. The matmul demo checks six numerical shapes including tails. The
existing NVVM compiler suite and production compiler/runtime corpora provide
broader regression coverage; measured results and the completed gates are
recorded below rather than inferred from test registration.

## Measured result and completed validation

On **Apple M4 Pro (16 GPU cores)**, macOS 26.7, Xcode 26.6, LLVM 23.1.0,
Release/shim-off, the final confirmation used the same CUDA source, inputs,
host runner, runtime, and FP32 arithmetic for both artifacts. These are local
worktree results based on `9c0a004`, including pre-existing uncommitted work;
they are not measurements of a published release.
[Configuration and source hashes](evidence/compiler-private-array-2026-09-06/metadata.json).

| Unchanged CUDA kernel | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| Large 8x8 accumulator tile (`block64_8x8`) | 584.675 | 96.320 | **6.07x** |
| Already-fast control (`block64_32`) | 67.078 | 66.330 | 1.01x |

The large-array kernel improves from **235.1 to 1,426.9 GFLOP/s**. This is a
compiler improvement to the previously slow kernel; it still trails the best
source-tuned kernel and MPS. Each number is the median of three round means,
with ten timed iterations and two warmups per round at 4096 cubed.
[All confirmation rounds](evidence/compiler-private-array-2026-09-06/confirmation-results.json)
and [logs](evidence/compiler-private-array-2026-09-06/confirmation-timings.log).

One earlier final-artifact run had large timing swings during desktop use,
including the unchanged control, so absolute timing from that run was not used
as the headline. Its entire
[results](evidence/compiler-private-array-2026-09-06/variable-desktop-results.json)
and [logs](evidence/compiler-private-array-2026-09-06/variable-desktop-timings.log)
are retained. The subsequent confirmation's large-array before means were
586.61, 584.44, and 584.67 ms; after means were 96.32, 96.60, and 95.08 ms.
No competing CuMetal tests or builds ran during either final-artifact timing
run; ordinary desktop activity was not disabled.

A further rerun with the drift guard enabled (`compare_compilers.py --max-drift`
default, three variants) passed with at most 4.9% round-to-round drift:
`block64_8x8` 584.003 -> 95.009 ms (6.15x), `block64_32` 66.163 -> 66.068 ms
(1.00x), and the article's register-tiled kernel 91.277 -> 90.215 ms (1.01x),
confirming the pass leaves kernels without a qualifying private array
unchanged within noise.
[Guarded summary](evidence/compiler-private-array-2026-09-06/guarded-summary.md),
[rows](evidence/compiler-private-array-2026-09-06/guarded-results.json), and
[artifact hashes](evidence/compiler-private-array-2026-09-06/guarded-metadata.json).

Generated MSL for the article's four kernels, the coalesced control, `block64`,
and the already-fast `block64_32` is text-identical to the baseline. The large
array kernel changes from 615 to 1,392 MSL lines as small loops expand. That is
code-generation evidence, not a measurement of physical register allocation
or spilling; no hardware-counter claim is made.
[Per-kernel comparison and hashes](evidence/compiler-private-array-2026-09-06/generated-code.json).

Completed gates:

- **285/285 Release CTests passed**, including the upstream CUDA samples,
  compiler backend matrices with Clang 21/22/23, and typed-PTX/native-AOT
  numerical corpora. [Full log](evidence/compiler-private-array-2026-09-06/ctest-release.log).
- Focused NVVM tests passed in **Debug and Release**; the new unit test also
  covers instruction-growth limits, explicit unroll counts, and preservation
  of an unrelated disabled loop.
  [Debug log](evidence/compiler-private-array-2026-09-06/ctest-debug-focused.log).
- **104 full-output comparisons across six matmul shapes** passed with the final
  compiler. All timed results also passed full-output comparisons with zero
  observed error. [Shape checks](evidence/compiler-private-array-2026-09-06/final-correctness.log).
- A separate trace confirms successful Apple-GPU launches of `block64_8x8`.
  [Provenance](evidence/compiler-private-array-2026-09-06/final-provenance.log).

PTX-import behavior, general optimization-level handling, automatic tiling,
SIMD-group matrix instruction selection, and universal speedups remain outside
this change. LLVM 23 libraries were used to build CuMetal; frontend checks
with CUDA Clang 21/22/23 are not LLVM-library-version build coverage.
