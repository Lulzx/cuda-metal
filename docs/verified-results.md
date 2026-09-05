# Verified results and downstream projects

This document records the performance and application claims summarized in the
README. A successful process exit is insufficient: first-party results require
numerical or domain-specific checks and Apple-GPU provenance.

## Covered kernels

Vector addition, SAXPY, reduction, matrix operations, atomics, shared memory,
warp operations, streams, events, and selected CUDA library calls have numerical
GPU tests. The suite includes negative cases because accepting a program is not
the same as implementing it correctly.

On 2026-08-29, `functional_cumetalc_link_executable` compiled the unmodified
`samples/vectorAdd/vectorAdd.cu` translation unit into a native-AOT executable
and passed all 16,384 numerical outputs on Apple M4 Pro. The gate checks that
the linked binary has no unresolved `__cudaRegister*` symbols, starts from an
empty cache, creates only a native-AOT materialization (not a registration-JIT
entry), and reports `device=apple_gpu` with `launch_success=true`.
`functional_cumetalc_native_aot_multi_kernel` separately launches four kernels
from one embedded native module and checks atomics, thread fences, and static
shared-memory results.
`functional_cumetalc_native_aot_symbols` verifies ABI-v3 constant and writable
device symbols end to end: bounded host-to/from-symbol copies, a 27,904-byte
constant buffer, symbol address/size queries, and persistent GPU updates across
two launches.

On 2026-08-31, the exact manifest-controlled 27-project corpus passed on Apple
M4 Pro with workload specializations disabled: 28/28 through typed PTX and
28/28 through direct native AOT. The device-call probes pass pointer
arguments and offsets through a scalar-returning noinline helper with a loop,
pointer merge, and early exit, and preserve every field through flat and
depth-two nested 12-byte aggregate returns followed by by-value aggregate
arguments. The nested probe also replaces an inner field of a materialized
return before consuming it. The flat probe
numerically consumes a CUDA Clang 21-23 module-private promoted aggregate
literal whose exact initializer and trailing zero bytes are embedded in the
metallib. The barrier-CFG probe specializes an unqualified helper pointer to
shared memory and checks uniform multi-exit barrier paths for all 32 lanes. The
initialized-global probe verifies exact nonzero/negative source bytes and
mutations persisting across two launches for both a visible array and
Clang-scalarized translation-unit-private integers, plus public symbol-copy
visibility. Native AOT performs that initialization before ABI registration
without PTX JIT; PTX registration recovers the bytes from the device image
rather than its zero-filled host shadow.
The GGML probe checks its complete output
operator set; the raytracer performs a 47-value differential preflight and a
full CPU-reference image comparison. The native-AOT device-`printf` gate also
checks all 32 coordinate records, dynamic width, a wide scalar, tracked string
materialization, a bounded unterminated string, a registration-backed writable
module string, and safe rejection of an arbitrary untracked address. Matching
legacy-PTX and typed-PTX gates exercise the same string cases on the Apple GPU.
All three paths also set the FIFO to three words, retain and drain a format-only
record, reject the following two-argument record, and observe device return
values `0` and `2`; a third statically null-format call returns `-1` without adding a ring
record. This separates CUDA return semantics from CuMetal's bounded record
acceptance. `functional_runtime_device_limits` independently
checks byte-to-word rounding, querying, invalid zero rejection, and reset.
These are numerical/runtime results, separate from the 29-file cross-Clang
compile matrix. Embedded read-only module-constant strings remain outside this
proved subset, as does CUDA's circular overwrite policy.

The reviewed Phase 4 manifest ran 185/185 required functional tests on
2026-08-29 with zero failures and zero skips. This includes native-AOT numerical
Apple-GPU execution for by-value aggregates containing device pointers, the
bounded monotonic `clock()` emulation, and a four-threadgroup cooperative grid
barrier. The barrier gate checks device-scope visibility across two barriers;
the clock gate checks monotonic progress rather than claiming GPU-cycle timing.

On 2026-08-30 the full 83-sample NVIDIA gate returned to 83/83 with zero
waivers. The `conjugateGradientMultiBlockCG` entry ran its unmodified
1,048,576-row workload through generic PTX and software `fast48` on Apple GPU;
the gate now requires the independently computed host equation error to be at
most `1e-4` in addition to upstream's device-reduced residual. The observed
printed error was `0.000000` and residual was `1.596402e-6`. This also covers a
warp-tile reduction invoked from only one warp of a larger block and a
guaranteed-progress occupancy-derived cooperative grid.

The 2026-08-31 rerun additionally checked the previously failing
`newdelete` and `UnifiedMemoryPerf` entries from a fresh registration-JIT cache.
`newdelete` reported 3/3, covering heap objects, placement-new in shared memory,
virtual calls, and its 16-byte user datatype. `UnifiedMemoryPerf` ran all eight
allocation modes at every upstream matrix size with `--kernel-iterations=1`;
the reduced iteration count keeps this correctness gate below the Metal GPU
watchdog and is not presented as a performance measurement. Volatile latch
loads, shared-pointer provenance, and the bounded aggregate-call ABI have
focused lowering regressions, and the JIT cache identity was advanced so older
artifacts cannot be reused.

On 2026-08-31, the complete Debug/shim-on non-benchmark inventory passed
278/278 serially, and the complete Debug/shim-off inventory passed 275/275.
Both runs had zero skips and included Phase 4, the 83-sample gate, all six
PhysX comparisons, the 29-file Clang-version/backend matrices, and the 27-project
typed-PTX and native-AOT numerical corpora. These are local Apple M4 Pro results,
not recurring CI evidence.

On 2026-08-29, `functional_typed_{direct,ptx}_{device,system}_atomics` and
`functional_typed_{direct,ptx}_fence` passed on Apple GPU. The device test
checks add/sub/min/max/CAS/inc/dec/and/or/xor across 16,384 contending threads;
the system test checks GPU atomics plus a host atomic on the same managed-memory
word; the fence test checks all payload words and the completion counter. These
tests execute the independently produced direct-NVVM and PTX typed metallibs,
not the legacy backend.

On the same date, `functional_typed_direct_constant_symbol` passed on Apple GPU.
It reads two host-populated `__constant__` locations 16 KiB apart and verifies a
writable `__device__` location retains GPU updates across two launches. The test
also checks that hidden symbol buffers do not leak into the caller-visible CUDA
kernel ABI.

`functional_typed_direct_sgemm_2d` independently compiles the reviewed 2D
block-tiled SGEMM through direct NVVM and compares every element of a
64x64x16 result on Apple GPU, including nontrivial alpha and beta scaling. This
exercises sequential and nested natural loops, static shared tiles, and private
register arrays without the legacy PTX backend.

`functional_typed_direct_flash_attention` checks the independently compiled
typed direct FlashAttention kernel against scaled dot-product attention on the
CPU for every element of a 32x16 query tile. It covers the same sequential-loop
fix with dynamic shared SRAM and exp/online-softmax arithmetic.

`functional_typed_{direct,ptx}_device_printf` independently compiles Clang's
device `printf` corpus through both typed frontends and launches a 2x2 grid of
2x2x2 threadgroups on Apple GPU. Each gate parses all 32 atomic ring records,
checks the format id and three payload words, requires every block/thread pair
exactly once, and verifies that an exactly-full record is rejected without any
payload write. This is record-level numerical/ABI evidence; the existing
registered-runtime test separately covers host formatting and drain behavior.

On 2026-08-30, `functional_driver_module_load_data_ptx` and
`functional_runtime_registration_fatbin_ptx` loaded direct and ELF-embedded
version-`0x0101` PTX entries compressed with both LZ4 and Zstd, launched vector
addition, and checked every output on Apple GPU. The same parser rejected
truncated payload ranges, corrupt compressed sizes, dual-codec flags, and a
declared expansion beyond the 64 MiB ceiling before allocation. Registration
also rejected an unsupported entry version while a valid fallback metallib was
configured, proving that recognized invalid images cannot silently escape
through `CUMETAL_FATBIN_METALLIB`.

On 2026-08-29, the `fp64_precision` contract probe passed twice on Apple GPU:
once with an 11-kernel metallib compiled directly from Clang NVVM through typed
CuMetal IR, and once with each kernel compiled from PTX through the typed
frontend at registration time. Both runs reported zero violations for the
`fast48` 2^-48 relative bound, idempotent raw-binary64 packing, signed
zero/infinity/NaN, a five-operation chain, float-pair joining, shared-memory and
32-lane shuffle reductions, store/reload, `uint64_t` aliasing, comparisons,
fma/sqrt/min/max, remainder, and rounding. Values outside binary32's exponent
envelope remain explicitly excluded by the documented mode contract.

## Performance gate

The Phase 5 gate compares CuMetal with hand-written Metal for the selected
five-kernel memory-bound release set. On an Apple M4 Pro with Metal 4, Xcode
26.6, and Apple Metal compiler 32023.883, rebuilt and measured on 2026-08-29:

| Kernel | Elements | CuMetal / native Metal |
| --- | ---: | ---: |
| vector add | 262,144 | 1.103x |
| SAXPY | 262,144 | 1.202x |
| STREAM copy | 262,144 | 1.154x |
| STREAM triad | 262,144 | 1.031x |
| FP32 reduction | 262,144 | 1.070x |

The gate uses the fastest of 50 synchronized wall-clock iterations. These
kernels take roughly 0.2 ms, so averages mostly measure scheduler interference.
The target is at most 2x native Metal, not a claim that translated code beats
the baseline.

Reproduce it from a Release build with:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target cumetal_bench
ctest --test-dir build-release -R bench_phase5_all_kernels --output-on-failure
```

## cuFFT backend comparison

Both cuFFT implementations sit behind the same `cufftExec*` entry points and are
selected by `CUMETAL_FFT_METAL`. The dense, out-of-place, single-precision 3-D
R2C/C2R production path now uses vendored VkFFT 1.3.4 through its public Metal
backend; the project-owned Stockham/Bluestein implementation remains the
fallback for other accepted layouts. The table below records the fallback
Metal implementation before the VkFFT integration, rather than current VkFFT
performance. `cumetal_fft_bench` measures one backend per process; run it twice
to compare. Apple M4 Pro, Debug build, 2026-08-30, median of 20-60 iterations
after a warm-up that excludes MSL compilation and Bluestein filter construction:

| 3-D grid | Accelerate (ms) | Metal (ms) | ratio |
| --- | ---: | ---: | ---: |
| 32x32x32 | 6.70 | 0.27 | 24.8x |
| 40x32x32 (villin PME) | 8.29 | 0.42 | 19.9x |
| 56x56x56 (rnase PME) | 393.59 | 1.52 | 258x |
| 64x64x64 | 44.75 | 0.55 | 81.7x |
| 96x96x96 | 169.48 | 6.69 | 25.3x |
| 128x128x128 | 317.55 | 2.68 | 119x |

The figure is one R2C plus one C2R with a single synchronize, which is the shape
PME uses: forward transform, solve, inverse transform, all on one stream.

Every pass of a transform is staged in threadgroup memory when one line fits
there, so a length-L axis costs one device round trip instead of log2(L). That
alone is worth about 1.75x over dispatching each pass separately (villin
0.71 -> 0.51 ms, 128^3 4.69 -> 2.70 ms) and cuts a 40x32x32 R2C from 30
dispatches to 10.

Two things this table does **not** say. The CPU column is CuMetal's own
implementation, which drives vDSP one line at a time and falls back to a scalar
Bluestein for lengths vDSP cannot factor -- 56 and 96 both carry a factor of 7 or
3 -- so the largest ratios are mostly a statement about that path, not about
Accelerate. And the Metal column is not a tuned GPU FFT: at these sizes it is
bound by command-buffer submit-and-wait latency, which is why 32^3 and 64^3 land
within a factor of two of each other despite an 8x difference in work.

```bash
DYLD_LIBRARY_PATH=build build/cumetal_fft_bench --cpu
DYLD_LIBRARY_PATH=build build/cumetal_fft_bench --metal
```

### End-to-end effect on GROMACS

Same 20-step runs as `demos/gromacs`, `-nb gpu -pme gpu -bonded gpu -update gpu`,
warm kernel cache, median of two interleaved runs per configuration:

| Run | villin wall (s) | rnase_cubic wall (s) |
| --- | ---: | ---: |
| `-pme cpu`, cuFFT never called | 0.42 | -- |
| `-pme gpu`, Accelerate cuFFT | 0.63 | 8.95 |
| `-pme gpu`, Metal cuFFT | 0.46 | 0.57 |

Both differences are just the transform arithmetic. villin saves 0.17 s, and its
FFT is 8.29 ms per step against 0.42 ms over 21 steps, which is 0.17 s. rnase
saves 8.4 s against a predicted 8.2 s. Nothing else needs to be invoked to
explain either.

**A correction.** An earlier revision of this section reported villin at 58.17 s
with the Accelerate backend and attributed the difference to CuMetal's batched
command buffers being drained by the CPU path's `cudaStreamSynchronize`. That
measurement was taken on a cold kernel cache: the first run of a GROMACS binary
JIT-compiles its kernels from a 5 MB PTX module, which takes about 56 s and which
GROMACS charges to `Launch PP GPU ops` because that is where the first launch
happens. Rerunning the same command warm gives 0.63 s. The synchronize is real
and it is why the CPU path cannot overlap with other GPU work, but it is not
worth 57 s, and the number that said so was measuring the JIT.

### Matched GROMACS backend comparisons

For a timestep of `dt` femtoseconds, GROMACS throughput is
`ns/day = 86.4 * dt / (ms/step)`. Higher `ns/day` and lower `ms/step` therefore
express the same result. On 2 fs water cases, the recorded matched warm medians
are:

| Placement and case | CuMetal | Comparator | Relative throughput |
| --- | ---: | ---: | ---: |
| Official 96k, GPU nonbonded only | 28.828 ns/day | AdaptiveCpp Metal, 3.142 ns/day | **9.175x** |
| Official 96k, GPU nonbonded + PME | 63.392 ns/day | Native Metal, 57.797 ns/day | **1.097x** |
| Reconstructed 98,319 atoms, full GPU | 46.853 ns/day | Native Metal, 41.183 ns/day | **1.138x** |
| 1,005,375 atoms, full-GPU structural stress | 5.615 ns/day | Native Metal, 5.274 ns/day | **1.065x provisional** |

Every row uses the same TPR and task placement within that row, and each has a
separate deterministic energy check. Rows are not compared across placements.
The large case is provisional because it was not equilibrated. The complete
all-cases target is not yet verified: AdaptiveCpp generic/Metal lacks the GPU
FFT/PME connection and the whole public corpus has not been run as paired warm
series through all three backends. Full commands, samples, and provenance are
in the [GROMACS guide](../demos/gromacs/README.md).

## Real programs

- Upstream `cuda-samples` vector addition builds without source changes and
  passes a numerical plus Apple-GPU provenance gate.
- Upstream `cuda-samples` `simplePrintf` builds without source changes and emits
  all 32 expected block/thread/value records through the Apple-GPU ring-buffer
  path. The focused in-tree Clang-ABI test independently checks the same record set.
- Upstream `cuda-samples` `simpleCUFFT` builds without source changes and passes
  its numerical convolution check. This exercises a 56-point cuFFT transform,
  a device pointwise-multiply kernel, and stream ordering between Metal work and
  the synchronous CPU/vDSP compatibility layer.
- Focused graph runtime tests pass allocation/free nodes, fixed returned
  addresses, linear copy nodes, synchronous and asynchronous external free,
  relaunch, cross-graph free, auto-free-on-launch, memory counters, trimming,
  and negative parameter/lifetime cases.
- llm.c GPT-2 FP32 passes logits, loss, tensor, and GPU-provenance checks on the
  tested path. It uses explicit workload specializations and is not proof of
  arbitrary PTX support.
- llama.cpp's unmodified GGML CUDA backend builds against CuMetal. SmolLM2-135M
  greedy decoding was coherent from one-layer offload through saturation on the
  verified Apple M4 Pro setup. FlashAttention is advertised as unsupported, so
  llama.cpp selects its ordinary attention path.
- A reduced PhysX 5.6 GRB path runs selected sphere, box, convex, and triangle
  mesh contacts on the GPU. It is a selected-shape conformance target, not
  general PhysX GPU support.

Exact commands, models, tolerances, provenance requirements, and scope
boundaries live in [the Apple-GPU execution record](apple-gpu-execution.md),
[the testing guide](testing.md), and [known gaps](known-gaps.md).

## NVIDIA cuda-samples conformance snapshot

The manifest snapshot recorded on 2026-08-29 classifies all 83 enrolled
headless samples as runtime passes, with no waivers or nonpassing entries.
Every entry must build, run, and satisfy its available numerical/output checks.
The manifest is an executable compatibility boundary: any regression from
`pass` fails, and classifications must be reviewed when the enrolled set changes.

This is complete coverage of the enrolled snapshot only. It is not a percentage
of the full CUDA API, all CUDA samples, or arbitrary CUDA applications. The
exact implementation boundaries remain in [known gaps](known-gaps.md).

## Projects using CuMetal

The following result is third-party work, verified by its author rather than by
this repository.

### cu_vslam_rs

[cu_vslam_rs](https://github.com/jeff-hykin/cu_vslam_rs) by
[@jeff-hykin](https://github.com/jeff-hykin) compiles NVIDIA's
[cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) visual-odometry stack for
Apple Silicon against CuMetal and packages it as an SDK with a Nix flake
(`nix build ...#sdk-metal`). NVIDIA ships no macOS build. Its CUDA kernels are
not rewritten in Metal; they use CuMetal's PTX path.

The project's `metal_smoke` test asserts actual camera motion rather than a
success status because CuMetal defects have historically returned success while
producing an identity pose. Stereo runs on either backend. RGB-D requires the
GPU because cuVSLAM v17 lifts depth into landmarks only in a CUDA kernel.

The working configuration required
`CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`: its feature detector builds texture
objects over linear memory and dereferences the resource pointer in device code,
which reads as zero under CuMetal's default addressing without an error.
`cudaCreateTextureObject` now warns about that case instead of failing silently.

If you have shipped something on CuMetal, open a pull request adding it here.
