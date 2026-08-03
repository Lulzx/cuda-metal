# Apollo demo

**CUDA programs. No NVIDIA GPU. Real Metal execution. Provenance on every launch.**

```bash
# from a built CuMetal tree
bash demos/apollo/run.sh
```

That is the whole pitch. The script compiles ordinary `.cu` files with `cumetalc`,
runs them on the Apple GPU, and refuses to call anything a pass unless the log
contains both:

```text
device=apple_gpu
launch_success=true
```

A correct number without that provenance is **not** a pass. CPU fallback is not a pass.

---

## What runs

| Stage | Workload | Why it is in the demo |
| --- | --- | --- |
| **1** | `samples/vectorAdd` | The hello-world. Ordinary CUDA, `<<<>>>` launch, numerical check. |
| **2a–d** | reduction, transpose, softmax, SGEMM | Shared memory, warps, real linear-algebra shapes — not a toy. |
| **3** | Ray Tracing in One Weekend | Branchy path tracer; GPU image must match CPU reference (~71 dB PSNR on M4 Pro). Writes `out/rtiow.ppm`. |
| **4** *(optional `--full`)* | llm.c GPT-2 FP32 | Full forward + backward + AdamW with logits/loss/tensor gates. |

Modes:

```bash
bash demos/apollo/run.sh            # stages 1–3  (~1–2 min)
bash demos/apollo/run.sh --quick    # vectorAdd + raytracer
bash demos/apollo/run.sh --full     # + llm.c (needs ../llm.c assets + binary)
```

Artifacts land in `demos/apollo/out/`:

| File | Contents |
| --- | --- |
| `report.txt` | Human-readable stage results |
| `provenance.log` | Every `CUMETAL_PROVENANCE` line |
| `rtiow.ppm` | GPU-rendered path-traced image |
| `*.log` | Per-stage compile + run output |

---

## Why this exists

CuMetal already had the hard work: compiler, runtime, numerical gates, llama.cpp
and llm.c paths. What it lacked was a **single front door** a stranger can run
and screenshot.

Apollo is that door. It does not claim general CUDA compatibility. It claims:

> On this machine, these CUDA programs compiled, launched on the Apple GPU,
> produced correct answers, and left a paper trail.

That is enough to be interesting. It is not enough to replace CUDA.

---

## Prerequisites

1. macOS 14+ on Apple Silicon  
2. Xcode Metal toolchain (`xcrun metal`)  
3. A Release (or equivalent) CuMetal build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)"
bash demos/apollo/run.sh --build-dir=build
```

The script also auto-detects `build-release`, `build`, `build-nosshim`, `build-noshim`.

### Full mode (llm.c)

```bash
# weights + prebuilt test binary (or build via scripts/build_llmc_test_gpt2fp32cu.sh)
export CUMETAL_LLMC_DIR=../llm.c
bash demos/apollo/run.sh --full
```

Requires `gpt2_124M.bin` (and the debug-state assets llm.c expects) under that directory.

---

## What a pass looks like

```text
PASS  1   vectorAdd (hello CUDA on Metal)
PASS  2a  parallel reduction (1M elements)
PASS  2b  matrix transpose (naive + shared mem)
PASS  2c  softmax (block + warp)
PASS  2d  SGEMM naive (siboehm kernel)
PASS  3   Ray Tracing in One Weekend (GPU == CPU reference)

APOLLO PASSED
```

Example provenance line (abridged):

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="_Z10vector_add..."
  device=apple_gpu device_name="Apple M4 Pro"
  launch_success=true duration_ns=...
```

Open `out/rtiow.ppm` in Preview. That image was rendered by a CUDA kernel on Metal.

---

## Scope (read this before tweeting)

- Covered paths only. See [docs/known-gaps.md](../../docs/known-gaps.md).
- No dynamic parallelism, no multi-GPU, no graphics interop, no SASS.
- llm.c uses explicit workload specializations on the verified path.
- FlashAttention and arbitrary llama.cpp offload are **not** part of this demo.
- FP64 is emulated (~44-bit mantissa), not IEEE binary64.

If a stage fails on your machine, file an issue with `out/report.txt` and the
failing `out/<stage>.log`. Do not paraphrase “it didn’t work.”

---

## One sentence

**Ordinary CUDA source, compiled by CuMetal, executed on Apple Silicon GPUs,
checked for answers and for `device=apple_gpu` on every launch.**
