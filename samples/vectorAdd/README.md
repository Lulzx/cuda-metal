vectorAdd Sample
================

`vectorAdd.cu` is an ordinary CUDA program — host code, `__global__` kernel, and a `<<<>>>`
launch in a single file, with no CuMetal-specific API calls. Build and run it:

```bash
./build/cumetalc samples/vectorAdd/vectorAdd.cu -o /tmp/vectorAdd
/tmp/vectorAdd
```

Expected output:

```
PASS: samples/vectorAdd produced correct output for 16384 elements
```

To confirm the kernel really executed on the Apple GPU rather than any host path:

```bash
CUMETAL_TRACE_GPU=1 /tmp/vectorAdd
```

which prints a provenance record naming the device and the semantic quality of the lowering.

How it works
------------

`cumetalc` builds an executable when the input is `.cu` and `-o` does not name a `.metallib`
(pass `--link` or `--no-link` to force either behavior). It drives Clang over the whole
translation unit: the host side compiles to the standard CUDA registration ABI, and the device
side goes through CuMetal's `ptxas`/`fatbinary` shims, which carry PTX into the fatbinary
envelope instead of assembling SASS. `libcumetal` translates that PTX to a `.metallib` on first
launch. The device code ships inside the executable, so nothing needs a metallib path at runtime.

Device code only
----------------

To stop at a `.metallib` instead — useful when you are inspecting the generated Metal or
loading the library yourself through `cumetalKernel_t`:

```bash
./build/cumetalc samples/vectorAdd/vectorAdd.cu -o /tmp/vectorAdd.metallib
./build/cumetalc samples/vectorAdd/vectorAdd.cu --emit msl -o /tmp/vectorAdd.metal
```
