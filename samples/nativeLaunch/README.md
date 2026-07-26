nativeLaunch Sample
===================

This sample uses CuMetal's **native** launch API (`cumetalKernel_t`, see
`runtime/api/cumetal_native.h`) instead of the CUDA registration path. Host and device code live
in separate files, and the host names the kernel and its argument layout explicitly, then passes
the `.metallib` path at runtime.

Use this when you want to load a prebuilt `.metallib` yourself — for embedding CuMetal in an
existing Metal application, or for testing a kernel without a CUDA host program.

**For ordinary CUDA programs, use [`../vectorAdd`](../vectorAdd) instead**: a single `.cu` file
built with `cumetalc vectorAdd.cu -o vectorAdd`, which needs none of this wiring.

Build and run
-------------

1. Compile the kernel to a `.metallib`:

```bash
./build/cumetalc --mode xcrun \
  --input samples/nativeLaunch/nativeLaunch.cu \
  --output /tmp/nativeLaunch.metallib \
  --overwrite
```

2. Compile and link the host program against `libcumetal`:

```bash
xcrun clang++ -std=c++20 \
  samples/nativeLaunch/nativeLaunch.cpp \
  -Iruntime/api \
  -Lbuild \
  -Wl,-rpath,build \
  -lcumetal \
  -o /tmp/nativeLaunch
```

3. Run, passing the metallib path:

```bash
/tmp/nativeLaunch /tmp/nativeLaunch.metallib
```
