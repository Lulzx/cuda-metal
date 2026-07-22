# cumetalc

`cumetalc` is the command-line compiler driver entrypoint.

Current implementation is a thin wrapper over `cumetal-air-emitter`:

```bash
cumetalc --mode xcrun --input kernel.metal --output kernel.metallib --overwrite
cumetalc --mode xcrun kernel.metal -o kernel.metallib --overwrite
cumetalc --mode experimental --input kernel.ptx --output kernel.metallib --entry kernel_name --ptx-strict --overwrite
cumetalc --mode experimental --input kernel.cu --output kernel.metallib --overwrite
cumetalc --cuda-device --mode experimental kernel.cu -o kernel.metallib \
  --entry kernel_name --ptx-strict -I dependency/include -D FEATURE=1 --overwrite
```

If `-o/--output` is omitted, output defaults to `<input-stem>.metallib`
(for `.metal`, `.ptx`, and `.cu` inputs).

For `.ptx` input, `cumetalc` lowers PTX to temporary LLVM IR internally (via phase1
pipeline) and then invokes the AIR emitter.

The legacy `.cu` input path invokes `xcrun clang++` with CUDA qualifiers stripped and
only supports simple prototype kernels.

`--cuda-device` selects the real CUDA source frontend. It invokes a CUDA-capable
Homebrew LLVM `clang++` (or `--cuda-clang`/`CUMETAL_CUDA_CLANG`) in device-only mode,
emits PTX, then runs CuMetal's PTX lowering and AIR emitter. `-I`, `-D`,
`--cuda-include`, and `--cuda-arch` are available for project builds.
`--cuda-inline-threshold <n>` forwards Clang's GPU inlining threshold and requests
inlining of every viable call reachable from the selected kernel. It is useful for
projects whose PTX contains helper `.func` definitions, which CuMetal does not yet
lower independently. Recursion, indirect calls, and explicitly non-inlineable
functions still fail during strict PTX lowering. Jump tables are disabled because
CuMetal currently lowers structured PTX branches rather than `brx.idx` target tables.

When built from a CuMetal source checkout, this `.cu` frontend path automatically adds
`runtime/api/` to the include path for clean-room CUDA headers.
