# frontend

CUDA source frontend integration lives here.

The canonical source path is:

```text
stock Clang CUDA device job (-nocudainc -nocudalib)
→ LLVM/NVVM device IR
→ CuMetal NVVM importer
→ verified CuMetal GPU IR
→ Metal legalization
→ typed MSL
→ Apple Metal compiler
```

`cumetalc --backend=cumetal-ir` uses CuMetal's clean-room runtime headers and
does not strip CUDA qualifiers. `--emit=llvm`, `cumetal-ir`, `metal-ir`, and
`msl` expose the intermediate stages. Unknown NVVM intrinsics and unsupported
LLVM constructs are diagnosed rather than replaced with traps or fallback.

The old qualifier-stripping path remains only under the temporary explicit
legacy backend during migration.
