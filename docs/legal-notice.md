# Legal Notice

CuMetal is a clean-room implementation. This document explains the project's legal posture
with respect to NVIDIA's CUDA EULA and Apple's AIR ABI.

---

## NVIDIA CUDA EULA — Translation Layer Warning

As of 2024, NVIDIA's CUDA EULA contains a clause prohibiting the use of translation layers
to run CUDA code on non-NVIDIA hardware. This clause applies to **binary drop-in shims**
that intercept CUDA API calls and redirect them to non-NVIDIA GPU backends (e.g., loading
a replacement `libcuda.dylib` that intercepts fatbinary registration).

**This clause does NOT apply to source-level recompilation.**

| Usage Model | Legal Risk | Notes |
|-------------|------------|-------|
| Recompile `.cu` source with `cumetalc` | **None** — you compiled your own source code with a different compiler | Primary recommended path |
| Link open-source CUDA programs against `libcumetal.dylib` | **Low** — no NVIDIA binary involved | Research and personal projects |
| Drop-in `libcuda.dylib` for closed-source PTX-shipping binaries | **High** — may violate NVIDIA EULA | Opt-in only; use at own risk |
| Drop-in for closed-source SASS-only binaries | Not supported | SASS is not translatable |

CuMetal's binary shim is the **`libcuda.dylib` alias**, and only that alias. It is provided as an
opt-in convenience and is **disabled by default** (`CUMETAL_ENABLE_BINARY_SHIM=OFF` in release
builds). Its use is at the user's own discretion and risk.

The host **registration ABI** (`__cudaRegisterFatBinary`, `__cudaRegisterFunction`,
`__cudaPopCallConfiguration`) is a separate thing and is built unconditionally
(`CUMETAL_ENABLE_CUDA_REGISTRATION=ON`). Clang emits calls to those symbols when it compiles
*your own* `.cu` source, so they are part of the source-recompilation path — the row marked
"None" in the table above — and carry none of the binary shim's exposure. No NVIDIA binary is
involved: you compiled your code with a different compiler, and these are that compiler's host
calling convention. The two switches were previously one, which meant a release build silently
disabled the source path along with the alias.

---

## Clean-Room Implementation

**No NVIDIA headers are shipped.** CuMetal's `cuda.h` and `cuda_runtime.h` are clean-room
implementations that match the public CUDA API specification without copying any NVIDIA
source material. Contributors are required to confirm clean-room status via CLA.

**No SASS decompilation.** CuMetal processes only PTX (NVIDIA's documented virtual ISA),
not SASS (native GPU machine code). PTX is a stable, documented, publicly specified
intermediate representation. No proprietary internal NVIDIA specifications are required.

**No NVIDIA source code was used.** The runtime shim, compiler passes, and PTX parser
were written without reference to any NVIDIA proprietary source. The PTX parser is derived
from the ZLUDA project's `ptx` crate (Apache 2.0), which is itself a clean-room parser.

---

## Apple AIR ABI

CuMetal generates `.metallib` files that conform to Apple's AIR (Apple Intermediate
Representation) ABI. This ABI was reverse-engineered from publicly distributed Apple
toolchain outputs (`.metallib` files produced by `xcrun metal`). No Apple proprietary
source code was accessed or copied.

**Legal basis for interoperability reverse engineering:**

- **United States**: *Sega v. Accolade*, 977 F.2d 1510 (9th Cir. 1992) — reverse
  engineering for interoperability is fair use under U.S. copyright law. *Sony Computer
  Entertainment v. Connectix*, 203 F.3d 596 (9th Cir. 2000) — intermediate copying
  during reverse engineering is permissible when the final product contains no copied code.
- **European Union**: Directive 2009/24/EC, Article 6 — decompilation for interoperability
  is permitted without authorization from the rightholder when necessary to achieve
  interoperability of an independently created computer program.
- **Key distinction**: CuMetal does not decompile or redistribute any Apple binary. It
  generates new `.metallib` files that conform to the reverse-engineered ABI specification.
  No Apple code is included in CuMetal's output.

The AIR ABI reverse engineering is documented in `docs/air-abi.md` and builds on the
open-source community work of:
- [MetalLibraryArchive](https://github.com/YuAo/MetalLibraryArchive) (YuAo) — MIT license
- [MetalShaderTools](https://github.com/zhuowei/MetalShaderTools) (zhuowei)
- [DougallJ's Apple GPU ISA documentation](https://github.com/dougallj/applegpu)

---

## ZLUDA PTX Parser

The PTX parser is derived from the ZLUDA project's `ptx` crate, licensed under Apache 2.0.
ZLUDA is used with attribution. Modifications to the parser are maintained in CuMetal's
fork and may be contributed back upstream.

ZLUDA's CUDA EULA history: ZLUDA initially provided a binary drop-in CUDA replacement for
AMD GPUs. AMD ceased funding ZLUDA in 2022 after NVIDIA added the translation-layer clause
to its EULA. CuMetal draws the same strategic lesson: the primary path is source
recompilation, not binary translation.

---

## Contributor License Agreement

The full text is in [docs/cla.md](cla.md), and [CONTRIBUTING.md](../CONTRIBUTING.md) explains
the workflow. Contributors sign it by adding `Signed-off-by` to each commit (`git commit -s`),
which certifies the Developer Certificate of Origin 1.1 plus CuMetal's clean-room clauses:

1. The contributed code is a clean-room implementation — no NVIDIA proprietary source
   material was referenced or copied.
2. No prior exposure to NVIDIA proprietary source code for the implemented API surface.
3. Nothing derived from NVIDIA SASS disassembly or decompilation.
4. Any AIR/`metallib` ABI knowledge came from publicly distributed Apple toolchain *output*,
   not from Apple source code or from decompiling an Apple binary.
5. The contribution is original work or is properly attributed, license-compatible open-source
   code.
6. The contributor has the legal right to make the certification.

There is no copyright assignment: contributors retain ownership and license under Apache 2.0.

---

## No Warranty

CuMetal is provided "as is" without warranty of any kind. The project makes no
representation regarding legal compliance in any specific jurisdiction. Users are
responsible for determining the legality of CuMetal's use in their context.
