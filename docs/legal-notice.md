# Legal Notice

CuMetal is a clean-room implementation. This is engineering policy, not legal advice.
The document records the intended technical boundaries of the project's source and
compatibility paths; it does not determine whether a particular use complies with a license
or law.

---

## Source Recompilation and Binary Compatibility

NVIDIA's license terms have included restrictions concerning translation layers on
non-NVIDIA platforms. Those terms can change and their application depends on the software,
license, jurisdiction, and facts involved. Users must review the terms that govern their use
and obtain legal advice when needed.

CuMetal's primary and recommended engineering path recompiles source that the user controls.
Its optional binary alias is a technically distinct compatibility path and is not presented as
having the same licensing or interoperability considerations.

| Usage model | Project engineering status | User responsibility |
|-------------|----------------------------|---------------------|
| Recompile controlled `.cu` source with `cumetalc` | Primary recommended path | Confirm rights to the source and applicable dependencies |
| Link a source-built program against `libcumetal.dylib` | Source-recompilation runtime path | Review the program and dependency licenses |
| Load the `libcuda.dylib` alias for a PTX-bearing binary | Bounded, opt-in compatibility path | Review all governing terms before use |
| Load a SASS-only binary | Unsupported | CuMetal does not translate or execute SASS |

CuMetal's binary shim is the **`libcuda.dylib` alias**, and only that alias. It is provided as an
opt-in convenience and is **disabled by default** (`CUMETAL_ENABLE_BINARY_SHIM=OFF` in release
builds). Its use is at the user's own discretion and risk.

The host **registration ABI** (`__cudaRegisterFatBinary`, `__cudaRegisterFunction`,
`__cudaPopCallConfiguration`) is a separate thing and is built unconditionally
(`CUMETAL_ENABLE_CUDA_REGISTRATION=ON`). Clang emits calls to those symbols when it compiles
controlled `.cu` source, so they are required by the source-recompilation path and are not the
optional alias. They implement the compiler's host calling convention; their presence alone
does not make an arbitrary binary compatible or determine its legal status.

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

CuMetal's production compiler emits typed Metal Shading Language and invokes Apple's public
`xcrun metal` and `xcrun metallib` tools to produce `.metallib` files. Direct AIR/container
generation, `air_inspect`, and `air_validate` are research and regression tooling only; the
project does not claim a private AIR ABI as a supported production interface.

AIR/metallib research inspects outputs produced by publicly distributed Apple tools. Project
policy prohibits accessing or copying Apple proprietary source, using private Apple APIs, or
redistributing Apple code. Whether a particular interoperability activity is permitted is a
fact- and jurisdiction-specific legal question outside this engineering notice.

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

CuMetal's architectural policy is independent of another project's legal history: the primary
path is source recompilation, and binary compatibility remains bounded and opt-in.

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
