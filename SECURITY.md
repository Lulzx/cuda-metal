# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |

CuMetal follows the latest release. Fixes land on `main` and ship in the next release; there are
no long-term support branches.

## Reporting a vulnerability

**Do not open a public issue.** Report privately through
[GitHub Security Advisories](https://github.com/lulzx/cumetal/security/advisories/new), which
opens a channel visible only to you and the maintainers.

Please include:

- CuMetal commit or release version
- macOS version, Xcode version, and chip (M1/M2/M3/M4)
- CMake configuration, in particular whether `CUMETAL_ENABLE_BINARY_SHIM` was `ON`
- A minimal reproducer — the `.cu`, PTX, or `.metallib` that triggers it
- What an attacker gains

You should get an acknowledgement within 7 days and an assessment within 30. If a fix is
warranted we will agree a disclosure date with you; the default is publication once a fix is
released. Credit is given unless you prefer otherwise.

## Threat model

CuMetal compiles and runs code you supply, so the boundary matters more than usual:

**In scope.** Bugs where processing input that a user did not author leads to compromise:

- Memory corruption in the compiler while parsing a malicious `.ptx`, `.cu`, NVVM `.ll`, or
  fatbinary — the compiler is the component most likely to meet untrusted input, since PTX and
  fatbinaries arrive inside third-party binaries.
- Memory corruption in the runtime parsing a malformed `.metallib`, fatbinary envelope, or
  registration record.
- Path traversal or arbitrary file write from attacker-controlled names in a fatbinary, module
  cache key, or JIT cache entry.
- A compiled kernel escaping the buffer bounds the host bound for it, where the CUDA source did
  not ask for that access.
- Cache poisoning: getting the JIT/module cache to return a metallib that did not come from the
  input it is keyed on.
- Privilege or sandbox escape from anything in the runtime.

**Out of scope.**

- A kernel reading or writing out of bounds *within its own process* when the CUDA source
  already did so. CUDA has no memory safety and neither does Metal; CuMetal does not add bounds
  checking, and reproducing an existing bug in the input program is not a CuMetal vulnerability.
- Compiling hostile source you deliberately fed to `cumetalc`. Running the compiler on untrusted
  input is equivalent to running any compiler on untrusted input — do it in a sandbox.
- Denial of service through compile time or memory use on pathological input.
- **Numerically wrong results.** These are taken very seriously — see
  [docs/correctness-audit-2026-07-26.md](docs/correctness-audit-2026-07-26.md) — but report them
  as ordinary bugs in the public tracker, with the expected value and how you obtained it. They
  get priority treatment without needing embargo.
- Issues in llama.cpp, llm.c, PhysX, or any other project CuMetal is tested against. Report
  those upstream.
- Anything requiring an already-compromised machine or a malicious `DYLD_LIBRARY_PATH`.

## A note on the binary shim

Building with `CUMETAL_ENABLE_BINARY_SHIM=ON` installs a `libcuda.dylib` alias so binaries
pre-linked against NVIDIA's driver load CuMetal instead. That means CuMetal parses fatbinaries
from software it did not compile, which is the largest untrusted-input surface in the project.
It is off by default in Release builds, and [docs/legal-notice.md](docs/legal-notice.md) covers
the separate licensing considerations. Treat fatbinaries from untrusted binaries as hostile
input; parser bugs reachable this way are in scope.
