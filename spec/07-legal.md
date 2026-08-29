# Legal and clean-room requirements

[Specification index](../spec.md)

This chapter records project engineering policy, not legal advice. The detailed
notice is `docs/legal-notice.md`.

## Source-first boundary

Recompiling source the user controls is the primary and recommended use. The
optional binary alias is a distinct compatibility mode with different legal and
technical risk. It must remain disabled by default in Release packaging and may
not shape the canonical compiler architecture.

## Clean-room rules

- Do not import, copy, or redistribute NVIDIA headers or proprietary source.
- Implement CUDA-facing declarations from public behavior/specification using
  project-owned clean-room headers.
- Process documented PTX only; do not decompile or translate SASS.
- Use only public Apple frameworks and tools. Do not call private Apple APIs.
- AIR/metallib research may inspect publicly produced toolchain outputs for
  interoperability; do not redistribute Apple code or claim a private ABI as a
  supported production interface.
- Preserve third-party licenses and attribution for incorporated code.
- Contributors must follow the repository CLA and provenance requirements.

## Documentation language

Do not state that source recompilation, binary interception, and closed-source
drop-in use have identical legal posture. Do not promise that an accepted
container or symbol table makes an application compatible. Link users to the
legal notice and describe the binary shim as opt-in and bounded.
