# CuMetal specification

**Version:** 0.4

**Target:** macOS 14+ on Apple Silicon

**License:** Apache 2.0

This file is the canonical specification index. The linked chapters are jointly
normative. If they conflict with `AGENTS.md`, README, status, or historical
notes, this index and its chapters win in the listed order.

## Normative chapters

1. [Purpose, principles, and scope](spec/01-scope.md)
2. [Compiler architecture](spec/02-compiler.md)
3. [Runtime architecture](spec/03-runtime.md)
4. [CUDA semantic contracts](spec/04-semantics.md)
5. [Build, verification, and release gates](spec/05-verification.md)
6. [Roadmap and closure criteria](spec/06-roadmap.md)
7. [Legal and clean-room requirements](spec/07-legal.md)

## Reading status correctly

The specification defines what CuMetal must become. It is not evidence that a
feature is implemented. Use these separate indexes:

- [Current status](docs/status.md) — implemented surfaces
- [Known gaps](docs/known-gaps.md) — partial, absent, and bounded behavior
- [Verified results](docs/verified-results.md) — measured results and provenance
- [Closure roadmap](docs/spec-closure-roadmap.md) — current priority order
- [Documentation index](docs/README.md) — all maintained guides and records

No README statement, registered test, skip, source stub, or API symbol is proof
of compatibility by itself. A claim is current only when its documented gate
passes with the required numerical and device evidence.

## Precedence and change control

1. `spec.md` and `spec/*.md` are canonical.
2. `AGENTS.md` governs repository workflow where the specification is silent.
3. `docs/status.md`, `docs/known-gaps.md`, and evidence pages describe the
   current implementation and may lag briefly, but must be reconciled before a
   release.
4. README is a concise entry point, not an independent contract.

Any behavior change must include focused tests and corresponding status/gap
updates. Partial behavior must be named as partial. Changes to durable platform,
legal, precision, or source-first boundaries require an explicit specification
edit.
