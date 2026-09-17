# Bounded address-demand and local-memory proofs: #76 follow-up

This follow-up completes the two proof-scaling gaps exposed after the scalar
zero-guard change. It is based on PR #139 at
`f7ceeeff731181112e691f3b3dbc872c17ef7743`. It preserves the original conversion,
alignment and production acceptance recorded in
[the earlier #76 report](issue-76-acceptance-tests.md).

## What changed

Address demand now indexes each SSA join once and expands predecessor edges
only when a demanded address reaches that join. Every relevant predecessor and
backedge is retained. Sparse local overrides avoid copying whole incoming
register environments. Every load remains available for independent typed-load
validation; demand is not evidence that an integer contains a pointer.

The work limit remains 1,000,000. Indexing, scans, lookups, insertions and expanded
edges consume that limit. Exhaustion returns no partial demand/load result.
Unit tests cover irrelevant large joins, genuinely demanded predecessors,
self-updating values, call returns, independently typed loads, and transactional
exhaustion.

Scalar range analysis now intersects all applicable dominating guards, trims
excluded interval endpoints, and relates bounded affine siblings of the same
SSA value. Plain 64-bit copies/additions/subtractions retain modulo-2^64
identity; an interval transfers only when its resulting signed representation
is representable. This does not infer a bounded base from a wrapping sibling.
Necessary Boolean facts follow true ANDs and false ORs, with predicate copies
and negation. Other Boolean outcomes, ambiguous definitions and incomplete
proofs remain unknown. Already completed cached facts remain usable after
later queries exhaust the unchanged work limit. The query limit does not bound
the preexisting eager constructor indexing.

Pointer iterators additionally support a positive literal endpoint and the
guard shape `select(end == next, 0, next) != 0`. Nonzero output with one literal
zero arm proves which selector branch was taken; it does not establish pointer
provenance for the other arm. The existing allocation, entry, every-backedge,
stride and overflow checks still apply. No proof budget was increased.

## Numerical and refusal checks

Apple M5, macOS 26.6.2, LLVM/Clang 21.1.8; both configurations build:

| Configuration | Selected unit suite | PTX functional suite |
| --- | --- | --- |
| Release, shim OFF | 57 pass, one benchmark-precondition skip | 57 pass after correcting/rerunning the negative fixture, one offline-tool skip |
| Debug, shim ON | 60 pass, same skip | 57 pass, same offline-tool skip |

The unit suites exclude `unit_metallib_parser` and `unit_cumetal_cli`, whose
Apple reference/CLI prerequisites are unavailable. These are scoped suite
results, not an all-project pass. The Release suite's only initial failure was
the negative-test construction corrected below; its focused rerun passes.

`functional_ptx_memory_ranges` contains 16 GPU fixtures with 65 lane positions
each. The six added fixtures cover direct/copied endpoint guards, affine sibling
guards, a compound guard, and variable/literal zero-sentinel iterators. Endpoint
fixtures check four output words per lane; iterator fixtures check the preserved
input plus all 64 array bytes. CPU expectations include zero, maximum valid
indices, excluded endpoints, signed-bit boundaries and wrapping affine inputs.
Input immutability, output guards and generic Apple GPU provenance are checked.
Both configurations pass all 16 fixtures and all 29 compile-only refusal cases:
1,040 lane positions per configuration, not unique inputs. The frozen parent
compiler `2632e682ea0bb08c0a93e3c474790bdf6d6073ec24305c7d762c1e2773afc33f`
rejects five of the six added positive fixtures. The small literal64 iterator
already emits MSL on that parent and is retained as regression coverage.

Compile-only refusal cases cover absent, bypassed and overwritten guards,
overlapping writes, foreign scalar bases, unknown lengths, nonzero sentinels,
overwritten sentinels and unsupported strides. They are never launched on the
GPU. A first negative test accidentally made its observed load unreachable by
creating an unconditional loop; it was corrected to keep the load reachable
while adding a backedge that bypasses the guard. All 16 numerical cases passed
in that first run; final suite results are recorded below.

Host tests additionally cover declaration/SSA identity, redefinitions,
ambiguous joins, foreign endpoint allocations, arithmetic limits, all relevant
guard polarities, demanded graph edges, and proof-budget exhaustion.

## Full-input scope

The unchanged LLVM7 inputs come from producer
`afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`,
[Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652):

| Input | SHA-256 | Next independently owned gate |
| --- | --- | --- |
| Bitcoin self-tests | `0bbd7000a8949ff1ca3776c938c9f72f14da477f388a13983bb8e5cce4bb25bc` | #136: helper-call write footprints |
| RSA-PSS self-tests | `af511dc9a746971a79e062e0165c82f78fe7e9b2d20f83067fd387b64aed4407` | #141: direct `clz.b64` normalization |

Final clean Release compiler SHA-256:
`9dbf05db25e2b3e2f8e510949116034f2043aaaa068b0574ca9354decc578425`.
Serial replays reject at the next gates in **97.974 s** (Bitcoin) and
**80.260 s** (RSA-PSS), within 300-second deadlines. Both input hashes remain
unchanged before/after. Temporary instrumentation is removed. These are
development-compiler checks; the matched immutable consumer replay is recorded
separately after publication.

```sh
cumetalc self_test_MODE.ptx --backend=cumetal-ir --ptx-strict \
  --entry kernel_self_test_MODE --emit=msl -o MODE.metal
```

Bitcoin's saved pointer occupies bytes `[256,264)` at PTX line 34534. The new
proofs clear byte stores at 33839, 33852, 31699 and 26729 without enumerating the
large initialized-prefix state. The next rejected call, field inversion at
24791, writes its output at `[496,537)` and reads `[696,736)`. Its two nested
`subtle::black_box` calls only read/write scalar parameter slots. Instantiating
that transitive parameter-relative footprint belongs to #136. Earlier calls
still need checking after it clears.

RSA-PSS clears address-demand graph construction and reaches unsupported
`clz.b64` at line 55734, owned by #141. Neither result is full translation,
Apple pipeline preparation or GPU workload acceptance. Downstream #30 and #35
must remain open. Mixed vector lane provenance (#118), helper provenance
(#140), helper writes (#136), direct CLZ (#141), and pipeline cost (#133) remain
separate work.

The prior native legacy numerical gate still requires unavailable offline
Apple Metal tools. This patch does not claim that gate or upstream integration
is complete.
