# CuMetal Contributor Agreement

This is the text you certify when you add `Signed-off-by` to a commit (`git commit -s`). It
combines the standard Developer Certificate of Origin with CuMetal's clean-room clauses, which
[docs/legal-notice.md](legal-notice.md) depends on.

There is no separate form and no copyright assignment: you keep ownership of your contribution
and license it under Apache 2.0, the same license as the project.

---

## Part A — Developer Certificate of Origin 1.1

> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I have the right to submit it
> under the open source license indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best of my knowledge, is covered
> under an appropriate open source license and I have the right under that license to submit that
> work with modifications, whether created in whole or in part by me, under the same open source
> license (unless I am permitted to submit under a different license), as indicated in the file; or
>
> (c) The contribution was provided directly to me by some other person who certified (a), (b) or
> (c) and I have not modified it.
>
> (d) I understand and agree that this project and the contribution are public and that a record
> of the contribution (including all personal information I submit with it, including my
> sign-off) is maintained indefinitely and may be redistributed consistent with this project or
> the open source license(s) involved.

The DCO is © 2004, 2006 The Linux Foundation, distributed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

## Part B — Clean-room certification

In addition, for each contribution I submit to CuMetal, I certify that:

1. **No NVIDIA proprietary source material** was referenced, copied, adapted, or consulted in
   creating the contribution. I may have used publicly published specifications and
   documentation — the CUDA Programming Guide, the PTX ISA specification, published API
   interface documentation — but not leaked headers, decompiled binaries, or internal NVIDIA
   source code.

2. **No prior exposure.** I have not had access to NVIDIA proprietary source code implementing
   the specific API surface this contribution implements. Where I have had such exposure, I have
   not contributed to that API surface.

3. **No SASS-derived work.** The contribution contains nothing derived from disassembling,
   decompiling, or reverse engineering NVIDIA SASS native machine code.

4. **No Apple proprietary source.** Any AIR/`metallib` ABI knowledge in the contribution was
   derived from publicly distributed Apple toolchain *outputs* — files produced by running
   `xcrun metal` — or from published open-source community research, not from Apple source code
   or from decompiling an Apple binary.

5. **Third-party code is disclosed.** Any code I did not write myself is identified in the
   contribution, is under a license compatible with Apache 2.0, and its origin is recorded in
   the file and in [docs/legal-notice.md](legal-notice.md).

6. I have the legal right to make this certification. If my employer has rights to intellectual
   property I create, I have received permission to contribute on their behalf, or my employer
   has waived those rights for this contribution.

---

## How to sign

Configure a real name and reachable email, then sign each commit:

```bash
git config user.name "Your Real Name"
git config user.email "you@example.com"
git commit -s -m "your message"
```

This appends:

```
Signed-off-by: Your Real Name <you@example.com>
```

Pseudonymous sign-offs are not accepted: the certification is only meaningful if it is
attributable.

## If you cannot certify a clause

Say so in the pull request rather than signing off anyway. Depending on the clause there is
often still a way to contribute — a different area of the codebase, a bug report with a
reproducer instead of a patch, or a design proposal someone else implements. A contribution we
cannot accept is a much smaller problem than a contribution that compromises the project's
provenance.
