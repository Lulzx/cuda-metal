"""Decide whether a Metal solve agrees with its CPU reference.

Prints `<rel-objective-diff>|<verdict>`.

Gated: model status class, primal objective, primal/dual infeasibility, relative
duality gap, and that every reported value is finite. Iteration count is not
gated -- the two builds legitimately take different paths to the same answer.
"""
import math
import sys

(cs, co, cpi, cdi, cg, gs, go, gpi, gdi, gg) = sys.argv[1:11]

# PDLP's own defaults here are 1e-4 on the gap and each residual, so two runs
# stop at different points inside tolerance; allow an order of magnitude for
# where they land.
TOL_OBJ = 1e-3
TOL_RESIDUAL = 1e-3
# How much worse than the CPU run the Metal run may be on any residual. This is
# the only residual test that applies to an unconverged run: a solve stopped at
# the iteration limit legitimately has a large gap (stair ends around 1e-2 on
# both builds), so an absolute ceiling there would fail a run that agrees
# perfectly well with its reference.
PARITY = 10.0


def num(text):
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


rel = "n/a"
cov, gov = num(co), num(go)
if cov is not None and gov is not None:
    rel = "%.1e" % (abs(cov - gov) / max(abs(cov), 1e-12))

values = {"objective": (cov, gov)}
for name, c, g in (("p.infeas", cpi, gpi), ("d.infeas", cdi, gdi), ("gap", cg, gg)):
    values[name] = (num(c), num(g))

verdict = "ok"
if any(v is None for pair in values.values() for v in pair):
    verdict = "NON-FINITE"
elif cs != gs:
    verdict = "STATUS %s!=%s" % (cs, gs)
elif abs(cov - gov) / max(abs(cov), 1e-12) > TOL_OBJ:
    verdict = "OBJECTIVE"
else:
    converged = cs.startswith("Optimal")
    for name in ("p.infeas", "d.infeas", "gap"):
        c, g = values[name]
        # Parity with the CPU run always applies, so a Metal run cannot pass by
        # being an order of magnitude less feasible than its reference. The
        # absolute ceiling applies only when the solver claims convergence.
        if abs(g) > PARITY * max(abs(c), 1e-12):
            verdict = name.upper()
            break
        if converged and abs(g) > TOL_RESIDUAL:
            verdict = name.upper()
            break

print("%s|%s" % (rel, verdict))
