"""Extract a plc run's solution summary as one whitespace-separated line.

Fields: status objective primal_infeas dual_infeas gap iterations

PDLP correctness is not the objective alone -- a run can report a plausible
objective while its residuals say the point is not feasible -- so the demo gates
on the KKT residuals and the relative duality gap too.
"""
import re
import sys

text = open(sys.argv[1], errors="replace").read()


def grab(pattern, default="n/a"):
    m = re.search(pattern, text, re.M)
    return m.group(1).strip() if m else default


print(
    # "Optimal current solution." vs "Optimal average solution." differ only in
    # which iterate PDLP accepted, so the status class is the first word.
    grab(r"^Solving information: *(\S+)"),
    grab(r"^ *Primal objective: *(\S+)"),
    # "abs / rel" pairs; the relative figure is the scale-free one.
    grab(r"^ *Primal infeas \(abs/rel\): *\S+ */ *(\S+)"),
    grab(r"^ *Dual infeas \(abs/rel\): *\S+ */ *(\S+)"),
    grab(r"^ *Duality gap \(abs/rel\): *\S+ */ *(\S+)"),
    grab(r"^ *Number of iterations: *(\S+)"),
)
