"""Extract `status objective iterations` from a plc run log, as one line."""
import re
import sys

text = open(sys.argv[1], errors="replace").read()


def grab(pattern, default="n/a"):
    m = re.search(pattern, text, re.M)
    return m.group(1).strip() if m else default


# "Optimal current solution." / "Optimal average solution." differ only in which
# iterate PDLP accepted, so the status class is the first word.
status = grab(r"^Solving information: *(\S+)")
print(status,
      grab(r"^ *Primal objective: *(\S+)"),
      grab(r"^ *Number of iterations: *(\S+)"))
