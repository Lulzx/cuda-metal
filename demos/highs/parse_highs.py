"""Extract a HiGHS PDLP run in the same six-field form as parse_plc.py.

Fields: status objective primal_infeas dual_infeas gap iterations
"""
import re
import sys

text = open(sys.argv[1], errors="replace").read()


def grab(pattern, default="n/a"):
    match = re.search(pattern, text, re.M)
    return match.group(1).strip() if match else default


print(
    # Normalize multiword statuses such as "Iteration limit" to the same
    # status class used by the standalone cuPDLP-C parser.
    grab(r"^Model status\s*:\s*(\S+)"),
    grab(r"^Objective value\s*:\s*(\S+)"),
    grab(r"^\s*Primal infeas \(abs/rel\):\s*\S+\s*/\s*(\S+)"),
    grab(r"^\s*Dual infeas \(abs/rel\):\s*\S+\s*/\s*(\S+)"),
    grab(r"^\s*Duality gap \(abs/rel\):\s*\S+\s*/\s*(\S+)"),
    grab(r"^PDLP\s+iterations:\s*(\d+)"),
)
