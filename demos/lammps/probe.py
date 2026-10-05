"""Compile the header reproducer with the configured Kokkos compiler flags."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main():
    build = Path(sys.argv[1]).resolve()
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/impl/Kokkos_Core.cpp"))
    command = entry.get("arguments") or shlex.split(entry["command"])
    source = str(Path(__file__).with_name("kokkos_header_probe.cpp").resolve())
    with tempfile.TemporaryDirectory(prefix="cumetal-kokkos-probe-") as scratch:
        output = str(Path(scratch) / "probe.o")
        args = []
        index = 0
        while index < len(command):
            arg = command[index]
            if arg == "-o":
                args.extend(["-o", output])
                index += 2
                continue
            args.append(source if arg == entry["file"] else arg)
            index += 1
        args.append("-ferror-limit=0")
        print("Header probe:", shlex.join(args), flush=True)
        return subprocess.run(args, cwd=entry["directory"]).returncode


if __name__ == "__main__":
    sys.exit(main())
