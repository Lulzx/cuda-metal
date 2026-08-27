#!/usr/bin/env python3
"""Download and expand Mittelmann LPfeas instances into a local cache.

Everything lands in --dir (default demos/highs/lpfeas/lps, gitignored) as
<name>.mps. Files are large -- the phase-1 set alone is a few GB expanded -- so
an instance already present is left alone unless --force.

Some archives on plato.asu.edu and Meszaros' site expand not to an MPS file but
to the compressed form NETLIB's `emps` utility reads. Those are marked emps=1 in
instances.tsv; emps itself is built once from netlib.org/lp/data/emps.c.
"""
import argparse, bz2, gzip, os, shutil, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    "plato":    "https://plato.asu.edu/ftp/",
    "miplib":   "https://miplib2010.zib.de/download/",
    "meszaros": "https://www.sztaki.hu/~meszaros/public_ftp/lptestset/",
}
EMPS_SRC = "https://www.netlib.org/lp/data/emps.c"


def instances():
    out = []
    for line in (HERE / "instances.tsv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, src, url, emps, rows, cols, nnz, phase = line.split()
        out.append(dict(name=name, source=src, url=url, emps=int(emps),
                        rows=int(rows), cols=int(cols), nnz=int(nnz),
                        phase=int(phase)))
    return out


def build_emps(workdir: Path) -> Path:
    exe = workdir / "emps"
    if exe.exists():
        return exe
    src = workdir / "emps.c"
    if not src.exists():
        urllib.request.urlretrieve(EMPS_SRC, src)
    # emps.c is 1980s K&R C; clang needs to be told not to reject it outright.
    subprocess.run(["cc", "-w", "-std=gnu89", "-Wno-implicit-function-declaration",
                    "-o", str(exe), str(src)], check=True)
    return exe


def download(url: str, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dest)


def decompress(archive: Path, dest: Path):
    opener = bz2.open if archive.suffix == ".bz2" else gzip.open
    with opener(archive, "rb") as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def fetch_one(inst, lp_dir: Path, cache: Path, emps_exe, force=False):
    mps = lp_dir / f"{inst['name']}.mps"
    if mps.exists() and not force:
        return mps, "cached"
    url = SOURCES[inst["source"]] + inst["url"]
    archive = cache / Path(inst["url"]).name
    if not archive.exists():
        download(url, archive)
    raw = cache / f"{inst['name']}.raw"
    decompress(archive, raw)
    if inst["emps"]:
        with open(mps, "wb") as out:
            subprocess.run([str(emps_exe), str(raw)], stdout=out, check=True)
        raw.unlink()
    else:
        raw.rename(mps)
    return mps, "fetched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "lps"))
    ap.add_argument("--phase", type=int, action="append",
                    help="fetch only these phases (repeatable); default 1")
    ap.add_argument("--only", action="append", help="fetch only these names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep the .bz2/.gz downloads (they are ~1/5 the size)")
    a = ap.parse_args()

    lp_dir = Path(a.dir); lp_dir.mkdir(parents=True, exist_ok=True)
    cache = lp_dir / ".cache"; cache.mkdir(exist_ok=True)
    want = instances()
    if a.only:
        want = [i for i in want if i["name"] in a.only]
    else:
        want = [i for i in want if i["phase"] in (a.phase or [1])]
    if not want:
        sys.exit("no instances selected")

    emps_exe = None
    if any(i["emps"] for i in want):
        emps_exe = build_emps(cache)

    fails = 0
    for i in want:
        try:
            mps, how = fetch_one(i, lp_dir, cache, emps_exe, a.force)
            print(f"{i['name']:<18} {how:<8} {mps.stat().st_size/1e6:8.1f} MB", flush=True)
        except Exception as e:
            fails += 1
            print(f"{i['name']:<18} FAIL     {type(e).__name__}: {e}", flush=True)
    if not a.keep_archives:
        shutil.rmtree(cache / "x", ignore_errors=True)
        for p in cache.glob("*.bz2"): p.unlink()
        for p in cache.glob("*.gz"): p.unlink()
    print(f"\n{len(want)-fails}/{len(want)} instances available in {lp_dir}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
