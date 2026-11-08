#!/usr/bin/env python3
"""Score COLMAP's GPU PatchMatch depth maps and fused cloud against exact truth.

COLMAP has no CPU PatchMatch, so there is no second build of the solver to
compare with. The reference is the scene itself: make_scene.py ray-casts every
view, so each pixel's true depth is known, and every fused 3-D point's distance
to the true surface can be computed in closed form.

PatchMatch is randomized and leaves pixels it cannot verify empty (depth 0), so
the gate measures what a correct solver must achieve rather than bit-equality:

  per view, over pixels COLMAP kept:
    coverage   fraction of the image with a depth          (>= --min-coverage)
    within1    fraction of kept depths within 1% of truth  (>= --min-within1)
    median     median relative depth error                 (<= --max-median)
  fused cloud (stereo_fusion):
    points     number of fused points                      (>= --min-points)
    on_surface fraction within 1 cm of the true surface    (>= --min-on-surface)

The defaults sit below the recorded CuMetal run (2026-11-09: coverage
0.81-0.96 and within1 0.96-0.98 per view, median 0.05-0.09%, 68,996 fused
points, 98.0% on the surface) with margin for PatchMatch's randomness. A miscompiled run measured
on the same scene scored within1 0.5% and median 72%.

A wrong texture fetch, a broken RNG or a mis-lowered cost kernel does not
degrade these numbers gracefully: the cost volume stops discriminating and the
depths collapse to noise, which fails every threshold by a wide margin.

Usage: gate.py --scene DIR --workspace DIR [--kind geometric|photometric]
               [--figure out.png] [--json out.json] [thresholds...]
Exit 0 only if every view and the fused cloud pass.
"""
import argparse
import glob
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_scene as scene  # noqa: E402  (same geometry constants)


def read_colmap_array(path):
    """COLMAP's dense .bin: ASCII 'W&H&C&' header, then float32, column-major."""
    with open(path, "rb") as fh:
        header = b""
        while header.count(b"&") < 3:
            byte = fh.read(1)
            if not byte:
                raise ValueError(f"{path}: truncated header")
            header += byte
        width, height, channels = (int(x) for x in header.split(b"&")[:3])
        data = np.fromfile(fh, dtype=np.float32)
    if data.size != width * height * channels:
        raise ValueError(f"{path}: {data.size} floats, header says {width}x{height}x{channels}")
    arr = data.reshape((width, height, channels), order="F")
    return np.transpose(arr, (1, 0, 2)).squeeze()


def read_ply_xyz(path):
    with open(path, "rb") as fh:
        props, count, fmt = [], 0, None
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                count = int(line.split()[2])
            elif line.startswith("property"):
                props.append(line.split())
            elif line == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
        sizes = {"float": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "int": "i4"}
        dtype = np.dtype([(p[2], "<" + sizes[p[1]]) for p in props])
        data = np.fromfile(fh, dtype=dtype, count=count)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def surface_distance(points):
    """Unsigned distance from each point to the nearest scene surface."""
    d_ground = np.abs(points[:, 2])
    d_ground[points[:, 1] > scene.WALL_Y] = np.inf
    d_wall = np.abs(points[:, 1] - scene.WALL_Y)
    d_wall[points[:, 2] < 0] = np.inf
    d_sphere = np.abs(np.linalg.norm(points - scene.SPHERE_C, axis=1) - scene.SPHERE_R)
    u, v = scene.panel_axes()
    rel = points - scene.PANEL_C
    pu = np.clip(rel @ u, -scene.PANEL_HALF, scene.PANEL_HALF)
    pv = np.clip(rel @ v, -scene.PANEL_HALF, scene.PANEL_HALF)
    closest = scene.PANEL_C + pu[:, None] * u + pv[:, None] * v
    d_panel = np.linalg.norm(points - closest, axis=1)
    return np.min(np.stack([d_ground, d_wall, d_sphere, d_panel]), axis=0)


def turbo(x):
    """Polynomial approximation of the Turbo colormap; x in [0,1] -> uint8 RGB."""
    x = np.clip(x, 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.stack([r, g, b], axis=-1), 0, 1) * 255).astype(np.uint8)


def write_png_rgb(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        fh.write(chunk(b"IEND", b""))


def read_png_gray(path):
    """Decode the 8-bit greyscale PNGs make_scene.py writes (filter 0 only)."""
    with open(path, "rb") as fh:
        data = fh.read()
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    rows = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, w + 1)
    return rows[:, 1:]


def figure(path, image, truth, depth):
    """image | true depth | CuMetal depth | relative error (0..2%), one row."""
    lo, hi = np.percentile(truth, 1), np.percentile(truth, 99)
    norm = lambda d: (d - lo) / (hi - lo)  # noqa: E731
    gray = np.repeat(image[:, :, None], 3, axis=2)
    t_rgb = turbo(norm(truth))
    d_rgb = turbo(norm(depth))
    d_rgb[depth <= 0] = 0
    err = np.where(depth > 0, np.abs(depth - truth) / truth, 0.0)
    e_rgb = turbo(err / 0.02)
    e_rgb[depth <= 0] = 0
    gap = np.full((truth.shape[0], 8, 3), 255, np.uint8)
    write_png_rgb(path, np.concatenate([gray, gap, t_rgb, gap, d_rgb, gap, e_rgb], axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--kind", default="geometric", choices=["geometric", "photometric"])
    ap.add_argument("--min-coverage", type=float, default=0.70)
    ap.add_argument("--min-within1", type=float, default=0.75)
    ap.add_argument("--max-median", type=float, default=0.002)
    ap.add_argument("--min-points", type=int, default=50000)
    ap.add_argument("--min-on-surface", type=float, default=0.95)
    ap.add_argument("--figure")
    ap.add_argument("--json")
    args = ap.parse_args()

    truths = sorted(glob.glob(os.path.join(args.scene, "truth", "*.npy")))
    if not truths:
        sys.exit(f"gate: no truth depth under {args.scene}/truth")
    rows, ok = [], True
    print(f"{'view':10} {'coverage':>9} {'within1%':>9} {'median':>9}")
    for tpath in truths:
        stem = os.path.splitext(os.path.basename(tpath))[0]
        truth = np.load(tpath).astype(np.float64)
        dpath = os.path.join(args.workspace, "stereo", "depth_maps", f"{stem}.png.{args.kind}.bin")
        if not os.path.exists(dpath):
            print(f"{stem:10} MISSING {dpath}")
            ok = False
            continue
        depth = read_colmap_array(dpath).astype(np.float64)
        if depth.shape != truth.shape:
            print(f"{stem:10} shape {depth.shape} != truth {truth.shape}")
            ok = False
            continue
        valid = depth > 0
        coverage = float(valid.mean())
        rel = np.abs(depth[valid] - truth[valid]) / truth[valid]
        within1 = float((rel < 0.01).mean()) if rel.size else 0.0
        median = float(np.median(rel)) if rel.size else float("inf")
        good = coverage >= args.min_coverage and within1 >= args.min_within1 and median <= args.max_median
        ok &= good
        rows.append({"view": stem, "coverage": coverage, "within1": within1, "median_rel": median, "pass": good})
        print(f"{stem:10} {coverage:9.3f} {within1:9.3f} {median:9.5f}{'' if good else '   FAIL'}")
        if args.figure and stem == os.path.splitext(os.path.basename(truths[len(truths) // 2]))[0]:
            image = read_png_gray(os.path.join(args.scene, "images", f"{stem}.png"))
            figure(args.figure, image, truth, depth)

    fused = {}
    ply = os.path.join(args.workspace, "fused.ply")
    if os.path.exists(ply):
        pts = read_ply_xyz(ply)
        dist = surface_distance(pts) if len(pts) else np.array([np.inf])
        on_surface = float((dist < 0.01).mean())
        fused = {"points": int(len(pts)), "on_surface_1cm": on_surface,
                 "median_distance_m": float(np.median(dist))}
        good = len(pts) >= args.min_points and on_surface >= args.min_on_surface
        ok &= good
        print(f"fused.ply  {len(pts)} points, {on_surface:.3f} within 1 cm of the surface, "
              f"median {fused['median_distance_m'] * 1000:.2f} mm{'' if good else '   FAIL'}")
    else:
        print(f"fused.ply  MISSING {ply}")
        ok = False

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"kind": args.kind, "views": rows, "fused": fused, "pass": ok,
                       "thresholds": {k: getattr(args, k) for k in
                                      ("min_coverage", "min_within1", "max_median", "min_points", "min_on_surface")}},
                      fh, indent=2)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
