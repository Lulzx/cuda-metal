#!/usr/bin/env python3
"""Render a synthetic multi-view scene with exact depth, as a COLMAP sparse model.

COLMAP has no CPU PatchMatch, so a CuMetal run cannot be checked against a CPU
build of the same solver. It is checked against the truth instead: every image
here is ray-cast from an analytic scene, so the depth of every pixel is known
exactly, and so are the camera poses. PatchMatch has to recover that depth from
the images alone.

The scene is a ground plane, a back wall, a tilted panel and a sphere. Each
surface carries a procedural albedo defined in world coordinates (value noise
over three octaves), so a surface point has the same colour in every view, as
PatchMatch's photo-consistency assumes. Each image is supersampled 3x3 to keep
the high-frequency texture from aliasing differently in different views.

Outputs (under --out):
  images/view_XX.png      8-bit greyscale renders
  sparse/{cameras,images,points3D}.txt
                          exact intrinsics and poses, plus surface samples with
                          their observations. COLMAP sets each image's depth
                          range and picks its source views from these points,
                          exactly as it would after SfM.
  truth/view_XX.npy       float32 ground-truth depth along the camera z axis,
                          the quantity COLMAP's depth maps store.

Usage: make_scene.py --out DIR [--views N] [--width W] [--height H] [--seed S]
"""
import argparse
import os
import struct
import sys
import zlib

import numpy as np

# Scene geometry (world frame: z up, cameras on the -y side looking toward +y).
WALL_Y = 1.6
SPHERE_C = np.array([0.15, 0.25, 0.55])
SPHERE_R = 0.55
# Tilted panel: a bounded quad leaning against nothing, between sphere and wall.
PANEL_C = np.array([-0.95, 0.7, 0.55])
PANEL_N = np.array([0.55, -0.6, 0.25])
PANEL_N = PANEL_N / np.linalg.norm(PANEL_N)
PANEL_HALF = 0.45
LOOK_AT = np.array([0.0, 0.5, 0.45])


def panel_axes():
    up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, PANEL_N)
    u /= np.linalg.norm(u)
    v = np.cross(PANEL_N, u)
    return u, v


def lattice_noise(p, freq, seed):
    """Trilinear value noise at points p (N,3), lattice spacing 1/freq."""
    q = p * freq
    i = np.floor(q).astype(np.int64)
    f = q - i
    f = f * f * (3.0 - 2.0 * f)  # smoothstep: continuous gradient across cells

    def h(ix, iy, iz):
        x = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
        x = (x ^ (x >> 13)) * 1274126177
        x = x ^ (x >> 16)
        return (x & 0xFFFF).astype(np.float64) / 65535.0

    out = 0.0
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                out = out + wx * wy * wz * h(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
    return out


def albedo(p, seed):
    a = (0.55 * lattice_noise(p, 6.0, seed) +
         0.30 * lattice_noise(p, 15.0, seed + 1) +
         0.15 * lattice_noise(p, 37.0, seed + 2))
    return 0.12 + 0.8 * a


def look_at(center, target):
    """World-to-camera (R, t) for a camera at center looking at target, image y down."""
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ center


def intersect(origins, dirs):
    """Nearest hit of rays (N,3) against the scene. Returns t, normal (N,3)."""
    n = dirs.shape[0]
    t_best = np.full(n, np.inf)
    normal = np.zeros((n, 3))

    def take(t, nrm):
        better = (t > 1e-6) & (t < t_best)
        t_best[better] = t[better]
        normal[better] = nrm[better] if nrm.ndim == 2 else nrm

    with np.errstate(divide="ignore", invalid="ignore"):
        # Ground z = 0, in front of the wall.
        t = -origins[:, 2] / dirs[:, 2]
        hit_y = origins[:, 1] + t * dirs[:, 1]
        take(np.where(hit_y < WALL_Y, t, np.inf), np.array([0.0, 0.0, 1.0]))
        # Back wall y = WALL_Y, above the ground.
        t = (WALL_Y - origins[:, 1]) / dirs[:, 1]
        hit_z = origins[:, 2] + t * dirs[:, 2]
        take(np.where(hit_z >= 0.0, t, np.inf), np.array([0.0, -1.0, 0.0]))
        # Tilted square panel.
        u, v = panel_axes()
        denom = dirs @ PANEL_N
        t = ((PANEL_C - origins) @ PANEL_N) / denom
        hit = origins + t[:, None] * dirs - PANEL_C
        inside = (np.abs(hit @ u) <= PANEL_HALF) & (np.abs(hit @ v) <= PANEL_HALF)
        take(np.where(inside, t, np.inf), PANEL_N)
        # Sphere.
        oc = origins - SPHERE_C
        b = np.sum(oc * dirs, axis=1)
        c = np.sum(oc * oc, axis=1) - SPHERE_R ** 2
        disc = b * b - c
        t = np.where(disc >= 0.0, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
        hit = origins + t[:, None] * dirs
        take(t, (hit - SPHERE_C) / SPHERE_R)
    return t_best, normal


def shade(points, normals, seed):
    # View-independent (Lambertian) so a point looks the same from every camera.
    light = np.array([-0.35, -0.5, 0.8])
    light /= np.linalg.norm(light)
    lambert = 0.45 + 0.55 * np.clip(normals @ light, 0.0, 1.0)
    return albedo(points, seed) * lambert


def render(R, t, K, width, height, seed, ss):
    center = -R.T @ t
    fx, fy, cx, cy = K
    # COLMAP's pixel (0,0) covers [0,1)x[0,1); its centre is (0.5, 0.5).
    offs = (np.arange(ss) + 0.5) / ss
    xs = (np.arange(width)[:, None] + offs[None, :]).reshape(-1)
    ys = (np.arange(height)[:, None] + offs[None, :]).reshape(-1)
    gx, gy = np.meshgrid(xs, ys)
    d_cam = np.stack([(gx - cx) / fx, (gy - cy) / fy, np.ones_like(gx)], axis=-1).reshape(-1, 3)
    d_world = d_cam @ R  # R^T d for each row
    d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
    origins = np.broadcast_to(center, d_world.shape)
    tt, nrm = intersect(origins, d_world)
    if not np.all(np.isfinite(tt)):
        sys.exit("make_scene: a camera ray missed the scene; adjust the rig")
    pts = origins + tt[:, None] * d_world
    color = shade(pts, nrm, seed).reshape(height * ss, width * ss)
    color = color.reshape(height, ss, width, ss).mean(axis=(1, 3))

    # Truth at the pixel centre (not the supersample mean, which would blur
    # depth across occlusion boundaries).
    px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    d_cam = np.stack([(px - cx) / fx, (py - cy) / fy, np.ones_like(px)], axis=-1).reshape(-1, 3)
    d_world = d_cam @ R
    scale = np.linalg.norm(d_world, axis=1)
    d_world /= scale[:, None]
    tt, _ = intersect(np.broadcast_to(center, d_world.shape), d_world)
    depth = (tt / scale).reshape(height, width)  # ray length -> camera-z depth
    return color, depth


def write_png_gray(path, img):
    data = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    h, w = data.shape
    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(h))

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def rot_to_quat(R):
    # Hamilton quaternion (w, x, y, z), the convention of COLMAP's images.txt.
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def sample_surfaces(rng, n):
    """Points spread over every surface, for the sparse model."""
    pts = []
    k = n // 4
    g = rng.uniform([-2.0, -0.8, 0.0], [2.0, WALL_Y, 0.0], size=(k, 3))
    pts.append(g)
    w = rng.uniform([-2.0, WALL_Y, 0.0], [2.0, WALL_Y, 2.0], size=(k, 3))
    pts.append(w)
    u, v = panel_axes()
    a = rng.uniform(-PANEL_HALF, PANEL_HALF, size=(k, 2))
    pts.append(PANEL_C + a[:, :1] * u + a[:, 1:] * v)
    s = rng.normal(size=(k, 3))
    pts.append(SPHERE_C + SPHERE_R * s / np.linalg.norm(s, axis=1, keepdims=True))
    return np.concatenate(pts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=10)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--supersample", type=int, default=3)
    args = ap.parse_args()

    for sub in ("images", "sparse", "truth"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    f = 0.9 * args.width
    K = (f, f, args.width / 2.0, args.height / 2.0)

    cams = []
    for i in range(args.views):
        # A 70-degree arc at two heights, the way a person walks past an object.
        ang = np.radians(-35.0 + 70.0 * i / max(1, args.views - 1))
        height = 1.15 if i % 2 == 0 else 1.45
        radius = 2.9
        center = np.array([radius * np.sin(ang), LOOK_AT[1] - radius * np.cos(ang), height])
        R, t = look_at(center, LOOK_AT)
        color, depth = render(R, t, K, args.width, args.height, args.seed, args.supersample)
        name = f"view_{i:02d}.png"
        write_png_gray(os.path.join(args.out, "images", name), color)
        np.save(os.path.join(args.out, "truth", f"view_{i:02d}.npy"), depth.astype(np.float32))
        cams.append((name, R, t, depth))
        print(f"  rendered {name}: depth {depth.min():.3f}..{depth.max():.3f}")

    # Sparse points with their exact observations. A point is observed by a
    # view only if it is the visible surface there (depth agrees with truth).
    X = sample_surfaces(rng, 4000)
    obs = {i: [] for i in range(len(cams))}
    tracks = []
    for pid, x in enumerate(X):
        track = []
        for ci, (_, R, t, depth) in enumerate(cams):
            pc = R @ x + t
            if pc[2] <= 0:
                continue
            u = K[0] * pc[0] / pc[2] + K[2]
            v = K[1] * pc[1] / pc[2] + K[3]
            if not (0 <= u < args.width and 0 <= v < args.height):
                continue
            if abs(depth[int(v), int(u)] - pc[2]) > 0.01 * pc[2]:
                continue  # occluded
            track.append((ci, len(obs[ci])))
            obs[ci].append((u, v, pid + 1))
        tracks.append(track)

    sp = os.path.join(args.out, "sparse")
    with open(os.path.join(sp, "cameras.txt"), "w") as fh:
        fh.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        fh.write(f"1 PINHOLE {args.width} {args.height} {K[0]:.10g} {K[1]:.10g} {K[2]:.10g} {K[3]:.10g}\n")
    with open(os.path.join(sp, "images.txt"), "w") as fh:
        fh.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for ci, (name, R, t, _) in enumerate(cams):
            q = rot_to_quat(R)
            fh.write(f"{ci + 1} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                     f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} 1 {name}\n")
            fh.write(" ".join(f"{u:.4f} {v:.4f} {p}" for u, v, p in obs[ci]) + "\n")
    kept = 0
    with open(os.path.join(sp, "points3D.txt"), "w") as fh:
        fh.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, (x, track) in enumerate(zip(X, tracks)):
            # A point seen once still owns its 2D entry in images.txt, so it
            # is written too; a point no view sees is referenced by nothing.
            if not track:
                continue
            kept += len(track) >= 2
            fh.write(f"{pid + 1} {x[0]:.9g} {x[1]:.9g} {x[2]:.9g} 128 128 128 0 " +
                     " ".join(f"{ci + 1} {idx}" for ci, idx in track) + "\n")
    print(f"  sparse model: {len(cams)} views, {len(X)} points ({kept} seen by >= 2 views)")


if __name__ == "__main__":
    main()
