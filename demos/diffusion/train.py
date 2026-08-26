#!/usr/bin/env python3
"""Train a tiny MNIST DDPM and export its weights for the CUDA sampler.

The network is deliberately small (~290k params, no attention) so that the
whole sampler fits in one readable .cu file.  Layer order here is the contract
main.cu implements -- change one and you must change the other.
"""
import argparse, gzip, os, struct, sys, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_STEPS = 1000
TDIM = 64

MNIST_DIRS = [
    os.path.expanduser("~/work/mnist/data/MNIST/raw"),
    os.path.join(os.path.dirname(__file__), "data"),
]
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"


def load_mnist():
    for d in MNIST_DIRS:
        p = os.path.join(d, "train-images-idx3-ubyte")
        if os.path.exists(p):
            raw = open(p, "rb").read()
            break
        if os.path.exists(p + ".gz"):
            raw = gzip.decompress(open(p + ".gz", "rb").read())
            break
    else:
        d = MNIST_DIRS[-1]
        os.makedirs(d, exist_ok=True)
        print(f"downloading MNIST -> {d}")
        gz = os.path.join(d, "train-images-idx3-ubyte.gz")
        urllib.request.urlretrieve(MNIST_URL, gz)
        raw = gzip.decompress(open(gz, "rb").read())
    magic, n, h, w = struct.unpack(">IIII", raw[:16])
    assert magic == 2051, magic
    x = np.frombuffer(raw[16:16 + n * h * w], dtype=np.uint8).reshape(n, 1, h, w)
    return torch.from_numpy(x.copy()).float() / 127.5 - 1.0


class Block(nn.Module):
    """conv -> +t -> groupnorm -> silu -> conv -> groupnorm -> silu"""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.gn1 = nn.GroupNorm(cout // 8, cout)
        self.gn2 = nn.GroupNorm(cout // 8, cout)
        self.proj = nn.Linear(TDIM, cout)

    def forward(self, x, t):
        h = self.conv1(x) + self.proj(t)[:, :, None, None]
        h = F.silu(self.gn1(h))
        return F.silu(self.gn2(self.conv2(h)))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(TDIM, TDIM), nn.Linear(TDIM, TDIM)
        self.d1 = Block(1, 32)
        self.d2 = Block(32, 64)
        self.m = Block(64, 64)
        self.u2 = Block(128, 64)
        self.u1 = Block(96, 32)
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def temb(self, t):
        half = TDIM // 2
        f = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None, :]
        e = torch.cat([torch.sin(a), torch.cos(a)], dim=1)
        return self.fc2(F.silu(self.fc1(e)))

    def forward(self, x, t):
        e = self.temb(t)
        h1 = self.d1(x, e)                                   # 28
        h2 = self.d2(F.avg_pool2d(h1, 2), e)                 # 14
        hm = self.m(F.avg_pool2d(h2, 2), e)                  # 7
        u2 = self.u2(torch.cat([F.interpolate(hm, scale_factor=2), h2], 1), e)
        u1 = self.u1(torch.cat([F.interpolate(u2, scale_factor=2), h1], 1), e)
        return self.out(u1)


def export(model, path):
    """Flat weight file: magic, count, then [name[32], ndim, dims[4], data]."""
    tensors = [(k, v.detach().cpu().float().numpy()) for k, v in model.state_dict().items()]
    with open(path, "wb") as f:
        f.write(b"CMDF" + struct.pack("<ii", 1, len(tensors)))
        for name, a in tensors:
            nb = name.encode()
            assert len(nb) < 32, name
            dims = list(a.shape) + [1] * (4 - a.ndim)
            f.write(nb.ljust(32, b"\0") + struct.pack("<i", a.ndim) + struct.pack("<4i", *dims))
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
    print(f"wrote {path}: {len(tensors)} tensors, "
          f"{sum(a.size for _, a in tensors)} params")


def export_check(model, dev, out_dir):
    """A fixed input + PyTorch's eps output, so main.cu --check can self-verify."""
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(2, 1, 28, 28, generator=g)
    t = torch.tensor([17, 640])
    with torch.no_grad():
        ref = model(x.to(dev), t.to(dev)).cpu()
    x.numpy().astype("<f4").tofile(os.path.join(out_dir, "check_in.bin"))
    t.numpy().astype("<i4").tofile(os.path.join(out_dir, "check_t.bin"))
    ref.numpy().astype("<f4").tofile(os.path.join(out_dir, "check_ref.bin"))
    print(f"check tensors written (ref |eps| mean {ref.abs().mean():.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    data = load_mnist().to(dev)
    print(f"device={dev}  data={tuple(data.shape)}")

    model = UNet().to(dev)
    print(f"params={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    betas = torch.linspace(1e-4, 0.02, T_STEPS, device=dev)
    abar = torch.cumprod(1.0 - betas, 0)

    n = data.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = cnt = 0.0
        for i in range(0, n - args.batch + 1, args.batch):
            x0 = data[perm[i:i + args.batch]]
            t = torch.randint(0, T_STEPS, (x0.shape[0],), device=dev)
            noise = torch.randn_like(x0)
            a = abar[t][:, None, None, None]
            xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
            loss = F.mse_loss(model(xt, t), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); cnt += 1
        print(f"epoch {ep + 1}/{args.epochs}  loss {tot / cnt:.4f}", flush=True)

    model.eval()
    export(model, os.path.join(args.out, "model.bin"))
    export_check(model, dev, args.out)


if __name__ == "__main__":
    sys.exit(main())
