"""python -m clew_ml.train — train SpeedNet on IO-VNBD (drive-level split)."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .data import load_processed, split_of
from .features import CHANNELS, WINDOW, make_dataset
from .model import SpeedNet, n_params

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
STILL = 0.3  # m/s


def loss_fn(o: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    speed, logvar, still = o[:, 0], o[:, 1], o[:, 2]
    nll = 0.5 * (logvar + (y - speed) ** 2 / logvar.exp())
    bce = torch.nn.functional.binary_cross_entropy_with_logits(still, (y < STILL).float())
    return nll.mean() + 0.3 * bce


def main(epochs: int = 20, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    drives = load_processed()
    tr = [d for d in drives if split_of(d.name) == "train"]
    va = [d for d in drives if split_of(d.name) == "val"]
    Xtr, ytr = make_dataset(tr, stride=3)
    Xva, yva = make_dataset(va, stride=10)
    mean = torch.tensor(Xtr.mean((0, 2)))
    std = torch.tensor(Xtr.std((0, 2)) + 1e-6)
    model = SpeedNet(mean, std)
    print(f"train {len(Xtr)} windows ({len(tr)} segs), val {len(Xva)} ({len(va)} segs), params {n_params(model)}")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(Xtr) // 256 + 1))
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        t0 = time.time()
        for i in range(0, len(perm), 256):
            b = perm[i : i + 256]
            xb = Xtr_t[b]
            # augmentation: ±10 % per-channel gain (calibration error) + sensor noise
            xb = xb * (1 + 0.1 * torch.randn(len(b), xb.shape[1], 1)) + 0.05 * std[None, :, None] * torch.randn_like(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), ytr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            o = model(Xva_t)
            rmse = float(((o[:, 0] - yva_t) ** 2).mean().sqrt())
            vl = float(loss_fn(o, yva_t))
        print(f"epoch {ep + 1:2d}  val loss {vl:.3f}  val speed RMSE {rmse:.2f} m/s  ({time.time() - t0:.0f}s)")
        if rmse < best:
            best, best_state = rmse, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # calibrate the predicted variance on validation data: s = mean(err² / var)
    model.eval()
    with torch.no_grad():
        o = model(Xva_t)
        scale = float(((o[:, 0] - yva_t) ** 2 / o[:, 1].exp()).mean())
        model.var_scale.fill_(scale)
        o = model(Xva_t)
        z = (o[:, 0] - yva_t) / o[:, 1].exp().sqrt()
        within = float((z.abs() < 1).float().mean())
    print(f"variance scale {scale:.2f}; after calibration {within:.0%} of val errors fall within ±1σ (ideal ≈68%)")
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "speednet.pt"))
    with open(os.path.join(OUT, "speednet.json"), "w") as f:
        json.dump({"channels": CHANNELS, "window": WINDOW, "hz": 10, "best_val_rmse": best, "var_scale": scale, "val_within_1sigma": within}, f, indent=2)
    print("saved artifacts/speednet.pt")


if __name__ == "__main__":
    main()
