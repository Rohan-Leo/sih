"""
The two IMU-window models that sit next to SpeedNet. Both use the same
mount-invariant 10 s feature windows (features.py) and a TCN encoder.

HeadingNet   corrects the calibrated gyro yaw rate. Output [δ, log-variance],
             where δ ≈ true heading rate − calibrated yaw rate, both averaged
             over the whole 10 s window: the slowly varying yaw error that
             integrates into heading drift during an outage (mount-scale
             error, vibration-induced bias) and that a 3-number regression
             can't absorb. The filter adds δ to the gyro for the next second.

MotionNet    classifies what the vehicle is doing over the last second:
             stationary / cruising / accelerating / braking / turning.
             Drives the "driving state" readout and backs up ZUPT.

python -m clew_ml.imu_models heading|motion   → artifacts/{headingnet,motionnet}.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .calib import heading_rate_truth
from .data import Drive, load_processed, split_of
from .features import CHANNELS, WINDOW, causal_mean, drive_features, windows
from .nets import TCN, fit, n_params

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MOTION_CLASSES = ["stationary", "cruising", "accelerating", "braking", "turning"]
ACC_THR = 0.7  # m/s² over 1 s
TURN_THR = 0.1  # rad/s (≈ 5.7°/s) over 1 s


def heading_target(d: Drive, feats: np.ndarray) -> np.ndarray:
    """δ = true − calibrated yaw rate, both as trailing 10 s means (rad/s)."""
    truth = causal_mean(heading_rate_truth(d), WINDOW)
    return (truth - causal_mean(feats[:, CHANNELS.index("yaw_rate")].astype(float), WINDOW)).astype(np.float32)


def motion_labels(d: Drive) -> np.ndarray:
    v = d.truth_speed
    dv = np.r_[np.zeros(10), (v[10:] - v[:-10])]  # m/s over the last 1 s
    hr = causal_mean(heading_rate_truth(d), 10)
    vmax = np.maximum.reduce([np.r_[np.full(k, v[0]), v[: len(v) - k]] for k in range(10)])
    lab = np.full(len(v), 1, np.int64)  # cruising
    lab[dv > ACC_THR] = 2
    lab[dv < -ACC_THR] = 3
    lab[(np.abs(hr) > TURN_THR) & (v > 2)] = 4
    lab[vmax < 0.5] = 0
    return lab


def dataset(drives: list[Drive], task: str, stride: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for d in drives:
        f, _ = drive_features(d)
        idx = np.arange(WINDOW - 1, d.T, stride)
        X.append(windows(f, idx))
        y.append((heading_target(d, f) if task == "heading" else motion_labels(d))[idx])
    return np.concatenate(X), np.concatenate(y)


def heading_loss(o: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu, lv = o[:, 0] * 0.1, o[:, 1].clamp(-12, 2)
    return (0.5 * (lv + (y - mu) ** 2 / lv.exp())).mean()


def heading_pred(o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(δ rad/s, variance) — the head predicts δ/0.1 for better conditioning."""
    return o[:, 0] * 0.1, o[:, 1].clamp(-12, 2).exp()


def macro_f1(pred: np.ndarray, y: np.ndarray, k: int) -> tuple[float, list[float]]:
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        p = tp / max(np.sum(pred == c), 1)
        r = tp / max(np.sum(y == c), 1)
        f1s.append(0.0 if tp == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)), f1s


def rule_motion(X: np.ndarray, still_thr: float) -> np.ndarray:
    """Hand-tuned baseline on the same calibrated signals (last 1 s of each window)."""
    ci = {c: i for i, c in enumerate(CHANNELS)}
    last = X[:, :, -10:]
    yaw = last[:, ci["yaw_rate"]].mean(1)
    along = last[:, ci["a_long"]].mean(1)
    jitter = X[:, ci["acc_mag"], -20:].std(1)
    out = np.full(len(X), 1)
    out[along > ACC_THR] = 2
    out[along < -ACC_THR] = 3
    out[np.abs(yaw) > TURN_THR] = 4
    out[(jitter < still_thr) & (np.abs(yaw) < 0.02)] = 0
    return out


def augment_fn(std: torch.Tensor):
    def aug(xb: torch.Tensor) -> torch.Tensor:
        return xb * (1 + 0.1 * torch.randn(len(xb), xb.shape[1], 1)) + 0.05 * std[None, :, None] * torch.randn_like(xb)

    return aug


def train(task: str, epochs: int = 15, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    drives = load_processed()
    by = {s: [d for d in drives if split_of(d.name) == s] for s in ("train", "val", "test")}
    Xtr, ytr = dataset(by["train"], task, 3)
    Xva, yva = dataset(by["val"], task, 10)
    Xte, yte = dataset(by["test"], task, 10)
    mean, std = torch.tensor(Xtr.mean((0, 2))), torch.tensor(Xtr.std((0, 2)) + 1e-6)
    tt = lambda a: torch.tensor(a)
    report: dict = {"task": task, "train_windows": len(Xtr)}
    if task == "heading":
        model = TCN(len(CHANNELS), 2, mean, std)
        print(f"HeadingNet: {len(Xtr)} train windows, {n_params(model)} params")
        score = lambda o, y: ((heading_pred(o)[0] - y) ** 2).mean().sqrt()
        fit(model, tt(Xtr), tt(ytr), tt(Xva), tt(yva), heading_loss, score, epochs, augment=augment_fn(std))
        with torch.no_grad():
            # calibrate predicted variance on validation data
            mu, var = heading_pred(model(tt(Xva)))
            s = float(((mu - tt(yva)) ** 2 / var).mean())
            model.register_buffer("var_scale", torch.tensor([s]))
            for name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
                mu, var = heading_pred(model(tt(X)))
                mu = mu.numpy()
                report[name] = {
                    "yaw_rmse_calibrated_gyro": float(np.sqrt(np.mean(y**2))),
                    "yaw_rmse_headingnet": float(np.sqrt(np.mean((y - mu) ** 2))),
                    "within_1sigma": float(np.mean(np.abs(y - mu) < np.sqrt(var.numpy() * s))),
                }
        report["var_scale"] = s
        name = "headingnet"
    else:
        # windows are chosen with the stationary-jitter threshold learned on train
        ci = CHANNELS.index("acc_mag")
        jit = Xtr[:, ci, -20:].std(1)
        still_thr = float(np.quantile(jit[ytr == 0], 0.9))
        counts = np.bincount(ytr, minlength=len(MOTION_CLASSES))
        w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
        w = w / w.mean()
        model = TCN(len(CHANNELS), len(MOTION_CLASSES), mean, std)
        print(f"MotionNet: {len(Xtr)} train windows, class counts {counts.tolist()}, {n_params(model)} params")
        loss = lambda o, y: torch.nn.functional.cross_entropy(o, y, weight=w)
        score = lambda o, y: 1 - macro_f1(o.argmax(1).numpy(), y.numpy(), len(MOTION_CLASSES))[0]
        fit(model, tt(Xtr), tt(ytr), tt(Xva), tt(yva), loss, score, epochs, augment=augment_fn(std))
        with torch.no_grad():
            for name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
                p = model(tt(X)).argmax(1).numpy()
                f, per = macro_f1(p, y, len(MOTION_CLASSES))
                fr, per_r = macro_f1(rule_motion(X, still_thr), y, len(MOTION_CLASSES))
                report[name] = {
                    "macro_f1_motionnet": f, "macro_f1_rules": fr,
                    "accuracy_motionnet": float(np.mean(p == y)),
                    "per_class_f1_motionnet": dict(zip(MOTION_CLASSES, per)),
                    "per_class_f1_rules": dict(zip(MOTION_CLASSES, per_r)),
                    "n": int(len(y)),
                }
        report["classes"] = MOTION_CLASSES
        name = "motionnet"
    os.makedirs(ART, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ART, f"{name}.pt"))
    with open(os.path.join(ART, f"{name}.json"), "w") as f:
        json.dump({"channels": CHANNELS, "window": WINDOW, "hz": 10, **report}, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k in ("val", "test")}, indent=1))
    return report


def load(name: str) -> TCN:
    sd = torch.load(os.path.join(ART, f"{name}.pt"), weights_only=True)
    m = TCN(len(CHANNELS), 2 if name == "headingnet" else len(MOTION_CLASSES))
    if "var_scale" in sd:
        m.register_buffer("var_scale", torch.ones(1))
    m.load_state_dict(sd)
    m.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["heading", "motion"])
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    train(a.task, a.epochs)
