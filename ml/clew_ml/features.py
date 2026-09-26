"""Mount-invariant IMU features and training windows."""
from __future__ import annotations

import numpy as np

from .calib import MountCalib, calibrate
from .data import Drive

CHANNELS = ["yaw_rate", "a_long", "a_lat", "a_vert", "acc_mag", "gyro_mag", "v_turn", "turning"]
WINDOW = 100  # 10 s at 10 Hz, causal (ends at the current sample)
CALIB_END = 3000  # first 5 min of each segment: GNSS-healthy calibration period


def causal_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing k-sample mean (what a live app can compute); shorter at the start."""
    c = np.cumsum(np.r_[0.0, x])
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - k)
    return (c[i] - c[lo]) / (i - lo)


def drive_features(d: Drive, calib: MountCalib | None = None) -> tuple[np.ndarray, MountCalib]:
    calib = calib or calibrate(d, CALIB_END)
    s = calib.signals(d)
    # Non-holonomic prior: with no sideways slip, a_lat = v·ψ̇, so in a turn
    # speed is directly observable. Trailing 1 s mean to beat vibration noise.
    yr, al = causal_mean(s["yaw_rate"], 10), causal_mean(s["a_lat"], 10)
    turning = np.abs(yr) > 0.05
    s["v_turn"] = np.where(turning, np.clip(al / np.where(turning, yr, 1.0), 0, 40), 0.0)
    s["turning"] = turning.astype(float)
    return np.stack([s[c] for c in CHANNELS], 1).astype(np.float32), calib


def windows(feats: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(N, C, WINDOW) windows ending at each index in idx (idx >= WINDOW - 1)."""
    offs = np.arange(-WINDOW + 1, 1)
    return np.transpose(feats[idx[:, None] + offs[None, :]], (0, 2, 1))


def make_dataset(drives: list[Drive], stride: int = 5) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for d in drives:
        f, _ = drive_features(d)
        idx = np.arange(WINDOW - 1, d.T, stride)
        X.append(windows(f, idx))
        y.append(d.truth_speed[idx].astype(np.float32))
    return np.concatenate(X), np.concatenate(y)
