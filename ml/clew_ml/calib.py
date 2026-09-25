"""
Mount calibration: how the phone sits in the vehicle.

Nothing here assumes axis names or signs. While GNSS is healthy we regress
  GNSS heading rate      ≈ gyro · w_yaw
  GNSS speed change      ≈ (acc − gravity) · u_fwd
  centripetal v·ψ̇        ≈ (acc − gravity) · u_lat
over ~1 s fix intervals. The same procedure runs on the phone at the start
of a trip (and again if the event classifier says the phone was disturbed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DT, Drive


@dataclass
class MountCalib:
    w_yaw: np.ndarray  # (3,) gyro → yaw rate, rad/s, clockwise-positive (compass sense)
    u_fwd: np.ndarray  # (3,) dynamic acc → longitudinal, m/s²
    u_lat: np.ndarray  # (3,) dynamic acc → lateral (rightward), m/s²
    n_intervals: int
    # in-sample fit quality (R²) — decides which aids are trustworthy for this mount
    r2_yaw: float = 0.0
    r2_long: float = 0.0
    r2_lat: float = 0.0

    def signals(self, d: Drive) -> dict[str, np.ndarray]:
        dyn = d.acc - d.grav
        g = d.grav / np.maximum(np.linalg.norm(d.grav, axis=1, keepdims=True), 1e-6)
        return {
            "yaw_rate": d.gyro @ self.w_yaw,
            "a_long": dyn @ self.u_fwd,
            "a_lat": dyn @ self.u_lat,
            "a_vert": (dyn * g).sum(1),
            "acc_mag": np.linalg.norm(dyn, axis=1),
            "gyro_mag": np.linalg.norm(d.gyro, axis=1),
        }


def _r2(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    if len(y) < 20:
        return 0.0
    return float(1 - np.var(y - X @ w) / max(np.var(y), 1e-12))


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-4) -> np.ndarray:
    A = X.T @ X
    return np.linalg.solve(A + lam * np.trace(A) / len(A) * np.eye(len(A)), X.T @ y)


def _spaced_fixes(fresh: np.ndarray, min_gap: int) -> np.ndarray:
    """Fresh-fix indices at least `min_gap` samples apart (some logs repeat fixes at 10 Hz)."""
    out, last = [], -min_gap
    for i in np.flatnonzero(fresh):
        if i - last >= min_gap:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def calibrate(d: Drive, end: int) -> MountCalib:
    """Fit on samples [0, end) using only phone GNSS + phone IMU."""
    fix = _spaced_fixes(d.gnss_fresh[:end], 10)
    dyn = d.acc - d.grav
    rows_g, ys_g, rows_a, ys_a, rows_l, ys_l = [], [], [], [], [], []
    for a, b in zip(fix[:-1], fix[1:]):
        dt = (b - a) * DT
        if not (0.5 <= dt <= 2.5):
            continue
        v0, v1 = d.gnss_speed[a], d.gnss_speed[b]
        g_mean = d.gyro[a:b].mean(0)
        a_mean = dyn[a:b].mean(0)
        rows_a.append(a_mean)
        ys_a.append((v1 - v0) / dt)
        if min(v0, v1) > 5:
            dpsi = (d.gnss_course[b] - d.gnss_course[a] + 540) % 360 - 180
            if abs(dpsi) < 45:
                rate = np.radians(dpsi) / dt
                rows_g.append(g_mean)
                ys_g.append(rate)
                rows_l.append(a_mean)
                ys_l.append(0.5 * (v0 + v1) * rate)
    w = _ridge(np.array(rows_g), np.array(ys_g)) if len(rows_g) > 20 else np.zeros(3)
    u = _ridge(np.array(rows_a), np.array(ys_a)) if len(rows_a) > 20 else np.zeros(3)
    ul = _ridge(np.array(rows_l), np.array(ys_l)) if len(rows_l) > 20 else np.zeros(3)
    G, A, L = np.array(rows_g), np.array(rows_a), np.array(rows_l)
    return MountCalib(
        w, u, ul, len(rows_a),
        r2_yaw=_r2(G, np.array(ys_g), w) if len(G) else 0.0,
        r2_long=_r2(A, np.array(ys_a), u) if len(A) else 0.0,
        r2_lat=_r2(L, np.array(ys_l), ul) if len(L) else 0.0,
    )


def heading_rate_truth(d: Drive) -> np.ndarray:
    h = np.unwrap(np.radians(d.truth_heading))
    r = np.gradient(h) / DT
    r[d.truth_speed < 3] = 0.0
    return r
