"""
DriftNet — how far off is the dead-reckoned dot right now?

During an outage the map draws a halo around the dot. The UKF's own
covariance is a poor radius: the filter assumes its speed and heading aids
are unbiased, so it is overconfident exactly when it drifts. DriftNet learns
the error distribution from thousands of simulated outages on training
drives and predicts the 50 / 68 / 95 % error radii from what the app knows
live: time and distance since GNSS, speed, how much the car turned, the speed
model's uncertainty, how well the speed correction was fitting before the
outage, the mount-calibration quality and the filter's own sigma.

python -m clew_ml.drift   → artifacts/driftnet.pt
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import torch

from .data import DT, Drive, load_processed, split_of
from .evaluate import OUTAGE, SpeedCorrector, Signals, aid, fit_still_threshold, load_model
from .features import CALIB_END, WINDOW, drive_features
from .nets import MLP, fit, n_params
from .ukf import UKF

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
FEATURES = [
    "elapsed",       # seconds since GNSS was lost / 120
    "log_dist",      # log1p dead-reckoned distance, m
    "mean_speed",    # dead-reckoned mean speed / 30
    "heading_change",  # |net heading change| since loss, rad
    "turn_amount",   # ∫|yaw rate| dt since loss, rad
    "speed_sd",      # mean SpeedNet σ during the outage, m/s
    "corr_resid",    # RMS residual of the online speed correction before the loss, m/s
    "log_sigma",     # log of the UKF's 1-σ position radius, m
    "still_frac",    # share of the outage the car was judged stationary
    "r2_yaw",        # mount-calibration fit quality
    "r2_long",
]
QUANTILES = (0.5, 0.68, 0.95)


def quantiles(o: torch.Tensor) -> torch.Tensor:
    """Head output → monotone log1p-error quantiles (N, 3)."""
    sp = torch.nn.functional.softplus
    q50 = o[:, 0]
    q68 = q50 + sp(o[:, 1])
    q95 = q68 + sp(o[:, 2])
    return torch.stack([q50, q68, q95], 1)


def pinball(o: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = quantiles(o)
    tau = torch.tensor(QUANTILES)[None, :]
    e = y[:, None] - q
    return torch.maximum(tau * e, (tau - 1) * e).mean()


class OutageTracker:
    """Live feature accumulator for one outage. Mirrors src/ml/drift.ts."""

    def __init__(self, psi0: float, corr: SpeedCorrector, r2_yaw: float, r2_long: float):
        self.psi0 = psi0
        self.t = 0.0
        self.dist = 0.0
        self.turn = 0.0
        self.sd_sum = 0.0
        self.sd_n = 0
        self.still = 0.0
        self.corr_resid = float(np.sqrt(corr.sr / corr.n)) if corr.n >= 20 else 3.0
        self.r2 = (r2_yaw, r2_long)

    def step(self, v: float, omega: float, still: bool, dt: float = DT):
        self.t += dt
        self.dist += v * dt
        self.turn += abs(omega) * dt
        self.still += dt if still else 0.0

    def speed_obs(self, var: float):
        if np.isfinite(var):
            self.sd_sum += float(np.sqrt(var))
            self.sd_n += 1

    def features(self, psi: float, sigma: float) -> np.ndarray:
        dpsi = abs((psi - self.psi0 + np.pi) % (2 * np.pi) - np.pi)
        return np.array([
            self.t / 120.0,
            np.log1p(self.dist),
            self.dist / max(self.t, 1e-3) / 30.0,
            dpsi,
            self.turn,
            self.sd_sum / self.sd_n if self.sd_n else 3.0,
            self.corr_resid,
            np.log(max(sigma, 0.5)),
            self.still / max(self.t, 1e-3),
            self.r2[0],
            self.r2[1],
        ], np.float32)


def segment_rows(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features, error m, UKF sigma m) at every second of simulated outages on one segment."""
    d, still_thr, every, use_heading = args
    model = load_model()
    heading = None
    if use_heading:
        from .imu_models import load as load_imu

        heading = load_imu("headingnet")
    sg = Signals(d, model, still_thr, heading)
    _, calib = drive_features(d)
    t0 = CALIB_END
    u = UKF()
    u.init(*d.gnss_en[t0], np.radians(d.gnss_course[t0]), d.gnss_speed[t0])
    corr = SpeedCorrector()
    for t in range(WINDOW, t0):
        if d.gnss_fresh[t]:
            corr.observe(d.gnss_speed[t], sg.v_hat[t])
    X, err, sig = [], [], []
    starts = set(range(t0 + every, d.T - OUTAGE, every))
    for t in range(t0 + 1, d.T - OUTAGE):
        u.predict(sg.omega[t], sg.a_long[t], DT)
        aid(u, sg, t, True, sg.still_ml, True, corr, use_heading)
        if d.gnss_fresh[t]:
            u.gnss(*d.gnss_en[t], d.gnss_acc[t], d.gnss_speed[t], d.gnss_course[t])
            corr.observe(d.gnss_speed[t], sg.v_hat[t])
        if t not in starts or d.truth_speed[t] <= 2:
            continue
        v = u.copy()
        c = corr.copy()
        tr = OutageTracker(v.x[2], c, calib.r2_yaw, calib.r2_long)
        for k in range(OUTAGE):
            s = t + 1 + k
            om = float(sg.omega[s])
            v.predict(om, sg.a_long[s], DT)
            aid(v, sg, s, True, sg.still_ml, True, c, use_heading)
            tr.step(v.x[3], om, bool(sg.still_ml[s]))
            if s % 10 == 0 and np.isfinite(sg.v_var[s]):
                tr.speed_obs(float(sg.v_var[s]))
            if k % 10 == 9:
                sigma = float(np.sqrt(np.linalg.eigvalsh(v.P[:2, :2]).max()))
                X.append(tr.features(v.x[2], sigma))
                err.append(float(np.linalg.norm(v.x[:2] - d.truth_en[s])))
                sig.append(sigma)
    return np.array(X, np.float32).reshape(-1, len(FEATURES)), np.array(err, np.float32), np.array(sig, np.float32)


def dataset(drives: list[Drive], still_thr: float, every: int, use_heading: bool):
    with Pool(min(4, os.cpu_count() or 1)) as p:
        parts = p.map(segment_rows, [(d, still_thr, every, use_heading) for d in drives])
    return tuple(np.concatenate([q[i] for q in parts]) for i in range(3))


def coverage(radius: np.ndarray, err: np.ndarray) -> float:
    return float(np.mean(err <= radius))


def main(use_heading: bool, epochs: int = 40, seed: int = 0):
    torch.manual_seed(seed)
    drives = load_processed()
    thr = fit_still_threshold([d for d in drives if split_of(d.name) == "train"])
    by = {s: [d for d in drives if split_of(d.name) == s] for s in ("train", "val", "test")}
    Xtr, etr, _ = dataset(by["train"], thr, 300, use_heading)
    Xva, eva, sva = dataset(by["val"], thr, 600, use_heading)
    Xte, ete, ste = dataset(by["test"], thr, 600, use_heading)
    ytr, yva = np.log1p(etr), np.log1p(eva)
    mean, std = torch.tensor(Xtr.mean(0)), torch.tensor(Xtr.std(0) + 1e-6)
    model = MLP(len(FEATURES), 3, mean, std)
    print(f"DriftNet: {len(Xtr)} train rows (outage-seconds), {n_params(model)} params")
    tt = torch.tensor
    fit(model, tt(Xtr), tt(ytr), tt(Xva), tt(yva), pinball, pinball, epochs, bs=512)
    report: dict = {"features": FEATURES, "quantiles": list(QUANTILES), "train_rows": int(len(Xtr)), "use_heading": use_heading}
    with torch.no_grad():
        for name, X, e, s in (("val", Xva, eva, sva), ("test", Xte, ete, ste)):
            q = np.expm1(quantiles(model(tt(X))).numpy())
            report[name] = {
                "n": int(len(e)),
                "driftnet": {"cover_68": coverage(q[:, 1], e), "cover_95": coverage(q[:, 2], e),
                             "median_r68_m": float(np.median(q[:, 1])), "median_r95_m": float(np.median(q[:, 2]))},
                # a 2-D Gaussian holds 68 % within 1.51σ and 95 % within 2.45σ
                "ukf_covariance": {"cover_68": coverage(1.51 * s, e), "cover_95": coverage(2.45 * s, e),
                                   "median_r68_m": float(np.median(1.51 * s)), "median_r95_m": float(np.median(2.45 * s))},
            }
    torch.save(model.state_dict(), os.path.join(ART, "driftnet.pt"))
    with open(os.path.join(ART, "driftnet.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))


def load() -> MLP:
    m = MLP(len(FEATURES), 3)
    m.load_state_dict(torch.load(os.path.join(ART, "driftnet.pt"), weights_only=True))
    m.eval()
    return m


if __name__ == "__main__":
    from .export_web import heading_helps

    main(heading_helps())
