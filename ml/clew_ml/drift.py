"""
DriftNet — how far off is the dead-reckoned dot right now?

During an outage the map draws a halo around the dot. The UKF's own
covariance is a poor radius: the filter assumes its speed and heading aids
are unbiased, so it is overconfident exactly when it drifts. DriftNet learns
the error distribution from thousands of simulated outages on training
drives and predicts the 50 / 68 / 95 % error radii from what the app knows
live: time and distance since GNSS, speed, how much the car turned and the
filter's own sigma.

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
from .features import CALIB_END, WINDOW
from .nets import MLP, fit, n_params
from .ukf import UKF

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
FEATURES = [
    "elapsed",         # seconds since GNSS was lost / 120
    "log_dist",        # log1p dead-reckoned distance, m
    "mean_speed",      # dead-reckoned mean speed / 30
    "heading_change",  # |net heading change| since loss, rad
    "turn_amount",     # ∫|yaw rate| dt since loss, rad
    "log_sigma",       # log of the UKF's 1-σ position radius, m
]
# Per-drive constants (mount-calibration R², speed-correction residual) and the speed
# model's σ were tried too: with ~20 training drives they let the net memorise drives,
# and test quantile loss got worse (0.34 vs 0.21) — see ml/README.md.
QUANTILES = (0.5, 0.68, 0.95)


def quantiles(o: torch.Tensor) -> torch.Tensor:
    """Head output → monotone log1p-error quantiles (N, 3)."""
    sp = torch.nn.functional.softplus
    q50 = o[:, 0]
    q68 = q50 + sp(o[:, 1])
    q95 = q68 + sp(o[:, 2])
    return torch.stack([q50, q68, q95], 1)


def radii(model: MLP, X: torch.Tensor) -> np.ndarray:
    """Calibrated error radii in metres (N, 3): quantiles + conformal offsets, kept monotone."""
    q = quantiles(model(X)) + model.offsets[None, :]
    q = torch.cummax(q, dim=1).values
    return np.expm1(q.numpy())


def pinball(o: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q = quantiles(o)
    tau = torch.tensor(QUANTILES)[None, :]
    e = y[:, None] - q
    return torch.maximum(tau * e, (tau - 1) * e).mean()


class OutageTracker:
    """Live feature accumulator for one outage. Mirrors src/ml/drift.ts."""

    def __init__(self, psi0: float):
        self.psi0 = psi0
        self.t = 0.0
        self.dist = 0.0
        self.turn = 0.0

    def step(self, v: float, omega: float, dt: float = DT):
        self.t += dt
        self.dist += v * dt
        self.turn += abs(omega) * dt

    def features(self, psi: float, sigma: float) -> np.ndarray:
        dpsi = abs((psi - self.psi0 + np.pi) % (2 * np.pi) - np.pi)
        return np.array([
            self.t / 120.0,
            np.log1p(self.dist),
            self.dist / max(self.t, 1e-3) / 30.0,
            dpsi,
            self.turn,
            np.log(max(sigma, 0.5)),
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
        tr = OutageTracker(v.x[2])
        for k in range(OUTAGE):
            s = t + 1 + k
            om = float(sg.omega[s])
            v.predict(om, sg.a_long[s], DT)
            aid(v, sg, s, True, sg.still_ml, True, c, use_heading)
            tr.step(v.x[3], om)
            if k % 10 == 9:
                sigma = float(np.sqrt(np.linalg.eigvalsh(v.P[:2, :2]).max()))
                X.append(tr.features(v.x[2], sigma))
                err.append(float(np.linalg.norm(v.x[:2] - d.truth_en[s])))
                sig.append(sigma)
    return np.array(X, np.float32).reshape(-1, len(FEATURES)), np.array(err, np.float32), np.array(sig, np.float32)


def dataset(drives: list[Drive], still_thr: float, every: int, use_heading: bool, cache: str | None = None):
    """Outage-second rows for these drives; cached in data/cache because the simulation is slow."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"drift_{cache}_{every}_{int(use_heading)}_f{len(FEATURES)}.npz") if cache else None
    if path and os.path.exists(path):
        z = np.load(path)
        return z["X"], z["e"], z["s"]
    with Pool(min(4, os.cpu_count() or 1)) as p:
        parts = p.map(segment_rows, [(d, still_thr, every, use_heading) for d in drives])
    X, e, s = (np.concatenate([q[i] for q in parts]) for i in range(3))
    if path:
        np.savez(path, X=X, e=e, s=s)
    return X, e, s


def coverage(radius: np.ndarray, err: np.ndarray) -> float:
    return float(np.mean(err <= radius))


# a 2-D isotropic Gaussian with 1-σ radius s holds 50 / 68 / 95 % within these multiples
RAYLEIGH = np.array([1.177, 1.510, 2.448])


def quantile_loss(r: np.ndarray, err: np.ndarray) -> float:
    """Mean pinball loss of log1p radii (N, 3) — a proper score: lower is better, rewards sharp *and* calibrated."""
    y = np.log1p(err)[:, None]
    q = np.log1p(r)
    tau = np.array(QUANTILES)[None, :]
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def summary(r: np.ndarray, err: np.ndarray) -> dict:
    return {"cover_50": coverage(r[:, 0], err), "cover_68": coverage(r[:, 1], err), "cover_95": coverage(r[:, 2], err),
            "median_r68_m": float(np.median(r[:, 1])), "median_r95_m": float(np.median(r[:, 2])),
            "quantile_loss": quantile_loss(r, err)}


def heuristic_radii(X: np.ndarray) -> np.ndarray:
    """positionEstimator.drSigma: σ = σ_loss + 0.4·t + 0.12·distance (σ_loss ≈ 5 m)."""
    t = X[:, FEATURES.index("elapsed")] * 120
    dist = np.expm1(X[:, FEATURES.index("log_dist")])
    s = np.minimum(300, 5 + 0.4 * t + 0.12 * dist)
    return s[:, None] * RAYLEIGH[None, :]


def evaluate(model: MLP, splits: dict) -> dict:
    out = {}
    with torch.no_grad():
        for name, (X, e, s) in splits.items():
            out[name] = {
                "n": int(len(e)),
                "driftnet": summary(radii(model, torch.tensor(X)), e),
                "ukf_covariance": summary(s[:, None] * RAYLEIGH[None, :], e),
                "heuristic": summary(heuristic_radii(X), e),
            }
    return out


def main(use_heading: bool, epochs: int = 40, seed: int = 0):
    torch.manual_seed(seed)
    drives = load_processed()
    thr = fit_still_threshold([d for d in drives if split_of(d.name) == "train"])
    by = {s: [d for d in drives if split_of(d.name) == s] for s in ("train", "val", "test")}
    Xtr, etr, _ = dataset(by["train"], thr, 300, use_heading, "train")
    Xva, eva, sva = dataset(by["val"], thr, 600, use_heading, "val")
    Xte, ete, ste = dataset(by["test"], thr, 600, use_heading, "test")
    ytr, yva = np.log1p(etr), np.log1p(eva)
    mean, std = torch.tensor(Xtr.mean(0)), torch.tensor(Xtr.std(0) + 1e-6)
    model = MLP(len(FEATURES), 3, mean, std)
    print(f"DriftNet: {len(Xtr)} train rows (outage-seconds), {n_params(model)} params")
    tt = torch.tensor
    fit(model, tt(Xtr), tt(ytr), tt(Xva), tt(yva), pinball, pinball, epochs, bs=512)
    # Conformal calibration on validation drives: shift each log-quantile so that it
    # covers its nominal share there. Training drives are optimistic (the speed model
    # has seen them), so the raw quantiles under-cover on unseen drives.
    with torch.no_grad():
        q = quantiles(model(tt(Xva))).numpy()
    offsets = [float(np.quantile(yva - q[:, i], tau)) for i, tau in enumerate(QUANTILES)]
    model.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32))
    report: dict = {"features": FEATURES, "quantiles": list(QUANTILES), "train_rows": int(len(Xtr)),
                    "use_heading": use_heading, "conformal_offsets": offsets}
    report.update(evaluate(model, {"val": (Xva, eva, sva), "test": (Xte, ete, ste)}))
    torch.save(model.state_dict(), os.path.join(ART, "driftnet.pt"))
    with open(os.path.join(ART, "driftnet.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))


def load() -> MLP:
    m = MLP(len(FEATURES), 3)
    m.register_buffer("offsets", torch.zeros(3))
    m.load_state_dict(torch.load(os.path.join(ART, "driftnet.pt"), weights_only=True))
    m.eval()
    return m


if __name__ == "__main__":
    import sys

    from .export_web import heading_helps

    if "--eval" in sys.argv:  # re-score the saved model without retraining
        drives = load_processed()
        thr = fit_still_threshold([d for d in drives if split_of(d.name) == "train"])
        with open(os.path.join(ART, "driftnet.json")) as f:
            rep = json.load(f)
        splits = {s: dataset([d for d in drives if split_of(d.name) == s], thr, 600, rep["use_heading"], s) for s in ("val", "test")}
        rep.update(evaluate(load(), splits))
        with open(os.path.join(ART, "driftnet.json"), "w") as f:
            json.dump(rep, f, indent=2)
        print(json.dumps({k: rep[k] for k in ("val", "test")}, indent=1))
    else:
        main(heading_helps())
