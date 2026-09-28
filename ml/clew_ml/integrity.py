"""
IntegrityNet — is this GNSS fix trustworthy?

Phones rarely lose GNSS cleanly. In urban canyons, flyovers and tunnel mouths
they keep reporting fixes that jump tens of metres (multipath), drift away,
freeze while the car moves, or carry a wrong speed — often with a confident
accuracy figure. Accepting those fixes drags the filter off the road.

For every incoming fix we compare it with what the IMU dead reckoning expects
since the last fix we trusted, and a small MLP turns those consistency checks
into P(fault). Everything is computable live in the browser; the feature code
below is mirrored line-for-line in src/ml/integrity.ts.

Training data: IO-VNBD drives with realistic faults injected into the 1 Hz fix
stream (jumps, drifts, noise bursts, frozen fixes, speed faults, outages).
A fix is "bad" if it is > 12 m off the vehicle track, its speed is > 3 m/s
wrong, or (while moving) its course is > 30° wrong.

python -m clew_ml.integrity   → artifacts/integritynet.pt
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .data import DT, Drive, load_processed, split_of
from .evaluate import SpeedCorrector, load_model, predict_speed
from .features import CALIB_END, drive_features
from .nets import MLP, fit, n_params

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
FEATURES = [
    "log_innov_m",        # log1p |fix − DR prediction from the last trusted fix|
    "log_innov_norm",     # log1p (innovation / expected spread)
    "log_acc",            # log reported accuracy
    "acc_ratio",          # accuracy / median of recent trusted fixes
    "log_speed_resid",    # log1p |fix speed − IMU speed|
    "course_resid",       # |fix course − DR heading| / 180 (0 when slow)
    "log_implied_resid",  # log1p |distance from previous fix / dt − fix speed|
    "dt_since_good",      # seconds since the last trusted fix (clipped 60) / 60
    "speed",              # fix speed / 30
]
BAD_POS_M = 12.0
BAD_SPEED = 3.0
BAD_COURSE = 30.0


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


class FixChecker:
    """Online consistency features for a stream of fixes. Mirrors src/ml/integrity.ts."""

    def __init__(self):
        self.good = None  # (E, N, speed, course) of the last trusted fix
        self.pred = None  # DR position since the last trusted fix
        self.psi = 0.0
        self.v = 0.0
        self.dt_good = 0.0
        self.prev = None  # (E, N, t) of the previous fix, trusted or not
        self.accs: list[float] = []
        self.corr = SpeedCorrector()

    def imu_step(self, yaw_rate: float, v_hat: float | None, dt: float = DT):
        """10 Hz: propagate the dead-reckoned expectation."""
        if self.good is None:
            return
        self.dt_good += dt
        if v_hat is not None and np.isfinite(v_hat):
            self.v = self.corr.k * v_hat
        self.psi += yaw_rate * dt
        self.pred = (self.pred[0] + self.v * np.sin(self.psi) * dt, self.pred[1] + self.v * np.cos(self.psi) * dt)

    def features(self, E: float, N: float, t: float, acc: float, speed: float, course: float, v_hat: float | None) -> np.ndarray | None:
        if self.good is None:
            return None
        innov = float(np.hypot(E - self.pred[0], N - self.pred[1]))
        spread = np.sqrt(max(acc, 2.0) ** 2 + (1.0 + 0.1 * self.v * self.dt_good) ** 2)
        med = float(np.median(self.accs)) if self.accs else acc
        vi = self.corr.k * v_hat if v_hat is not None and np.isfinite(v_hat) else self.v
        cres = abs(float(wrap180(course - np.degrees(self.psi)))) / 180.0 if speed > 3 and self.v > 3 else 0.0
        if self.prev is not None and t - self.prev[2] > 0.2:
            implied = float(np.hypot(E - self.prev[0], N - self.prev[1])) / (t - self.prev[2])
            ires = abs(implied - speed)
        else:
            ires = 0.0
        return np.array([
            np.log1p(innov),
            np.log1p(innov / spread),
            np.log(max(acc, 0.5)),
            acc / max(med, 0.5),
            np.log1p(abs(speed - vi)),
            cres,
            np.log1p(ires),
            min(self.dt_good, 60.0) / 60.0,
            speed / 30.0,
        ], np.float32)

    def after_fix(self, E: float, N: float, t: float, acc: float, speed: float, course: float, v_hat: float | None, trusted: bool):
        self.prev = (E, N, t)
        if not trusted:
            return
        if v_hat is not None and np.isfinite(v_hat):
            self.corr.observe(speed, v_hat)
        self.good = (E, N, speed, course)
        self.pred = (E, N)
        self.psi = np.radians(course)
        self.v = speed
        self.dt_good = 0.0
        self.accs.append(acc)
        if len(self.accs) > 20:
            self.accs.pop(0)


def inject_faults(d: Drive, rng: np.random.Generator) -> dict:
    """Corrupted copy of the drive's 1 Hz fix stream + per-fix truth labels."""
    fix = np.flatnonzero(d.gnss_fresh)
    fix = fix[fix >= CALIB_END]
    n = len(fix)
    base_acc = rng.uniform(3, 8)
    en = d.gnss_en[fix].copy()
    sp = d.gnss_speed[fix].copy()
    co = d.gnss_course[fix].copy()
    acc = base_acc * np.exp(rng.normal(0, 0.15, n))
    present = np.ones(n, bool)
    kind = np.zeros(n, np.int8)
    i = int(rng.integers(5, 40))
    while i < n:
        L = int(rng.integers(3, 31))
        j = min(n, i + L)
        f = rng.choice(["jump", "drift", "noise", "frozen", "speed", "outage", "outage_long"], p=[0.24, 0.16, 0.14, 0.12, 0.1, 0.14, 0.1])
        if f == "jump":
            ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(15, 120)
            en[i:j] += mag * np.array([np.sin(ang), np.cos(ang)]) + rng.normal(0, 2, (j - i, 2))
            if rng.random() < 0.4:
                acc[i:j] *= rng.uniform(2, 6)
        elif f == "drift":
            ang, rate = rng.uniform(0, 2 * np.pi), rng.uniform(1.5, 6)
            k = np.arange(1, j - i + 1)[:, None]
            en[i:j] += rate * k * np.array([np.sin(ang), np.cos(ang)])
        elif f == "noise":
            en[i:j] += rng.normal(0, rng.uniform(15, 45), (j - i, 2))
            acc[i:j] *= rng.uniform(1, 4)
        elif f == "frozen":
            en[i:j], sp[i:j], co[i:j] = en[i], sp[i], co[i]
        elif f == "speed":
            sp[i:j] = np.maximum(0, sp[i:j] * rng.uniform(0.2, 0.6) if rng.random() < 0.5 else sp[i:j] + rng.uniform(4, 12))
        elif f == "outage":
            present[i:j] = False
        else:  # tunnel-length outage: fixes return far from where DR left off
            j = min(n, i + int(rng.integers(40, 150)))
            present[i:j] = False
        kind[i:j] = ["jump", "drift", "noise", "frozen", "speed", "outage", "outage_long"].index(f) + 1
        i = j + int(rng.integers(20, 90))
    t_sp = d.truth_speed[fix]
    pos_err = np.linalg.norm(en - d.truth_en[fix], axis=1)
    c_err = np.abs(wrap180(co - d.truth_heading[fix]))
    bad = (pos_err > BAD_POS_M) | (np.abs(sp - t_sp) > BAD_SPEED) | ((t_sp > 3) & (c_err > BAD_COURSE))
    return dict(fix=fix, en=en, speed=sp, course=co, acc=acc, present=present, bad=bad, kind=kind)


def drive_dataset(d: Drive, speednet, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequential simulation with teacher forcing: trust exactly the clean fixes."""
    feats, calib = drive_features(d)
    yaw = calib.signals(d)["yaw_rate"]
    v_hat, _, _ = predict_speed(speednet, feats)
    s = inject_faults(d, rng)
    ck = FixChecker()
    X, y, kinds = [], [], []
    fixes = {int(t): k for k, t in enumerate(s["fix"])}
    # warm-up: trust the calibration-period fixes
    for t in np.flatnonzero(d.gnss_fresh[:CALIB_END])[-60:]:
        ck.after_fix(*d.gnss_en[t], t * DT, 4.0, d.gnss_speed[t], d.gnss_course[t], v_hat[t], True)
    start = int(np.flatnonzero(d.gnss_fresh[:CALIB_END])[-1]) + 1
    for t in range(start, d.T):
        ck.imu_step(yaw[t], v_hat[t - t % 10] if np.isfinite(v_hat[t - t % 10]) else None)
        k = fixes.get(t)
        if k is None or not s["present"][k]:
            continue
        E, N = s["en"][k]
        args = (E, N, t * DT, s["acc"][k], s["speed"][k], s["course"][k], v_hat[t])
        f = ck.features(*args)
        if f is not None:
            X.append(f)
            y.append(bool(s["bad"][k]))
            kinds.append(int(s["kind"][k]))
        ck.after_fix(*args, trusted=not s["bad"][k])
    return np.array(X, np.float32), np.array(y, np.float32), np.array(kinds)


def dataset(drives: list[Drive], speednet, seed: int, reps: int = 1):
    rng = np.random.default_rng(seed)
    parts = [drive_dataset(d, speednet, rng) for d in drives for _ in range(reps)]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def prf(pred: np.ndarray, y: np.ndarray) -> dict:
    pos = y > 0.5
    tp = np.sum(pred & pos)
    p = tp / max(pred.sum(), 1)
    r = tp / max(pos.sum(), 1)
    return {
        "precision": float(p), "recall": float(r), "f1": float(0 if tp == 0 else 2 * p * r / (p + r)),
        "good_fixes_rejected": float(np.mean(pred[~pos])),
    }


def baselines(X: np.ndarray) -> dict[str, np.ndarray]:
    fi = {f: i for i, f in enumerate(FEATURES)}
    acc = np.exp(X[:, fi["log_acc"]])
    ratio = X[:, fi["acc_ratio"]]
    med = acc / np.maximum(ratio, 1e-6)
    norm = np.expm1(X[:, fi["log_innov_norm"]])
    return {
        "accuracy threshold (previous app rule)": acc > np.minimum(150, np.maximum(30, 3 * med)),
        "innovation gate (5σ, UKF-style)": norm > 5,
    }


def main(epochs: int = 30, seed: int = 0):
    torch.manual_seed(seed)
    drives = load_processed()
    sn = load_model()
    by = {s: [d for d in drives if split_of(d.name) == s] for s in ("train", "val", "test")}
    Xtr, ytr, _ = dataset(by["train"], sn, seed, reps=3)
    Xva, yva, _ = dataset(by["val"], sn, seed + 1)
    Xte, yte, kte = dataset(by["test"], sn, seed + 2)
    mean, std = torch.tensor(Xtr.mean(0)), torch.tensor(Xtr.std(0) + 1e-6)
    model = MLP(len(FEATURES), 1, mean, std)
    print(f"IntegrityNet: {len(Xtr)} train fixes ({ytr.mean():.0%} faulty), {n_params(model)} params")
    loss = lambda o, y: torch.nn.functional.binary_cross_entropy_with_logits(o[:, 0], y)
    score = lambda o, y: loss(o, y)
    tt = torch.tensor
    fit(model, tt(Xtr), tt(ytr), tt(Xva), tt(yva), loss, score, epochs, bs=512)
    report: dict = {"features": FEATURES, "train_fixes": int(len(Xtr))}
    with torch.no_grad():
        for name, X, y in (("val", Xva, yva), ("test", Xte, yte)):
            p = torch.sigmoid(model(tt(X))[:, 0]).numpy()
            r = {"n_fixes": int(len(y)), "faulty_share": float(y.mean()), "auc": auc(p, y),
                 "integritynet": prf(p > 0.5, y)}
            for bname, pred in baselines(X).items():
                r[bname] = prf(pred, y)
            report[name] = r
    kinds = ["clean", "jump", "drift", "noise", "frozen", "speed", "after outage", "after long outage"]
    with torch.no_grad():
        p = torch.sigmoid(model(tt(Xte))[:, 0]).numpy() > 0.5
    report["test_by_fault"] = {
        kinds[k]: {"n": int((kte == k).sum()), "flagged": float(p[kte == k].mean()),
                   "truly_bad": float(yte[kte == k].mean())} for k in range(len(kinds)) if (kte == k).any()
    }
    torch.save(model.state_dict(), os.path.join(ART, "integritynet.pt"))
    with open(os.path.join(ART, "integritynet.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))


def load() -> MLP:
    m = MLP(len(FEATURES), 1)
    m.load_state_dict(torch.load(os.path.join(ART, "integritynet.pt"), weights_only=True))
    m.eval()
    return m


if __name__ == "__main__":
    main()
