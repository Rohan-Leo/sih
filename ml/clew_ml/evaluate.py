"""
python -m clew_ml.evaluate [--split test]

Outage benchmark on held-out IO-VNBD drives.

For every segment: calibrate the mount on the first 5 min, run the GNSS-aided
filter, and at regular points while moving, cut GNSS for 120 s. Every method
starts from the same GNSS-aided state; we report horizontal error against the
vehicle's GNSS track at 10 / 30 / 60 / 120 s into the outage.
"""
from __future__ import annotations

import argparse
import json
import os

from multiprocessing import Pool

import numpy as np
import torch

from .data import DT, Drive, load_processed, split_of
from .features import CALIB_END, WINDOW, drive_features, windows
from .mapmatch import OnlineHMM, RoadGraph
from .model import SpeedNet
from .osm import load_roads
from .ukf import UKF

HORIZONS_S = (10, 30, 60, 120)
OUTAGE = int(max(HORIZONS_S) / DT)
START_EVERY = 600
NHC_MIN_R2 = 0.6  # one outage start per minute of driving
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

METHODS = {
    "freeze": "Freeze (typical map app: dot stops)",
    "hold": "Hold last GNSS speed & heading",
    "ins": "IMU integration only (UKF predict, no aiding)",
    "ukf_nhc_zupt": "UKF, non-holonomic model + ZUPT (no ML)",
    "ukf_ml_raw": "UKF + ZUPT + learned speed, no bias correction",
    "ukf_full": "UKF + ZUPT + learned speed + online bias correction (Clew)",
    "ukf_full_hd": "Clew + HeadingNet gyro-bias measurement",
    "ukf_full_mm": "Clew + HMM map matching (OSM roads)",
}


def causal_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.r_[np.zeros(k), x])
    return (c[k:] - c[:-k]) / k


def causal_std(x: np.ndarray, k: int) -> np.ndarray:
    m = causal_mean(x, k)
    m2 = causal_mean(x * x, k)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def load_model() -> SpeedNet:
    m = SpeedNet()
    m.load_state_dict(torch.load(os.path.join(ART, "speednet.pt"), weights_only=True))
    m.eval()
    return m


def predict_speed(model: SpeedNet, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(feats)
    v = np.full(T, np.nan)
    var = np.full(T, np.nan)
    p_still = np.zeros(T)
    idx = np.arange(WINDOW - 1, T)
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            b = idx[i : i + 4096]
            o = model(torch.tensor(windows(feats, b))).numpy()
            v[b], var[b], p_still[b] = o[:, 0], np.exp(o[:, 1]), 1 / (1 + np.exp(-o[:, 2]))
    return v, var, p_still


class Signals:
    """Everything the filters consume at runtime (all causal)."""

    def __init__(self, d: Drive, model: SpeedNet, still_thr: float, heading=None):
        feats, calib = drive_features(d)
        s = calib.signals(d)
        # the centripetal pseudo-measurement is only as good as the lateral mount calibration
        self.nhc_ok = calib.r2_lat > NHC_MIN_R2
        self.omega = s["yaw_rate"]
        self.a_long = s["a_long"]
        self.omega_s = causal_mean(s["yaw_rate"], 10)
        self.a_lat_s = causal_mean(s["a_lat"], 10)
        self.still_heur = (causal_std(s["acc_mag"], 20) < still_thr) & (np.abs(self.omega_s) < 0.02)
        self.v_hat, self.v_var, p = predict_speed(model, feats)
        self.still_ml = p > 0.8
        # HeadingNet yaw correction, predicted at 1 Hz and held for the next second
        self.dpsi = np.full(len(feats), np.nan)
        self.dpsi_var = np.full(len(feats), np.nan)
        if heading is not None:
            self.dpsi, self.dpsi_var = predict_heading(heading, feats)


def predict_heading(model, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HeadingNet δ̂ (rad/s) and its variance on 1 Hz ticks (t % 10 == 0); NaN elsewhere."""
    T = len(feats)
    ticks = np.arange(WINDOW, T, 10)
    with torch.no_grad():
        o = model(torch.tensor(windows(feats, ticks))).numpy()
    d, v = np.full(T, np.nan), np.full(T, np.nan)
    d[ticks] = o[:, 0] * 0.1
    v[ticks] = np.exp(np.clip(o[:, 1], -12, 2)) * float(model.var_scale[0])
    return d, v


def fit_still_threshold(drives: list[Drive]) -> float:
    """Heuristic stationarity threshold on acc-magnitude jitter, fitted on training drives."""
    still, moving = [], []
    for d in drives:
        _, calib = drive_features(d)
        sd = causal_std(calib.signals(d)["acc_mag"], 20)
        still.append(sd[d.truth_speed < 0.3])
        moving.append(sd[d.truth_speed > 3])
    s, m = np.concatenate(still), np.concatenate(moving)
    best, thr = -1.0, 0.1
    for t in np.quantile(s, np.linspace(0.05, 0.95, 37)):
        score = (s < t).mean() - 3 * (m < t).mean()  # false "still" while moving is costly
        if score > best:
            best, thr = score, float(t)
    return thr


class SpeedCorrector:
    """
    Online bias correction for the learned speed ("Predicted speed + bias
    correction" in the deployment pipeline). While GNSS is healthy it fits
    v_gnss ≈ k · v̂ with exponential forgetting (~2 min), plus the residual
    spread. During an outage the filter receives k·v̂ with that spread as noise,
    so a model that under-reads on, say, a motorway is rescaled for this drive.
    """

    def __init__(self, forget: float = 0.992):
        self.f = forget
        self.sxy = self.sxx = 0.0
        self.sr = 0.0
        self.n = 0.0

    def copy(self) -> "SpeedCorrector":
        c = SpeedCorrector(self.f)
        c.__dict__.update(self.__dict__)
        return c

    @property
    def k(self) -> float:
        return float(np.clip(self.sxy / self.sxx, 0.5, 2.5)) if self.sxx > 50 else 1.0

    def observe(self, v_gnss: float, v_hat: float):
        """Call at ~1 Hz while GNSS is good."""
        if not np.isfinite(v_hat) or v_gnss < 1.0:
            return
        r = v_gnss - self.k * v_hat
        self.sxy = self.f * self.sxy + v_gnss * v_hat
        self.sxx = self.f * self.sxx + v_hat * v_hat
        self.sr = self.f * self.sr + r * r
        self.n = self.f * self.n + 1

    def measurement(self, v_hat: float, model_var: float) -> tuple[float, float]:
        if self.n < 20:
            return v_hat, max(model_var, 1.0) * 2.0
        resid_var = self.sr / self.n
        return self.k * v_hat, max(resid_var, 1.0) * 2.0


HEADING_VAR_INFLATE = 4.0  # 1 Hz predictions from overlapping windows are correlated


def aid(u: UKF, sg: Signals, t: int, use_nhc: bool, zupt: np.ndarray | None, use_speed: bool, corr: "SpeedCorrector | None" = None, use_heading: bool = False):
    if use_heading and np.isfinite(sg.dpsi[t]):
        # HeadingNet: δ = true − calibrated yaw rate over the last 10 s, i.e. the gyro bias is −δ
        u.gyro_bias_obs(-float(sg.dpsi[t]), float(sg.dpsi_var[t]) * HEADING_VAR_INFLATE)
    if use_nhc and sg.nhc_ok and abs(sg.omega_s[t]) > 0.05:
        u.nhc_centripetal(sg.a_lat_s[t], sg.omega_s[t])
    if zupt is not None and zupt[t]:
        u.zupt()
    if use_speed and t % 10 == 0 and np.isfinite(sg.v_hat[t]):
        # predictions from overlapping windows are correlated → apply at 1 Hz, inflate variance
        if corr is not None:
            z, R = corr.measurement(float(sg.v_hat[t]), float(sg.v_var[t]))
        else:
            z, R = float(sg.v_hat[t]), max(float(sg.v_var[t]), 1.0) * 2.0
        u.speed_obs(z, R)


def run_outage(method: str, u0: UKF, sg: Signals, s: int, graph: RoadGraph | None = None, corr: SpeedCorrector | None = None) -> np.ndarray:
    """Positions (OUTAGE, 2) during a GNSS outage starting at sample s."""
    out = np.empty((OUTAGE, 2))
    if method == "freeze":
        out[:] = u0.x[:2]
        return out
    if method == "hold":
        E, N, psi, v = u0.x[:4]
        k = np.arange(1, OUTAGE + 1) * DT
        return np.stack([E + v * np.sin(psi) * k, N + v * np.cos(psi) * k], 1)
    u = u0.copy()
    hmm = OnlineHMM(graph) if (method == "ukf_full_mm" and graph is not None) else None
    if hmm:
        hmm.step(u.x[:2], float(np.sqrt(np.linalg.eigvalsh(u.P[:2, :2]).max())), 0.0)
    travelled = 0.0
    for k in range(OUTAGE):
        t = s + k
        u.predict(sg.omega[t], sg.a_long[t], DT)
        travelled += u.x[3] * DT
        if method == "ukf_nhc_zupt":
            aid(u, sg, t, True, sg.still_heur, False)
        elif method == "ukf_ml_raw":
            aid(u, sg, t, True, sg.still_ml, True)
        elif method in ("ukf_full", "ukf_full_mm"):
            aid(u, sg, t, True, sg.still_ml, True, corr)
        elif method == "ukf_full_hd":
            aid(u, sg, t, True, sg.still_ml, True, corr, use_heading=True)
        if hmm and k % 10 == 9:
            sigma = float(np.sqrt(np.linalg.eigvalsh(u.P[:2, :2]).max()))
            c = hmm.step(u.x[:2], sigma, travelled)
            travelled = 0.0
            if c is not None:
                # matched road position → soft position measurement
                u.update(c.point, lambda X: X[:, :2], np.eye(2) * 8.0**2)
        out[k] = u.x[:2]
    return out


def evaluate_segment(d: Drive, model: SpeedNet, still_thr: float, heading=None) -> list[dict]:
    sg = Signals(d, model, still_thr, heading)
    roads = load_roads(d)
    graph = RoadGraph(roads) if roads else None
    t0 = CALIB_END
    u = UKF()
    u.init(*d.gnss_en[t0], np.radians(d.gnss_course[t0]), d.gnss_speed[t0])
    starts = set(range(t0 + START_EVERY, d.T - OUTAGE, START_EVERY))
    rows = []
    corr = SpeedCorrector()
    for t in range(WINDOW, t0):  # calibration period: GNSS is healthy, learn the model's bias
        if d.gnss_fresh[t]:
            corr.observe(d.gnss_speed[t], sg.v_hat[t])
    # HeadingNet runs in its own GNSS-aided filter the whole time (as in the app)
    u_hd = u.copy() if heading is not None else None
    for t in range(t0 + 1, d.T - OUTAGE):
        u.predict(sg.omega[t], sg.a_long[t], DT)
        aid(u, sg, t, True, sg.still_ml, True, corr)
        if u_hd is not None:
            u_hd.predict(sg.omega[t], sg.a_long[t], DT)
            aid(u_hd, sg, t, True, sg.still_ml, True, corr, use_heading=True)
        if d.gnss_fresh[t]:
            u.gnss(*d.gnss_en[t], d.gnss_acc[t], d.gnss_speed[t], d.gnss_course[t])
            if u_hd is not None:
                u_hd.gnss(*d.gnss_en[t], d.gnss_acc[t], d.gnss_speed[t], d.gnss_course[t])
            corr.observe(d.gnss_speed[t], sg.v_hat[t])
        if t in starts and d.truth_speed[t] > 2:
            truth = d.truth_en[t + 1 : t + 1 + OUTAGE]
            dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))]
            row = {"segment": d.name, "t": t, "speed": float(d.truth_speed[t])}
            for m in METHODS:
                if (m == "ukf_full_mm" and graph is None) or (m == "ukf_full_hd" and heading is None):
                    continue
                path = run_outage(m, u_hd if m == "ukf_full_hd" else u, sg, t + 1, graph, corr.copy())
                err = np.linalg.norm(path - truth, axis=1)
                for h in HORIZONS_S:
                    i = int(h / DT) - 1
                    row[f"{m}@{h}"] = float(err[i])
                    row[f"dist@{h}"] = float(dist[i])
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for m in METHODS:
        rs = [r for r in rows if f"{m}@{HORIZONS_S[0]}" in r]
        if not rs:
            continue
        out[m] = {"n": len(rs)}
        for h in HORIZONS_S:
            e = np.array([r[f"{m}@{h}"] for r in rs])
            dist = np.array([r[f"dist@{h}"] for r in rs])
            out[m][h] = {
                "median_m": float(np.median(e)),
                "mean_m": float(np.mean(e)),
                "p90_m": float(np.percentile(e, 90)),
                "pct_of_distance": float(100 * np.median(e / np.maximum(dist, 1))),
            }
    return out


def to_markdown(summary: dict, n: int, split: str, hours: float) -> str:
    lines = [
        f"Outage benchmark — IO-VNBD {split} split ({hours:.1f} h of held-out driving, {n} simulated outages)",
        "",
        "Median horizontal error (m) after GNSS is lost; in brackets: median error as % of distance driven.",
        "",
        "| Method | " + " | ".join(f"{h} s" for h in HORIZONS_S) + " |",
        "|---|" + "---:|" * len(HORIZONS_S),
    ]
    for m, label in METHODS.items():
        if m not in summary:
            continue
        cells = [f"{summary[m][h]['median_m']:.0f} ({summary[m][h]['pct_of_distance']:.1f}%)" for h in HORIZONS_S]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    drives = load_processed()
    model = load_model()
    thr = fit_still_threshold([d for d in drives if split_of(d.name) == "train"])
    segs = [d for d in drives if split_of(d.name) == args.split]
    heading = None
    if os.path.exists(os.path.join(ART, "headingnet.pt")):
        from .imu_models import load as load_imu

        heading = load_imu("headingnet")
    with Pool(min(4, os.cpu_count() or 1)) as p:
        per_seg = p.starmap(evaluate_segment, [(d, model, thr, heading) for d in segs])
    rows = []
    for d, r in zip(segs, per_seg):
        rows += r
        if r:
            print(f"{d.name:32s} {len(r):3d} outages  clew@60s median {np.median([x['ukf_full@60'] for x in r]):6.1f} m   hold {np.median([x['hold@60'] for x in r]):6.1f} m")
    summary = summarize(rows)
    hours = sum(d.T for d in segs) * DT / 3600
    md = to_markdown(summary, len(rows), args.split, hours)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"benchmark_{args.split}.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows, "still_threshold": thr}, f, indent=1)
    with open(os.path.join(RESULTS, f"benchmark_{args.split}.md"), "w") as f:
        f.write(md)
    print()
    print(md)


if __name__ == "__main__":
    main()
