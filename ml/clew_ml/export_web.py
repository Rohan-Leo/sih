"""
python -m clew_ml.export_web — assets for the browser app (../public/ml, ../src/ml).

  public/ml/<model>.bin         float32 weights, state_dict order, for each of the
                                five models (speednet, headingnet, motionnet,
                                integritynet, driftnet)
  public/ml/<model>.json        manifest: tensor names/shapes/offsets + model metadata
  public/ml/replay-y1-3.bin     14 min of a held-out IO-VNBD drive, float32 rows
  public/ml/replay-y1-3.json    replay manifest: columns, origin, dead zones, route
  src/ml/speednet.vectors.json  input/output pairs to verify the TypeScript port
  src/ml/models.vectors.json    the same for the other four models
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import drift, imu_models, integrity
from .data import DT, load_processed
from .features import CHANNELS, WINDOW, drive_features, windows
from .model import SpeedNet

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PUB = os.path.join(ROOT, "public", "ml")
SRC = os.path.join(ROOT, "src", "ml")
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")

REPLAY_SEGMENT = "Y (Driver D)/Y1#3"
REPLAY_SAMPLES = 14 * 60 * 10
REPLAY_COLS = [
    "acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z", "gyro_x", "gyro_y", "gyro_z",
    "truth_e", "truth_n", "truth_speed", "truth_heading",
    "gnss_e", "gnss_n", "gnss_speed", "gnss_course", "gnss_fresh",
]


def en_to_lnglat(e: np.ndarray, n: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    R = 6371008.8
    lat = lat0 + np.degrees(n / R)
    lon = lon0 + np.degrees(e / (R * np.cos(np.radians(lat0))))
    return np.stack([lon, lat], 1)


def route_from_track(en: np.ndarray, heading: np.ndarray, lat0: float, lon0: float) -> dict:
    """OSRM-shaped route (geometry + turn steps) derived from the recorded track."""
    step = [0]
    for i in range(1, len(en)):
        if np.linalg.norm(en[i] - en[step[-1]]) >= 8:
            step.append(i)
    idx = np.array(step + [len(en) - 1])
    pts = en[idx]
    coords = en_to_lnglat(pts[:, 0], pts[:, 1], lat0, lon0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    brg = np.degrees(np.arctan2(np.diff(pts[:, 0]), np.diff(pts[:, 1]))) % 360
    # turn = heading change of > 55° across ~40 m, at least 120 m apart
    turns = []
    j = 0
    for i in range(len(brg)):
        while cum[i] - cum[j] > 40:
            j += 1
        k = i
        while k < len(brg) - 1 and cum[k] - cum[i] < 40:
            k += 1
        d = (brg[k] - brg[j] + 540) % 360 - 180
        if abs(d) > 55 and (not turns or cum[i] - turns[-1][0] > 120):
            turns.append((cum[i], i, d))
    steps = [{"name": "", "distance": 0.0, "duration": 0.0, "maneuver": {"type": "depart", "location": coords[0].tolist(), "bearing_after": float(brg[0])}}]
    for c, i, d in turns:
        mod = ("sharp " if abs(d) > 120 else "") + ("right" if d > 0 else "left")
        if abs(d) <= 70:
            mod = "slight " + ("right" if d > 0 else "left")
        steps.append({"name": "", "distance": 0.0, "duration": 0.0, "maneuver": {"type": "turn", "modifier": mod, "location": coords[i].tolist()}})
    steps.append({"name": "", "distance": 0.0, "duration": 0.0, "maneuver": {"type": "arrive", "location": coords[-1].tolist()}})
    ats = [0.0] + [t[0] for t in turns] + [float(cum[-1])]
    for s, a, b in zip(steps, ats, ats[1:] + [ats[-1]]):
        s["distance"] = float(b - a)
        s["duration"] = float((b - a) / 9.0)
    return {
        "geometry": {"coordinates": coords.round(7).tolist()},
        "distance": float(cum[-1]),
        "duration": float(cum[-1] / 9.0),
        "legs": [{"steps": steps}],
    }


def write_model(name: str, m: torch.nn.Module, meta: dict) -> None:
    tensors, off, blobs = [], 0, []
    for k, v in m.state_dict().items():
        a = v.detach().numpy().astype(np.float32).ravel()
        tensors.append({"name": k, "shape": list(v.shape), "offset": off, "size": int(a.size)})
        off += a.size
        blobs.append(a)
    np.concatenate(blobs).tofile(os.path.join(PUB, f"{name}.bin"))
    with open(os.path.join(PUB, f"{name}.json"), "w") as f:
        json.dump({**meta, "tensors": tensors}, f)
    print(f"{name}.bin {off * 4 / 1024:.0f} KB, {len(tensors)} tensors")


def fixchecker_trace(d, speednet, n: int = 1500) -> list:
    """A recorded FixChecker session (IMU steps and fixes with faults) so the TS port can be diffed."""
    from .evaluate import predict_speed
    from .features import CALIB_END

    feats, calib = drive_features(d)
    yaw = calib.signals(d)["yaw_rate"]
    v_hat, _, _ = predict_speed(speednet, feats)
    s = integrity.inject_faults(d, np.random.default_rng(3))
    fixes = {int(t): k for k, t in enumerate(s["fix"])}
    ck = integrity.FixChecker()
    ops = []
    for t in range(CALIB_END, CALIB_END + n):
        vh = v_hat[t - t % 10]
        vh = float(vh) if np.isfinite(vh) else None
        ck.imu_step(float(yaw[t]), vh)
        ops.append({"imu": [float(yaw[t]), vh]})
        k = fixes.get(t)
        if k is None or not s["present"][k]:
            continue
        E, N = map(float, s["en"][k])
        args = [E, N, t * DT, float(s["acc"][k]), float(s["speed"][k]), float(s["course"][k]), float(v_hat[t])]
        f = ck.features(*args)
        trusted = bool(f is None or not s["bad"][k])
        ck.after_fix(*args, trusted=trusted)
        ops.append({"fix": args, "trusted": trusted, "features": None if f is None else f.tolist()})
    return ops


def outage_trace() -> dict:
    from .evaluate import SpeedCorrector

    c = SpeedCorrector()
    rng = np.random.default_rng(4)
    for _ in range(40):
        c.observe(float(rng.uniform(5, 20)), float(rng.uniform(5, 20)))
    tr = drift.OutageTracker(0.3, c, 0.8, 0.4)
    steps, out = [], []
    for k in range(300):
        v, om, still = float(rng.uniform(0, 15)), float(rng.normal(0, 0.1)), bool(rng.random() < 0.1)
        tr.step(v, om, still)
        var = float(rng.uniform(0.5, 9))
        if k % 10 == 9:
            tr.speed_obs(var)
            out.append(tr.features(0.3 + 0.01 * k, 5 + 0.1 * k).tolist())
        steps.append([v, om, still, var])
    return {"corrector": {"sxy": c.sxy, "sxx": c.sxx, "sr": c.sr, "n": c.n}, "steps": steps, "features": out}


def heading_helps() -> bool:
    """Use HeadingNet in the filter only if the validation benchmark says it reduces 60 s error."""
    path = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_val.json")
    if not os.path.exists(path):
        return False
    s = json.load(open(path))["summary"]
    return "ukf_full_hd" in s and s["ukf_full_hd"]["60"]["median_m"] < s["ukf_full"]["60"]["median_m"]


def main():
    os.makedirs(PUB, exist_ok=True)
    os.makedirs(SRC, exist_ok=True)

    # ── models ──
    m = SpeedNet()
    m.load_state_dict(torch.load(os.path.join(ART, "speednet.pt"), weights_only=True))
    m.eval()
    write_model("speednet", m, {"channels": CHANNELS, "window": WINDOW, "hz": 10})
    hn, mn = imu_models.load("headingnet"), imu_models.load("motionnet")
    inn, dn = integrity.load(), drift.load()
    use_hd = heading_helps()
    write_model("headingnet", hn, {"channels": CHANNELS, "window": WINDOW, "hz": 10, "use_in_filter": use_hd})
    write_model("motionnet", mn, {"channels": CHANNELS, "window": WINDOW, "hz": 10, "classes": imu_models.MOTION_CLASSES})
    write_model("integritynet", inn, {"features": integrity.FEATURES, "threshold": 0.5})
    write_model("driftnet", dn, {"features": drift.FEATURES, "quantiles": list(drift.QUANTILES)})
    print(f"HeadingNet in filter: {use_hd}")

    # ── replay drive + test vectors ──
    d = next(x for x in load_processed() if x.name == REPLAY_SEGMENT)
    n = min(REPLAY_SAMPLES, d.T)
    feats, _ = drive_features(d)
    idx = np.array([400, 2500, 4200, 6100])
    X = windows(feats, idx)
    with torch.no_grad():
        Y = m(torch.tensor(X)).numpy()
    with open(os.path.join(SRC, "speednet.vectors.json"), "w") as f:
        json.dump({"inputs": X.round(6).tolist(), "outputs": Y.tolist()}, f)
    Xr = torch.tensor(X.round(6))
    rng = np.random.default_rng(0)
    tab = lambda mod: torch.tensor((mod.mean.numpy() + mod.std.numpy() * rng.normal(0, 1, (4, len(mod.mean)))).astype(np.float32).round(5))
    Xi, Xd = tab(inn), tab(dn)
    with torch.no_grad():
        vec = {
            "headingnet": {"inputs": X.round(6).tolist(), "outputs": hn(Xr).numpy().tolist()},
            "motionnet": {"inputs": X.round(6).tolist(), "outputs": mn(Xr).numpy().tolist()},
            "integritynet": {"inputs": Xi.numpy().tolist(), "outputs": inn(Xi).numpy().tolist()},
            "driftnet": {"inputs": Xd.numpy().tolist(), "outputs": dn(Xd).numpy().tolist()},
        }
    vec["fixchecker"] = fixchecker_trace(d, m)
    vec["outagetracker"] = outage_trace()
    with open(os.path.join(SRC, "models.vectors.json"), "w") as f:
        json.dump(vec, f)

    cols = np.column_stack([
        d.acc[:n], d.grav[:n], d.gyro[:n],
        d.truth_en[:n], d.truth_speed[:n], d.truth_heading[:n],
        d.gnss_en[:n], d.gnss_speed[:n], d.gnss_course[:n], d.gnss_fresh[:n].astype(float),
    ]).astype(np.float32)
    cols.tofile(os.path.join(PUB, "replay-y1-3.bin"))
    lat0, lon0 = map(float, d.origin)
    # dead zones as sample ranges, chosen after the 5-min calibration while the car is moving
    zones = []
    for start, length, label in [(6 * 600 + 300, 300, "Dead zone A — replayed drive, GNSS withheld 30 s"), (10 * 600, 400, "Dead zone B — replayed drive, GNSS withheld 40 s")]:
        s0 = start
        while d.truth_speed[s0] < 4:
            s0 += 10
        zones.append({"from": int(s0), "to": int(s0 + length), "label": label})
    # a multipath episode: fixes keep arriving with ±4 m "accuracy" but are pulled off the road
    f0 = 5000
    while d.truth_speed[f0] < 4:
        f0 += 10
    faults = [{"from": int(f0), "to": int(f0 + 250), "label": "Multipath — fixes pulled ~50 m off the road for 25 s",
               "offset": [35.0, -30.0], "driftPerS": [1.0, 0.8]}]
    with open(os.path.join(PUB, "replay-y1-3.json"), "w") as f:
        json.dump(
            {
                "name": "IO-VNBD held-out drive Y1 (Coventry, UK) — real phone IMU",
                "segment": REPLAY_SEGMENT,
                "hz": 10,
                "samples": int(n),
                "columns": REPLAY_COLS,
                "origin": [lat0, lon0],
                "deadZones": zones,
                "faultZones": faults,
                "calibrationSamples": 3000,
                "osrm": route_from_track(d.truth_en[:n], d.truth_heading[:n], lat0, lon0),
            },
            f,
        )
    print(f"replay {n} samples ({n * DT / 60:.0f} min), {cols.nbytes / 1024:.0f} KB, zones {zones}")


if __name__ == "__main__":
    main()
