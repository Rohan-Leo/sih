"""
python -m clew_ml.export_web — assets for the browser app (../public/ml, ../src/ml).

  public/ml/speednet.bin        float32 weights, state_dict order
  public/ml/speednet.json       manifest: tensor names/shapes/offsets, channels, window
  public/ml/replay-y1-3.bin     14 min of a held-out IO-VNBD drive, float32 rows
  public/ml/replay-y1-3.json    replay manifest: columns, origin, dead zones, route
  src/ml/speednet.vectors.json  input/output pairs to verify the TypeScript port
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

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


def main():
    os.makedirs(PUB, exist_ok=True)
    os.makedirs(SRC, exist_ok=True)

    # ── model ──
    m = SpeedNet()
    m.load_state_dict(torch.load(os.path.join(ART, "speednet.pt"), weights_only=True))
    m.eval()
    tensors, off, blobs = [], 0, []
    for k, v in m.state_dict().items():
        a = v.detach().numpy().astype(np.float32).ravel()
        tensors.append({"name": k, "shape": list(v.shape), "offset": off, "size": int(a.size)})
        off += a.size
        blobs.append(a)
    np.concatenate(blobs).tofile(os.path.join(PUB, "speednet.bin"))
    with open(os.path.join(PUB, "speednet.json"), "w") as f:
        json.dump({"channels": CHANNELS, "window": WINDOW, "hz": 10, "tensors": tensors}, f)
    print(f"speednet.bin {off * 4 / 1024:.0f} KB, {len(tensors)} tensors")

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
                "calibrationSamples": 3000,
                "osrm": route_from_track(d.truth_en[:n], d.truth_heading[:n], lat0, lon0),
            },
            f,
        )
    print(f"replay {n} samples ({n * DT / 60:.0f} min), {cols.nbytes / 1024:.0f} KB, zones {zones}")


if __name__ == "__main__":
    main()
