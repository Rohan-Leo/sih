"""
python -m clew_ml.report — one table per model from artifacts/*.json and
results/benchmark_*.json, written to results/models.md.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, "..", "artifacts")
RES = os.path.join(HERE, "..", "results")


def j(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def main():
    L: list[str] = ["# Clew's five models — held-out results", ""]
    L.append("All numbers are on IO-VNBD drives the models never trained on: *val* (4.7 h) tunes, *test* (2.9 h) reports.")
    L.append("")

    b = {s: j(os.path.join(RES, f"benchmark_{s}.json"))["summary"] for s in ("val", "test")}
    L += ["## 1. SpeedNet — learned speed (drives the whole outage benchmark)", "",
          "Median position error after GNSS is lost (m):", "",
          "| Split | Method | 10 s | 30 s | 60 s | 120 s |", "|---|---|---:|---:|---:|---:|"]
    for s in ("val", "test"):
        for m, label in (("ins", "IMU integration only"), ("ukf_nhc_zupt", "UKF, no ML"), ("ukf_full", "**Clew (UKF + SpeedNet + bias correction)**")):
            L.append(f"| {s} | {label} | " + " | ".join(f"{b[s][m][h]['median_m']:.0f}" for h in ("10", "30", "60", "120")) + " |")
    L.append("")

    h = j(os.path.join(ART, "headingnet.json"))
    pub = os.path.join(HERE, "..", "..", "public", "ml", "headingnet.json")
    hd_on = os.path.exists(pub) and j(pub).get("use_in_filter", False)
    L += ["## 2. HeadingNet — gyro yaw-rate error", "",
          "| Split | Yaw-rate RMSE, calibrated gyro | with HeadingNet | errors within ±1σ |", "|---|---:|---:|---:|"]
    for s in ("val", "test"):
        r = h[s]
        L.append(f"| {s} | {r['yaw_rmse_calibrated_gyro'] * 57.3:.2f} °/s | {r['yaw_rmse_headingnet'] * 57.3:.2f} °/s | {pct(r['within_1sigma'])} |")
    L += ["", "In the outage benchmark, as a gyro-bias measurement in the UKF (median m at 10 / 30 / 60 / 120 s):", ""]
    for s in ("val", "test"):
        if "ukf_full_hd" in b[s]:
            a = " / ".join(f"{b[s]['ukf_full'][x]['median_m']:.0f}" for x in ("10", "30", "60", "120"))
            c = " / ".join(f"{b[s]['ukf_full_hd'][x]['median_m']:.0f}" for x in ("10", "30", "60", "120"))
            L.append(f"- {s}: Clew {a} → with HeadingNet {c}")
    L += ["", "It predicts the gyro error better than the raw calibration, but the UKF already learns the gyro bias from GNSS "
          "before an outage, so the outage error doesn't improve. The export step turns it on in the filter only if the "
          f"validation benchmark improves by ≥ 2 % at 60 s without getting worse at 120 s. Currently: **{'on' if hd_on else 'off'}**; "
          "the app still shows its gyro-error estimate.", ""]

    m = j(os.path.join(ART, "motionnet.json"))
    cls = m["classes"]
    L += ["## 3. MotionNet — driving state", "",
          "| Split | Method | Macro-F1 | " + " | ".join(cls) + " |", "|---|---|---:|" + "---:|" * len(cls)]
    for s in ("val", "test"):
        r = m[s]
        L.append(f"| {s} | hand-tuned rules | {r['macro_f1_rules']:.2f} | " + " | ".join(f"{r['per_class_f1_rules'][c]:.2f}" for c in cls) + " |")
        L.append(f"| {s} | **MotionNet** | **{r['macro_f1_motionnet']:.2f}** | " + " | ".join(f"{r['per_class_f1_motionnet'][c]:.2f}" for c in cls) + " |")
    L.append("")

    i = j(os.path.join(ART, "integritynet.json"))
    L += ["## 4. IntegrityNet — GNSS fault detection", "",
          "Faults injected into the 1 Hz fix stream of held-out drives (jumps, drifts, noise bursts, frozen fixes, speed faults, outages).", "",
          "| Split | Detector | Precision | Recall | F1 | Good fixes rejected |", "|---|---|---:|---:|---:|---:|"]
    for s in ("val", "test"):
        r = i[s]
        for k in [k for k in r if isinstance(r[k], dict)]:
            name = "**IntegrityNet**" if k == "integritynet" else k
            q = r[k]
            L.append(f"| {s} | {name} | {q['precision']:.2f} | {q['recall']:.2f} | {q['f1']:.2f} | {pct(q['good_fixes_rejected'])} |")
    L += ["", f"Test AUC {i['test']['auc']:.3f}. Share of each fault type flagged on test:", ""]
    L.append("| Injected fault | Fixes | Flagged | Truly faulty (> 12 m / 3 m/s / 30° off) |")
    L.append("|---|---:|---:|---:|")
    for k, q in i["test_by_fault"].items():
        L.append(f"| {k} | {q['n']} | {pct(q['flagged'])} | {pct(q['truly_bad'])} |")
    L.append("")

    d = j(os.path.join(ART, "driftnet.json"))
    L += ["## 5. DriftNet — error radius while dead reckoning", "",
          "Share of outage-seconds whose true error falls inside the radius (ideal: 68 % and 95 %).", "",
          "| Split | Radius from | inside r68 | inside r95 | median r68 | median r95 | quantile loss ↓ |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in ("val", "test"):
        for k, label in (("heuristic", "previous app heuristic"), ("ukf_covariance", "UKF covariance"), ("driftnet", "**DriftNet**")):
            if k in d[s]:
                q = d[s][k]
                L.append(f"| {s} | {label} | {pct(q['cover_68'])} | {pct(q['cover_95'])} | {q['median_r68_m']:.0f} m | {q['median_r95_m']:.0f} m | {q['quantile_loss']:.3f} |")
    L += ["", "Quantile (pinball) loss scores the 50/68/95 % radii together and rewards radii that are both calibrated and tight. "
          "DriftNet's quantiles are conformally calibrated on the validation drives, so val coverage matches by construction; test is the honest check.", ""]

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "models.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
