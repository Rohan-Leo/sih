# Clew — dead-reckoning pipeline (ML + fusion)

The real positioning pipeline behind Clew, trained and benchmarked on the PS-mandated **IO-VNBD** dataset: phone IMU → mount calibration → learned speed (PyTorch → ONNX / TFLite) → UKF with a non-holonomic motion model, ZUPT and online bias correction → HMM map matching.

## Results (held-out drives)

GNSS is cut for 120 s at one-minute intervals while driving. Every method starts from the same GNSS-aided state. Errors are horizontal, measured against the vehicle's GNSS track.

**Test split** — 2.9 h of driving the model never saw, 94 outages. Median error in metres, with median error as a % of distance driven in brackets:

| Method | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| Freeze (typical map app: dot stops) | 97 (100%) | 301 (98%) | 490 (87%) | 772 (78%) |
| Hold last GNSS speed & heading | 21 (22%) | 132 (40%) | 326 (54%) | 802 (72%) |
| IMU integration only | 17 (13%) | 62 (22%) | 135 (24%) | 268 (23%) |
| UKF + ZUPT, no ML | 19 (20%) | 98 (43%) | 235 (53%) | 533 (53%) |
| UKF + ZUPT + learned speed | 16 (19%) | 55 (22%) | 104 (20%) | 187 (17%) |
| **Clew: + online bias correction** | **15 (16%)** | **48 (17%)** | **81 (14%)** | **154 (14%)** |

At 60 s, Clew's mean error is 138 m (vs 187 m for IMU-only) and its 90th percentile is 377 m (vs 383 m). The validation split (4.7 h, 144 outages) shows the same ordering: Clew 77 m median at 60 s, vs 157 m for IMU-only. All features are strictly causal, i.e. computable live. Full tables are in `results/`.

### Read these numbers honestly

- **Map matching isn't in these numbers yet.** The HMM matcher is implemented and tested on a synthetic city (mean error 15.2 m → 6.4 m), but benchmarking it needs OpenStreetMap roads. Run `python -m clew_ml.osm` on a machine with internet, then re-run the benchmark; it adds a "+ HMM map matching" row automatically.
- **The test split is not fully blind.** Online bias correction was added after the first test run exposed a failure: the model read ~14 m/s on a motorway at a true 26 m/s. The fix was tuned on validation only, but test numbers were seen twice.
- **The GNSS input is simulated, the IMU is real.** In most IO-VNBD drives the phone's own GNSS updates only every ~9 s, so the "phone GNSS" here is the vehicle receiver downsampled to 1 Hz with ~3 m correlated noise. The IMU input is 100% the phone's.
- **The phone data is harder than a live app would see.**
  - The IMU is 10 Hz, and the accelerometer is vibration-dominated: longitudinal R² ≈ 0.3 after calibration.
  - That weak accelerometer is why the centripetal non-holonomic update is gated off on most drives (it's used only when lateral calibration R² > 0.6).
  - Android phones deliver 100–200 Hz, which should help the learned model a lot.
- **The no-ML ZUPT baseline does badly on test** because its stationarity heuristic fires falsely at times. The learned stationary head doesn't have this problem.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Data | `clew_ml/data.py` | Loads synchronised phone + vehicle logs and repairs them, then applies a drive-level train/val/test split. See repairs below. |
| Mount calibration | `clew_ml/calib.py` | While GNSS is healthy, regresses phone gyro/accel onto GNSS heading rate, speed change and centripetal acceleration. No axis assumptions. Reports R² per aid. |
| Features | `clew_ml/features.py` | 10 s windows of mount-invariant channels, including the physics feature v = a_lat / ψ̇ (no side-slip). |
| Speed model | `clew_ml/model.py`, `train.py` | Dilated 1D CNN, 51k parameters. Outputs speed + its variance + a stationary flag. Trained with Gaussian NLL, with variance calibrated post hoc (79% of validation errors fall within ±1σ). |
| Fusion | `clew_ml/ukf.py` | UKF with state [E, N, ψ, v, gyro bias, accel bias]. Non-holonomic unicycle motion. Updates: GNSS with innovation gating, ZUPT, learned speed, and an optional centripetal constraint. |
| Bias correction | `evaluate.py: SpeedCorrector` | Learns v_gnss ≈ k·v̂ online while GNSS is good (~2 min memory) and applies it during outages. |
| Map matching | `clew_ml/mapmatch.py` | Online HMM (Newson & Krumm): candidates on nearby road segments, emission from filter σ, transitions from network vs travelled distance. The matched point feeds back into the UKF. |
| Web export | `clew_ml/export_web.py` | Weights as float32 (`public/ml/speednet.bin`), a 14-min held-out drive for the in-app replay demo, and test vectors. The TypeScript port (`src/ml/`) matches PyTorch to 6e-6. |
| Export | `clew_ml/export.py` | `speednet.onnx` and `speednet.tflite` (216 KB each; both match PyTorch to <1e-5; 0.16 ms per inference on CPU), plus `speednet.weights.json`. |

Dataset repairs in `data.py`, each discovered along the way:

- Phone and vehicle streams are offset by several seconds, and in some drives that offset changes mid-recording. Alignment is done per 20-minute chunk by cross-correlating speed.
- The phone IMU is logged ~4–5 s after the phone GNSS. It's aligned separately against vehicle turning.
- Some logs store phone speed in m/s instead of km/h.
- Segments whose aligned IMU doesn't track vehicle turning (|r| < 0.6) are dropped.

Result: 46 usable segments (15 h). The split is 7.3 h train, 4.7 h validation and 2.9 h test.

## Reproduce

```bash
cd ml
pip install -r requirements.txt
./download_iovnbd.sh                    # ~410 MB into data/raw/
python -m clew_ml.preprocess            # align + cache → data/processed/
python -m clew_ml.train                 # ~6 min on a laptop CPU → artifacts/speednet.pt
python -m clew_ml.evaluate --split val  # tune here
python -m clew_ml.evaluate --split test # report here → results/
python -m clew_ml.export                # ONNX (+ TFLite with `pip install litert-torch`)
python -m clew_ml.osm                   # optional: OSM roads for map-matched evaluation
pytest tests/
```

Model input is float32 `(1, 8, 100)`: 10 s of features at 10 Hz, raw, since normalisation is inside the graph. The channel order is in `artifacts/speednet.json`. Output is `[speed m/s, log-variance, stationary logit]`.
