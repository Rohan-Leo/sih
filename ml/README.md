# Clew — dead-reckoning pipeline (ML + fusion)

The real positioning pipeline behind Clew, trained and benchmarked on the PS-mandated **IO-VNBD** dataset: phone IMU → mount calibration → learned speed (PyTorch → ONNX / TFLite) → UKF with a non-holonomic motion model, ZUPT and online bias correction → HMM map matching. Four more models sit around that core: HeadingNet (gyro error), MotionNet (driving state), IntegrityNet (GNSS fault detection) and DriftNet (error radius). See [Five models](#five-models).

## Results (held-out drives)

GNSS is cut for 120 s at one-minute intervals while driving. Every method starts from the same GNSS-aided state. Errors are horizontal, measured against the vehicle's GNSS track.

**Test split** — 2.9 h of driving the model never saw, 94 outages. Median error in metres, with median error as a % of distance driven in brackets:

| Method | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| Freeze (typical map app: dot stops) | 97 (100%) | 302 (98%) | 490 (87%) | 771 (78%) |
| Hold last GNSS speed & heading | 22 (23%) | 131 (41%) | 335 (55%) | 808 (72%) |
| IMU integration only | 16 (13%) | 64 (20%) | 143 (25%) | 282 (23%) |
| UKF + ZUPT, no ML | 21 (23%) | 98 (43%) | 236 (52%) | 527 (53%) |
| UKF + ZUPT + learned speed | 16 (18%) | 53 (23%) | 95 (19%) | 167 (17%) |
| **Clew: + online bias correction** | **15 (16%)** | **49 (17%)** | **80 (13%)** | **154 (13%)** |
| Clew + HeadingNet as gyro-bias measurement | 15 (16%) | 50 (16%) | 81 (15%) | 150 (15%) |

At 60 s, Clew's mean error is 136 m (vs 186 m for IMU-only) and its 90th percentile is 374 m (vs 380 m). The validation split (4.7 h, 144 outages) shows the same ordering: Clew 76 m median at 60 s, vs 150 m for IMU-only. All features are strictly causal, i.e. computable live. Full tables are in `results/`.

### Read these numbers honestly

- **Map matching isn't in these numbers yet.** The HMM matcher is implemented and tested on a synthetic city (mean error 15.2 m → 6.4 m), but benchmarking it needs OpenStreetMap roads. Run `python -m clew_ml.osm` on a machine with internet, then re-run the benchmark; it adds a "+ HMM map matching" row automatically.
- **The test split is not fully blind.** Online bias correction was added after the first test run exposed a failure: the model read ~14 m/s on a motorway at a true 26 m/s. The fix was tuned on validation only, but test numbers were seen twice.
- **The GNSS input is simulated, the IMU is real.** In most IO-VNBD drives the phone's own GNSS updates only every ~9 s, so the "phone GNSS" here is the vehicle receiver downsampled to 1 Hz with ~3 m correlated noise. The IMU input is 100% the phone's.
- **The phone data is harder than a live app would see.**
  - The IMU is 10 Hz, and the accelerometer is vibration-dominated: longitudinal R² ≈ 0.3 after calibration.
  - That weak accelerometer is why the centripetal non-holonomic update is gated off on most drives (it's used only when lateral calibration R² > 0.6).
  - Android phones deliver 100–200 Hz, which should help the learned model a lot.
- **The no-ML ZUPT baseline does badly on test** because its stationarity heuristic fires falsely at times. The learned stationary head doesn't have this problem.
- **Numbers move by a few metres between preprocessing runs.** The simulated GNSS noise is seeded with Python's `hash()` of the drive name, which is randomised per process, so re-running `preprocess` redraws it. The ordering of methods has been stable across runs.

## Five models

Each model is an encoder plus a task head, in one shared layout (`clew_ml/nets.py`), so the browser runs all of them with one ~150-line TypeScript runner (`src/ml/tinynet.ts`). The encoder is a dilated 1D CNN (TCN) over the 10 s IMU feature window, or an MLP over a feature vector. Full results: [`results/models.md`](results/models.md).

| Model | Encoder → head | Trained on | Test result | In the app |
|---|---|---|---|---|
| **SpeedNet** (`model.py`, `train.py`) | TCN, 51k params → speed, log-variance, stationary logit | wheel-speed labels | outage error at 60 s: 80 m vs 143 m IMU-only | UKF speed updates, ZUPT |
| **HeadingNet** (`imu_models.py heading`) | TCN, 24k → δ yaw rate, log-variance | true − calibrated yaw rate over 10 s | yaw RMSE 1.07 vs 1.15 °/s; outage error unchanged (81 vs 80 m) | readout only; kept out of the filter by the export gate |
| **MotionNet** (`imu_models.py motion`) | TCN, 24k → 5 classes | states labelled from vehicle speed and heading | macro-F1 0.65 vs 0.56 rules | "Driving" readout |
| **IntegrityNet** (`integrity.py`) | MLP, 5k → P(fault) | 9 consistency features per fix, with injected faults | F1 0.94 vs 0.37 innovation gate; 1 % of good fixes rejected | rejects faulty fixes before they reach the UKF |
| **DriftNet** (`drift.py`) | MLP, 5k → 50/68/95 % log-error quantiles | 55k outage-seconds from simulated outages on training drives | coverage 67 % / 93 %; quantile loss 0.213 vs 0.252 UKF covariance | halo radius, "95% within" readout |

What didn't work, kept for the record:

- **HeadingNet as a direct gyro correction** (adding δ to the gyro) made 60 s error worse on validation: 89–93 m vs 76 m. As a UKF measurement of the gyro bias it is neutral (76 vs 76 m, and worse at 120 s). The UKF already learns the bias from GNSS before an outage, so the learned estimate adds little. The export step (`export_web.heading_helps`) turns it on only if validation improves ≥ 2 % at 60 s without getting worse at 120 s.
- **DriftNet with per-drive features** (mount-calibration R², speed-correction residual, the speed model's σ) memorised the ~20 training drives: test quantile loss was 0.34, worse than both baselines. The six features it uses now are all outage-local.
- **DriftNet without calibration** under-covers on unseen drives. The quantiles are shifted by conformal offsets fitted on validation drives.

IntegrityNet's caveats:

- **Faults are injected, not recorded.** IO-VNBD has no labelled multipath, so the fault types, magnitudes and rates are ours. Drift faults are the hardest: 67 % flagged, and some aren't bad yet when they start.
- **It is more permissive after a long gap.** The next fix after an outage is usually good, and the model learned that. In the replay demo, two faulty fixes got through at the end of the 25 s multipath episode. The app protects against the fallout two ways:
  - a fix trusted after a > 10 s gap is *provisional*, and is rolled back if the next fixes disagree with it but agree with the view before it
  - the UKF is restarted at the fix after 3 trusted fixes in a row fail its innovation gate

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
| Web export | `clew_ml/export_web.py` | All five models as float32 blobs + manifests (`public/ml/<model>.{bin,json}`), a 14-min held-out drive (with a multipath episode) for the in-app replay demo, and test vectors for the networks and the live feature code. The TypeScript port (`src/ml/`) matches PyTorch to < 1e-5. |
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
python -m clew_ml.train                 # SpeedNet, ~6 min on a laptop CPU → artifacts/speednet.pt
python -m clew_ml.imu_models heading    # HeadingNet   → artifacts/headingnet.pt
python -m clew_ml.imu_models motion     # MotionNet    → artifacts/motionnet.pt
python -m clew_ml.integrity             # IntegrityNet → artifacts/integritynet.pt
python -m clew_ml.evaluate --split val  # tune here (also decides whether HeadingNet goes in the filter)
python -m clew_ml.evaluate --split test # report here → results/
python -m clew_ml.drift                 # DriftNet     → artifacts/driftnet.pt (simulates outages; slow the first time)
python -m clew_ml.export_web            # all five models + replay drive → ../public/ml, test vectors → ../src/ml
python -m clew_ml.report                # → results/models.md
python -m clew_ml.export                # SpeedNet as ONNX (+ TFLite with `pip install litert-torch`)
python -m clew_ml.osm                   # optional: OSM roads for map-matched evaluation
pytest tests/
```

SpeedNet, HeadingNet and MotionNet take float32 `(1, 8, 100)`: 10 s of features at 10 Hz, raw, since normalisation is inside the graph. The channel order is in `artifacts/speednet.json`. SpeedNet outputs `[speed m/s, log-variance, stationary logit]`. IntegrityNet and DriftNet take the feature vectors listed in `artifacts/{integritynet,driftnet}.json`. Then run `npm run check:model` from the repo root to verify the browser port.
