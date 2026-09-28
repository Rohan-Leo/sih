# Clew's five models — held-out results

All numbers are on IO-VNBD drives the models never trained on: *val* (4.7 h) tunes, *test* (2.9 h) reports.

## 1. SpeedNet — learned speed (drives the whole outage benchmark)

Median position error after GNSS is lost (m):

| Split | Method | 10 s | 30 s | 60 s | 120 s |
|---|---|---:|---:|---:|---:|
| val | IMU integration only | 14 | 66 | 150 | 350 |
| val | UKF, no ML | 14 | 70 | 158 | 404 |
| val | **Clew (UKF + SpeedNet + bias correction)** | 14 | 39 | 76 | 169 |
| test | IMU integration only | 16 | 64 | 143 | 282 |
| test | UKF, no ML | 21 | 98 | 236 | 527 |
| test | **Clew (UKF + SpeedNet + bias correction)** | 15 | 49 | 80 | 154 |

## 2. HeadingNet — gyro yaw-rate error

| Split | Yaw-rate RMSE, calibrated gyro | with HeadingNet | errors within ±1σ |
|---|---:|---:|---:|
| val | 1.30 °/s | 0.98 °/s | 73% |
| test | 1.15 °/s | 1.07 °/s | 78% |

In the outage benchmark, as a gyro-bias measurement in the UKF (median m at 10 / 30 / 60 / 120 s):

- val: Clew 14 / 39 / 76 / 169 → with HeadingNet 15 / 40 / 76 / 180
- test: Clew 15 / 49 / 80 / 154 → with HeadingNet 15 / 50 / 81 / 150

It predicts the gyro error better than the raw calibration, but the UKF already learns the gyro bias from GNSS before an outage, so the outage error doesn't improve. The export step turns it on in the filter only if the validation benchmark improves by ≥ 2 % at 60 s without getting worse at 120 s. Currently: **off**; the app still shows its gyro-error estimate.

## 3. MotionNet — driving state

| Split | Method | Macro-F1 | stationary | cruising | accelerating | braking | turning |
|---|---|---:|---:|---:|---:|---:|---:|
| val | hand-tuned rules | 0.61 | 0.86 | 0.83 | 0.21 | 0.28 | 0.86 |
| val | **MotionNet** | **0.77** | 0.90 | 0.85 | 0.59 | 0.62 | 0.87 |
| test | hand-tuned rules | 0.56 | 0.70 | 0.84 | 0.22 | 0.20 | 0.86 |
| test | **MotionNet** | **0.65** | 0.88 | 0.85 | 0.39 | 0.30 | 0.84 |

## 4. IntegrityNet — GNSS fault detection

Faults injected into the 1 Hz fix stream of held-out drives (jumps, drifts, noise bursts, frozen fixes, speed faults, outages).

| Split | Detector | Precision | Recall | F1 | Good fixes rejected |
|---|---|---:|---:|---:|---:|
| val | **IntegrityNet** | 0.96 | 0.97 | 0.96 | 1% |
| val | accuracy threshold (previous app rule) | 1.00 | 0.04 | 0.08 | 0% |
| val | innovation gate (5σ, UKF-style) | 0.98 | 0.38 | 0.55 | 0% |
| test | **IntegrityNet** | 0.94 | 0.94 | 0.94 | 1% |
| test | accuracy threshold (previous app rule) | 0.95 | 0.03 | 0.07 | 0% |
| test | innovation gate (5σ, UKF-style) | 0.96 | 0.23 | 0.37 | 0% |

Test AUC 0.995. Share of each fault type flagged on test:

| Injected fault | Fixes | Flagged | Truly faulty (> 12 m / 3 m/s / 30° off) |
|---|---:|---:|---:|
| clean | 5143 | 1% | 0% |
| jump | 354 | 97% | 100% |
| drift | 240 | 67% | 77% |
| noise | 360 | 93% | 90% |
| frozen | 89 | 89% | 91% |
| speed | 147 | 83% | 84% |

## 5. DriftNet — error radius while dead reckoning

Share of outage-seconds whose true error falls inside the radius (ideal: 68 % and 95 %).

| Split | Radius from | inside r68 | inside r95 | median r68 | median r95 | quantile loss ↓ |
|---|---|---:|---:|---:|---:|---:|
| val | previous app heuristic | 83% | 94% | 138 m | 223 m | 0.232 |
| val | UKF covariance | 51% | 74% | 71 m | 114 m | 0.237 |
| val | **DriftNet** | 68% | 95% | 94 m | 204 m | 0.197 |
| test | previous app heuristic | 79% | 90% | 145 m | 236 m | 0.260 |
| test | UKF covariance | 54% | 75% | 75 m | 122 m | 0.252 |
| test | **DriftNet** | 67% | 93% | 98 m | 213 m | 0.213 |

Quantile (pinball) loss scores the 50/68/95 % radii together and rewards radii that are both calibrated and tight. DriftNet's quantiles are conformally calibrated on the validation drives, so val coverage matches by construction; test is the honest check.
