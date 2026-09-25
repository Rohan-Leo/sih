Outage benchmark — IO-VNBD val split (4.7 h of held-out driving, 144 simulated outages)

Median horizontal error (m) after GNSS is lost; in brackets: median error as % of distance driven.

| Method | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| Freeze (typical map app: dot stops) | 90 (99.8%) | 235 (96.0%) | 403 (84.8%) | 703 (72.7%) |
| Hold last GNSS speed & heading | 26 (24.0%) | 147 (61.4%) | 412 (82.8%) | 886 (88.9%) |
| IMU integration only (UKF predict, no aiding) | 14 (16.8%) | 65 (25.9%) | 160 (30.2%) | 357 (30.2%) |
| UKF, non-holonomic model + ZUPT (no ML) | 14 (16.9%) | 68 (27.0%) | 166 (36.2%) | 402 (37.9%) |
| UKF + ZUPT + learned speed, no bias correction | 13 (13.7%) | 43 (17.2%) | 86 (17.3%) | 170 (18.7%) |
| UKF + ZUPT + learned speed + online bias correction (Clew) | 13 (14.4%) | 39 (13.8%) | 72 (14.0%) | 160 (16.0%) |
