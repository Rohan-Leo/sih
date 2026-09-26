Outage benchmark — IO-VNBD val split (4.7 h of held-out driving, 144 simulated outages)

Median horizontal error (m) after GNSS is lost; in brackets: median error as % of distance driven.

| Method | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| Freeze (typical map app: dot stops) | 90 (99.8%) | 235 (96.0%) | 403 (84.8%) | 703 (72.7%) |
| Hold last GNSS speed & heading | 26 (23.9%) | 147 (61.3%) | 412 (82.8%) | 886 (89.0%) |
| IMU integration only (UKF predict, no aiding) | 14 (16.8%) | 64 (25.8%) | 157 (30.6%) | 351 (30.8%) |
| UKF, non-holonomic model + ZUPT (no ML) | 14 (16.9%) | 67 (26.8%) | 166 (35.6%) | 400 (37.9%) |
| UKF + ZUPT + learned speed, no bias correction | 13 (14.5%) | 40 (17.9%) | 83 (18.4%) | 180 (20.2%) |
| UKF + ZUPT + learned speed + online bias correction (Clew) | 15 (14.0%) | 40 (13.9%) | 77 (14.2%) | 166 (16.0%) |
