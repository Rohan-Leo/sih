Outage benchmark — IO-VNBD test split (2.9 h of held-out driving, 94 simulated outages)

Median horizontal error (m) after GNSS is lost; in brackets: median error as % of distance driven.

| Method | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| Freeze (typical map app: dot stops) | 97 (100.2%) | 301 (98.0%) | 490 (86.8%) | 772 (77.8%) |
| Hold last GNSS speed & heading | 21 (21.8%) | 132 (40.1%) | 326 (53.8%) | 802 (72.0%) |
| IMU integration only (UKF predict, no aiding) | 17 (13.4%) | 62 (21.8%) | 135 (23.9%) | 268 (23.3%) |
| UKF, non-holonomic model + ZUPT (no ML) | 19 (19.5%) | 98 (42.6%) | 235 (52.9%) | 533 (52.5%) |
| UKF + ZUPT + learned speed, no bias correction | 16 (18.5%) | 55 (22.0%) | 104 (20.3%) | 187 (17.2%) |
| UKF + ZUPT + learned speed + online bias correction (Clew) | 15 (16.2%) | 48 (17.0%) | 81 (13.5%) | 154 (13.5%) |
