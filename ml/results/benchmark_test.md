Outage benchmark — IO-VNBD test split (2.9 h of held-out driving, 94 simulated outages)

Median horizontal error (m) after GNSS is lost; in brackets: median error as % of distance driven.

| Method | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| Freeze (typical map app: dot stops) | 97 (100.1%) | 302 (98.0%) | 490 (86.8%) | 772 (77.8%) |
| Hold last GNSS speed & heading | 21 (21.8%) | 132 (40.1%) | 326 (53.8%) | 801 (72.0%) |
| IMU integration only (UKF predict, no aiding) | 17 (13.5%) | 63 (21.8%) | 135 (23.9%) | 267 (23.4%) |
| UKF, non-holonomic model + ZUPT (no ML) | 19 (19.5%) | 98 (42.6%) | 235 (52.9%) | 538 (52.5%) |
| UKF + ZUPT + learned speed, no bias correction | 17 (19.2%) | 61 (24.1%) | 111 (21.0%) | 219 (18.9%) |
| UKF + ZUPT + learned speed + online bias correction (Clew) | 15 (17.0%) | 49 (17.6%) | 80 (14.1%) | 161 (13.8%) |
