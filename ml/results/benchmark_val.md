Outage benchmark — IO-VNBD val split (4.7 h of held-out driving, 144 simulated outages)

Median horizontal error (m) after GNSS is lost; in brackets: median error as % of distance driven.

| Method | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| Freeze (typical map app: dot stops) | 92 (99.9%) | 234 (95.8%) | 402 (85.1%) | 704 (72.7%) |
| Hold last GNSS speed & heading | 26 (27.0%) | 144 (61.2%) | 418 (84.0%) | 892 (88.7%) |
| IMU integration only (UKF predict, no aiding) | 14 (16.1%) | 66 (25.0%) | 150 (30.2%) | 350 (29.9%) |
| UKF, non-holonomic model + ZUPT (no ML) | 14 (16.4%) | 70 (25.7%) | 158 (36.7%) | 404 (37.7%) |
| UKF + ZUPT + learned speed, no bias correction | 13 (14.1%) | 40 (17.1%) | 83 (19.3%) | 174 (20.1%) |
| UKF + ZUPT + learned speed + online bias correction (Clew) | 14 (13.4%) | 39 (13.9%) | 76 (14.5%) | 169 (16.4%) |
| Clew + HeadingNet gyro-bias measurement | 15 (13.7%) | 40 (14.6%) | 76 (16.0%) | 180 (17.7%) |
