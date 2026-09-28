Outage benchmark — IO-VNBD test split (2.9 h of held-out driving, 94 simulated outages)

Median horizontal error (m) after GNSS is lost; in brackets: median error as % of distance driven.

| Method | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| Freeze (typical map app: dot stops) | 97 (99.9%) | 302 (98.1%) | 490 (86.9%) | 771 (77.7%) |
| Hold last GNSS speed & heading | 22 (22.5%) | 131 (40.6%) | 335 (54.6%) | 808 (71.7%) |
| IMU integration only (UKF predict, no aiding) | 16 (12.7%) | 64 (20.2%) | 143 (25.0%) | 282 (22.9%) |
| UKF, non-holonomic model + ZUPT (no ML) | 21 (23.1%) | 98 (42.5%) | 236 (51.9%) | 527 (52.8%) |
| UKF + ZUPT + learned speed, no bias correction | 16 (18.3%) | 53 (23.2%) | 95 (19.3%) | 167 (17.2%) |
| UKF + ZUPT + learned speed + online bias correction (Clew) | 15 (15.9%) | 49 (16.8%) | 80 (13.4%) | 154 (12.6%) |
| Clew + HeadingNet gyro-bias measurement | 15 (16.1%) | 50 (16.0%) | 81 (14.7%) | 150 (14.7%) |
