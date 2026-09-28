# Clew

**Navigation that keeps its thread through GNSS dead zones.**
Smart India Hackathon 2026 · PS **26168** — *AI-ML based Intelligent Dead Reckoning system for seamless navigation* · Theme: Smart Vehicles · Team **IcarusSipsTea**

Clew is a working map-navigation web app: search a destination, get a real driving route with turn-by-turn directions, and follow yourself along it. When the GNSS fix goes stale, errors out or its accuracy collapses (tunnels, basements, urban canyons), Clew switches to dead reckoning from the phone's own accelerometer, gyroscope and compass — so the dot keeps moving instead of freezing — and blends back smoothly when satellites return.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # static site in dist/
npm run preview      # serve the production build
```

No API keys or sign-ups. The app uses public, keyless services:

| What | Service |
| --- | --- |
| Map tiles | [OpenFreeMap](https://openfreemap.org) (positron / dark styles, recoloured to Clew's palette) |
| Search | [Photon](https://photon.komoot.io) |
| Routing & turn-by-turn | [OSRM demo server](https://project-osrm.org) |

If the basemap can't be reached (offline, restricted venue Wi-Fi), the map falls back to a plain background so the route, thread and position still render.

## Modes

- **Live** — real `navigator.geolocation.watchPosition` + `DeviceMotionEvent` + `DeviceOrientationEvent`. Search → pick a result → **Start navigation**.
- **Demo** — for a judges' table where nobody moves. It has two scenarios:
  - **Recorded drive** (default) replays 14 minutes of a real IO-VNBD test drive that the model never saw: the phone's own accelerometer and gyroscope, plus 1 Hz GNSS. It runs through the **learned engine**. The first 5 minutes calibrate the phone mount, and **Skip mount calibration** jumps past them. After that, GNSS drops in two dead zones, and the dashed line shows where the car really went.
  - **New Delhi route** is a scripted drive on synthetic sensors (`src/demo/demoRoute.json`), run through the heuristic engine.

  Both scenarios go through the **same** estimator, map and UI. The recorded drive also contains a 25 s multipath episode, where GNSS keeps reporting but is wrong, for IntegrityNet to catch. Play/pause, 0.5×–4× speed, two hatched dead zones where the simulated GNSS feed is withheld, and a **Simulate GNSS loss now** button for an outage on cue. Demo mode also shows the error against ground truth, which Live mode can't know.
  Open straight into it with `/?mode=demo`.

### Regenerating the demo route

```bash
npm run bake:demo
```

This asks OSRM for the real route (Vijay Chowk → Kartavya Path → C-Hexagon → Purana Qila, New Delhi), converts it into a timed `{lat, lon, t}` trace with a realistic speed profile, and writes `src/demo/demoRoute.json`. The file currently checked in was generated on a machine without access to OSRM, so it holds a **hand-traced fallback** (`"source": "hand-traced"`). In that case the app tries OSRM from the browser at runtime and caches the real route. Run the bake once with internet access to make the real route permanent.

## Permissions on a phone

Geolocation and motion sensors only work in a **secure context** (HTTPS or `localhost`). To test on a phone, deploy (below) or put the dev server behind an HTTPS tunnel. `npm run dev` already listens on your LAN, but plain `http://192.168.x.x` won't be given sensor access.

The app asks for:

- **Location** — prompted when you first search, tap *Use my location* or start navigation.
- **Motion & Orientation** (iPhone / iPad, iOS 13+) — Safari only allows this from a tap, so it's requested when you press **Start navigation** (or *Enable motion sensors*). Android Chrome grants it without a prompt.

On a laptop there are usually no motion sensors. Clew still works, and says so: during an outage it holds the last speed and course. Desktop browsers also report position only when it changes, so on a laptop a "stale" fix counts as GNSS loss only if there's an error or accuracy degrades. Phones stream about 1 Hz, and there silence does count.

## Deploy (HTTPS)

The app is a static site: `npm run build` writes `dist/`, including the five models (~430 KB) and the replay drive (~600 KB) under `dist/ml/`. It needs no server, API keys or environment variables.

- **Vercel:** *Add New → Project*, import the GitHub repo, keep the detected settings (`vercel.json` sets `npm run build` → `dist`) and deploy. Vercel builds the default branch, so merge into `main` first, or choose the branch under *Settings → Git*.
- **Netlify:** *Add new site → Import an existing project*; `netlify.toml` sets the same build, plus an SPA fallback.
- **CLI instead:** `npx vercel --prod` or `npx netlify deploy --prod --dir dist` from the repo root.

Share `https://<your-app>/?mode=demo` for the judges' demo.

## The learned engine: five models in the browser

`src/ml/` runs the trained pipeline from `ml/` in plain TypeScript. There's no ML runtime: one small runner, `src/ml/tinynet.ts`, executes all five networks. Each network is an encoder (a dilated 1D CNN over a 10 s IMU window, or an MLP over a feature vector) plus a task head.

| # | Model | What it does in the app | Held-out result (test drives) |
|---|---|---|---|
| 1 | **SpeedNet** | forward speed, its uncertainty and a stationary flag → UKF speed updates and ZUPT | median error 60 s into an outage: **80 m** vs 143 m (IMU only) and 335 m (hold last speed/heading) |
| 2 | **HeadingNet** | gyro yaw-rate error → "Gyro err" readout | yaw-rate RMSE 1.07 °/s vs 1.15 °/s for the calibrated gyro. **Not used in the filter**: it didn't reduce outage error, because the UKF already learns the gyro bias from GNSS |
| 3 | **MotionNet** | stationary / cruising / accelerating / braking / turning → "Driving" readout | macro-F1 **0.65** vs 0.56 for hand-tuned rules; braking F1 0.30 vs 0.20 |
| 4 | **IntegrityNet** | P(fault) for every GNSS fix, from how well it agrees with the IMU → rejects multipath, drift, frozen and speed faults | F1 **0.94** vs 0.37 for a UKF-style innovation gate and 0.07 for the previous accuracy-threshold rule; rejects 1 % of good fixes |
| 5 | **DriftNet** | 50 / 68 / 95 % error radius while dead reckoning → the halo and the "95% within" readout | 67 % / 93 % of true errors fall inside the 68 / 95 % radii; quantile loss **0.213** vs 0.252 (UKF covariance) and 0.260 (previous heuristic) |

The learned engine takes over from the heuristic automatically once the phone mount is calibrated, which needs about 5 minutes of driving with GNSS. You'll then see **Engine: LEARNED · UKF + 5 models**. Without motion sensors (most laptops) it stays on the heuristic. SpeedNet is required; if any of the other four fails to load, the app logs it and runs without that model.

Full tables and caveats are in [`ml/results/models.md`](ml/results/models.md) and [`ml/README.md`](ml/README.md).

`npm run check:model` checks the TypeScript port end to end:
- all five networks against PyTorch outputs (max deviation < 1e-5)
- the IntegrityNet and DriftNet feature code against the Python originals
- a headless replay of the recorded drive through both engines

On that drive, the learned engine ends dead zones A and B with 41 m and 17 m of error; the heuristic ends them with 177 m and 102 m.

In the recorded-drive demo, a **multipath episode** (orange hatching on the progress strip) keeps fixes arriving with a confident ±4 m while pulling them about 50 m off the road. IntegrityNet rejects them and the app dead-reckons through; the heuristic follows them off the road. `scripts/scan-fault.ts` slides that episode along the drive. At its median placement, the learned engine ends it 52 m off versus 61 m for following the bad fixes. The gain in position is modest over 25 s, because dead reckoning drifts too. The bigger win is that the bad fixes never reach the filter.

## How the heuristic fallback works — and what it isn't

The fusion logic lives in one swappable module, `src/engine/positionEstimator.ts`:

- **Loss detection:** no fix for 3 s (for streaming receivers), geolocation errors, or reported accuracy jumping well past its recent baseline.
- **Speed:** last GNSS speed. It decays slowly while the accelerometer detects motion and quickly when it reads still, and follows step cadence at walking speeds.
- **Heading:** compass, calibrated against the last GNSS course to absorb phone-mount misalignment, or integrated gyro yaw.
- **"Map matching":** the estimate is held to the active route polyline, with turn anchoring (when the sensors report a turn, it slides to the corner where the route takes that heading).
- **Reacquisition:** eases from the dead-reckoned position back onto the GNSS track and logs outage length and drift.

This is a **lightweight, demo-grade heuristic**, not production-grade dead reckoning. In our submission it stands in for the real pipeline: a learned IMU speed model (PyTorch → TFLite), a UKF with non-holonomic constraints, and HMM road-network map matching. `PositionEstimator` has a small input surface (`pushFix` / `pushMotion` / `pushHeading` / `setRoute` / `tick`), so the trained model can replace its internals without touching the map, UI or sensor code.

## The real pipeline (`ml/`)

[`ml/`](ml/README.md) is where all five models are trained and benchmarked on the PS-mandated IO-VNBD dataset. It covers:
- repairing the dataset
- mount calibration
- the models, exported to the browser (and SpeedNet also to ONNX/TFLite)
- the UKF with a non-holonomic motion model, ZUPT and online bias correction
- HMM map matching

On 2.9 h of held-out drives, the median position error 60 s after GNSS is lost is:

| Method | Median error |
|---|---:|
| Clew pipeline | **80 m** |
| IMU integration alone | 143 m |
| Holding the last speed and heading | 335 m |
| A frozen dot | 490 m |

Map matching isn't in these numbers yet, because it needs OSM road data. See `ml/README.md` for the full table and caveats.

## Project layout

```
src/
  engine/
    positionEstimator.ts  GNSS health + dead reckoning + blend-back (heuristic engine, hooks for the learned one)
    navEngine.ts          clock, sources, route progress, thread, event log
    liveSource.ts         Geolocation / DeviceMotion / DeviceOrientation (+ iOS permission)
    demoSource.ts         scripted GNSS/IMU feed with dead zones
    replaySource.ts       recorded IO-VNBD drive with dead zones and a multipath episode
  ml/
    tinynet.ts            one runner for all five networks
    learnedEstimator.ts   UKF + the five models wired into the estimator
    speednet.ts, imuModels.ts, integrity.ts, drift.ts   per-model wrappers and live feature code
    calib.ts, ukf.ts      online mount calibration, UKF
  services/               Photon search, OSRM routing + instruction text
  components/             map, search, turn banner, status badge, console, About drawer
  demo/                   baked demo route + shared timeline generator
scripts/bake-demo-route.mjs
```

Map data © OpenStreetMap contributors.
