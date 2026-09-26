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

  Both scenarios go through the **same** estimator, map and UI. Play/pause, 0.5×–4× speed, two hatched dead zones where the simulated GNSS feed is withheld, and a **Simulate GNSS loss now** button for an outage on cue. Demo mode also shows the error against ground truth, which Live mode can't know.
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

- **Vercel:** import the repo; `vercel.json` sets `npm run build` → `dist`.
- **Netlify:** import the repo; `netlify.toml` sets the same, plus an SPA fallback.

## The learned engine (in the browser)

`src/ml/` is a TypeScript port of the trained pipeline from `ml/`:
- online phone-mount calibration
- the speed network, loaded from `public/ml/speednet.bin` (201 KB) and run once per second in about 20 ms, with no ML runtime
- a UKF with zero-velocity updates and online speed-bias correction

It takes over from the heuristic automatically once the mount is calibrated, which needs about 5 minutes of driving with GNSS. You'll see **Engine: LEARNED** under the readouts. Without motion sensors (most laptops) it stays on the heuristic.

`npm run check:model` checks the TypeScript network against PyTorch outputs, then replays the recorded drive through both engines headlessly. On that drive, the learned engine ends dead zones A and B with 42 m and 5 m of error; the heuristic ends them with 183 m and 101 m.

## How the heuristic fallback works — and what it isn't

The fusion logic lives in one swappable module, `src/engine/positionEstimator.ts`:

- **Loss detection:** no fix for 3 s (for streaming receivers), geolocation errors, or reported accuracy jumping well past its recent baseline.
- **Speed:** last GNSS speed. It decays slowly while the accelerometer detects motion and quickly when it reads still, and follows step cadence at walking speeds.
- **Heading:** compass, calibrated against the last GNSS course to absorb phone-mount misalignment, or integrated gyro yaw.
- **"Map matching":** the estimate is held to the active route polyline, with turn anchoring (when the sensors report a turn, it slides to the corner where the route takes that heading).
- **Reacquisition:** eases from the dead-reckoned position back onto the GNSS track and logs outage length and drift.

This is a **lightweight, demo-grade heuristic**, not production-grade dead reckoning. In our submission it stands in for the real pipeline: a learned IMU speed model (PyTorch → TFLite), a UKF with non-holonomic constraints, and HMM road-network map matching. `PositionEstimator` has a small input surface (`pushFix` / `pushMotion` / `pushHeading` / `setRoute` / `tick`), so the trained model can replace its internals without touching the map, UI or sensor code.

## The real pipeline (`ml/`)

The in-app fallback is a lightweight heuristic; the actual dead-reckoning pipeline lives in [`ml/`](ml/README.md). It covers IO-VNBD data repair, mount calibration, a learned speed model exported to ONNX/TFLite, a UKF with a non-holonomic motion model, ZUPT and online bias correction, and HMM map matching.

On 2.9 h of held-out drives, after 60 s without GNSS, the median position error is:

| Method | Median error |
|---|---:|
| Clew pipeline | **81 m** |
| IMU integration alone | 135 m |
| Holding the last speed and heading | 326 m |
| A frozen dot | 490 m |

Map matching isn't in these numbers yet, because it needs OSM road data. See `ml/README.md` for the full table and caveats.

## Project layout

```
src/
  engine/
    positionEstimator.ts  GNSS health + dead reckoning + blend-back (swap point for the real model)
    navEngine.ts          clock, sources, route progress, thread, event log
    liveSource.ts         Geolocation / DeviceMotion / DeviceOrientation (+ iOS permission)
    demoSource.ts         scripted GNSS/IMU feed with dead zones
  services/               Photon search, OSRM routing + instruction text
  components/             map, search, turn banner, status badge, console, About drawer
  demo/                   baked demo route + shared timeline generator
scripts/bake-demo-route.mjs
```

Map data © OpenStreetMap contributors.
