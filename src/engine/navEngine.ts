/**
 * NavEngine — owns the estimator, the active sensor source (Live or Demo),
 * the route and navigation progress, the thread (trail) and the event log.
 *
 * Live and Demo differ ONLY in where sensor readings come from and which
 * clock stamps them; everything downstream is shared.
 */
import { type LngLat, Polyline, haversine } from '../lib/geo'
import { type Route, type RouteStep, fetchRoute, fromOsrm } from '../services/osrm'
import type { EstimatorState, FixMode } from './positionEstimator'
import { DemoSource } from './demoSource'
import { LiveSource, type LiveStatus } from './liveSource'
import { ReplaySource } from './replaySource'
import type { DeadZone, ScriptedSource, SensorSink } from './sources'
import { LearnedEstimator } from '../ml/learnedEstimator'
import { SpeedNet } from '../ml/speednet'
import demoData from '../demo/demoRoute.json'
import { buildTimeline } from '../demo/timeline.js'

export type AppMode = 'live' | 'demo'
/** recorded: a real held-out IO-VNBD drive (real phone IMU); delhi: synthetic sensors on a baked route */
export type DemoScenario = 'recorded' | 'delhi'
const SCENARIO_KEY = 'clew.demoScenario'
const ML_BASE = `${import.meta.env.BASE_URL}ml/`

export interface ActiveRoute {
  route: Route
  line: Polyline
  steps: RouteStep[]
  destination: LngLat
}

export interface Progress {
  along: number
  remaining: number
  remainingTime: number
  etaClock: Date
  nextStepIdx: number
  nextStep: RouteStep | null
  thenStep: RouteStep | null
  distToManeuver: number
  offRoute: number
  arrived: boolean
}

export interface LogEntry {
  id: number
  at: Date
  kind: 'info' | 'lost' | 'reacquired' | 'warn' | 'route'
  message: string
}

export interface TrailSegment {
  kind: 'gnss' | 'dr'
  coords: LngLat[]
}

export interface DemoInfo {
  playing: boolean
  rate: number
  simT: number
  duration: number
  zone: DeadZone | null
  withheld: boolean
  manualRemaining: number
  truthError: number | null
  source: string
  deadZones: DeadZone[]
  pathLength: number
  truthAlong: number
  origin: string
  destination: string
  scenario: DemoScenario
  /** sim time (ms) at which the recorded drive's mount calibration completes */
  calibrationEnd: number | null
}

export interface Snapshot {
  mode: AppMode
  est: EstimatorState
  route: ActiveRoute | null
  navigating: boolean
  progress: Progress | null
  demo: DemoInfo | null
  live: LiveStatus
  log: LogEntry[]
  routing: boolean
  routeError: string | null
  demoScenario: DemoScenario
  demoLoading: boolean
  version: number
}

export interface Frame {
  est: EstimatorState
  truth: LngLat | null
}

interface DemoFile {
  name: string
  source: string
  origin: { name: string; coord: number[] }
  destination: { name: string; coord: number[] }
  waypoints: number[][]
  deadZones: DeadZone[]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  osrm: any
  samples: { lat: number; lon: number; t: number }[]
}

const DEMO_CACHE_KEY = 'clew.demoRoute.osrm.v1'
const TRAIL_MIN_STEP = 2.5
const TRAIL_MAX_POINTS = 6000

export class NavEngine {
  mode: AppMode = 'live'
  readonly estimator = new LearnedEstimator()
  readonly live: LiveSource
  private demo: ScriptedSource | null = null
  private demoFile: DemoFile = demoData as DemoFile
  scenario: DemoScenario = readScenario()
  demoLoading = false
  private replay: Promise<ReplaySource> | null = null
  private loadToken = 0

  route: ActiveRoute | null = null
  navigating = false
  routing = false
  routeError: string | null = null

  trail: TrailSegment[] = []
  truthTrail: LngLat[] = []
  trailVersion = 0
  log: LogEntry[] = []
  private logId = 0

  // demo clock
  simT = 0
  playing = false
  rate = 1

  private raf = 0
  private lastRealT = 0
  private lastEst: EstimatorState
  private lastProgress: Progress | null = null
  private arrivedLogged = false
  private offRouteSince: number | null = null
  private rerouteInFlight = false

  private frameListeners = new Set<(f: Frame) => void>()
  private snapListeners = new Set<() => void>()
  private snapshot: Snapshot
  private version = 0
  private lastPublish = 0
  private dirty = true

  private sink: SensorSink = {
    pushFix: (f) => this.estimator.pushFix(f),
    pushMotion: (m) => this.estimator.pushMotion(m),
    pushHeading: (h) => this.estimator.pushHeading(h),
    gnssError: (r) => this.estimator.noteGnssError(r),
  }

  constructor() {
    this.live = new LiveSource(() => (this.dirty = true))
    this.lastEst = this.estimator.tick(0)
    this.snapshot = this.buildSnapshot()
    SpeedNet.load(ML_BASE)
      .then((net) => {
        this.estimator.net = net
        this.dirty = true
      })
      .catch(() => this.addLog('warn', 'Speed model failed to load — using the heuristic fallback'))
    this.estimator.onEvent((e) => {
      const kind: LogEntry['kind'] =
        e.kind === 'lost' ? 'lost' : e.kind === 'reacquired' ? 'reacquired' : e.kind === 'acquired' ? 'info' : 'warn'
      this.addLog(kind, e.message)
    })
  }

  // ── clock ─────────────────────────────────────────────────────────────────

  now(): number {
    return this.mode === 'demo' ? this.simT : performance.now()
  }

  start() {
    const loop = (t: number) => {
      this.raf = requestAnimationFrame(loop)
      this.step(t)
    }
    this.lastRealT = performance.now()
    this.raf = requestAnimationFrame(loop)
  }

  destroy() {
    cancelAnimationFrame(this.raf)
    this.live.stop()
  }

  private step(realT: number) {
    const realDt = Math.min(250, realT - this.lastRealT)
    this.lastRealT = realT

    if (this.mode === 'demo' && this.demo) {
      if (this.playing) {
        const end = this.demo.duration + 4000
        const target = Math.min(end, this.simT + realDt * this.rate)
        // Sub-step so slow frames at 4× don't hand the estimator huge dt jumps.
        while (this.simT < target) {
          this.simT = Math.min(target, this.simT + 50)
          this.demo.advance(this.simT, this.sink)
          if (this.simT < target) this.estimator.tick(this.simT)
        }
        if (this.simT >= end) {
          this.playing = false
          this.dirty = true
        }
      }
      this.demo.advance(this.simT, this.sink)
    }

    const t = this.now()
    const est = this.estimator.tick(t)
    this.lastEst = est
    this.updateTrail(est)
    this.lastProgress = this.computeProgress(est, t)
    let truth: LngLat | null = null
    if (this.mode === 'demo' && this.demo) {
      truth = this.demo.truthAt(this.simT).pos
      const last = this.truthTrail[this.truthTrail.length - 1]
      if (!last || haversine(last, truth) > TRAIL_MIN_STEP) this.truthTrail.push(truth)
    }

    this.frameListeners.forEach((l) => l({ est, truth }))

    // React snapshot at ~8 Hz (or immediately on discrete changes)
    if (this.dirty || realT - this.lastPublish > 120) {
      this.dirty = false
      this.lastPublish = realT
      this.version++
      this.snapshot = this.buildSnapshot()
      this.snapListeners.forEach((l) => l())
    }
  }

  // ── subscriptions ─────────────────────────────────────────────────────────

  onFrame(cb: (f: Frame) => void): () => void {
    this.frameListeners.add(cb)
    return () => this.frameListeners.delete(cb)
  }

  subscribe = (cb: () => void): (() => void) => {
    this.snapListeners.add(cb)
    return () => this.snapListeners.delete(cb)
  }

  getSnapshot = (): Snapshot => this.snapshot

  private buildSnapshot(): Snapshot {
    let demo: DemoInfo | null = null
    if (this.mode === 'demo' && this.demo) {
      const truth = this.demo.truthAt(this.simT)
      demo = {
        playing: this.playing,
        rate: this.rate,
        simT: this.simT,
        duration: this.demo.duration,
        zone: this.demo.deadZoneAt(this.simT),
        withheld: this.demo.gnssWithheld(this.simT),
        manualRemaining: this.demo.manualOutageRemaining(this.simT),
        truthError: this.demo.errorAgainstTruth(this.lastEst.position, this.simT),
        source: this.scenario === 'delhi' ? this.demoFile.source : 'recorded',
        deadZones: this.demo.deadZones,
        pathLength: this.demo.path.length,
        truthAlong: truth.along,
        origin: this.scenario === 'delhi' ? this.demoFile.origin.name : 'Recorded drive · Coventry, UK',
        destination: this.scenario === 'delhi' ? this.demoFile.destination.name : 'IO-VNBD held-out test drive',
        scenario: this.scenario,
        calibrationEnd: this.demo instanceof ReplaySource ? this.demo.manifest.calibrationSamples * 100 : null,
      }
    }
    return {
      mode: this.mode,
      est: this.lastEst,
      route: this.route,
      navigating: this.navigating,
      progress: this.lastProgress,
      demo,
      live: this.live.status,
      log: this.log,
      routing: this.routing,
      demoScenario: this.scenario,
      demoLoading: this.demoLoading,
      routeError: this.routeError,
      version: this.version,
    }
  }

  private addLog(kind: LogEntry['kind'], message: string) {
    this.log = [{ id: ++this.logId, at: new Date(), kind, message }, ...this.log].slice(0, 60)
    this.dirty = true
  }

  // ── trail (the thread) ────────────────────────────────────────────────────

  private updateTrail(est: EstimatorState) {
    if (!est.position) return
    const kind: TrailSegment['kind'] = est.mode === 'DR' ? 'dr' : 'gnss'
    let seg = this.trail[this.trail.length - 1]
    if (!seg || seg.kind !== kind) {
      const joint = seg?.coords[seg.coords.length - 1]
      seg = { kind, coords: joint ? [joint] : [] }
      this.trail.push(seg)
    }
    const last = seg.coords[seg.coords.length - 1]
    if (!last || haversine(last, est.position) >= TRAIL_MIN_STEP) {
      seg.coords.push(est.position)
      this.trailVersion++
      let total = this.trail.reduce((a, s) => a + s.coords.length, 0)
      while (total > TRAIL_MAX_POINTS && this.trail.length) {
        const first = this.trail[0]
        first.coords.shift()
        total--
        if (first.coords.length < 2 && this.trail.length > 1) {
          total -= first.coords.length
          this.trail.shift()
        }
      }
    }
  }

  clearTrail() {
    this.trail = []
    this.truthTrail = []
    this.trailVersion++
  }

  // ── routing & progress ────────────────────────────────────────────────────

  private setRoute(route: Route, destination: LngLat) {
    const line = new Polyline(route.coords)
    let prev = 0
    const steps = route.steps.map((s, i) => {
      let at = i === 0 ? 0 : line.project(s.maneuver.location, prev, 3000).along
      if (s.maneuver.type === 'arrive') at = line.length
      at = Math.max(prev, at)
      prev = at
      return { ...s, at }
    })
    this.route = { route, line, steps, destination }
    this.estimator.setRoute(line)
    this.arrivedLogged = false
    this.offRouteSince = null
    this.dirty = true
  }

  /** Request a route from the current estimate to `dest` (Live mode). */
  async routeTo(dest: LngLat, name: string): Promise<void> {
    const from = this.lastEst.position
    if (!from) {
      this.routeError = 'Waiting for your location — allow location access, then try again.'
      this.dirty = true
      return
    }
    this.routing = true
    this.routeError = null
    this.dirty = true
    try {
      const r = await fetchRoute([from, dest], name)
      if (this.mode !== 'live') return
      this.setRoute(r, dest)
      this.addLog('route', `Route to ${name}: ${(r.distance / 1000).toFixed(1)} km, ${Math.round(r.duration / 60)} min`)
    } catch (e) {
      this.routeError = e instanceof Error ? e.message : 'Routing failed'
    } finally {
      this.routing = false
      this.dirty = true
    }
  }

  clearRoute() {
    this.route = null
    this.navigating = false
    this.estimator.setRoute(null)
    this.routeError = null
    this.dirty = true
  }

  startNavigation() {
    if (!this.route) return
    this.navigating = true
    this.arrivedLogged = false
    this.clearTrail()
    this.addLog('info', `Navigation started — ${this.route.route.destinationName}`)
  }

  stopNavigation() {
    this.navigating = false
    this.dirty = true
    this.addLog('info', 'Navigation ended')
  }

  private computeProgress(est: EstimatorState, t: number): Progress | null {
    const r = this.route
    if (!r || !est.position) return null
    const along = Math.min(r.line.length, this.estimator.routeAlong)
    const onLine = r.line.pointAt(along).point
    const offRoute = haversine(onLine, est.position)
    const remaining = Math.max(0, r.line.length - along)
    let idx = r.steps.findIndex((s, i) => i > 0 && s.at > along + 8)
    if (idx === -1) idx = r.steps.length - 1
    const nextStep = r.steps[idx] ?? null
    const thenStep = r.steps[idx + 1] ?? null
    const avg = r.route.distance / Math.max(1, r.route.duration)
    const v = est.speed > 2 ? 0.5 * est.speed + 0.5 * avg : avg
    const remainingTime = remaining / Math.max(1, v)
    const arrived = this.navigating && remaining < 25

    if (arrived && !this.arrivedLogged) {
      this.arrivedLogged = true
      this.addLog('info', `Arrived at ${r.route.destinationName}`)
    }

    // Live-mode reroute when clearly off the route with a good fix.
    if (this.mode === 'live' && this.navigating && est.mode === 'LIVE' && offRoute > 70 && !arrived) {
      this.offRouteSince ??= t
      if (t - this.offRouteSince > 8000 && !this.rerouteInFlight) {
        this.rerouteInFlight = true
        this.addLog('route', 'Off route — rerouting')
        void this.routeTo(r.destination, r.route.destinationName).finally(() => {
          this.rerouteInFlight = false
          this.offRouteSince = null
        })
      }
    } else {
      this.offRouteSince = null
    }

    return {
      along,
      remaining,
      remainingTime,
      etaClock: new Date(Date.now() + remainingTime * 1000),
      nextStepIdx: idx,
      nextStep,
      thenStep,
      distToManeuver: nextStep ? Math.max(0, nextStep.at - along) : remaining,
      offRoute,
      arrived,
    }
  }

  // ── modes ─────────────────────────────────────────────────────────────────

  setMode(mode: AppMode) {
    if (mode === this.mode) return
    this.live.stop()
    this.estimator.reset()
    this.clearRoute()
    this.clearTrail()
    this.playing = false
    this.simT = 0
    this.mode = mode
    if (mode === 'live') this.estimator.configure({ enabled: true, routeConstraint: true })
    this.lastEst = this.estimator.tick(this.now())
    this.lastProgress = null
    this.log = []
    if (mode === 'demo') this.loadDemo()
    this.dirty = true
  }

  /** Begin real GNSS (and motion sensors where no prompt is needed). */
  startLiveSensors() {
    if (this.mode !== 'live') return
    this.live.startGeolocation(this.sink, () => performance.now())
    this.live.attachSensorsIfAllowed()
    this.dirty = true
  }

  async requestMotionPermission() {
    const ok = await this.live.requestMotionPermission()
    this.addLog(ok ? 'info' : 'warn', ok ? 'Motion sensors enabled' : 'Motion permission denied — dead reckoning will hold last speed & course')
  }

  // ── demo ──────────────────────────────────────────────────────────────────

  private loadDemo() {
    const token = ++this.loadToken
    if (this.scenario === 'recorded') {
      // Real phone IMU → the learned engine. No route snapping: the "route"
      // here is the recorded track itself, so using it would be cheating.
      this.estimator.configure({ enabled: true, routeConstraint: false })
      this.demo = null
      this.demoLoading = true
      this.dirty = true
      this.replay ??= ReplaySource.load(`${ML_BASE}replay-y1-3`)
      this.replay
        .then((src) => {
          if (token !== this.loadToken || this.mode !== 'demo') return
          src.reset()
          this.demo = src
          this.demoLoading = false
          const coords = src.manifest.osrm.geometry.coordinates
          this.setRoute(fromOsrm(src.manifest.osrm, 'End of recorded drive'), coords[coords.length - 1] as LngLat)
          this.navigating = true
          this.addLog('info', `Loaded ${src.manifest.name}. The phone mount calibrates during the first 5 min, then GNSS drops in two dead zones.`)
        })
        .catch(() => {
          this.demoLoading = false
          this.replay = null
          this.addLog('warn', 'Could not load the recorded drive — switch to the Delhi scenario')
        })
      return
    }
    // Synthetic sensors: the learned model was never meant for them, use the heuristic.
    this.estimator.configure({ enabled: false, routeConstraint: true })
    const cached = readCache()
    if (cached) this.demoFile = cached
    const f = this.demoFile
    this.demo = new DemoSource(f.samples, f.deadZones)
    const dest = f.destination.coord as LngLat
    this.setRoute(fromOsrm(f.osrm, f.destination.name), dest)
    this.navigating = true
    this.addLog('info', `Demo route loaded: ${f.origin.name} → ${f.destination.name} (${f.source === 'osrm' ? 'OSRM route' : 'hand-traced route'})`)
    if (f.source !== 'osrm') void this.upgradeDemoRoute()
  }

  setScenario(s: DemoScenario) {
    if (s === this.scenario) return
    this.scenario = s
    try {
      localStorage.setItem(SCENARIO_KEY, s)
    } catch {
      /* ignore */
    }
    this.resetDemo()
  }

  /** Fast-forward the demo clock (e.g. past the recorded drive's mount calibration). */
  skipTo(target: number) {
    if (this.mode !== 'demo' || !this.demo || target <= this.simT) return
    while (this.simT < target) {
      this.simT = Math.min(target, this.simT + 50)
      this.demo.advance(this.simT, this.sink)
      this.lastEst = this.estimator.tick(this.simT)
      this.updateTrail(this.lastEst)
      const truth = this.demo.truthAt(this.simT).pos
      const last = this.truthTrail[this.truthTrail.length - 1]
      if (!last || haversine(last, truth) > TRAIL_MIN_STEP) this.truthTrail.push(truth)
    }
    this.addLog('info', `Skipped ahead to ${Math.floor(target / 60000)}:${String(Math.floor((target % 60000) / 1000)).padStart(2, '0')}`)
    this.dirty = true
  }

  /** If the bundled route is the offline hand-traced fallback, fetch the real one. */
  private async upgradeDemoRoute() {
    const f = this.demoFile
    try {
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), 7000)
      const path = f.waypoints.map((w) => w.join(',')).join(';')
      const res = await fetch(
        `https://router.project-osrm.org/route/v1/driving/${path}?overview=full&geometries=geojson&steps=true`,
        { signal: ctrl.signal },
      )
      clearTimeout(timer)
      const data = await res.json()
      if (data.code !== 'Ok') return
      const osrm = data.routes[0]
      const upgraded: DemoFile = {
        ...f,
        source: 'osrm',
        osrm,
        samples: buildTimeline(osrm.geometry.coordinates),
      }
      try {
        localStorage.setItem(DEMO_CACHE_KEY, JSON.stringify(upgraded))
      } catch {
        /* storage unavailable — fine */
      }
      this.demoFile = upgraded
      if (this.mode === 'demo' && this.simT === 0 && !this.playing) this.resetDemo()
      this.addLog('route', 'Demo route refreshed from OSRM')
    } catch {
      /* offline — keep the bundled route */
    }
  }

  resetDemo() {
    if (this.mode !== 'demo') return
    this.estimator.reset()
    this.clearRoute()
    this.clearTrail()
    this.simT = 0
    this.playing = false
    this.log = []
    this.lastEst = this.estimator.tick(0)
    this.loadDemo()
  }

  setPlaying(p: boolean) {
    if (this.mode !== 'demo' || !this.demo) return
    if (p && this.simT >= this.demo.duration) this.resetDemo()
    this.playing = p
    this.dirty = true
  }

  setRate(r: number) {
    this.rate = r
    this.dirty = true
  }

  toggleSimulatedOutage() {
    if (!this.demo) return
    const wasActive = this.demo.manualOutageRemaining(this.simT) > 0
    this.demo.toggleManualOutage(this.simT)
    const secs = Math.round(this.demo.manualOutageRemaining(this.simT) / 1000)
    this.addLog('warn', wasActive ? 'Presenter restored GNSS' : `Presenter triggered GNSS loss (${secs} s)`)
  }

  get lastFrameMode(): FixMode {
    return this.lastEst.mode
  }
}

function readScenario(): DemoScenario {
  try {
    const q = new URLSearchParams(location.search).get('scenario')
    const v = q ?? localStorage.getItem(SCENARIO_KEY)
    return v === 'delhi' ? 'delhi' : 'recorded'
  } catch {
    return 'recorded'
  }
}

function readCache(): DemoFile | null {
  try {
    const s = localStorage.getItem(DEMO_CACHE_KEY)
    if (!s) return null
    const f = JSON.parse(s) as DemoFile
    // invalidate if the bundled scenario changed
    if (JSON.stringify(f.waypoints) !== JSON.stringify((demoData as DemoFile).waypoints)) return null
    if ((demoData as DemoFile).source === 'osrm') return null
    return f
  } catch {
    return null
  }
}
