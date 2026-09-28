/**
 * LearnedEstimator — the trained pipeline running in the browser.
 *
 *   phone IMU ─► 10 Hz rows ─► online mount calibration ─► features (10 s window)
 *                                                            │  1 Hz
 *                     ┌──────────────┬───────────────────────┼──────────────┐
 *                     ▼              ▼                       ▼              │
 *                 SpeedNet       HeadingNet               MotionNet         │
 *              speed, σ², still  gyro-bias estimate     driving state      │
 *                     │              ┆ (only if it beat the benchmark)      │
 *                     └──► UKF (non-holonomic) ◄─ GNSS fixes ◄─ IntegrityNet (P fault)
 *                          + ZUPT + online speed-bias correction
 *                                │ during outages
 *                                ▼
 *                            DriftNet ─► 68 / 95 % error radius (the halo)
 *
 * It extends the heuristic PositionEstimator, so GNSS-loss detection,
 * reacquisition blending and the drift log are shared. Until the mount is
 * calibrated (≈5 min of driving with GNSS), or when the device has no motion
 * sensors, it falls back to the heuristic dead reckoning.
 *
 * Same maths as ml/clew_ml (see ml/README.md for the offline benchmarks).
 * Not included here: HMM map matching on an OSM road graph — during an
 * outage the active route is used as a soft position constraint instead.
 */
import { type LngLat, destination } from '../lib/geo'
import {
  type EstimatorConfig,
  type EngineName,
  type GnssFix,
  type HeadingSource,
  type MlInfo,
  type MotionSample,
  type MotionState,
  PositionEstimator,
} from '../engine/positionEstimator'
import { OnlineMountCalibrator } from './calib'
import { type DriftNet, type DriftRadii, OutageTracker } from './drift'
import type { HeadingNet, MotionNet, DrivingState } from './imuModels'
import { FixChecker, type IntegrityNet } from './integrity'
import { SpeedCorrector } from './speedCorrector'
import type { SpeedNet, SpeedPrediction } from './speednet'
import { UKF } from './ukf'

type V3 = [number, number, number]
const R_EARTH = 6371008.8
const D2R = Math.PI / 180
/** P(fault) above which IntegrityNet rejects a fix */
const FAULT_THRESHOLD = 0.5
/** after this long without a trusted fix, accept fixes again (the model is advisory, never a lock-out) */
const TRUST_OVERRIDE_S = 45
/** HeadingNet predictions from overlapping windows are correlated (evaluate.HEADING_VAR_INFLATE) */
const HEADING_VAR_INFLATE = 4

interface Row {
  acc: V3
  grav: V3
  gyro: V3
}

export interface LearnedOptions {
  /** use the learned engine at all (off for synthetic demo sensors) */
  enabled: boolean
  /** use the active route as a soft position constraint during outages */
  routeConstraint: boolean
}

/** The five trained models. Only SpeedNet is required; the rest are used when loaded. */
export interface ModelSet {
  speed: SpeedNet
  heading: HeadingNet | null
  motion: MotionNet | null
  integrity: IntegrityNet | null
  drift: DriftNet | null
}

export class LearnedEstimator extends PositionEstimator {
  models: ModelSet | null = null
  opts: LearnedOptions = { enabled: true, routeConstraint: true }

  private origin: [number, number] | null = null
  private bin: { key: number; acc: V3; grav: V3; gyro: V3; n: number; hasGrav: boolean } | null = null
  private gravLp: V3 | null = null
  private yawHist: number[] = []
  private latHist: number[] = []
  private feats: number[][] = []
  private calib = new OnlineMountCalibrator()
  private ukf: UKF | null = null
  private corr = new SpeedCorrector()
  private pred: SpeedPrediction | null = null
  private heading: { delta: number; variance: number } | null = null
  private driving: { state: DrivingState; p: number } | null = null
  private checker = new FixChecker()
  private trust: number | null = null
  private tracker: OutageTracker | null = null
  private radii: DriftRadii | null = null
  private rowCount = 0
  private imuSeen = false

  constructor(cfg: Partial<EstimatorConfig> = {}) {
    super(cfg)
  }

  /** Back-compat: the speed model alone. */
  get net(): SpeedNet | null {
    return this.models?.speed ?? null
  }
  set net(n: SpeedNet | null) {
    this.models = n ? { heading: null, motion: null, integrity: null, drift: null, ...this.models, speed: n } : null
  }

  configure(opts: Partial<LearnedOptions>) {
    this.opts = { ...this.opts, ...opts }
  }

  override reset() {
    const models = this.models
    const opts = this.opts
    super.reset()
    this.models = models
    this.opts = opts
    this.origin = null
    this.bin = null
    this.gravLp = null
    this.yawHist = []
    this.latHist = []
    this.feats = []
    this.calib = new OnlineMountCalibrator()
    this.ukf = null
    this.corr = new SpeedCorrector()
    this.pred = null
    this.heading = null
    this.driving = null
    this.checker = new FixChecker()
    this.trust = null
    this.tracker = null
    this.radii = null
    this.rowCount = 0
    this.imuSeen = false
  }

  private get ready(): boolean {
    return this.opts.enabled && this.ukf !== null && this.models !== null
  }

  private get useHeading(): boolean {
    return !!this.models?.heading?.useInFilter
  }

  // ── coordinates ──
  private toEN(lon: number, lat: number): [number, number] {
    const [lat0, lon0] = this.origin!
    return [(lon - lon0) * D2R * R_EARTH * Math.cos(lat0 * D2R), (lat - lat0) * D2R * R_EARTH]
  }
  private toLngLat(e: number, n: number): LngLat {
    const [lat0, lon0] = this.origin!
    return [lon0 + e / (R_EARTH * Math.cos(lat0 * D2R)) / D2R, lat0 + n / R_EARTH / D2R]
  }

  // ── inputs ──
  override pushMotion(m: MotionSample) {
    super.pushMotion(m)
    if (!this.opts.enabled) return
    this.imuSeen = true
    const acc: V3 = [m.ax, m.ay, m.az]
    const gyro: V3 = m.gyro ?? [0, 0, ((m.gyroZ ?? 0) * Math.PI) / 180]
    const key = Math.floor(m.t / 100) // 10 Hz bins, the models' sample rate
    if (this.bin && key !== this.bin.key) this.flushBin()
    if (!this.bin) this.bin = { key, acc: [0, 0, 0], grav: [0, 0, 0], gyro: [0, 0, 0], n: 0, hasGrav: !!m.gravity }
    const b = this.bin
    for (let i = 0; i < 3; i++) {
      b.acc[i] += acc[i]
      b.gyro[i] += gyro[i]
      if (m.gravity) b.grav[i] += m.gravity[i]
    }
    b.n++
  }

  private flushBin() {
    const b = this.bin!
    this.bin = null
    const acc = b.acc.map((v) => v / b.n) as V3
    const gyro = b.gyro.map((v) => v / b.n) as V3
    let grav: V3
    if (b.hasGrav) grav = b.grav.map((v) => v / b.n) as V3
    else {
      // no separate gravity from the device: low-pass the accelerometer
      this.gravLp = this.gravLp ? (this.gravLp.map((g, i) => g + 0.05 * (acc[i] - g)) as V3) : acc
      grav = this.gravLp
    }
    this.onRow({ acc, grav, gyro })
  }

  /** IntegrityNet gate, called by the base class before a fix is accepted. */
  protected override rejectFix(fix: GnssFix): { reason: string; message: string } | null {
    const integ = this.models?.integrity
    if (!this.opts.enabled || !integ || !this.origin || !this.calib.result) return null
    const [e, n] = this.toEN(fix.lon, fix.lat)
    const speed = fix.speed ?? this.gnssSpeed
    const course = fix.course ?? this.gnssCourse ?? 0
    const vHat = this.pred?.speed ?? null
    const f = this.checker.features(e, n, fix.t / 1000, fix.accuracy, speed, course, vHat)
    let trusted = true
    let p = 0
    if (f) {
      p = integ.pFault(f)
      this.trust = 1 - p
      trusted = p < FAULT_THRESHOLD || this.checker.secondsSinceTrusted > TRUST_OVERRIDE_S
    }
    this.checker.afterFix(e, n, fix.t / 1000, fix.accuracy, speed, course, vHat, trusted)
    if (trusted) return null
    return {
      reason: `IntegrityNet: fix inconsistent with the IMU (P fault ${(p * 100).toFixed(0)}%)`,
      message: `IntegrityNet rejected a GNSS fix that disagrees with the IMU dead reckoning (P fault ${(p * 100).toFixed(0)}%, reported ±${Math.round(fix.accuracy)} m)`,
    }
  }

  override pushFix(fix: GnssFix) {
    super.pushFix(fix)
    if (!this.opts.enabled || this.lastGoodFix !== fix) return // rejected fixes don't aid
    if (!this.origin) this.origin = [fix.lat, fix.lon]
    const speed = fix.speed ?? this.gnssSpeed
    const course = fix.course ?? this.gnssCourse
    this.calib.addFix(fix.t, speed, course)
    if (this.calib.result && !this.ukf && course !== null) {
      this.ukf = new UKF()
      const [e, n] = this.toEN(fix.lon, fix.lat)
      this.ukf.init(e, n, course * D2R, speed)
    }
    if (this.ukf) {
      const [e, n] = this.toEN(fix.lon, fix.lat)
      this.ukf.gnss(e, n, fix.accuracy, speed, course)
      if (this.pred) this.corr.observe(speed, this.pred.speed)
    }
  }

  // ── the 10 Hz pipeline ──
  private onRow(r: Row) {
    const dyn: V3 = [r.acc[0] - r.grav[0], r.acc[1] - r.grav[1], r.acc[2] - r.grav[2]]
    this.calib.addImu(r.gyro, dyn)
    const c = this.calib.result
    if (!c) return
    this.rowCount++
    const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    const gn = Math.hypot(...r.grav) || 1
    const yaw = dot(r.gyro, c.wYaw)
    const aLong = dot(dyn, c.uFwd)
    const aLat = dot(dyn, c.uLat)
    this.yawHist.push(yaw)
    this.latHist.push(aLat)
    if (this.yawHist.length > 10) {
      this.yawHist.shift()
      this.latHist.shift()
    }
    const yr = this.yawHist.reduce((a, b) => a + b, 0) / this.yawHist.length
    const al = this.latHist.reduce((a, b) => a + b, 0) / this.latHist.length
    const turning = Math.abs(yr) > 0.05
    // channel order must match ml/clew_ml/features.py CHANNELS
    this.feats.push([
      yaw,
      aLong,
      aLat,
      dot(dyn, [r.grav[0] / gn, r.grav[1] / gn, r.grav[2] / gn]),
      Math.hypot(...dyn),
      Math.hypot(...r.gyro),
      turning ? Math.min(40, Math.max(0, al / yr)) : 0,
      turning ? 1 : 0,
    ])
    const M = this.models
    const W = M?.speed.window ?? 100
    if (this.feats.length > W) this.feats.shift()

    // the window models at 1 Hz
    const tick = this.rowCount % 10 === 0
    if (M && this.feats.length === W && tick) {
      const C = this.feats[0].length
      const x = new Float32Array(C * W)
      for (let t = 0; t < W; t++) for (let ch = 0; ch < C; ch++) x[ch * W + t] = this.feats[t][ch]
      this.pred = M.speed.predict(x)
      if (M.heading) this.heading = M.heading.predict(x, W)
      if (M.motion) {
        const d = M.motion.predict(x, W)
        this.driving = { state: d.state, p: d.p }
      }
    }
    this.checker.imuStep(yaw, this.pred?.speed ?? null)

    const u = this.ukf
    if (!u) return
    u.predict(yaw, aLong, 0.1)
    if (this.useHeading && this.heading && tick) u.gyroBiasObs(-this.heading.delta, this.heading.variance * HEADING_VAR_INFLATE)
    const still = !!this.pred && this.pred.pStill > 0.8
    if (still) u.zupt()
    if (this.pred && tick) {
      const [z, R] = this.corr.measurement(this.pred.speed, this.pred.variance)
      u.speedObs(z, R)
    }
    // during an outage, the planned route acts as a soft map-matching constraint
    if (this.drActive && this.opts.routeConstraint && this.route && tick) {
      const p = this.toLngLat(u.x[0], u.x[1])
      const pr = this.route.project(p, this.along, 400)
      if (pr.offset < 50) {
        const [e, n] = this.toEN(pr.point[0], pr.point[1])
        u.position(e, n, 8)
      }
    }
    if (this.drActive && this.tracker) {
      this.tracker.step(u.x[3], yaw, still)
      if (tick && this.pred) this.tracker.speedObs(this.pred.variance)
      if (tick && M?.drift) this.radii = M.drift.radii(this.tracker.features(u.x[2], u.sigmaPos))
    }
  }

  // ── hooks into the shared estimator ──
  protected override onEnterDeadReckoning(t: number) {
    super.onEnterDeadReckoning(t)
    this.radii = null
    this.tracker = null
    if (!this.ready) return
    const u = this.ukf!
    const c = this.calib.result!
    this.tracker = new OutageTracker(u.x[2], this.corr, c.r2Yaw, c.r2Long)
    const p = this.toLngLat(u.x[0], u.x[1])
    // display starts where the dot was and eases onto the filter's estimate
    this.snapOffset = this.est ? [this.est[0] - p[0], this.est[1] - p[1]] : [0, 0]
    this.drBase = p
    this.snapped = false
    this.drSpeed = u.x[3]
  }

  protected override propagateDeadReckoning(
    t: number,
    dt: number,
    motion: MotionState,
    from: LngLat,
    along: number,
    snapped: boolean,
    primary: boolean,
  ): { pos: LngLat; along: number; snapped: boolean; heading: number; source: HeadingSource } {
    if (!this.ready) return super.propagateDeadReckoning(t, dt, motion, from, along, snapped, primary)
    const u = this.ukf!
    const heading = ((u.x[2] / D2R) % 360 + 360) % 360
    if (!primary) {
      // ghost for the reacquisition blend: keep coasting on the filter's velocity
      return { pos: destination(from, heading, u.x[3] * dt), along, snapped: false, heading, source: 'ukf' }
    }
    const pos = this.toLngLat(u.x[0], u.x[1])
    this.drSpeed = u.x[3]
    this.drDistance += u.x[3] * dt
    let nextAlong = along
    let onRoute = false
    if (this.route) {
      const pr = this.route.project(pos, along, 400)
      nextAlong = pr.along
      onRoute = this.opts.routeConstraint && pr.offset < 50
    }
    return { pos, along: nextAlong, snapped: onRoute, heading, source: 'ukf' }
  }

  protected override drSigma(outageS: number): number {
    if (!this.ready) return super.drSigma(outageS)
    // DriftNet's 68 % radius when available — the filter's own σ is overconfident in outages
    if (this.radii) return Math.max(3, this.radii.r68)
    return Math.max(3, this.ukf!.sigmaPos)
  }

  protected override mlInfo(): MlInfo | null {
    if (!this.opts.enabled || !this.models || !this.calib.result) return null
    return {
      driving: this.driving?.state ?? null,
      drivingP: this.driving?.p ?? null,
      gnssTrust: this.trust,
      radius95: this.drActive && this.radii ? this.radii.r95 : null,
      headingCorr: this.heading ? (this.heading.delta * 180) / Math.PI : null,
    }
  }

  protected override engineInfo(): { engine: EngineName; engineNote: string } {
    if (!this.opts.enabled) return super.engineInfo()
    const M = this.models
    if (!M) return { engine: 'heuristic', engineNote: 'Loading models…' }
    if (!this.imuSeen) return { engine: 'heuristic', engineNote: 'No motion sensors — learned engine unavailable' }
    if (!this.calib.result) {
      return {
        engine: 'heuristic',
        engineNote: `Calibrating phone mount ${Math.round(this.calib.progress * 100)}% (needs GNSS while driving)`,
      }
    }
    if (!this.ukf) return { engine: 'heuristic', engineNote: 'Waiting for a GNSS course to start the filter' }
    const p = this.pred
    const n = 1 + [M.heading, M.motion, M.integrity, M.drift].filter(Boolean).length
    return {
      engine: 'learned',
      engineNote: p
        ? `UKF + ${n} models · v̂ ${(p.speed * 3.6).toFixed(0)} km/h · bias ×${this.corr.k.toFixed(2)}`
        : `UKF + ${n} models · warming up`,
    }
  }
}
