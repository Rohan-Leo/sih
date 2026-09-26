/**
 * LearnedEstimator — the trained pipeline running in the browser.
 *
 *   phone IMU ─► 10 Hz rows ─► online mount calibration ─► features ─► SpeedNet (1 Hz)
 *                                     │                                    │
 *                                     └──► UKF (non-holonomic motion) ◄────┘ learned speed
 *                                          ▲  + ZUPT + online bias correction
 *                              GNSS fixes ─┘
 *
 * It extends the heuristic PositionEstimator, so GNSS-loss detection,
 * reacquisition blending and the drift log are shared. Until the mount is
 * calibrated (≈5 min of driving with GNSS), or when the device has no motion
 * sensors, it falls back to the heuristic dead reckoning.
 *
 * Same maths as ml/clew_ml (see ml/README.md for the offline benchmark).
 * Not included here: HMM map matching on an OSM road graph — during an
 * outage the active route is used as a soft position constraint instead.
 */
import { type LngLat, destination } from '../lib/geo'
import {
  type EstimatorConfig,
  type EngineName,
  type GnssFix,
  type HeadingSource,
  type MotionSample,
  type MotionState,
  PositionEstimator,
} from '../engine/positionEstimator'
import { OnlineMountCalibrator } from './calib'
import type { SpeedNet, SpeedPrediction } from './speednet'
import { UKF } from './ukf'

type V3 = [number, number, number]
const R_EARTH = 6371008.8
const D2R = Math.PI / 180

interface Row {
  acc: V3
  grav: V3
  gyro: V3
}

/** v_gnss ≈ k·v̂ learned online while GNSS is good (port of evaluate.SpeedCorrector). */
class SpeedCorrector {
  private sxy = 0
  private sxx = 0
  private sr = 0
  private n = 0
  get k(): number {
    return this.sxx > 50 ? Math.min(2.5, Math.max(0.5, this.sxy / this.sxx)) : 1
  }
  observe(vGnss: number, vHat: number) {
    if (!Number.isFinite(vHat) || vGnss < 1) return
    const f = 0.992
    const r = vGnss - this.k * vHat
    this.sxy = f * this.sxy + vGnss * vHat
    this.sxx = f * this.sxx + vHat * vHat
    this.sr = f * this.sr + r * r
    this.n = f * this.n + 1
  }
  measurement(vHat: number, modelVar: number): [number, number] {
    if (this.n < 20) return [vHat, Math.max(modelVar, 1) * 2]
    return [this.k * vHat, Math.max(this.sr / this.n, 1) * 2]
  }
}

export interface LearnedOptions {
  /** use the learned engine at all (off for synthetic demo sensors) */
  enabled: boolean
  /** use the active route as a soft position constraint during outages */
  routeConstraint: boolean
}

export class LearnedEstimator extends PositionEstimator {
  net: SpeedNet | null = null
  opts: LearnedOptions = { enabled: true, routeConstraint: true }

  private origin: [number, number] | null = null
  private bin: { key: number; acc: V3; grav: V3; gyro: V3; n: number; hasGrav: boolean } | null = null
  private gravLp: V3 | null = null
  private rows: Row[] = []
  private yawHist: number[] = []
  private latHist: number[] = []
  private feats: number[][] = []
  private calib = new OnlineMountCalibrator()
  private ukf: UKF | null = null
  private corr = new SpeedCorrector()
  private pred: SpeedPrediction | null = null
  private rowCount = 0
  private imuSeen = false

  constructor(cfg: Partial<EstimatorConfig> = {}) {
    super(cfg)
  }

  configure(opts: Partial<LearnedOptions>) {
    this.opts = { ...this.opts, ...opts }
  }

  override reset() {
    const net = this.net
    const opts = this.opts
    super.reset()
    this.net = net
    this.opts = opts
    this.origin = null
    this.bin = null
    this.gravLp = null
    this.rows = []
    this.yawHist = []
    this.latHist = []
    this.feats = []
    this.calib = new OnlineMountCalibrator()
    this.ukf = null
    this.corr = new SpeedCorrector()
    this.pred = null
    this.rowCount = 0
    this.imuSeen = false
  }

  private get ready(): boolean {
    return this.opts.enabled && this.ukf !== null && this.net !== null
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
    const key = Math.floor(m.t / 100) // 10 Hz bins, the model's sample rate
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

  override pushFix(fix: GnssFix) {
    super.pushFix(fix)
    if (!this.opts.enabled || this.lastGoodFix !== fix) return // rejected (degraded) fixes don't aid
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
    this.rows.push(r)
    if (this.rows.length > 12) this.rows.shift()
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
    const W = this.net?.window ?? 100
    if (this.feats.length > W) this.feats.shift()

    // speed model at 1 Hz
    if (this.net && this.feats.length === W && this.rowCount % 10 === 0) {
      const C = this.feats[0].length
      const x = new Float32Array(C * W)
      for (let t = 0; t < W; t++) for (let ch = 0; ch < C; ch++) x[ch * W + t] = this.feats[t][ch]
      this.pred = this.net.predict(x)
    }

    const u = this.ukf
    if (!u) return
    u.predict(yaw, aLong, 0.1)
    if (this.pred && this.pred.pStill > 0.8) u.zupt()
    if (this.pred && this.rowCount % 10 === 0) {
      const [z, R] = this.corr.measurement(this.pred.speed, this.pred.variance)
      u.speedObs(z, R)
    }
    // during an outage, the planned route acts as a soft map-matching constraint
    if (this.drActive && this.opts.routeConstraint && this.route && this.rowCount % 10 === 0) {
      const p = this.toLngLat(u.x[0], u.x[1])
      const pr = this.route.project(p, this.along, 400)
      if (pr.offset < 50) {
        const [e, n] = this.toEN(pr.point[0], pr.point[1])
        u.position(e, n, 8)
      }
    }
  }

  // ── hooks into the shared estimator ──
  protected override onEnterDeadReckoning(t: number) {
    super.onEnterDeadReckoning(t)
    if (!this.ready) return
    const p = this.toLngLat(this.ukf!.x[0], this.ukf!.x[1])
    // display starts where the dot was and eases onto the filter's estimate
    this.snapOffset = this.est ? [this.est[0] - p[0], this.est[1] - p[1]] : [0, 0]
    this.drBase = p
    this.snapped = false
    this.drSpeed = this.ukf!.x[3]
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
    return this.ready ? Math.max(3, this.ukf!.sigmaPos) : super.drSigma(outageS)
  }

  protected override engineInfo(): { engine: EngineName; engineNote: string } {
    if (!this.opts.enabled) return super.engineInfo()
    if (!this.net) return { engine: 'heuristic', engineNote: 'Loading speed model…' }
    if (!this.imuSeen) return { engine: 'heuristic', engineNote: 'No motion sensors — learned engine unavailable' }
    if (!this.calib.result) {
      return {
        engine: 'heuristic',
        engineNote: `Calibrating phone mount ${Math.round(this.calib.progress * 100)}% (needs GNSS while driving)`,
      }
    }
    if (!this.ukf) return { engine: 'heuristic', engineNote: 'Waiting for a GNSS course to start the filter' }
    const p = this.pred
    return {
      engine: 'learned',
      engineNote: p ? `UKF + SpeedNet · v̂ ${(p.speed * 3.6).toFixed(0)} km/h · bias ×${this.corr.k.toFixed(2)}` : 'UKF + SpeedNet · warming up',
    }
  }
}
