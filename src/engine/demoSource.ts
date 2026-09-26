/**
 * Scripted sensor feed for Judge Demo mode.
 *
 * Plays back a prerecorded {lat, lon, t} trace (baked from an OSRM route)
 * and synthesises what a phone would report along it: 1 Hz GNSS fixes with
 * realistic noise, 25 Hz accelerometer (road vibration) + gyro yaw rate, and
 * a compass that's deliberately mis-mounted by a few degrees.
 *
 * The ONLY thing it does differently in a dead zone is stop emitting GNSS
 * fixes — the estimator has to notice and fall back on its own, exactly as
 * in Live mode.
 */
import { type LngLat, Polyline, angleDiff, bearing, destination, haversine } from '../lib/geo'
import type { DeadZone, ScriptedSource, SensorSink } from './sources'

export interface DemoSample {
  lat: number
  lon: number
  t: number
}

export type { DeadZone }

const MOUNT_OFFSET_DEG = 7 // phone isn't perfectly aligned with the car
const FIX_PERIOD = 1000
const IMU_PERIOD = 40

export class DemoSource implements ScriptedSource {
  readonly samples: DemoSample[]
  readonly path: Polyline
  readonly duration: number
  /** dead zones in metres along the trace */
  readonly deadZones: DeadZone[]
  private cumAtSample: number[]
  private nextFixT = 0
  private nextImuT = 0
  private manualUntil = -Infinity
  private rng: () => number

  constructor(samples: DemoSample[], deadZoneFractions: DeadZone[]) {
    this.samples = samples
    this.path = new Polyline(samples.map((s) => [s.lon, s.lat] as LngLat))
    this.cumAtSample = this.path.cum
    this.duration = samples[samples.length - 1]?.t ?? 0
    this.deadZones = deadZoneFractions.map((z) => ({
      ...z,
      from: z.from * this.path.length,
      to: z.to * this.path.length,
    }))
    this.rng = mulberry32(26168)
  }

  reset() {
    this.nextFixT = 0
    this.nextImuT = 0
    this.manualUntil = -Infinity
    this.rng = mulberry32(26168)
  }

  /** Ground-truth state at sim time t (ms). */
  truthAt(t: number): { pos: LngLat; speed: number; course: number; along: number } {
    const s = this.samples
    if (t <= 0) return { pos: [s[0].lon, s[0].lat], speed: 0, course: this.path.pointAt(0).bearing, along: 0 }
    let lo = 0
    let hi = s.length - 1
    if (t >= s[hi].t) {
      return { pos: [s[hi].lon, s[hi].lat], speed: 0, course: this.path.pointAt(this.path.length).bearing, along: this.path.length }
    }
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (s[mid].t <= t) lo = mid
      else hi = mid
    }
    const a = s[lo]
    const b = s[hi]
    const f = (t - a.t) / Math.max(1, b.t - a.t)
    const along = this.cumAtSample[lo] + (this.cumAtSample[hi] - this.cumAtSample[lo]) * f
    const segD = this.cumAtSample[hi] - this.cumAtSample[lo]
    const speed = segD / Math.max(0.001, (b.t - a.t) / 1000)
    const p = this.path.pointAt(along)
    const course = segD > 0.3 ? bearing([a.lon, a.lat], [b.lon, b.lat]) : p.bearing
    return { pos: p.point, speed, course, along }
  }

  deadZoneAt(t: number): DeadZone | null {
    const { along } = this.truthAt(t)
    return this.deadZones.find((z) => along >= z.from && along <= z.to) ?? null
  }

  /** Presenter button: withhold GNSS for `ms` of sim time (or cancel). */
  toggleManualOutage(now: number, ms = 12000) {
    this.manualUntil = this.manualUntil > now ? -Infinity : now + ms
  }

  manualOutageRemaining(now: number): number {
    return Math.max(0, this.manualUntil - now)
  }

  gnssWithheld(t: number): boolean {
    return t < this.manualUntil || this.deadZoneAt(t) !== null
  }

  /** Emit every sensor reading due between the last call and sim time `t`. */
  advance(t: number, sink: SensorSink) {
    while (this.nextImuT <= t) {
      this.emitImu(this.nextImuT, sink)
      this.nextImuT += IMU_PERIOD
    }
    while (this.nextFixT <= t) {
      if (!this.gnssWithheld(this.nextFixT)) this.emitFix(this.nextFixT, sink)
      this.nextFixT += FIX_PERIOD
    }
  }

  private gauss(): number {
    // Box–Muller
    const u = Math.max(1e-9, this.rng())
    const v = this.rng()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }

  private emitFix(t: number, sink: SensorSink) {
    const truth = this.truthAt(t)
    const errM = Math.abs(this.gauss()) * 2.2
    const pos = destination(truth.pos, this.rng() * 360, errM)
    sink.pushFix({
      lon: pos[0],
      lat: pos[1],
      accuracy: 4 + this.rng() * 2.5,
      speed: Math.max(0, truth.speed + this.gauss() * 0.25),
      course: truth.speed > 0.5 ? truth.course + this.gauss() * 1.5 : null,
      t,
    })
  }

  private emitImu(t: number, sink: SensorSink) {
    const now = this.truthAt(t)
    const prev = this.truthAt(Math.max(0, t - IMU_PERIOD))
    const dt = IMU_PERIOD / 1000
    const moving = now.speed > 0.4
    const vib = moving ? 0.45 + Math.min(0.5, now.speed / 30) : 0.02
    const longAcc = (now.speed - prev.speed) / dt
    // yaw rate, counter-clockwise positive (DeviceMotion convention)
    const yawRate = -angleDiff(prev.course, now.course) / dt
    sink.pushMotion({
      ax: this.gauss() * vib * 0.5,
      ay: longAcc + this.gauss() * vib * 0.5,
      az: 9.80665 + this.gauss() * vib,
      includesGravity: true,
      gyroZ: yawRate + this.gauss() * 0.4,
      t,
    })
    if (Math.round(t) % 200 === 0) {
      sink.pushHeading({ heading: now.course + MOUNT_OFFSET_DEG + this.gauss() * 2, absolute: true, t })
    }
  }

  /** Metres between the fused estimate and ground truth — only knowable in demo. */
  errorAgainstTruth(est: LngLat | null, t: number): number | null {
    if (!est) return null
    return haversine(est, this.truthAt(t).pos)
  }
}

function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
