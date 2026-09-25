/**
 * PositionEstimator — GNSS/IMU fusion for Clew.
 *
 * ─── Honesty note ───────────────────────────────────────────────────────────
 * This is a deliberately LIGHTWEIGHT, real-time HEURISTIC. It is a stand-in
 * for the pipeline in our SIH 2026 submission (PS 26168), which is:
 *   1. a learned IMU speed model (PyTorch, exported to TFLite) that regresses
 *      forward speed from raw accelerometer/gyroscope windows,
 *   2. an Unscented Kalman Filter with non-holonomic constraints (a car does
 *      not slide sideways or jump vertically) fusing that speed with gyro yaw,
 *   3. HMM map matching over the road network (Viterbi over candidate road
 *      segments) instead of snapping to a single known route.
 * None of that is implemented here. What IS implemented, and actually runs on
 * every tick in both Live and Demo mode:
 *   - GNSS health monitoring (stale fixes, error callbacks, accuracy jumps),
 *   - dead reckoning from a decaying last-known speed, modulated by an
 *     accelerometer motion/step detector,
 *   - heading from the compass (calibrated against the last GNSS course to
 *     absorb phone-mount misalignment) or integrated gyro yaw,
 *   - "map matching" by constraining the estimate to the active route polyline,
 *   - smooth blend-back on reacquisition with a drift report.
 *
 * The public surface (pushFix / pushMotion / pushHeading / setRoute / tick)
 * is intentionally small so the real model can replace the internals of
 * `propagateDeadReckoning` (speed + heading) and `setRoute` (matching)
 * without touching the map, UI or sensor plumbing.
 * ────────────────────────────────────────────────────────────────────────────
 *
 * Time: every method takes timestamps from the engine clock (ms). In Live
 * mode that is performance.now(); in Demo mode it is the simulated clock,
 * so playback speed and pause behave correctly with the same thresholds.
 */
import {
  type LngLat,
  Polyline,
  angleDiff,
  bearing,
  clamp,
  destination,
  haversine,
  lerp,
  lerpAngle,
  lerpLngLat,
  normDeg,
} from '../lib/geo'

export type FixMode = 'SEARCHING' | 'LIVE' | 'DR'
export type MotionState = 'moving' | 'still' | 'unknown'
export type HeadingSource = 'gnss' | 'compass' | 'gyro' | 'route' | 'held'

export interface GnssFix {
  lon: number
  lat: number
  /** 1-σ horizontal accuracy reported by the receiver, metres */
  accuracy: number
  /** m/s, null if the receiver didn't report it */
  speed: number | null
  /** course over ground, degrees, null if unknown */
  course: number | null
  t: number
}

export interface MotionSample {
  /** accelerometer, m/s² */
  ax: number
  ay: number
  az: number
  /** true if the sample includes gravity (accelerationIncludingGravity) */
  includesGravity: boolean
  /** yaw rate about the device z axis, deg/s (DeviceMotionEvent.rotationRate.alpha) */
  gyroZ: number | null
  t: number
}

export interface HeadingSample {
  /** degrees clockwise from north */
  heading: number
  /** true = magnetic/true north referenced; false = arbitrary reference */
  absolute: boolean
  t: number
}

export interface EstimatorConfig {
  /** fix older than this → dead reckoning */
  staleMs: number
  /** fix older than this (but < staleMs) → SEARCHING, predicted forward */
  lateMs: number
  /** accuracy is "degraded" above max(floor, ratio × baseline) … */
  accuracyFloor: number
  accuracyRatio: number
  /** … and always above this */
  accuracyHard: number
  /** blend-back duration on reacquisition */
  blendMs: number
  /** max distance from route to accept snapping during DR */
  snapMaxOffset: number
  /** assumed stride for step-based speed (pedestrian) */
  strideM: number
}

export const DEFAULT_CONFIG: EstimatorConfig = {
  staleMs: 3000,
  lateMs: 1500,
  accuracyFloor: 30,
  accuracyRatio: 3,
  accuracyHard: 150,
  blendMs: 1600,
  snapMaxOffset: 60,
  strideM: 0.72,
}

export interface EstimatorState {
  position: LngLat | null
  heading: number
  speed: number
  mode: FixMode
  /** 1-σ uncertainty radius of the fused estimate, metres */
  sigma: number
  gnssAccuracy: number | null
  rawFix: LngLat | null
  fixAgeMs: number | null
  outageMs: number
  drDistance: number
  motion: MotionState
  headingSource: HeadingSource
  snapped: boolean
  blending: boolean
}

export interface EstimatorEvent {
  kind: 'acquired' | 'lost' | 'reacquired' | 'degraded' | 'offroute'
  t: number
  message: string
  outageMs?: number
  driftM?: number
}

interface MagSample {
  t: number
  mag: number
}

const G = 9.80665

export class PositionEstimator {
  readonly cfg: EstimatorConfig
  private route: Polyline | null = null
  private listeners = new Set<(e: EstimatorEvent) => void>()

  // GNSS
  private lastGoodFix: GnssFix | null = null
  private prevFixForSpeed: GnssFix | null = null
  private accHistory: number[] = []
  private gnssSpeed = 0
  private gnssCourse: number | null = null
  private newGoodFix = false
  private degradedReason: string | null = null
  /** one log line per degradation episode, not per rejected fix */
  private degradedLogged = false
  private everHadFix = false
  /** recent intervals between good fixes — tells a 1 Hz GNSS stream from on-change (Wi-Fi/IP) location */
  private fixIntervals: number[] = []
  private errorSinceFix = false

  // fused estimate
  private est: LngLat | null = null
  private estHeading = 0
  private estSpeed = 0
  private sigma = 50
  private along = 0
  private snapped = false
  private headingSource: HeadingSource = 'held'
  private lastTickT: number | null = null

  // dead reckoning
  private drActive = false
  private drStartT = 0
  private lastGoodBeforeOutageT = 0
  private drDistance = 0
  private drSpeed = 0
  private sigmaAtLoss = 5
  private headingAtLoss = 0
  private speedAtLoss = 0
  private compassOffset = 0
  private gyroYaw = 0
  private offHeadingSince: number | null = null
  private pendingShift = 0
  private drBase: LngLat | null = null
  private snapOffset: [number, number] = [0, 0]
  private track: LngLat | null = null

  // blend-back after reacquisition
  private blend: {
    startT: number
    ghost: LngLat
    ghostAlong: number
    ghostSnapped: boolean
    fromSigma: number
    durationMs: number
  } | null = null

  // sensors
  private compass: HeadingSample | null = null
  private lastGyroT: number | null = null
  private lastMotionT: number | null = null
  private magWindow: MagSample[] = []
  private magEma = 0
  private stepTimes: number[] = []
  private lastStepT = 0
  private stepArmed = true

  constructor(cfg: Partial<EstimatorConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...cfg }
  }

  onEvent(cb: (e: EstimatorEvent) => void): () => void {
    this.listeners.add(cb)
    return () => this.listeners.delete(cb)
  }

  private emit(e: EstimatorEvent) {
    this.listeners.forEach((l) => l(e))
  }

  reset() {
    const listeners = this.listeners
    const route = this.route
    Object.assign(this, new PositionEstimator(this.cfg))
    this.listeners = listeners
    this.route = route
  }

  /** The active route acts as the "road network" for our simplified map matching. */
  setRoute(line: Polyline | null) {
    this.route = line
    this.snapped = false
    this.along = 0
    if (line && this.est) this.along = line.project(this.est).along
  }

  /** Notify of a geolocation error (permission, timeout, unavailable). */
  noteGnssError(reason: string) {
    this.degradedReason = reason
    this.errorSinceFix = true
  }

  // ── inputs ────────────────────────────────────────────────────────────────

  pushFix(fix: GnssFix) {
    const baseline = this.baselineAccuracy()
    const limit = Math.min(
      this.cfg.accuracyHard,
      Math.max(this.cfg.accuracyFloor, baseline !== null ? baseline * this.cfg.accuracyRatio : Infinity),
    )
    if (!(fix.accuracy <= limit)) {
      // Degraded: do not refresh the "good fix" clock. If it stays degraded
      // for staleMs, we fall into dead reckoning exactly like a lost signal.
      if (!this.degradedLogged) {
        this.degradedLogged = true
        this.emit({
          kind: 'degraded',
          t: fix.t,
          message: `GNSS accuracy degraded to ±${Math.round(fix.accuracy)} m (baseline ±${Math.round(baseline ?? 0)} m) — fix rejected`,
        })
      }
      this.degradedReason = `accuracy degraded to ±${Math.round(fix.accuracy)} m`
      return
    }

    this.accHistory.push(fix.accuracy)
    if (this.accHistory.length > 20) this.accHistory.shift()

    // speed: prefer receiver doppler speed, else differentiate positions
    const p: LngLat = [fix.lon, fix.lat]
    let speed = fix.speed
    if ((speed === null || !Number.isFinite(speed)) && this.prevFixForSpeed) {
      const dt = (fix.t - this.prevFixForSpeed.t) / 1000
      if (dt > 0.2) speed = haversine([this.prevFixForSpeed.lon, this.prevFixForSpeed.lat], p) / dt
    }
    if (speed !== null && Number.isFinite(speed)) {
      // light smoothing — raw position-difference speed is noisy
      this.gnssSpeed = this.everHadFix ? lerp(this.gnssSpeed, speed, 0.6) : speed
    }
    let course = fix.course
    if ((course === null || !Number.isFinite(course)) && this.prevFixForSpeed) {
      const prev: LngLat = [this.prevFixForSpeed.lon, this.prevFixForSpeed.lat]
      if (haversine(prev, p) > Math.max(3, fix.accuracy * 0.5)) {
        course = bearing(prev, p)
      }
    }
    if (course !== null && Number.isFinite(course) && this.gnssSpeed > 1.0) this.gnssCourse = normDeg(course)
    this.prevFixForSpeed = fix

    if (!this.everHadFix) {
      this.emit({ kind: 'acquired', t: fix.t, message: `GNSS fix acquired — ±${Math.round(fix.accuracy)} m` })
    }
    this.everHadFix = true
    if (this.lastGoodFix) {
      this.fixIntervals.push(fix.t - this.lastGoodFix.t)
      if (this.fixIntervals.length > 6) this.fixIntervals.shift()
    }
    this.errorSinceFix = false
    this.lastGoodFix = fix
    this.newGoodFix = true
    this.degradedReason = null
    this.degradedLogged = false
  }

  pushHeading(h: HeadingSample) {
    if (h.absolute) this.compass = { ...h, heading: normDeg(h.heading) }
  }

  pushMotion(m: MotionSample) {
    const dt = this.lastMotionT !== null ? clamp((m.t - this.lastMotionT) / 1000, 0, 0.2) : 0
    this.lastMotionT = m.t

    // gyro yaw integration (valid when the phone lies roughly flat).
    // rotationRate.alpha is counter-clockwise-positive about +z (out of the
    // screen); compass headings are clockwise-positive, hence the minus.
    if (m.gyroZ !== null && Number.isFinite(m.gyroZ)) {
      if (this.drActive && this.lastGyroT !== null) this.gyroYaw -= m.gyroZ * dt
      this.lastGyroT = m.t
    }

    const raw = Math.hypot(m.ax, m.ay, m.az)
    // remove gravity (or its slowly varying estimate) → dynamic acceleration
    this.magEma = this.magEma === 0 ? raw : lerp(this.magEma, raw, 0.02)
    const dyn = m.includesGravity ? raw - (this.magEma || G) : raw
    this.magWindow.push({ t: m.t, mag: dyn })
    while (this.magWindow.length && this.magWindow[0].t < m.t - 1500) this.magWindow.shift()

    // naïve step detector: upward threshold crossing with refractory period
    if (dyn > 1.1 && this.stepArmed && m.t - this.lastStepT > 300) {
      this.stepArmed = false
      this.lastStepT = m.t
      this.stepTimes.push(m.t)
      while (this.stepTimes.length && this.stepTimes[0] < m.t - 4000) this.stepTimes.shift()
    }
    if (dyn < 0.3) this.stepArmed = true
  }

  // ── derived sensor features ───────────────────────────────────────────────

  private baselineAccuracy(): number | null {
    if (this.accHistory.length < 3) return null
    const s = [...this.accHistory].sort((a, b) => a - b)
    return s[Math.floor(s.length / 2)]
  }

  /**
   * Phones stream GNSS at ~1 Hz, so silence means loss. Desktop browsers only
   * report on change, so silence there means "hasn't moved" — only errors or
   * degraded accuracy count as loss for such sources.
   */
  private isStreaming(): boolean {
    if (this.fixIntervals.length < 3) return false
    const s = [...this.fixIntervals].sort((a, b) => a - b)
    return s[Math.floor(s.length / 2)] < 1600
  }

  private motionState(t: number): MotionState {
    if (this.lastMotionT === null || t - this.lastMotionT > 1500 || this.magWindow.length < 8) return 'unknown'
    const n = this.magWindow.length
    const mean = this.magWindow.reduce((a, s) => a + s.mag, 0) / n
    const variance = this.magWindow.reduce((a, s) => a + (s.mag - mean) ** 2, 0) / n
    return Math.sqrt(variance) > 0.12 ? 'moving' : 'still'
  }

  private stepSpeed(t: number): number {
    const recent = this.stepTimes.filter((s) => s > t - 3000)
    if (recent.length < 3) return 0
    return (recent.length / 3) * this.cfg.strideM
  }

  private compassFresh(t: number): boolean {
    return this.compass !== null && t - this.compass.t < 1500
  }

  // ── main loop ─────────────────────────────────────────────────────────────

  tick(t: number): EstimatorState {
    const dt = this.lastTickT === null ? 0 : clamp((t - this.lastTickT) / 1000, 0, 0.5)
    this.lastTickT = t
    const motion = this.motionState(t)

    if (!this.lastGoodFix) {
      return this.snapshot(t, 'SEARCHING', motion)
    }

    const age = t - this.lastGoodFix.t
    const fixPos: LngLat = [this.lastGoodFix.lon, this.lastGoodFix.lat]

    if (this.newGoodFix) {
      this.newGoodFix = false
      if (this.drActive && this.est) {
        const outageMs = this.lastGoodFix.t - this.lastGoodBeforeOutageT
        const drift = haversine(this.est, fixPos)
        this.emit({
          kind: 'reacquired',
          t,
          outageMs,
          driftM: drift,
          message: `GNSS reacquired — ${(outageMs / 1000).toFixed(1)} s outage, ~${Math.round(drift)} m drift corrected (dead-reckoned ${Math.round(this.drDistance)} m)`,
        })
        this.blend = {
          startT: t,
          ghost: this.drBase ?? this.est,
          ghostAlong: this.along,
          ghostSnapped: this.snapped,
          fromSigma: this.sigma,
          // bigger corrections glide in over longer, so nothing visibly jumps
          durationMs: clamp(this.cfg.blendMs + drift * 25, this.cfg.blendMs, 5000),
        }
        this.drActive = false
      }
    }

    let mode: FixMode
    const lost =
      age > this.cfg.staleMs && (this.isStreaming() || this.errorSinceFix || this.degradedReason !== null || this.drActive)
    if (lost) {
      if (!this.drActive) this.enterDeadReckoning(t, age)
      const next = this.propagateDeadReckoning(t, dt, motion, this.drBase ?? this.est ?? fixPos, this.along, this.snapped, true)
      this.drBase = next.pos
      // ease out the lateral jump from "GNSS track" onto the route line
      const k = Math.exp(-dt / 0.6)
      this.snapOffset = [this.snapOffset[0] * k, this.snapOffset[1] * k]
      this.est = [next.pos[0] + this.snapOffset[0], next.pos[1] + this.snapOffset[1]]
      this.along = next.along
      this.snapped = next.snapped
      this.estHeading = next.heading
      this.headingSource = next.source
      this.estSpeed = this.drSpeed
      const outageS = (t - this.lastGoodBeforeOutageT) / 1000
      this.sigma = Math.min(
        300,
        this.sigmaAtLoss + 0.4 * outageS + (this.snapped ? 0.05 : 0.12) * this.drDistance,
      )
      mode = 'DR'
    } else {
      // GNSS usable. Predict the fix forward a little so the dot glides
      // between 1 Hz updates instead of stepping.
      const lead = Math.min(age, this.cfg.staleMs) / 1000
      const target =
        this.gnssCourse !== null && this.gnssSpeed > 0.8
          ? destination(fixPos, this.gnssCourse, this.gnssSpeed * lead)
          : fixPos
      const acc = this.lastGoodFix.accuracy

      // Smoothed GNSS track: absorbs fix-to-fix noise so nothing steps.
      if (!this.track || haversine(this.track, target) > 150) this.track = target
      else this.track = lerpLngLat(this.track, target, 1 - Math.exp(-dt / 0.3))

      if (this.blend) {
        // keep dead-reckoning a "ghost" so the blend starts from where the
        // DR estimate would be *now*, then ease into the GNSS track
        const g = this.propagateDeadReckoning(t, dt, motion, this.blend.ghost, this.blend.ghostAlong, this.blend.ghostSnapped, false)
        this.blend.ghost = g.pos
        this.blend.ghostAlong = g.along
        const k = clamp((t - this.blend.startT) / this.blend.durationMs, 0, 1)
        const e = easeInOut(k)
        this.est = lerpLngLat(g.pos, this.track, e)
        this.sigma = lerp(this.blend.fromSigma, acc, e)
        if (k >= 1) this.blend = null
      } else {
        this.est = this.track
        this.sigma = lerp(this.sigma, acc, 1 - Math.exp(-dt / 0.8))
      }

      if (this.gnssCourse !== null && this.gnssSpeed > 1.2) {
        this.estHeading = lerpAngle(this.estHeading, this.gnssCourse, 1 - Math.exp(-dt / 0.3))
        this.headingSource = 'gnss'
      } else if (this.compassFresh(t)) {
        this.estHeading = lerpAngle(this.estHeading, this.compass!.heading, 1 - Math.exp(-dt / 0.3))
        this.headingSource = 'compass'
      }
      this.estSpeed = this.gnssSpeed
      if (this.route && this.est) {
        const pr = this.route.project(this.est, this.along, 400)
        this.along = pr.offset < this.cfg.snapMaxOffset ? pr.along : this.route.project(this.est).along
      }
      mode = age > this.cfg.lateMs && this.isStreaming() ? 'SEARCHING' : 'LIVE'
    }

    return this.snapshot(t, mode, motion)
  }

  private enterDeadReckoning(t: number, age: number) {
    this.drActive = true
    this.drStartT = t
    this.lastGoodBeforeOutageT = this.lastGoodFix!.t
    this.drDistance = 0
    this.gyroYaw = 0
    this.blend = null
    this.offHeadingSince = null
    this.pendingShift = 0
    this.speedAtLoss = this.gnssSpeed
    this.drSpeed = this.gnssSpeed
    this.sigmaAtLoss = this.sigma
    this.headingAtLoss = this.gnssCourse ?? this.estHeading
    // Calibrate compass → vehicle heading offset from the last good course.
    // A phone in a cradle rarely points exactly along the direction of travel.
    if (this.compassFresh(t) && this.gnssCourse !== null && this.gnssSpeed > 1.5) {
      this.compassOffset = angleDiff(this.compass!.heading, this.gnssCourse)
    } else {
      this.compassOffset = 0
    }
    // Snap to the route if we're plausibly on it.
    this.snapped = false
    this.snapOffset = [0, 0]
    this.drBase = this.est
    if (this.route && this.est) {
      const pr = this.route.project(this.est, this.along, 400)
      if (pr.offset <= this.cfg.snapMaxOffset) {
        this.snapped = true
        this.along = pr.along
        this.snapOffset = [this.est[0] - pr.point[0], this.est[1] - pr.point[1]]
        this.drBase = pr.point
      }
    }
    const reason = this.degradedReason ?? `no fix for ${(age / 1000).toFixed(1)} s`
    this.emit({
      kind: 'lost',
      t,
      message: `GNSS lost (${reason}) — dead reckoning from ${(this.speedAtLoss * 3.6).toFixed(0)} km/h, ${Math.round(this.headingAtLoss)}°${this.snapped ? ', constrained to route' : ''}`,
    })
  }

  /**
   * One dead-reckoning step. THIS is the function the trained model replaces:
   * speed ← learned IMU speed regressor, heading ← UKF yaw, matching ← HMM.
   */
  private propagateDeadReckoning(
    t: number,
    dt: number,
    motion: MotionState,
    from: LngLat,
    along: number,
    snapped: boolean,
    primary: boolean,
  ): { pos: LngLat; along: number; snapped: boolean; heading: number; source: HeadingSource } {
    // ── speed: decaying last-known speed, modulated by motion detection ──
    // Moving → hold speed almost constant (a car in a tunnel keeps going);
    // still → bleed it off quickly; no sensors → slow, conservative decay.
    const tau = motion === 'still' ? 1.5 : motion === 'moving' ? 240 : 45
    this.drSpeed *= Math.exp(-dt / tau)
    if (motion === 'moving' && this.drSpeed < this.speedAtLoss * 0.6) {
      // pulled away again after a stop: assume we resume near the prior cruise speed
      this.drSpeed = lerp(this.drSpeed, this.speedAtLoss * 0.8, 1 - Math.exp(-dt / 5))
    }
    const steps = this.stepSpeed(t)
    if (this.speedAtLoss < 3 && steps > 0 && motion !== 'still') {
      // walking: trust the step cadence more than the stale GNSS speed
      this.drSpeed = lerp(this.drSpeed, steps, 1 - Math.exp(-dt / 1.0))
    }
    if (motion === 'still' && this.drSpeed < 0.2) this.drSpeed = 0

    // ── heading ──
    let heading = this.headingAtLoss
    let source: HeadingSource = 'held'
    if (this.compassFresh(t)) {
      heading = normDeg(this.compass!.heading + this.compassOffset)
      source = 'compass'
    } else if (this.lastGyroT !== null && t - this.lastGyroT < 1500) {
      heading = normDeg(this.headingAtLoss + this.gyroYaw)
      source = 'gyro'
    }

    const dist = this.drSpeed * dt
    if (primary) this.drDistance += dist

    if (snapped && this.route) {
      let advance = dist
      if (primary && source !== 'held' && this.drSpeed > 1) {
        const routeBrg = this.route.pointAt(along).bearing
        const diverge = Math.abs(angleDiff(routeBrg, heading))
        if (diverge > 35) {
          this.offHeadingSince ??= t
          const held = t - this.offHeadingSince
          if (held > 800 && this.pendingShift === 0) {
            // Turn anchoring: the sensors say we turned, so we must be at a
            // corner. Find where the route takes that heading and slide there.
            const anchor = this.findHeadingMatch(along, heading)
            if (anchor !== null) {
              this.pendingShift = anchor - along
              this.offHeadingSince = null
            } else if (diverge > 75 && held > 8000) {
              this.emit({ kind: 'offroute', t, message: 'Heading diverged from route — releasing route constraint' })
              return { pos: destination(from, heading, dist), along, snapped: false, heading, source }
            }
          }
        } else {
          this.offHeadingSince = null
        }
      }
      if (primary && this.pendingShift !== 0) {
        if (this.pendingShift > 0) {
          // behind the corner: catch up at up to +12 m/s
          const s = Math.min(this.pendingShift, 12 * dt)
          advance += s
          this.pendingShift -= s
        } else {
          // past the corner: hold back by consuming forward progress (never reverse)
          const s = Math.min(-this.pendingShift, advance)
          advance -= s
          this.pendingShift += s
          if (advance === 0 && this.drSpeed < 0.5) this.pendingShift = 0
        }
      }
      const nextAlong = Math.min(this.route.length, along + advance)
      const pt = this.route.pointAt(nextAlong)
      return { pos: pt.point, along: nextAlong, snapped: true, heading: pt.bearing, source: 'route' }
    }
    return { pos: dist > 0 ? destination(from, heading, dist) : from, along, snapped: false, heading, source }
  }

  /** Nearest along-distance (−80 m … +250 m) where the route runs on `heading` (±25°). */
  private findHeadingMatch(along: number, heading: number): number | null {
    if (!this.route) return null
    let best: number | null = null
    for (let d = Math.max(0, along - 80); d <= Math.min(this.route.length, along + 250); d += 4) {
      if (Math.abs(angleDiff(this.route.pointAt(d).bearing, heading)) < 25) {
        if (best === null || Math.abs(d - along) < Math.abs(best - along)) best = d
      }
    }
    return best
  }

  private snapshot(t: number, mode: FixMode, motion: MotionState): EstimatorState {
    return {
      position: this.est,
      heading: this.estHeading,
      speed: this.estSpeed,
      mode,
      sigma: this.sigma,
      gnssAccuracy: this.lastGoodFix?.accuracy ?? null,
      rawFix: this.lastGoodFix ? [this.lastGoodFix.lon, this.lastGoodFix.lat] : null,
      fixAgeMs: this.lastGoodFix ? t - this.lastGoodFix.t : null,
      outageMs: this.drActive ? t - this.lastGoodBeforeOutageT : 0,
      drDistance: this.drActive ? this.drDistance : 0,
      motion,
      headingSource: this.headingSource,
      snapped: this.drActive && this.snapped,
      blending: this.blend !== null,
    }
  }

  /** Where along the active route the estimate currently is (metres). */
  get routeAlong(): number {
    return this.along
  }

  get drStartedAt(): number | null {
    return this.drActive ? this.drStartT : null
  }
}

function easeInOut(k: number): number {
  return k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2
}
