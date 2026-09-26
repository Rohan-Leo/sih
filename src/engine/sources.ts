import type { LngLat, Polyline } from '../lib/geo'
import type { GnssFix, HeadingSample, MotionSample } from './positionEstimator'

/** Where raw sensor readings go. Both Live and Demo sources feed the same sink. */
export interface SensorSink {
  pushFix(f: GnssFix): void
  pushMotion(m: MotionSample): void
  pushHeading(h: HeadingSample): void
  gnssError(reason: string): void
}

export interface DeadZone {
  /** metres along the scripted path */
  from: number
  to: number
  label: string
}

/** A timed sensor feed for Demo mode (synthetic route or recorded drive). */
export interface ScriptedSource {
  readonly path: Polyline
  readonly duration: number
  readonly deadZones: DeadZone[]
  advance(t: number, sink: SensorSink): void
  truthAt(t: number): { pos: LngLat; speed: number; course: number; along: number }
  deadZoneAt(t: number): DeadZone | null
  gnssWithheld(t: number): boolean
  toggleManualOutage(now: number, ms?: number): void
  manualOutageRemaining(now: number): number
  errorAgainstTruth(est: LngLat | null, t: number): number | null
  reset(): void
}
