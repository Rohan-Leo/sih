import type { GnssFix, HeadingSample, MotionSample } from './positionEstimator'

/** Where raw sensor readings go. Both Live and Demo sources feed the same sink. */
export interface SensorSink {
  pushFix(f: GnssFix): void
  pushMotion(m: MotionSample): void
  pushHeading(h: HeadingSample): void
  gnssError(reason: string): void
}
