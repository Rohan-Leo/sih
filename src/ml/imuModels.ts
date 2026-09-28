/**
 * The two IMU-window models next to SpeedNet (ml/clew_ml/imu_models.py).
 * Both read the same 10 s mount-invariant feature window as SpeedNet.
 */
import { TinyNet } from './tinynet'

export class HeadingNet {
  /** whether the benchmark showed the correction helps; set at export time */
  readonly useInFilter: boolean
  private readonly varScale: number

  constructor(readonly net: TinyNet) {
    this.useInFilter = net.manifest.use_in_filter === true
    this.varScale = net.buffer('var_scale')?.[0] ?? 1
  }

  /**
   * δ ≈ true − calibrated yaw rate over the last 10 s (rad/s), and its variance.
   * The filter's gyro bias is −δ; with useInFilter it is fed to the UKF as a measurement.
   */
  predict(x: Float32Array, window: number): { delta: number; variance: number } {
    const o = this.net.run(x, window)
    return { delta: o[0] * 0.1, variance: Math.exp(Math.min(2, Math.max(-12, o[1]))) * this.varScale }
  }
}

export type DrivingState = 'stationary' | 'cruising' | 'accelerating' | 'braking' | 'turning'

export class MotionNet {
  readonly classes: DrivingState[]

  constructor(readonly net: TinyNet) {
    this.classes = (net.manifest.classes as DrivingState[]) ?? ['stationary', 'cruising', 'accelerating', 'braking', 'turning']
  }

  predict(x: Float32Array, window: number): { state: DrivingState; p: number; probs: number[] } {
    const o = this.net.run(x, window)
    const m = Math.max(...o)
    const e = Array.from(o, (v) => Math.exp(v - m))
    const s = e.reduce((a, b) => a + b, 0)
    const probs = e.map((v) => v / s)
    const i = probs.indexOf(Math.max(...probs))
    return { state: this.classes[i], p: probs[i], probs }
  }
}
