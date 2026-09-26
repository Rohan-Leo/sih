/**
 * Online phone-mount calibration — port of ml/clew_ml/calib.py.
 *
 * Between consecutive GNSS fixes (0.5–2.5 s apart) we average the IMU and
 * regress, with no assumptions about axes:
 *   GNSS heading rate      ≈ gyro · w_yaw
 *   GNSS speed change      ≈ (acc − gravity) · u_fwd
 *   centripetal v·ψ̇        ≈ (acc − gravity) · u_lat
 * It freezes after `targetIntervals` (≈5 min of driving at 1 Hz GNSS), which
 * matches how the speed model's training features were calibrated.
 */
type V3 = [number, number, number]

class RidgeStats {
  xtx = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  xty = [0, 0, 0]
  n = 0
  add(x: V3, y: number) {
    for (let i = 0; i < 3; i++) {
      this.xty[i] += x[i] * y
      for (let j = 0; j < 3; j++) this.xtx[i * 3 + j] += x[i] * x[j]
    }
    this.n++
  }
  solve(lam = 1e-4): V3 {
    if (this.n <= 20) return [0, 0, 0]
    const A = this.xtx.slice()
    const tr = (A[0] + A[4] + A[8]) / 3
    A[0] += lam * tr
    A[4] += lam * tr
    A[8] += lam * tr
    return solve3(A, this.xty)
  }
}

function solve3(A: number[], b: number[]): V3 {
  const [a, bb, c, d, e, f, g, h, i] = A
  const det = a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g)
  if (Math.abs(det) < 1e-18) return [0, 0, 0]
  const inv = [
    e * i - f * h, c * h - bb * i, bb * f - c * e,
    f * g - d * i, a * i - c * g, c * d - a * f,
    d * h - e * g, bb * g - a * h, a * e - bb * d,
  ].map((v) => v / det)
  return [
    inv[0] * b[0] + inv[1] * b[1] + inv[2] * b[2],
    inv[3] * b[0] + inv[4] * b[1] + inv[5] * b[2],
    inv[6] * b[0] + inv[7] * b[1] + inv[8] * b[2],
  ]
}

export interface MountCalib {
  wYaw: V3
  uFwd: V3
  uLat: V3
}

export class OnlineMountCalibrator {
  private yaw = new RidgeStats()
  private long = new RidgeStats()
  private lat = new RidgeStats()
  private gyroSum: V3 = [0, 0, 0]
  private dynSum: V3 = [0, 0, 0]
  private count = 0
  private prev: { t: number; speed: number; course: number | null } | null = null
  result: MountCalib | null = null

  constructor(readonly targetIntervals = 299) {}

  get progress(): number {
    return Math.min(1, this.long.n / this.targetIntervals)
  }

  /** Every 10 Hz IMU row. */
  addImu(gyro: V3, dyn: V3) {
    if (this.result) return
    for (let i = 0; i < 3; i++) {
      this.gyroSum[i] += gyro[i]
      this.dynSum[i] += dyn[i]
    }
    this.count++
  }

  /** Every accepted GNSS fix (ms clock). */
  addFix(t: number, speed: number, course: number | null) {
    if (this.result) return
    const p = this.prev
    // fixes closer than 1 s are skipped (some receivers repeat fixes at 10 Hz)
    if (p && t - p.t < 1000 - 50) return
    if (p && this.count > 0) {
      const dt = (t - p.t) / 1000
      if (dt >= 0.5 && dt <= 2.5) {
        const g = this.gyroSum.map((v) => v / this.count) as V3
        const a = this.dynSum.map((v) => v / this.count) as V3
        this.long.add(a, (speed - p.speed) / dt)
        if (Math.min(speed, p.speed) > 5 && course !== null && p.course !== null) {
          const dpsi = ((((course - p.course + 540) % 360) + 360) % 360) - 180
          if (Math.abs(dpsi) < 45) {
            const rate = ((dpsi * Math.PI) / 180) / dt
            this.yaw.add(g, rate)
            this.lat.add(a, 0.5 * (speed + p.speed) * rate)
          }
        }
      }
    }
    this.prev = { t, speed, course }
    this.gyroSum = [0, 0, 0]
    this.dynSum = [0, 0, 0]
    this.count = 0
    if (this.long.n >= this.targetIntervals) {
      this.result = { wYaw: this.yaw.solve(), uFwd: this.long.solve(), uLat: this.lat.solve() }
    }
  }
}
