/**
 * IntegrityNet — P(this GNSS fix is faulty), from how well it agrees with the
 * IMU dead reckoning since the last fix we trusted. Mirrors
 * ml/clew_ml/integrity.py (FixChecker + FEATURES) line for line.
 * Positions are local east/north metres, times seconds, angles clockwise.
 */
import { SpeedCorrector } from './speedCorrector'
import { TinyNet, sigmoid } from './tinynet'

const wrap180 = (a: number) => ((((a + 180) % 360) + 360) % 360) - 180

export class FixChecker {
  private good = false
  private pred: [number, number] = [0, 0]
  private psi = 0
  private v = 0
  private dtGood = 0
  private prev: [number, number, number] | null = null
  private accs: number[] = []
  readonly corr = new SpeedCorrector()

  /** 10 Hz: propagate the dead-reckoned expectation. vHat is the latest SpeedNet speed. */
  imuStep(yawRate: number, vHat: number | null, dt = 0.1) {
    if (!this.good) return
    this.dtGood += dt
    if (vHat !== null && Number.isFinite(vHat)) this.v = this.corr.k * vHat
    this.psi += yawRate * dt
    this.pred = [this.pred[0] + this.v * Math.sin(this.psi) * dt, this.pred[1] + this.v * Math.cos(this.psi) * dt]
  }

  features(E: number, N: number, t: number, acc: number, speed: number, course: number, vHat: number | null): number[] | null {
    if (!this.good) return null
    const innov = Math.hypot(E - this.pred[0], N - this.pred[1])
    const spread = Math.sqrt(Math.max(acc, 2) ** 2 + (1 + 0.1 * this.v * this.dtGood) ** 2)
    const sorted = [...this.accs].sort((a, b) => a - b)
    const m = sorted.length
    const med = m ? (m % 2 ? sorted[(m - 1) / 2] : 0.5 * (sorted[m / 2 - 1] + sorted[m / 2])) : acc
    const vi = vHat !== null && Number.isFinite(vHat) ? this.corr.k * vHat : this.v
    const cres = speed > 3 && this.v > 3 ? Math.abs(wrap180(course - (this.psi * 180) / Math.PI)) / 180 : 0
    let ires = 0
    if (this.prev && t - this.prev[2] > 0.2) {
      ires = Math.abs(Math.hypot(E - this.prev[0], N - this.prev[1]) / (t - this.prev[2]) - speed)
    }
    return [
      Math.log1p(innov),
      Math.log1p(innov / spread),
      Math.log(Math.max(acc, 0.5)),
      acc / Math.max(med, 0.5),
      Math.log1p(Math.abs(speed - vi)),
      cres,
      Math.log1p(ires),
      Math.min(this.dtGood, 60) / 60,
      speed / 30,
    ]
  }

  afterFix(E: number, N: number, t: number, acc: number, speed: number, course: number, vHat: number | null, trusted: boolean) {
    this.prev = [E, N, t]
    if (!trusted) return
    if (vHat !== null && Number.isFinite(vHat)) this.corr.observe(speed, vHat)
    this.good = true
    this.pred = [E, N]
    this.psi = (course * Math.PI) / 180
    this.v = speed
    this.dtGood = 0
    this.accs.push(acc)
    if (this.accs.length > 20) this.accs.shift()
  }

  get secondsSinceTrusted(): number {
    return this.dtGood
  }

  clone(): FixChecker {
    const c = Object.assign(Object.create(FixChecker.prototype), this) as FixChecker
    Object.assign(c, {
      pred: [...this.pred],
      prev: this.prev ? [...this.prev] : null,
      accs: [...this.accs],
      corr: Object.assign(new SpeedCorrector(), this.corr),
    })
    return c
  }
}

export class IntegrityNet {
  constructor(readonly net: TinyNet) {}

  /** Probability that the fix is faulty. */
  pFault(features: number[]): number {
    return sigmoid(this.net.run(features)[0])
  }
}
