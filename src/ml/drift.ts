/**
 * DriftNet — learned error radius of the dead-reckoned position.
 * OutageTracker mirrors ml/clew_ml/drift.py; the net outputs log1p-error
 * quantiles (50 / 68 / 95 %), made monotone with softplus increments.
 */
import type { SpeedCorrector } from './speedCorrector'
import { TinyNet, softplus } from './tinynet'

export class OutageTracker {
  private t = 0
  private dist = 0
  private turn = 0
  private sdSum = 0
  private sdN = 0
  private still = 0
  private readonly corrResid: number

  constructor(
    private readonly psi0: number,
    corr: SpeedCorrector,
    private readonly r2Yaw: number,
    private readonly r2Long: number,
  ) {
    this.corrResid = corr.residual ?? 3
  }

  step(v: number, omega: number, still: boolean, dt = 0.1) {
    this.t += dt
    this.dist += v * dt
    this.turn += Math.abs(omega) * dt
    if (still) this.still += dt
  }

  speedObs(variance: number) {
    if (Number.isFinite(variance)) {
      this.sdSum += Math.sqrt(variance)
      this.sdN++
    }
  }

  features(psi: number, sigma: number): number[] {
    const d = psi - this.psi0 + Math.PI
    const dpsi = Math.abs((((d % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI)
    return [
      this.t / 120,
      Math.log1p(this.dist),
      this.dist / Math.max(this.t, 1e-3) / 30,
      dpsi,
      this.turn,
      this.sdN ? this.sdSum / this.sdN : 3,
      this.corrResid,
      Math.log(Math.max(sigma, 0.5)),
      this.still / Math.max(this.t, 1e-3),
      this.r2Yaw,
      this.r2Long,
    ]
  }
}

export interface DriftRadii {
  r50: number
  r68: number
  r95: number
}

export class DriftNet {
  constructor(readonly net: TinyNet) {}

  radii(features: number[]): DriftRadii {
    const o = this.net.run(features)
    const q50 = o[0]
    const q68 = q50 + softplus(o[1])
    const q95 = q68 + softplus(o[2])
    return { r50: Math.expm1(q50), r68: Math.expm1(q68), r95: Math.expm1(q95) }
  }
}
