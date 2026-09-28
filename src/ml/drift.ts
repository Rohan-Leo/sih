/**
 * DriftNet — learned error radius of the dead-reckoned position.
 * OutageTracker mirrors ml/clew_ml/drift.py; the net outputs log1p-error
 * quantiles (50 / 68 / 95 %), made monotone with softplus increments, then
 * shifted by conformal offsets calibrated on validation drives.
 */
import { TinyNet, softplus } from './tinynet'

export class OutageTracker {
  private t = 0
  private dist = 0
  private turn = 0

  constructor(private readonly psi0: number) {}

  step(v: number, omega: number, dt = 0.1) {
    this.t += dt
    this.dist += v * dt
    this.turn += Math.abs(omega) * dt
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
      Math.log(Math.max(sigma, 0.5)),
    ]
  }
}

export interface DriftRadii {
  r50: number
  r68: number
  r95: number
}

export class DriftNet {
  private readonly offsets: number[]

  constructor(readonly net: TinyNet) {
    // conformal offsets fitted on validation drives (drift.py), so the radii cover their nominal share
    this.offsets = Array.from(net.buffer('offsets') ?? [0, 0, 0])
  }

  radii(features: number[]): DriftRadii {
    const o = this.net.run(features)
    const q50 = o[0] + this.offsets[0]
    const q68 = Math.max(q50, o[0] + softplus(o[1]) + this.offsets[1])
    const q95 = Math.max(q68, o[0] + softplus(o[1]) + softplus(o[2]) + this.offsets[2])
    return { r50: Math.expm1(q50), r68: Math.expm1(q68), r95: Math.expm1(q95) }
  }
}
