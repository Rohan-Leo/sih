/**
 * Online bias correction for the learned speed — port of evaluate.SpeedCorrector.
 * While GNSS is good it fits v_gnss ≈ k·v̂ with ~2 min memory, plus the residual
 * spread; during an outage the filter receives k·v̂ with that spread as noise.
 */
export class SpeedCorrector {
  sxy = 0
  sxx = 0
  sr = 0
  n = 0

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
