/**
 * SpeedNet inference in plain TypeScript — no ML runtime needed in the browser.
 *
 * Same graph as ml/clew_ml/model.py: 5 dilated Conv1d (+GELU) layers, then
 * [mean-pool ‖ last-step] → Linear → GELU → Linear → [speed, log-variance,
 * stationary logit]. Weights come from public/ml/speednet.{json,bin}
 * (exported by `python -m clew_ml.export_web`). ~5 M multiply-adds per call;
 * we run it once per second.
 */

interface TensorInfo {
  name: string
  shape: number[]
  offset: number
  size: number
}

export interface SpeedNetManifest {
  channels: string[]
  window: number
  hz: number
  tensors: TensorInfo[]
}

export interface SpeedPrediction {
  speed: number
  variance: number
  pStill: number
}

// erf via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7) — torch's GELU is the exact erf form
function erf(x: number): number {
  const s = x < 0 ? -1 : 1
  const a = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * a)
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)
  return s * y
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2))

export class SpeedNet {
  readonly channels: string[]
  readonly window: number
  private w: Record<string, { data: Float32Array; shape: number[] }> = {}
  private convs: { weight: Float32Array; bias: Float32Array; cout: number; cin: number; k: number; dil: number }[] = []

  constructor(manifest: SpeedNetManifest, weights: Float32Array) {
    this.channels = manifest.channels
    this.window = manifest.window
    for (const t of manifest.tensors) {
      this.w[t.name] = { data: weights.subarray(t.offset, t.offset + t.size), shape: t.shape }
    }
    for (let i = 0, dil = 1; this.w[`body.${i}.weight`]; i += 2, dil *= 2) {
      const W = this.w[`body.${i}.weight`]
      this.convs.push({ weight: W.data, bias: this.w[`body.${i}.bias`].data, cout: W.shape[0], cin: W.shape[1], k: W.shape[2], dil })
    }
  }

  static async load(base = '/ml/'): Promise<SpeedNet> {
    const [m, b] = await Promise.all([
      fetch(`${base}speednet.json`).then((r) => r.json() as Promise<SpeedNetManifest>),
      fetch(`${base}speednet.bin`).then((r) => r.arrayBuffer()),
    ])
    return new SpeedNet(m, new Float32Array(b))
  }

  /** x: channel-major (C × window) raw features. */
  predict(x: Float32Array): SpeedPrediction {
    const C = this.channels.length
    const T = this.window
    const mean = this.w.mean.data
    const std = this.w.std.data
    let h = new Float32Array(C * T)
    for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) h[c * T + t] = (x[c * T + t] - mean[c]) / std[c]

    for (const L of this.convs) {
      const out = new Float32Array(L.cout * T)
      const pad = ((L.k - 1) / 2) * L.dil // "same" padding
      for (let o = 0; o < L.cout; o++) {
        const base = o * L.cin * L.k
        for (let t = 0; t < T; t++) {
          let acc = L.bias[o]
          for (let i = 0; i < L.cin; i++) {
            const wi = base + i * L.k
            const hi = i * T
            for (let k = 0; k < L.k; k++) {
              const s = t + k * L.dil - pad
              if (s >= 0 && s < T) acc += L.weight[wi + k] * h[hi + s]
            }
          }
          out[o * T + t] = gelu(acc)
        }
      }
      h = out
    }

    const width = this.convs[this.convs.length - 1].cout
    const z = new Float32Array(2 * width)
    for (let c = 0; c < width; c++) {
      let s = 0
      for (let t = 0; t < T; t++) s += h[c * T + t]
      z[c] = s / T
      z[width + c] = h[c * T + T - 1]
    }
    const dense = (inp: Float32Array, name: string, act: boolean) => {
      const W = this.w[`${name}.weight`]
      const B = this.w[`${name}.bias`].data
      const [nout, nin] = W.shape
      const y = new Float32Array(nout)
      for (let o = 0; o < nout; o++) {
        let a = B[o]
        for (let i = 0; i < nin; i++) a += W.data[o * nin + i] * inp[i]
        y[o] = act ? gelu(a) : a
      }
      return y
    }
    const o = dense(dense(z, 'head.0', true), 'head.3', false)
    const speed = Math.log1p(Math.exp(-Math.abs(o[0]))) + Math.max(o[0], 0) // softplus
    const logvar = Math.min(6, Math.max(-4, o[1])) + Math.log(this.w.var_scale.data[0])
    return { speed, variance: Math.exp(logvar), pStill: 1 / (1 + Math.exp(-o[2])) }
  }
}
