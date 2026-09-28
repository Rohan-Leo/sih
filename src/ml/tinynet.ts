/**
 * TinyNet — one dependency-free runner for every Clew model in the browser.
 *
 * All five models share the layout of ml/clew_ml/nets.py:
 *   mean, std      input normalisation (models take raw features)
 *   body.{0,2,…}   Conv1d, dilation 1, 2, 4, …, each followed by GELU   (TCN encoder, optional)
 *   head.{i}       Linear layers with GELU between them                  (task head)
 * A TCN encoder hands the head [mean over time ‖ last time step].
 *
 * Weights are float32 blobs exported by `python -m clew_ml.export_web`
 * (public/ml/<name>.bin + <name>.json). Model-specific post-processing
 * (softplus, sigmoid, quantiles…) lives with each model's wrapper.
 */

interface TensorInfo {
  name: string
  shape: number[]
  offset: number
  size: number
}

export interface TinyNetManifest {
  tensors: TensorInfo[]
  [meta: string]: unknown
}

// erf via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7) — torch's GELU is the exact erf form
function erf(x: number): number {
  const s = x < 0 ? -1 : 1
  const a = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * a)
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)
  return s * y
}
export const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2))
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
export const softplus = (x: number) => Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0)

interface Conv {
  weight: Float32Array
  bias: Float32Array
  cout: number
  cin: number
  k: number
  dil: number
}
interface Dense {
  weight: Float32Array
  bias: Float32Array
  nout: number
  nin: number
}

export class TinyNet {
  readonly manifest: TinyNetManifest
  private w: Record<string, { data: Float32Array; shape: number[] }> = {}
  private convs: Conv[] = []
  private dense: Dense[] = []

  constructor(manifest: TinyNetManifest, weights: Float32Array) {
    this.manifest = manifest
    for (const t of manifest.tensors) {
      this.w[t.name] = { data: weights.subarray(t.offset, t.offset + t.size), shape: t.shape }
    }
    for (let i = 0, dil = 1; this.w[`body.${i}.weight`]; i += 2, dil *= 2) {
      const W = this.w[`body.${i}.weight`]
      this.convs.push({ weight: W.data, bias: this.w[`body.${i}.bias`].data, cout: W.shape[0], cin: W.shape[1], k: W.shape[2], dil })
    }
    const heads = Object.keys(this.w)
      .map((k) => /^head\.(\d+)\.weight$/.exec(k))
      .filter((m): m is RegExpExecArray => m !== null)
      .map((m) => Number(m[1]))
      .sort((a, b) => a - b)
    for (const i of heads) {
      const W = this.w[`head.${i}.weight`]
      this.dense.push({ weight: W.data, bias: this.w[`head.${i}.bias`].data, nout: W.shape[0], nin: W.shape[1] })
    }
  }

  static async load(base: string, name: string): Promise<TinyNet> {
    const [m, b] = await Promise.all([
      fetch(`${base}${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`${name}.json: HTTP ${r.status}`)
        return r.json() as Promise<TinyNetManifest>
      }),
      fetch(`${base}${name}.bin`).then((r) => {
        if (!r.ok) throw new Error(`${name}.bin: HTTP ${r.status}`)
        return r.arrayBuffer()
      }),
    ])
    return new TinyNet(m, new Float32Array(b))
  }

  /** A stored buffer such as `var_scale`. */
  buffer(name: string): Float32Array | null {
    return this.w[name]?.data ?? null
  }

  get isSequence(): boolean {
    return this.convs.length > 0
  }

  /**
   * Raw head outputs. For a TCN, x is channel-major (C × T) raw features;
   * for a tabular model, x is the raw feature vector.
   */
  run(x: Float32Array | number[], T = 1): Float32Array {
    const mean = this.w.mean.data
    const std = this.w.std.data
    const C = mean.length
    let z: Float32Array
    if (this.convs.length) {
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
      z = new Float32Array(2 * width)
      for (let c = 0; c < width; c++) {
        let s = 0
        for (let t = 0; t < T; t++) s += h[c * T + t]
        z[c] = s / T
        z[width + c] = h[c * T + T - 1]
      }
    } else {
      z = new Float32Array(C)
      for (let c = 0; c < C; c++) z[c] = (x[c] - mean[c]) / std[c]
    }
    this.dense.forEach((L, li) => {
      const y = new Float32Array(L.nout)
      const act = li < this.dense.length - 1
      for (let o = 0; o < L.nout; o++) {
        let a = L.bias[o]
        for (let i = 0; i < L.nin; i++) a += L.weight[o * L.nin + i] * z[i]
        y[o] = act ? gelu(a) : a
      }
      z = y
    })
    return z
  }
}
