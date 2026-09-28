/**
 * SpeedNet — forward speed, its variance and a stationary flag from a 10 s
 * IMU window. Graph: ml/clew_ml/model.py (5 dilated Conv1d + GELU, then
 * [mean-pool ‖ last-step] → Linear → GELU → Linear). Runs on the shared
 * TinyNet runner, ~5 M multiply-adds per call, once per second.
 */
import { TinyNet, type TinyNetManifest, softplus, sigmoid } from './tinynet'

export interface SpeedNetManifest extends TinyNetManifest {
  channels: string[]
  window: number
  hz: number
}

export interface SpeedPrediction {
  speed: number
  variance: number
  pStill: number
}

export class SpeedNet {
  readonly channels: string[]
  readonly window: number
  private net: TinyNet

  constructor(manifest: SpeedNetManifest, weights: Float32Array) {
    this.net = new TinyNet(manifest, weights)
    this.channels = manifest.channels
    this.window = manifest.window
  }

  static async load(base = '/ml/'): Promise<SpeedNet> {
    const t = await TinyNet.load(base, 'speednet')
    return SpeedNet.fromNet(t)
  }

  static fromNet(t: TinyNet): SpeedNet {
    const s = Object.create(SpeedNet.prototype) as SpeedNet
    const m = t.manifest as SpeedNetManifest
    Object.assign(s, { net: t, channels: m.channels, window: m.window })
    return s
  }

  /** x: channel-major (C × window) raw features. */
  predict(x: Float32Array): SpeedPrediction {
    const o = this.net.run(x, this.window)
    const logvar = Math.min(6, Math.max(-4, o[1])) + Math.log(this.net.buffer('var_scale')![0])
    return { speed: softplus(o[0]), variance: Math.exp(logvar), pStill: sigmoid(o[2]) }
  }
}
