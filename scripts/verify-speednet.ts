// npx jiti scripts/verify-speednet.ts — TS SpeedNet vs PyTorch reference outputs
import { readFileSync } from 'node:fs'
import { SpeedNet } from '../src/ml/speednet'

const manifest = JSON.parse(readFileSync('public/ml/speednet.json', 'utf8'))
const buf = readFileSync('public/ml/speednet.bin')
const net = new SpeedNet(manifest, new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4))
const vec = JSON.parse(readFileSync('src/ml/speednet.vectors.json', 'utf8'))
let worst = 0
const t0 = performance.now()
vec.inputs.forEach((x: number[][], i: number) => {
  const p = net.predict(new Float32Array(x.flat()))
  const [s, lv, st] = vec.outputs[i]
  const ref = { speed: s, variance: Math.exp(lv), pStill: 1 / (1 + Math.exp(-st)) }
  const d = Math.max(Math.abs(p.speed - ref.speed), Math.abs(p.variance - ref.variance) / ref.variance, Math.abs(p.pStill - ref.pStill))
  worst = Math.max(worst, d)
  console.log(`#${i} ts speed ${p.speed.toFixed(4)} torch ${ref.speed.toFixed(4)} | var ${p.variance.toFixed(3)} vs ${ref.variance.toFixed(3)} | pStill ${p.pStill.toFixed(4)} vs ${ref.pStill.toFixed(4)}`)
})
console.log(`max deviation ${worst.toExponential(2)}; ${((performance.now() - t0) / vec.inputs.length).toFixed(1)} ms per inference`)
if (worst > 1e-3) process.exit(1)
