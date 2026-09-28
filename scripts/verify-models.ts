// npx jiti scripts/verify-models.ts — the TypeScript TinyNet runner vs PyTorch for the other four models
import { readFileSync } from 'node:fs'
import { TinyNet } from '../src/ml/tinynet'

const vec = JSON.parse(readFileSync('src/ml/models.vectors.json', 'utf8'))
let worst = 0
for (const name of ['headingnet', 'motionnet', 'integritynet', 'driftnet']) {
  const buf = readFileSync(`public/ml/${name}.bin`)
  const net = new TinyNet(JSON.parse(readFileSync(`public/ml/${name}.json`, 'utf8')), new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4))
  const t0 = performance.now()
  let dev = 0
  vec[name].inputs.forEach((x: number[] | number[][], i: number) => {
    const seq = Array.isArray(x[0])
    const flat = seq ? (x as number[][]).flat() : (x as number[])
    const out = net.run(new Float32Array(flat), seq ? (x as number[][])[0].length : 1)
    vec[name].outputs[i].forEach((r: number, j: number) => (dev = Math.max(dev, Math.abs(out[j] - r) / Math.max(1, Math.abs(r)))))
  })
  worst = Math.max(worst, dev)
  console.log(`${name.padEnd(13)} max deviation ${dev.toExponential(2)}; ${((performance.now() - t0) / vec[name].inputs.length).toFixed(2)} ms per inference`)
}
if (worst > 1e-3) process.exit(1)
