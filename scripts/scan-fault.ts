// npx jiti scripts/scan-fault.ts — slide the multipath zone along the replay drive and compare the
// learned engine (IntegrityNet rejects the fixes, dead reckoning) with the heuristic (follows them).
import { readFileSync, existsSync } from 'node:fs'
import { LearnedEstimator } from '../src/ml/learnedEstimator'
import { MODEL_FILES, modelsFrom } from '../src/ml/models'
import { TinyNet } from '../src/ml/tinynet'
import { ReplaySource } from '../src/engine/replaySource'
const bin = (p: string) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) }
const nets = Object.fromEntries(MODEL_FILES.filter((n) => existsSync(`public/ml/${n}.bin`)).map((n) => [n, new TinyNet(JSON.parse(readFileSync(`public/ml/${n}.json`, 'utf8')), bin(`public/ml/${n}.bin`))]))
const models = modelsFrom(nets)
const base = JSON.parse(readFileSync('public/ml/replay-y1-3.json', 'utf8'))
const data = bin('public/ml/replay-y1-3.bin')
for (let f0 = 4400; f0 <= 5700; f0 += 100) {
  const m = { ...base, faultZones: [{ ...base.faultZones[0], from: f0, to: f0 + 250 }] }
  const row: string[] = []
  for (const learned of [true, false]) {
    const src = new ReplaySource(m, data)
    const est = new LearnedEstimator(); est.models = models; est.configure({ enabled: learned, routeConstraint: false })
    const sink = { pushFix: (f: any) => est.pushFix(f), pushMotion: (x: any) => est.pushMotion(x), pushHeading: () => {}, gnssError: () => {} }
    let worst = 0, end = 0
    for (let t = 0; t <= (f0 + 260) * 100; t += 50) { src.advance(t, sink); const s = est.tick(t); if (t >= f0 * 100 && t < (f0 + 250) * 100) { end = src.errorAgainstTruth(s.position, t)!; worst = Math.max(worst, end) } }
    row.push(`${learned ? 'learned' : 'heuristic'} end ${end.toFixed(0)} max ${worst.toFixed(0)}`)
  }
  console.log(f0, row.join(' | '))
}
