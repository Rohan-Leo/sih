// npx jiti scripts/replay-check.ts — run the recorded drive through both engines, headless
import { readFileSync } from 'node:fs'
import { SpeedNet } from '../src/ml/speednet'
import { LearnedEstimator } from '../src/ml/learnedEstimator'
import { ReplaySource } from '../src/engine/replaySource'

const bin = (p: string) => {
  const b = readFileSync(p)
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4)
}
const net = new SpeedNet(JSON.parse(readFileSync('public/ml/speednet.json', 'utf8')), bin('public/ml/speednet.bin'))
const manifest = JSON.parse(readFileSync('public/ml/replay-y1-3.json', 'utf8'))
const data = bin('public/ml/replay-y1-3.bin')
const manualAt = process.argv[2] ? Number(process.argv[2]) * 1000 : -1

for (const learned of [true, false]) {
  const src = new ReplaySource(manifest, data)
  const est = new LearnedEstimator()
  est.net = net
  est.configure({ enabled: learned, routeConstraint: false })
  const log: string[] = []
  est.onEvent((e) => log.push(`[${(e.t / 1000).toFixed(0)}s] ${e.message}`))
  const sink = { pushFix: (f: any) => est.pushFix(f), pushMotion: (m: any) => est.pushMotion(m), pushHeading: () => {}, gnssError: () => {} }
  const errs: Record<string, number[]> = {}
  let note = ''
  const t0 = performance.now()
  for (let t = 0; t <= src.duration; t += 50) {
    if (manualAt >= 0 && Math.abs(t - manualAt) < 25) src.toggleManualOutage(t)
    src.advance(t, sink)
    const s = est.tick(t)
    if (s.engine !== note) log.push(`   engine → ${s.engine.toUpperCase()} (${s.engineNote}) @${(t / 1000).toFixed(0)}s`)
    note = s.engine
    const z = src.deadZoneAt(t)
    if (s.mode === 'DR') (errs[z?.label ?? 'manual'] ??= []).push(src.errorAgainstTruth(s.position, t)!)
  }
  console.log(`\n=== ${learned ? 'LEARNED (UKF + SpeedNet)' : 'HEURISTIC'} — ${((performance.now() - t0) / 1000).toFixed(1)} s CPU for ${(src.duration / 60000).toFixed(0)} min of drive`)
  log.forEach((l) => console.log(l))
  for (const [k, v] of Object.entries(errs)) console.log(`${k}: error at end ${v[v.length - 1].toFixed(1)} m, max ${Math.max(...v).toFixed(1)} m over ${(v.length * 0.05).toFixed(0)} s of DR`)
}
