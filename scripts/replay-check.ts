// npx jiti scripts/replay-check.ts — run the recorded drive through both engines, headless
import { readFileSync } from 'node:fs'
import { existsSync } from 'node:fs'
import { LearnedEstimator } from '../src/ml/learnedEstimator'
import { MODEL_FILES, modelsFrom } from '../src/ml/models'
import { TinyNet } from '../src/ml/tinynet'
import { ReplaySource } from '../src/engine/replaySource'

const bin = (p: string) => {
  const b = readFileSync(p)
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4)
}
const nets = Object.fromEntries(
  MODEL_FILES.filter((n) => existsSync(`public/ml/${n}.bin`)).map((n) => [n, new TinyNet(JSON.parse(readFileSync(`public/ml/${n}.json`, 'utf8')), bin(`public/ml/${n}.bin`))]),
)
const models = modelsFrom(nets)
console.log(`models: ${Object.keys(nets).join(', ')}${models.heading?.useInFilter ? ' (HeadingNet in filter)' : ''}`)
const manifest = JSON.parse(readFileSync('public/ml/replay-y1-3.json', 'utf8'))
const data = bin('public/ml/replay-y1-3.bin')
const manualAt = process.argv[2] ? Number(process.argv[2]) * 1000 : -1

for (const learned of [true, false]) {
  const src = new ReplaySource(manifest, data)
  const est = new LearnedEstimator()
  est.models = models
  est.configure({ enabled: learned, routeConstraint: false })
  const log: string[] = []
  est.onEvent((e) => log.push(`[${(e.t / 1000).toFixed(0)}s] ${e.message}`))
  const sink = { pushFix: (f: any) => est.pushFix(f), pushMotion: (m: any) => est.pushMotion(m), pushHeading: () => {}, gnssError: () => {} }
  const errs: Record<string, number[]> = {}
  const ml = { states: {} as Record<string, number>, lowTrust: 0, r95: [] as [number, number][] }
  let note = ''
  const t0 = performance.now()
  for (let t = 0; t <= src.duration; t += 50) {
    if (manualAt >= 0 && Math.abs(t - manualAt) < 25) src.toggleManualOutage(t)
    src.advance(t, sink)
    const s = est.tick(t)
    if (s.engine !== note) log.push(`   engine → ${s.engine.toUpperCase()} (${s.engineNote}) @${(t / 1000).toFixed(0)}s`)
    note = s.engine
    const z = src.deadZoneAt(t)
    const err = src.errorAgainstTruth(s.position, t)!
    if (s.mode === 'DR' || z?.kind === 'fault') (errs[`${z?.label ?? 'manual'}${s.mode === 'DR' ? '' : ' (GNSS mode)'}`] ??= []).push(err)
    if (s.ml && t % 1000 === 0) {
      if (s.ml.driving) ml.states[s.ml.driving] = (ml.states[s.ml.driving] ?? 0) + 1
      if (s.ml.gnssTrust !== null && s.ml.gnssTrust < 0.5) ml.lowTrust++
      if (s.ml.radius95 !== null) ml.r95.push([err, s.ml.radius95])
    }
  }
  console.log(`\n=== ${learned ? 'LEARNED (UKF + SpeedNet)' : 'HEURISTIC'} — ${((performance.now() - t0) / 1000).toFixed(1)} s CPU for ${(src.duration / 60000).toFixed(0)} min of drive`)
  log.forEach((l) => console.log(l))
  for (const [k, v] of Object.entries(errs)) console.log(`${k}: error at end ${v[v.length - 1].toFixed(1)} m, max ${Math.max(...v).toFixed(1)} m over ${(v.length * 0.05).toFixed(0)} s`)
  if (learned) {
    console.log(`MotionNet seconds per state: ${JSON.stringify(ml.states)}`)
    console.log(`IntegrityNet: ${ml.lowTrust} s with GNSS trust < 50%`)
    if (ml.r95.length) console.log(`DriftNet: true error inside the 95% radius ${((100 * ml.r95.filter(([e, r]) => e <= r).length) / ml.r95.length).toFixed(0)}% of ${ml.r95.length} DR seconds`)
  }
}
