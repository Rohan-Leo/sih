// npx jiti scripts/verify-features.ts — TS FixChecker / OutageTracker vs the Python originals
import { readFileSync } from 'node:fs'
import { OutageTracker } from '../src/ml/drift'
import { FixChecker } from '../src/ml/integrity'
import { SpeedCorrector } from '../src/ml/speedCorrector'

const vec = JSON.parse(readFileSync('src/ml/models.vectors.json', 'utf8'))
const diff = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i]) / Math.max(1, Math.abs(b[i]))))

let worst = 0
let nFix = 0
const ck = new FixChecker()
for (const op of vec.fixchecker) {
  if (op.imu) ck.imuStep(op.imu[0], op.imu[1])
  else {
    const [E, N, t, acc, speed, course, vHat] = op.fix
    const f = ck.features(E, N, t, acc, speed, course, vHat)
    if ((f === null) !== (op.features === null)) throw new Error('FixChecker: anchor state differs')
    if (f) {
      worst = Math.max(worst, diff(f, op.features))
      nFix++
    }
    ck.afterFix(E, N, t, acc, speed, course, vHat, op.trusted)
  }
}
console.log(`FixChecker     ${nFix} fixes, max deviation ${worst.toExponential(2)}`)

const o = vec.outagetracker
const corr = Object.assign(new SpeedCorrector(), o.corrector)
const tr = new OutageTracker(0.3, corr, 0.8, 0.4)
let w2 = 0
let j = 0
o.steps.forEach(([v, om, still, variance]: [number, number, boolean, number], k: number) => {
  tr.step(v, om, still)
  if (k % 10 === 9) {
    tr.speedObs(variance)
    w2 = Math.max(w2, diff(tr.features(0.3 + 0.01 * k, 5 + 0.1 * k), o.features[j++]))
  }
})
console.log(`OutageTracker  ${j} rows, max deviation ${w2.toExponential(2)}`)
if (Math.max(worst, w2) > 1e-4) process.exit(1)
