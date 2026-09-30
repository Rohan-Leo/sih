// npx jiti scripts/gnss-loss-sim.ts [--write]
// Checks every correlation rule of the scripted GNSS-loss scenario at 60 Hz,
// prints the values at the spec's reference times, and with --write saves
// public/sim/gnss-loss-73s.json (10 Hz frames) for UIs that prefer plain data.
import { mkdirSync, writeFileSync } from 'node:fs'
import { DEFAULT_SEED, DURATION_S, T, TRANSITIONS, frameAt, framesAt, type SimFrame } from '../src/sim/gnssLossScenario'

const frames = framesAt(60)
const fails: string[] = []
const check = (ok: boolean, msg: string) => {
  if (!ok) fails.push(msg)
}
const inPhase = (lo: number, hi: number) => frames.filter((f) => f.t >= lo && f.t < hi)
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const between = (v: number, lo: number, hi: number) => v >= lo - 1e-9 && v <= hi + 1e-9

// ── per-frame ranges & consistency ──
for (const f of frames) {
  const at = `t=${f.t.toFixed(3)}`
  check(Math.abs(f.speedError - Math.abs(f.cnnSpeed - f.wheelSpeed)) < 1e-12, `${at} speedError ≠ |cnn − truth|`)
  check(between(f.cnnSpeed, 15.0, 18.5), `${at} CNN ${f.cnnSpeed} outside 15.0–18.5`)
  check(between(f.speedError, 0.1, 2.5), `${at} speed error ${f.speedError} outside 0.1–2.5`)
  check(f.driftPct < 10, `${at} drift % ${f.driftPct} ≥ 10`)
  check(between(f.ukfConfidence, 0, 1), `${at} confidence outside 0–1`)
  // tile 4
  if (f.t < 6.8 || f.t > 9.2) check(between(f.wheelSpeed, 16.5, 17.0), `${at} wheel ${f.wheelSpeed} outside 16.5–17.0`)
  // enum ↔ number pairing (tile 1 ↔ tile 2)
  if (f.fixStatus === 'FIX' && f.t < T.blendStart) check(f.cn0 >= 35, `${at} FIX but C/N0 ${f.cn0} < 35`)
  if (f.fixStatus === 'WEAK' && f.t < T.blackoutStart) check(f.cn0 < 35, `${at} WEAK but C/N0 ${f.cn0} ≥ 35`)
  if (f.fixStatus === 'NO_FIX') check(f.cn0 <= 18, `${at} NO_FIX but C/N0 ${f.cn0} > 18`)
  // rule 5: never NO_FIX unless PRIMING or DEAD_RECKONING already reached; PRIMING only before NO_FIX
  if (f.fixStatus === 'NO_FIX') check(f.state === 'DEAD_RECKONING', `${at} NO_FIX while state ${f.state}`)
  if (f.state === 'PRIMING') check(f.fixStatus !== 'NO_FIX', `${at} PRIMING while NO_FIX`)
  // rule 2
  if (f.t < T.blackoutStart) check(f.drift === 0 && f.driftPct === 0, `${at} drift before blackout`)
}
// baseline ranges
for (const f of inPhase(0, T.dropStart)) {
  check(between(f.cn0, 42, 44), `t=${f.t.toFixed(2)} baseline C/N0 ${f.cn0}`)
  check(between(f.ukfConfidence, 0.97, 0.99), `t=${f.t.toFixed(2)} baseline confidence ${f.ukfConfidence}`)
}
for (const f of inPhase(12, T.blendStart - 0.6)) check(between(f.cn0, 6, 10), `t=${f.t.toFixed(2)} blackout floor C/N0 ${f.cn0}`)
for (const f of inPhase(T.stableStart + 0.01, DURATION_S + 1)) check(between(f.cn0, 42, 44), `t=${f.t.toFixed(2)} stabilized C/N0 ${f.cn0}`)

// rule 5: ordering of the first appearances
const first = (pred: (f: SimFrame) => boolean) => frames.find(pred)?.t ?? Infinity
const tPriming = first((f) => f.state === 'PRIMING')
const tNoFix = first((f) => f.fixStatus === 'NO_FIX')
check(tNoFix - tPriming >= 1.9, `PRIMING at ${tPriming}s is not ~2 s before NO_FIX at ${tNoFix}s`)

// rule 2: drift starts from exactly 0 at DR entry
const tDR = first((f) => f.state === 'DEAD_RECKONING')
check(frameAt(tDR).drift === 0 && tDR === T.blackoutStart, `drift at DR entry is ${frameAt(tDR).drift}`)

// rule 1: speed-error statistics are the same in every phase
const errMean = (lo: number, hi: number) => mean(inPhase(lo, hi).map((f) => f.speedError))
const eBase = errMean(0, T.blackoutStart)
const eBlack = errMean(T.blackoutStart, T.blendStart)
const eRec = errMean(T.blendStart, DURATION_S + 1)
check(between(eBase, 0.85, 1.15) && between(eBlack, 0.85, 1.15) && between(eRec, 0.85, 1.15), `speed-error means ${eBase}, ${eBlack}, ${eRec}`)
// no jump at the GNSS drop: error ±1 s around 7 s stays inside the normal range
const jumpWin = inPhase(6, 8).map((f) => f.speedError)
check(Math.max(...jumpWin) <= 1.5, 'speed error spikes at blackout entry')

// rule 3: confidence ↔ drift inversely correlated (blackout + blending)
const dr = inPhase(T.blackoutStart, DURATION_S + 1)
const xs = dr.map((f) => f.drift)
const ys = dr.map((f) => f.ukfConfidence)
const mx = mean(xs)
const my = mean(ys)
const r =
  xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) /
  Math.sqrt(xs.reduce((a, x) => a + (x - mx) ** 2, 0) * ys.reduce((a, y) => a + (y - my) ** 2, 0))
check(r < -0.9, `confidence–drift correlation ${r.toFixed(3)} not strongly negative`)

// rule 4: drift % peaks mid-blackout, then settles
const bl = inPhase(T.blackoutStart, T.blendStart)
const peak = bl.reduce((a, f) => (f.driftPct > a.driftPct ? f : a))
const endPct = frameAt(T.blendStart - 1e-6).driftPct
check(peak.t > 30 && peak.t < 60 && endPct < peak.driftPct, `drift % peak ${peak.driftPct.toFixed(2)} at ${peak.t}, end ${endPct.toFixed(2)}`)
check(frameAt(T.blendStart + 5).driftPct < 0.5, 'drift % not < 0.5 by +5 s after reacquisition')

// rule 6: drift recovery (to within 3 m) vs GNSS recovery (C/N0 back to ≥42)
const tCn0Back = first((f) => f.t >= T.blendStart && f.cn0 >= 42) - T.blendStart
const tDriftBack = first((f) => f.t >= T.blendStart && f.drift <= 3) - T.blendStart
const tGnssBack = T.stableStart - T.blendStart // BLENDING → ACTIVE

// ── report ──
const row = (t: number) => {
  const f = frameAt(t)
  return [
    t.toFixed(1).padStart(5),
    f.phase.padEnd(10),
    f.cn0.toFixed(1).padStart(5),
    f.fixStatus.padEnd(6),
    f.state.padEnd(14),
    f.wheelSpeed.toFixed(2),
    f.cnnSpeed.toFixed(2),
    f.speedError.toFixed(2),
    f.drift.toFixed(1).padStart(5),
    f.driftPct.toFixed(2).padStart(5),
    f.ukfConfidence.toFixed(3),
  ].join('  ')
}
console.log('    t  phase        C/N0  fix     state          wheel  CNN    err    drift  drift%  conf')
for (const t of [0, 2.5, 5, 5.5, 6, 6.5, 7, 8, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 66.9, 67, 68, 69, 70, 72, 73]) console.log(row(t))
console.log()
console.log(`transitions: ${TRANSITIONS.map((x) => `${x.t}s ${x.tile} ${x.from}→${x.to}`).join(' · ')}`)
console.log(`PRIMING at ${tPriming.toFixed(2)} s, NO_FIX at ${tNoFix.toFixed(2)} s (lead ${(tNoFix - tPriming).toFixed(2)} s)`)
console.log(`speed error mean — before ${eBase.toFixed(2)}, blackout ${eBlack.toFixed(2)}, after ${eRec.toFixed(2)} m/s`)
console.log(`confidence vs drift: r = ${r.toFixed(3)}`)
console.log(`drift % peak ${peak.driftPct.toFixed(2)}% at t=${peak.t.toFixed(1)} s, ${endPct.toFixed(2)}% at blackout end, ${frameAt(DURATION_S).driftPct.toFixed(2)}% at 73 s`)
console.log(
  `recovery: GNSS back (FIX) after ${tGnssBack} s (C/N0 ≥42 after ${tCn0Back.toFixed(1)} s); drift ≤3 m after ${tDriftBack.toFixed(1)} s, ` +
    `and it reaches the spec's settled 2 m at +8 s (×${(8 / tGnssBack).toFixed(1)}), 2 s after the 73 s window ends — ${frameAt(DURATION_S).drift.toFixed(1)} m at 73 s`,
)
console.log(fails.length ? `\n${fails.length} FAILED checks:\n${fails.slice(0, 20).join('\n')}` : `\nall checks passed over ${frames.length} frames (60 Hz)`)

if (process.argv.includes('--write')) {
  mkdirSync('public/sim', { recursive: true })
  const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d
  const out = {
    description: 'Scripted GNSS-loss scenario for the 3×3 tile grid — hand-authored target curves, NOT measured results.',
    seed: DEFAULT_SEED,
    hz: 10,
    durationS: DURATION_S,
    transitions: TRANSITIONS,
    frames: framesAt(10).map((f) => ({
      t: round(f.t, 1),
      phase: f.phase,
      cn0: round(f.cn0, 1),
      fixStatus: f.fixStatus,
      state: f.state,
      wheelSpeed: round(f.wheelSpeed, 2),
      cnnSpeed: round(f.cnnSpeed, 2),
      speedError: round(f.speedError, 2),
      drift: round(f.drift, 2),
      distanceSinceBlackout: round(f.distanceSinceBlackout, 1),
      driftPct: round(f.driftPct, 2),
      ukfConfidence: round(f.ukfConfidence, 3),
    })),
  }
  writeFileSync('public/sim/gnss-loss-73s.json', JSON.stringify(out))
  console.log(`wrote public/sim/gnss-loss-73s.json (${out.frames.length} frames)`)
}
if (fails.length) process.exit(1)
