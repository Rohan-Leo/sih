/**
 * Scripted "Simulate GNSS loss" scenario — 73 s of values for the 3×3 tile grid.
 *
 * SCRIPTED, NOT MEASURED. Every number here follows a hand-written target curve.
 * Nothing comes from the estimator or the trained model. Label it as a simulation
 * wherever it is shown, and quote measured results from ml/results/ instead.
 *
 * Usage
 *   frameAt(t)                    pure: the frame at t seconds after the click (0 ≤ t ≤ 73)
 *   framesAt(hz)                  the whole run sampled at a fixed rate
 *   new GnssLossSimulation()      wall-clock helper for the button: start() on click, frame() per render
 *   runGnssLossSimulation(cb)     timer helper: calls cb(frame) at a fixed rate until 73 s, returns stop()
 *
 * Deterministic: the same seed always gives the same run, at any frame rate.
 * Jitter is smooth value-noise over time, not per-frame noise, so readouts don't flicker at 60 fps.
 */

export type GnssFixStatus = 'FIX' | 'WEAK' | 'NO_FIX'
export type NavState = 'ACTIVE' | 'PRIMING' | 'DEAD_RECKONING' | 'BLENDING'
export type SimPhase = 'baseline' | 'drop' | 'blackout' | 'blending' | 'stabilized'

export interface SimFrame {
  /** seconds since the button click, 0–73 */
  t: number
  phase: SimPhase
  /** Tile 1 — median C/N0 of the top-4 satellites, dB-Hz */
  cn0: number
  /** Tile 2 */
  fixStatus: GnssFixStatus
  /** Tile 3 */
  state: NavState
  /** Tile 4 — wheel-speed ground truth, m/s */
  wheelSpeed: number
  /** Tile 5 — CNN speed estimate, m/s (independent of GNSS state) */
  cnnSpeed: number
  /** Tile 6 — |cnnSpeed − wheelSpeed|, m/s */
  speedError: number
  /** Tile 7 — position drift, m (0 until DEAD_RECKONING begins) */
  drift: number
  /** distance travelled since blackout entry, m (∫ wheelSpeed dt) — denominator of tile 8 */
  distanceSinceBlackout: number
  /** Tile 8 — drift as % of distanceSinceBlackout */
  driftPct: number
  /** Tile 9 — UKF confidence, 0–1 */
  ukfConfidence: number
}

export const DURATION_S = 73
export const DEFAULT_SEED = 26168

/** Phase boundaries (s). */
export const T = {
  dropStart: 5,
  weakAt: 5.5,
  blackoutStart: 7,
  blendStart: 67,
  stableStart: 70,
  end: DURATION_S,
} as const

// ── helpers ──────────────────────────────────────────────────────────────────

type Key = [number, number]

/** Monotone cubic (Fritsch–Carlson) interpolation: smooth, and never overshoots the keys. */
function pchip(keys: Key[]): (x: number) => number {
  const n = keys.length
  const xs = keys.map((k) => k[0])
  const ys = keys.map((k) => k[1])
  const h = xs.slice(1).map((x, i) => x - xs[i])
  const d = h.map((hi, i) => (ys[i + 1] - ys[i]) / hi)
  const m = new Array<number>(n)
  m[0] = d[0]
  m[n - 1] = d[n - 2]
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] <= 0) m[i] = 0
    else {
      const w1 = 2 * h[i] + h[i - 1]
      const w2 = h[i] + 2 * h[i - 1]
      m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    }
  }
  return (x: number) => {
    if (x <= xs[0]) return ys[0]
    if (x >= xs[n - 1]) return ys[n - 1]
    let i = 0
    while (x > xs[i + 1]) i++
    const t = (x - xs[i]) / h[i]
    const t2 = t * t
    const t3 = t2 * t
    return (
      (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1]
    )
  }
}

/** Piecewise-linear envelope (for jitter amplitudes). */
function linear(keys: Key[]): (x: number) => number {
  return (x: number) => {
    if (x <= keys[0][0]) return keys[0][1]
    for (let i = 0; i < keys.length - 1; i++) {
      const [x0, y0] = keys[i]
      const [x1, y1] = keys[i + 1]
      if (x <= x1) return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0)
    }
    return keys[keys.length - 1][1]
  }
}

/** Deterministic hash → uniform [0, 1). */
function hash01(seed: number, channel: number, k: number): number {
  let x = (seed ^ Math.imul(channel + 1, 0x9e3779b1) ^ Math.imul(k + 0x632be5ab, 0x85ebca6b)) >>> 0
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0
  x = (x ^ (x >>> 16)) >>> 0
  return x / 4294967296
}

const smooth = (u: number) => u * u * (3 - 2 * u)

/** Smooth value-noise in [-1, 1]: random knots every 1/rateHz s, eased between. */
function noise(seed: number, channel: number, t: number, rateHz: number): number {
  const x = t * rateHz
  const k = Math.floor(x)
  const a = hash01(seed, channel, k) * 2 - 1
  const b = hash01(seed, channel, k + 1) * 2 - 1
  return a + (b - a) * smooth(x - k)
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

// ── target curves (straight from the spec) ───────────────────────────────────

const CN0 = pchip([
  [0, 43], [5, 43],
  // drop: S-curve 43 → 18 over 5–7 s (crosses 35 at ≈5.5 s, reaches 18 at 7 s)
  [5.25, 39], [5.5, 34], [6.0, 28], [6.5, 22], [7.0, 18],
  // blackout: 16 → 12 → 8, then an ~8 dB-Hz floor
  [7.5, 16], [9, 12], [12, 8], [66.4, 8],
  // recovery 10 → 18 → 28 → 38 → 42 → 43
  [67, 10], [67.6, 18], [68.2, 28], [68.8, 38], [69.4, 42], [70, 43], [73, 43],
])
// jitter amplitudes: ±1 steady, small during fast transitions, ±2 on the blackout floor
const CN0_JITTER = linear([
  [0, 1], [5, 1], [5.2, 0.4], [7.5, 0.4], [12, 2], [66, 2], [67, 0.4], [70, 0.4], [71, 1], [73, 1],
])

const DRIFT_BLACKOUT = pchip([
  [0, 0], [5, 0.5], [10, 2], [15, 5], [20, 9], [25, 13], [30, 18], [35, 23], [40, 27], [45, 31], [50, 34], [55, 37], [60, 39],
])
const DRIFT_RECOVERY = pchip([[0, 39], [1, 30], [2, 18], [3, 8], [5, 3], [8, 2]])

const CONF_PRE = pchip([[0, 0.98], [4.8, 0.98], [5.3, 0.95], [6.8, 0.95], [7, 0.93]])
const CONF_BLACKOUT = pchip([[0, 0.93], [15, 0.85], [25, 0.78], [35, 0.68], [45, 0.58], [55, 0.48], [60, 0.42]])
const CONF_RECOVERY = pchip([[0, 0.42], [1, 0.55], [2, 0.72], [3, 0.88], [5, 0.94], [8, 0.98]])
// ±0.01 while steady (0.97–0.99), ±0.02 elsewhere; fades near the blackout-end handover so the curve stays continuous
const CONF_JITTER = linear([[0, 0.01], [4.8, 0.01], [5.3, 0.02], [66.5, 0.02], [67, 0], [67.3, 0.02], [73, 0.02]])

// ── channels ─────────────────────────────────────────────────────────────────

function phaseAt(t: number): SimPhase {
  if (t < T.dropStart) return 'baseline'
  if (t < T.blackoutStart) return 'drop'
  if (t < T.blendStart) return 'blackout'
  if (t < T.stableStart) return 'blending'
  return 'stabilized'
}

function stateAt(t: number): NavState {
  if (t < T.dropStart) return 'ACTIVE'
  if (t < T.blackoutStart) return 'PRIMING' // fires 2 s before NO_FIX
  if (t < T.blendStart) return 'DEAD_RECKONING'
  if (t < T.stableStart) return 'BLENDING'
  return 'ACTIVE'
}

function fixStatusAt(t: number): GnssFixStatus {
  if (t < T.weakAt) return 'FIX' // C/N0 ≥ 35
  if (t < T.blackoutStart) return 'WEAK' // C/N0 < 35
  if (t < T.blendStart) return 'NO_FIX' // C/N0 ≤ 18
  if (t < T.stableStart) return 'WEAK' // reacquiring (hysteresis: held WEAK until 70 s)
  return 'FIX'
}

function cn0At(t: number, seed: number): number {
  const v = CN0(t) + CN0_JITTER(t) * noise(seed, 1, t, 3)
  // keep the enum/number pairing consistent at every frame
  if (t >= T.blackoutStart && t < T.blendStart) return clamp(v, 6, 18)
  if (t < T.dropStart || t >= T.stableStart) return clamp(v, 42, 44)
  // drop phase: FIX ⇔ C/N0 ≥ 35, WEAK ⇔ C/N0 < 35, on exactly the same frame
  if (t < T.blackoutStart) return t < T.weakAt ? Math.max(v, 35) : Math.min(v, 34.9)
  return v
}

/** Wheel-speed truth: 16.5–17.0 m/s, with a ~2 s dip to 16.0 at tunnel entry (7 s). */
function wheelSpeedAt(t: number, seed: number): number {
  // raised-cosine dip window: full depth 7.4–8.6 s, eased in/out over 0.6 s
  const dip =
    t < 6.8 || t > 9.2 ? 0 : t < 7.4 ? 0.5 - 0.5 * Math.cos(((t - 6.8) / 0.6) * Math.PI) : t <= 8.6 ? 1 : 0.5 + 0.5 * Math.cos(((t - 8.6) / 0.6) * Math.PI)
  const jitter = 0.25 * (1 - 0.8 * dip) * noise(seed, 2, t, 1.5)
  return 16.75 - 0.75 * dip + jitter
}

/**
 * CNN estimate: truth + an error that updates once per second (the model runs
 * at 1 Hz) and is held in between, like a real inference readout.
 * Error magnitudes are drawn in balanced 6-second blocks: each block uses every
 * value in ERR_BLOCK once, in a seeded random order with random signs. So the
 * error averages 1.0 m/s over any few seconds, in every phase. The process
 * never looks at GNSS state (rule 1).
 */
const ERR_BLOCK = [0.3, 0.65, 0.95, 1.15, 1.45, 1.5] // mean 1.0 m/s, max 1.5 keeps CNN in 15.0–18.5
function cnnErrorKnot(seed: number, k: number): number {
  const block = Math.floor(k / ERR_BLOCK.length)
  const order = ERR_BLOCK.map((_, i) => i).sort((a, b) => hash01(seed, 3, block * 16 + a) - hash01(seed, 3, block * 16 + b))
  const mag = ERR_BLOCK[order[((k % ERR_BLOCK.length) + ERR_BLOCK.length) % ERR_BLOCK.length]]
  return hash01(seed, 4, k) < 0.5 ? -mag : mag
}
function cnnSpeedAt(t: number, seed: number, truth: number): number {
  return clamp(truth + cnnErrorKnot(seed, Math.floor(t)), 15.0, 18.5)
}

function driftAt(t: number): number {
  if (t < T.blackoutStart) return 0
  if (t < T.blendStart) return DRIFT_BLACKOUT(t - T.blackoutStart)
  return DRIFT_RECOVERY(t - T.blendStart)
}

function confidenceAt(t: number, seed: number): number {
  const base =
    t < T.blackoutStart ? CONF_PRE(t) : t < T.blendStart ? CONF_BLACKOUT(t - T.blackoutStart) : CONF_RECOVERY(t - T.blendStart)
  const v = base + CONF_JITTER(t) * noise(seed, 5, t, 1)
  return t < 4.8 ? clamp(v, 0.97, 0.99) : clamp(v, 0, 1)
}

// distance since blackout entry = ∫ wheel speed dt (tabulated once per seed at 100 Hz)
const DIST_STEP = 0.01
const distanceTables = new Map<number, Float64Array>()
function distanceTable(seed: number): Float64Array {
  let tab = distanceTables.get(seed)
  if (!tab) {
    const n = Math.ceil((DURATION_S - T.blackoutStart) / DIST_STEP) + 1
    tab = new Float64Array(n)
    for (let i = 1; i < n; i++) {
      const t0 = T.blackoutStart + (i - 1) * DIST_STEP
      tab[i] = tab[i - 1] + 0.5 * (wheelSpeedAt(t0, seed) + wheelSpeedAt(t0 + DIST_STEP, seed)) * DIST_STEP
    }
    distanceTables.set(seed, tab)
  }
  return tab
}
function distanceSinceBlackout(t: number, seed: number): number {
  if (t <= T.blackoutStart) return 0
  const tab = distanceTable(seed)
  const x = (t - T.blackoutStart) / DIST_STEP
  const i = Math.min(Math.floor(x), tab.length - 2)
  return tab[i] + (tab[i + 1] - tab[i]) * (x - i)
}

// ── public API ───────────────────────────────────────────────────────────────

/** The frame at t seconds after the button click. Pure and deterministic. */
export function frameAt(t: number, seed = DEFAULT_SEED): SimFrame {
  const tt = clamp(t, 0, DURATION_S)
  const wheelSpeed = wheelSpeedAt(tt, seed)
  const cnnSpeed = cnnSpeedAt(tt, seed, wheelSpeed)
  const drift = driftAt(tt)
  const dist = distanceSinceBlackout(tt, seed)
  return {
    t: tt,
    phase: phaseAt(tt),
    cn0: cn0At(tt, seed),
    fixStatus: fixStatusAt(tt),
    state: stateAt(tt),
    wheelSpeed,
    cnnSpeed,
    speedError: Math.abs(cnnSpeed - wheelSpeed),
    drift,
    distanceSinceBlackout: dist,
    driftPct: dist >= 1 ? (100 * drift) / dist : 0,
    ukfConfidence: confidenceAt(tt, seed),
  }
}

/** The whole 73 s run at a fixed sample rate (t = 0, 1/hz, …, 73). */
export function framesAt(hz = 10, seed = DEFAULT_SEED): SimFrame[] {
  const n = Math.round(DURATION_S * hz)
  return Array.from({ length: n + 1 }, (_, i) => frameAt(i / hz, seed))
}

/** Discrete transitions, for anything that wants to log or announce them. */
export const TRANSITIONS: { t: number; tile: 'state' | 'fixStatus'; from: string; to: string }[] = [
  { t: T.dropStart, tile: 'state', from: 'ACTIVE', to: 'PRIMING' },
  { t: T.weakAt, tile: 'fixStatus', from: 'FIX', to: 'WEAK' },
  { t: T.blackoutStart, tile: 'state', from: 'PRIMING', to: 'DEAD_RECKONING' },
  { t: T.blackoutStart, tile: 'fixStatus', from: 'WEAK', to: 'NO_FIX' },
  { t: T.blendStart, tile: 'state', from: 'DEAD_RECKONING', to: 'BLENDING' },
  { t: T.blendStart, tile: 'fixStatus', from: 'NO_FIX', to: 'WEAK' },
  { t: T.stableStart, tile: 'state', from: 'BLENDING', to: 'ACTIVE' },
  { t: T.stableStart, tile: 'fixStatus', from: 'WEAK', to: 'FIX' },
]

/**
 * Wall-clock driver for the button. Call start() on click, then frame() on each
 * render. `speed` scales simulated time (1 = real time, 2 = twice as fast).
 */
export class GnssLossSimulation {
  private startMs: number | null = null

  constructor(readonly opts: { seed?: number; speed?: number } = {}) {}

  start(nowMs: number = performance.now()) {
    this.startMs = nowMs
  }

  stop() {
    this.startMs = null
  }

  get running(): boolean {
    return this.startMs !== null
  }

  /** Simulated seconds since start (clamped to 73), or null if not started. */
  elapsed(nowMs: number = performance.now()): number | null {
    if (this.startMs === null) return null
    return Math.min(DURATION_S, ((nowMs - this.startMs) / 1000) * (this.opts.speed ?? 1))
  }

  done(nowMs: number = performance.now()): boolean {
    return this.elapsed(nowMs) === DURATION_S
  }

  frame(nowMs: number = performance.now()): SimFrame | null {
    const t = this.elapsed(nowMs)
    return t === null ? null : frameAt(t, this.opts.seed)
  }
}

/**
 * Timer driver: calls onFrame at `hz` until the 73 s run ends (the last frame is
 * exactly t = 73). Returns a function that stops it early.
 */
export function runGnssLossSimulation(
  onFrame: (f: SimFrame) => void,
  opts: { hz?: number; speed?: number; seed?: number; onDone?: () => void } = {},
): () => void {
  const sim = new GnssLossSimulation({ seed: opts.seed, speed: opts.speed })
  sim.start()
  onFrame(sim.frame()!)
  const timer = setInterval(() => {
    onFrame(sim.frame()!)
    if (sim.done()) {
      clearInterval(timer)
      opts.onDone?.()
    }
  }, 1000 / (opts.hz ?? 10))
  return () => clearInterval(timer)
}
