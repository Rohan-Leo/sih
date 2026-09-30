// @ts-check
/**
 * Turn a route polyline into a timed {lat, lon, t} trace with a plausible
 * urban driving speed profile (slows for turns, accelerates on straights).
 * Plain JS so the Node bake script and the browser app share it verbatim.
 */

const R = 6371008.8
const D2R = Math.PI / 180

/** @param {[number, number]} a @param {[number, number]} b */
function dist(a, b) {
  const dLat = (b[1] - a[1]) * D2R
  const dLon = (b[0] - a[0]) * D2R
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(a[1] * D2R) * Math.cos(b[1] * D2R) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)))
}

/** @param {[number, number]} a @param {[number, number]} b */
function brg(a, b) {
  const y = Math.sin((b[0] - a[0]) * D2R) * Math.cos(b[1] * D2R)
  const x =
    Math.cos(a[1] * D2R) * Math.sin(b[1] * D2R) -
    Math.sin(a[1] * D2R) * Math.cos(b[1] * D2R) * Math.cos((b[0] - a[0]) * D2R)
  return (Math.atan2(y, x) / D2R + 360) % 360
}

/**
 * @param {[number, number][]} coords  [lon, lat]
 * @param {{cruise?: number, turn?: number, accel?: number, dtMs?: number}} [opts]
 * @returns {{lat: number, lon: number, t: number}[]}
 */
export function buildTimeline(coords, opts = {}) {
  const cruise = opts.cruise ?? 12.5 // m/s ≈ 45 km/h
  const turn = opts.turn ?? 4.5 // m/s through a 90° turn
  const accel = opts.accel ?? 1.4 // m/s²
  const dtMs = opts.dtMs ?? 500

  // drop duplicate vertices
  /** @type {[number, number][]} */
  const pts = []
  for (const c of coords) if (!pts.length || dist(pts[pts.length - 1], c) > 0.5) pts.push(c)
  if (pts.length < 2) return pts.map((p) => ({ lon: p[0], lat: p[1], t: 0 }))

  const cum = [0]
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + dist(pts[i - 1], pts[i]))

  // speed cap at each vertex from the turn angle
  const cap = pts.map((_, i) => {
    if (i === 0 || i === pts.length - 1) return 0
    let d = Math.abs(brg(pts[i], pts[i + 1]) - brg(pts[i - 1], pts[i]))
    if (d > 180) d = 360 - d
    if (d < 12) return cruise
    const k = Math.min(1, (d - 12) / 78)
    return cruise + (turn - cruise) * k
  })
  // forward / backward passes enforce the acceleration limit: v² ≤ v0² + 2·a·s
  const v = cap.slice()
  for (let i = 1; i < v.length; i++) v[i] = Math.min(v[i], Math.sqrt(v[i - 1] ** 2 + 2 * accel * (cum[i] - cum[i - 1])))
  for (let i = v.length - 2; i >= 0; i--) v[i] = Math.min(v[i], Math.sqrt(v[i + 1] ** 2 + 2 * accel * (cum[i + 1] - cum[i])))

  // integrate time along distance at 1 m resolution
  const total = cum[cum.length - 1]
  /** @type {{lat: number, lon: number, t: number}[]} */
  const out = []
  let seg = 0
  let t = 0
  let nextEmit = 0
  const step = 1
  for (let s = 0; s <= total + 1e-6; s += step) {
    while (seg < pts.length - 2 && cum[seg + 1] < s) seg++
    const len = cum[seg + 1] - cum[seg]
    const f = len > 0 ? Math.min(1, (s - cum[seg]) / len) : 0
    // speed between vertices: limited by accel from both ends and by cruise
    const vs = Math.min(
      cruise,
      Math.sqrt(v[seg] ** 2 + 2 * accel * (s - cum[seg])),
      Math.sqrt(v[seg + 1] ** 2 + 2 * accel * (cum[seg + 1] - s)),
    )
    if (t >= nextEmit || s + step > total) {
      out.push({
        lon: +(pts[seg][0] + (pts[seg + 1][0] - pts[seg][0]) * f).toFixed(7),
        lat: +(pts[seg][1] + (pts[seg + 1][1] - pts[seg][1]) * f).toFixed(7),
        t: Math.round(t),
      })
      nextEmit += dtMs
    }
    t += (step / Math.max(vs, 1.2)) * 1000
  }
  return out
}
