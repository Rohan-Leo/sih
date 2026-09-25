/**
 * Small, dependency-free geodesy helpers. All coordinates are [lon, lat]
 * (GeoJSON order) in degrees; all distances are metres.
 */
export type LngLat = [number, number]

const R = 6371008.8
const D2R = Math.PI / 180
const R2D = 180 / Math.PI

export function haversine(a: LngLat, b: LngLat): number {
  const dLat = (b[1] - a[1]) * D2R
  const dLon = (b[0] - a[0]) * D2R
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a[1] * D2R) * Math.cos(b[1] * D2R) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)))
}

/** Initial bearing a → b, degrees clockwise from true north, [0, 360). */
export function bearing(a: LngLat, b: LngLat): number {
  const φ1 = a[1] * D2R
  const φ2 = b[1] * D2R
  const Δλ = (b[0] - a[0]) * D2R
  const y = Math.sin(Δλ) * Math.cos(φ2)
  const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ)
  return normDeg(Math.atan2(y, x) * R2D)
}

/** Point reached travelling `dist` metres from `p` on `brg` degrees. */
export function destination(p: LngLat, brg: number, dist: number): LngLat {
  const δ = dist / R
  const θ = brg * D2R
  const φ1 = p[1] * D2R
  const λ1 = p[0] * D2R
  const φ2 = Math.asin(Math.sin(φ1) * Math.cos(δ) + Math.cos(φ1) * Math.sin(δ) * Math.cos(θ))
  const λ2 =
    λ1 + Math.atan2(Math.sin(θ) * Math.sin(δ) * Math.cos(φ1), Math.cos(δ) - Math.sin(φ1) * Math.sin(φ2))
  return [λ2 * R2D, φ2 * R2D]
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

export function lerpLngLat(a: LngLat, b: LngLat, t: number): LngLat {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t)]
}

export function normDeg(d: number): number {
  return ((d % 360) + 360) % 360
}

/** Signed smallest difference b − a in degrees, (−180, 180]. */
export function angleDiff(a: number, b: number): number {
  let d = normDeg(b) - normDeg(a)
  if (d > 180) d -= 360
  if (d <= -180) d += 360
  return d
}

/** Interpolate headings along the short arc. */
export function lerpAngle(a: number, b: number, t: number): number {
  return normDeg(a + angleDiff(a, b) * t)
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** Circle polygon (for accuracy / uncertainty halos). */
export function circlePolygon(center: LngLat, radius: number, steps = 64): LngLat[] {
  const ring: LngLat[] = []
  for (let i = 0; i <= steps; i++) ring.push(destination(center, (i / steps) * 360, radius))
  return ring
}

export interface Projection {
  point: LngLat
  /** distance along the polyline from its start, metres */
  along: number
  /** perpendicular distance from the query point, metres */
  offset: number
  segment: number
}

/**
 * Polyline with cumulative distances, used for route progress,
 * "point at distance" lookups and nearest-point projection
 * (our stand-in for map matching).
 */
export class Polyline {
  readonly coords: LngLat[]
  readonly cum: number[]
  readonly length: number

  constructor(coords: LngLat[]) {
    this.coords = coords
    this.cum = [0]
    for (let i = 1; i < coords.length; i++) {
      this.cum.push(this.cum[i - 1] + haversine(coords[i - 1], coords[i]))
    }
    this.length = this.cum[this.cum.length - 1] ?? 0
  }

  /** Segment index containing distance `d` (binary search). */
  private segmentAt(d: number): number {
    let lo = 0
    let hi = this.cum.length - 1
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (this.cum[mid] <= d) lo = mid
      else hi = mid
    }
    return lo
  }

  pointAt(d: number): { point: LngLat; bearing: number } {
    const n = this.coords.length
    if (n === 0) return { point: [0, 0], bearing: 0 }
    if (n === 1) return { point: this.coords[0], bearing: 0 }
    const dd = clamp(d, 0, this.length)
    const i = Math.min(this.segmentAt(dd), n - 2)
    const segLen = this.cum[i + 1] - this.cum[i]
    const t = segLen > 0 ? (dd - this.cum[i]) / segLen : 0
    return {
      point: lerpLngLat(this.coords[i], this.coords[i + 1], t),
      bearing: bearing(this.coords[i], this.coords[i + 1]),
    }
  }

  /**
   * Nearest point on the polyline to `p`. Uses a local equirectangular
   * projection per segment — plenty accurate at street scale.
   * `hintAlong`/`window` restrict the search to a stretch of the route so a
   * vehicle on a road that doubles back doesn't snap to the other carriageway.
   */
  project(p: LngLat, hintAlong?: number, window = Infinity): Projection {
    const kx = Math.cos(p[1] * D2R) * R * D2R
    const ky = R * D2R
    let best: Projection = { point: this.coords[0], along: 0, offset: Infinity, segment: 0 }
    for (let i = 0; i < this.coords.length - 1; i++) {
      if (hintAlong !== undefined && window !== Infinity) {
        if (this.cum[i + 1] < hintAlong - window || this.cum[i] > hintAlong + window) continue
      }
      const a = this.coords[i]
      const b = this.coords[i + 1]
      const ax = (a[0] - p[0]) * kx
      const ay = (a[1] - p[1]) * ky
      const bx = (b[0] - p[0]) * kx
      const by = (b[1] - p[1]) * ky
      const dx = bx - ax
      const dy = by - ay
      const len2 = dx * dx + dy * dy
      let t = len2 > 0 ? -(ax * dx + ay * dy) / len2 : 0
      t = clamp(t, 0, 1)
      const cx = ax + dx * t
      const cy = ay + dy * t
      const off = Math.hypot(cx, cy)
      if (off < best.offset) {
        best = {
          point: lerpLngLat(a, b, t),
          along: this.cum[i] + (this.cum[i + 1] - this.cum[i]) * t,
          offset: off,
          segment: i,
        }
      }
    }
    return best
  }

  /** Sub-line between two along-distances (for dead-zone overlays). */
  slice(from: number, to: number): LngLat[] {
    const a = clamp(Math.min(from, to), 0, this.length)
    const b = clamp(Math.max(from, to), 0, this.length)
    const out: LngLat[] = [this.pointAt(a).point]
    for (let i = 0; i < this.coords.length; i++) {
      if (this.cum[i] > a && this.cum[i] < b) out.push(this.coords[i])
    }
    out.push(this.pointAt(b).point)
    return out
  }
}

export function formatDistance(m: number): string {
  if (!Number.isFinite(m)) return '—'
  if (m < 950) return `${Math.max(0, Math.round(m / 10) * 10)} m`
  return `${(m / 1000).toFixed(m < 9950 ? 1 : 0)} km`
}

export function formatDuration(s: number): string {
  if (!Number.isFinite(s)) return '—'
  const min = Math.round(s / 60)
  if (min < 1) return '<1 min'
  if (min < 60) return `${min} min`
  return `${Math.floor(min / 60)} h ${min % 60} min`
}

export function compass(deg: number): string {
  const pts = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
  return pts[Math.round(normDeg(deg) / 45) % 8]
}
