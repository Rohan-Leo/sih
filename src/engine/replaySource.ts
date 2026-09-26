/**
 * Replays a real recorded drive (held-out IO-VNBD test drive) as a sensor feed:
 * the phone's own accelerometer / gravity / gyroscope at 10 Hz, plus 1 Hz GNSS.
 * GNSS is withheld inside the dead zones (or on the presenter's cue) and the
 * estimator has to cope with the real IMU — no synthetic motion at all.
 */
import { type LngLat, Polyline, bearing, haversine } from '../lib/geo'
import type { DeadZone, ScriptedSource, SensorSink } from './sources'

export interface ReplayManifest {
  name: string
  segment: string
  hz: number
  samples: number
  columns: string[]
  origin: [number, number]
  deadZones: { from: number; to: number; label: string }[]
  calibrationSamples: number
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  osrm: any
}

const R = 6371008.8
const D2R = Math.PI / 180

export class ReplaySource implements ScriptedSource {
  readonly path: Polyline
  readonly duration: number
  readonly deadZones: DeadZone[]
  private col: Record<string, number> = {}
  private truth: LngLat[]
  private next = 0
  private manualUntil = -Infinity
  private zoneSamples: { from: number; to: number; zone: DeadZone }[]
  private readonly period: number

  constructor(
    readonly manifest: ReplayManifest,
    private data: Float32Array,
  ) {
    manifest.columns.forEach((c, i) => (this.col[c] = i))
    this.period = 1000 / manifest.hz
    this.duration = (manifest.samples - 1) * this.period
    this.truth = Array.from({ length: manifest.samples }, (_, i) => this.lngLat(this.v(i, 'truth_e'), this.v(i, 'truth_n')))
    this.path = new Polyline(this.truth)
    this.zoneSamples = manifest.deadZones.map((z) => {
      const zone = { from: this.path.cum[z.from], to: this.path.cum[Math.min(z.to, manifest.samples - 1)], label: z.label }
      return { from: z.from, to: z.to, zone }
    })
    this.deadZones = this.zoneSamples.map((z) => z.zone)
  }

  static async load(base = '/ml/replay-y1-3'): Promise<ReplaySource> {
    const [m, b] = await Promise.all([
      fetch(`${base}.json`).then((r) => r.json() as Promise<ReplayManifest>),
      fetch(`${base}.bin`).then((r) => r.arrayBuffer()),
    ])
    return new ReplaySource(m, new Float32Array(b))
  }

  private v(i: number, c: string): number {
    return this.data[i * this.manifest.columns.length + this.col[c]]
  }

  private lngLat(e: number, n: number): LngLat {
    const [lat0, lon0] = this.manifest.origin
    return [lon0 + e / (R * Math.cos(lat0 * D2R)) / D2R, lat0 + n / R / D2R]
  }

  private index(t: number): number {
    return Math.max(0, Math.min(this.manifest.samples - 1, Math.round(t / this.period)))
  }

  reset() {
    this.next = 0
    this.manualUntil = -Infinity
  }

  truthAt(t: number) {
    const i = this.index(t)
    const j = Math.min(i + 1, this.manifest.samples - 1)
    return {
      pos: this.truth[i],
      speed: this.v(i, 'truth_speed'),
      course: j > i && haversine(this.truth[i], this.truth[j]) > 0.2 ? bearing(this.truth[i], this.truth[j]) : this.v(i, 'truth_heading'),
      along: this.path.cum[i],
    }
  }

  deadZoneAt(t: number): DeadZone | null {
    const i = this.index(t)
    return this.zoneSamples.find((z) => i >= z.from && i < z.to)?.zone ?? null
  }

  gnssWithheld(t: number): boolean {
    return t < this.manualUntil || this.deadZoneAt(t) !== null
  }

  toggleManualOutage(now: number, ms = 20000) {
    this.manualUntil = this.manualUntil > now ? -Infinity : now + ms
  }

  manualOutageRemaining(now: number): number {
    return Math.max(0, this.manualUntil - now)
  }

  errorAgainstTruth(est: LngLat | null, t: number): number | null {
    return est ? haversine(est, this.truthAt(t).pos) : null
  }

  advance(t: number, sink: SensorSink) {
    while (this.next < this.manifest.samples && this.next * this.period <= t) {
      const i = this.next++
      const ts = i * this.period
      sink.pushMotion({
        ax: this.v(i, 'acc_x'),
        ay: this.v(i, 'acc_y'),
        az: this.v(i, 'acc_z'),
        includesGravity: true,
        gravity: [this.v(i, 'grav_x'), this.v(i, 'grav_y'), this.v(i, 'grav_z')],
        gyro: [this.v(i, 'gyro_x'), this.v(i, 'gyro_y'), this.v(i, 'gyro_z')],
        gyroZ: null,
        t: ts,
      })
      if (this.v(i, 'gnss_fresh') > 0.5 && !this.gnssWithheld(ts)) {
        const [lon, lat] = this.lngLat(this.v(i, 'gnss_e'), this.v(i, 'gnss_n'))
        sink.pushFix({ lon, lat, accuracy: 4, speed: this.v(i, 'gnss_speed'), course: this.v(i, 'gnss_course'), t: ts })
      }
    }
  }
}
