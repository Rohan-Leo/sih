/**
 * Real device sensors: Geolocation (GNSS), DeviceMotion (accelerometer +
 * gyroscope) and DeviceOrientation (compass), including the iOS 13+
 * gesture-gated permission flow.
 */
import type { SensorSink } from './sources'

export type GeoStatus = 'idle' | 'waiting' | 'ok' | 'denied' | 'unavailable' | 'insecure'
export type MotionStatus = 'idle' | 'needs-permission' | 'waiting' | 'active' | 'none' | 'denied'

export interface LiveStatus {
  geo: GeoStatus
  geoMessage: string | null
  motion: MotionStatus
  compass: 'none' | 'absolute' | 'relative'
}

type PermissionRequester = { requestPermission?: () => Promise<'granted' | 'denied'> }

export function motionPermissionRequired(): boolean {
  const dm = (globalThis as unknown as { DeviceMotionEvent?: PermissionRequester }).DeviceMotionEvent
  return typeof dm?.requestPermission === 'function'
}

export class LiveSource {
  status: LiveStatus = { geo: 'idle', geoMessage: null, motion: 'idle', compass: 'none' }
  private sink: SensorSink | null = null
  private now: () => number = () => performance.now()
  private watchId: number | null = null
  private lastFixStamp = 0
  private sensorsAttached = false
  private noMotionTimer: number | null = null
  private onChange: () => void

  constructor(onChange: () => void) {
    this.onChange = onChange
  }

  private set(p: Partial<LiveStatus>) {
    this.status = { ...this.status, ...p }
    this.onChange()
  }

  get geoRunning(): boolean {
    return this.watchId !== null
  }

  /** Start GNSS. Safe to call repeatedly. */
  startGeolocation(sink: SensorSink, now: () => number) {
    this.sink = sink
    this.now = now
    if (this.watchId !== null) return
    if (!window.isSecureContext) {
      this.set({ geo: 'insecure', geoMessage: 'Location needs HTTPS (or localhost).' })
      return
    }
    if (!('geolocation' in navigator)) {
      this.set({ geo: 'unavailable', geoMessage: 'This browser has no Geolocation API.' })
      return
    }
    this.set({ geo: 'waiting', geoMessage: null })
    const opts: PositionOptions = { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
    this.watchId = navigator.geolocation.watchPosition(this.handlePos, this.handleErr, opts)
  }

  private handlePos = (p: GeolocationPosition) => {
    // some browsers re-deliver an identical reading; ignore exact repeats
    const now = this.now()
    if (p.timestamp === this.lastFixStamp) return
    this.lastFixStamp = p.timestamp
    if (this.status.geo !== 'ok' || this.status.geoMessage) this.set({ geo: 'ok', geoMessage: null })
    const c = p.coords
    this.sink?.pushFix({
      lon: c.longitude,
      lat: c.latitude,
      accuracy: c.accuracy,
      speed: c.speed ?? null,
      course: c.heading !== null && !Number.isNaN(c.heading) ? c.heading : null,
      t: now,
    })
  }

  private handleErr = (e: GeolocationPositionError) => {
    if (e.code === e.PERMISSION_DENIED) {
      this.set({ geo: 'denied', geoMessage: 'Location permission denied.' })
    } else {
      this.set({ geoMessage: e.code === e.TIMEOUT ? 'GNSS timeout' : 'Position unavailable' })
    }
    this.sink?.gnssError(e.code === e.PERMISSION_DENIED ? 'permission denied' : e.code === e.TIMEOUT ? 'receiver timeout' : 'position unavailable')
  }

  /**
   * iOS requires DeviceMotion/Orientation permission from a user gesture.
   * Call this from a click handler (we use "Start navigation").
   */
  async requestMotionPermission(): Promise<boolean> {
    const dm = (window as unknown as { DeviceMotionEvent?: PermissionRequester }).DeviceMotionEvent
    const dor = (window as unknown as { DeviceOrientationEvent?: PermissionRequester }).DeviceOrientationEvent
    try {
      if (typeof dm?.requestPermission === 'function') {
        const r = await dm.requestPermission()
        if (r !== 'granted') {
          this.set({ motion: 'denied' })
          return false
        }
      }
      if (typeof dor?.requestPermission === 'function') await dor.requestPermission()
    } catch {
      this.set({ motion: 'denied' })
      return false
    }
    this.attachSensors()
    return true
  }

  /** Attach motion listeners where no permission prompt is needed (Android, desktop). */
  attachSensorsIfAllowed() {
    if (motionPermissionRequired()) {
      if (this.status.motion === 'idle') this.set({ motion: 'needs-permission' })
      return
    }
    this.attachSensors()
  }

  private attachSensors() {
    if (this.sensorsAttached) return
    this.sensorsAttached = true
    if (!('DeviceMotionEvent' in window)) {
      this.set({ motion: 'none' })
      return
    }
    this.set({ motion: 'waiting' })
    window.addEventListener('devicemotion', this.handleMotion)
    window.addEventListener('deviceorientationabsolute', this.handleOrientation as EventListener)
    window.addEventListener('deviceorientation', this.handleOrientation)
    // Laptops expose the API but never fire events.
    this.noMotionTimer = window.setTimeout(() => {
      if (this.status.motion === 'waiting') this.set({ motion: 'none' })
    }, 2500)
  }

  private handleMotion = (e: DeviceMotionEvent) => {
    const a = e.accelerationIncludingGravity ?? e.acceleration
    if (!a || a.x === null || a.y === null || a.z === null) return
    if (this.status.motion !== 'active') this.set({ motion: 'active' })
    this.sink?.pushMotion({
      ax: a.x,
      ay: a.y,
      az: a.z,
      includesGravity: e.accelerationIncludingGravity !== null,
      gyroZ: e.rotationRate?.alpha ?? null,
      t: this.now(),
    })
  }

  private handleOrientation = (e: DeviceOrientationEvent) => {
    const ios = (e as DeviceOrientationEvent & { webkitCompassHeading?: number }).webkitCompassHeading
    let heading: number | null = null
    let absolute = false
    if (typeof ios === 'number' && !Number.isNaN(ios)) {
      heading = ios
      absolute = true
    } else if (e.alpha !== null) {
      // alpha is counter-clockwise; compass heading is clockwise from north.
      heading = (360 - e.alpha) % 360
      absolute = e.absolute || e.type === 'deviceorientationabsolute'
    }
    if (heading === null) return
    const next = absolute ? 'absolute' : 'relative'
    if (this.status.compass !== next && !(this.status.compass === 'absolute' && !absolute)) {
      this.set({ compass: next })
    }
    // Relative-only orientation is ignored for heading; the estimator uses
    // the gyro yaw integral instead.
    if (absolute) this.sink?.pushHeading({ heading, absolute, t: this.now() })
  }

  stop() {
    if (this.watchId !== null) navigator.geolocation.clearWatch(this.watchId)
    if (this.noMotionTimer !== null) clearTimeout(this.noMotionTimer)
    this.watchId = null
    window.removeEventListener('devicemotion', this.handleMotion)
    window.removeEventListener('deviceorientationabsolute', this.handleOrientation as EventListener)
    window.removeEventListener('deviceorientation', this.handleOrientation)
    this.sensorsAttached = false
    this.status = { geo: 'idle', geoMessage: null, motion: 'idle', compass: 'none' }
  }
}
