/** Driving routes + turn-by-turn steps from the public OSRM demo server — no API key. */
import type { LngLat } from '../lib/geo'

export interface Maneuver {
  type: string
  modifier?: string
  exit?: number
  location: LngLat
  bearingAfter?: number
}

export interface RouteStep {
  instruction: string
  name: string
  distance: number
  duration: number
  maneuver: Maneuver
  /** distance along the whole route at which this step's maneuver happens (filled by buildRoute) */
  at: number
}

export interface Route {
  coords: LngLat[]
  steps: RouteStep[]
  distance: number
  duration: number
  destinationName: string
}

interface OsrmStep {
  name: string
  ref?: string
  distance: number
  duration: number
  rotary_name?: string
  destinations?: string
  maneuver: { type: string; modifier?: string; exit?: number; location: [number, number]; bearing_after?: number }
}

export async function fetchRoute(
  waypoints: LngLat[],
  destinationName: string,
  signal?: AbortSignal,
): Promise<Route> {
  const path = waypoints.map((w) => `${w[0].toFixed(6)},${w[1].toFixed(6)}`).join(';')
  const url = `https://router.project-osrm.org/route/v1/driving/${path}?overview=full&geometries=geojson&steps=true`
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`Routing failed (${res.status})`)
  const data = await res.json()
  if (data.code !== 'Ok' || !data.routes?.length) throw new Error(data.message ?? 'No route found')
  return fromOsrm(data.routes[0], destinationName)
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function fromOsrm(r: any, destinationName: string): Route {
  const steps: RouteStep[] = []
  for (const leg of r.legs as { steps: OsrmStep[] }[]) {
    for (const s of leg.steps) {
      // intermediate "arrive" at via points is noise for the driver
      if (s.maneuver.type === 'arrive' && leg !== r.legs[r.legs.length - 1]) continue
      if (s.maneuver.type === 'depart' && steps.length > 0) continue
      steps.push({
        instruction: instructionFor(s),
        name: roadName(s),
        distance: s.distance,
        duration: s.duration,
        maneuver: {
          type: s.maneuver.type,
          modifier: s.maneuver.modifier,
          exit: s.maneuver.exit,
          location: s.maneuver.location,
          bearingAfter: s.maneuver.bearing_after,
        },
        at: 0,
      })
    }
  }
  return {
    coords: r.geometry.coordinates,
    steps,
    distance: r.distance,
    duration: r.duration,
    destinationName,
  }
}

function roadName(s: OsrmStep): string {
  if (s.name && s.ref) return `${s.name} (${s.ref})`
  return s.name || s.ref || ''
}

const ORD = ['', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']

/** Turn an OSRM step into a Google-Maps-shaped sentence. */
export function instructionFor(s: OsrmStep): string {
  const { type, modifier, exit } = s.maneuver
  const road = roadName(s)
  const onto = road ? ` onto ${road}` : ''
  const dir = modifier ? modifier.replace('uturn', 'U-turn') : 'straight'
  const turnWord = (m?: string) => {
    switch (m) {
      case 'uturn':
        return 'Make a U-turn'
      case 'sharp right':
        return 'Turn sharp right'
      case 'right':
        return 'Turn right'
      case 'slight right':
        return 'Bear right'
      case 'straight':
        return 'Continue straight'
      case 'slight left':
        return 'Bear left'
      case 'left':
        return 'Turn left'
      case 'sharp left':
        return 'Turn sharp left'
      default:
        return 'Continue'
    }
  }
  switch (type) {
    case 'depart':
      return road ? `Head ${headingWord(s.maneuver.bearing_after)} on ${road}` : `Head ${headingWord(s.maneuver.bearing_after)}`
    case 'arrive':
      return modifier && modifier !== 'straight' ? `Arrive at destination, on the ${modifier.replace('slight ', '').replace('sharp ', '')}` : 'Arrive at destination'
    case 'turn':
    case 'end of road':
      return `${turnWord(modifier)}${onto}`
    case 'new name':
    case 'continue':
      return modifier === 'straight' || !modifier ? `Continue${onto}` : `${turnWord(modifier)}${onto}`
    case 'merge':
      return `Merge ${dir.includes('left') ? 'left' : dir.includes('right') ? 'right' : ''}${onto}`.replace('  ', ' ')
    case 'on ramp':
      return `Take the ramp${onto}`
    case 'off ramp':
      return `Take the exit${onto}`
    case 'fork':
      return `Keep ${dir.includes('left') ? 'left' : 'right'} at the fork${onto}`
    case 'roundabout':
    case 'rotary': {
      const name = s.rotary_name ? ` (${s.rotary_name})` : ''
      return exit ? `At the roundabout${name}, take the ${ORD[exit] ?? exit + 'th'} exit${onto}` : `Enter the roundabout${name}${onto}`
    }
    case 'roundabout turn':
      return `At the roundabout, ${turnWord(modifier).toLowerCase()}${onto}`
    case 'exit roundabout':
    case 'exit rotary':
      return `Exit the roundabout${onto}`
    default:
      return `${turnWord(modifier)}${onto}`
  }
}

function headingWord(b?: number): string {
  if (b === undefined) return 'out'
  const words = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
  return words[Math.round((((b % 360) + 360) % 360) / 45) % 8]
}
