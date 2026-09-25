#!/usr/bin/env node
/**
 * Bakes the Judge Demo route into src/demo/demoRoute.json.
 *
 *   npm run bake:demo
 *
 * 1. Asks the public OSRM server for a real driving route between the demo
 *    waypoints (with turn-by-turn steps).
 * 2. Converts the geometry into a timed {lat, lon, t} trace with a realistic
 *    urban speed profile (src/demo/timeline.js).
 * 3. Marks GNSS dead zones as fractions of route length.
 *
 * If OSRM is unreachable (offline / firewalled build machine) it writes a
 * hand-traced fallback along the same streets and flags it
 * `"source": "hand-traced"`; the app then retries OSRM in the browser and
 * upgrades the route at runtime. Re-run this script on any machine with
 * internet access to bake the real OSRM route permanently.
 */
import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { buildTimeline } from '../src/demo/timeline.js'

const here = dirname(fileURLToPath(import.meta.url))
const OUT = resolve(here, '../src/demo/demoRoute.json')

// ── Demo scenario: Vijay Chowk → Kartavya Path → C-Hexagon → Purana Qila ──
const ORIGIN = { name: 'Vijay Chowk', coord: [77.2100, 28.61381] }
const DESTINATION = { name: 'Purana Qila, Mathura Road', coord: [77.2410, 28.6079] }
const VIA = [[77.2262, 28.61305]] // keep the route on Kartavya Path
const DEAD_ZONES = [
  { from: 0.2, to: 0.28, label: 'Dead zone A — simulated underpass' },
  { from: 0.72, to: 0.79, label: 'Dead zone B — simulated urban canyon' },
]

async function fetchOsrm() {
  const wps = [ORIGIN.coord, ...VIA, DESTINATION.coord].map((c) => c.join(',')).join(';')
  const url = `https://router.project-osrm.org/route/v1/driving/${wps}?overview=full&geometries=geojson&steps=true`
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 15000)
  try {
    const res = await fetch(url, { signal: ctrl.signal })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    if (data.code !== 'Ok') throw new Error(data.message ?? data.code)
    return data.routes[0]
  } finally {
    clearTimeout(timer)
  }
}

/** Hand-traced fallback in OSRM's response shape so the app parses both identically. */
function handTraced() {
  const kartavya = (lon) => [lon, +(28.6143 - 0.0465 * (lon - 77.1994)).toFixed(6)]
  const coords = []
  for (let lon = 77.21; lon < 77.2262; lon += 0.0015) coords.push(kartavya(+lon.toFixed(5)))
  // C-Hexagon ring around India Gate, clockwise (left-hand traffic)
  const c = [77.22953, 28.61294]
  const rx = 0.0033
  const ry = 0.0029
  const ring = [180, 120, 60, 0, -60].map((deg) => [
    +(c[0] + rx * Math.cos((deg * Math.PI) / 180)).toFixed(6),
    +(c[1] + ry * Math.sin((deg * Math.PI) / 180)).toFixed(6),
  ])
  coords.push(...ring)
  const se = ring[ring.length - 1]
  coords.push([77.2338, 28.6095], [77.2365, 28.6086], [77.239, 28.6081], DESTINATION.coord)

  let d = 0
  const segLen = (a, b) => {
    const R = 6371008.8
    const r = Math.PI / 180
    const x = (b[0] - a[0]) * r * Math.cos(((a[1] + b[1]) / 2) * r)
    const y = (b[1] - a[1]) * r
    return Math.hypot(x, y) * R
  }
  const cumTo = (pt) => {
    let acc = 0
    for (let i = 1; i < coords.length; i++) {
      acc += segLen(coords[i - 1], coords[i])
      if (coords[i] === pt) return acc
    }
    return acc
  }
  for (let i = 1; i < coords.length; i++) d += segLen(coords[i - 1], coords[i])
  const west = ring[0]
  const dWest = cumTo(west)
  const dSe = cumTo(se)
  const step = (type, loc, name, distance, extra = {}) => ({
    name,
    distance,
    duration: distance / 10,
    maneuver: { type, location: loc, bearing_after: 95, ...extra },
  })
  return {
    geometry: { coordinates: coords },
    distance: d,
    duration: d / 10,
    legs: [
      {
        steps: [
          step('depart', coords[0], 'Kartavya Path', dWest),
          step('rotary', west, 'Purana Qila Road', dSe - dWest, { exit: 3, rotary_name: 'C-Hexagon' }),
          step('exit rotary', se, 'Purana Qila Road', d - dSe, { modifier: 'slight right' }),
          step('arrive', DESTINATION.coord, 'Purana Qila Road', 0),
        ],
      },
    ],
  }
}

let route
let source = 'osrm'
try {
  route = await fetchOsrm()
  console.log(`OSRM route: ${(route.distance / 1000).toFixed(2)} km, ${route.legs.flatMap((l) => l.steps).length} steps`)
} catch (err) {
  console.warn(`OSRM unreachable (${err.message}) — writing hand-traced fallback`)
  route = handTraced()
  source = 'hand-traced'
}

// Keep only what the app uses (smaller JSON).
const slim = {
  geometry: { coordinates: route.geometry.coordinates },
  distance: route.distance,
  duration: route.duration,
  legs: route.legs.map((l) => ({
    steps: l.steps.map((s) => ({
      name: s.name,
      ref: s.ref,
      distance: s.distance,
      duration: s.duration,
      rotary_name: s.rotary_name,
      maneuver: {
        type: s.maneuver.type,
        modifier: s.maneuver.modifier,
        exit: s.maneuver.exit,
        location: s.maneuver.location,
        bearing_after: s.maneuver.bearing_after,
      },
    })),
  })),
}

const samples = buildTimeline(route.geometry.coordinates)
const out = {
  name: 'Kartavya Path loop, New Delhi',
  source,
  generatedAt: new Date().toISOString(),
  origin: ORIGIN,
  destination: DESTINATION,
  waypoints: [ORIGIN.coord, ...VIA, DESTINATION.coord],
  deadZones: DEAD_ZONES,
  osrm: slim,
  samples,
}
writeFileSync(OUT, JSON.stringify(out) + '\n')
console.log(`Wrote ${OUT} — ${samples.length} samples, ${(samples[samples.length - 1].t / 1000).toFixed(0)} s of driving (${source})`)
