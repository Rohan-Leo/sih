import { useEffect, useRef } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import '../lib/maplibreWorker'
import type { Feature, FeatureCollection } from 'geojson'
import type { NavEngine, Snapshot } from '../engine/navEngine'
import { type LngLat, circlePolygon } from '../lib/geo'
import { STYLE_URL, type Theme, applyPaletteOverrides, fallbackStyle } from '../lib/mapStyle'

interface Props {
  engine: NavEngine
  snap: Snapshot
  theme: Theme
  follow: boolean
  headingUp: boolean
  onUserPan: () => void
  padding: { top: number; bottom: number; left: number; right: number }
  onMap?: (m: maplibregl.Map | null) => void
}

const EMPTY: FeatureCollection = { type: 'FeatureCollection', features: [] }

const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

function lineFeature(coords: LngLat[], props: Record<string, unknown> = {}): Feature {
  return { type: 'Feature', properties: props, geometry: { type: 'LineString', coordinates: coords } }
}

/** Add Clew's own sources/layers on top of whatever basemap is loaded. */
function ensureOverlay(map: maplibregl.Map) {
  const thread = cssVar('--thread') || '#C1622D'
  const fused = cssVar('--fused') || '#2F7A78'
  const ink = cssVar('--ink') || '#1B2430'
  const paper = cssVar('--paper') || '#F4EFE6'

  const src = (id: string) => {
    if (!map.getSource(id)) map.addSource(id, { type: 'geojson', data: EMPTY })
  }
  ;['route', 'deadzones', 'truth', 'trail', 'halo', 'maneuver'].forEach(src)

  const add = (layer: maplibregl.AddLayerObject) => {
    if (!map.getLayer(layer.id)) map.addLayer(layer)
  }
  add({
    id: 'route-casing',
    type: 'line',
    source: 'route',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': paper, 'line-width': ['interpolate', ['linear'], ['zoom'], 10, 5, 16, 12] },
  })
  add({
    id: 'route-line',
    type: 'line',
    source: 'route',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': fused,
      'line-opacity': 0.55,
      'line-width': ['interpolate', ['linear'], ['zoom'], 10, 2.5, 16, 7],
    },
  })
  add({
    id: 'deadzone-line',
    type: 'line',
    source: 'deadzones',
    layout: { 'line-cap': 'butt' },
    paint: {
      'line-color': ink,
      'line-opacity': 0.55,
      'line-width': ['interpolate', ['linear'], ['zoom'], 10, 4, 16, 13],
      'line-dasharray': [0.35, 0.35],
    },
  })
  if (map.getStyle().glyphs) add({
    id: 'deadzone-label',
    type: 'symbol',
    source: 'deadzones',
    layout: {
      'symbol-placement': 'line-center',
      'text-field': ['get', 'short'],
      'text-size': 11,
      'text-font': ['Noto Sans Regular'],
      'text-offset': [0, -1.4],
      'text-letter-spacing': 0.08,
    },
    paint: { 'text-color': ink, 'text-halo-color': paper, 'text-halo-width': 2 },
  })
  add({
    id: 'truth-line',
    type: 'line',
    source: 'truth',
    layout: { 'line-cap': 'round' },
    paint: { 'line-color': ink, 'line-opacity': 0.35, 'line-width': 1.2, 'line-dasharray': [1, 2.5] },
  })
  add({
    id: 'halo-fill',
    type: 'fill',
    source: 'halo',
    paint: { 'fill-color': ['get', 'color'], 'fill-opacity': ['get', 'opacity'] },
  })
  add({
    id: 'halo-edge',
    type: 'line',
    source: 'halo',
    paint: { 'line-color': ['get', 'color'], 'line-opacity': 0.55, 'line-width': 1, 'line-dasharray': [2, 2] },
  })
  // The thread: solid where GNSS was trusted, stitched where we dead-reckoned.
  add({
    id: 'trail-under',
    type: 'line',
    source: 'trail',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': paper, 'line-width': 6, 'line-opacity': 0.85 },
  })
  add({
    id: 'trail-gnss',
    type: 'line',
    source: 'trail',
    filter: ['==', ['get', 'kind'], 'gnss'],
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': thread, 'line-width': 3 },
  })
  add({
    id: 'trail-dr',
    type: 'line',
    source: 'trail',
    filter: ['==', ['get', 'kind'], 'dr'],
    layout: { 'line-cap': 'butt', 'line-join': 'round' },
    paint: { 'line-color': thread, 'line-width': 3, 'line-dasharray': [1.2, 1] },
  })
  add({
    id: 'maneuver-pt',
    type: 'circle',
    source: 'maneuver',
    paint: {
      'circle-radius': 5,
      'circle-color': paper,
      'circle-stroke-color': ink,
      'circle-stroke-width': 1.5,
    },
  })
}

export default function MapView({ engine, snap, theme, follow, headingUp, onUserPan, padding, onMap }: Props) {
  const el = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const markerRef = useRef<maplibregl.Marker | null>(null)
  const markerElRef = useRef<HTMLDivElement | null>(null)
  const rawRef = useRef<maplibregl.Marker | null>(null)
  const truthRef = useRef<maplibregl.Marker | null>(null)
  const destRef = useRef<maplibregl.Marker | null>(null)
  const followRef = useRef(follow)
  const headingUpRef = useRef(headingUp)
  const styleReady = useRef(false)
  const lastTrailVersion = useRef(-1)
  const lastTrailPush = useRef(0)
  const centeredOnce = useRef(false)
  const onUserPanRef = useRef(onUserPan)
  const snapRef = useRef(snap)
  snapRef.current = snap
  followRef.current = follow
  headingUpRef.current = headingUp
  onUserPanRef.current = onUserPan

  // ── create map once ──
  useEffect(() => {
    if (!el.current) return
    const map = new maplibregl.Map({
      container: el.current,
      style: STYLE_URL[theme],
      center: [78.96, 22.6],
      zoom: 4,
      attributionControl: { compact: true },
      pitchWithRotate: false,
    })
    mapRef.current = map
    onMap?.(map)

    const dot = document.createElement('div')
    dot.className = 'clew-dot'
    dot.innerHTML = `
      <svg width="44" height="44" viewBox="-22 -22 44 44" style="overflow:visible;display:block">
        <circle class="dr-ring" r="13" fill="none" stroke="var(--thread)" stroke-width="1.5" stroke-dasharray="3 3" style="display:none"/>
        <circle r="11" fill="var(--paper)" stroke="var(--fused)" stroke-width="1.5"/>
        <path class="dot-arrow" d="M0 -8 L6 6 L0 3 L-6 6 Z" fill="var(--fused)"/>
      </svg>`
    markerElRef.current = dot
    markerRef.current = new maplibregl.Marker({ element: dot, rotationAlignment: 'map', pitchAlignment: 'map' })

    const raw = document.createElement('div')
    raw.innerHTML = `<svg width="14" height="14" viewBox="-7 -7 14 14" style="display:block"><path d="M-6 0H6M0 -6V6" stroke="var(--ink)" stroke-width="1.2" opacity="0.7"/><rect x="-3" y="-3" width="6" height="6" fill="none" stroke="var(--ink)" stroke-width="1" opacity="0.7"/></svg>`
    raw.title = 'Raw GNSS fix'
    rawRef.current = new maplibregl.Marker({ element: raw })

    const truth = document.createElement('div')
    truth.innerHTML = `<svg width="12" height="12" viewBox="-6 -6 12 12" style="display:block"><circle r="4" fill="none" stroke="var(--ink)" stroke-width="1.2" stroke-dasharray="2 1.5" opacity="0.8"/></svg>`
    truth.title = 'Ground truth (demo only)'
    truthRef.current = new maplibregl.Marker({ element: truth })

    const dest = document.createElement('div')
    dest.innerHTML = `<svg width="26" height="34" viewBox="0 0 26 34" style="display:block"><path d="M13 33 L13 14" stroke="var(--ink)" stroke-width="1.5"/><rect x="3" y="2" width="20" height="13" rx="2" fill="var(--ink)"/><path d="M8 8.5 h10 M13 5 v7" stroke="var(--paper)" stroke-width="1.4"/></svg>`
    destRef.current = new maplibregl.Marker({ element: dest, anchor: 'bottom' })

    map.on('style.load', () => {
      applyPaletteOverrides(map, document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light')
      ensureOverlay(map)
      styleReady.current = true
      lastTrailVersion.current = -1
      pushStatic(map, snapRef.current, engine)
    })
    // Basemap unreachable (offline, firewalled venue Wi-Fi)? Swap in a plain
    // paper style so the route, thread and position still render.
    map.on('error', (e) => {
      const msg = String((e as unknown as { error?: { message?: string } }).error?.message ?? '')
      if (!styleReady.current && msg.includes('/styles/')) {
        map.setStyle(fallbackStyle(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'), { diff: false })
      }
    })
    const userMove = (e: { originalEvent?: unknown }) => {
      if (e.originalEvent) onUserPanRef.current()
    }
    map.on('dragstart', userMove)
    map.on('rotatestart', userMove)

    return () => {
      onMap?.(null)
      map.remove()
      mapRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── theme switch → swap basemap, re-add overlay on style.load ──
  const firstTheme = useRef(true)
  useEffect(() => {
    if (firstTheme.current) {
      firstTheme.current = false
      return
    }
    const map = mapRef.current
    if (!map) return
    styleReady.current = false
    map.setStyle(STYLE_URL[theme], { diff: false })
  }, [theme])

  // ── padding for side panel / bottom sheet ──
  useEffect(() => {
    mapRef.current?.setPadding(padding)
  }, [padding.top, padding.bottom, padding.left, padding.right]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── route, dead zones, destination (change rarely) ──
  useEffect(() => {
    const map = mapRef.current
    if (!map || !styleReady.current) return
    pushStatic(map, snap, engine)
  }, [snap.route, snap.demo?.source, snap.mode]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const map = mapRef.current
    const d = snap.route?.destination
    if (!map || !destRef.current) return
    if (d) destRef.current.setLngLat(d).addTo(map)
    else destRef.current.remove()
  }, [snap.route])

  // Fit the route when a new one arrives (and we're not following).
  useEffect(() => {
    const map = mapRef.current
    const r = snap.route
    if (!map || !r) return
    const b = new maplibregl.LngLatBounds()
    r.line.coords.forEach((c) => b.extend(c))
    centeredOnce.current = true
    map.fitBounds(b, { padding: 60, duration: 700, maxZoom: 16 })
  }, [snap.route])

  // ── next maneuver marker ──
  useEffect(() => {
    const map = mapRef.current
    if (!map || !styleReady.current) return
    const s = map.getSource('maneuver') as maplibregl.GeoJSONSource | undefined
    const st = snap.progress?.nextStep
    s?.setData(
      st && st.maneuver.type !== 'arrive'
        ? { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: st.maneuver.location } }
        : EMPTY,
    )
  }, [snap.progress?.nextStepIdx, snap.route]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── per-frame updates ──
  useEffect(() => {
    return engine.onFrame(({ est, truth }) => {
      const map = mapRef.current
      if (!map) return
      const pos = est.position
      const marker = markerRef.current!
      if (pos) {
        marker.setLngLat(pos)
        if (!marker.getElement().isConnected) marker.addTo(map)
        marker.setRotation(est.heading)
        const ring = markerElRef.current?.querySelector('.dr-ring') as SVGElement | null
        const arrow = markerElRef.current?.querySelector('.dot-arrow') as SVGElement | null
        if (ring) ring.style.display = est.mode === 'DR' ? 'block' : 'none'
        if (arrow) arrow.setAttribute('fill', est.mode === 'DR' ? 'var(--thread)' : 'var(--fused)')

        if (!centeredOnce.current) {
          centeredOnce.current = true
          map.jumpTo({ center: pos, zoom: 16 })
        } else if (followRef.current) {
          map.jumpTo({ center: pos, bearing: headingUpRef.current ? est.heading : map.getBearing() })
        }
      } else if (marker.getElement().isConnected) {
        marker.remove()
        centeredOnce.current = false
      }

      // raw GNSS fix crosshair (only while it's meaningfully distinct)
      if (est.rawFix && est.mode !== 'DR' && est.mode !== 'SEARCHING') {
        rawRef.current!.setLngLat(est.rawFix)
        if (!rawRef.current!.getElement().isConnected) rawRef.current!.addTo(map)
      } else if (rawRef.current!.getElement().isConnected) rawRef.current!.remove()

      if (truth) {
        truthRef.current!.setLngLat(truth)
        if (!truthRef.current!.getElement().isConnected) truthRef.current!.addTo(map)
      } else if (truthRef.current!.getElement().isConnected) truthRef.current!.remove()

      if (!styleReady.current) return
      const halo = map.getSource('halo') as maplibregl.GeoJSONSource | undefined
      if (halo && pos) {
        const dr = est.mode === 'DR' || est.blending
        halo.setData({
          type: 'Feature',
          properties: {
            color: dr ? cssVar('--thread') : cssVar('--fused'),
            opacity: dr ? 0.14 : 0.1,
          },
          geometry: { type: 'Polygon', coordinates: [circlePolygon(pos, Math.max(3, est.sigma))] },
        })
      }

      const now = performance.now()
      if (engine.trailVersion !== lastTrailVersion.current && now - lastTrailPush.current > 100) {
        lastTrailVersion.current = engine.trailVersion
        lastTrailPush.current = now
        const trail = map.getSource('trail') as maplibregl.GeoJSONSource | undefined
        trail?.setData({
          type: 'FeatureCollection',
          features: engine.trail
            .filter((s) => s.coords.length > 1)
            .map((s) => lineFeature(s.coords, { kind: s.kind })),
        })
        const tr = map.getSource('truth') as maplibregl.GeoJSONSource | undefined
        tr?.setData(engine.truthTrail.length > 1 ? lineFeature(engine.truthTrail) : EMPTY)
      }
    })
  }, [engine])

  return (
    <div className="absolute inset-0">
      <div ref={el} className="h-full w-full" aria-label="Map" />
    </div>
  )
}

function pushStatic(map: maplibregl.Map, snap: Snapshot, _engine: NavEngine) {
  const route = map.getSource('route') as maplibregl.GeoJSONSource | undefined
  route?.setData(snap.route ? lineFeature(snap.route.line.coords) : EMPTY)
  const dz = map.getSource('deadzones') as maplibregl.GeoJSONSource | undefined
  if (dz) {
    const zones = snap.demo && snap.route ? snap.demo.deadZones : []
    const line = snap.route?.line
    dz.setData({
      type: 'FeatureCollection',
      features: line
        ? zones.map((z) =>
            lineFeature(line.slice((z.from / snap.demo!.pathLength) * line.length, (z.to / snap.demo!.pathLength) * line.length), {
              label: z.label,
              short: z.label.split('—')[0].trim().toUpperCase(),
            }),
          )
        : [],
    })
  }
}
