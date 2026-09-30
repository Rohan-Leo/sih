/** Destination search via the public Photon geocoder (komoot) — no API key. */
import type { LngLat } from '../lib/geo'

export interface Place {
  id: string
  name: string
  detail: string
  coord: LngLat
  kind: string
}

interface PhotonFeature {
  geometry: { coordinates: [number, number] }
  properties: Record<string, string | number | undefined>
}

export async function searchPlaces(q: string, bias: LngLat | null, signal?: AbortSignal): Promise<Place[]> {
  const url = new URL('https://photon.komoot.io/api/')
  url.searchParams.set('q', q)
  url.searchParams.set('limit', '7')
  url.searchParams.set('lang', 'en')
  if (bias) {
    url.searchParams.set('lon', bias[0].toFixed(5))
    url.searchParams.set('lat', bias[1].toFixed(5))
  }
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`Search failed (${res.status})`)
  const data = (await res.json()) as { features: PhotonFeature[] }
  const seen = new Set<string>()
  const out: Place[] = []
  for (const f of data.features ?? []) {
    const p = f.properties
    const name = String(p.name ?? p.street ?? p.city ?? 'Unnamed place')
    const street = p.street ? `${p.housenumber ? p.housenumber + ' ' : ''}${p.street}` : ''
    const parts = [street && street !== name ? street : '', p.district, p.city ?? p.county, p.state, p.country]
      .filter((x): x is string => typeof x === 'string' && x.length > 0 && x !== name)
    const detail = [...new Set(parts)].join(', ')
    const key = `${name}|${detail}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push({
      id: `${p.osm_type ?? ''}${p.osm_id ?? out.length}`,
      name,
      detail,
      coord: f.geometry.coordinates,
      kind: String(p.osm_value ?? p.type ?? ''),
    })
  }
  return out
}
