/**
 * OpenFreeMap basemaps (keyless) + paint overrides that pull them into
 * Clew's warm paper / warm-black palette.
 */
import type { Map as MLMap, StyleSpecification } from 'maplibre-gl'

export type Theme = 'light' | 'dark'

export const STYLE_URL: Record<Theme, string> = {
  light: 'https://tiles.openfreemap.org/styles/positron',
  dark: 'https://tiles.openfreemap.org/styles/dark',
}

const PALETTE = {
  light: {
    background: '#F1EBDF',
    water: '#C9D8D4',
    park: '#E3E4CE',
    building: '#E6DCCB',
    landuse: '#EEE6D8',
    road: '#FFFDF8',
    roadCasing: '#DCD1BE',
    roadMajor: '#FBF4E6',
    text: '#3C4450',
    textHalo: '#F4EFE6',
    boundary: '#B9AE9C',
  },
  dark: {
    background: '#1B1711',
    water: '#132020',
    park: '#1F2117',
    building: '#2A231A',
    landuse: '#201B14',
    road: '#3A3126',
    roadCasing: '#231D16',
    roadMajor: '#4A3E2F',
    text: '#C9BFAE',
    textHalo: '#19150F',
    boundary: '#5B5042',
  },
}

/** Minimal offline style: paper background + blueprint grid feel, no tiles. */
export function fallbackStyle(theme: Theme): StyleSpecification {
  return {
    version: 8,
    name: 'clew-offline',
    sources: {},
    layers: [{ id: 'background', type: 'background', paint: { 'background-color': PALETTE[theme].background } }],
  }
}

function safe(fn: () => void) {
  try {
    fn()
  } catch {
    /* layer lacks this property in this style — ignore */
  }
}

export function applyPaletteOverrides(map: MLMap, theme: Theme) {
  const p = PALETTE[theme]
  const layers = map.getStyle()?.layers ?? []
  for (const layer of layers) {
    const id = layer.id.toLowerCase()
    const type = layer.type
    if (type === 'background') {
      safe(() => map.setPaintProperty(layer.id, 'background-color', p.background))
    } else if (type === 'fill') {
      let color: string | null = null
      if (id.includes('water') || id.includes('ocean')) color = p.water
      else if (/park|wood|grass|forest|landcover|green|cemetery|pitch|golf/.test(id)) color = p.park
      else if (id.includes('building')) color = p.building
      else if (/landuse|residential|aeroway|industrial|commercial/.test(id)) color = p.landuse
      if (color) {
        const c = color
        safe(() => map.setPaintProperty(layer.id, 'fill-color', c))
        if (id.includes('building')) safe(() => map.setPaintProperty(layer.id, 'fill-outline-color', p.roadCasing))
      }
    } else if (type === 'fill-extrusion') {
      safe(() => map.setPaintProperty(layer.id, 'fill-extrusion-color', p.building))
    } else if (type === 'line') {
      if (id.includes('water') || id.includes('river') || id.includes('stream')) {
        safe(() => map.setPaintProperty(layer.id, 'line-color', p.water))
      } else if (/boundary|admin/.test(id)) {
        safe(() => map.setPaintProperty(layer.id, 'line-color', p.boundary))
      } else if (/road|highway|street|transportation|tunnel|bridge|motorway|trunk|primary|secondary|tertiary|minor|service|path/.test(id)) {
        const casing = id.includes('casing') || id.includes('outline')
        const major = /motorway|trunk|primary/.test(id)
        const col = casing ? p.roadCasing : major ? p.roadMajor : p.road
        safe(() => map.setPaintProperty(layer.id, 'line-color', col))
      }
    } else if (type === 'symbol') {
      safe(() => map.setPaintProperty(layer.id, 'text-color', p.text))
      safe(() => map.setPaintProperty(layer.id, 'text-halo-color', p.textHalo))
    }
  }
}
