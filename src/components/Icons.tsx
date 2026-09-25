/** Hairline instrument glyphs — drawn for Clew, 1.25px strokes, no fills-in-circles. */
import type { SVGProps } from 'react'

type P = SVGProps<SVGSVGElement>
const base = (p: P): P => ({
  width: 16,
  height: 16,
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.25,
  strokeLinecap: 'square',
  ...p,
})

export const IconSearch = (p: P) => (
  <svg {...base(p)}>
    <rect x="2.5" y="2.5" width="8" height="8" rx="1" />
    <path d="M10.5 10.5 14 14" />
  </svg>
)
export const IconSun = (p: P) => (
  <svg {...base(p)}>
    <rect x="5" y="5" width="6" height="6" rx="1" />
    <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6 13 13M3 13l1.4-1.4M11.6 4.4 13 3" />
  </svg>
)
export const IconMoon = (p: P) => (
  <svg {...base(p)}>
    <path d="M12.5 10.5A5.5 5.5 0 0 1 5.5 3.5a5.5 5.5 0 1 0 7 7Z" />
  </svg>
)
export const IconPlay = (p: P) => (
  <svg {...base(p)}>
    <path d="M4.5 3v10l8-5-8-5Z" strokeLinejoin="round" />
  </svg>
)
export const IconPause = (p: P) => (
  <svg {...base(p)}>
    <path d="M5 3v10M11 3v10" />
  </svg>
)
export const IconReset = (p: P) => (
  <svg {...base(p)}>
    <path d="M3 8a5 5 0 1 0 1.5-3.6" />
    <path d="M3 2.5v2.5h2.5" />
  </svg>
)
export const IconRecenter = (p: P) => (
  <svg {...base(p)}>
    <path d="M8 1v4M8 11v4M1 8h4M11 8h4" />
    <rect x="6" y="6" width="4" height="4" />
  </svg>
)
export const IconNorth = (p: P) => (
  <svg {...base(p)}>
    <path d="M8 2 12 13 8 10.5 4 13 8 2Z" strokeLinejoin="round" />
  </svg>
)
export const IconPlus = (p: P) => (
  <svg {...base(p)}>
    <path d="M8 3v10M3 8h10" />
  </svg>
)
export const IconMinus = (p: P) => (
  <svg {...base(p)}>
    <path d="M3 8h10" />
  </svg>
)
export const IconClose = (p: P) => (
  <svg {...base(p)}>
    <path d="m3.5 3.5 9 9M12.5 3.5l-9 9" />
  </svg>
)
export const IconChevron = (p: P) => (
  <svg {...base(p)}>
    <path d="m4 6 4 4 4-4" />
  </svg>
)

/** Ball-of-thread mark */
export const ClewMark = (p: P) => (
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none" {...p}>
    <circle cx="10" cy="11" r="7.25" stroke="var(--thread)" strokeWidth="1.5" />
    <path d="M3.6 8.6c4.2 1.6 8.8-1.6 13 .8M3.9 13.6c4.1-1.5 8.2 1.6 12.3-.6M7.4 4.4c1.4 4.4-1.1 8.6.6 13.1" stroke="var(--thread)" strokeWidth="1.1" />
    <path d="M17.2 11c2.3 0 2.8 3.6 4.3 4.6" stroke="var(--fused)" strokeWidth="1.4" strokeLinecap="round" />
  </svg>
)

/** Maneuver arrow, drawn per OSRM type/modifier. */
export function ManeuverGlyph({ type, modifier, size = 36 }: { type?: string; modifier?: string; size?: number }) {
  const common = { stroke: 'currentColor', strokeWidth: 2.2, fill: 'none', strokeLinecap: 'square' as const, strokeLinejoin: 'miter' as const }
  let d = 'M18 32V6M11 13l7-7 7 7'
  if (type === 'arrive') {
    return (
      <svg width={size} height={size} viewBox="0 0 36 36">
        <path d="M18 32V14" {...common} />
        <rect x="10" y="4" width="16" height="10" {...common} strokeWidth={2} />
      </svg>
    )
  }
  if (type === 'roundabout' || type === 'rotary' || type === 'exit roundabout' || type === 'exit rotary') {
    return (
      <svg width={size} height={size} viewBox="0 0 36 36">
        <path d="M14 33V23a7.5 7.5 0 1 1 11 -5.5L31 11" {...common} />
        <path d="M25.5 10H31v5.5" {...common} />
      </svg>
    )
  }
  switch (modifier) {
    case 'right':
      d = 'M12 32V16h16M22 10l6 6-6 6'
      break
    case 'sharp right':
      d = 'M12 6v18l14-4M20 14l6 6-6 4'
      break
    case 'slight right':
      d = 'M14 32V20l10-12M18 8h6v6'
      break
    case 'left':
      d = 'M24 32V16H8M14 10l-6 6 6 6'
      break
    case 'sharp left':
      d = 'M24 6v18l-14-4M16 14l-6 6 6 4'
      break
    case 'slight left':
      d = 'M22 32V20L12 8M18 8h-6v6'
      break
    case 'uturn':
      d = 'M24 32V12a6 6 0 0 0-12 0v8M8 16l4 5 4-5'
      break
  }
  return (
    <svg width={size} height={size} viewBox="0 0 36 36">
      <path d={d} {...common} />
    </svg>
  )
}
