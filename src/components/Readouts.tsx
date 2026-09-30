import type { Snapshot } from '../engine/navEngine'
import { compass } from '../lib/geo'

function Cell({ label, value, unit, accent }: { label: string; value: string; unit?: string; accent?: string }) {
  return (
    <div className="border-b border-r border-hair px-2.5 py-2">
      <div className="label">{label}</div>
      <div className="num mt-0.5 truncate text-[15px] leading-tight" style={accent ? { color: accent } : undefined}>
        {value}
        {unit && <span className="ml-1 text-[11px] text-muted">{unit}</span>}
      </div>
    </div>
  )
}

export default function Readouts({ snap }: { snap: Snapshot }) {
  const e = snap.est
  const dr = e.mode === 'DR'
  const modeColor = e.mode === 'LIVE' ? 'var(--fused)' : dr ? 'var(--thread)' : 'var(--warn)'
  const cells = [
    { label: 'Source', value: e.mode === 'LIVE' ? 'GNSS' : dr ? (e.snapped ? 'DR·ROUTE' : 'DR·FREE') : e.position ? 'HOLDOVER' : '—', accent: modeColor },
    { label: 'σ fused', value: e.position ? e.sigma.toFixed(e.sigma < 10 ? 1 : 0) : '—', unit: 'm' },
    { label: 'GNSS acc', value: e.gnssAccuracy !== null ? e.gnssAccuracy.toFixed(0) : '—', unit: 'm' },
    { label: 'Speed', value: e.position ? (e.speed * 3.6).toFixed(0) : '—', unit: 'km/h' },
    { label: 'Heading', value: e.position ? `${Math.round(e.heading).toString().padStart(3, '0')}°` : '—', unit: e.position ? `${compass(e.heading)}·${e.headingSource}` : undefined },
    { label: 'Fix age', value: e.fixAgeMs !== null ? (e.fixAgeMs / 1000).toFixed(1) : '—', unit: 's' },
    { label: 'Outage', value: dr ? (e.outageMs / 1000).toFixed(1) : '0.0', unit: 's', accent: dr ? 'var(--thread)' : undefined },
    { label: 'DR dist', value: dr ? e.drDistance.toFixed(0) : '0', unit: 'm' },
    snap.demo
      ? {
          label: 'Err vs truth',
          value: snap.demo.truthError !== null ? snap.demo.truthError.toFixed(1) : '—',
          unit: 'm',
          accent: dr ? 'var(--thread)' : undefined,
        }
      : { label: 'Motion', value: e.motion.toUpperCase() },
  ]
  return (
    <div>
      <div className="grid grid-cols-3 border-l border-t border-hair">
        {cells.map((c) => (
          <Cell key={c.label} {...c} />
        ))}
      </div>
      <div className="mt-2 flex items-baseline gap-2 text-[11.5px] leading-snug">
        <span className="label shrink-0">Engine</span>
        <span className="num shrink-0" style={{ color: e.engine === 'learned' ? 'var(--fused)' : 'var(--muted)' }}>
          {e.engine === 'learned' ? 'LEARNED' : 'HEURISTIC'}
        </span>
        <span className="min-w-0 text-muted">{e.engineNote}</span>
      </div>
    </div>
  )
}
