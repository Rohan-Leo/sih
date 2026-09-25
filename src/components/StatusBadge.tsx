import { motion } from 'framer-motion'
import type { EstimatorState } from '../engine/positionEstimator'

const LABEL = { LIVE: 'GNSS LIVE', SEARCHING: 'SEARCHING', DR: 'DEAD RECKONING' } as const

export default function StatusBadge({ est, compact = false }: { est: EstimatorState; compact?: boolean }) {
  const color = est.mode === 'LIVE' ? 'var(--fused)' : est.mode === 'DR' ? 'var(--thread)' : 'var(--warn)'
  const detail =
    est.mode === 'LIVE'
      ? `±${Math.round(est.gnssAccuracy ?? est.sigma)} m`
      : est.mode === 'DR'
        ? `${(est.outageMs / 1000).toFixed(1)} s · σ ${Math.round(est.sigma)} m`
        : est.position
          ? `fix ${((est.fixAgeMs ?? 0) / 1000).toFixed(1)} s old`
          : 'acquiring'
  return (
    <div
      role="status"
      aria-live="polite"
      className="ticks flex items-stretch rounded-sm border bg-panel text-ink"
      style={{ borderColor: color }}
    >
      <div className="flex items-center gap-2 px-2.5 py-1.5" style={{ background: `color-mix(in srgb, ${color} 12%, transparent)` }}>
        <span className="relative inline-flex h-2 w-2">
          {est.mode !== 'LIVE' && (
            <span className="dr-pulse absolute inset-0 rounded-[1px]" style={{ background: color }} />
          )}
          <span className="relative inline-block h-2 w-2 rounded-[1px]" style={{ background: color }} />
        </span>
        <motion.span
          key={est.mode}
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18 }}
          className="font-mono text-[11px] font-medium tracking-[0.14em]"
          style={{ color }}
        >
          {LABEL[est.mode]}
        </motion.span>
      </div>
      {!compact && <div className="num flex items-center border-l border-hair px-2.5 text-[11px] text-muted">{detail}</div>}
    </div>
  )
}
