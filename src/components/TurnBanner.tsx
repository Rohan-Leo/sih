import { motion } from 'framer-motion'
import type { Progress } from '../engine/navEngine'
import { formatDistance } from '../lib/geo'
import { ManeuverGlyph } from './Icons'

export default function TurnBanner({ progress, dr }: { progress: Progress; dr: boolean }) {
  const s = progress.nextStep
  if (!s) return null
  if (progress.arrived) {
    return (
      <div className="ticks flex items-center gap-3 rounded-sm bg-ink px-4 py-3 text-paper">
        <ManeuverGlyph type="arrive" size={32} />
        <div>
          <div className="font-display text-lg leading-tight">You have arrived</div>
          <div className="text-xs opacity-75">{s.name || 'Destination'}</div>
        </div>
      </div>
    )
  }
  return (
    <div className="ticks overflow-hidden rounded-sm bg-ink text-paper">
      <motion.div
          key={progress.nextStepIdx}
          initial={{ opacity: 0, x: 12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.2 }}
          className="flex items-center gap-3 px-3.5 py-3"
        >
          <div className="shrink-0">
            <ManeuverGlyph type={s.maneuver.type} modifier={s.maneuver.modifier} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="num text-2xl leading-none">{formatDistance(progress.distToManeuver)}</div>
            <div className="mt-1 truncate text-[15px] leading-snug">{s.instruction}</div>
          </div>
        </motion.div>
      {(progress.thenStep || dr) && (
        <div className="flex items-center justify-between gap-3 border-t border-white/15 px-3.5 py-1.5 text-[11px]">
          <span className="truncate opacity-70">{progress.thenStep ? `Then: ${progress.thenStep.instruction}` : ''}</span>
          {dr && (
            <span className="num shrink-0 tracking-wider" style={{ color: 'var(--thread)' }}>
              EST. POSITION
            </span>
          )}
        </div>
      )}
    </div>
  )
}
