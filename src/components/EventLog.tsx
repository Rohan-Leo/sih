import { AnimatePresence, motion } from 'framer-motion'
import type { LogEntry } from '../engine/navEngine'

const COLOR: Record<LogEntry['kind'], string> = {
  info: 'var(--muted)',
  route: 'var(--ink)',
  lost: 'var(--thread)',
  reacquired: 'var(--fused)',
  warn: 'var(--warn)',
}

export default function EventLog({ log }: { log: LogEntry[] }) {
  if (!log.length) return <div className="px-1 py-2 text-xs text-faint">No events yet.</div>
  return (
    <ul className="space-y-0">
      <AnimatePresence initial={false}>
        {log.slice(0, 14).map((l) => (
          <motion.li
            key={l.id}
            layout
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="flex gap-2 border-b border-hair py-1.5 text-[12px] leading-snug last:border-b-0"
          >
            <span className="num shrink-0 text-[10.5px] text-faint">
              {l.at.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
            <span style={{ color: COLOR[l.kind] }}>{l.message}</span>
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  )
}
