import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { type Place, searchPlaces } from '../services/photon'
import type { LngLat } from '../lib/geo'
import { formatDistance, haversine } from '../lib/geo'
import { IconClose, IconSearch } from './Icons'

interface Props {
  bias: LngLat | null
  onPick: (p: Place) => void
  onClear: () => void
  onFocus: () => void
  hasRoute: boolean
  routing: boolean
  error: string | null
}

export default function SearchBox({ bias, onPick, onClear, onFocus, hasRoute, routing, error }: Props) {
  const [q, setQ] = useState('')
  const [results, setResults] = useState<Place[]>([])
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [searchErr, setSearchErr] = useState<string | null>(null)
  const [active, setActive] = useState(0)
  const biasRef = useRef(bias)
  biasRef.current = bias
  const picked = useRef(false)

  useEffect(() => {
    if (picked.current) {
      picked.current = false
      return
    }
    const query = q.trim()
    if (query.length < 2) {
      setResults([])
      setSearchErr(null)
      return
    }
    const ctrl = new AbortController()
    const timer = setTimeout(async () => {
      setBusy(true)
      try {
        const r = await searchPlaces(query, biasRef.current, ctrl.signal)
        setResults(r)
        setActive(0)
        setSearchErr(r.length ? null : 'No matches')
        setOpen(true)
      } catch (e) {
        if ((e as Error).name !== 'AbortError') setSearchErr('Search unavailable — check your connection')
      } finally {
        setBusy(false)
      }
    }, 280)
    return () => {
      clearTimeout(timer)
      ctrl.abort()
    }
  }, [q])

  const pick = (p: Place) => {
    picked.current = true
    setQ(p.name)
    setOpen(false)
    setResults([])
    onPick(p)
  }

  return (
    <div className="relative">
      <div className="ticks flex items-center gap-2 rounded-sm border border-hair-strong bg-panel px-3">
        <IconSearch className="shrink-0 text-muted" />
        <input
          value={q}
          onChange={(e) => {
            setQ(e.target.value)
            setOpen(true)
          }}
          onFocus={() => {
            onFocus()
            if (results.length) setOpen(true)
          }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') setActive((a) => Math.min(results.length - 1, a + 1))
            else if (e.key === 'ArrowUp') setActive((a) => Math.max(0, a - 1))
            else if (e.key === 'Enter' && results[active]) pick(results[active])
            else if (e.key === 'Escape') setOpen(false)
          }}
          placeholder="Search a destination"
          aria-label="Search a destination"
          className="h-11 min-w-0 flex-1 bg-transparent text-[15px] text-ink outline-none placeholder:text-faint"
          autoComplete="off"
          spellCheck={false}
        />
        {(busy || routing) && <span className="num text-[10px] tracking-widest text-muted">{routing ? 'ROUTING' : '···'}</span>}
        {(q || hasRoute) && (
          <button
            type="button"
            aria-label="Clear"
            className="-mr-1 p-1.5 text-muted hover:text-ink"
            onClick={() => {
              setQ('')
              setResults([])
              onClear()
            }}
          >
            <IconClose />
          </button>
        )}
      </div>
      {error && <div className="mt-1 rounded-sm border border-thread/50 bg-panel px-3 py-2 text-xs text-thread">{error}</div>}
      <AnimatePresence>
        {open && (results.length > 0 || searchErr) && q.trim().length >= 2 && (
          <motion.ul
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            className="scrollbar-thin absolute inset-x-0 top-full z-30 mt-1 max-h-[50vh] overflow-auto rounded-sm border border-hair-strong bg-panel"
            role="listbox"
          >
            {searchErr && !results.length && <li className="px-3 py-3 text-sm text-muted">{searchErr}</li>}
            {results.map((r, i) => (
              <li key={r.id + i} role="option" aria-selected={i === active}>
                <button
                  type="button"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => pick(r)}
                  onMouseEnter={() => setActive(i)}
                  className={`flex w-full items-baseline gap-3 border-b border-hair px-3 py-2.5 text-left last:border-b-0 ${i === active ? 'bg-panel-2' : ''}`}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[14px] text-ink">{r.name}</span>
                    <span className="block truncate text-xs text-muted">{r.detail || r.kind}</span>
                  </span>
                  {bias && <span className="num shrink-0 text-[11px] text-faint">{formatDistance(haversine(bias, r.coord))}</span>}
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
