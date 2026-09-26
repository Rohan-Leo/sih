import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import type { Map as MLMap } from 'maplibre-gl'
import { type AppMode, NavEngine, type Snapshot } from './engine/navEngine'
import { motionPermissionRequired } from './engine/liveSource'
import { useTheme } from './hooks/useTheme'
import { useMedia } from './hooks/useMedia'
import { formatDistance, formatDuration } from './lib/geo'
import MapView from './components/MapView'
import SearchBox from './components/SearchBox'
import StatusBadge from './components/StatusBadge'
import TurnBanner from './components/TurnBanner'
import Readouts from './components/Readouts'
import DemoControls, { ScenarioPicker } from './components/DemoControls'
import EventLog from './components/EventLog'
import AboutDrawer from './components/AboutDrawer'
import {
  ClewMark,
  IconChevron,
  IconMinus,
  IconMoon,
  IconNorth,
  IconPlus,
  IconRecenter,
  IconSun,
} from './components/Icons'

const MODE_KEY = 'clew.mode'

function initialMode(): AppMode {
  try {
    const m = new URLSearchParams(location.search).get('mode') ?? localStorage.getItem(MODE_KEY)
    return m === 'demo' ? 'demo' : 'live'
  } catch {
    return 'live'
  }
}

export default function App() {
  const engine = useMemo(() => new NavEngine(), [])
  const snap = useSyncExternalStore(engine.subscribe, engine.getSnapshot)
  const [theme, toggleTheme] = useTheme()
  const desktop = useMedia('(min-width: 900px)')
  const [follow, setFollow] = useState(true)
  const [headingUp, setHeadingUp] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)
  const [sheetOpen, setSheetOpen] = useState(false)
  const [liveWanted, setLiveWanted] = useState(false)
  const mapRef = useRef<MLMap | null>(null)

  useEffect(() => {
    engine.start()
    const m = initialMode()
    if (m === 'demo') engine.setMode('demo')
    // If location permission was granted before, start GNSS straight away.
    navigator.permissions
      ?.query({ name: 'geolocation' as PermissionName })
      .then((p) => {
        if (p.state === 'granted') setLiveWanted(true)
      })
      .catch(() => {})
    return () => engine.destroy()
  }, [engine])

  useEffect(() => {
    if (liveWanted && snap.mode === 'live') engine.startLiveSensors()
  }, [liveWanted, snap.mode, engine])

  const switchMode = (m: AppMode) => {
    engine.setMode(m)
    try {
      localStorage.setItem(MODE_KEY, m)
    } catch {
      /* ignore */
    }
    setFollow(m === 'live')
    setHeadingUp(false)
  }

  // A fresh route should be seen whole, not followed.
  useEffect(() => {
    if (snap.route && !snap.navigating) setFollow(false)
  }, [snap.route, snap.navigating])

  const startNav = async () => {
    // iOS: motion permission must be requested inside this click handler.
    if (motionPermissionRequired() && snap.live.motion !== 'active') await engine.requestMotionPermission()
    setLiveWanted(true)
    engine.startLiveSensors()
    engine.startNavigation()
    setFollow(true)
  }

  const onUserPan = useCallback(() => setFollow(false), [])

  const padding = useMemo(
    () =>
      desktop
        ? { top: 70, bottom: 30, left: 420, right: 70 }
        : { top: snap.navigating ? 190 : 130, bottom: sheetOpen ? 380 : 200, left: 20, right: 20 },
    [desktop, snap.navigating, sheetOpen],
  )

  const dr = snap.est.mode === 'DR'
  const showBanner = snap.navigating && snap.progress && (snap.mode === 'demo' ? snap.demo!.simT > 0 || snap.demo!.playing : true)

  const search =
    snap.mode === 'live' ? (
      <SearchBox
        bias={snap.est.position}
        hasRoute={!!snap.route}
        routing={snap.routing}
        error={snap.routeError}
        onFocus={() => setLiveWanted(true)}
        onPick={(p) => engine.routeTo(p.coord, p.name)}
        onClear={() => engine.clearRoute()}
      />
    ) : (
      <div className="ticks flex h-11 items-center gap-2 rounded-sm border border-hair-strong bg-panel px-3 text-[14px]">
        <span className="label shrink-0">Demo</span>
        <span className="truncate">
          {snap.demo ? (
            <>
              {snap.demo.origin} <span className="text-muted">→</span> {snap.demo.destination}
            </>
          ) : (
            'Loading…'
          )}
        </span>
      </div>
    )

  const banner = (
    <AnimatePresence>
      {showBanner && snap.progress && (
        <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}>
          <TurnBanner progress={snap.progress} dr={dr} />
        </motion.div>
      )}
    </AnimatePresence>
  )

  const mapControls = (
      <div
        className={`absolute right-3 z-10 flex flex-col overflow-hidden rounded-sm border border-hair-strong bg-panel ${
          desktop ? 'top-16' : 'bottom-full mb-3'
        }`}
      >
        <MapBtn label="Zoom in" onClick={() => mapRef.current?.zoomIn()}>
          <IconPlus />
        </MapBtn>
        <MapBtn label="Zoom out" onClick={() => mapRef.current?.zoomOut()}>
          <IconMinus />
        </MapBtn>
        <MapBtn
          label={headingUp ? 'North up' : 'Heading up'}
          active={headingUp}
          onClick={() => {
            const next = !headingUp
            setHeadingUp(next)
            if (!next) mapRef.current?.easeTo({ bearing: 0, duration: 400 })
            else setFollow(true)
          }}
        >
          <IconNorth style={{ transform: headingUp ? 'none' : `rotate(${-(mapRef.current?.getBearing() ?? 0)}deg)` }} />
        </MapBtn>
        <MapBtn
          label="Follow my position"
          active={follow}
          onClick={() => {
            setFollow(true)
            if (snap.mode === 'live') setLiveWanted(true)
            const p = snap.est.position
            if (p) mapRef.current?.easeTo({ center: p, zoom: Math.max(15, mapRef.current.getZoom()), duration: 500 })
          }}
        >
          <IconRecenter />
        </MapBtn>
      </div>
  )

  const consoleBody = <ConsoleBody snap={snap} engine={engine} onStartNav={startNav} onLocate={() => setLiveWanted(true)} compact={!desktop && !sheetOpen} />

  return (
    <div className="relative h-full w-full overflow-hidden bg-paper">
      <MapView
        engine={engine}
        snap={snap}
        theme={theme}
        follow={follow}
        headingUp={headingUp}
        onUserPan={onUserPan}
        padding={padding}
        onMap={(m) => (mapRef.current = m)}
      />

      {/* ── top bar ── */}
      <header className="absolute inset-x-0 top-0 z-20 flex h-12 items-center gap-3 border-b border-hair-strong bg-paper/95 px-3 sm:px-4">
        <div className="flex items-center gap-2">
          <ClewMark />
          <span className="font-display text-[22px] leading-none tracking-tight">Clew</span>
          <span className="num hidden text-[10px] tracking-[0.14em] text-muted sm:inline">PS 26168 · DEAD-RECKONING NAVIGATOR</span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {desktop && <StatusBadge est={snap.est} />}
          <div role="radiogroup" aria-label="Position source" className="flex rounded-sm border border-hair-strong p-0.5">
            {(['live', 'demo'] as AppMode[]).map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={snap.mode === m}
                onClick={() => switchMode(m)}
                className={`rounded-[2px] px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.12em] ${
                  snap.mode === m ? 'bg-ink text-paper' : 'text-muted hover:text-ink'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className="flex h-8 w-8 items-center justify-center rounded-sm border border-hair-strong text-ink"
          >
            {theme === 'dark' ? <IconSun /> : <IconMoon />}
          </button>
          <button
            type="button"
            onClick={() => setAboutOpen(true)}
            className="h-8 rounded-sm border border-hair-strong px-2.5 font-mono text-[11px] uppercase tracking-[0.12em] text-ink"
          >
            About
          </button>
        </div>
      </header>

      {desktop ? (
        <aside className="absolute bottom-3 left-3 top-[60px] z-10 flex w-[392px] flex-col gap-2">
          {search}
          {banner}
          <div className="blueprint ticks scrollbar-thin min-h-0 flex-1 overflow-y-auto rounded-sm border border-hair-strong">
            {consoleBody}
          </div>
        </aside>
      ) : null}
      {desktop && mapControls}
      {!desktop && (
        <>
          <div className="absolute inset-x-2 top-14 z-10 space-y-2">
            {search}
            {banner}
            <div className="flex">
              <StatusBadge est={snap.est} />
            </div>
          </div>
          <motion.div
            className="blueprint absolute inset-x-0 bottom-0 z-10 flex max-h-[55vh] flex-col rounded-t-[4px] border-t border-hair-strong"
            layout
            transition={{ duration: 0.2 }}
          >
            {!sheetOpen && mapControls}
            <button
              type="button"
              onClick={() => setSheetOpen((o) => !o)}
              className="flex w-full items-center justify-center gap-2 py-1.5 text-muted"
              aria-label={sheetOpen ? 'Collapse panel' : 'Expand panel'}
            >
              <span className="h-[3px] w-10 rounded-[1px] bg-hair-strong" />
              <IconChevron className={`transition-transform ${sheetOpen ? '' : 'rotate-180'}`} width={12} height={12} />
            </button>
            <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto pb-[env(safe-area-inset-bottom)]">{consoleBody}</div>
          </motion.div>
        </>
      )}

      <AboutDrawer open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  )
}

function MapBtn({
  label,
  onClick,
  active,
  children,
}: {
  label: string
  onClick: () => void
  active?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={active}
      onClick={onClick}
      className={`flex h-10 w-10 items-center justify-center border-b border-hair last:border-b-0 ${
        active ? 'text-fused' : 'text-ink'
      }`}
    >
      {children}
    </button>
  )
}

function Section({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="border-t border-hair px-3 py-3 first:border-t-0">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="label">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  )
}

function ConsoleBody({
  snap,
  engine,
  onStartNav,
  onLocate,
  compact,
}: {
  snap: Snapshot
  engine: NavEngine
  onStartNav: () => void
  onLocate: () => void
  compact: boolean
}) {
  const [showSteps, setShowSteps] = useState(false)
  const r = snap.route
  const p = snap.progress
  const live = snap.live

  const sensorNotes: React.ReactNode[] = []
  if (snap.mode === 'live') {
    if (live.geo === 'idle')
      sensorNotes.push(
        <button key="loc" type="button" onClick={onLocate} className="h-9 w-full rounded-sm border border-fused text-sm font-medium text-fused">
          Use my location
        </button>,
      )
    if (live.geo === 'waiting') sensorNotes.push(<Note key="w">Waiting for a GNSS fix…</Note>)
    if (live.geoMessage) sensorNotes.push(<Note key="g" tone="warn">{live.geoMessage}</Note>)
    if (live.motion === 'needs-permission')
      sensorNotes.push(
        <button
          key="m"
          type="button"
          onClick={() => engine.requestMotionPermission()}
          className="h-9 w-full rounded-sm border border-hair-strong text-sm"
        >
          Enable motion sensors
        </button>,
      )
    if (live.motion === 'none')
      sensorNotes.push(
        <Note key="n">No motion sensors on this device — dead reckoning will hold the last speed and course.</Note>,
      )
  }

  return (
    <div>
      {snap.mode === 'demo' && (
        <Section title="Judge demo">
          <div className="space-y-3">
            {!compact && <ScenarioPicker scenario={snap.demoScenario} engine={engine} />}
            {snap.demo ? (
              <DemoControls demo={snap.demo} engine={engine} compact={compact} />
            ) : (
              <p className="text-[13px] text-muted">{snap.demoLoading ? 'Loading recorded drive…' : 'Demo unavailable.'}</p>
            )}
          </div>
        </Section>
      )}

      {snap.mode === 'live' && (
        <Section title={r ? 'Route' : 'Navigate'}>
          {!r && !snap.routing && (
            <p className="text-[13px] leading-relaxed text-muted">
              Search for a destination above. Clew follows GNSS while it's healthy and switches to on-device dead
              reckoning when the fix goes stale, degrades or drops.
            </p>
          )}
          {r && (
            <div>
              <div className="font-display text-lg leading-tight">{r.route.destinationName}</div>
              <div className="num mt-1 flex gap-4 text-sm">
                {p && snap.navigating ? (
                  <>
                    <span>{formatDistance(p.remaining)}</span>
                    <span>{formatDuration(p.remainingTime)}</span>
                    <span className="text-muted">
                      ETA {p.etaClock.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </>
                ) : (
                  <>
                    <span>{formatDistance(r.route.distance)}</span>
                    <span>{formatDuration(r.route.duration)}</span>
                    <span className="text-muted">driving</span>
                  </>
                )}
              </div>
              <div className="mt-3 flex gap-2">
                {snap.navigating ? (
                  <button type="button" onClick={() => engine.stopNavigation()} className="h-10 flex-1 rounded-sm border border-ink text-sm font-medium">
                    End navigation
                  </button>
                ) : (
                  <button type="button" onClick={onStartNav} className="h-10 flex-1 rounded-sm bg-fused text-sm font-medium text-paper">
                    Start navigation
                  </button>
                )}
              </div>
            </div>
          )}
          {sensorNotes.length > 0 && <div className="mt-3 space-y-2">{sensorNotes}</div>}
        </Section>
      )}

      {snap.mode === 'demo' && p && !compact && (
        <Section title="Route">
          <div className="num flex gap-4 text-sm">
            <span>{formatDistance(p.remaining)} left</span>
            <span>{formatDuration(p.remainingTime)}</span>
          </div>
        </Section>
      )}

      {!compact && (
        <>
          <Section title="Position fix">
            <Readouts snap={snap} />
          </Section>

          {r && (
            <Section
              title={`Directions · ${r.steps.length} steps`}
              right={
                <button type="button" onClick={() => setShowSteps((s) => !s)} className="label hover:text-ink">
                  {showSteps ? 'Hide' : 'Show'}
                </button>
              }
            >
              {showSteps && (
                <ol className="text-[13px]">
                  {r.steps.map((s, i) => (
                    <li
                      key={i}
                      className={`flex gap-3 border-b border-hair py-1.5 last:border-b-0 ${
                        p && i < p.nextStepIdx ? 'text-faint' : p && i === p.nextStepIdx ? 'text-ink' : 'text-muted'
                      }`}
                    >
                      <span className="num w-5 shrink-0 text-[11px] text-faint">{String(i + 1).padStart(2, '0')}</span>
                      <span className="flex-1">{s.instruction}</span>
                      <span className="num shrink-0 text-[11px]">{s.distance > 0 ? formatDistance(s.distance) : ''}</span>
                    </li>
                  ))}
                </ol>
              )}
            </Section>
          )}

          <Section title="Event log">
            <EventLog log={snap.log} />
          </Section>
        </>
      )}
    </div>
  )
}

function Note({ children, tone }: { children: React.ReactNode; tone?: 'warn' }) {
  return (
    <div className={`border-l-2 pl-2 text-[12px] leading-snug ${tone === 'warn' ? 'border-warn text-warn' : 'border-hair-strong text-muted'}`}>
      {children}
    </div>
  )
}
