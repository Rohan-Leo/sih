import type { DemoInfo, DemoScenario, NavEngine } from '../engine/navEngine'
import { IconPause, IconPlay, IconReset } from './Icons'

const RATES = [0.5, 1, 2, 3, 4]

function mmss(ms: number) {
  const s = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export function ScenarioPicker({ scenario, engine }: { scenario: DemoScenario; engine: NavEngine }) {
  const opts: { id: DemoScenario; title: string; sub: string }[] = [
    { id: 'recorded', title: 'Recorded drive', sub: 'real phone IMU · learned engine' },
    { id: 'delhi', title: 'New Delhi route', sub: 'synthetic sensors · heuristic' },
  ]
  return (
    <div role="radiogroup" aria-label="Demo scenario" className="grid grid-cols-2 gap-1.5">
      {opts.map((o) => (
        <button
          key={o.id}
          type="button"
          role="radio"
          aria-checked={scenario === o.id}
          onClick={() => engine.setScenario(o.id)}
          className={`rounded-sm border px-2.5 py-1.5 text-left ${scenario === o.id ? 'border-ink bg-panel-2' : 'border-hair text-muted hover:border-hair-strong'}`}
        >
          <span className="block text-[13px] text-ink">{o.title}</span>
          <span className="block text-[10.5px] text-muted">{o.sub}</span>
        </button>
      ))}
    </div>
  )
}

export default function DemoControls({ demo, engine, compact = false }: { demo: DemoInfo; engine: NavEngine; compact?: boolean }) {
  const frac = Math.min(1, demo.truthAlong / Math.max(1, demo.pathLength))
  const manual = demo.manualRemaining > 0
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => engine.setPlaying(!demo.playing)}
          className="flex h-10 items-center gap-2 rounded-sm bg-ink px-4 text-sm font-medium text-paper"
        >
          {demo.playing ? <IconPause /> : <IconPlay />}
          {demo.playing ? 'Pause' : demo.simT > 0 && demo.simT < demo.duration ? 'Resume' : 'Play route'}
        </button>
        <button
          type="button"
          aria-label="Restart demo"
          title="Restart demo"
          onClick={() => engine.resetDemo()}
          className="flex h-10 w-10 items-center justify-center rounded-sm border border-hair-strong text-ink"
        >
          <IconReset />
        </button>
        <div className="num ml-auto text-right text-[12px] text-muted">
          {mmss(demo.simT)} <span className="text-faint">/ {mmss(demo.duration)}</span>
        </div>
      </div>

      {/* route progress strip with dead zones */}
      <div>
        <div className="relative h-3 rounded-[1px] border border-hair-strong">
          {demo.deadZones.map((z) => (
            <div
              key={z.label}
              title={z.label}
              className="absolute inset-y-0"
              style={{
                left: `${(z.from / demo.pathLength) * 100}%`,
                width: `${((z.to - z.from) / demo.pathLength) * 100}%`,
                backgroundImage:
                  z.kind === 'fault'
                    ? 'repeating-linear-gradient(45deg, var(--warn) 0 2px, transparent 2px 4px)'
                    : 'repeating-linear-gradient(135deg, var(--ink) 0 1px, transparent 1px 4px)',
                opacity: z.kind === 'fault' ? 0.7 : 0.45,
              }}
            />
          ))}
          <div className="absolute inset-y-0 left-0 bg-thread/70" style={{ width: `${frac * 100}%` }} />
          <div className="absolute -inset-y-1 w-[2px] bg-ink" style={{ left: `calc(${frac * 100}% - 1px)` }} />
        </div>
        <div className="mt-1 flex justify-between text-[10px] text-muted">
          <span className="truncate">{demo.origin}</span>
          <span className="truncate">{demo.destination}</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <span className="label shrink-0">Speed</span>
        <input
          type="range"
          min={0}
          max={RATES.length - 1}
          step={1}
          value={Math.max(0, RATES.indexOf(demo.rate))}
          onChange={(e) => engine.setRate(RATES[Number(e.target.value)])}
          className="min-w-0 flex-1"
          aria-label="Playback speed"
        />
        <span className="num w-10 text-right text-sm">{demo.rate}×</span>
      </div>

      {demo.calibrationEnd !== null && demo.simT < demo.calibrationEnd && (
        <button
          type="button"
          onClick={() => engine.skipTo(demo.calibrationEnd! + 1000)}
          className="flex h-9 w-full items-center justify-center rounded-sm border border-hair-strong text-[13px] text-ink"
        >
          Skip mount calibration → {mmss(demo.calibrationEnd + 1000)}
        </button>
      )}

      <button
        type="button"
        onClick={() => engine.toggleSimulatedOutage()}
        disabled={demo.simT >= demo.duration}
        className={`flex h-10 w-full items-center justify-center gap-2 rounded-sm border text-sm font-medium transition-colors ${
          manual ? 'border-thread bg-thread text-paper' : 'border-thread text-thread hover:bg-thread-soft'
        }`}
      >
        {manual ? `Restore GNSS  ·  ${(demo.manualRemaining / 1000).toFixed(0)} s left` : 'Simulate GNSS loss now'}
      </button>

      <div className="text-[11px] leading-relaxed text-muted">
        {demo.zone ? (
          <span style={{ color: 'var(--thread)' }}>{demo.zone.label} — GNSS feed withheld.</span>
        ) : demo.withheld ? (
          <span style={{ color: 'var(--thread)' }}>GNSS feed withheld by presenter.</span>
        ) : compact ? null : (
          demo.scenario === 'recorded' ? (
            <>A real drive the model never saw in training, replayed from the phone's own sensors. GNSS is withheld in hatched segments; the dashed line is where the car really went.</>
          ) : (
            <>Scripted drive, real pipeline: fixes are withheld in hatched segments and the same estimator used in Live mode must notice and dead-reckon.</>
          )
        )}
        {demo.source === 'hand-traced' && !compact && (
          <span className="mt-1 block text-faint">Route geometry: offline hand-traced fallback. Run <span className="num">npm run bake:demo</span> with internet to bake the OSRM route.</span>
        )}
      </div>
    </div>
  )
}
