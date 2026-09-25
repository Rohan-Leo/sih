import { AnimatePresence, motion } from 'framer-motion'
import { ClewMark, IconClose } from './Icons'

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-hair pt-4">
      <h3 className="label mb-2">{title}</h3>
      <div className="space-y-2 text-[13.5px] leading-relaxed text-ink">{children}</div>
    </section>
  )
}

export default function AboutDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-ink/25"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            role="dialog"
            aria-label="About Clew"
            className="blueprint scrollbar-thin fixed inset-y-0 right-0 z-50 w-full max-w-[440px] overflow-y-auto border-l border-hair-strong"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'tween', duration: 0.25, ease: [0.2, 0.7, 0.2, 1] }}
          >
            <div className="space-y-5 p-5 pb-10">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2.5">
                  <ClewMark width={30} height={30} />
                  <div>
                    <div className="font-display text-2xl leading-none">Clew</div>
                    <div className="mt-1 text-xs text-muted">a thread through the labyrinth</div>
                  </div>
                </div>
                <button type="button" onClick={onClose} aria-label="Close" className="p-1.5 text-muted hover:text-ink">
                  <IconClose />
                </button>
              </div>

              <p className="font-display text-[17px] leading-snug">
                Theseus found his way out of the Labyrinth by unwinding a ball of thread. Clew does the same through
                tunnels, basements and urban canyons: when satellites go dark, it keeps paying out the thread from what
                the phone itself can feel.
              </p>

              <Section title="Submission">
                <dl className="grid grid-cols-[110px_1fr] gap-y-1.5 text-[13px]">
                  <dt className="text-muted">Event</dt>
                  <dd>Smart India Hackathon 2026</dd>
                  <dt className="text-muted">PS ID</dt>
                  <dd className="num">26168</dd>
                  <dt className="text-muted">Title</dt>
                  <dd>AI-ML based Intelligent Dead Reckoning system for seamless navigation</dd>
                  <dt className="text-muted">Theme</dt>
                  <dd>Smart Vehicles</dd>
                  <dt className="text-muted">Team</dt>
                  <dd>IcarusSipsTea</dd>
                  <dt className="text-muted">Product</dt>
                  <dd>Clew</dd>
                </dl>
              </Section>

              <Section title="The full pipeline (our submission)">
                <ol className="list-decimal space-y-1.5 pl-5">
                  <li>
                    <strong className="font-medium">Learned IMU speed model</strong> — a network trained in PyTorch on
                    accelerometer + gyroscope windows, exported to TFLite for on-device inference, regressing forward
                    speed without wheel odometry.
                  </li>
                  <li>
                    <strong className="font-medium">UKF with non-holonomic constraints</strong> — an Unscented Kalman
                    Filter fusing GNSS, learned speed and gyro yaw, with the constraint that a road vehicle does not slip
                    sideways or move vertically.
                  </li>
                  <li>
                    <strong className="font-medium">HMM road-network map matching</strong> — a Hidden Markov Model over
                    candidate road segments (Viterbi decoding) to keep the estimate on physically reachable roads.
                  </li>
                </ol>
              </Section>

              <Section title="What this app actually runs">
                <p>
                  The in-app fallback is a <strong className="font-medium">simplified, real-time stand-in</strong> for
                  that pipeline — a lightweight, demo-grade heuristic, not production-grade dead reckoning. It does
                  run for real on every frame, in both Live and Demo mode:
                </p>
                <ul className="list-disc space-y-1 pl-5">
                  <li>GNSS is declared lost after 3 s without a fix, on geolocation errors, or when reported accuracy jumps well past its recent baseline.</li>
                  <li>Speed is the last GNSS speed, decaying slowly while the accelerometer detects motion and quickly when it reads still; step cadence is used at walking speeds.</li>
                  <li>Heading comes from the compass (offset-calibrated against the last GNSS course) or integrated gyro yaw.</li>
                  <li>Instead of HMM matching, the estimate is constrained to the active route polyline.</li>
                  <li>On reacquisition the dot eases back onto the GNSS track and the drift is logged.</li>
                </ul>
                <p className="text-muted">
                  Expect drift of metres to tens of metres over a short outage, growing with time — that is what the
                  widening halo shows. The estimator lives in one module
                  (<span className="num text-[12px]">src/engine/positionEstimator.ts</span>) so the trained model can be
                  dropped in without touching the map or UI.
                </p>
              </Section>

              <Section title="Reading the map">
                <ul className="space-y-1.5">
                  <li className="flex items-center gap-3">
                    <svg width="36" height="6"><path d="M0 3h36" stroke="var(--thread)" strokeWidth="3" /></svg>
                    Thread — path travelled on GNSS
                  </li>
                  <li className="flex items-center gap-3">
                    <svg width="36" height="6"><path d="M0 3h36" stroke="var(--thread)" strokeWidth="3" strokeDasharray="4 3" /></svg>
                    Thread — dead-reckoned stretch
                  </li>
                  <li className="flex items-center gap-3">
                    <svg width="36" height="12"><circle cx="18" cy="6" r="5" fill="var(--fused-soft)" stroke="var(--fused)" strokeDasharray="2 2" /></svg>
                    Uncertainty halo (1-σ), grows during dead reckoning
                  </li>
                  <li className="flex items-center gap-3">
                    <svg width="36" height="10"><path d="M0 5h36" stroke="var(--ink)" strokeOpacity="0.55" strokeWidth="6" strokeDasharray="3 3" /></svg>
                    Demo dead zone (GNSS withheld)
                  </li>
                  <li className="flex items-center gap-3">
                    <svg width="36" height="6"><path d="M0 3h36" stroke="var(--ink)" strokeOpacity="0.5" strokeDasharray="1 3" /></svg>
                    Ground truth (demo only)
                  </li>
                </ul>
              </Section>

              <Section title="Permissions">
                <p>
                  Location (GNSS) and, on iPhone, Motion &amp; Orientation access. Both need HTTPS. On a laptop there are
                  usually no motion sensors — dead reckoning then holds the last speed and course.
                </p>
              </Section>

              <Section title="Data & services">
                <p className="text-muted">
                  Map data © OpenStreetMap contributors · tiles by OpenFreeMap · search by Photon (komoot) · routing by
                  the OSRM demo server. All keyless; please use fairly.
                </p>
              </Section>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
