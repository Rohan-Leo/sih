/** Load Clew's five models from public/ml/ (exported by `python -m clew_ml.export_web`). */
import { DriftNet } from './drift'
import { HeadingNet, MotionNet } from './imuModels'
import { IntegrityNet } from './integrity'
import type { ModelSet } from './learnedEstimator'
import { SpeedNet } from './speednet'
import { TinyNet } from './tinynet'

export const MODEL_FILES = ['speednet', 'headingnet', 'motionnet', 'integritynet', 'driftnet'] as const

export function modelsFrom(nets: Partial<Record<(typeof MODEL_FILES)[number], TinyNet>>): ModelSet {
  if (!nets.speednet) throw new Error('SpeedNet is required')
  return {
    speed: SpeedNet.fromNet(nets.speednet),
    heading: nets.headingnet ? new HeadingNet(nets.headingnet) : null,
    motion: nets.motionnet ? new MotionNet(nets.motionnet) : null,
    integrity: nets.integritynet ? new IntegrityNet(nets.integritynet) : null,
    drift: nets.driftnet ? new DriftNet(nets.driftnet) : null,
  }
}

/** SpeedNet must load; the other four are optional, so a missing file degrades gracefully. */
export async function loadModels(base = '/ml/'): Promise<{ models: ModelSet; missing: string[] }> {
  const results = await Promise.allSettled(MODEL_FILES.map((n) => TinyNet.load(base, n)))
  const nets: Partial<Record<(typeof MODEL_FILES)[number], TinyNet>> = {}
  const missing: string[] = []
  results.forEach((r, i) => {
    if (r.status === 'fulfilled') nets[MODEL_FILES[i]] = r.value
    else missing.push(MODEL_FILES[i])
  })
  if (!nets.speednet) throw new Error('SpeedNet failed to load')
  return { models: modelsFrom(nets), missing }
}
