/**
 * MapLibre v6 locates its worker relative to its own module URL, which breaks
 * once Vite pre-bundles / bundles the library. Point it at a worker that Vite
 * builds (with its shared-chunk import inlined) instead.
 */
import { setWorkerUrl } from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'

setWorkerUrl(workerUrl)
