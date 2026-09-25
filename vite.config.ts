import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // MapLibre v6 runs its tile worker as an ES module that imports a shared
  // chunk; we bundle it ourselves (see src/lib/maplibreWorker.ts).
  worker: { format: 'es' },
  // MapLibre alone is ~1 MB minified; that's expected for a map app.
  build: { chunkSizeWarningLimit: 1600 },
  server: {
    // Expose on the LAN so a phone on the same Wi-Fi can open the dev server.
    // Note: phones only grant geolocation/motion over HTTPS (see README).
    host: true,
  },
})
