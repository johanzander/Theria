import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built into the backend's static directory; served at /static/plan/ (also behind HA ingress).
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5175,
    proxy: { '/api': { target: process.env.VITE_API_TARGET ?? 'http://localhost:8081' } },
  },
  build: { outDir: '../backend/static/plan', emptyOutDir: true },
  test: { environment: 'node' },
})
