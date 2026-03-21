import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    proxy: {
      // Proxy all /ws websocket connections to FastAPi
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
      // Proxy all REST calls (/count, /history, /reset) to FastAPI
      '/count': { target: 'https://localhost:8000', changeOrigin: true },
      '/history': { target: 'https://localhost:8000', changeOrigin: true },
      '/reset': { target: 'https://localhost:8000', changeOrigin: true },
    }
  }
})
