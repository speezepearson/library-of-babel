/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/llamacpp': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/llamacpp/, ''),
      },
    },
  },
  test: {
    environment: 'node',
    testTimeout: 120_000,
  },
})
