import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // An inline config stops Vite searching parent folders, where a sibling app's
  // Tailwind PostCSS config would otherwise process these styles.
  css: { postcss: {} },
})
