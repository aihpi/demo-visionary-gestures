import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/beatsheldon/',
  plugins: [tailwindcss(), react()],
  build: {
    // Copy public directory contents to dist root
    copyPublicDir: true,
    rollupOptions: {
      output: {
        assetFileNames: (assetInfo) => {
          // Keep images organized
          if (assetInfo.name.match(/\.(png|jpe?g|svg|gif|webp)$/)) {
            return 'images/[name].[hash][extname]';
          }
          // Keep model files with original names (no hash)
          if (assetInfo.name.match(/\.(bin|json)$/) && assetInfo.name.includes('group1-shard')) {
            return 'model/[name][extname]';
          }
          return 'assets/[name].[hash][extname]';
        }
      }
    },
    // Ensure binary files are not inlined
    assetsInlineLimit: (filePath, content) => {
      // Never inline .bin files
      if (filePath.endsWith('.bin')) {
        return false;
      }
      return 4096; // Default threshold
    }
  },
  // Explicitly exclude .bin files from being processed as assets
  assetsInclude: ['**/*.json', '**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.gif', '**/*.svg', '**/*.webp'],
})