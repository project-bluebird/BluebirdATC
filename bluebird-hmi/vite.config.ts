/// <reference types="node" />

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ command}) => ({
  plugins: [react()],
  base: command === "build" ? "/hmi/" : "/",
  resolve: {
    alias: {
      api: path.resolve(__dirname, "src/api"),
      app: path.resolve(__dirname, "src/app"),
      assets: path.resolve(__dirname, "src/assets"),
      components: path.resolve(__dirname, "src/components"),
      data: path.resolve(__dirname, "src/data"),
      pages: path.resolve(__dirname, "src/pages"),
      slices: path.resolve(__dirname, "src/slices"),
      utils: path.resolve(__dirname, "src/utils"),
    }
  },
  build: {
    outDir: "../bluebird-api/bluebird_api/hmi",
    emptyOutDir: true,
  }
}));
