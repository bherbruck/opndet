import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Built assets are emitted into the Python package so they ship in the wheel
// and FastAPI can mount them statically. `base: "./"` keeps asset URLs relative
// so it works no matter what path the dashboard is mounted under.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  build: {
    outDir: "../src/opndet/dashboard_static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    // `bun run dev` → proxy API + static files to a locally-running
    // `opndet dashboard` (default port 5000).
    proxy: {
      "/api": "http://127.0.0.1:5000",
      "/files": "http://127.0.0.1:5000",
    },
  },
});
