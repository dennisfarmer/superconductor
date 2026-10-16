import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: this server on :8000, the API (uvicorn --reload) on :8001
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8000,
    strictPort: true,
    proxy: {
      "/function": "http://127.0.0.1:8001",
      "/healthz": "http://127.0.0.1:8001",
    },
  },
  build: { outDir: "dist" },
});
