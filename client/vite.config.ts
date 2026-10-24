import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: this server on :8764, the API (uvicorn --reload) on :8763 (Makefile DEV_API_PORT)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8764,
    strictPort: true,
    proxy: {
      "/function": "http://127.0.0.1:8763",
      "/healthz": "http://127.0.0.1:8763",
    },
  },
  build: { outDir: "dist" },
});
