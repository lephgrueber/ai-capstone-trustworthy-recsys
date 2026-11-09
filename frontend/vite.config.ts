import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const apiTarget = process.env.RECSYS_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": apiTarget, "/health": apiTarget, "/ready": apiTarget }
  },
  test: { environment: "jsdom", setupFiles: "./src/test/setup.ts", include: ["src/**/*.test.{ts,tsx}"], css: true }
});
