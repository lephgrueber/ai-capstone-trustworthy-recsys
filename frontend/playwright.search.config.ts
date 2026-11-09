import { defineConfig, devices } from "@playwright/test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const python = resolve(here, "..", process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");

export default defineConfig({
  testDir: "./e2e", testMatch: "search.spec.ts", timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:15173", screenshot: "only-on-failure", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], channel: process.env.PLAYWRIGHT_CHANNEL || undefined } }],
  webServer: [
    { command: `"${python}" -m uvicorn trustworthy_recsys.serving.app:app --host 127.0.0.1 --port 18000`,
      cwd: resolve(here, ".."), url: "http://127.0.0.1:18000/ready", reuseExistingServer: false,
      env: { TRUSTWORTHY_RECSYS_BACKEND: "fixture" }, timeout: 60_000 },
    { command: "npm run dev -- --host 127.0.0.1 --port 15173 --strictPort", cwd: here,
      env: { RECSYS_API_TARGET: "http://127.0.0.1:18000" },
      url: "http://127.0.0.1:15173", reuseExistingServer: false, timeout: 60_000 }
  ]
});
