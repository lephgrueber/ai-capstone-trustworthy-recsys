import { defineConfig, devices } from "@playwright/test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure"
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "TRUSTWORTHY_RECSYS_BACKEND=baseline TRUSTWORTHY_RECSYS_RUN_DIR=data/processed/m3_ratio_80_10_10_20261002 TRUSTWORTHY_RECSYS_SCENARIO=ratio_80_10_10 .venv/bin/python -m uvicorn trustworthy_recsys.serving.app:app --host 127.0.0.1 --port 8000",
      cwd: resolve(here, ".."),
      url: "http://127.0.0.1:8000/ready",
      reuseExistingServer: true,
      timeout: 30_000
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5173",
      cwd: here,
      url: "http://127.0.0.1:5173",
      reuseExistingServer: true,
      timeout: 30_000
    }
  ]
});
