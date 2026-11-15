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
  projects: [
    {
      name: "release-artwork",
      testMatch: "release.learned.spec.ts",
      use: { ...devices["Desktop Chrome"] }
    },
    {
      name: "release-no-artwork",
      testMatch: "release.no-artwork.spec.ts",
      use: { ...devices["Desktop Chrome"] }
    }
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5173",
    cwd: resolve(here),
    url: "http://127.0.0.1:5173",
    reuseExistingServer: true,
    timeout: 30_000
  }
});

