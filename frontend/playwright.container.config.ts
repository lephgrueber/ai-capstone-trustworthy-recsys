import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.CONTAINER_APP_URL ?? "http://127.0.0.1:15173";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "container-artwork",
      testMatch: "release.learned.spec.ts",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "container-no-artwork",
      testMatch: "release.no-artwork.spec.ts",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
