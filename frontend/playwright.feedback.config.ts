import { defineConfig } from "@playwright/test";
import { randomUUID } from "node:crypto";
import base from "./playwright.search.config";

const servers = Array.isArray(base.webServer) ? base.webServer : [];
export default defineConfig({
  ...base,
  testMatch: "feedback.spec.ts",
  webServer: [{ ...servers[0], env: { ...servers[0].env,
    TRUSTWORTHY_RECSYS_FEEDBACK_MODE: "randomized",
    TRUSTWORTHY_RECSYS_FEEDBACK_TRAFFIC: "synthetic",
    TRUSTWORTHY_RECSYS_FEEDBACK_DB: `data/feedback/browser-${randomUUID()}.sqlite3`
  } }, servers[1]]
});
