import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const uniqueRun = `${new Date().toISOString().replace(/[:.]/g, "-")}-${process.pid}`;
const evidenceDir = resolve(
  process.cwd(),
  process.env.M4_RELEASE_EVIDENCE_DIR ?? `../tmp/m4_release_evidence/${uniqueRun}`
);
const evidenceFile = resolve(evidenceDir, "browser-flow-artwork.json");

if (existsSync(evidenceFile)) {
  throw new Error(`Refusing to replace existing browser evidence: ${evidenceFile}`);
}

async function chooseGenre(page: Page, kind: "Include" | "Avoid", value: string) {
  const drawer = page.getByRole("dialog", { name: "Personalize your recommendations" });
  await drawer.getByRole("button", { name: new RegExp(`${kind} genres`) }).click();
  const selector = page.getByRole("dialog", { name: `${kind} genres` });
  await selector.getByRole("button", { name: value }).click();
  await selector.getByRole("button", { name: "Done" }).click();
}

test("real learned flow uses LambdaRank, filters, search, hides and live TMDb artwork", async ({ page, request }) => {
  const statusResponse = await request.get("/api/v1/application/status");
  expect(statusResponse.ok(), "release-artwork requires a ready application status endpoint").toBe(true);
  const status = await statusResponse.json();
  expect(status.backend_mode, "release-artwork requires the learned backend").toBe("learned");
  expect(status.posters_configured, "release-artwork requires backend TMDb artwork configuration").toBe(true);

  const initialResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await page.goto("/");
  const initial = await (await initialResponse).json();
  expect(initial.backend_mode).toBe("learned");
  expect(initial.retrieval.name).toBe("two-tower-weaviate");
  expect(initial.reranker.name).toBe("lambdarank");
  await expect(page.locator(".feature-hero h1")).toBeVisible();
  await expect(page.locator('img[src^="https://image.tmdb.org/"]').first()).toBeVisible({ timeout: 15_000 });

  await page.locator(".portrait-card").first().getByRole("button", { name: "Hide this movie" }).click();
  await page.getByRole("button", { name: /Personalize/ }).click();
  const drawer = page.getByRole("dialog", { name: "Personalize your recommendations" });
  const favorites = drawer.getByLabel("Movies you like");
  await favorites.fill("Toy Story (1995)");
  await drawer.getByRole("button", { name: /Toy Story \(1995\).*Add/ }).click();
  await chooseGenre(page, "Include", "Adventure");
  await chooseGenre(page, "Avoid", "Horror");

  const submittedRequest = page.waitForRequest((request) => request.url().endsWith("/recommendations") && request.method() === "POST");
  const learnedResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await drawer.getByRole("button", { name: "Update recommendations" }).click();
  const submitted = (await submittedRequest).postDataJSON();
  const learned = await (await learnedResponse).json();
  expect(submitted.preferences.history_items).toContain("1");
  expect(submitted.preferences.excluded_movie_ids.length).toBeGreaterThan(0);
  const hiddenMovieId = submitted.preferences.excluded_movie_ids[0];
  expect(submitted.preferences.include_genres).toContain("Adventure");
  expect(submitted.preferences.exclude_genres).toContain("Horror");
  expect(learned.retrieval.mode).toBe("learned");
  expect(learned.reranker.name).toBe("lambdarank");
  expect(learned.warnings).not.toContain("Popularity fallback used because no supported history items were supplied.");
  expect(learned.recommendations.every((item: { movie: { genres: string[]; movie_id: string } }) =>
    item.movie.genres.includes("Adventure") &&
    !item.movie.genres.includes("Horror") &&
    item.movie.movie_id !== hiddenMovieId)).toBe(true);
  await expect(page.getByRole("heading", { name: "Your recommendations", level: 2 })).toBeVisible();

  const search = page.getByRole("search", { name: "Search movie recommendations" });
  const browserStarted = Date.now();
  const searchResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await search.getByRole("textbox").fill("show comedy without thriller");
  await search.getByRole("textbox").press("Enter");
  const searched = await (await searchResponse).json();
  await expect(page.getByText("Including Comedy; Avoiding Thriller.")).toBeVisible();
  const browserCompleted = Date.now();
  expect(searched.retrieval.mode).toBe("learned");
  expect(searched.reranker.name).toBe("lambdarank");
  expect(searched.applied_preferences.include_genres).toEqual(expect.arrayContaining(["Adventure", "Comedy"]));
  expect(searched.applied_preferences.exclude_genres).toEqual(expect.arrayContaining(["Horror", "Thriller"]));
  const included = new Set<string>(searched.applied_preferences.include_genres);
  const excluded = new Set<string>(searched.applied_preferences.exclude_genres);
  expect(searched.recommendations.every((item: { movie: { genres: string[] } }) =>
    item.movie.genres.some((genre) => included.has(genre)) &&
    item.movie.genres.every((genre) => !excluded.has(genre)))).toBe(true);

  const artworkResources = await page.evaluate(() => performance.getEntriesByType("resource")
    .filter((entry) => entry.name.startsWith("https://image.tmdb.org/"))
    .map((entry) => ({ name: entry.name, startTime: entry.startTime, duration: entry.duration })));
  mkdirSync(evidenceDir, { recursive: true });
  writeFileSync(evidenceFile, JSON.stringify({
    text_entry_submission_to_response_and_summary_ms: browserCompleted - browserStarted,
    timing_boundary: "Before search-field fill through recommendation response receipt and updated preference-summary visibility in Chromium; excludes completion of all result cards and artwork.",
    artwork_enabled: true,
    artwork_resource_count: artworkResources.length,
    artwork_resources: artworkResources,
    final_request_id: searched.request_id,
    retrieval: searched.retrieval,
    reranker: searched.reranker,
  }, null, 2) + "\n", { flag: "wx" });
});

test("one controlled recommendation failure recovers through Retry", async ({ page, request }) => {
  const statusResponse = await request.get("/api/v1/application/status");
  expect(statusResponse.ok(), "release-artwork requires a ready application status endpoint").toBe(true);
  const status = await statusResponse.json();
  expect(status.backend_mode, "release-artwork requires the learned backend").toBe("learned");
  expect(status.posters_configured, "release-artwork requires backend TMDb artwork configuration").toBe(true);

  let failed = false;
  await page.route("**/api/v1/recommendations", async (route) => {
    if (!failed) {
      failed = true;
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({
        error: { code: "controlled_release_check", message: "Controlled release-check failure." },
      }) });
      return;
    }
    await route.continue();
  });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Controlled release-check failure.");
  const recovery = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await page.getByRole("button", { name: "Retry" }).click();
  const recovered = await (await recovery).json();
  expect(recovered.backend_mode).toBe("learned");
  expect(recovered.reranker.name).toBe("lambdarank");
  await expect(page.locator(".feature-hero h1")).toBeVisible();
});

