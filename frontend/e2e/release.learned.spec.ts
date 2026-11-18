import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

const evidenceFile = resolve(process.cwd(), process.env.M4_RELEASE_EVIDENCE_DIR ?? "../tmp/rating_profile_verification_20261117", "browser-flow-artwork.json");
const containerEvidenceDir = process.env.M4_CONTAINER_EVIDENCE_DIR;

test("real learned browser flow submits ratings to two-tower and LambdaRank", async ({ page, request }) => {
  const status = await (await request.get("/api/v1/application/status")).json();
  expect(status.backend_mode).toBe("learned");
  expect(status.posters_configured).toBe(true);
  await page.addInitScript(() => localStorage.clear());

  const initialResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await page.goto("/");
  const initial = await (await initialResponse).json();
  expect(initial.retrieval.name).toBe("two-tower-weaviate");
  expect(initial.reranker.name).toBe("lambdarank");
  await expect(page.locator('img[src^="https://image.tmdb.org/"]').first()).toBeVisible({ timeout: 15_000 });

  const search = page.getByRole("search", { name: "Rate a movie" });
  await search.getByRole("combobox").fill("Toy Story");
  await page.getByLabel("Movie search results").getByRole("button", { name: /Toy Story \(1995\)/ }).first().click();
  const editor = page.getByRole("dialog", { name: /Rate Toy Story/ });
  await editor.getByLabel("Your rating").selectOption("4.5");
  const submittedRequest = page.waitForRequest((candidate) => candidate.url().endsWith("/recommendations") &&
    candidate.postDataJSON().preferences.ratings?.[0]?.rating === 4.5);
  const learnedResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  const started = Date.now();
  await editor.getByRole("button", { name: "Save rating" }).click();
  const submitted = (await submittedRequest).postDataJSON();
  const learned = await (await learnedResponse).json();
  expect(submitted.known_user_id).toBeNull();
  expect(submitted.preferences.ratings[0]).toMatchObject({ movie_id: "1", rating: 4.5 });
  expect(learned.retrieval.name).toBe("two-tower-weaviate");
  expect(learned.reranker.name).toBe("lambdarank");
  expect(learned.recommendations.every((item: { movie: { movie_id: string } }) => item.movie.movie_id !== "1")).toBe(true);
  await expect(editor).toBeHidden();

  await page.getByRole("button", { name: "Preferences" }).click();
  const preferences = page.getByRole("dialog", { name: "Recommendation preferences" });
  await preferences.getByRole("button", { name: "Include genres" }).click();
  const genreDialog = page.getByRole("dialog", { name: "Include genres" });
  const genreRequest = page.waitForRequest((candidate) => candidate.url().endsWith("/recommendations") &&
    candidate.postDataJSON().preferences.include_genres.includes("Adventure"));
  await genreDialog.getByRole("button", { name: "Adventure" }).click();
  expect((await genreRequest).postDataJSON().preferences.ratings[0].rating).toBe(4.5);
  await expect(genreDialog).toBeVisible();

  if (!existsSync(evidenceFile)) {
    mkdirSync(resolve(evidenceFile, ".."), { recursive: true });
    writeFileSync(evidenceFile, JSON.stringify({
      verification: "candidate working tree against saved learned artifacts",
      elapsed_rating_submission_to_response_ms: Date.now() - started,
      request: submitted,
      retrieval: learned.retrieval,
      reranker: learned.reranker,
      rated_item_excluded: true,
      artwork_enabled: true,
    }, null, 2) + "\n", { flag: "wx" });
  }
});

test("one controlled recommendation failure preserves a saved rating and retries", async ({ page, request }) => {
  expect((await (await request.get("/api/v1/application/status")).json()).backend_mode).toBe("learned");
  let recommendationCount = 0;
  await page.route("**/api/v1/recommendations", async (route) => {
    recommendationCount += 1;
    if (recommendationCount === 2) {
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { code: "controlled_release_check", message: "Controlled release-check failure." } }) });
    } else await route.continue();
  });
  await page.addInitScript(() => localStorage.clear());
  await page.goto("/");
  const search = page.getByRole("search", { name: "Rate a movie" });
  await search.getByRole("combobox").fill("Toy Story");
  await page.getByLabel("Movie search results").getByRole("button", { name: /Toy Story \(1995\)/ }).first().click();
  const editor = page.getByRole("dialog", { name: /Rate Toy Story/ });
  await editor.getByLabel("Your rating").selectOption("3.5");
  await editor.getByRole("button", { name: "Save rating" }).click();
  await expect(editor).toBeHidden();
  const notice = page.locator(".rating-notice");
  await expect(notice).toContainText("Recommendations couldn’t update.");
  const recovery = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await notice.getByRole("button", { name: "Retry" }).click();
  const recovered = await (await recovery).json();
  expect(recovered.reranker.name).toBe("lambdarank");
  expect(recovered.applied_preferences.ratings[0].rating).toBe(3.5);
});

test("title-only recommendation failure exposes main-page Retry with durable state unchanged", async ({ page, request }) => {
  expect((await (await request.get("/api/v1/application/status")).json()).backend_mode).toBe("learned");
  let failTitleRecommendation = true;
  await page.route("**/api/v1/recommendations", async (route) => {
    const body = route.request().postDataJSON();
    if (failTitleRecommendation && body.temporary_history_items?.includes("2571")) {
      failTitleRecommendation = false;
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({
        error: { code: "controlled_title_refresh", message: "Controlled title refresh failure." },
      }) });
    } else await route.continue();
  });
  await page.addInitScript(() => localStorage.clear());
  const initialResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await page.goto("/");
  await initialResponse;
  const durableBefore = await page.evaluate(() => localStorage.getItem("trustworthy-recsys-browser-profile-v1"));

  await page.getByRole("tab", { name: "Find recommendations" }).click();
  await page.getByRole("textbox", { name: "Describe the movies you want" }).fill("movies like The Matrix");
  const failedRequest = page.waitForRequest((candidate) => candidate.url().endsWith("/recommendations") &&
    candidate.postDataJSON().temporary_history_items?.includes("2571"));
  await page.getByRole("button", { name: "Find recommendations", exact: true }).click();
  expect((await failedRequest).postDataJSON()).toMatchObject({
    preference_revision: 1,
    temporary_history_items: ["2571"],
    preferences: { ratings: [], history_items: [] },
  });

  const alert = page.locator(".personalization-summary").getByRole("alert");
  await expect(alert).toContainText("Controlled title refresh failure.");
  await expect(page.locator(".recommendation-section article").first()).toBeVisible();
  const retry = page.locator(".personalization-summary").getByRole("button", { name: "Retry" });
  await expect(retry).toBeVisible();
  if (containerEvidenceDir) {
    mkdirSync(containerEvidenceDir, { recursive: true });
    await page.screenshot({ path: resolve(containerEvidenceDir, "title-only-failure-main-page.png"), fullPage: true });
  }
  const retryRequest = page.waitForRequest((candidate) => candidate.url().endsWith("/recommendations") &&
    candidate.postDataJSON().temporary_history_items?.includes("2571"));
  const retryResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await retry.click();
  expect((await retryRequest).postDataJSON()).toMatchObject({ preference_revision: 1, temporary_history_items: ["2571"] });
  const recovered = await (await retryResponse).json();
  expect(recovered.retrieval.name).toBe("two-tower-weaviate");
  expect(recovered.reranker.name).toBe("lambdarank");
  expect(recovered.recommendations.every((item: { movie: { movie_id: string } }) => item.movie.movie_id !== "2571")).toBe(true);
  expect(await page.evaluate(() => localStorage.getItem("trustworthy-recsys-browser-profile-v1"))).toBe(durableBefore);
  if (containerEvidenceDir) {
    await page.screenshot({ path: resolve(containerEvidenceDir, "title-only-retry-recovered.png"), fullPage: true });
    const payloadEvidence = resolve(containerEvidenceDir, "title-only-failure-retry.json");
    if (!existsSync(payloadEvidence)) {
      writeFileSync(payloadEvidence, JSON.stringify({
        durable_profile_before: durableBefore,
        durable_profile_after: await page.evaluate(() => localStorage.getItem("trustworthy-recsys-browser-profile-v1")),
        failed_request: (await failedRequest).postDataJSON(),
        retry_request: (await retryRequest).postDataJSON(),
        recovery: { retrieval: recovered.retrieval, reranker: recovered.reranker, matrix_excluded: true },
      }, null, 2) + "\n", { flag: "wx" });
    }
  }
});
