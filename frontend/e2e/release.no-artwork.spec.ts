import { expect, test } from "@playwright/test";

test("learned flow remains ready and searchable with neutral artwork", async ({ page, request }) => {
  const statusResponse = await request.get("/api/v1/application/status");
  expect(statusResponse.ok(), "release-no-artwork requires a ready application status endpoint").toBe(true);
  const status = await statusResponse.json();
  expect(status.backend_mode, "release-no-artwork requires the learned backend").toBe("learned");
  expect(status.posters_configured, "release-no-artwork requires TMDb to be absent from the backend").toBe(false);

  const initialResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await page.goto("/");
  const initial = await (await initialResponse).json();
  expect(initial.backend_mode).toBe("learned");
  expect(initial.retrieval.name).toBe("two-tower-weaviate");
  expect(initial.reranker.name).toBe("lambdarank");
  await expect(page.locator(".hero-neutral")).toBeVisible();
  await expect(page.locator('img[src^="https://image.tmdb.org/"]')).toHaveCount(0);

  await page.getByRole("button", { name: /Personalize/ }).click();
  const drawer = page.getByRole("dialog", { name: "Personalize your recommendations" });
  const favorites = drawer.getByLabel("Movies you like");
  await favorites.fill("Toy Story (1995)");
  await drawer.getByRole("button", { name: /Toy Story \(1995\).*Add/ }).click();
  const favoriteResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await drawer.getByRole("button", { name: "Update recommendations" }).click();
  const favoriteResult = await (await favoriteResponse).json();
  expect(favoriteResult.retrieval.mode).toBe("learned");
  expect(favoriteResult.reranker.name).toBe("lambdarank");
  expect(favoriteResult.warnings).not.toContain("Popularity fallback used because no supported history items were supplied.");
  await expect(page.locator('img[src^="https://image.tmdb.org/"]')).toHaveCount(0);

  await page.getByRole("button", { name: "About" }).click();
  const about = page.getByRole("dialog", { name: "About Trustworthy Recommendation System" });
  await about.getByText("System details").click();
  await expect(about.getByText("TMDB not configured")).toBeVisible();
  await about.getByRole("button", { name: "Close details" }).click();

  const search = page.getByRole("search", { name: "Search movie recommendations" });
  const resultResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await search.getByRole("textbox").fill("show comedy without thriller");
  await search.getByRole("textbox").press("Enter");
  const result = await (await resultResponse).json();
  expect(result.retrieval.mode).toBe("learned");
  expect(result.reranker.name).toBe("lambdarank");
  expect(result.applied_preferences.include_genres).toContain("Comedy");
  expect(result.applied_preferences.exclude_genres).toContain("Thriller");
  expect(result.recommendations.every((item: { movie: { genres: string[] } }) =>
    item.movie.genres.includes("Comedy") && !item.movie.genres.includes("Thriller"))).toBe(true);
  await expect(page.getByRole("heading", { name: "Your recommendations", level: 2 })).toBeVisible();
});

