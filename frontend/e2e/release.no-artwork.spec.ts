import { expect, test } from "@playwright/test";

test("learned rating flow remains usable without TMDB artwork", async ({ page, request }) => {
  const status = await (await request.get("/api/v1/application/status")).json();
  expect(status.backend_mode).toBe("learned");
  expect(status.posters_configured).toBe(false);
  await page.addInitScript(() => localStorage.clear());
  const initialResponse = page.waitForResponse((response) => response.url().endsWith("/recommendations") && response.ok());
  await page.goto("/");
  const initial = await (await initialResponse).json();
  expect(initial.retrieval.name).toBe("two-tower-weaviate");
  expect(initial.reranker.name).toBe("lambdarank");
  await expect(page.locator(".hero-neutral")).toBeVisible();
  await expect(page.locator('img[src^="https://image.tmdb.org/"]')).toHaveCount(0);

  const search = page.getByRole("search", { name: "Rate a movie" });
  await search.getByRole("combobox").fill("Toy Story");
  await page.getByLabel("Movie search results").getByRole("button", { name: /Toy Story \(1995\)/ }).first().click();
  const editor = page.getByRole("dialog", { name: /Rate Toy Story/ });
  await editor.getByLabel("Your rating").selectOption("4");
  const response = page.waitForResponse((candidate) => candidate.url().endsWith("/recommendations") && candidate.ok());
  await editor.getByRole("button", { name: "Save rating" }).click();
  const rated = await (await response).json();
  expect(rated.reranker.name).toBe("lambdarank");
  expect(rated.applied_preferences.ratings[0].rating).toBe(4);
  expect(rated.recommendations.every((item: { movie: { movie_id: string } }) => item.movie.movie_id !== "1")).toBe(true);
  await expect(page.locator('img[src^="https://image.tmdb.org/"]')).toHaveCount(0);

  await expect(editor).toBeHidden();
  await page.getByRole("button", { name: "About" }).click();
  const about = page.getByRole("dialog", { name: "About Trustworthy Recommendation System" });
  await about.getByText("System details").click();
  await expect(about.getByText("TMDB not configured")).toBeVisible();
});
