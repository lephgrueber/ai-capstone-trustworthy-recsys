import { expect, test, type Page } from "@playwright/test";

async function rateMovie(page: Page, query: string, title: RegExp, rating: string) {
  const search = page.getByRole("search", { name: "Rate a movie" });
  const input = search.getByRole("combobox");
  await input.fill(query);
  await input.click();
  await page.getByLabel("Movie search results").getByRole("button", { name: title }).first().click();
  const editor = page.getByRole("dialog", { name: new RegExp(`Rate ${query}`, "i") });
  await editor.getByLabel("Your rating").selectOption(rating);
  const request = page.waitForRequest((candidate) =>
    candidate.url().endsWith("/recommendations") &&
    candidate.postDataJSON().preferences.ratings?.some((record: { rating: number }) => record.rating === Number(rating))
  );
  await editor.getByRole("button", { name: "Save rating" }).click();
  return { request: await request };
}

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/catalog/posters", async (route) => {
    const body = route.request().postDataJSON() as { movie_ids?: string[] };
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({
      provider: "tmdb", configured: true,
      items: (body.movie_ids ?? []).map((movie_id, index) => ({ movie_id, tmdb_id: movie_id,
        poster_url: index ? null : "/test-poster.svg", backdrop_url: index ? null : "/test-backdrop.svg",
        backdrop_url_small: index ? null : "/test-backdrop.svg", status: index ? "missing" : "available" }))
    }) });
  });
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
});

test("search, explicit rating, editing, removal and reload use one browser profile", async ({ page }) => {
  const requests: Array<Record<string, any>> = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/recommendations")) requests.push(request.postDataJSON());
  });
  await page.goto("/");
  await expect(page.getByRole("tab", { name: "Rate a movie" })).toBeVisible();
  await expect.poll(() => requests.length).toBeGreaterThan(0);
  expect(requests[0].preferences.ratings).toEqual([]);

  const search = page.getByRole("search", { name: "Rate a movie" });
  await search.getByRole("combobox").fill("Toy Story");
  const result = page.getByLabel("Movie search results").getByRole("button", { name: /Toy Story \(1995\)/ }).first();
  await expect(result).toBeVisible();
  await result.click();
  expect(requests).toHaveLength(1);
  await page.getByRole("dialog", { name: /Rate Toy Story/ }).getByRole("button", { name: "Close rating" }).click();
  await expect(page.getByRole("button", { name: "Manage ratings" })).toHaveCount(0);

  const first = await rateMovie(page, "Toy Story", /Toy Story \(1995\)/, "4");
  expect(first.request.postDataJSON().preferences.ratings).toHaveLength(1);
  await expect(page.getByRole("button", { name: "Manage ratings" })).toBeVisible();

  await page.getByRole("button", { name: "Manage ratings" }).click();
  const ratings = page.getByRole("dialog", { name: /Your ratings/ });
  await expect(ratings.getByText("4.0 ★")).toBeVisible();
  await ratings.getByRole("button", { name: "Edit" }).click();
  const edit = page.getByRole("dialog", { name: /Rate Toy Story/ });
  await edit.getByLabel("Your rating").selectOption("5");
  const edited = page.waitForRequest((candidate) => candidate.url().endsWith("/recommendations") &&
    candidate.postDataJSON().preferences.ratings?.[0]?.rating === 5);
  await edit.getByRole("button", { name: "Save rating" }).click();
  expect((await edited).postDataJSON().preferences.ratings).toHaveLength(1);
  await expect(page.getByRole("dialog", { name: /Your ratings/ }).getByText("5.0 ★")).toBeVisible();
  await page.getByRole("button", { name: "Close ratings" }).click();

  const beforeReload = requests.length;
  await page.reload();
  await expect.poll(() => requests.length).toBeGreaterThan(beforeReload);
  expect(requests[beforeReload].preferences.ratings[0].rating).toBe(5);
  await page.getByRole("button", { name: "Manage ratings" }).click();
  await expect(page.getByRole("dialog", { name: /Your ratings/ }).getByText("5.0 ★")).toBeVisible();

  const removal = page.waitForRequest((candidate) => candidate.url().endsWith("/recommendations") &&
    candidate.postDataJSON().preferences.ratings?.length === 0);
  await page.getByRole("dialog", { name: /Your ratings/ }).getByRole("button", { name: "Remove" }).click();
  expect((await removal).postDataJSON().preferences.ratings).toEqual([]);
  await expect(page.getByRole("button", { name: "Manage ratings" })).toHaveCount(0);
});

test("genre changes preserve ratings, refresh automatically, and keep focus in the dialog", async ({ page }) => {
  await page.goto("/");
  await rateMovie(page, "Toy Story", /Toy Story \(1995\)/, "2.5");
  await page.getByRole("button", { name: "Preferences" }).click();
  const preferences = page.getByRole("dialog", { name: "Recommendation preferences" });
  await preferences.getByRole("button", { name: "Include genres" }).click();
  const genres = page.getByRole("dialog", { name: "Include genres" });
  const refresh = page.waitForRequest((request) => request.url().endsWith("/recommendations") &&
    request.postDataJSON().preferences.include_genres.includes("Adventure"));
  await genres.getByRole("button", { name: "Adventure" }).click();
  const payload = (await refresh).postDataJSON();
  expect(payload.preferences.ratings[0].rating).toBe(2.5);
  await expect(genres).toBeVisible();
  await expect(genres.getByRole("button", { name: "Adventure" })).toBeFocused();
});

test("responsive artwork fallback and keyboard controls remain usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator(".feature-hero h1")).toBeVisible();
  await expect(page.locator(".portrait-card").first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const first = page.locator(".portrait-card").first();
  const details = first.getByRole("button", { name: /Details for/ });
  await details.focus();
  await expect(details).toBeFocused();
  await details.click();
  await expect(page.getByRole("dialog").filter({ hasText: "Opening details does not save feedback" })).toBeVisible();
});
