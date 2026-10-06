import { expect, test, type Page } from "@playwright/test";

const screenshots = "test-results/screenshots";

async function openDrawer(page: Page) {
  await page.getByRole("button", { name: /Personalize/ }).click();
  return page.getByRole("dialog", { name: "Personalize your recommendations" });
}

async function addToyStory(page: Page) {
  const drawer = await openDrawer(page);
  const search = drawer.getByLabel("Movies you like");
  await expect(search).toBeFocused();
  await search.fill("Toy Story (1995)");
  await drawer.getByRole("option", { name: /Toy Story \(1995\)/ }).click();
  await expect(drawer.getByRole("button", { name: "Remove Toy Story (1995)" })).toBeVisible();
  return drawer;
}

async function chooseGenre(page: Page, drawer: ReturnType<Page["getByRole"]>, kind: "Include" | "Avoid", value: string) {
  await drawer.getByRole("button", { name: new RegExp(`${kind} genres`) }).click();
  const selector = page.getByRole("dialog", { name: `${kind} genres` });
  await selector.getByRole("button", { name: value }).click();
  await selector.getByRole("button", { name: "Done" }).click();
}

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/catalog/posters", async (route) => {
    const body = JSON.parse(route.request().postData() ?? "{}") as { movie_ids?: string[] };
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ provider: "tmdb", configured: true, items: (body.movie_ids ?? []).map((movie_id, index) => ({ movie_id, tmdb_id: movie_id, poster_url: "/test-poster.svg", backdrop_url: index === 0 ? "/test-backdrop.svg" : null, backdrop_url_small: index === 0 ? "/test-backdrop.svg" : null, status: "available" })) }) });
  });
});

test("initial visit immediately shows a ranked feature and movie row", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Explore movies" })).toBeVisible();
  await expect(page.locator(".feature-hero h1")).toBeVisible();
  await expect(page.locator(".feature-hero")).toHaveCSS("opacity", "1");
  await expect(page.locator(".site-header")).toHaveCSS("position", "relative");
  await expect(page.locator(".site-header")).toHaveCSS("backdrop-filter", "none");
  const headerBottom = await page.locator(".site-header").evaluate((element) => element.getBoundingClientRect().bottom); const heroTop = await page.locator(".feature-hero").evaluate((element) => element.getBoundingClientRect().top); expect(heroTop).toBeGreaterThanOrEqual(headerBottom);
  const heroBox = await page.locator(".feature-hero").boundingBox(); expect(heroBox?.x).toBeGreaterThanOrEqual(24); expect(heroBox?.width).toBeLessThan(1440); expect(parseFloat(await page.locator(".feature-hero").evaluate((element) => getComputedStyle(element).borderRadius))).toBeGreaterThanOrEqual(20);
  await expect(page.locator(".personalization-summary")).toHaveCount(0);
  await expect(page.locator(".portrait-card").first()).toBeVisible();
  await expect(page.locator(".rank-badge")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Explore movies", level: 2 })).toHaveCount(1);
  await expect(page.getByText(/More to explore|More recommendations/)).toHaveCount(0);
  await expect(page.locator(".row-track").first()).toHaveCSS("scrollbar-width", "none");
  await expect(page.getByRole("button", { name: /Next Explore movies/ })).toBeVisible();
  expect((await page.locator(".movie-row").first().boundingBox())?.y).toBeLessThan(900);
  await page.locator(".feature-media .hero-backdrop").waitFor({ state: "visible", timeout: 5_000 }).catch(() => undefined);
  await page.screenshot({ path: `${screenshots}/movie-first-initial-desktop.png` });
});

test("cards keep direct actions and subtle emphasis without a floating preview", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 }); await page.goto("/");
  const first = page.locator(".portrait-card").first(); const second = page.locator(".portrait-card").nth(1);
  const originalHero = await page.locator(".feature-hero h1").textContent();
  await expect(first).toBeVisible(); const before = await second.evaluate((element) => ({ x: (element as HTMLElement).offsetLeft, y: (element as HTMLElement).offsetTop })); await first.hover();
  await expect(page.locator(".expanded-preview")).toHaveCount(0);
  await expect(first.locator(".card-actions")).toHaveCSS("opacity", "1");
  await expect(first).not.toHaveCSS("transform", "none");
  await page.screenshot({ path: `${screenshots}/movie-first-card-hover-desktop.png` });
  await expect(second).toBeVisible();
  await expect.poll(async () => second.evaluate((element) => (element as HTMLElement).offsetLeft)).toBe(before.x);
  await expect.poll(async () => second.evaluate((element) => (element as HTMLElement).offsetTop)).toBe(before.y);
  await expect(page.locator(".feature-hero h1")).toHaveText(originalHero ?? "");
  await page.evaluate(() => window.scrollTo(0, 0));
  await first.evaluate((element) => element.focus({ preventScroll: true }));
  await expect(first).toBeFocused();
  await expect(first.locator(".card-actions")).toHaveCSS("opacity", "1");
  await first.getByRole("button", { name: /Details for/ }).click(); const details = page.getByRole("dialog"); await expect(details).toBeVisible(); await details.getByRole("button", { name: "Close details" }).click();
  await first.getByRole("button", { name: /Add .* to favorites/ }).click(); await expect(first.getByRole("button", { name: /is in favorites/ })).toBeDisabled();
  await second.getByRole("button", { name: "Hide this movie" }).click(); await expect(page.getByText("Pending changes. Update to apply them.")).toBeVisible();
  await expect(page.locator(".feature-hero h1")).toHaveText(originalHero ?? "");
});

test("About explains the active system and loads the bundled TMDb logo", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 }); await page.goto("/"); await page.getByRole("button", { name: "About" }).click();
  const about = page.getByRole("dialog", { name: "About Trustworthy Recommendation System" }); await expect(about.getByRole("heading", { name: "How it works" })).toBeVisible(); await expect(about.getByText(/TMDB supplies display artwork only/)).toBeVisible(); await expect(about.getByRole("link", { name: /System card/i })).toHaveCount(0);
  const logo = about.getByAltText("The Movie Database (TMDB)"); await expect(logo).toBeVisible(); const logoState = await logo.evaluate((image: HTMLImageElement) => ({ complete: image.complete, width: image.naturalWidth, src: image.currentSrc })); expect(logoState.complete).toBe(true); expect(logoState.width).toBeGreaterThan(0); expect(logoState.src).toContain("/tmdb-logo.svg"); await expect(logo).toHaveAttribute("src", "/tmdb-logo.svg"); await logo.scrollIntoViewIfNeeded(); await page.screenshot({ path: `${screenshots}/movie-first-about-desktop.png` });
  await about.getByText("System details").click(); await expect(about.getByText(/Backend mode/)).toBeVisible();
});

test("one drawer update applies its exact pending preferences", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 }); await page.goto("/");
  const originalHero = await page.locator(".feature-hero h1").textContent();
  const drawer = await addToyStory(page); await chooseGenre(page, drawer, "Include", "Comedy"); await chooseGenre(page, drawer, "Avoid", "Horror");
  await expect(drawer.getByText(/Text refinement is coming soon/)).toHaveCount(0);
  await page.screenshot({ path: `${screenshots}/movie-first-drawer-desktop.png` });
  await expect(page.locator(".feature-hero h1")).toHaveText(originalHero ?? "");
  const submissions = page.waitForRequest((request) => request.url().includes("/api/v1/recommendations") && request.method() === "POST");
  await drawer.getByRole("button", { name: "Update recommendations" }).click();
  const submitted = await submissions; const payload = submitted.postDataJSON();
  expect(payload.preferences.history_items.length).toBe(1); expect(payload.preferences.include_genres).toContain("Comedy"); expect(payload.preferences.exclude_genres).toContain("Horror");
  await expect(drawer).toBeHidden();
  await expect(page.getByRole("heading", { name: "Your recommendations", level: 2 })).toBeVisible();
  await expect(page.getByText(/More to explore|More recommendations/)).toHaveCount(0);
  await expect(page.getByText("Pending changes. Update to apply them.")).toHaveCount(0);
  await page.waitForTimeout(200);
  await page.screenshot({ path: `${screenshots}/movie-first-applied-desktop.png` });
  const resetRequest = page.waitForRequest((request) => request.url().includes("/api/v1/recommendations") && request.method() === "POST" && request.postDataJSON().preferences.history_items.length === 0);
  await page.getByRole("button", { name: "Reset all" }).click(); await resetRequest;
  await expect(page.getByRole("heading", { name: "Explore movies", level: 2 })).toBeVisible();
});

test("mocked backdrops enhance display and failed artwork retains movies", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.route("**/api/v1/catalog/posters", async (route) => {
    const body = JSON.parse(route.request().postData() ?? "{}") as { movie_ids?: string[] };
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ provider: "tmdb", configured: true, items: (body.movie_ids ?? []).map((movie_id, index) => ({ movie_id, tmdb_id: "862", poster_url: index ? null : "/test-poster.svg", backdrop_url: index ? null : "/test-backdrop.svg", status: index ? "missing" : "available" })) }) });
  });
  await page.goto("/"); await expect(page.locator(".feature-media img")).toHaveAttribute("src", "/test-backdrop.svg");
  const titles = await page.locator(".portrait-card h3").allTextContents(); expect(titles.length).toBeGreaterThan(1);
  await page.screenshot({ path: `${screenshots}/movie-first-artwork-and-fallback-desktop.png` });
});

test("successful empty results retain applied preferences and recover with one reset", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 }); let recommendationRequests = 0;
  await page.route("**/api/v1/recommendations", async (route) => { recommendationRequests += 1; const request = route.request().postDataJSON(); await route.fulfill({ contentType: "application/json", body: JSON.stringify({ request_id: `empty-${recommendationRequests}`, preference_revision: request.preference_revision, recommendations: [], applied_preferences: request.preferences, retrieval: { name: "fixture", version: "1", mode: "fixture" }, reranker: { name: "pass-through", version: "1", mode: "pass_through" }, backend_mode: "fixture", partial: true, warnings: ["bounded candidate pool exhausted"], timings: { retrieval_ms: 1, filtering_ms: 1, reranking_ms: 1, total_ms: 3 } }) }); });
  await page.goto("/"); await expect(page.getByRole("heading", { name: "No recommendations found" })).toBeVisible();
  await page.getByRole("button", { name: "Edit preferences" }).click(); const drawer = page.getByRole("dialog", { name: "Personalize your recommendations" }); await chooseGenre(page, drawer, "Include", "Comedy"); await drawer.getByRole("button", { name: "Update recommendations" }).click();
  await expect(page.getByText("Include Comedy")).toBeVisible(); await expect(page.getByText(/bounded candidate|constraints were not relaxed/i)).toHaveCount(0); await expect(page.getByRole("button", { name: "Edit preferences" })).toBeVisible();
  await page.screenshot({ path: `${screenshots}/movie-first-empty-results-desktop.png` });
  const beforeReset = recommendationRequests; await page.getByRole("button", { name: "Reset all" }).first().click(); await expect.poll(() => recommendationRequests).toBe(beforeReset + 1); await expect(page.locator(".personalization-summary")).toHaveCount(0);
});

test("long-title hero uses intentional poster-only fallback", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.route("**/api/v1/recommendations", async (route) => {
    const request = route.request().postDataJSON();
    const movies = Array.from({ length: 7 }, (_, index) => ({ movie_id: String(1000 + index), title: index === 0 ? "An Extremely Long Catalog Movie Title That Still Stays Inside Its Readable Hero Area (2004)" : `Movie ${index}, The (${1990 + index})`, genres: ["Drama"], poster_url: null }));
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ request_id: "long-title", preference_revision: request.preference_revision, recommendations: movies.map((movie, index) => ({ rank: index + 1, movie, retrieval_score: null, ranking_score: null })), applied_preferences: request.preferences, retrieval: { name: "fixture", version: "1", mode: "fixture" }, reranker: { name: "pass-through", version: "1", mode: "pass_through" }, backend_mode: "fixture", partial: false, warnings: [], timings: { retrieval_ms: 1, filtering_ms: 1, reranking_ms: 1, total_ms: 3 } }) });
  });
  await page.route("**/api/v1/catalog/posters", async (route) => { const ids = route.request().postDataJSON().movie_ids as string[]; await route.fulfill({ contentType: "application/json", body: JSON.stringify({ provider: "tmdb", configured: true, items: ids.map((movie_id, index) => ({ movie_id, tmdb_id: movie_id, poster_url: index === 0 ? "/test-poster.svg" : null, backdrop_url: null, backdrop_url_small: null, status: index === 0 ? "available" : "missing" })) }) }); });
  await page.goto("/"); await expect(page.locator(".hero-poster-fallback img")).toHaveAttribute("src", "/test-poster.svg");
  await expect(page.locator(".feature-hero h1")).not.toContainText("(2004)");
  const headerBottom = await page.locator(".site-header").evaluate((element) => element.getBoundingClientRect().bottom); const titleTop = await page.locator(".feature-hero h1").evaluate((element) => element.getBoundingClientRect().top); expect(titleTop).toBeGreaterThan(headerBottom);
  await page.screenshot({ path: `${screenshots}/movie-first-long-title-poster-fallback-desktop.png` });
});

test("update failure preserves pending edits and prior movies", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 }); await page.goto("/");
  const original = await page.locator(".feature-hero h1").textContent(); const drawer = await addToyStory(page); await drawer.getByRole("button", { name: "Close personalization" }).click();
  await page.route("**/api/v1/recommendations", (route) => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { code: "backend_unavailable", message: "Recommendations are temporarily unavailable." } }) }));
  await page.getByRole("button", { name: "Update recommendations" }).click();
  await expect(page.getByRole("alert")).toContainText("Recommendations are temporarily unavailable."); await expect(page.locator(".feature-hero h1")).toHaveText(original ?? ""); await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
});

test("mobile browsing and drawer avoid page overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 }); await page.goto("/"); await expect(page.locator(".feature-hero h1")).toBeVisible(); await expect(page.locator(".portrait-card").first()).toBeVisible();
  const mobileHero = await page.locator(".feature-hero").boundingBox(); expect(mobileHero?.x).toBeGreaterThanOrEqual(15); expect(parseFloat(await page.locator(".feature-hero").evaluate((element) => getComputedStyle(element).borderRadius))).toBeGreaterThanOrEqual(12);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const drawer = await openDrawer(page); await expect(drawer).toBeVisible(); await expect(drawer.getByRole("button", { name: "Include genres" })).toHaveText("Include"); await expect(drawer.getByRole("button", { name: "Avoid genres" })).toHaveText("Avoid"); expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true); expect(await drawer.getByRole("button", { name: "Include genres" }).evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true); expect(await drawer.getByRole("button", { name: "Avoid genres" }).evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true); await page.screenshot({ path: `${screenshots}/movie-first-drawer-mobile.png` });
  await drawer.getByRole("button", { name: "Close personalization" }).click(); const firstCard = page.locator(".portrait-card").first(); await firstCard.focus(); await expect(firstCard.locator(".card-actions")).toHaveCSS("opacity", "1"); await expect(page.locator(".expanded-preview")).toHaveCount(0); await page.screenshot({ path: `${screenshots}/movie-first-card-mobile.png` }); await page.locator(".card-art").first().dispatchEvent("click"); const details = page.getByRole("dialog"); await expect(details).toBeVisible(); await details.getByRole("button", { name: "Close details" }).click();
});
