import { expect, test, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const evidenceDir = resolve(process.cwd(), "..", "tmp", "search-rating-correction-20261117");
const profile = {
  version: 1,
  ratings: [
    { movie_id: "1", rating: 4.5, rated_at: "2026-11-15T12:00:00.000Z" },
    { movie_id: "260", rating: 5, rated_at: "2026-11-16T12:00:00.000Z" },
    { movie_id: "1196", rating: 4, rated_at: "2026-11-17T12:00:00.000Z" },
  ],
  legacy_favorites: [],
  movies: {
    "1": { movie_id: "1", title: "Toy Story (1995)", genres: ["Adventure", "Animation", "Children", "Comedy", "Fantasy"], poster_url: null },
    "260": { movie_id: "260", title: "Star Wars: Episode IV - A New Hope (1977)", genres: ["Action", "Adventure", "Sci-Fi"], poster_url: null },
    "1196": { movie_id: "1196", title: "Star Wars: Episode V - The Empire Strikes Back (1980)", genres: ["Action", "Adventure", "Sci-Fi"], poster_url: null },
  },
};

async function assertInsideViewport(page: Page, selector: string) {
  const bounds = await page.locator(selector).boundingBox();
  const viewport = page.viewportSize();
  expect(bounds).not.toBeNull();
  expect(viewport).not.toBeNull();
  expect(bounds!.x).toBeGreaterThanOrEqual(15);
  expect(bounds!.y).toBeGreaterThanOrEqual(15);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport!.width - 15);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport!.height - 15);
  expect(await page.locator(selector).evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
}

async function openLongTitleDialog(page: Page) {
  await page.route("**/api/v1/catalog/search**", async (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ backend_mode: "baseline", total: 1, items: [{
      movie_id: "1",
      title: "The Improbably Long Movie Title That Still Wraps Cleanly Beside the Close Button (1995)",
      genres: ["Adventure", "Comedy"],
      poster_url: null,
    }] }),
  }));
  const input = page.getByRole("combobox", { name: "Search for a movie by title or release year" });
  await input.fill("overflow test");
  await input.click();
  await page.getByLabel("Movie search results").getByRole("button").click();
  return page.getByRole("dialog", { name: /Rate The Improbably Long Movie Title/ });
}

test.beforeAll(() => mkdirSync(evidenceDir, { recursive: true }));

test("compact desktop homepage, aligned search, and ratings management", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.addInitScript((saved) => localStorage.setItem("trustworthy-recsys-browser-profile-v1", JSON.stringify(saved)), profile);
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Manage ratings" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.screenshot({ path: resolve(evidenceDir, "desktop-rate-mode.png"), fullPage: true });
  await page.getByRole("tab", { name: "Find recommendations" }).click();
  await expect(page.getByRole("textbox", { name: "Describe the movies you want" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Use microphone for recommendations" })).toBeVisible();
  await page.screenshot({ path: resolve(evidenceDir, "desktop-find-recommendations-mode.png"), fullPage: true });
  await page.getByRole("tab", { name: "Rate a movie" }).click();

  const input = page.getByRole("combobox", { name: "Search for a movie by title or release year" });
  await input.fill("sleepless");
  const results = page.getByLabel("Movie search results");
  await expect(results.getByRole("button").first()).toBeVisible();
  const inputBox = await page.locator(".main-search-bar").boundingBox();
  const resultsBox = await results.boundingBox();
  expect(Math.abs(inputBox!.width - resultsBox!.width)).toBeLessThanOrEqual(1);
  expect(resultsBox!.height).toBeLessThanOrEqual(360);
  await input.press("Escape");
  await expect(results).toBeHidden();

  await page.screenshot({ path: resolve(evidenceDir, "desktop-homepage.png"), fullPage: true });
  await page.getByRole("button", { name: "Manage ratings" }).click();
  const manager = page.getByRole("dialog", { name: "Your ratings · 3" });
  await expect(manager.getByText("Saved in this browser.")).toBeVisible();
  await expect(manager.getByRole("button", { name: "Edit" })).toHaveCount(3);
  await expect(manager.getByRole("button", { name: "Remove" })).toHaveCount(3);
  await assertInsideViewport(page, ".ratings-manager-panel");
  await page.screenshot({ path: resolve(evidenceDir, "desktop-ratings-management.png"), fullPage: true });
});

test("rating dialog stays contained on laptop, mobile, and enlarged text", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 700 });
  await page.goto("/");
  let dialog = await openLongTitleDialog(page);
  await expect(dialog).toBeVisible();
  await assertInsideViewport(page, ".rating-panel");
  await expect(dialog.getByRole("button", { name: "Save rating" })).toBeVisible();
  await expect(dialog.getByLabel("Your rating")).toBeVisible();
  await dialog.getByRole("button", { name: "Close rating" }).click();
  await expect(page.getByRole("combobox", { name: "Search for a movie by title or release year" })).toBeFocused();

  await page.setViewportSize({ width: 390, height: 844 });
  dialog = await openLongTitleDialog(page);
  await assertInsideViewport(page, ".rating-panel");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: resolve(evidenceDir, "mobile-rating-dialog.png"), fullPage: true });

  await page.evaluate(() => { document.documentElement.style.fontSize = "200%"; });
  await assertInsideViewport(page, ".rating-panel");
  expect(await page.locator(".rating-panel").evaluate((node) => node.scrollHeight >= node.clientHeight)).toBe(true);
});

test("empty homepage stays compact", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByRole("tab", { name: "Rate a movie" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Find recommendations" })).toBeVisible();
  await expect(page.getByText("Saved in this browser.")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Manage ratings" })).toHaveCount(0);
  await expect(page.locator(".ratings-panel")).toHaveCount(0);
  const hero = await page.locator(".feature-hero").boundingBox();
  expect(hero!.y).toBeLessThan(240);
});

test("saved rating reports a delayed recommendation failure and keeps Retry", async ({ page }) => {
  let recommendationCount = 0;
  await page.route("**/api/v1/recommendations", async (route) => {
    recommendationCount += 1;
    if (recommendationCount === 1) return route.continue();
    await new Promise((resolve) => setTimeout(resolve, 250));
    return route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { code: "temporary", message: "Temporary failure" } }) });
  });
  await page.goto("/");
  await page.getByRole("combobox", { name: "Search for a movie by title or release year" }).fill("Toy Story");
  await page.getByLabel("Movie search results").getByRole("button", { name: /Toy Story \(1995\)/ }).first().click();
  const dialog = page.getByRole("dialog", { name: /Rate Toy Story/ });
  await dialog.getByRole("button", { name: "Save rating" }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByText("Rating saved. Updating recommendations…")).toBeVisible();
  const notice = page.locator(".rating-notice");
  await expect(notice).toContainText("Rating saved. Recommendations couldn’t update.");
  await expect(notice.getByRole("button", { name: "Retry" })).toBeVisible();
});

test("successful save closes immediately and confirms on the page", async ({ page }) => {
  let recommendationCount = 0;
  await page.route("**/api/v1/recommendations", async (route) => {
    recommendationCount += 1;
    if (recommendationCount === 2) await new Promise((resolve) => setTimeout(resolve, 700));
    await route.continue();
  });
  await page.goto("/");
  await page.getByRole("combobox", { name: "Search for a movie by title or release year" }).fill("Toy Story");
  await page.getByLabel("Movie search results").getByRole("button", { name: /Toy Story \(1995\)/ }).first().click();
  const dialog = page.getByRole("dialog", { name: /Rate Toy Story/ });
  await dialog.getByLabel("Your rating").selectOption("4.5");
  await dialog.getByRole("button", { name: "Save rating" }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByText("Rating saved. Updating recommendations…")).toBeVisible();
  await expect(page.getByRole("button", { name: "Manage ratings" })).toBeVisible();
  await page.screenshot({ path: resolve(evidenceDir, "after-successful-rating-save.png"), fullPage: true });
  await expect(page.getByText("Rating saved.", { exact: true })).toBeVisible();
});
