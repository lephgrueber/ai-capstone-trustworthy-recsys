import { expect, test } from "@playwright/test";

test("main-page text search applies filters without opening Personalize", async ({ page }) => {
  await page.goto("/");
  const search = page.getByRole("search", { name: "Search movie recommendations" });
  await expect(search.getByRole("textbox")).toBeEnabled();
  await search.getByRole("textbox").fill("show comedy without thriller");
  await search.getByRole("textbox").press("Enter");
  await expect(page.getByText("Including Comedy; Avoiding Thriller.")).toBeVisible();
  await expect(page.locator(".personalization-summary")).toContainText("Include Comedy");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.screenshot({ path: "test-results/search-desktop.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(search).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: "test-results/search-mobile.png" });
});

test("voice transcript waits for submission and resolved titles remain editable", async ({ page }) => {
  await page.addInitScript(() => {
    class MockSpeech {
      onresult?: (event: { results: { transcript: string }[][] }) => void;
      onend?: () => void;
      start() { this.onresult?.({ results: [[{ transcript: "movies like Orbit House" }]] }); this.onend?.(); }
      abort() { this.onend?.(); }
    }
    Object.defineProperty(window, "SpeechRecognition", { value: MockSpeech });
  });
  await page.goto("/");
  const search = page.getByRole("search", { name: "Search movie recommendations" });
  await expect(search.getByRole("textbox")).toBeEnabled();
  await search.getByRole("button", { name: "Use microphone" }).click();
  await expect(search.getByRole("textbox")).toHaveValue("movies like Orbit House");
  await expect(page.locator(".personalization-summary")).toHaveCount(0);
  await search.getByRole("button", { name: "Search movies" }).click();
  await expect(page.locator(".personalization-summary")).toContainText("1 favorite");
  await page.getByRole("button", { name: "Personalize" }).click();
  const drawer = page.getByRole("dialog", { name: "Personalize your recommendations" });
  await expect(drawer.getByRole("button", { name: "Remove Orbit House" })).toBeVisible();
  await expect(drawer.getByRole("button", { name: "Use microphone" })).toHaveCount(0);
});
