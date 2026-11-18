import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    type Handler = ((event: any) => void) | null;
    type MockRecognition = { lang: string; continuous: boolean; interimResults: boolean; onresult: Handler; onerror: Handler; onend: (() => void) | null; start(): void; abort(): void; aborted: boolean };
    const instances: MockRecognition[] = [];
    class MockSpeechRecognition implements MockRecognition {
      lang = ""; continuous = false; interimResults = false; onresult: Handler = null; onerror: Handler = null; onend: (() => void) | null = null; aborted = false;
      constructor() { instances.push(this); }
      start() { this.aborted = false; }
      abort() { this.aborted = true; }
    }
    Object.defineProperty(window, "SpeechRecognition", { configurable: true, value: MockSpeechRecognition });
    Object.defineProperty(window, "__voiceTest", { configurable: true, value: {
      result(index: number, transcript: string) { instances[index]?.onresult?.({ results: [[{ transcript }]] }); },
      end(index: number) { instances[index]?.onend?.(); },
      error(index: number, error: string) { instances[index]?.onerror?.({ error }); },
      wasAborted(index: number) { return instances[index]?.aborted ?? false; },
    } });
  });
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
});

test("main search selection does not save feedback", async ({ page }) => {
  const recommendationRequests: unknown[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/recommendations")) recommendationRequests.push(request.postDataJSON());
  });
  await page.goto("/");
  await expect.poll(() => recommendationRequests.length).toBe(1);
  const search = page.getByRole("search", { name: "Rate a movie" });
  await search.getByRole("combobox").fill("Signal in the Fog");
  const result = page.getByLabel("Movie search results").getByRole("button", { name: /Signal in the Fog/ }).first();
  await expect(result).toContainText("Signal in the Fog");
  await result.click();
  await expect(page.getByRole("dialog", { name: /Rate Signal in the Fog/ })).toBeVisible();
  expect(recommendationRequests).toHaveLength(1);
  expect(await page.evaluate(() => localStorage.length)).toBe(0);
});

test("main-page recommendation search preserves the numeric profile", async ({ page }) => {
  await page.goto("/");
  const movieSearch = page.getByRole("search", { name: "Rate a movie" });
  await movieSearch.getByRole("combobox").fill("Signal in the Fog");
  await page.getByLabel("Movie search results").getByRole("button", { name: /Signal in the Fog/ }).first().click();
  const rating = page.getByRole("dialog", { name: /Rate Signal in the Fog/ });
  await rating.getByLabel("Your rating").selectOption("4.5");
  await rating.getByRole("button", { name: "Save rating" }).click();
  await expect(rating).toBeHidden();
  await page.getByRole("tab", { name: "Find recommendations" }).click();
  const requestInput = page.getByRole("textbox", { name: "Describe the movies you want" });
  await requestInput.fill("show comedy without thriller");
  const request = page.waitForRequest((candidate) => candidate.url().endsWith("/recommendations") &&
    candidate.postDataJSON().preferences.include_genres.includes("Comedy"));
  await page.getByRole("button", { name: "Find recommendations" }).click();
  const payload = (await request).postDataJSON();
  expect(payload.preferences.ratings[0].rating).toBe(4.5);
  expect(payload.preferences.exclude_genres).toContain("Thriller");
  await expect(page.getByText("Including Comedy; Avoiding Thriller.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Use microphone for recommendations" })).toBeVisible();
});

test("voice fills each mode without selecting, submitting, or saving automatically", async ({ page }) => {
  const recommendationRequests: unknown[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/recommendations")) recommendationRequests.push(request.postDataJSON());
  });
  await page.goto("/");
  await expect.poll(() => recommendationRequests.length).toBe(1);

  await page.getByRole("button", { name: "Use microphone for movie title" }).click();
  await page.evaluate(() => (window as any).__voiceTest.result(0, "Signal in the Fog"));
  await page.evaluate(() => (window as any).__voiceTest.end(0));
  const titleInput = page.getByRole("combobox", { name: "Search for a movie by title or release year" });
  await expect(titleInput).toHaveValue("Signal in the Fog");
  await expect(page.getByLabel("Movie search results").getByRole("button", { name: /Signal in the Fog/ }).first()).toBeVisible();
  await expect(page.getByRole("dialog", { name: /Rate Signal in the Fog/ })).toHaveCount(0);
  expect(await page.evaluate(() => localStorage.length)).toBe(0);
  expect(recommendationRequests).toHaveLength(1);

  await page.getByRole("button", { name: "Use microphone for movie title" }).click();
  await page.getByRole("tab", { name: "Find recommendations" }).click();
  const recommendationInput = page.getByRole("textbox", { name: "Describe the movies you want" });
  await recommendationInput.fill("typed request");
  await page.evaluate(() => (window as any).__voiceTest.result(1, "late title"));
  await expect(recommendationInput).toHaveValue("typed request");
  expect(await page.evaluate(() => (window as any).__voiceTest.wasAborted(1))).toBe(true);

  await page.getByRole("button", { name: "Use microphone for recommendations" }).click();
  await page.evaluate(() => (window as any).__voiceTest.result(2, "show comedy without thriller"));
  await page.evaluate(() => (window as any).__voiceTest.end(2));
  await expect(recommendationInput).toHaveValue("show comedy without thriller");
  expect(recommendationRequests).toHaveLength(1);
  expect(await page.evaluate(() => localStorage.length)).toBe(0);

  const submitted = page.waitForRequest((request) => request.url().endsWith("/recommendations") && request.postDataJSON().preferences.include_genres.includes("Comedy"));
  await page.getByRole("button", { name: "Find recommendations" }).click();
  expect((await submitted).postDataJSON().preferences.exclude_genres).toContain("Thriller");
  await page.getByRole("tab", { name: "Rate a movie" }).click();
  await expect(titleInput).toHaveValue("Signal in the Fog");
  expect(await page.evaluate(() => localStorage.length)).toBe(0);
});

test("title-based recommendation search uses temporary context without saving feedback", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("tab", { name: "Find recommendations" }).click();
  const input = page.getByRole("textbox", { name: "Describe the movies you want" });
  await input.fill("movies like Orbit House");
  const request = page.waitForRequest((candidate) => {
    if (!candidate.url().endsWith("/recommendations")) return false;
    return candidate.postDataJSON().temporary_history_items?.includes("102");
  });
  await page.getByRole("button", { name: "Find recommendations" }).click();
  const payload = (await request).postDataJSON();
  expect(payload.temporary_history_items).toEqual(["102"]);
  expect(payload.preferences.history_items).toEqual([]);
  expect(payload.preferences.ratings).toEqual([]);
  expect(await page.evaluate(() => localStorage.length)).toBe(0);
  await expect(page.getByText("Finding movies like Orbit House.")).toBeVisible();
});
