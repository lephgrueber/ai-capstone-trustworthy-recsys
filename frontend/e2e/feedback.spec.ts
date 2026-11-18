import { expect, test } from "@playwright/test";

test("real feedback endpoint persists hero events while the grid stays separate", async ({ page }) => {
  const accepted: Array<{ kind: string; decision_id: string; movie_id: string }> = [];
  page.on("response", async (response) => {
    if (response.url().endsWith("/feedback/events") && response.ok()) {
      accepted.push(response.request().postDataJSON());
    }
  });
  const responsePromise = page.waitForResponse((r) => r.url().endsWith("/recommendations") && r.ok());
  await page.goto("/");
  const response = await (await responsePromise).json();
  expect(response.feedback.synthetic).toBe(true);
  await expect(page.getByText(/featured-recommendation feedback pilot is active/i)).toBeVisible();
  await expect.poll(() => accepted.some((e) => e.kind === "heartbeat")).toBe(true);
  const hero = page.locator(".feature-hero");
  await hero.scrollIntoViewIfNeeded();
  await expect.poll(() => accepted.some((e) => e.kind === "display")).toBe(true);
  await hero.getByRole("button", { name: "Details", exact: true }).click();
  await expect.poll(() => accepted.some((e) => e.kind === "details_open")).toBe(true);
  const details = accepted.find((e) => e.kind === "details_open")!;
  expect(details.decision_id).toBe(response.feedback.decision_id);
  expect(details.movie_id).toBe(response.feedback.movie_id);
  expect(response.recommendations.slice(1).every((item: { movie: { movie_id: string } }) => item.movie.movie_id !== response.feedback.movie_id)).toBe(true);
  await page.getByRole("button", { name: "Close details" }).click();
  await page.locator(".portrait-card").first().getByRole("button", { name: /^Details for/ }).click();
  await page.waitForTimeout(250);
  expect(accepted.filter((e) => e.kind === "details_open")).toHaveLength(1);

  await page.getByRole("button", { name: "Close details" }).click();
  const eventCount = accepted.length;
  await hero.getByRole("button", { name: "Rate", exact: true }).click();
  const rating = page.getByRole("dialog", { name: /Rate / });
  await rating.getByLabel("Your rating").selectOption("4.5");
  await rating.getByRole("button", { name: "Save rating" }).click();
  await page.waitForTimeout(250);
  expect(accepted.slice(eventCount).every((event) => event.kind !== "favorite_add" && event.kind !== "hide")).toBe(true);
});
