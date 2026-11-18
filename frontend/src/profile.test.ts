import { describe, expect, it } from "vitest";
import { loadBrowserProfile, removeRating, saveBrowserProfile, setRating, type BrowserProfile } from "./profile";

const movie = { movie_id: "1", title: "One (2001)", genres: ["Drama"], poster_url: null };
const empty = (): BrowserProfile => ({ version: 1, ratings: [], legacy_favorites: [], movies: {} });

describe("browser rating profile", () => {
  it("saves half-star ratings and no-op saves preserve the genuine timestamp", () => {
    const first = setRating(empty(), movie, 4, new Date("2026-11-17T12:00:00Z"));
    const same = setRating(first.profile, movie, 4, new Date("2026-11-18T12:00:00Z"));
    expect(same.changed).toBe(false);
    expect(same.profile.ratings[0].rated_at).toBe("2026-11-17T12:00:00.000Z");
    const edited = setRating(same.profile, movie, 5, new Date("2026-11-19T12:00:00Z"));
    expect(edited.profile.ratings).toEqual([{ movie_id: "1", rating: 5, rated_at: "2026-11-19T12:00:00.000Z" }]);
  });

  it("explicit ratings supersede legacy favorites and removal updates the active profile", () => {
    const profile = { ...empty(), legacy_favorites: ["1"] };
    const rated = setRating(profile, movie, 2.5).profile;
    expect(rated.legacy_favorites).toEqual([]);
    expect(removeRating(rated, "1").ratings).toEqual([]);
  });

  it("recovers valid records while ignoring malformed and duplicate older entries", () => {
    const storage = new Map<string, string>();
    const fake = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => { storage.set(key, value); } } as unknown as Storage;
    fake.setItem("trustworthy-recsys-browser-profile-v1", JSON.stringify({ version: 1, ratings: [
      { movie_id: "1", rating: 4, rated_at: "2026-11-17T10:00:00Z" },
      { movie_id: "1", rating: 5, rated_at: "2026-11-17T11:00:00Z" },
      { movie_id: "2", rating: 7, rated_at: "bad" },
    ], legacy_favorites: ["3", "3"] }));
    const loaded = loadBrowserProfile(fake);
    expect(loaded.profile.ratings).toEqual([{ movie_id: "1", rating: 5, rated_at: "2026-11-17T11:00:00Z" }]);
    expect(loaded.profile.legacy_favorites).toEqual(["3"]);
    expect(loaded.warning).not.toBeNull();
    expect(saveBrowserProfile(loaded.profile, fake)).toBe(true);
  });

  it("does not describe unavailable storage as session-only persistence", () => {
    const unavailable = { getItem: () => { throw new Error("blocked"); } } as unknown as Storage;
    const loaded = loadBrowserProfile(unavailable);
    expect(loaded.persistent).toBe(false);
    expect(loaded.warning).toBe("Browser storage is unavailable or unreadable. Ratings cannot be saved until storage is available.");
  });
});
