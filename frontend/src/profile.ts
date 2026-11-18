import type { CatalogMovie } from "./api/client";

export const PROFILE_STORAGE_KEY = "trustworthy-recsys-browser-profile-v1";
const LEGACY_KEYS = ["trustworthy-recsys-preferences", "trustworthy-recsys-profile"];

export type BrowserRating = { movie_id: string; rating: number; rated_at: string };
export type BrowserProfile = {
  version: 1;
  ratings: BrowserRating[];
  legacy_favorites: string[];
  movies: Record<string, CatalogMovie>;
};
export type LoadedProfile = { profile: BrowserProfile; persistent: boolean; warning: string | null };

export const EMPTY_PROFILE: BrowserProfile = { version: 1, ratings: [], legacy_favorites: [], movies: {} };

function validRating(value: unknown): value is BrowserRating {
  if (!value || typeof value !== "object") return false;
  const record = value as Partial<BrowserRating>;
  const timestamp = typeof record.rated_at === "string" ? Date.parse(record.rated_at) : NaN;
  return typeof record.movie_id === "string" && !!record.movie_id && !/\s/.test(record.movie_id)
    && typeof record.rating === "number" && Number.isFinite(record.rating)
    && record.rating >= .5 && record.rating <= 5 && Number.isInteger(record.rating * 2)
    && Number.isFinite(timestamp) && /(?:Z|[+-]00:00)$/.test(record.rated_at!);
}

function sanitizeMovies(value: unknown): Record<string, CatalogMovie> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(Object.entries(value).filter(([id, movie]) => {
    const item = movie as Partial<CatalogMovie>;
    return item && item.movie_id === id && typeof item.title === "string" && Array.isArray(item.genres);
  })) as Record<string, CatalogMovie>;
}

function sanitize(raw: unknown): BrowserProfile {
  const value = raw && typeof raw === "object" ? raw as Partial<BrowserProfile> : {};
  const byMovie = new Map<string, BrowserRating>();
  for (const candidate of Array.isArray(value.ratings) ? value.ratings : []) {
    if (!validRating(candidate)) continue;
    const current = byMovie.get(candidate.movie_id);
    if (!current || Date.parse(candidate.rated_at) >= Date.parse(current.rated_at)) byMovie.set(candidate.movie_id, candidate);
  }
  const legacy = Array.isArray(value.legacy_favorites)
    ? value.legacy_favorites.filter((id): id is string => typeof id === "string" && !!id && !/\s/.test(id))
    : [];
  return {
    version: 1,
    ratings: [...byMovie.values()],
    legacy_favorites: [...new Set(legacy)].filter((id) => !byMovie.has(id)),
    movies: sanitizeMovies(value.movies),
  };
}

export function loadBrowserProfile(storage?: Storage): LoadedProfile {
  try {
    const target = storage ?? window.localStorage;
    const saved = target.getItem(PROFILE_STORAGE_KEY);
    if (saved !== null) {
      const parsed = JSON.parse(saved);
      const profile = sanitize(parsed);
      return {
        profile,
        persistent: true,
        warning: JSON.stringify(profile) === JSON.stringify(parsed) ? null : "Some unreadable saved profile entries were ignored.",
      };
    }
    for (const key of LEGACY_KEYS) {
      const legacy = target.getItem(key);
      if (legacy === null) continue;
      const parsed = JSON.parse(legacy);
      const history = Array.isArray(parsed?.history_items) ? parsed.history_items : [];
      return {
        profile: sanitize({ legacy_favorites: history, movies: parsed?.movies }),
        persistent: true,
        warning: history.length ? "Existing favorites were preserved as unrated preferences." : null,
      };
    }
    return { profile: { ...EMPTY_PROFILE, ratings: [], legacy_favorites: [], movies: {} }, persistent: true, warning: null };
  } catch {
    return {
      profile: { ...EMPTY_PROFILE, ratings: [], legacy_favorites: [], movies: {} },
      persistent: false,
      warning: "Browser storage is unavailable or unreadable. Ratings cannot be saved until storage is available.",
    };
  }
}

export function saveBrowserProfile(profile: BrowserProfile, storage?: Storage): boolean {
  try {
    const target = storage ?? window.localStorage;
    target.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profile));
    return target.getItem(PROFILE_STORAGE_KEY) === JSON.stringify(profile);
  } catch {
    return false;
  }
}

export function setRating(profile: BrowserProfile, movie: CatalogMovie, rating: number, now = new Date()): { profile: BrowserProfile; changed: boolean } {
  if (!Number.isFinite(rating) || rating < .5 || rating > 5 || !Number.isInteger(rating * 2)) throw new Error("Rating must be 0.5–5 in half-star increments.");
  const existing = profile.ratings.find((record) => record.movie_id === movie.movie_id);
  if (existing?.rating === rating) return { profile, changed: false };
  const record = { movie_id: movie.movie_id, rating, rated_at: now.toISOString() };
  return {
    changed: true,
    profile: {
      ...profile,
      ratings: [...profile.ratings.filter((item) => item.movie_id !== movie.movie_id), record],
      legacy_favorites: profile.legacy_favorites.filter((id) => id !== movie.movie_id),
      movies: { ...profile.movies, [movie.movie_id]: movie },
    },
  };
}

export function removeRating(profile: BrowserProfile, movieId: string): BrowserProfile {
  return { ...profile, ratings: profile.ratings.filter((record) => record.movie_id !== movieId) };
}
