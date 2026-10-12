import type { components } from "./schema";

type GeneratedMovie = components["schemas"]["CatalogMovie"];
type GeneratedPreferences = components["schemas"]["PreferenceState"];
export type CatalogMovie = Omit<GeneratedMovie, "genres" | "poster_url"> & { genres: string[]; poster_url: string | null };
export type PreferenceState = Omit<GeneratedPreferences, "history_items" | "excluded_movie_ids" | "include_genres" | "exclude_genres"> & {
  history_items: string[];
  excluded_movie_ids: string[];
  include_genres: string[];
  exclude_genres: string[];
};
export type RecommendationRequest = components["schemas"]["RecommendationRequest"];
type GeneratedResponse = components["schemas"]["RecommendationResponse"];
type GeneratedItem = components["schemas"]["RecommendationItem"];
export type RecommendationResponse = Omit<GeneratedResponse, "applied_preferences" | "recommendations"> & {
  applied_preferences: PreferenceState;
  recommendations: Array<Omit<GeneratedItem, "movie"> & { movie: CatalogMovie }>;
};
export type RefinementResponse =
  | (Omit<components["schemas"]["AppliedRefinement"], "preferences"> & { preferences: PreferenceState })
  | components["schemas"]["ClarificationRefinement"]
  | components["schemas"]["UnsupportedRefinement"]
  | components["schemas"]["UnavailableRefinement"];
export type ApplicationStatus = components["schemas"]["ApplicationStatus"];
export type PosterMetadata = components["schemas"]["PosterMetadata"];

export class ApiError extends Error {
  constructor(message: string, public readonly status: number, public readonly code = "request_failed") {
    super(message);
  }
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers }
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(body?.error?.message ?? `Request failed (${response.status})`, response.status, body?.error?.code);
  }
  return body as T;
}

export async function searchCatalog(query: string, signal?: AbortSignal): Promise<{ items: CatalogMovie[]; total: number; backend_mode: string }> {
  const params = new URLSearchParams({ q: query, limit: "50" });
  return json(`/api/v1/catalog/search?${params}`, { signal });
}

export function getApplicationStatus(signal?: AbortSignal) {
  return json<ApplicationStatus>("/api/v1/application/status", { signal });
}

export function getPosters(movieIds: string[], signal?: AbortSignal) {
  return json<components["schemas"]["PosterResponse"]>("/api/v1/catalog/posters", {
    method: "POST",
    body: JSON.stringify({ movie_ids: movieIds.slice(0, 50) }),
    signal
  });
}

export function getRecommendations(request: RecommendationRequest, signal?: AbortSignal) {
  return json<RecommendationResponse>("/api/v1/recommendations", {
    method: "POST",
    body: JSON.stringify(request),
    signal
  });
}

export function refine(sessionId: string, preferenceRevision: number, preferences: PreferenceState, text: string) {
  return json<RefinementResponse>("/api/v1/refinements", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, preference_revision: preferenceRevision, preferences, text })
  });
}
