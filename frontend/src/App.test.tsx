import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App, formatCatalogTitle } from "./App";
import type { PreferenceState } from "./api/client";
import { PROFILE_STORAGE_KEY } from "./profile";

const MOVIES = [
  { movie_id: "101", title: "Signal in the Fog (2004)", genres: ["Mystery", "Thriller"], poster_url: null },
  { movie_id: "102", title: "Orbit House (2011)", genres: ["Drama", "Science Fiction"], poster_url: null },
  { movie_id: "103", title: "The Last Matinee (1999)", genres: ["Comedy", "Drama"], poster_url: null },
];
const MATRIX = { movie_id: "2571", title: "Matrix, The (1999)", genres: ["Action", "Science Fiction"], poster_url: null };
const STAR_WARS = { movie_id: "260", title: "Star Wars: Episode IV - A New Hope (1977)", genres: ["Action", "Adventure", "Science Fiction"], poster_url: null };
const HIDDEN_MOVIES = [
  { movie_id: "104", title: "Hidden Harbor (2005)", genres: ["Drama"], poster_url: null },
  { movie_id: "105", title: "Second Signal (2012)", genres: ["Comedy"], poster_url: null },
];
const EMPTY: PreferenceState = { history_items: [], excluded_movie_ids: [], include_genres: [], exclude_genres: [], ratings: [] };
const STATUS = { backend_mode: "fixture", model_label: "Synthetic demo", data_source_label: "Synthetic fixture catalog", refinement_available: true, poster_provider: "tmdb", posters_configured: false, available_genres: ["Comedy", "Drama", "Mystery", "Science Fiction", "Thriller"], feedback_mode: "off" };

function rec(revision: number, preferences = EMPTY, movies = MOVIES) {
  return { request_id: `request-${revision}-${preferences.ratings?.[0]?.rating ?? "empty"}`, preference_revision: revision,
    recommendations: movies.map((movie, index) => ({ rank: index + 1, movie, retrieval_score: null, ranking_score: null })),
    applied_preferences: preferences, retrieval: { name: "retrieval", version: "1", mode: "fixture" },
    reranker: { name: "pass-through", version: "1", mode: "pass_through" }, backend_mode: "fixture", partial: false,
    warnings: [], timings: { retrieval_ms: 1, filtering_ms: 1, reranking_ms: 1, total_ms: 3 } };
}
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

function mockApi(recommend?: (body: any) => Promise<Response> | Response) {
  return vi.fn((url: string | URL | Request, init?: RequestInit) => {
    const path = String(url);
    if (path.includes("application/status")) return json(STATUS);
    if (path.includes("catalog/search")) return json({ items: MOVIES, total: MOVIES.length, backend_mode: "fixture" });
    if (path.includes("catalog/posters")) return json({ provider: "tmdb", configured: false, items: [] });
    if (path.includes("refinements")) {
      const body = JSON.parse(String(init?.body));
      if (String(body.text).includes("movies like")) {
        const resolved = String(body.text).includes("The Matrix") ? MATRIX : String(body.text).includes("Star Wars") ? STAR_WARS : MOVIES[1];
        return json({ outcome: "applied", preference_revision: body.preference_revision + 1, preferences: body.preferences, summary: `Finding movies like ${resolved.title}.`, resolved_movies: [resolved] });
      }
      return json({ outcome: "applied", preference_revision: body.preference_revision + 1, preferences: { ...body.preferences, include_genres: ["Comedy"] }, summary: "Including Comedy.", resolved_movies: [] });
    }
    const body = JSON.parse(String(init?.body));
    return recommend?.(body) ?? json(rec(body.preference_revision, body.preferences));
  });
}

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); localStorage.clear(); sessionStorage.clear(); });

type MockRecognition = {
  start: ReturnType<typeof vi.fn>; abort: ReturnType<typeof vi.fn>; lang: string; continuous: boolean; interimResults: boolean;
  onresult: null | ((event: { results: { transcript: string }[][] }) => void);
  onerror: null | ((event: { error: string }) => void); onend: null | (() => void);
};

function installSpeechRecognition() {
  const instances: MockRecognition[] = [];
  vi.stubGlobal("SpeechRecognition", class {
    constructor() {
      const instance: MockRecognition = { start: vi.fn(), abort: vi.fn(), lang: "", continuous: true, interimResults: true,
        onresult: null, onerror: null, onend: null };
      instances.push(instance);
      return instance;
    }
  });
  return instances;
}

async function searchAndOpenRating(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByRole("combobox", { name: "Search for a movie by title or release year" }), "Signal");
  const results = await screen.findByLabelText("Movie search results");
  await user.click(await within(results).findByRole("button", { name: /Signal in the Fog/ }));
  return screen.getByRole("dialog", { name: "Rate Signal in the Fog" });
}

async function saveCurrentRating(user: ReturnType<typeof userEvent.setup>, value: string) {
  const dialog = screen.getByRole("dialog", { name: /Rate Signal in the Fog/ });
  await user.selectOptions(within(dialog).getByLabelText("Your rating"), value);
  await user.click(within(dialog).getByRole("button", { name: "Save rating" }));
}

describe("browser rating workflow", () => {
  it("hydrates an explicit empty profile before the first request", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" });
    const calls = fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0][1]?.body)).preferences).toEqual(EMPTY);
  });

  it("search and selection save nothing until explicit Save rating", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" });
    const dialog = await searchAndOpenRating(user);
    expect(localStorage.getItem(PROFILE_STORAGE_KEY)).toBeNull();
    expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(1);
    expect(within(dialog).getByLabelText("Your rating")).toHaveValue("4");
    await saveCurrentRating(user, "2.5");
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(2));
    const request = JSON.parse(String(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))[1][1]?.body));
    expect(request.preferences.ratings[0]).toMatchObject({ movie_id: "101", rating: 2.5 });
    expect(screen.queryByRole("dialog", { name: "Rate Signal in the Fog" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Manage ratings" })).toBeInTheDocument();
  });

  it("reload sends the restored profile in the first request", async () => {
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify({ version: 1, ratings: [{ movie_id: "101", rating: 4.5, rated_at: "2026-11-17T12:00:00.000Z" }], legacy_favorites: [], movies: { "101": MOVIES[0] } }));
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); render(<App />);
    await screen.findByRole("button", { name: "Manage ratings" });
    const call = fetch.mock.calls.find(([url]) => String(url).includes("recommendations"))!;
    expect(JSON.parse(String(call[1]?.body)).preferences.ratings).toEqual([{ movie_id: "101", rating: 4.5, rated_at: "2026-11-17T12:00:00.000Z" }]);
  });

  it("a 4-to-5 edit triggers a revision while keeping one record", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" }); await searchAndOpenRating(user); await saveCurrentRating(user, "4");
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(2));
    await user.click(screen.getByRole("button", { name: "Manage ratings" }));
    const ratings = screen.getByRole("dialog", { name: /Your ratings/ });
    await user.click(within(ratings).getByRole("button", { name: "Edit" })); await saveCurrentRating(user, "5");
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(3));
    const calls = fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    const latest = JSON.parse(String(calls.at(-1)![1]?.body));
    expect(calls).toHaveLength(3); expect(latest.preferences.ratings).toHaveLength(1);
    expect(latest.preferences.ratings[0].rating).toBe(5); expect(latest.preference_revision).toBe(2);
  });

  it("an identical save preserves timestamp and sends no request", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" }); await searchAndOpenRating(user); await saveCurrentRating(user, "4");
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(2));
    const before = JSON.parse(localStorage.getItem(PROFILE_STORAGE_KEY)!).ratings[0].rated_at;
    await user.click(screen.getByRole("button", { name: "Manage ratings" }));
    await user.click(within(screen.getByRole("dialog", { name: /Your ratings/ })).getByRole("button", { name: "Edit" }));
    await user.click(screen.getByRole("button", { name: "Save rating" }));
    expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(2);
    expect(JSON.parse(localStorage.getItem(PROFILE_STORAGE_KEY)!).ratings[0].rated_at).toBe(before);
    expect(screen.getByText("Rating saved.")).toBeInTheDocument();
  });

  it("removing a rating persists and refreshes", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" }); await searchAndOpenRating(user); await saveCurrentRating(user, "3");
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(2));
    await user.click(screen.getByRole("button", { name: "Manage ratings" }));
    await user.click(within(screen.getByRole("dialog", { name: /Your ratings/ })).getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Manage ratings" })).not.toBeInTheDocument());
    const calls = fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    expect(JSON.parse(String(calls.at(-1)![1]?.body)).preferences.ratings).toEqual([]);
  });

  it("clear saved ratings is distinct and refreshes with an empty profile", async () => {
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify({ version: 1, ratings: [
      { movie_id: "101", rating: 4.5, rated_at: "2026-11-17T12:00:00.000Z" },
    ], legacy_favorites: [], movies: { "101": MOVIES[0] } }));
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("button", { name: "Manage ratings" });
    await user.click(screen.getByRole("button", { name: "Manage ratings" }));
    await user.click(within(screen.getByRole("dialog", { name: /Your ratings/ })).getByRole("button", { name: "Clear saved ratings" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Manage ratings" })).not.toBeInTheDocument());
    const calls = fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    expect(JSON.parse(String(calls.at(-1)![1]?.body)).preferences.ratings).toEqual([]);
  });

  it("genre changes preserve ratings and keep the dialog open", async () => {
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify({ version: 1, ratings: [{ movie_id: "101", rating: 4.5, rated_at: "2026-11-17T12:00:00.000Z" }], legacy_favorites: [], movies: { "101": MOVIES[0] } }));
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />); await screen.findByRole("button", { name: "Manage ratings" });
    await user.click(screen.getByRole("button", { name: "Preferences" }));
    const panel = screen.getByRole("dialog", { name: "Recommendation preferences" });
    await user.click(within(panel).getByRole("button", { name: "Include genres" }));
    const selector = screen.getByRole("dialog", { name: "Include genres" }); await user.click(within(selector).getByRole("button", { name: "Drama" }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(2));
    expect(screen.getByRole("dialog", { name: "Include genres" })).toBeInTheDocument();
    const request = JSON.parse(String(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))[1][1]?.body));
    expect(request.preferences.include_genres).toEqual(["Drama"]); expect(request.preferences.ratings[0].rating).toBe(4.5);
  });

  it("rapid edits cannot display a stale response", async () => {
    let resolveFour!: (response: Response) => void;
    const fetch = mockApi((body) => { const rating = body.preferences.ratings?.[0]?.rating;
      if (rating === 4) return new Promise<Response>((resolve) => { resolveFour = resolve; });
      return json(rec(body.preference_revision, body.preferences, rating === 5 ? [MOVIES[2]] : MOVIES)); });
    vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Signal in the Fog", level: 1 });
    await searchAndOpenRating(user); await saveCurrentRating(user, "4");
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(2));
    await user.click(screen.getByRole("button", { name: "Manage ratings" }));
    await user.click(within(screen.getByRole("dialog", { name: /Your ratings/ })).getByRole("button", { name: "Edit" }));
    await saveCurrentRating(user, "5");
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(3));
    await user.click(screen.getByRole("button", { name: "Close ratings" }));
    expect(await screen.findByRole("heading", { name: "The Last Matinee", level: 1 })).toBeInTheDocument();
    resolveFour(await json(rec(1, { ...EMPTY, ratings: [{ movie_id: "101", rating: 4, rated_at: "2026-11-17T12:00:00Z" }] }, [MOVIES[1]])));
    await new Promise((resolve) => setTimeout(resolve, 0)); expect(screen.getByRole("heading", { name: "The Last Matinee", level: 1 })).toBeInTheDocument();
  });

  it("failed refresh preserves the rating and Retry sends it again", async () => {
    let count = 0; const fetch = mockApi((body) => { count += 1; if (count === 2) return json({ error: { message: "Temporary failure" } }, 503); return json(rec(body.preference_revision, body.preferences)); });
    vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Explore movies" });
    await searchAndOpenRating(user); await saveCurrentRating(user, "3.5");
    const notice = (await screen.findByText("Rating saved. Recommendations couldn’t update.")).closest<HTMLElement>("[role=alert]")!;
    await user.click(within(notice).getByRole("button", { name: "Retry" })); await waitFor(() => expect(count).toBe(3));
    const calls = fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    expect(JSON.parse(String(calls.at(-1)![1]?.body)).preferences.ratings[0].rating).toBe(3.5);
  });

  it("shows pending and complete save messages from the actual refresh state", async () => {
    let finishRefresh!: (response: Response) => void;
    let pendingBody: any;
    const fetch = mockApi((body) => body.preference_revision === 0
      ? json(rec(0, body.preferences))
      : new Promise<Response>((resolve) => { pendingBody = body; finishRefresh = resolve; }));
    vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" });
    await searchAndOpenRating(user); await saveCurrentRating(user, "4.5");
    expect(screen.queryByRole("dialog", { name: "Rate Signal in the Fog" })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Rating saved. Updating recommendations…");
    finishRefresh(await json(rec(1, pendingBody.preferences)));
    expect(await screen.findByText("Rating saved.")).toBeInTheDocument();
    expect(screen.queryByText(/Rating saved. Updating recommendations/)).not.toBeInTheDocument();
  });

  it("keeps the rating dialog open when browser persistence fails", async () => {
    const original = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (this: Storage, key, value) {
      if (key === PROFILE_STORAGE_KEY) throw new DOMException("blocked");
      return original.call(this, key, value);
    });
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" });
    const dialog = await searchAndOpenRating(user);
    await user.selectOptions(within(dialog).getByLabelText("Your rating"), "4.5");
    await user.click(within(dialog).getByRole("button", { name: "Save rating" }));
    expect(dialog).toBeInTheDocument();
    expect(within(dialog).getByRole("alert")).toHaveTextContent("couldn’t save this rating");
    expect(within(dialog).getByLabelText("Your rating")).toHaveValue("4.5");
    expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(1);
  });

  it("keeps title and recommendation drafts separate and sends title context without saving feedback", async () => {
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify({ version: 1, ratings: [{ movie_id: "101", rating: 4.5, rated_at: "2026-11-17T12:00:00.000Z" }], legacy_favorites: [], movies: { "101": MOVIES[0] } }));
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("button", { name: "Manage ratings" });
    const titleInput = screen.getByRole("combobox", { name: "Search for a movie by title or release year" });
    await user.type(titleInput, "Signal");
    await user.click(screen.getByRole("tab", { name: "Find recommendations" }));
    const recommendationInput = screen.getByRole("textbox", { name: "Describe the movies you want" });
    expect(screen.getByRole("button", { name: "Use microphone for recommendations" })).toBeInTheDocument();
    await user.type(recommendationInput, "movies like Orbit House");
    await user.click(screen.getByRole("button", { name: "Find recommendations" }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(2));
    const request = JSON.parse(String(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations")).at(-1)![1]?.body));
    expect(request.temporary_history_items).toEqual(["102"]);
    expect(request.preferences.ratings[0].rating).toBe(4.5);
    expect(request.preferences.history_items).toEqual([]);
    expect(JSON.parse(localStorage.getItem(PROFILE_STORAGE_KEY)!).ratings).toHaveLength(1);
    await user.click(screen.getByRole("tab", { name: "Rate a movie" }));
    expect(screen.getByRole("combobox", { name: "Search for a movie by title or release year" })).toHaveValue("Signal");
  });

  it("Undo restores an empty temporary title context and advances the request revision", async () => {
    const ratedAt = "2026-11-17T12:00:00.000Z";
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify({ version: 1, ratings: [{ movie_id: "101", rating: 4.5, rated_at: ratedAt }], legacy_favorites: [], movies: { "101": MOVIES[0] } }));
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("button", { name: "Manage ratings" });
    await user.click(screen.getByRole("tab", { name: "Find recommendations" }));
    const input = screen.getByRole("textbox", { name: "Describe the movies you want" });
    await user.type(input, "movies like The Matrix");
    await user.click(screen.getByRole("button", { name: "Find recommendations" }));
    const recommendationCalls = () => fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(2));
    expect(JSON.parse(String(recommendationCalls()[1][1]?.body))).toMatchObject({ preference_revision: 1, temporary_history_items: ["2571"] });

    await user.click(screen.getByRole("button", { name: "Preferences" }));
    await user.click(within(screen.getByRole("dialog", { name: "Recommendation preferences" })).getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(3));
    const restored = JSON.parse(String(recommendationCalls()[2][1]?.body));
    expect(restored).toMatchObject({ preference_revision: 2, temporary_history_items: [] });
    expect(restored.preferences.ratings).toEqual([{ movie_id: "101", rating: 4.5, rated_at: ratedAt }]);
    expect(JSON.parse(localStorage.getItem(PROFILE_STORAGE_KEY)!).ratings).toEqual([{ movie_id: "101", rating: 4.5, rated_at: ratedAt }]);
  });

  it("Undo restores the preceding temporary title context", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" });
    await user.click(screen.getByRole("tab", { name: "Find recommendations" }));
    const input = screen.getByRole("textbox", { name: "Describe the movies you want" });
    const recommendationCalls = () => fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));

    await user.type(input, "movies like The Matrix");
    await user.click(screen.getByRole("button", { name: "Find recommendations" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(2));
    await user.clear(input); await user.type(input, "movies like Star Wars");
    await user.click(screen.getByRole("button", { name: "Find recommendations" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(3));
    expect(JSON.parse(String(recommendationCalls()[2][1]?.body))).toMatchObject({ preference_revision: 2, temporary_history_items: ["260"] });

    await user.click(screen.getByRole("button", { name: "Preferences" }));
    await user.click(within(screen.getByRole("dialog", { name: "Recommendation preferences" })).getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(4));
    expect(JSON.parse(String(recommendationCalls()[3][1]?.body))).toMatchObject({ preference_revision: 3, temporary_history_items: ["2571"] });
  });

  it("hides, unhides, and undoes visit-only exclusions without changing other personalization", async () => {
    const ratedAt = "2026-11-17T12:00:00.000Z";
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify({ version: 1, ratings: [{ movie_id: "101", rating: 4.5, rated_at: ratedAt }], legacy_favorites: ["103"], movies: { "101": MOVIES[0], "103": MOVIES[2] } }));
    const allMovies = [...MOVIES, ...HIDDEN_MOVIES];
    const fetch = mockApi((body) => json(rec(body.preference_revision, body.preferences, allMovies.filter((movie) => !body.preferences.excluded_movie_ids.includes(movie.movie_id)))));
    vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("button", { name: "Manage ratings" });
    const recommendationCalls = () => fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    const requestAt = (index: number) => JSON.parse(String(recommendationCalls()[index][1]?.body));
    const expectPreserved = (request: any, excluded: string[], revision: number) => {
      expect(request).toMatchObject({ preference_revision: revision, temporary_history_items: ["102"], preferences: {
        excluded_movie_ids: excluded, include_genres: ["Comedy"], history_items: ["103"],
        ratings: [{ movie_id: "101", rating: 4.5, rated_at: ratedAt }],
      } });
    };

    await user.click(screen.getByRole("tab", { name: "Find recommendations" }));
    const refinement = screen.getByRole("textbox", { name: "Describe the movies you want" });
    await user.type(refinement, "show comedy"); await user.click(screen.getByRole("button", { name: "Find recommendations" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(2));
    await user.clear(refinement); await user.type(refinement, "movies like Orbit House"); await user.click(screen.getByRole("button", { name: "Find recommendations" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(3));
    await user.click(screen.getByRole("tab", { name: "Rate a movie" }));

    await user.click(screen.getByRole("button", { name: "Hide Hidden Harbor (2005) for now" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(4));
    expectPreserved(requestAt(3), ["104"], 3);
    await user.click(screen.getByRole("button", { name: "Preferences" }));
    let preferences = screen.getByRole("dialog", { name: "Recommendation preferences" });
    expect(within(preferences).getByRole("heading", { name: /Hidden this visit 1/ })).toBeInTheDocument();
    expect(within(preferences).getByText("Hidden Harbor")).toBeInTheDocument();
    expect(within(preferences).getByText("2005")).toBeInTheDocument();
    await user.click(within(preferences).getByRole("button", { name: "Close preferences" }));

    await user.click(screen.getByRole("button", { name: "Hide Second Signal (2012) for now" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(5));
    expectPreserved(requestAt(4), ["104", "105"], 4);
    await user.click(screen.getByRole("button", { name: "Preferences" }));
    preferences = screen.getByRole("dialog", { name: "Recommendation preferences" });
    await user.click(within(preferences).getByRole("button", { name: "Unhide Hidden Harbor" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(6));
    expectPreserved(requestAt(5), ["105"], 5);
    expect(within(preferences).queryByText("Hidden Harbor")).not.toBeInTheDocument();
    expect(within(preferences).getByText("Second Signal")).toBeInTheDocument();

    await user.click(within(preferences).getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(7));
    expectPreserved(requestAt(6), ["104", "105"], 6);
    await user.click(within(preferences).getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(8));
    expectPreserved(requestAt(7), ["104"], 7);
    await user.click(within(preferences).getByRole("button", { name: "Unhide Hidden Harbor" }));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(9));
    expectPreserved(requestAt(8), [], 8);
    expect(within(preferences).queryByText("Hidden this visit")).not.toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem(PROFILE_STORAGE_KEY)!)).toMatchObject({ ratings: [{ movie_id: "101", rating: 4.5, rated_at: ratedAt }], legacy_favorites: ["103"] });
    expect(localStorage.getItem(PROFILE_STORAGE_KEY)).not.toContain("excluded_movie_ids");
  });

  it("reload clears hidden exclusions while restoring saved ratings and favorites", async () => {
    const ratedAt = "2026-11-17T12:00:00.000Z";
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify({ version: 1, ratings: [{ movie_id: "101", rating: 4.5, rated_at: ratedAt }], legacy_favorites: ["103"], movies: { "101": MOVIES[0], "103": MOVIES[2] } }));
    const allMovies = [...MOVIES, ...HIDDEN_MOVIES];
    const fetch = mockApi((body) => json(rec(body.preference_revision, body.preferences, allMovies.filter((movie) => !body.preferences.excluded_movie_ids.includes(movie.movie_id)))));
    vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); const first = render(<App />);
    await screen.findByRole("button", { name: "Hide Hidden Harbor (2005) for now" });
    await user.click(screen.getByRole("button", { name: "Hide Hidden Harbor (2005) for now" }));
    const recommendationCalls = () => fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    await waitFor(() => expect(recommendationCalls()).toHaveLength(2));
    expect(JSON.parse(String(recommendationCalls()[1][1]?.body)).preferences.excluded_movie_ids).toEqual(["104"]);

    first.unmount(); render(<App />);
    await waitFor(() => expect(recommendationCalls()).toHaveLength(3));
    const restored = JSON.parse(String(recommendationCalls()[2][1]?.body));
    expect(restored.preferences).toMatchObject({ excluded_movie_ids: [], history_items: ["103"], ratings: [{ movie_id: "101", rating: 4.5, rated_at: ratedAt }] });
    await user.click(screen.getByRole("button", { name: "Preferences" }));
    expect(within(screen.getByRole("dialog", { name: "Recommendation preferences" })).queryByText("Hidden this visit")).not.toBeInTheDocument();
    expect(localStorage.getItem(PROFILE_STORAGE_KEY)).not.toContain("excluded_movie_ids");
  });

  it("fills both search drafts from speech without selecting, submitting, or accepting a late cross-mode result", async () => {
    const recognition = installSpeechRecognition();
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" });
    const recommendationCalls = () => fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    expect(recommendationCalls()).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "Use microphone for movie title" }));
    const titleSession = recognition[0];
    act(() => titleSession.onresult?.({ results: [[{ transcript: "Signal in the Fog" }]] }));
    act(() => titleSession.onend?.());
    expect(screen.getByRole("combobox", { name: "Search for a movie by title or release year" })).toHaveValue("Signal in the Fog");
    expect(await screen.findByLabelText("Movie search results")).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /Rate Signal in the Fog/ })).not.toBeInTheDocument();
    expect(localStorage.getItem(PROFILE_STORAGE_KEY)).toBeNull();
    expect(recommendationCalls()).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: "Use microphone for movie title" }));
    const interruptedTitleSession = recognition[1];
    const lateTitleResult = interruptedTitleSession.onresult!;
    await user.click(screen.getByRole("tab", { name: "Find recommendations" }));
    expect(interruptedTitleSession.abort).toHaveBeenCalledOnce();
    await user.type(screen.getByRole("textbox", { name: "Describe the movies you want" }), "typed request");
    act(() => lateTitleResult({ results: [[{ transcript: "late title" }]] }));
    expect(screen.getByRole("textbox", { name: "Describe the movies you want" })).toHaveValue("typed request");

    await user.click(screen.getByRole("button", { name: "Use microphone for recommendations" }));
    const recommendationSession = recognition[2];
    act(() => recommendationSession.onresult?.({ results: [[{ transcript: "show comedy without horror" }]] }));
    expect(screen.getByRole("textbox", { name: "Describe the movies you want" })).toHaveValue("show comedy without horror");
    expect(recommendationCalls()).toHaveLength(1);
    expect(localStorage.getItem(PROFILE_STORAGE_KEY)).toBeNull();

    act(() => recommendationSession.onerror?.({ error: "not-allowed" }));
    expect(screen.getByRole("textbox", { name: "Describe the movies you want" })).toHaveValue("show comedy without horror");
    expect(screen.getByRole("status")).toHaveTextContent(/denied.*type your request/i);
    await user.click(screen.getByRole("tab", { name: "Rate a movie" }));
    expect(screen.getByRole("combobox", { name: "Search for a movie by title or release year" })).toHaveValue("Signal in the Fog");
  });

  it("a completed older refresh cannot close a newly opened rating dialog", async () => {
    let finishRefresh!: (response: Response) => void; let pendingBody: any;
    const fetch = mockApi((body) => body.preference_revision === 0 ? json(rec(0, body.preferences))
      : new Promise<Response>((resolve) => { pendingBody = body; finishRefresh = resolve; }));
    vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" });
    await searchAndOpenRating(user); await saveCurrentRating(user, "4.5");
    await user.click(screen.getByRole("button", { name: "Rate" }));
    const newerDialog = screen.getByRole("dialog", { name: /Rate Signal in the Fog/ });
    finishRefresh(await json(rec(1, pendingBody.preferences)));
    await waitFor(() => expect(newerDialog).toBeInTheDocument());
  });

  it("hides empty unrated favorites and keeps recommendation search out of Preferences", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await screen.findByRole("heading", { name: "Explore movies" });
    await user.click(screen.getByRole("button", { name: "Preferences" }));
    const preferences = screen.getByRole("dialog", { name: "Recommendation preferences" });
    expect(within(preferences).queryByText("Unrated favorites")).not.toBeInTheDocument();
    expect(within(preferences).queryByRole("textbox", { name: "Describe the movies you want" })).not.toBeInTheDocument();
  });

  it("opening details never saves feedback", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    await user.click(await screen.findByRole("button", { name: `View details for ${MOVIES[0].title}` }));
    expect(screen.getByRole("dialog", { name: "Signal in the Fog" })).toBeInTheDocument();
    expect(localStorage.getItem(PROFILE_STORAGE_KEY)).toBeNull(); expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(1);
  });

  it("search results support arrow navigation and Escape dismissal", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    const input = screen.getByRole("combobox", { name: "Search for a movie by title or release year" });
    await user.type(input, "Signal");
    const results = await screen.findByLabelText("Movie search results");
    await within(results).findAllByRole("button");
    await user.keyboard("{ArrowDown}");
    expect(within(results).getAllByRole("button")[0]).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    expect(within(results).getAllByRole("button")[1]).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(input).toHaveFocus();
    expect(screen.queryByLabelText("Movie search results")).not.toBeInTheDocument();
  });

  it("formats catalog articles and years", () => {
    expect(formatCatalogTitle("Princess Bride, The (1987)")).toEqual({ title: "The Princess Bride", year: "1987" });
    expect(formatCatalogTitle("A Very Long, Uncertain Title (2004)")).toEqual({ title: "A Very Long, Uncertain Title", year: "2004" });
  });
});
