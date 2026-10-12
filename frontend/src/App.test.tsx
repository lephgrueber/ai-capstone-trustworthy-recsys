import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App, formatCatalogTitle } from "./App";

const MOVIES = [
  { movie_id: "101", title: "Signal in the Fog", genres: ["Mystery", "Thriller"], poster_url: null },
  { movie_id: "102", title: "Orbit House", genres: ["Drama", "Science Fiction"], poster_url: null },
  { movie_id: "103", title: "The Last Matinee", genres: ["Comedy", "Drama"], poster_url: null }
];
const EMPTY = { history_items: [], excluded_movie_ids: [], include_genres: [], exclude_genres: [] };
const STATUS = { backend_mode: "fixture", model_label: "Synthetic demo", data_source_label: "Synthetic fixture catalog", refinement_available: false, poster_provider: "tmdb", posters_configured: false, available_genres: ["Comedy", "Drama", "Mystery", "Science Fiction", "Thriller"] };

function rec(revision: number, preferences = EMPTY, movies = MOVIES, mode = "fixture") {
  return { request_id: `request-${revision}`, preference_revision: revision, recommendations: movies.map((movie, index) => ({ rank: index + 1, movie, retrieval_score: null, ranking_score: null })), applied_preferences: preferences, retrieval: { name: "retrieval", version: "1", mode }, reranker: { name: "pass-through", version: "1", mode: "pass_through" }, backend_mode: mode, partial: false, warnings: [], timings: { retrieval_ms: 1, filtering_ms: 1, reranking_ms: 1, total_ms: 3 } };
}
const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

function mockApi(options: { status?: typeof STATUS; artwork?: boolean; recommend?: (revision: number, preferences: typeof EMPTY) => unknown } = {}) {
  return vi.fn((url: string | URL | Request, init?: RequestInit) => {
    const path = String(url);
    if (path.includes("application/status")) return json(options.status ?? STATUS);
    if (path.includes("catalog/search")) return json({ items: MOVIES, total: 3, backend_mode: "fixture" });
    if (path.includes("catalog/posters")) {
      const ids = JSON.parse(String(init?.body)).movie_ids as string[];
      return json({ provider: "tmdb", configured: !!options.artwork, items: ids.map((id) => ({ movie_id: id, tmdb_id: id, poster_url: options.artwork ? `https://image.tmdb.org/t/p/w500/${id}.jpg` : null, backdrop_url: options.artwork ? `https://image.tmdb.org/t/p/w1280/${id}.jpg` : null, status: options.artwork ? "available" : "missing" })) });
    }
    const body = JSON.parse(String(init?.body));
    return json(options.recommend?.(body.preference_revision, body.preferences) ?? rec(body.preference_revision, body.preferences));
  });
}

afterEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });

async function drawer(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /Personalize/ }));
  return screen.getByRole("dialog", { name: "Personalize your recommendations" });
}
async function genre(user: ReturnType<typeof userEvent.setup>, kind: "Include" | "Avoid", value: string) {
  const panel = screen.getByRole("dialog", { name: "Personalize your recommendations" });
  await user.click(within(panel).getByRole("button", { name: new RegExp(`${kind} genres`) }));
  const selector = screen.getByRole("dialog", { name: `${kind} genres` });
  await user.click(within(selector).getByRole("button", { name: value }));
  await user.click(within(selector).getByRole("button", { name: "Done" }));
}

describe("movie-first discovery", () => {
  it("loads empty-history recommendations once and focuses drawer search", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />);
    expect(await screen.findByRole("heading", { name: "Explore movies" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Signal in the Fog", level: 1 })).toBeInTheDocument();
    expect(document.querySelector(".personalization-summary")).not.toBeInTheDocument();
    expect(within(screen.getByRole("banner")).getByRole("link", { name: "Trustworthy Recommendation System" })).toBeInTheDocument(); expect(within(screen.getByRole("banner")).getByRole("button", { name: "Personalize" })).toBeInTheDocument(); expect(within(screen.getByRole("banner")).getByRole("button", { name: "About" })).toBeInTheDocument();
    expect(screen.getByRole("banner")).not.toHaveTextContent("Synthetic demo");
    expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(1);
    const panel = await drawer(user); const search = within(panel).getByPlaceholderText("Search for a movie..."); expect(search).toHaveFocus();
    expect(within(panel).getByText("No favorites yet. You can still explore movies.")).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: "Include genres" })).toHaveTextContent("Include");
    expect(within(panel).getByRole("button", { name: "Include genres" })).not.toHaveTextContent("Include genres");
    expect(within(panel).getByRole("button", { name: "Avoid genres" })).toHaveTextContent("Avoid");
    await user.type(search, "Signal"); await user.click(await within(panel).findByRole("option", { name: /Signal in the Fog/ }));
    expect(within(panel).getByRole("button", { name: "Remove Signal in the Fog" })).toBeInTheDocument();
  });

  it("prevents Include/Avoid contradictions and supports undo/reset", async () => {
    vi.stubGlobal("fetch", mockApi()); const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Explore movies" }); await drawer(user);
    await genre(user, "Include", "Drama"); const panel = screen.getByRole("dialog", { name: "Personalize your recommendations" });
    expect(within(panel).getByRole("button", { name: "Include: Drama" })).toBeInTheDocument();
    await user.click(within(panel).getByRole("button", { name: /Avoid genres/ })); const selector = screen.getByRole("dialog", { name: "Avoid genres" });
    expect(within(selector).getByRole("button", { name: "Drama" })).toBeDisabled(); await user.click(within(selector).getByRole("button", { name: "Close genre selector" }));
    await user.click(within(panel).getByRole("button", { name: /Undo/ })); expect(within(panel).queryByRole("button", { name: "Include: Drama" })).not.toBeInTheDocument();
    await genre(user, "Include", "Comedy"); await user.click(within(panel).getByRole("button", { name: /Reset all/ }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Personalize your recommendations" })).not.toBeInTheDocument());
    await waitFor(() => expect(document.querySelector(".personalization-summary")).not.toBeInTheDocument());
  });

  it("retains old movies during edits and replaces them only after success", async () => {
    let calls = 0; vi.stubGlobal("fetch", mockApi({ recommend: (revision, preferences) => rec(revision, preferences, calls++ === 0 ? [MOVIES[0], MOVIES[1]] : [MOVIES[2]]) }));
    const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Signal in the Fog", level: 1 }); await drawer(user); await genre(user, "Include", "Drama");
    const panel = screen.getByRole("dialog", { name: "Personalize your recommendations" }); await user.click(within(panel).getByRole("button", { name: "Close personalization" }));
    expect(screen.getByRole("heading", { name: "Signal in the Fog", level: 1 })).toBeInTheDocument(); expect(screen.getByText("Pending changes. Update to apply them.")).toBeInTheDocument(); expect(screen.getByText("Current results use no personalization")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Update recommendations" })); expect(await screen.findByRole("heading", { name: "The Last Matinee", level: 1 })).toBeInTheDocument();
  });

  it("keeps the applied hero visible until a hidden movie edit is submitted", async () => {
    vi.stubGlobal("fetch", mockApi()); const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Signal in the Fog", level: 1 });
    const card = screen.getByRole("button", { name: "View details for Signal in the Fog" }).closest("article")!; await user.click(within(card).getByRole("button", { name: "Hide this movie" }));
    expect(screen.getByRole("heading", { name: "Signal in the Fog", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Pending changes. Update to apply them.")).toBeInTheDocument();
  });

  it("ignores a response superseded by a preference edit", async () => {
    let resolveOld!: (value: Response) => void;
    vi.stubGlobal("fetch", vi.fn((url: string | URL | Request) => { const path = String(url); if (path.includes("application/status")) return json(STATUS); if (path.includes("catalog/search")) return json({ items: MOVIES, total: 3, backend_mode: "fixture" }); if (path.includes("catalog/posters")) return json({ provider: "tmdb", configured: false, items: [] }); return new Promise<Response>((resolve) => { resolveOld = resolve; }); }));
    const user = userEvent.setup(); render(<App />); await screen.findByText(/Synthetic demo data/); await drawer(user); await genre(user, "Include", "Comedy"); resolveOld(await json(rec(0))); await waitFor(() => expect(screen.queryByRole("heading", { name: "Signal in the Fog", level: 1 })).not.toBeInTheDocument());
  });

  it("uses backdrops, preserves order, and falls back to a poster on failure", async () => {
    vi.stubGlobal("fetch", mockApi({ artwork: true })); render(<App />); await screen.findByRole("heading", { name: "Explore movies" });
    const backdrop = await waitFor(() => document.querySelector('img[src="https://image.tmdb.org/t/p/w1280/101.jpg"]') as HTMLImageElement); expect(backdrop).toBeTruthy(); fireEvent.error(backdrop);
    await waitFor(() => expect(document.querySelector('img[src="https://image.tmdb.org/t/p/w500/101.jpg"]')).toBeTruthy());
    expect(screen.getAllByRole("button", { name: /View details for/ }).map((button) => button.getAttribute("aria-label"))).toEqual(["View details for Signal in the Fog", "View details for Orbit House", "View details for The Last Matinee"]);
  });

  it("keeps hover and focus on the original tile without changing the hero", async () => {
    vi.stubGlobal("fetch", mockApi({ artwork: true })); const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Signal in the Fog", level: 1 });
    const secondCard = screen.getByRole("button", { name: "View details for Orbit House" }).closest("article")!;
    fireEvent.mouseOver(secondCard); fireEvent.focus(secondCard);
    expect(document.querySelector(".expanded-preview")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Signal in the Fog", level: 1 })).toBeInTheDocument();
    await user.click(within(secondCard).getByRole("button", { name: "Details for Orbit House" }));
    const details = screen.getByRole("dialog", { name: "Orbit House" }); expect(details).toBeInTheDocument();
    await user.click(within(details).getByRole("button", { name: "Close details" }));
    await user.click(within(secondCard).getByRole("button", { name: "Add Orbit House to favorites" }));
    expect(within(secondCard).getByRole("button", { name: "Orbit House is in favorites" })).toBeDisabled();
    expect(screen.queryByRole("dialog", { name: "Orbit House" })).not.toBeInTheDocument();
    await user.click(within(secondCard).getByRole("button", { name: "Hide this movie" }));
    expect(screen.getByText("Pending changes. Update to apply them.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Signal in the Fog", level: 1 })).toBeInTheDocument();
  });

  it("uses one recommendation heading across visual rows and preserves response order", async () => {
    const movies = Array.from({ length: 10 }, (_, index) => ({ movie_id: String(200 + index), title: `Movie ${index + 1} (${2000 + index})`, genres: ["Drama"], poster_url: null }));
    vi.stubGlobal("fetch", mockApi({ recommend: (revision, preferences) => rec(revision, preferences, movies) })); const user = userEvent.setup(); render(<App />);
    expect(await screen.findByRole("heading", { name: "Explore movies", level: 2 })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { name: "Explore movies", level: 2 })).toHaveLength(1);
    expect(screen.queryByText(/More to explore|More recommendations/)).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /View details for/ }).map((button) => button.getAttribute("aria-label"))).toEqual(movies.map((movie) => `View details for ${movie.title}`));
    expect(document.querySelector(".rank-badge")).not.toBeInTheDocument();
    await drawer(user); await genre(user, "Include", "Drama"); const panel = screen.getByRole("dialog", { name: "Personalize your recommendations" }); await user.click(within(panel).getByRole("button", { name: "Update recommendations" }));
    expect(await screen.findByRole("heading", { name: "Your recommendations", level: 2 })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { name: "Your recommendations", level: 2 })).toHaveLength(1);
    expect(screen.queryByText(/More to explore|More recommendations/)).not.toBeInTheDocument();
  });

  it("distinguishes first-load failure from a successful empty response", async () => {
    const base = mockApi(); const failed = vi.fn((url: string | URL | Request, init?: RequestInit) => String(url).includes("recommendations") ? json({ error: { message: "Backend unavailable" } }, 503) : base(url, init));
    vi.stubGlobal("fetch", failed); const view = render(<App />); expect(await screen.findByRole("alert")).toHaveTextContent("Backend unavailable"); expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument(); expect(screen.queryByRole("heading", { name: "No recommendations found" })).not.toBeInTheDocument(); view.unmount();
    vi.stubGlobal("fetch", mockApi({ recommend: (revision, preferences) => rec(revision, preferences, []) })); render(<App />); expect(await screen.findByRole("heading", { name: "No recommendations found" })).toBeInTheDocument(); expect(screen.getByText("Try adjusting your genre preferences, or reset them to explore movies again.")).toBeInTheDocument(); expect(screen.getByRole("button", { name: "Edit preferences" })).toBeInTheDocument(); expect(screen.getByRole("button", { name: "Reset all" })).toBeInTheDocument(); expect(screen.queryByText(/bounded candidate|constraints were not relaxed/i)).not.toBeInTheDocument();
  });

  it("keeps unavailable text refinement in the drawer and backend details accurate", async () => {
    const baseline = { ...STATUS, backend_mode: "baseline", model_label: "Baseline model", data_source_label: "MovieLens 32M · ratio_80_10_10" }; vi.stubGlobal("fetch", mockApi({ status: baseline }));
    const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Explore movies" }); expect(screen.getByRole("banner")).not.toHaveTextContent("Baseline model"); expect(screen.queryByPlaceholderText("Keep the sci-fi, but leave out horror.")).not.toBeInTheDocument();
    const panel = await drawer(user); expect(within(panel).queryByText(/Text refinement is coming soon/)).not.toBeInTheDocument(); await user.click(within(panel).getByRole("button", { name: "Close personalization" }));
    await user.click(screen.getByRole("button", { name: "About" })); const about = screen.getByRole("dialog", { name: "About Trustworthy Recommendation System" }); expect(about).toHaveTextContent("transparent, preference-driven movie discovery"); expect(about).toHaveTextContent("TMDB supplies display artwork only and does not affect ranking"); expect(within(about).queryByRole("link", { name: /System card/i })).not.toBeInTheDocument(); await user.click(within(about).getByText("System details")); expect(about).toHaveTextContent("Baseline model"); expect(about).toHaveTextContent("Text refinementUnavailable"); expect(about).toHaveTextContent("genre-weighted popularity"); expect(within(about).getByRole("link", { name: "Project repository" })).toBeInTheDocument(); const logo = screen.getByAltText("The Movie Database (TMDB)"); expect(logo).toHaveAttribute("src", "/tmdb-logo.svg");
  });

  it("keeps applied preferences above an empty result and resets them with one request", async () => {
    let calls = 0; const fetch = mockApi({ recommend: (revision, preferences) => rec(revision, preferences, calls++ === 1 ? [] : MOVIES) }); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Explore movies" }); await drawer(user); await genre(user, "Include", "Drama"); const panel = screen.getByRole("dialog", { name: "Personalize your recommendations" }); await user.click(within(panel).getByRole("button", { name: "Update recommendations" }));
    expect(await screen.findByRole("heading", { name: "No recommendations found" })).toBeInTheDocument(); expect(screen.getByText("Include Drama")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Edit preferences" })); expect(within(screen.getByRole("dialog", { name: "Personalize your recommendations" })).getByRole("button", { name: "Include: Drama" })).toBeInTheDocument(); await user.click(screen.getByRole("button", { name: "Close personalization" }));
    const beforeReset = fetch.mock.calls.filter(([url]) => String(url).includes("recommendations")).length; await user.click(screen.getAllByRole("button", { name: "Reset all" })[0]); await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(beforeReset + 1)); expect(await screen.findByRole("heading", { name: "Explore movies" })).toBeInTheDocument();
  });

  it("submits the drawer snapshot once and resets applied preferences once from the summary", async () => {
    const fetch = mockApi(); vi.stubGlobal("fetch", fetch); const user = userEvent.setup(); render(<App />); await screen.findByRole("heading", { name: "Explore movies" });
    const panel = await drawer(user); const search = within(panel).getByLabelText("Movies you like"); await user.type(search, "Signal"); await user.click(await within(panel).findByRole("option", { name: /Signal in the Fog/ })); await genre(user, "Include", "Drama");
    await user.click(within(panel).getByRole("button", { name: "Update recommendations" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Personalize your recommendations" })).not.toBeInTheDocument());
    const recommendationCalls = fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"));
    expect(recommendationCalls).toHaveLength(2);
    expect(JSON.parse(String(recommendationCalls[1][1]?.body)).preferences).toEqual({ ...EMPTY, history_items: ["101"], include_genres: ["Drama"] });
    await user.click(screen.getByRole("button", { name: "Reset all" }));
    await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))).toHaveLength(3));
    expect(JSON.parse(String(fetch.mock.calls.filter(([url]) => String(url).includes("recommendations"))[2][1]?.body)).preferences).toEqual(EMPTY);
  });

  it("formats only unambiguous trailing catalog articles and years", () => {
    expect(formatCatalogTitle("Princess Bride, The (1987)")).toEqual({ title: "The Princess Bride", year: "1987" });
    expect(formatCatalogTitle("A Very Long, Uncertain Title (2004)")).toEqual({ title: "A Very Long, Uncertain Title", year: "2004" });
    expect(formatCatalogTitle("Film, Maybe (1999)")).toEqual({ title: "Film, Maybe", year: "1999" });
  });
});
