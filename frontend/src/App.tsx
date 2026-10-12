import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Ban, Check, ChevronDown, ChevronLeft, ChevronRight, Film, ImageOff, Info, LoaderCircle, Pencil, Plus, RotateCcw, Search, SlidersHorizontal, Undo2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getApplicationStatus, getPosters, getRecommendations, refine, searchCatalog, type ApplicationStatus, type CatalogMovie, type PreferenceState, type RecommendationResponse } from "./api/client";
import { Button } from "./components/ui/button";
import { MovieDialog } from "./components/ui/dialog";

const EMPTY: PreferenceState = { history_items: [], excluded_movie_ids: [], include_genres: [], exclude_genres: [] };
const TMDB_LOGO = `${import.meta.env.BASE_URL}tmdb-logo.svg`;
type Artwork = { poster: string | null; backdrop: string | null; backdropSmall: string | null };
type Recommendation = RecommendationResponse["recommendations"][number];

export function formatCatalogTitle(canonical: string) {
  const yearMatch = canonical.match(/\s+\((\d{4})\)$/);
  const year = yearMatch?.[1] ?? null;
  const withoutYear = yearMatch ? canonical.slice(0, yearMatch.index).trim() : canonical;
  const articleMatch = withoutYear.match(/^(.+),\s+(The|An|A)$/);
  return { title: articleMatch ? `${articleMatch[2]} ${articleMatch[1]}` : withoutYear, year };
}

function samePreferences(left: PreferenceState, right: PreferenceState) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function summarizePreferences(value: PreferenceState) {
  const genreSummary = (label: string, genres: string[]) => genres.length
    ? `${label} ${genres.slice(0, 2).join(", ")}${genres.length > 2 ? ` +${genres.length - 2}` : ""}`
    : "";
  return [
    value.history_items.length ? `${value.history_items.length} favorite${value.history_items.length === 1 ? "" : "s"}` : "",
    genreSummary("Include", value.include_genres),
    genreSummary("Avoid", value.exclude_genres),
    value.excluded_movie_ids.length ? `${value.excluded_movie_ids.length} hidden` : "",
  ].filter(Boolean).join(" · ") || "No personalization";
}

function sessionId() {
  const stored = sessionStorage.getItem("trustworthy-recsys-session");
  if (stored) return stored;
  const created = crypto.randomUUID();
  sessionStorage.setItem("trustworthy-recsys-session", created);
  return created;
}

function hasPreferences(value: PreferenceState) {
  return value.history_items.length + value.excluded_movie_ids.length + value.include_genres.length + value.exclude_genres.length > 0;
}

function Portrait({ movie, artwork, compact = false }: { movie: CatalogMovie; artwork?: Artwork; compact?: boolean }) {
  const source = artwork?.poster ?? movie.poster_url;
  const [broken, setBroken] = useState(false);
  useEffect(() => setBroken(false), [source]);
  if (source && !broken) return <img src={source} alt={`Poster for ${movie.title}`} loading="lazy" decoding="async" onError={() => setBroken(true)} className="h-full w-full object-cover" />;
  return <div className="poster-fallback h-full w-full" aria-label={`No artwork available for ${movie.title}`}><ImageOff size={compact ? 14 : 22} aria-hidden="true" />{!compact && <span>{movie.title}</span>}</div>;
}

function HeroArtwork({ movie, artwork }: { movie: CatalogMovie; artwork?: Artwork }) {
  const [backdropBroken, setBackdropBroken] = useState(false);
  const [posterBroken, setPosterBroken] = useState(false);
  useEffect(() => { setBackdropBroken(false); setPosterBroken(false); }, [artwork?.backdrop, artwork?.poster, movie.poster_url]);
  if (artwork?.backdrop && !backdropBroken) return <img src={artwork.backdrop} srcSet={artwork.backdropSmall ? `${artwork.backdropSmall} 780w, ${artwork.backdrop} 1280w` : undefined} sizes="(max-width: 780px) 780px, 1280px" alt="" loading="eager" decoding="async" onError={() => setBackdropBroken(true)} className="hero-backdrop" />;
  const poster = artwork?.poster ?? movie.poster_url;
  if (poster && !posterBroken) return <div className="hero-poster-fallback"><img src={poster} alt="" loading="eager" decoding="async" onError={() => setPosterBroken(true)} /></div>;
  return <div className="hero-neutral" aria-label={`No artwork available for ${movie.title}`}><Film aria-hidden="true" /></div>;
}

type MovieCarouselProps = {
  ariaLabel: string;
  items: Recommendation[];
  artwork: Record<string, Artwork>;
  favoriteIds: string[];
  reducedMotion: boolean | null;
  onDetail: (movie: CatalogMovie) => void;
  onFavorite: (movie: CatalogMovie) => void;
  onHide: (movie: CatalogMovie) => void;
  continuation?: boolean;
};

function MovieCarousel({ ariaLabel, items, artwork, favoriteIds, reducedMotion, onDetail, onFavorite, onHide, continuation = false }: MovieCarouselProps) {
  const row = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ previous: false, next: true });
  const updateEdges = useCallback(() => {
    const node = row.current;
    if (node) setEdges({ previous: node.scrollLeft > 3, next: node.scrollLeft + node.clientWidth < node.scrollWidth - 3 });
  }, []);
  useEffect(() => { updateEdges(); window.addEventListener("resize", updateEdges); return () => window.removeEventListener("resize", updateEdges); }, [items.length, updateEdges]);
  const scroll = (direction: number) => { if (row.current) row.current.scrollBy({ left: direction * row.current.clientWidth * .82, behavior: reducedMotion ? "auto" : "smooth" }); };
  return <div className={`movie-row ${continuation ? "continuation" : ""}`}><div className="row-carousel"><button className="edge-arrow previous" aria-label={`Previous ${ariaLabel}`} disabled={!edges.previous} onClick={() => scroll(-1)}><ChevronLeft /></button><div className="row-track" ref={row} tabIndex={0} aria-label={ariaLabel} onScroll={updateEdges} onKeyDown={(event) => { if (event.key === "ArrowRight") scroll(1); if (event.key === "ArrowLeft") scroll(-1); }}>{items.map(({ movie }) => {
    const formatted = formatCatalogTitle(movie.title);
    const favorite = favoriteIds.includes(movie.movie_id);
    return <motion.article
      tabIndex={0}
      aria-label={movie.title}
      data-movie-id={movie.movie_id}
      key={movie.movie_id}
      className="portrait-card"
      whileHover={reducedMotion ? undefined : { scale: 1.04, y: -4 }}
      whileFocus={reducedMotion ? undefined : { scale: 1.04, y: -4 }}
      transition={{ duration: .18, ease: "easeOut" }}
    ><button className="card-art" onClick={() => onDetail(movie)} aria-label={`View details for ${movie.title}`}><Portrait movie={movie} artwork={artwork[movie.movie_id]} /></button><div className="card-copy"><h3>{formatted.title}</h3><p>{formatted.year ?? "Year unavailable"}</p></div><div className="card-actions"><button onClick={() => onDetail(movie)} aria-label={`Details for ${movie.title}`} title="Details"><Info /></button><button disabled={favorite} onClick={() => onFavorite(movie)} aria-label={favorite ? `${movie.title} is in favorites` : `Add ${movie.title} to favorites`} title={favorite ? "In favorites" : "Add to favorites"}><Plus /></button><button onClick={() => onHide(movie)} aria-label="Hide this movie" title="Hide this movie"><Ban /></button></div></motion.article>;
  })}</div><button className="edge-arrow next" aria-label={`Next ${ariaLabel}`} disabled={!edges.next} onClick={() => scroll(1)}><ChevronRight /></button></div></div>;
}

function GenreChoices({ kind, genres, selected, blocked, onToggle }: { kind: "include" | "avoid"; genres: string[]; selected: string[]; blocked: string[]; onToggle: (genre: string, kind: "include" | "avoid") => void }) {
  return <div className="genre-dialog-options">{genres.map((genre) => {
    const active = selected.includes(genre); const disabled = blocked.includes(genre);
    return <button type="button" key={genre} aria-pressed={active} disabled={disabled} title={disabled ? `Remove ${genre} from ${kind === "include" ? "Avoid" : "Include"} first` : undefined} onClick={() => onToggle(genre, kind)} className={`genre-option ${kind} ${active ? "active" : ""}`}><span className="genre-check" aria-hidden="true">{active ? (kind === "include" ? <Check size={14} /> : <Ban size={14} />) : null}</span>{genre}</button>;
  })}</div>;
}

export function App() {
  const reducedMotion = useReducedMotion();
  const [status, setStatus] = useState<ApplicationStatus | null>(null);
  const [preferences, setPreferences] = useState<PreferenceState>(EMPTY);
  const [revision, setRevision] = useState(0);
  const [undoStack, setUndoStack] = useState<PreferenceState[]>([]);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState<string | null>(null);
  const [browserMs, setBrowserMs] = useState<number | null>(null);
  const [personalizeOpen, setPersonalizeOpen] = useState(false);
  const [detail, setDetail] = useState<CatalogMovie | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [genreDialog, setGenreDialog] = useState<"include" | "avoid" | null>(null);
  const [query, setQuery] = useState("");
  const [catalog, setCatalog] = useState<CatalogMovie[]>([]);
  const [knownMovies, setKnownMovies] = useState<Record<string, CatalogMovie>>({});
  const [artwork, setArtwork] = useState<Record<string, Artwork>>({});
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [refinement, setRefinement] = useState("");
  const [refinementMessage, setRefinementMessage] = useState<string | null>(null);
  const [tmdbLogoFailed, setTmdbLogoFailed] = useState(false);
  const recommendationRequest = useRef<AbortController | null>(null);
  const searchRequest = useRef<AbortController | null>(null);
  const requestSequence = useRef(0);
  const initialRequested = useRef(false);
  const requestedArtwork = useRef(new Set<string>());
  const revisionRef = useRef(0);
  const sid = useMemo(sessionId, []);

  useEffect(() => { const controller = new AbortController(); void getApplicationStatus(controller.signal).then(setStatus).catch(() => undefined); return () => controller.abort(); }, []);

  const enrichArtwork = useCallback(async (ids: string[]) => {
    const pending = [...new Set(ids)].filter((id) => !requestedArtwork.current.has(id)).slice(0, 30);
    if (!pending.length) return;
    pending.forEach((id) => requestedArtwork.current.add(id));
    try {
      const response = await getPosters(pending);
      setArtwork((current) => ({ ...current, ...Object.fromEntries(response.items.map((item) => [item.movie_id, { poster: item.poster_url ?? null, backdrop: item.backdrop_url ?? null, backdropSmall: item.backdrop_url_small ?? null }])) }));
    } catch {
      // A transient batch failure must not suppress display-art retries forever.
      pending.forEach((id) => requestedArtwork.current.delete(id));
    }
  }, []);

  const recommend = useCallback(async (state: PreferenceState, stateRevision: number) => {
    recommendationRequest.current?.abort();
    const controller = new AbortController(); recommendationRequest.current = controller;
    const sequence = ++requestSequence.current; const started = performance.now();
    setLoading(true); setFailure(null);
    try {
      const response = await getRecommendations({ session_id: sid, preference_revision: stateRevision, preferences: state, result_count: 12, known_user_id: null }, controller.signal);
      if (sequence !== requestSequence.current || response.preference_revision !== stateRevision) return false;
      setResult(response);
      setKnownMovies((current) => ({ ...current, ...Object.fromEntries(response.recommendations.map(({ movie }) => [movie.movie_id, movie])) }));
      void enrichArtwork(response.recommendations.map(({ movie }) => movie.movie_id));
      setBrowserMs(performance.now() - started);
      return true;
    } catch (error) {
      if ((error as Error).name !== "AbortError" && sequence === requestSequence.current) setFailure(error instanceof ApiError ? error.message : "The recommendation service could not be reached.");
      return false;
    } finally { if (sequence === requestSequence.current) setLoading(false); }
  }, [enrichArtwork, sid]);

  useEffect(() => { if (!initialRequested.current) { initialRequested.current = true; void recommend(EMPTY, 0); } }, [recommend]);

  const loadCatalog = useCallback(async (text: string) => {
    searchRequest.current?.abort();
    if (!text.trim()) { setCatalog([]); setSearching(false); setSearchError(null); return; }
    const controller = new AbortController(); searchRequest.current = controller; setSearching(true); setSearchError(null);
    try {
      const response = await searchCatalog(text, controller.signal);
      setCatalog(response.items); setKnownMovies((current) => ({ ...current, ...Object.fromEntries(response.items.map((movie) => [movie.movie_id, movie])) }));
      void enrichArtwork(response.items.map((movie) => movie.movie_id));
    } catch (error) { if ((error as Error).name !== "AbortError") setSearchError((error as Error).message); }
    finally { if (!controller.signal.aborted) setSearching(false); }
  }, [enrichArtwork]);
  useEffect(() => { const timer = window.setTimeout(() => void loadCatalog(query), 180); return () => window.clearTimeout(timer); }, [query, loadCatalog]);

  const changePreferences = (next: PreferenceState) => {
    if (samePreferences(next, preferences)) return revisionRef.current;
    recommendationRequest.current?.abort(); requestSequence.current += 1; setLoading(false);
    revisionRef.current += 1;
    setUndoStack((stack) => [...stack.slice(-19), preferences]); setPreferences(next); setRevision(revisionRef.current); setFailure(null);
    return revisionRef.current;
  };
  const undo = () => { const previous = undoStack.at(-1); if (!previous) return; recommendationRequest.current?.abort(); requestSequence.current += 1; revisionRef.current += 1; setUndoStack((stack) => stack.slice(0, -1)); setPreferences(previous); setRevision(revisionRef.current); setFailure(null); };
  const submitPreferences = async (snapshot: PreferenceState, snapshotRevision: number, closeOnSuccess = false) => {
    const exactSnapshot = { history_items: [...snapshot.history_items], excluded_movie_ids: [...snapshot.excluded_movie_ids], include_genres: [...snapshot.include_genres], exclude_genres: [...snapshot.exclude_genres] };
    const succeeded = await recommend(exactSnapshot, snapshotRevision);
    if (succeeded && closeOnSuccess) setPersonalizeOpen(false);
    return succeeded;
  };
  const resetAll = () => {
    const nextRevision = samePreferences(preferences, EMPTY) ? revisionRef.current : changePreferences(EMPTY);
    void submitPreferences(EMPTY, nextRevision, personalizeOpen);
  };
  const toggleGenre = (genre: string, kind: "include" | "avoid") => {
    if (kind === "include") { if (preferences.exclude_genres.includes(genre)) return; changePreferences({ ...preferences, include_genres: preferences.include_genres.includes(genre) ? preferences.include_genres.filter((value) => value !== genre) : [...preferences.include_genres, genre] }); }
    else { if (preferences.include_genres.includes(genre)) return; changePreferences({ ...preferences, exclude_genres: preferences.exclude_genres.includes(genre) ? preferences.exclude_genres.filter((value) => value !== genre) : [...preferences.exclude_genres, genre] }); }
  };
  const submitRefinement = async () => {
    if (!status?.refinement_available || !refinement.trim()) return;
    try { const response = await refine(sid, revision, preferences, refinement.trim()); if (response.outcome === "applied") { changePreferences(response.preferences); setRefinementMessage(response.summary); } else setRefinementMessage(response.outcome === "clarification" ? response.question : response.message); }
    catch (error) { setRefinementMessage((error as Error).message); }
  };

  const dirty = result !== null && !samePreferences(preferences, result.applied_preferences);
  const displayed = result?.recommendations ?? [];
  const feature = displayed[0]?.movie ?? null;
  const selected = preferences.history_items.map((id) => knownMovies[id]).filter(Boolean) as CatalogMovie[];
  const genres = status?.available_genres ?? [...new Set(Object.values(knownMovies).flatMap((movie) => movie.genres))].sort();
  const applied = result?.applied_preferences;
  const appliedSummary = summarizePreferences(applied ?? EMPTY);
  const appliedHasPreferences = Boolean(applied && hasPreferences(applied));
  const showPreferenceSummary = appliedHasPreferences || dirty;
  const activeChips = [...preferences.include_genres.map((genre) => ({ label: `Include: ${genre}`, remove: () => toggleGenre(genre, "include") })), ...preferences.exclude_genres.map((genre) => ({ label: `Avoid: ${genre}`, remove: () => toggleGenre(genre, "avoid") }))];
  const addFavorite = (movie: CatalogMovie) => { if (!preferences.history_items.includes(movie.movie_id)) { setKnownMovies((current) => ({ ...current, [movie.movie_id]: movie })); changePreferences({ ...preferences, history_items: [...preferences.history_items, movie.movie_id] }); } };
  const hideMovie = (movie: CatalogMovie) => { if (!preferences.excluded_movie_ids.includes(movie.movie_id)) changePreferences({ ...preferences, excluded_movie_ids: [...preferences.excluded_movie_ids, movie.movie_id] }); };

  return <main id="top" className="app-shell">
    <header className="site-header"><div className="header-inner"><a href="#top" className="brand">Trustworthy Recommendation System</a><nav className="header-actions" aria-label="Primary"><Button variant="ghost" onClick={() => setPersonalizeOpen(true)}><SlidersHorizontal size={16} /> Personalize{dirty && <span className="pending-dot" aria-label="Pending changes" />}</Button><Button variant="quiet" onClick={() => setAboutOpen(true)}>About</Button></nav></div></header>

    {loading && !result && <section className="hero-skeleton" aria-label="Loading movies"><div /><div className="skeleton-copy"><span /><span /><span /></div></section>}
    {!result && failure && <section className="first-error" role="alert"><Film /><h1>Movies are unavailable right now</h1><p>{failure}</p><Button onClick={() => void recommend(preferences, revision)}>Retry</Button></section>}
    {feature && <motion.section key={`${result?.request_id}-${feature.movie_id}`} className="feature-hero" initial={reducedMotion ? false : { opacity: 0 }} animate={{ opacity: 1 }}><div className="feature-media"><HeroArtwork movie={feature} artwork={artwork[feature.movie_id]} /></div><div className="feature-scrim" /><div className={`feature-content ${formatCatalogTitle(feature.title).title.length > 55 ? "long-title" : ""}`}><p className="feature-context">{applied && hasPreferences(applied) ? "Your recommendations" : "Explore movies"}</p><h1>{formatCatalogTitle(feature.title).title}</h1><p className="feature-genres">{[formatCatalogTitle(feature.title).year, ...feature.genres].filter(Boolean).join(" · ") || "Metadata unavailable"}</p><div className="feature-actions"><Button onClick={() => setDetail(feature)}><Info /> Details</Button><Button variant="ghost" disabled={preferences.history_items.includes(feature.movie_id)} onClick={() => addFavorite(feature)}><Plus /> {preferences.history_items.includes(feature.movie_id) ? "In favorites" : "Add to favorites"}</Button></div></div></motion.section>}

    {result && <div className={`browse-content ${!feature ? "no-feature" : ""} ${!showPreferenceSummary ? "no-summary" : ""}`}>{showPreferenceSummary && <div className="personalization-summary"><SlidersHorizontal className="summary-icon" aria-hidden="true" /><div><strong>{appliedHasPreferences ? appliedSummary : "Current results use no personalization"}</strong>{dirty && <span>Pending changes. Update to apply them.</span>}{loading && <span>Updating—previous results remain visible.</span>}{failure && <span role="alert">Update failed: {failure} Previous results remain visible.</span>}</div><Button variant="quiet" onClick={() => setPersonalizeOpen(true)}><Pencil size={15} /> Edit</Button><Button variant="quiet" onClick={resetAll} disabled={loading}><RotateCcw size={15} /> Reset all</Button>{dirty && <Button onClick={() => void submitPreferences(preferences, revision)} disabled={loading}>{loading ? <LoaderCircle className="animate-spin" /> : null}Update recommendations</Button>}{failure && !loading && <Button onClick={() => void submitPreferences(preferences, revision)}>Retry</Button>}</div>}
      {displayed.length > 0 ? <section className="recommendation-section" aria-labelledby="recommendations-heading"><div className="row-heading"><h2 id="recommendations-heading">{appliedHasPreferences ? "Your recommendations" : "Explore movies"}</h2></div><MovieCarousel ariaLabel={appliedHasPreferences ? "Your recommendations" : "Explore movies"} items={displayed.slice(0, 8)} artwork={artwork} favoriteIds={preferences.history_items} reducedMotion={reducedMotion} onDetail={setDetail} onFavorite={addFavorite} onHide={hideMovie} />{displayed.length > 8 && <MovieCarousel ariaLabel={`${appliedHasPreferences ? "Your recommendations" : "Explore movies"} continuation`} items={displayed.slice(8)} artwork={artwork} favoriteIds={preferences.history_items} reducedMotion={reducedMotion} onDetail={setDetail} onFavorite={addFavorite} onHide={hideMovie} continuation />}</section> : <section className="no-matches"><Film /><h2>No recommendations found</h2><p>Try adjusting your genre preferences, or reset them to explore movies again.</p><div className="empty-actions"><Button variant="ghost" onClick={() => setPersonalizeOpen(true)}>Edit preferences</Button><Button onClick={resetAll} disabled={loading}>{loading ? <LoaderCircle className="animate-spin" /> : <RotateCcw />} Reset all</Button></div></section>}
    </div>}
    <footer>Trustworthy Recommendation System · A movie-discovery research prototype.{status?.backend_mode === "fixture" && <span> Synthetic demo data.</span>}</footer>

    <MovieDialog open={personalizeOpen} onOpenChange={setPersonalizeOpen} title="Personalize" accessibleName="Personalize your recommendations" description="Choose movies and genres to get better recommendations." closeLabel="Close personalization" contentClassName="personalize-panel"><div className="personalize-drawer"><label htmlFor="movie-search">Add favorite movies</label><div className="drawer-search"><Search /><input autoFocus id="movie-search" aria-label="Movies you like" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search for a movie..." /></div>
      {query.trim() && <div className="drawer-search-results" role="listbox" aria-label="Movie search results">{searching && <p>Searching movies…</p>}{searchError && <p>{searchError} <button onClick={() => void loadCatalog(query)}>Retry</button></p>}{!searching && !searchError && catalog.map((movie) => <button key={movie.movie_id} role="option" aria-selected={preferences.history_items.includes(movie.movie_id)} disabled={preferences.history_items.includes(movie.movie_id)} onClick={() => { addFavorite(movie); setQuery(""); }}><span className="drawer-result-poster"><Portrait movie={movie} artwork={artwork[movie.movie_id]} compact /></span><span><strong>{movie.title}</strong><small>{movie.genres.join(" · ") || "Genre unavailable"}</small></span><em>{preferences.history_items.includes(movie.movie_id) ? "Selected" : "Add"}</em></button>)}{!searching && !searchError && !catalog.length && <p>No supported movies match “{query}”.</p>}</div>}
      <div className="drawer-section"><h3>Your favorites <span>{selected.length}</span></h3><div className="drawer-favorites"><AnimatePresence initial={false}>{selected.map((movie) => <motion.div layout key={movie.movie_id} className="drawer-favorite"><span><Portrait movie={movie} artwork={artwork[movie.movie_id]} compact /></span><strong>{movie.title}</strong><button aria-label={`Remove ${movie.title}`} onClick={() => changePreferences({ ...preferences, history_items: preferences.history_items.filter((id) => id !== movie.movie_id) })}><X /></button></motion.div>)}</AnimatePresence>{!selected.length && <p>No favorites yet. You can still explore movies.</p>}</div></div>
      <div className="drawer-section"><h3>Genre preferences</h3><div className="drawer-genre-buttons"><Button variant="ghost" aria-label="Include genres" onClick={() => setGenreDialog("include")}><Check /> <span>Include</span> <ChevronDown /></Button><Button variant="ghost" aria-label="Avoid genres" onClick={() => setGenreDialog("avoid")}><Ban /> <span>Avoid</span> <ChevronDown /></Button></div><div className="drawer-chips">{activeChips.map((chip) => <button key={chip.label} onClick={chip.remove}>{chip.label}<X /></button>)}{!activeChips.length && <span>No genre filters selected</span>}</div></div>
      {status?.refinement_available && <div className="drawer-section"><label htmlFor="refinement">Refine with text</label><div className="compact-refinement"><input id="refinement" value={refinement} onChange={(event) => setRefinement(event.target.value)} placeholder="Keep the sci-fi, but leave out horror." /><Button disabled={!refinement.trim()} onClick={() => void submitRefinement()}>Apply</Button></div>{refinementMessage && <p>{refinementMessage}</p>}</div>}
      <div className="drawer-spacer" /><p className="drawer-save-note">Changes stay pending until you update. Current recommendations remain visible underneath.</p><div className="drawer-actions"><Button variant="quiet" disabled={!undoStack.length || loading} onClick={undo}><Undo2 /> Undo</Button><Button variant="quiet" disabled={loading || (!hasPreferences(preferences) && !result?.applied_preferences.excluded_movie_ids.length && !hasPreferences(result?.applied_preferences ?? EMPTY))} onClick={resetAll}><RotateCcw /> Reset all</Button><Button disabled={loading || !dirty} onClick={() => void submitPreferences(preferences, revision, true)}>{loading && <LoaderCircle className="animate-spin" />}{result ? "Update recommendations" : "Find recommendations"}</Button></div>{dirty && <p className="drawer-pending" role="status">Pending changes</p>}{failure && <p className="drawer-error" role="alert">{failure} Your edits are still here.</p>}</div></MovieDialog>

    <MovieDialog open={genreDialog !== null} onOpenChange={(open) => !open && setGenreDialog(null)} title={genreDialog === "include" ? "Include genres" : "Avoid genres"} description={genreDialog === "include" ? "Recommendations will match at least one selected genre." : "Recommendations will match none of the selected genres."} closeLabel="Close genre selector">{genreDialog && <div className="genre-dialog-content"><GenreChoices kind={genreDialog} genres={genres} selected={genreDialog === "include" ? preferences.include_genres : preferences.exclude_genres} blocked={genreDialog === "include" ? preferences.exclude_genres : preferences.include_genres} onToggle={toggleGenre} /><Button onClick={() => setGenreDialog(null)}>Done</Button></div>}</MovieDialog>
    <MovieDialog open={detail !== null} onOpenChange={(open) => !open && setDetail(null)} title={detail?.title ?? "Movie details"} description="Catalog information">{detail && <div className="detail-content"><div className="detail-poster"><Portrait movie={detail} artwork={artwork[detail.movie_id]} /></div><dl><dt>Genres</dt><dd>{detail.genres.join(", ") || "Unavailable"}</dd><dt>Artwork</dt><dd>{artwork[detail.movie_id]?.poster || artwork[detail.movie_id]?.backdrop ? "Provided by TMDB" : "Not available"}</dd></dl></div>}</MovieDialog>
    <MovieDialog open={aboutOpen} onOpenChange={setAboutOpen} title="About" accessibleName="About Trustworthy Recommendation System" description="Trustworthy Recommendation System is a research prototype for transparent, preference-driven movie discovery."><div className="about-content"><section className="how-it-works"><h3>How it works</h3><p>{status?.backend_mode === "fixture" ? "The active demo backend ranks a small synthetic fixture catalog." : "The active recommendation backend ranks movies from MovieLens data."} You can explicitly apply favorite movies, include or avoid genres, and hide individual titles; TMDB supplies display artwork only and does not affect ranking.</p>{status?.backend_mode === "baseline" && <p>The active baseline uses genre-weighted popularity. Learned retrieval, learned reranking, and language-based preference interpretation are future capabilities, not features of this baseline.</p>}{status?.backend_mode === "fixture" && <p>Synthetic fixture results demonstrate the interface only; they are not MovieLens recommendations or measured model output.</p>}{status?.backend_mode === "learned" && <p>The configured learned backend is active. Language-based preference interpretation remains unavailable unless System details reports it as available.</p>}</section><p className="documentation-links"><a href="https://github.com/lephgrueber/ai-capstone-trustworthy-recsys" target="_blank" rel="noreferrer">Project repository</a></p><details className="system-details"><summary>System details</summary><section><p className="data-flow-note">The browser stores a random session identifier in session storage and sends each submitted preference snapshot to this application’s API. The interface does not create a user account. Artwork is resolved separately through TMDB when configured.</p><dl><dt>Model</dt><dd>{status?.model_label ?? "Unknown"}</dd><dt>Data</dt><dd>{status?.data_source_label ?? "Unknown"}</dd><dt>Backend mode</dt><dd>{status?.backend_mode ?? "Unknown"}</dd><dt>Text refinement</dt><dd>{status?.refinement_available ? "Available" : "Unavailable"}</dd><dt>Artwork</dt><dd>{status?.posters_configured ? "TMDB configured" : "TMDB not configured"}</dd>{result && <><dt>Retrieval</dt><dd>{result.retrieval.name} · {result.retrieval.version}</dd><dt>Reranking</dt><dd>{result.reranker.name} · {result.reranker.version}</dd><dt>Timing</dt><dd>{result.timings.total_ms.toFixed(2)} ms server · {browserMs?.toFixed(2)} ms browser</dd><dt>Request</dt><dd>{result.request_id.slice(0, 8)} · revision {result.preference_revision}</dd>{result.warnings.length > 0 && <><dt>Warnings</dt><dd>{result.warnings.join(" ")}</dd></>}</>}</dl></section>{status?.backend_mode === "baseline" && <p className="system-note">The active baseline uses genre-weighted popularity. No learned retrieval or reranking is active.</p>}{status?.backend_mode === "fixture" && <p className="system-note">This mode uses synthetic demo data and does not represent MovieLens recommendations.</p>}</details><section className="tmdb-credit"><a href="https://www.themoviedb.org" target="_blank" rel="noreferrer">{!tmdbLogoFailed ? <img src={TMDB_LOGO} alt="The Movie Database (TMDB)" onError={() => setTmdbLogoFailed(true)} /> : <span className="tmdb-text-fallback">The Movie Database (TMDB)</span>}</a><p>This product uses the TMDB API but is not endorsed or certified by TMDB.</p></section><p className="course-note">University of Pennsylvania · CIS 5980 AI Engineering Capstone</p></div></MovieDialog>
  </main>;
}
