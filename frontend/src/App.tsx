import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Ban, Check, ChevronDown, ChevronLeft, ChevronRight, Film, ImageOff, Info, LoaderCircle, Pencil, Plus, RotateCcw, Search, SlidersHorizontal, Star, Trash2, Undo2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { ApiError, getApplicationStatus, getPosters, getRecommendations, refine, searchCatalog, type ApplicationStatus, type CatalogMovie, type PreferenceState, type RecommendationResponse } from "./api/client";
import { Button } from "./components/ui/button";
import { MovieDialog } from "./components/ui/dialog";
import { VoiceInput } from "./components/VoiceInput";
import { FeedbackCollector } from "./feedback";
import { loadBrowserProfile, removeRating, saveBrowserProfile, setRating, type BrowserProfile } from "./profile";

const EMPTY: PreferenceState = { history_items: [], excluded_movie_ids: [], include_genres: [], exclude_genres: [], ratings: [] };
const TMDB_LOGO = `${import.meta.env.BASE_URL}tmdb-logo.svg`;
const GENERIC_RECOMMENDATION_FAILURE = "We couldn’t load movies. Please try again.";
type Artwork = { poster: string | null; backdrop: string | null; backdropSmall: string | null };
type Recommendation = RecommendationResponse["recommendations"][number];
type UndoSnapshot = { preferences: PreferenceState; temporaryHistory: string[] };

export function formatCatalogTitle(canonical: string) {
  const yearMatch = canonical.match(/\s+\((\d{4})\)$/);
  const year = yearMatch?.[1] ?? null;
  const withoutYear = yearMatch ? canonical.slice(0, yearMatch.index).trim() : canonical;
  const articleMatch = withoutYear.match(/^(.+),\s+(The|An|A)$/);
  return { title: articleMatch ? `${articleMatch[2]} ${articleMatch[1]}` : withoutYear, year };
}

function samePreferences(left: PreferenceState, right: PreferenceState) {
  const normalize = (value: PreferenceState) => ({
    history_items: value.history_items,
    excluded_movie_ids: value.excluded_movie_ids,
    include_genres: value.include_genres,
    exclude_genres: value.exclude_genres,
    ratings: value.ratings === null || value.ratings === undefined ? value.ratings : value.ratings.map((record) => ({
      movie_id: record.movie_id,
      rating: record.rating,
      rated_at: new Date(record.rated_at).toISOString(),
    })),
  });
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}

function summarizePreferences(value: PreferenceState) {
  const genreSummary = (label: string, genres: string[]) => genres.length
    ? `${label} ${genres.slice(0, 2).join(", ")}${genres.length > 2 ? ` +${genres.length - 2}` : ""}`
    : "";
  return [
    value.ratings?.length ? `${value.ratings.length} rating${value.ratings.length === 1 ? "" : "s"}` : "",
    value.history_items.length ? `${value.history_items.length} unrated favorite${value.history_items.length === 1 ? "" : "s"}` : "",
    genreSummary("Include", value.include_genres),
    genreSummary("Avoid", value.exclude_genres),
    value.excluded_movie_ids.length ? `${value.excluded_movie_ids.length} hidden this visit` : "",
  ].filter(Boolean).join(" · ") || "No personalization";
}

function sessionId() {
  const created = crypto.randomUUID();
  try {
    const stored = sessionStorage.getItem("trustworthy-recsys-session");
    if (stored) return stored;
    sessionStorage.setItem("trustworthy-recsys-session", created);
  } catch { /* A transient session ID still permits non-persistent use. */ }
  return created;
}

function hasPreferences(value: PreferenceState) {
  return (value.ratings?.length ?? 0) + value.history_items.length + value.excluded_movie_ids.length + value.include_genres.length + value.exclude_genres.length > 0;
}

function recommendationFailureMessage(error: unknown) {
  if (!(error instanceof ApiError)) return GENERIC_RECOMMENDATION_FAILURE;
  return error.message === `Request failed (${error.status})` ? GENERIC_RECOMMENDATION_FAILURE : error.message;
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
      aria-label={movie.title}
      data-movie-id={movie.movie_id}
      key={movie.movie_id}
      className="portrait-card"
      whileHover={reducedMotion ? undefined : { scale: 1.04, y: -4 }}
      transition={{ duration: .18, ease: "easeOut" }}
    ><button className="card-art" onClick={() => onDetail(movie)} aria-label={`View details for ${movie.title}`}><Portrait movie={movie} artwork={artwork[movie.movie_id]} /></button><div className="card-copy"><h3>{formatted.title}</h3><p>{formatted.year ?? "Year unavailable"}</p></div><div className="card-actions"><button onClick={() => onDetail(movie)} aria-label={`Details for ${movie.title}`} title="Details"><Info /></button><button disabled={favorite} onClick={() => onFavorite(movie)} aria-label={favorite ? `${movie.title} already has saved feedback` : `Rate ${movie.title}`} title={favorite ? "Already personalized" : "Rate movie"}><Star /></button><button onClick={() => onHide(movie)} aria-label={`Hide ${movie.title} for now`} title="Hide for now"><Ban /></button></div></motion.article>;
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
  const loaded = useMemo(() => loadBrowserProfile(), []);
  const initialPreferences = useMemo<PreferenceState>(() => ({
    ...EMPTY,
    history_items: [...loaded.profile.legacy_favorites],
    ratings: loaded.profile.ratings.map(({ movie_id, rating, rated_at }) => ({ movie_id, rating, rated_at })),
  }), [loaded.profile]);
  const [profile, setProfile] = useState<BrowserProfile>(loaded.profile);
  const [profilePersistent, setProfilePersistent] = useState(loaded.persistent);
  const [profileMessage, setProfileMessage] = useState<string | null>(loaded.warning);
  const [status, setStatus] = useState<ApplicationStatus | null>(null);
  const [preferences, setPreferences] = useState<PreferenceState>(initialPreferences);
  const preferencesRef = useRef(initialPreferences);
  const [revision, setRevision] = useState(0);
  const [undoStack, setUndoStack] = useState<UndoSnapshot[]>([]);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState<string | null>(null);
  const [browserMs, setBrowserMs] = useState<number | null>(null);
  const [personalizeOpen, setPersonalizeOpen] = useState(false);
  const [ratingsOpen, setRatingsOpen] = useState(false);
  const [detail, setDetail] = useState<CatalogMovie | null>(null);
  const [ratingMovie, setRatingMovie] = useState<CatalogMovie | null>(null);
  const [ratingDraft, setRatingDraft] = useState(4);
  const [ratingSaveError, setRatingSaveError] = useState<string | null>(null);
  const [ratingNoticeActive, setRatingNoticeActive] = useState(false);
  const [ratingNoticeRefresh, setRatingNoticeRefresh] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [genreDialog, setGenreDialog] = useState<"include" | "avoid" | null>(null);
  const [query, setQuery] = useState("");
  const [catalog, setCatalog] = useState<CatalogMovie[]>([]);
  const [knownMovies, setKnownMovies] = useState<Record<string, CatalogMovie>>(loaded.profile.movies);
  const [artwork, setArtwork] = useState<Record<string, Artwork>>({});
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeSearchIndex, setActiveSearchIndex] = useState(-1);
  const [searchMode, setSearchMode] = useState<"rate" | "recommend">("rate");
  const [refinement, setRefinement] = useState("");
  const [refinementMessage, setRefinementMessage] = useState<string | null>(null);
  const [refining, setRefining] = useState(false);
  const [tmdbLogoFailed, setTmdbLogoFailed] = useState(false);
  const recommendationRequest = useRef<AbortController | null>(null);
  const searchRequest = useRef<AbortController | null>(null);
  const searchDismissed = useRef(false);
  const searchInput = useRef<HTMLInputElement | null>(null);
  const ratingReturnFocus = useRef<HTMLElement | null>(null);
  const ratingFallbackFocus = useRef<HTMLElement | null>(null);
  const searchResultRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const requestSequence = useRef(0);
  const refinementSequence = useRef(0);
  const initialRequested = useRef(false);
  const requestedArtwork = useRef(new Set<string>());
  const revisionRef = useRef(0);
  const temporaryHistoryRef = useRef<string[]>([]);
  const ratingReturnToManager = useRef(false);
  const sid = useMemo(sessionId, []);
  const feedbackCollector = useRef<FeedbackCollector | null>(null);
  const featuredElement = useRef<HTMLElement | null>(null);
  const assignmentKeys = useRef(new Map<number, string>());

  useEffect(() => { feedbackCollector.current = new FeedbackCollector(); return () => feedbackCollector.current?.stop(); }, []);
  useEffect(() => {
    const assignment = result?.feedback; const element = featuredElement.current;
    if (!assignment || !element || !("IntersectionObserver" in window)) return;
    let visible = false; let timer: ReturnType<typeof setTimeout> | undefined;
    const update = () => { clearTimeout(timer); if (visible && document.visibilityState === "visible") timer = setTimeout(() => feedbackCollector.current?.record(assignment, "display"), 1000); };
    const observer = new IntersectionObserver(([entry]) => { visible = entry.intersectionRatio >= .5; update(); }, { threshold: [.5] });
    observer.observe(element); document.addEventListener("visibilitychange", update);
    return () => { observer.disconnect(); clearTimeout(timer); document.removeEventListener("visibilitychange", update); };
  }, [result?.feedback?.decision_id]);
  useEffect(() => { const controller = new AbortController(); void getApplicationStatus(controller.signal).then(setStatus).catch(() => undefined); return () => controller.abort(); }, []);

  const enrichArtwork = useCallback(async (ids: string[]) => {
    const pending = [...new Set(ids)].filter((id) => !requestedArtwork.current.has(id)).slice(0, 30);
    if (!pending.length) return;
    pending.forEach((id) => requestedArtwork.current.add(id));
    try {
      const response = await getPosters(pending);
      setArtwork((current) => ({ ...current, ...Object.fromEntries(response.items.map((item) => [item.movie_id, { poster: item.poster_url ?? null, backdrop: item.backdrop_url ?? null, backdropSmall: item.backdrop_url_small ?? null }])) }));
    } catch { pending.forEach((id) => requestedArtwork.current.delete(id)); }
  }, []);

  const recommend = useCallback(async (state: PreferenceState, stateRevision: number, temporaryHistory = temporaryHistoryRef.current) => {
    recommendationRequest.current?.abort();
    const controller = new AbortController(); recommendationRequest.current = controller;
    const sequence = ++requestSequence.current; const started = performance.now();
    setLoading(true); setFailure(null);
    try {
      if (!assignmentKeys.current.has(stateRevision)) assignmentKeys.current.set(stateRevision, crypto.randomUUID());
      const response = await getRecommendations({ assignment_key: assignmentKeys.current.get(stateRevision), session_id: sid, preference_revision: stateRevision, preferences: state, result_count: 12, known_user_id: null, temporary_history_items: temporaryHistory }, controller.signal);
      if (sequence !== requestSequence.current || response.preference_revision !== stateRevision) return false;
      if (response.feedback) feedbackCollector.current?.receive(response.feedback);
      setResult(response);
      setKnownMovies((current) => ({ ...current, ...Object.fromEntries(response.recommendations.map(({ movie }) => [movie.movie_id, movie])) }));
      void enrichArtwork(response.recommendations.map(({ movie }) => movie.movie_id));
      setBrowserMs(performance.now() - started);
      return true;
    } catch (error) {
      if ((error as Error).name !== "AbortError" && sequence === requestSequence.current) setFailure(recommendationFailureMessage(error));
      return false;
    } finally { if (sequence === requestSequence.current) setLoading(false); }
  }, [enrichArtwork, sid]);

  useEffect(() => { if (!initialRequested.current) { initialRequested.current = true; void recommend(initialPreferences, 0, []); } }, [initialPreferences, recommend]);
  useEffect(() => {
    if (!ratingNoticeActive || loading || failure) return;
    const timer = window.setTimeout(() => setRatingNoticeActive(false), 4000);
    return () => window.clearTimeout(timer);
  }, [failure, loading, ratingNoticeActive]);

  const loadCatalog = useCallback(async (text: string) => {
    searchRequest.current?.abort();
    if (!text.trim()) { setCatalog([]); setSearching(false); setSearchError(null); setSearchOpen(false); return; }
    const controller = new AbortController(); searchRequest.current = controller; setSearching(true); setSearchError(null);
    try {
      const response = await searchCatalog(text, controller.signal);
      setCatalog(response.items); if (!searchDismissed.current) setSearchOpen(true); setKnownMovies((current) => ({ ...current, ...Object.fromEntries(response.items.map((movie) => [movie.movie_id, movie])) }));
      void enrichArtwork(response.items.map((movie) => movie.movie_id));
    } catch (error) { if ((error as Error).name !== "AbortError") setSearchError((error as Error).message); }
    finally { if (!controller.signal.aborted) setSearching(false); }
  }, [enrichArtwork]);
  useEffect(() => { const timer = window.setTimeout(() => void loadCatalog(query), 180); return () => window.clearTimeout(timer); }, [query, loadCatalog]);
  useEffect(() => { setActiveSearchIndex(-1); searchResultRefs.current = searchResultRefs.current.slice(0, catalog.length); }, [catalog]);

  const persist = (next: BrowserProfile) => {
    const saved = saveBrowserProfile(next);
    setProfilePersistent(saved);
    setProfileMessage(saved ? null : "We couldn’t save changes in this browser.");
    if (saved) setProfile(next);
    return saved;
  };
  const changePreferences = (next: PreferenceState, remember = true, nextTemporaryHistory = temporaryHistoryRef.current) => {
    const current = preferencesRef.current;
    const currentTemporaryHistory = [...temporaryHistoryRef.current];
    const sameTemporaryHistory = JSON.stringify(nextTemporaryHistory) === JSON.stringify(temporaryHistoryRef.current);
    if (samePreferences(next, current) && sameTemporaryHistory) return revisionRef.current;
    recommendationRequest.current?.abort(); requestSequence.current += 1; setLoading(false);
    revisionRef.current += 1; preferencesRef.current = next; temporaryHistoryRef.current = [...nextTemporaryHistory];
    if (remember) setUndoStack((stack) => [...stack.slice(-19), { preferences: current, temporaryHistory: currentTemporaryHistory }]);
    setPreferences(next); setRevision(revisionRef.current); setFailure(null);
    void recommend(next, revisionRef.current, nextTemporaryHistory);
    return revisionRef.current;
  };
  const profileForPreferences = (state: PreferenceState): BrowserProfile => ({
    ...profile,
    ratings: (state.ratings ?? []).map(({ movie_id, rating, rated_at }) => ({ movie_id, rating, rated_at })),
    legacy_favorites: [...state.history_items],
  });
  const applyProfile = (next: BrowserProfile) => {
    if (!persist(next)) return false;
    const state = preferencesRef.current;
    changePreferences({
      ...state,
      history_items: [...next.legacy_favorites],
      ratings: next.ratings.map(({ movie_id, rating, rated_at }) => ({ movie_id, rating, rated_at })),
    });
    return true;
  };
  const undo = () => {
    const previous = undoStack.at(-1); if (!previous) return;
    if (!persist(profileForPreferences(previous.preferences))) return;
    setUndoStack((stack) => stack.slice(0, -1));
    changePreferences(previous.preferences, false, previous.temporaryHistory);
  };
  const resetAll = () => {
    const nextProfile = { ...profile, ratings: [], legacy_favorites: [] };
    if (persist(nextProfile)) changePreferences({ ...EMPTY }, true, []);
  };
  const clearRatings = () => {
    const next = { ...profile, ratings: [] };
    if (persist(next)) changePreferences({ ...preferencesRef.current, ratings: [] });
  };
  const openRating = (movie: CatalogMovie) => {
    const existing = profile.ratings.find((record) => record.movie_id === movie.movie_id);
    ratingReturnToManager.current = ratingsOpen;
    ratingReturnFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    ratingFallbackFocus.current = searchInput.current;
    searchDismissed.current = true;
    setKnownMovies((current) => ({ ...current, [movie.movie_id]: movie }));
    setDetail(null); setRatingsOpen(false); setSearchOpen(false); setRatingMovie(movie); setRatingDraft(existing?.rating ?? 4); setRatingSaveError(null);
  };
  const closeRating = () => {
    setRatingMovie(null);
    window.setTimeout(() => {
      if (ratingReturnFocus.current?.isConnected) ratingReturnFocus.current.focus();
      else if (ratingFallbackFocus.current?.isConnected) ratingFallbackFocus.current.focus();
    }, 0);
  };
  const saveRating = () => {
    if (!ratingMovie) return;
    const updated = setRating(profile, ratingMovie, ratingDraft);
    if (updated.changed && !applyProfile(updated.profile)) {
      setRatingSaveError("We couldn’t save this rating in your browser. Your selection is still here; try again.");
      return;
    }
    setRatingNoticeActive(true); setRatingNoticeRefresh(updated.changed); setRatingSaveError(null); setRatingMovie(null);
    window.setTimeout(() => {
      if (ratingReturnToManager.current) setRatingsOpen(true);
      else if (ratingReturnFocus.current?.isConnected) ratingReturnFocus.current.focus();
      else if (ratingFallbackFocus.current?.isConnected) ratingFallbackFocus.current.focus();
    }, 0);
  };
  const deleteRating = (movieId: string) => applyProfile(removeRating(profile, movieId));
  const addFavorite = (movie: CatalogMovie) => {
    if (profile.ratings.some((record) => record.movie_id === movie.movie_id) || profile.legacy_favorites.includes(movie.movie_id)) return;
    setKnownMovies((current) => ({ ...current, [movie.movie_id]: movie }));
    applyProfile({ ...profile, legacy_favorites: [...profile.legacy_favorites, movie.movie_id], movies: { ...profile.movies, [movie.movie_id]: movie } });
  };
  const removeFavorite = (movieId: string) => applyProfile({ ...profile, legacy_favorites: profile.legacy_favorites.filter((id) => id !== movieId) });
  const toggleGenre = (genre: string, kind: "include" | "avoid") => {
    const state = preferencesRef.current;
    if (kind === "include") {
      if (state.exclude_genres.includes(genre)) return;
      changePreferences({ ...state, include_genres: state.include_genres.includes(genre) ? state.include_genres.filter((value) => value !== genre) : [...state.include_genres, genre] });
    } else {
      if (state.include_genres.includes(genre)) return;
      changePreferences({ ...state, exclude_genres: state.exclude_genres.includes(genre) ? state.exclude_genres.filter((value) => value !== genre) : [...state.exclude_genres, genre] });
    }
  };
  const hideMovie = (movie: CatalogMovie) => { const state = preferencesRef.current; if (!state.excluded_movie_ids.includes(movie.movie_id)) changePreferences({ ...state, excluded_movie_ids: [...state.excluded_movie_ids, movie.movie_id] }); };
  const unhideMovie = (movieId: string) => { const state = preferencesRef.current; if (state.excluded_movie_ids.includes(movieId)) changePreferences({ ...state, excluded_movie_ids: state.excluded_movie_ids.filter((id) => id !== movieId) }); };
  const searchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Escape") { searchDismissed.current = true; searchRequest.current?.abort(); setSearchOpen(false); setActiveSearchIndex(-1); return; }
    if (event.key === "ArrowDown" && searchOpen && catalog.length) {
      event.preventDefault(); setActiveSearchIndex(0); searchResultRefs.current[0]?.focus();
    }
  };
  const resultKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === "Escape") { event.preventDefault(); searchDismissed.current = true; searchRequest.current?.abort(); setSearchOpen(false); setActiveSearchIndex(-1); searchInput.current?.focus(); return; }
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? catalog.length - 1
      : event.key === "ArrowDown" ? Math.min(catalog.length - 1, index + 1) : Math.max(0, index - 1);
    setActiveSearchIndex(next); searchResultRefs.current[next]?.focus();
  };
  const acceptTitleTranscript = (text: string) => {
    const title = text.slice(0, 100);
    searchDismissed.current = false;
    setQuery(title);
    setSearching(Boolean(title.trim()));
    setSearchOpen(Boolean(title.trim()));
  };
  const searchModeKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const next = searchMode === "rate" ? "recommend" : "rate";
    setSearchMode(next); setSearchOpen(false);
    const tabs = event.currentTarget.querySelectorAll<HTMLButtonElement>("[role=tab]");
    tabs[next === "rate" ? 0 : 1]?.focus();
  };
  const submitRefinement = async () => {
    if (!status?.refinement_available || !refinement.trim() || refining) return;
    const sourceRevision = revisionRef.current; const sequence = ++refinementSequence.current;
    setRefining(true); setRefinementMessage(null);
    try {
      const response = await refine(sid, sourceRevision, preferencesRef.current, refinement.trim());
      if (sequence !== refinementSequence.current) return;
      if (sourceRevision !== revisionRef.current) { setRefinementMessage("Your preferences changed while refining. Please submit again."); return; }
      if (response.outcome === "applied") {
        const resolved = response.resolved_movies ?? [];
        setKnownMovies((current) => ({ ...current, ...Object.fromEntries(resolved.map((movie) => [movie.movie_id, { ...movie, genres: movie.genres ?? [], poster_url: movie.poster_url ?? null }])) }));
        changePreferences(response.preferences, true, resolved.map((movie) => movie.movie_id));
        setRefinementMessage(response.summary);
      }
      else setRefinementMessage(response.outcome === "clarification" ? response.question : response.message);
    } catch (error) { if (sequence === refinementSequence.current) setRefinementMessage((error as Error).message); }
    finally { if (sequence === refinementSequence.current) setRefining(false); }
  };

  const dirty = result !== null && !samePreferences(preferences, result.applied_preferences);
  const displayed = result?.recommendations ?? [];
  const feature = displayed[0]?.movie ?? null;
  const grid = result?.feedback ? displayed.slice(1) : displayed;
  const selected = profile.legacy_favorites.map((id) => knownMovies[id] ?? profile.movies[id]).filter(Boolean) as CatalogMovie[];
  const savedRatings = profile.ratings.map((record) => ({ record, movie: knownMovies[record.movie_id] ?? profile.movies[record.movie_id] }));
  const hiddenMovies = preferences.excluded_movie_ids.map((movieId) => ({ movieId, movie: knownMovies[movieId] ?? profile.movies[movieId] }));
  const genres = status?.available_genres ?? [...new Set(Object.values(knownMovies).flatMap((movie) => movie.genres))].sort();
  const applied = result?.applied_preferences;
  const appliedSummary = summarizePreferences(applied ?? EMPTY);
  const appliedHasPreferences = Boolean(applied && hasPreferences(applied));
  const showPreferenceSummary = appliedHasPreferences || dirty || hasPreferences(preferences);
  const activeChips = [...preferences.include_genres.map((genre) => ({ kind: "include", label: `Include: ${genre}`, remove: () => toggleGenre(genre, "include") })), ...preferences.exclude_genres.map((genre) => ({ kind: "avoid", label: `Avoid: ${genre}`, remove: () => toggleGenre(genre, "avoid") }))];
  const formattedDetail = detail ? formatCatalogTitle(detail.title) : null;
  const ratingNotice = ratingNoticeActive ? <div className={`rating-notice ${failure && ratingNoticeRefresh ? "error" : ""}`} role={failure && ratingNoticeRefresh ? "alert" : "status"}><span>{ratingNoticeRefresh && loading ? "Rating saved. Updating recommendations…" : ratingNoticeRefresh && failure ? "Rating saved. Recommendations couldn’t update." : "Rating saved."}</span>{ratingNoticeRefresh && failure && !loading ? <Button onClick={() => void recommend(preferencesRef.current, revisionRef.current)}>Retry</Button> : null}</div> : null;

  return <main id="top" className="app-shell">
    <header className="site-header"><div className="header-inner"><a href="#top" className="brand">Trustworthy Recommendation System</a><nav className="header-actions" aria-label="Primary"><Button variant="ghost" onClick={() => setPersonalizeOpen(true)}><SlidersHorizontal size={16} /> Preferences{dirty && <span className="pending-dot" aria-label="Updating preferences" />}</Button><Button variant="quiet" onClick={() => setAboutOpen(true)}>About</Button></nav></div></header>

    <section className="main-search" aria-label="Movie search and recommendations">
      <div className="search-modes" role="tablist" aria-label="Search purpose" onKeyDown={searchModeKeyDown}><button type="button" role="tab" aria-selected={searchMode === "rate"} aria-controls="rate-search-panel" tabIndex={searchMode === "rate" ? 0 : -1} onClick={() => setSearchMode("rate")}>Rate a movie</button><button type="button" role="tab" aria-selected={searchMode === "recommend"} aria-controls="recommend-search-panel" tabIndex={searchMode === "recommend" ? 0 : -1} onClick={() => { setSearchMode("recommend"); setSearchOpen(false); }}>Find recommendations</button></div>
      {searchMode === "rate" ? <form id="rate-search-panel" role="search" aria-label="Rate a movie" onSubmit={(event) => event.preventDefault()}>
        <div className="main-search-anchor">
          <div className="main-search-bar"><Search aria-hidden="true" className="search-leading-icon" /><input ref={searchInput} role="combobox" aria-autocomplete="list" aria-expanded={searchOpen && Boolean(query.trim())} aria-controls="movie-search-results" aria-label="Search for a movie by title or release year" value={query} maxLength={100} onFocus={() => { if (!searchDismissed.current && query.trim()) setSearchOpen(true); }} onClick={() => { searchDismissed.current = false; if (query.trim()) setSearchOpen(true); }} onKeyDown={searchKeyDown} onChange={(event) => { searchDismissed.current = false; setQuery(event.target.value); setSearching(Boolean(event.target.value.trim())); setSearchOpen(true); }} placeholder="Search movie titles…" />{searching ? <LoaderCircle className="animate-spin search-status-icon" aria-label="Searching movies" /> : null}<VoiceInput purpose="movie-title" sessionKey={searchMode} onTranscript={acceptTitleTranscript} /></div>
          {query.trim() && searchOpen && <ul id="movie-search-results" className="main-search-results" aria-label="Movie search results">{searchError && <li><p role="alert">{searchError} <button onClick={() => void loadCatalog(query)}>Retry</button></p></li>}{!searching && !searchError && catalog.map((movie, index) => { const formatted = formatCatalogTitle(movie.title); return <li key={movie.movie_id}><button ref={(node) => { searchResultRefs.current[index] = node; }} className={activeSearchIndex === index ? "is-active" : ""} aria-current={activeSearchIndex === index ? "true" : undefined} onFocus={() => setActiveSearchIndex(index)} onKeyDown={(event) => resultKeyDown(event, index)} onClick={() => openRating(movie)}><span className="search-result-poster"><Portrait movie={movie} artwork={artwork[movie.movie_id]} compact /></span><span><strong>{formatted.title}</strong><small>{[formatted.year, ...movie.genres].filter(Boolean).join(" · ")}</small></span><em>{profile.ratings.some((record) => record.movie_id === movie.movie_id) ? "Edit rating" : "Rate"}</em></button></li>; })}{!searching && !searchError && !catalog.length && <li><p>No supported movies match “{query}”.</p></li>}</ul>}
        </div>
      </form> : <form id="recommend-search-panel" role="search" aria-label="Find recommendations" onSubmit={(event) => { event.preventDefault(); void submitRefinement(); }}><div className="main-search-bar"><Search aria-hidden="true" className="search-leading-icon" /><input aria-label="Describe the movies you want" value={refinement} maxLength={500} disabled={refining || !status?.refinement_available} onChange={(event) => setRefinement(event.target.value)} placeholder="Try comedy without horror…" /><VoiceInput purpose="recommendations" sessionKey={searchMode} onTranscript={setRefinement} disabled={refining || !status?.refinement_available} /><button className="search-submit" type="submit" disabled={refining || !status?.refinement_available || !refinement.trim()} aria-label="Find recommendations">{refining ? <LoaderCircle className="animate-spin" /> : <Search />}</button></div>{refinementMessage && <p className="main-search-message" role="status">{refinementMessage}</p>}</form>}
      {profile.ratings.length > 0 ? <div className="ratings-summary" role="region" aria-label="Your ratings"><strong>Your ratings <span aria-hidden="true">·</span> {profile.ratings.length}</strong><Button variant="quiet" onClick={() => setRatingsOpen(true)}>Manage ratings</Button></div> : null}
      {!ratingsOpen ? ratingNotice : null}
    </section>

    {loading && !result && <div className="loading-state" role="status" aria-label="Loading movies"><section className="hero-skeleton" aria-hidden="true"><div /><div className="skeleton-copy"><span /><span /><span /></div></section><section className="loading-browse" aria-hidden="true"><span className="loading-heading" /><div className="loading-cards">{Array.from({ length: 6 }, (_, index) => <div className="loading-card" key={index}><span className="loading-card-art" /><span className="loading-line" /><span className="loading-line short" /></div>)}</div></section></div>}
    {!result && failure && <section className="first-error" role="alert"><Film /><h1>Movies are unavailable right now</h1><p>{failure}</p><p>Your saved ratings are unchanged.</p><Button onClick={() => void recommend(preferencesRef.current, revisionRef.current)}>Retry</Button></section>}
    {feature && <motion.section ref={featuredElement} key={`${result?.request_id}-${feature.movie_id}`} className="feature-hero" initial={reducedMotion ? false : { opacity: 0 }} animate={{ opacity: 1 }}><div className="feature-media"><HeroArtwork movie={feature} artwork={artwork[feature.movie_id]} /></div><div className="feature-scrim" /><div className={`feature-content ${formatCatalogTitle(feature.title).title.length > 55 ? "long-title" : ""}`}><p className="feature-context">{applied && hasPreferences(applied) ? "Your recommendations" : "Explore movies"}</p><h1>{formatCatalogTitle(feature.title).title}</h1><p className="feature-genres">{[formatCatalogTitle(feature.title).year, ...feature.genres].filter(Boolean).join(" · ") || "Metadata unavailable"}</p><div className="feature-actions"><Button onClick={() => { if (result?.feedback) feedbackCollector.current?.record(result.feedback, "details_open"); setDetail(feature); }}><Info /> Details</Button><Button variant="ghost" onClick={() => openRating(feature)}><Star /> Rate</Button><Button variant="quiet" disabled={profile.legacy_favorites.includes(feature.movie_id) || profile.ratings.some((record) => record.movie_id === feature.movie_id)} onClick={() => { if (result?.feedback) feedbackCollector.current?.record(result.feedback, "favorite_add"); addFavorite(feature); }}><Plus /> Add favorite</Button>{result?.feedback && <Button variant="quiet" disabled={preferences.excluded_movie_ids.includes(feature.movie_id)} onClick={() => { feedbackCollector.current?.record(result.feedback!, "hide"); hideMovie(feature); }}><Ban /> Hide for now</Button>}</div></div></motion.section>}

    {result && <div className={`browse-content ${!feature ? "no-feature" : ""} ${!showPreferenceSummary ? "no-summary" : ""}`}>{showPreferenceSummary && <div className="personalization-summary"><SlidersHorizontal className="summary-icon" aria-hidden="true" /><div><strong>{appliedHasPreferences ? appliedSummary : "Current results use no personalization"}</strong>{loading && <span>Updating—previous results remain visible.</span>}{dirty && !loading && !failure && <span>Waiting for the latest preference revision.</span>}{failure && !ratingNoticeActive && <span role="alert">Update failed: {failure} Previous results and saved ratings remain available.</span>}</div><Button variant="quiet" onClick={() => setPersonalizeOpen(true)}><Pencil size={15} /> Edit</Button><Button variant="quiet" onClick={resetAll}><RotateCcw size={15} /> Reset all personalization</Button>{failure && !loading && !ratingNoticeActive && <Button onClick={() => void recommend(preferencesRef.current, revisionRef.current)}>Retry</Button>}</div>}
      {displayed.length > 0 ? <section className="recommendation-section" aria-labelledby="recommendations-heading"><div className="row-heading"><h2 id="recommendations-heading">{appliedHasPreferences ? "Your recommendations" : "Explore movies"}</h2></div><MovieCarousel ariaLabel={appliedHasPreferences ? "Your recommendations" : "Explore movies"} items={grid.slice(0, 8)} artwork={artwork} favoriteIds={[...preferences.history_items, ...(preferences.ratings ?? []).map((record) => record.movie_id)]} reducedMotion={reducedMotion} onDetail={setDetail} onFavorite={openRating} onHide={hideMovie} />{grid.length > 8 && <MovieCarousel ariaLabel={`${appliedHasPreferences ? "Your recommendations" : "Explore movies"} continuation`} items={grid.slice(8)} artwork={artwork} favoriteIds={[...preferences.history_items, ...(preferences.ratings ?? []).map((record) => record.movie_id)]} reducedMotion={reducedMotion} onDetail={setDetail} onFavorite={openRating} onHide={hideMovie} continuation />}</section> : <section className="no-matches"><Film /><h2>No recommendations found</h2><p>Try adjusting your genre preferences or removing a filter.</p><div className="empty-actions"><Button variant="ghost" onClick={() => setPersonalizeOpen(true)}>Edit preferences</Button><Button onClick={resetAll}><RotateCcw /> Reset all personalization</Button></div></section>}
    </div>}
    <footer>Trustworthy Recommendation System · A movie-discovery research prototype.{status?.feedback_mode && status.feedback_mode !== "off" && <span> The separate featured-recommendation feedback pilot is active. Personal star ratings are not research reward events.</span>}{status?.backend_mode === "fixture" && <span> Synthetic demo data.</span>}</footer>

    <MovieDialog open={personalizeOpen} onOpenChange={setPersonalizeOpen} title="Preferences" accessibleName="Recommendation preferences" description="Adjust your recommendation preferences." closeLabel="Close preferences" contentClassName="personalize-panel"><div className="personalize-drawer">
      <div className="drawer-section"><h3>Genre preferences</h3><p>Changes refresh recommendations automatically.</p><div className="drawer-genre-buttons"><Button variant="ghost" aria-label="Include genres" onClick={() => setGenreDialog("include")}><Check /> <span>Include</span> <ChevronDown /></Button><Button variant="ghost" aria-label="Avoid genres" onClick={() => setGenreDialog("avoid")}><Ban /> <span>Avoid</span> <ChevronDown /></Button></div><div className="drawer-chips">{activeChips.map((chip) => <button className={`preference-chip ${chip.kind}`} key={chip.label} onClick={chip.remove}>{chip.label}<X /></button>)}{!activeChips.length && <span>No genre filters selected</span>}</div></div>
      {selected.length > 0 && <div className="drawer-section"><h3>Unrated favorites <span>{selected.length}</span></h3><p>Rate these movies to complete your ratings.</p><div className="drawer-favorites"><AnimatePresence initial={false}>{selected.map((movie) => <motion.div layout key={movie.movie_id} className="drawer-favorite"><span><Portrait movie={movie} artwork={artwork[movie.movie_id]} compact /></span><strong>{movie.title}</strong><Button variant="quiet" onClick={() => openRating(movie)}>Rate</Button><button aria-label={`Remove favorite ${movie.title}`} onClick={() => removeFavorite(movie.movie_id)}><X /></button></motion.div>)}</AnimatePresence></div></div>}
      {hiddenMovies.length > 0 && <div className="drawer-section"><h3>Hidden this visit <span>{hiddenMovies.length}</span></h3><p>These movies can return after a full page reload.</p><div className="drawer-favorites">{hiddenMovies.map(({ movieId, movie }) => { const formatted = movie ? formatCatalogTitle(movie.title) : null; const label = formatted?.title ?? `Movie ${movieId}`; return <div key={movieId} className="drawer-favorite"><span>{movie ? <Portrait movie={movie} artwork={artwork[movieId]} compact /> : <span className="drawer-movie-fallback" aria-label={`No artwork available for Movie ${movieId}`}><Film aria-hidden="true" /></span>}</span><div className="drawer-favorite-copy"><strong>{label}</strong><small>{formatted?.year ?? (movie ? "Year unavailable" : `ID ${movieId}`)}</small></div><Button variant="quiet" aria-label={`Unhide ${label}`} onClick={() => unhideMovie(movieId)}>Unhide</Button></div>; })}</div></div>}
      <div className="drawer-spacer" />{loading && <p className="drawer-pending" role="status">Updating recommendations while this dialog remains open.</p>}{failure && <p className="drawer-error" role="alert">{failure} Your edits and ratings are preserved.</p>}<div className="drawer-actions"><Button variant="quiet" disabled={!undoStack.length} onClick={undo}><Undo2 /> Undo</Button><Button variant="quiet" onClick={resetAll}><RotateCcw /> Reset all personalization</Button>{failure && !loading && <Button onClick={() => void recommend(preferencesRef.current, revisionRef.current)}>Retry</Button>}</div></div></MovieDialog>

    <MovieDialog open={ratingsOpen} onOpenChange={setRatingsOpen} title={`Your ratings · ${profile.ratings.length}`} description={profilePersistent ? "Saved in this browser." : "Browser storage is unavailable."} closeLabel="Close ratings" presentation="modal" contentClassName="ratings-manager-panel"><div className="ratings-manager">
      {ratingsOpen ? ratingNotice : null}
      {profileMessage && <p className="profile-warning" role="status">{profileMessage}</p>}
      <div className="rating-list">{savedRatings.map(({ record, movie }) => { const formatted = movie ? formatCatalogTitle(movie.title) : null; return <article key={record.movie_id}><span className="managed-rating-poster">{movie ? <Portrait movie={movie} artwork={artwork[record.movie_id]} compact /> : <Film aria-hidden="true" />}</span><div className="managed-rating-copy"><strong>{formatted?.title ?? `Movie ${record.movie_id}`}</strong><small>{formatted?.year ?? "Year unavailable"}</small></div><span className="managed-rating-value">{record.rating.toFixed(1)} ★</span><div className="managed-rating-actions"><Button variant="quiet" onClick={() => movie && openRating(movie)} disabled={!movie}><Pencil size={14} /> Edit</Button><Button variant="quiet" onClick={() => deleteRating(record.movie_id)}><X size={14} /> Remove</Button></div></article>; })}{!savedRatings.length && <p className="empty-ratings">No saved ratings.</p>}</div>
      {profile.ratings.length > 0 && <div className="ratings-manager-footer"><Button variant="quiet" onClick={clearRatings}><Trash2 size={15} /> Clear saved ratings</Button></div>}
    </div></MovieDialog>

    <MovieDialog open={genreDialog !== null} onOpenChange={(open) => !open && setGenreDialog(null)} title={genreDialog === "include" ? "Include genres" : "Avoid genres"} description={genreDialog === "include" ? "Recommendations will match at least one selected genre." : "Recommendations will match none of the selected genres."} closeLabel="Close genre selector">{genreDialog && <div className="genre-dialog-content"><GenreChoices kind={genreDialog} genres={genres} selected={genreDialog === "include" ? preferences.include_genres : preferences.exclude_genres} blocked={genreDialog === "include" ? preferences.exclude_genres : preferences.include_genres} onToggle={toggleGenre} /><Button onClick={() => setGenreDialog(null)}>Done</Button></div>}</MovieDialog>
    <MovieDialog open={ratingMovie !== null} onOpenChange={(open) => !open && closeRating()} title={ratingMovie ? `Rate ${formatCatalogTitle(ratingMovie.title).title}` : "Rate movie"} description="Choose your rating." closeLabel="Close rating" presentation="modal" contentClassName="rating-panel"><div className="rating-editor">{ratingMovie && <><span className="rating-editor-poster"><Portrait movie={ratingMovie} artwork={artwork[ratingMovie.movie_id]} compact /></span><div className="rating-control"><label htmlFor="your-rating">Your rating</label><select id="your-rating" value={ratingDraft} onChange={(event) => { setRatingDraft(Number(event.target.value)); setRatingSaveError(null); }}>{Array.from({ length: 10 }, (_, index) => (index + 1) / 2).map((value) => <option value={value} key={value}>{value.toFixed(1)} stars</option>)}</select><Button className="rating-save" onClick={saveRating}><Star /> Save rating</Button></div>{ratingSaveError && <p className="rating-save-error" role="alert">{ratingSaveError}</p>}</>}</div></MovieDialog>
    <MovieDialog open={detail !== null} onOpenChange={(open) => !open && setDetail(null)} title={formattedDetail?.title ?? "Movie details"} description="Movie details and artwork information. Opening details does not save feedback." contentClassName="detail-panel">{detail && <div className="detail-content"><div className="detail-poster"><Portrait movie={detail} artwork={artwork[detail.movie_id]} /></div><dl>{formattedDetail?.year && <><dt>Year</dt><dd>{formattedDetail.year}</dd></>}<dt>Genres</dt><dd>{detail.genres.join(", ") || "Unavailable"}</dd><dt>Artwork</dt><dd>{artwork[detail.movie_id]?.poster || artwork[detail.movie_id]?.backdrop ? "Provided by TMDB" : "Not available"}</dd></dl><Button onClick={() => openRating(detail)}><Star /> Rate this movie</Button></div>}</MovieDialog>
    <MovieDialog open={aboutOpen} onOpenChange={setAboutOpen} title="About" accessibleName="About Trustworthy Recommendation System" description="Trustworthy Recommendation System is a research prototype for transparent, preference-driven movie discovery."><div className="about-content"><section className="how-it-works"><h3>How it works</h3><p>{status?.backend_mode === "fixture" ? "The active demo backend ranks a small synthetic fixture catalog." : "The active recommendation backend ranks movies from MovieLens data."} Explicit ratings are stored locally in this browser and sent with recommendation requests. Rated movies are excluded from results; ratings of 4 or more also provide positive history. TMDB supplies display artwork only.</p>{status?.backend_mode === "baseline" && <p className="capability-note">The active baseline uses genre-weighted popularity. Learned retrieval and reranking are not active.</p>}{status?.backend_mode === "fixture" && <p className="capability-note">Synthetic fixture results demonstrate the interface only.</p>}{status?.backend_mode === "learned" && <p className="capability-note">The configured two-tower retrieval and LambdaRank reranker are active.</p>}</section><p className="documentation-links"><a href="https://github.com/lephgrueber/ai-capstone-trustworthy-recsys" target="_blank" rel="noreferrer">Project repository</a></p><details className="system-details"><summary>System details</summary><section><p className="data-flow-note">The browser restores the versioned rating profile before its first recommendation request. Numeric ratings and their genuine submission times remain separate from the optional featured-slot research event stream. No account or cross-device synchronization is provided.</p><dl><dt>Model</dt><dd>{status?.model_label ?? "Unknown"}</dd><dt>Data</dt><dd>{status?.data_source_label ?? "Unknown"}</dd><dt>Backend mode</dt><dd>{status?.backend_mode ?? "Unknown"}</dd><dt>Genre refinement</dt><dd>{status?.refinement_available ? "Available" : "Unavailable"}</dd><dt>Artwork</dt><dd>{status?.posters_configured ? "TMDB configured" : "TMDB not configured"}</dd>{result && <><dt>Retrieval</dt><dd>{result.retrieval.name} · {result.retrieval.version}</dd><dt>Reranking</dt><dd>{result.reranker.name} · {result.reranker.version}</dd><dt>Timing</dt><dd>{result.timings.total_ms.toFixed(2)} ms server · {browserMs?.toFixed(2)} ms browser</dd><dt>Request</dt><dd>{result.request_id.slice(0, 8)} · revision {result.preference_revision}</dd>{result.warnings.length > 0 && <><dt>Warnings</dt><dd>{result.warnings.join(" ")}</dd></>}</>}</dl></section></details><section className="tmdb-credit"><a href="https://www.themoviedb.org" target="_blank" rel="noreferrer">{!tmdbLogoFailed ? <img src={TMDB_LOGO} alt="The Movie Database (TMDB)" onError={() => setTmdbLogoFailed(true)} /> : <span className="tmdb-text-fallback">The Movie Database (TMDB)</span>}</a><p>This product uses the TMDB API but is not endorsed or certified by TMDB.</p></section><p className="course-note">University of Pennsylvania · CIS 5980 AI Engineering Capstone</p></div></MovieDialog>
  </main>;
}
