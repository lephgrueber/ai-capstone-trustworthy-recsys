# M3 feedback: implementation and results

This document answers the M3 feedback item by item: what changed, where it lives, how to
reproduce it, and what the development (dev) experiments showed. Test results appear only
for the final refit, once settings were locked (section 1, step 5).

All runs used MovieLens 32M from `data/processed/movielens_phase1` on one 8-core laptop.
Generated artifacts live under `artifacts/m3_feedback/` (git-ignored); every artifact
records the hashes of its inputs.

<!-- RESULTS -->

## 1. Separate training from model selection

### Windows

`trustworthy_recsys/data/windows.py` sorts every rating globally by timestamp and places one
cutoff for all viewers; events with equal timestamps always fall on the same side. The
existing dev and test boundaries are unchanged, so Train A + Train B is exactly the M3
training partition.

| Window | Span (UTC) | Ratings | Share | Role |
|---|---|---:|---:|---|
| Train A | 1995-01-09 → 2016-10-14 01:56 | 22,402,509 | 70.0% | tower weights, historical feature statistics |
| Train B | 2016-10-14 01:56 → 2018-10-04 | 3,200,358 | 10.0% | candidates from frozen towers; reranker weights |
| Dev | 2018-10-04 → 2020-11-06 | 3,198,273 | 10.0% | choose settings, compare pipelines |
| Test | 2020-11-06 → 2023-10-13 | 3,199,064 | 10.0% | final evaluation after choices are fixed |
| Final A / Final B | A+B+dev split 7:1 at 2018-06-17 16:00 | 25.2M / 3.6M | | final refit |
| Deployment A / B | all data split 7:1 at 2020-05-10 17:45 | 28.0M / 4.0M | | browser release fit |

Each export (`train_b`, `dev`, `final_b`, `test`, `deployment_b`) has one record per viewer
with at least one like in its label window. Inputs come only from ratings before the
window's history end; labels are the window's likes (rating ≥ 4). Every record also keeps
the viewer's other earlier ratings (for exclusion, rating statistics and dislikes) and the
star ratings behind each label. A JSON sidecar records the boundaries and the file hash, and
every later stage reads its feature cutoff from that sidecar.

The dev export reproduces the pipeline's own `validation_all` population exactly: 22,720
viewers and 1,616,509 later likes.

### Workflow

`python -m trustworthy_recsys.workflow {development,final,deployment}` runs the stages:

1. **Towers on Train A.** Inputs come from `data/retrieval_inputs.py` with
   `--window-end <Train A end>`. Checkpoints are chosen by Recall@500 over **every** dev
   viewer, not a 512-viewer sample at Recall@100 (`retrieval/train.py`, `--selection-examples`).
2. **Reranker on Train B.** Train B viewers get the frozen towers' actual top 500 from their
   Train A history, and features use Train A ratings only. A candidate is labeled 1 if it
   is a later Train B like; lists with no hit are dropped from fitting. The boosting-round
   count is chosen by end-to-end dev NDCG@12, and dev labels never fit weights
   (`ranking/train.py --valid-table-dir`).
3. **Selection on the full dev set.** Every dev viewer is scored, including viewers with no
   earlier likes (popularity fallback). Inputs and rating features come from A + B only.
   Popularity, unchanged two-tower order and reranking are reported for every viewer and for
   the 0 / 1–20 / 21–100 / 101+ usable-favorite groups (`ranking/evaluate.py`). The chosen
   epoch and round count are written to `locked.json`.
4. **Final refit.** A + B + dev is repartitioned once into Final A / Final B with the same
   A:B ratio. Both stages are refit with the locked settings and no selection: the towers
   train the selected number of epochs, the reranker trains the selected number of rounds,
   and candidates are regenerated from the new frozen towers. The result is packaged as one
   version, `pipeline.json`, covering inputs and the movie-ID mapping, towers and movie
   vectors, the reranker, the fitting windows and file hashes.
5. **Test, once.** History and features come from Final A + Final B.

The deployment stage repeats step 4 on every observation (Deployment A / B) and records
that no held-out score exists for that fit. Benchmark artifacts and scores stay separate
from it. At serving time the reranker computes features over its whole fitting window
(`serving_features` in `training.json`).

Reranker fitting and evaluation run in a separate process without torch. On macOS LightGBM
is otherwise limited to one thread.

## 2. Feature design

### Viewer tower (`retrieval/model.py`, `HistoryConfig`)

- **Learned favorite-movie vectors** over the 50 most recent favorites, pooled four ways:
  - equal weights;
  - by rating, `w = r − 3.5`;
  - by recency, `w = 0.5^(age/365 d)`;
  - learned: a softmax of `a·(r−4) + b·log1p(age)`.

  Weights are normalized to sum to one. Age is measured from the prediction time: the target
  rating in training, the request time in serving.
- **Several interests:** k-means over the favorites' genre vectors; each group's pooled
  vector is fed alongside the overall summary.
- **Dislikes:** a separate pooled dislike vector, the dislikes' genre mean and a has-dislikes
  flag. An optional loss term penalizes a liked target scoring below the viewer's latest
  *strictly earlier* dislike.
- **Favorite metadata:** the favorites' credit, runtime and tag summaries from the movie
  tower, pooled with the same weights.

### Movie tower (`retrieval/movie_side.py`)

- **Credits:** per-role person vectors for directors, writers and cast, mean-pooled, with a
  missing flag per role.
- **Runtime:** scaled with the fitting catalog's median and standard deviation; a missing
  runtime takes the median and sets a flag.
- **Tags:** the projected tag summary plus a has-tags flag.

Train A targets see tags first used before Jan 1 of the target's year. Every later stage
uses tags from before its own cutoff.

### Reranker (`ranking/features.py`)

All new features are appended, so older rerankers still select their own columns.

| Signal | Features |
|---|---|
| Movie vectors | max / mean cosine between the candidate's frozen movie-tower vector and each earlier favorite's |
| Credit IDs (per role) | shared-person count; share of favorites (with any credit in the role) sharing someone |
| Semantic tags | max / mean cosine over tagged favorites, the number compared, and a phrase-level best-match score against the 5 most recent tagged favorites |
| Runtime | gap to the mean favorite runtime |
| Dislikes | dislike count, tower and audience cosines, director share and tag cosine against disliked movies |
| Rating history | days since the last like (section 3) |

Release year and movie age are compared by dropping one or the other (see results).

## 3. Match training inputs to browser requests

- **Rating-history values passed directly.** `rating_statistics_from_record` computes mean
  rating, count, days since the last rating and days since the last like from a viewer's
  earlier ratings. Tables use it instead of the MovieLens-ID lookup (tested to give identical
  values), and browser profiles derive the same four values.
- **App availability profiles.** `features.AVAILABILITY` lists which values each app version
  can calculate:
  - `favorites_only`: none;
  - `likes_with_dates`: days since the last like only;
  - `star_ratings`: all four. This repository's browser already saves dated star ratings.

  Dev comparisons score each version with only its own values, and fitting can mask a value
  for a share of viewers.
- **Title search** (`serving/title_search.py`):
  - normalization: case, accents, punctuation and year are removed, and a trailing ", The"
    moves to the front;
  - ranking tiers: exact, then prefix, then whole-word, then substring or Damerau–Levenshtein
    fuzzy matches. "Matrix" and "The Matrix" find the same film, and "Alein" finds *Alien*.
- **Browser:**
  - Searching never saves feedback.
  - Ratings use a star scale and persist in local storage.
  - Every rated movie is excluded from recommendations.
  - Recommendations refresh on every change.
  - New: genre include/avoid dialogs narrow as you type, and a rating can carry up to five
    optional tags.

## 4. Movie metadata and rating labels

- **Credits and runtime** (`data/metadata.py`): built from the 2026-11-20 IMDb snapshot.
  Principal cast is the actor and actress rows, with equal weights. Coverage on the M3
  training catalog is 99.6% runtime, 99.6% director, 95.0% writer and 91.9% cast, matching
  the feedback's figures.
- **Tags** (`data/tags.py`):
  - 131,620 distinct normalized phrases, each encoded once with the local
    `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). No tag text leaves the machine.
  - Each movie–phrase pair keeps its first-use time; summaries are as of any cutoff.
  - Encoder, normalization, averaging rule and the 32-dimension tower projection are saved
    with the artifact.
  - Also provided: k-means theme clusters and phrase-level best-match scores.
- **Review tags:** visitors can tag a movie when rating it. With
  `TRUSTWORTHY_RECSYS_VISITOR_TAGS=record` (off by default), new phrases are stored with their
  time, encoded with the same encoder (known phrases reuse their vector) and folded into the
  reranker's tag summaries.
- **Labels:**
  - The dislike threshold is one setting, default ≤ 2.0; 2.5–3.5 is neither a like nor a
    dislike, and an unrated movie gives no evidence.
  - Five-star likes can be weighted more when fitting either stage.
  - Graded NDCG uses gain `r − 3` and is reported next to binary NDCG.

## 5. Missing and unseen inputs

| Case | Handling |
|---|---|
| Writer the model has not seen | Credit matching uses IMDb IDs, so a shared writer still counts. In the tower, people with fewer than 2 catalog credits, and new people, share a per-role UNKNOWN vector, trained by masking known people 10% of the time. |
| No writer listed | Reranker writer features are NaN, not 0. The tower uses a zero role vector plus a missing flag. |
| Missing runtime | The tower uses the fitting median plus a missing flag, and the same saved median is used when serving. Reranker runtime gap is NaN. |
| No tags | The tower uses a zero tag vector plus a has-tags flag of 0. Reranker tag similarities are NaN when the candidate has no tags. |
| Only some favorites comparable | Means cover the comparable favorites only, with their count as a separate feature; NaN when none is comparable. |

A test blanks one movie's runtime, then all of its optional metadata while keeping its ID.
It checks that training and serving fill the same values and that the movie still gets a
finite vector and score.

## 6. Target sampling and tower settings

`data/retrieval_inputs.py --max-targets-per-user 0` stores every eligible target: 10,969,594
for Train A. `TargetSampler` then draws per epoch:

- `spaced`: the M3 rule;
- `fixed`: one random draw;
- `rotate`: the same first draw, redrawn each later epoch.

For rotation, the logQ correction uses each movie's expected draw frequency. `retrieval/sweep.py`
compares settings on the full dev set by Recall@500, with the epoch and number of updates
that reached it, training time and target coverage.

## 7. Display-length tuning and retrieval depth

- Reranker rounds are chosen by end-to-end NDCG@12 for the 12-movie display, and the
  LambdaRank truncation level is a setting (15 vs 30 is compared below).
- `retrieval/candidate_recall.py` measures recall at several depths with exclusions and full
  denominators. Optional browser genre filters are applied after retrieval, reporting the
  share of that genre's later likes left in the pool. Exact-search latency is reported per
  depth.
- The server candidate budget and candidate batches now allow up to 1,000.

## 8. Comparisons and diagnostics

- **Paired comparisons** (`ranking/compare.py`, `ranking/ablations.py`): per-viewer
  differences, a viewer-bootstrap 95% interval for each ablation, the usable-favorite groups,
  graded NDCG and the spread across seed repeats.
- **Group diagnostics** (`ranking/diagnostics.py`): same personalized dev requests; candidate
  pool, two-tower top-K and reranked top-K recall.
  - Movie groups: popularity, genre, tags, runtime.
  - Viewer groups: history length, era of favorites, interest range.
  - Also: catalog coverage, and the share of displayed titles later rated low next to the
    share rated at all.
- **Recency-weighted popularity** (`ranking/popularity.py`) for viewers with no usable
  favorites.
- **Cold start** (`retrieval/cold_start.py`, optional): movie-ID dropout and metadata-only
  vectors for movies first rated in Train B.
- **CI:** the Python job already installs the `rank` extra. `.github/workflows/frontend.yml`
  runs the frontend unit tests, type check and build when frontend files change.
