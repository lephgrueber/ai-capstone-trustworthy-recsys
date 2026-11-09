# Movie recommendation feedback and OPE plan

Status: implemented for the local pilot. This file preserves the original design rationale; [movie_ope.md](movie_ope.md) is the implemented contract and run guide. Collection is configurable (off / observe / randomized); real outcome collection is still required before a movie-policy improvement can be assessed.

## Purpose

Record what the application decided to show, what actually appeared, and how the
visitor responded. This will let us study movie recommendations using our own
evidence. Open Bandit remains a separate fashion benchmark; its chosen random
policy is not a recommendation to replace the movie model.

The first movie outcome will be opening the featured movie's details within
30 minutes of assignment. This is an interest signal, not proof of watching or
satisfaction. Favorites and hides will be separate secondary outcomes. Freeze
these definitions before collection rather than choosing the best-looking one later.

## What already exists

- `serving/contracts.py`: request session ID, preference revision, preference
  snapshot, known MovieLens user ID, response request ID, ranks and component versions.
- `serving/service.py`: candidate retrieval, hard filtering, reranking and final
  constraint validation. These are the boundaries for recording the decision.
- `serving/app.py`: `POST /api/v1/recommendations` and request logging.
  The request ID can come from a client header, so it must not be the trusted
  unique identifier of an experimental decision.
- `frontend/src/App.tsx`: featured movie, detail dialog, favorites, hides and
  asynchronous recommendation updates. A session ID is already stored in the browser.

The design below called for a durable decision/event store, display acknowledgments, response
events tied to the originating recommendation, known randomized selection
probabilities, and a dataset builder for mature outcomes. Existing service logs
and similarity scores do not supply those probabilities.

## Step 1: capture the current application's behavior

Introduce a separate server-generated `decision_id` for every recommendation
decision. Preserve `request_id` for troubleshooting. Store the decision before
returning a response, including the exact ordered results and frozen pre-request
preferences. Link later browser events to this decision rather than to the newest
response in memory. Stale responses discarded by the browser are not displays.

Initially label the existing ranking `deterministic_observation_v1` and
`ope_eligible=false`. Its chosen action has conditional probability one and
alternatives zero under the fixed pipeline. Observational logging is useful for
debugging and click-rate reporting but does not create support for evaluating
different choices. Never turn similarity scores or observed item frequencies into
invented logging probabilities.

## Step 2: a bounded, known-randomness movie experiment

After instrumentation is verified, introduce an explicitly versioned experiment
for one featured slot. Keep the learned model and user constraints in place.

1. Retrieve and filter candidates as today. Freeze the ordinary recommendation
   grid first. It must be identical regardless of which featured item is drawn.
2. The featured-slot candidate set is the remaining eligible retrieved movies,
   excluding every movie in that fixed grid. Save its complete IDs, order, scores,
   and a content hash. This avoids duplicates and avoids changing the grid as a
   side effect of the sampled featured item.
3. With proposed exploration rate 0.1, choose the highest-ranked eligible movie
   with probability 0.9, otherwise choose uniformly from the complete remaining
   candidate set. This number is an initial design choice, not selected from the
   Open Bandit results. Record the exact configured rate with every decision.
4. If there are K eligible candidates, the preferred movie's probability is
   `0.9 + 0.1/K`; each other movie's probability is `0.1/K`. Record the total
   probability, not only the probability of the random branch that happened to run.
5. Perform no item-dependent post-draw substitution. If a draw cannot be served,
   log the failure. Replacing it requires a new decision with the correct mechanism.
   Empty or single-item candidate sets are recorded explicitly; they provide no
   useful comparison between alternatives. Fixture traffic is excluded from real
   evidence. Baseline and learned traffic are recorded as separate populations.

For example, with ten candidates the preferred item has probability 0.91 and
each other item 0.01. A later policy can only be assessed if it chooses from the
same saved candidate set and leaves the rest of the presentation mechanism fixed.
These logs evaluate featured-slot selection conditional on the recorded retrieval
pool. They cannot establish the value of changing the retrieval catalog, the
Weaviate index, or an entire recommendation slate.

## Event contract design

Persist decisions separately from browser events, with versioned schemas.
Use a transactional SQLite store for the local single-service prototype; use
managed transactional storage before deploying multiple independent instances.

| Record | Required information |
|---|---|
| Decision | Server decision ID, server UTC time, pseudonymous session, preference revision, pre-decision preference/context snapshot, backend mode, retrieval/reranker/model/index fingerprints, policy version, experiment version, fixed grid, complete eligible featured candidates, selected movie, slot, exact probability, configured exploration rate, status |
| Display | Unique event ID, decision ID, movie ID, slot, client event time, server receipt time, display surface, visibility rule version |
| Response | Unique event ID, decision ID, movie ID, slot, response type (`details_open`, `favorite_add`, `hide`), client and server times, originating display surface |
| Collection health | Decision ID, client acknowledgment/flush state, delivery failures, server persistence failures and outcome-window status |

Store the exact context needed for later replay, not only a hash. A hash is an
integrity reference and cannot reconstruct missing preferences. Store only the
project's recommendation context and pseudonymous identifiers; credentials and
unrelated personal text have no role in these records. Explain collection in the
application's data-flow description. Keep synthetic/automated test traffic labeled.

The browser may report interaction facts, but must not choose the propensity,
eligible candidates, experiment version or reward value. The backend derives
these from the persisted decision and validates membership and attribution.
Reject events whose decision/movie/slot combinations do not match. Use a server-
issued session binding/token rather than treating a guessable ID as authorization.

## Display and response rules

For display diagnostics, the featured card must be at least 50% visible for one
continuous second while the document is visible. This is a proposed measurable
rule, not proof that a person read it. Details opened sooner should still retain
their response event; do not discard a real click because the display timer had
not completed. Keep decision, visibility and response events separate.

Carry the originating decision ID into the detail dialog and its controls.
Clicks from catalog search or a preference picker are not recommendation rewards.
React rerenders, retries, repeated observations, and duplicate requests must not
create extra exposures or rewards. Generate stable event IDs per logical event
and enforce uniqueness in storage. Multiple details opens count at most once per
decision for the binary primary outcome; raw response events may still be retained.

Do not silently condition OPE on browser-confirmed visibility: whether an item
renders or becomes visible can depend on the chosen item and user behavior.
The initial target is assignment-level details-open probability under the fixed
presentation mechanism. Report visibility and delivery separately. If complete
outcome observation is unavailable, explicitly report missing/censored outcomes
and sensitivity bounds rather than pretending the observed-only subset identifies
the assignment-level effect. A later viewed-only analysis needs its own assumptions.

## Reward construction and dataset export

After the 30-minute attribution window and a fixed 24-hour late-arrival allowance,
join decisions to deduplicated events. Preserve collection status, not just a
binary label. A qualifying details-open response within the window gives reward
one. Reward zero requires a mature, sufficiently observed window without such a
response. A missing display acknowledgment or broken event stream is not evidence
of a dislike and must not silently become zero. Browser closure/offline delivery
can leave outcomes unknown; define and report the observation rule before a pilot.

Export assigned, observed, missing, positive and negative counts, event-join rates,
duplicates, late events, policy support, action probabilities and session/time
coverage. Save `reward_status` so an evaluator cannot accidentally consume unknown
outcomes as negatives. Export source and code hashes with the dataset.

The single-position estimator formulas in `ope/estimators.py` are reusable, but
the existing Open Bandit importer is not. Add a separate movie-log importer and
candidate-set masks. Target probabilities must be zero outside each event's
eligible set. Do not widen the candidate set after observing the reward. Hold out
later collection windows, and use session-aware uncertainty analysis where feasible.

## Original implementation sequence and acceptance checks

1. Extend serving contracts with server decision IDs and experimental metadata;
   add decision/event persistence and a validated event endpoint.
2. Instrument the featured card and existing details/favorite/hide controls in
   `App.tsx`; preserve source attribution across asynchronous updates.
3. Test duplicate events, stale responses, reloads, offline retries, invalid movie
   IDs, missing windows, and server-write failures. No persistence failure should
   silently create an OPE-eligible unlogged assignment.
4. Test the selector separately: probabilities sum to one, forbidden/grid movies
   are never drawn, sampled frequencies match probabilities, and the preferred
   item gets the combined probability. Test empty and single-item pools.
5. Run a labeled synthetic pilot to reconcile decisions, displays and responses.
   Synthetic events verify plumbing; they are not evidence of user preference.
6. Enable a bounded real pilot with frozen outcome and policy definitions, then
   inspect data quality, informative click counts and effective sample size before
   choosing an evaluation date. Do not promise sufficient evidence after a fixed
   arbitrary number of days or infer sufficient power from raw row count alone.

Deliverables: serving/event contracts, selector and storage, browser attribution,
movie-log exporter, data-quality report, movie-feedback Data Card and run guide.
Only after that collection can a separate movie-policy OPE report be produced.


## Implementation mapping

`serving/feedback.py` implements SQLite persistence, capability-bound events and featured selection. `frontend/src/feedback.ts` implements tab-scoped delivery and observation tracking; `App.tsx` binds the featured controls. `ope/movie.py` exports and evaluates mature observations; `ope/movie_pilot.py` exercises the path with labeled synthetic outcomes. The initial experiment is configurable rather than silently enabled for every deployment. See the run guide for the precise negative-outcome observation rule and current limitations.
