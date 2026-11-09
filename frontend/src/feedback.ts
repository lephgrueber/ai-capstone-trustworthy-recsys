import type { components } from "./api/schema";

export type Assignment = components["schemas"]["FeedbackAssignment"];
type Event = components["schemas"]["FeedbackEvent"];
type Kind = Event["kind"];
type Tracker = { assignment: Assignment; origin: number; displayed: boolean; healthy: boolean };
const STORAGE = "movie-feedback-outbox-v1";
const WINDOW_MS = 1_800_000;

/** Tab-scoped durable outbox. Tokens never enter URLs or console logs. */
export class FeedbackCollector {
  private trackers: Record<string, Tracker> = {};
  private queue: Event[] = [];
  private busy = false;
  private timer: ReturnType<typeof setInterval> | undefined;
  private stopped = false;
  constructor() {
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORAGE) ?? "null");
      if (saved?.version === 1) { this.trackers = saved.trackers; this.queue = saved.queue; }
    } catch { /* An unreadable outbox cannot be claimed as an observed window. */ }
    this.timer = setInterval(() => this.tick(), 15_000);
    void this.flush();
  }
  stop() { this.stopped = true; clearInterval(this.timer); this.save(); }
  private save() {
    try { sessionStorage.setItem(STORAGE, JSON.stringify({ version: 1, trackers: this.trackers, queue: this.queue })); }
    catch { Object.values(this.trackers).forEach((tracker) => { tracker.healthy = false; }); }
  }
  receive(assignment: Assignment) {
    if (this.trackers[assignment.decision_id]) return;
    this.trackers[assignment.decision_id] = { assignment, origin: Date.now()-(assignment.age_ms ?? 0), displayed: false, healthy: true };
    this.record(assignment, "heartbeat");
  }
  record(assignment: Assignment, kind: Kind) {
    const tracker = this.trackers[assignment.decision_id];
    if (!tracker) return;
    const elapsed = Math.max(0, Date.now()-tracker.origin);
    if (kind !== "window_end" && elapsed > WINDOW_MS) return;
    if (kind === "display" && tracker.displayed) return;
    if (kind === "display") tracker.displayed = true;
    if (this.queue.length >= 2000) { tracker.healthy = false; this.save(); return; }
    this.queue.push({ event_id: crypto.randomUUID(), decision_id: assignment.decision_id,
      token: assignment.token, movie_id: assignment.movie_id, kind, surface: "featured",
      elapsed_ms: Math.min(elapsed, WINDOW_MS+5000), stream_ok: tracker.healthy });
    this.save();
    void this.flush();
  }
  private tick() {
    for (const tracker of Object.values(this.trackers)) {
      const elapsed = Date.now()-tracker.origin;
      if (elapsed >= WINDOW_MS) {
        // A throttled timer beyond the tolerance cannot certify a full window.
        if (elapsed <= WINDOW_MS+5000) this.record(tracker.assignment, "window_end");
        delete this.trackers[tracker.assignment.decision_id];
      } else this.record(tracker.assignment, "heartbeat");
    }
    this.save(); void this.flush();
  }
  private async flush() {
    if (this.busy || this.stopped) return;
    this.busy = true;
    try {
      while (this.queue.length && !this.stopped) {
        const event = this.queue[0];
        // Expired unsent events cannot become delayed negative evidence.
        const tracker = this.trackers[event.decision_id];
        if (tracker && Date.now()-tracker.origin > WINDOW_MS+86_400_000) {
          this.queue = this.queue.filter((e) => e.decision_id !== event.decision_id);
          delete this.trackers[event.decision_id]; this.save(); continue;
        }
        let response: Response;
        try {
          response = await fetch("/api/v1/feedback/events", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(event), keepalive: true });
        } catch { break; } // Retain the same ID and payload for the next retry.
        if (this.stopped) break; // A replacement collector now owns sessionStorage.
        if (response.ok) this.queue.shift();
        else if (response.status >= 400 && response.status < 500) {
          // Do not send a window_end after any permanently lost event.
          this.queue = this.queue.filter((e) => e.decision_id !== event.decision_id);
          delete this.trackers[event.decision_id];
        } else break;
        this.save();
      }
    } finally { this.busy = false; }
  }
}
