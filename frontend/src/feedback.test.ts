import { afterEach, expect, it, vi } from "vitest";
import { FeedbackCollector, type Assignment } from "./feedback";

const assignment: Assignment = { decision_id: "decision-12345678", token: "x".repeat(64),
  assigned_at: "2026-11-09T00:00:00Z", movie_id: "101", policy: "featured_epsilon_v1",
  propensity: .91, eligible_count: 10, outcome_window_seconds: 1800, synthetic: true, age_ms: 0 };

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); sessionStorage.clear(); });

it("retries an event with the same ID and restores the outbox after reload", async () => {
  vi.useFakeTimers();
  const fetch = vi.fn().mockRejectedValue(new Error("offline")); vi.stubGlobal("fetch", fetch);
  const collector = new FeedbackCollector();
  collector.receive(assignment);
  collector.record(assignment, "details_open");
  await vi.advanceTimersByTimeAsync(1);
  const first = JSON.parse(String(fetch.mock.calls[0][1].body));
  collector.stop();
  fetch.mockResolvedValue(new Response("{}", { status: 200 }));
  const restored = new FeedbackCollector();
  await vi.advanceTimersByTimeAsync(1);
  expect(JSON.parse(String(fetch.mock.calls[1][1].body)).event_id).toBe(first.event_id);
  expect(fetch.mock.calls.map((c) => JSON.parse(String(c[1].body)).kind)).toContain("details_open");
  restored.stop();
});

it("deduplicates display acknowledgments and preserves the original decision", async () => {
  vi.useFakeTimers();
  const fetch = vi.fn().mockResolvedValue(new Response("{}", { status: 200 })); vi.stubGlobal("fetch", fetch);
  const collector = new FeedbackCollector();
  collector.receive(assignment);
  collector.record(assignment, "display");
  collector.record(assignment, "display");
  collector.receive({ ...assignment, decision_id: "new-decision-12345" });
  collector.record(assignment, "details_open");
  await vi.advanceTimersByTimeAsync(1);
  const events = fetch.mock.calls.map((c) => JSON.parse(String(c[1].body)));
  expect(events.filter((e) => e.kind === "display")).toHaveLength(1);
  expect(events.find((e) => e.kind === "details_open").decision_id).toBe(assignment.decision_id);
  collector.stop();
});

it("does not certify an observation window after a permanently rejected event", async () => {
  vi.useFakeTimers();
  const fetch = vi.fn().mockResolvedValue(new Response("{}", { status: 422 })); vi.stubGlobal("fetch", fetch);
  const collector = new FeedbackCollector(); collector.receive(assignment);
  await vi.advanceTimersByTimeAsync(1_800_000);
  expect(fetch.mock.calls).toHaveLength(1);
  collector.stop();
});

it("closes a healthy window after heartbeats without inventing a click", async () => {
  vi.useFakeTimers();
  const fetch = vi.fn().mockResolvedValue(new Response("{}", { status: 200 })); vi.stubGlobal("fetch", fetch);
  const collector = new FeedbackCollector(); collector.receive(assignment);
  await vi.advanceTimersByTimeAsync(1_800_000);
  const events = fetch.mock.calls.map((c) => JSON.parse(String(c[1].body)));
  expect(events.at(-1).kind).toBe("window_end");
  expect(events.some((e) => e.kind === "details_open")).toBe(false);
  collector.stop();
});

it("an old in-flight flush cannot overwrite a replacement collector's outbox", async () => {
  vi.useFakeTimers();
  let resolveOld!: (value: Response) => void;
  const fetch = vi.fn().mockImplementationOnce(() => new Promise<Response>((resolve) => { resolveOld = resolve; }))
    .mockRejectedValue(new Error("offline"));
  vi.stubGlobal("fetch", fetch);
  const old = new FeedbackCollector(); old.receive(assignment); old.stop();
  const current = new FeedbackCollector(); current.record(assignment, "details_open");
  await vi.advanceTimersByTimeAsync(1);
  resolveOld(new Response("{}", { status: 200 }));
  await vi.advanceTimersByTimeAsync(1);
  const saved = JSON.parse(sessionStorage.getItem("movie-feedback-outbox-v1")!);
  expect(saved.queue.some((event: { kind: string }) => event.kind === "details_open")).toBe(true);
  current.stop();
});
