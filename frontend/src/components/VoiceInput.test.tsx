import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { VoiceInput } from "./VoiceInput";

afterEach(() => vi.unstubAllGlobals());

it("keeps typed input available when speech is unsupported", () => {
  vi.stubGlobal("SpeechRecognition", undefined);
  vi.stubGlobal("webkitSpeechRecognition", undefined);
  render(<><input aria-label="Movie title" defaultValue="typed title" /><VoiceInput purpose="movie-title" sessionKey="rate" onTranscript={vi.fn()} /></>);
  expect(screen.getByRole("button", { name: "Use microphone for movie title" })).toBeDisabled();
  expect(screen.getByText(/not supported in this browser.*movie title/i)).toBeInTheDocument();
  expect(screen.getByLabelText("Movie title")).toHaveValue("typed title");
});

it("starts only on click, returns editable text, handles denial, and aborts on unmount", () => {
  const start = vi.fn(), abort = vi.fn(), onTranscript = vi.fn();
  const instance = { start, abort, lang: "", continuous: true, interimResults: true,
    onresult: null as null | ((event: { results: { transcript: string }[][] }) => void),
    onerror: null as null | ((event: { error: string }) => void), onend: null as null | (() => void) };
  vi.stubGlobal("SpeechRecognition", class { constructor() { return instance; } });
  const { unmount } = render(<VoiceInput purpose="recommendations" sessionKey="recommend" onTranscript={onTranscript} />);
  expect(start).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Use microphone for recommendations" }));
  expect(start).toHaveBeenCalledOnce();
  expect(screen.getByRole("button", { name: "Stop listening for recommendations" })).toHaveAttribute("aria-pressed", "true");
  act(() => instance.onresult?.({ results: [[{ transcript: "show comedy" }]] }));
  expect(onTranscript).toHaveBeenCalledWith("show comedy");
  expect(screen.getByRole("status")).toHaveTextContent("Review the text, then choose Find recommendations");
  act(() => instance.onerror?.({ error: "not-allowed" }));
  expect(screen.getByRole("status")).toHaveTextContent(/denied.*type your request/i);
  unmount();
  expect(abort).toHaveBeenCalledOnce();
  expect(instance.onresult).toBeNull();
});

it("aborts on a mode change and ignores a queued result from the previous mode", () => {
  const onTitle = vi.fn(), onRecommendations = vi.fn();
  const instance = { start: vi.fn(), abort: vi.fn(), lang: "", continuous: true, interimResults: true,
    onresult: null as null | ((event: { results: { transcript: string }[][] }) => void),
    onerror: null as null | ((event: { error: string }) => void), onend: null as null | (() => void) };
  vi.stubGlobal("SpeechRecognition", class { constructor() { return instance; } });
  const { rerender } = render(<VoiceInput purpose="movie-title" sessionKey="rate" onTranscript={onTitle} />);
  fireEvent.click(screen.getByRole("button", { name: "Use microphone for movie title" }));
  const lateResult = instance.onresult!;
  rerender(<VoiceInput purpose="recommendations" sessionKey="recommend" onTranscript={onRecommendations} />);
  expect(instance.abort).toHaveBeenCalledOnce();
  act(() => lateResult({ results: [[{ transcript: "late title" }]] }));
  expect(onTitle).not.toHaveBeenCalled();
  expect(onRecommendations).not.toHaveBeenCalled();
});
