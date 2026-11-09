import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { VoiceInput } from "./VoiceInput";

afterEach(() => vi.unstubAllGlobals());

it("keeps typed input available when speech is unsupported", () => {
  vi.stubGlobal("SpeechRecognition", undefined);
  vi.stubGlobal("webkitSpeechRecognition", undefined);
  render(<VoiceInput onTranscript={vi.fn()} />);
  expect(screen.getByRole("button", { name: "Use microphone" })).toBeDisabled();
  expect(screen.getByText(/not supported in this browser/)).toBeInTheDocument();
});

it("starts only on click, returns editable text, handles denial, and aborts on unmount", () => {
  const start = vi.fn(), abort = vi.fn(), onTranscript = vi.fn();
  const instance = { start, abort, lang: "", continuous: true, interimResults: true,
    onresult: null as null | ((event: { results: { transcript: string }[][] }) => void),
    onerror: null as null | ((event: { error: string }) => void), onend: null as null | (() => void) };
  vi.stubGlobal("SpeechRecognition", class { constructor() { return instance; } });
  const { unmount } = render(<VoiceInput onTranscript={onTranscript} />);
  expect(start).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Use microphone" }));
  expect(start).toHaveBeenCalledOnce();
  act(() => instance.onresult?.({ results: [[{ transcript: "show comedy" }]] }));
  expect(onTranscript).toHaveBeenCalledWith("show comedy");
  expect(screen.getByRole("status")).toHaveTextContent("Review the text");
  act(() => instance.onerror?.({ error: "not-allowed" }));
  expect(screen.getByRole("status")).toHaveTextContent("denied");
  unmount();
  expect(abort).toHaveBeenCalledOnce();
  expect(instance.onresult).toBeNull();
});
