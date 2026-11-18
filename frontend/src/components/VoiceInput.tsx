import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, Square } from "lucide-react";

type Recognition = {
  lang: string; continuous: boolean; interimResults: boolean;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void; abort(): void;
};
type SpeechWindow = Window & { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition };
type VoicePurpose = "movie-title" | "recommendations";
type VoiceInputProps = {
  onTranscript: (text: string) => void;
  purpose: VoicePurpose;
  sessionKey: string;
  disabled?: boolean;
};

let activeVoiceSession: (() => void) | null = null;

const copy = {
  "movie-title": {
    target: "movie title",
    review: "Review the title, then choose a movie.",
    typed: "movie title",
    help: "Review the transcript before choosing a movie.",
  },
  recommendations: {
    target: "recommendations",
    review: "Review the text, then choose Find recommendations.",
    typed: "request",
    help: "Review the transcript before searching.",
  },
} satisfies Record<VoicePurpose, { target: string; review: string; typed: string; help: string }>;

export function VoiceInput({ onTranscript, purpose, sessionKey, disabled = false }: VoiceInputProps) {
  const [listening, setListening] = useState(false);
  const [message, setMessage] = useState("");
  const recognition = useRef<Recognition | null>(null);
  const generation = useRef(0);
  const browser = window as SpeechWindow;
  const Constructor = browser.SpeechRecognition ?? browser.webkitSpeechRecognition;
  const wording = copy[purpose];
  const stop = useCallback((clearMessage = false) => {
    generation.current += 1;
    const instance = recognition.current;
    recognition.current = null;
    if (instance) {
      instance.onresult = null; instance.onerror = null; instance.onend = null;
      instance.abort();
    }
    if (activeVoiceSession === stop) activeVoiceSession = null;
    setListening(false);
    if (clearMessage) setMessage("");
  }, []);
  useEffect(() => {
    stop(true);
    return () => {
      generation.current += 1;
      const instance = recognition.current;
      recognition.current = null;
      if (instance) {
        instance.onresult = null; instance.onerror = null; instance.onend = null;
        instance.abort();
      }
      if (activeVoiceSession === stop) activeVoiceSession = null;
    };
  }, [sessionKey, stop]);
  useEffect(() => { if (disabled) stop(); }, [disabled, stop]);
  useEffect(() => {
    if (message !== wording.review && message !== "Voice input stopped. You can edit the field or try again.") return;
    const timer = window.setTimeout(() => setMessage(""), 4000);
    return () => window.clearTimeout(timer);
  }, [message, wording.review]);
  const start = () => {
    if (!Constructor) return;
    activeVoiceSession?.();
    stop();
    const instance = new Constructor();
    const token = generation.current;
    recognition.current = instance;
    activeVoiceSession = stop;
    instance.lang = "en-US"; instance.continuous = false; instance.interimResults = false;
    instance.onresult = (event) => {
      if (recognition.current !== instance || generation.current !== token) return;
      const transcript = Array.from(event.results).map((result) => result[0].transcript).join(" ").slice(0, 500);
      onTranscript(transcript); setMessage(wording.review);
    };
    instance.onerror = (event) => {
      if (recognition.current !== instance || generation.current !== token) return;
      setListening(false);
      setMessage(event.error === "not-allowed" ? `Microphone access was denied. You can type your ${wording.typed} instead.` : `Voice input could not finish. Try again or type your ${wording.typed}.`);
    };
    instance.onend = () => {
      if (recognition.current !== instance || generation.current !== token) return;
      recognition.current = null;
      if (activeVoiceSession === stop) activeVoiceSession = null;
      setListening(false);
    };
    try { setListening(true); setMessage("Listening…"); instance.start(); }
    catch { stop(); setMessage(`Voice input is unavailable. Type your ${wording.typed} instead.`); }
  };
  const stopByUser = () => { stop(); setMessage("Voice input stopped. You can edit the field or try again."); };
  const help = Constructor ? `Voice input uses your browser’s speech service and may send audio to its provider. ${wording.help}` : `Voice input is not supported in this browser. Type your ${wording.typed} instead.`;
  return <div className="voice-input">
    <button className={`voice-button ${listening ? "listening" : ""}`} type="button" aria-label={listening ? `Stop listening for ${wording.target}` : `Use microphone for ${wording.target}`} aria-pressed={listening} title={help} disabled={disabled || !Constructor} onClick={() => listening ? stopByUser() : start()}>
      {listening ? <Square aria-hidden="true" /> : <Mic aria-hidden="true" />}
    </button>
    <span className="voice-help">{help}</span>
    {message && <p className="voice-message" role="status">{message}</p>}
  </div>;
}
