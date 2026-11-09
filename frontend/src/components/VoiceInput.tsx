import { useEffect, useRef, useState } from "react";
import { Mic, Square } from "lucide-react";

type Recognition = {
  lang: string; continuous: boolean; interimResults: boolean;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void; abort(): void;
};
type SpeechWindow = Window & { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition };

export function VoiceInput({ onTranscript, disabled = false }: { onTranscript: (text: string) => void; disabled?: boolean }) {
  const [listening, setListening] = useState(false);
  const [message, setMessage] = useState("");
  const recognition = useRef<Recognition | null>(null);
  const browser = window as SpeechWindow;
  const Constructor = browser.SpeechRecognition ?? browser.webkitSpeechRecognition;
  useEffect(() => () => {
    if (recognition.current) {
      recognition.current.onresult = null; recognition.current.onerror = null; recognition.current.onend = null;
      recognition.current.abort();
    }
  }, []);
  const start = () => {
    if (!Constructor) return;
    const instance = new Constructor(); recognition.current = instance;
    instance.lang = "en-US"; instance.continuous = false; instance.interimResults = false;
    instance.onresult = (event) => {
      const transcript = Array.from(event.results).map((result) => result[0].transcript).join(" ").slice(0, 500);
      onTranscript(transcript); setMessage("Review the text, then choose Search movies.");
    };
    instance.onerror = (event) => {
      setListening(false);
      setMessage(event.error === "not-allowed" ? "Microphone access was denied. You can type your request instead." : "Voice input could not finish. Try again or type your request.");
    };
    instance.onend = () => setListening(false);
    try { setListening(true); setMessage("Listening…"); instance.start(); }
    catch { setListening(false); setMessage("Voice input is unavailable. Type your request instead."); }
  };
  const help = Constructor ? "Voice input uses your browser’s speech service and may send audio to its provider. Review the transcript before searching." : "Voice input is not supported in this browser. Type your request instead.";
  return <div className="voice-input">
    <button className={`voice-button ${listening ? "listening" : ""}`} type="button" aria-label={listening ? "Stop listening" : "Use microphone"} title={help} disabled={disabled || !Constructor} onClick={() => listening ? recognition.current?.abort() : start()}>
      {listening ? <Square aria-hidden="true" /> : <Mic aria-hidden="true" />}
    </button>
    <span className="voice-help">{help}</span>
    {message && <p className="voice-message" role="status">{message}</p>}
  </div>;
}
