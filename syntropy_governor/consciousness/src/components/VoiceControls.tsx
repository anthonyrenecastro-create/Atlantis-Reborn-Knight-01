import { type Dispatch, type SetStateAction, useEffect, useState } from "react";
import { useVoice } from "../hooks/useVoice";

type VoiceControlsProps = {
  onTranscript: Dispatch<SetStateAction<string>>;
  onSend: () => void;
  onClearInput: () => void;
  speechText: string;
};

const LANGUAGE_CHOICES = [
  { value: "en-US", label: "English (US)" },
  { value: "en-GB", label: "English (UK)" },
  { value: "es-ES", label: "Spanish" },
  { value: "fr-FR", label: "French" },
];

export default function VoiceControls({ onTranscript, onSend, onClearInput, speechText }: VoiceControlsProps) {
  const {
    supported,
    listening,
    speaking,
    transcript,
    language,
    error,
    setLanguage,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    clearTranscript,
  } = useVoice();
  const [insertMode, setInsertMode] = useState<"replace" | "append">("replace");

  useEffect(() => {
    if (!transcript) {
      return;
    }
    const normalized = transcript.toLowerCase().trim();
    if (normalized === "send" || normalized === "submit" || normalized === "run query") {
      onSend();
      return;
    }
    if (normalized === "clear" || normalized === "reset") {
      onClearInput();
      return;
    }
    if (insertMode === "append") {
      onTranscript((prev) => (prev ? `${prev} ${transcript}` : transcript));
      return;
    }
    onTranscript(() => transcript);
  }, [insertMode, onClearInput, onSend, onTranscript, transcript]);

  return (
    <section className="voice-shell">
      <div className="voice-header">
        <strong>Voice</strong>
        <span className={`status-pill ${supported ? "healthy" : "down"}`}>{supported ? "Enabled" : "Unavailable"}</span>
      </div>
      <div className="voice-actions">
        <button
          type="button"
          className="ghost-btn"
          onClick={listening ? stopListening : startListening}
          disabled={!supported}
        >
          {listening ? "Stop Mic" : "Start Mic"}
        </button>
        <button type="button" className="ghost-btn" onClick={() => speak(speechText)} disabled={!supported || !speechText}>
          {speaking ? "Speaking..." : "Speak Reply"}
        </button>
        <button type="button" className="ghost-btn" onClick={stopSpeaking} disabled={!supported || !speaking}>
          Stop Voice
        </button>
        <button type="button" className="ghost-btn" onClick={clearTranscript} disabled={!transcript}>
          Clear Transcript
        </button>
      </div>
      <div className="voice-options">
        <label>
          Language
          <select value={language} onChange={(e) => setLanguage(e.target.value)} disabled={!supported || listening}>
            {LANGUAGE_CHOICES.map((choice) => (
              <option key={choice.value} value={choice.value}>
                {choice.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Transcript Mode
          <select value={insertMode} onChange={(e) => setInsertMode(e.target.value as "replace" | "append")}>
            <option value="replace">Replace Input</option>
            <option value="append">Append Input</option>
          </select>
        </label>
      </div>
      {transcript ? <p className="voice-transcript">Transcript: {transcript}</p> : null}
      <p className="voice-hint">Voice commands: "send" to submit, "clear" to reset input.</p>
      {error ? <p className="warning">{error}</p> : null}
    </section>
  );
}
