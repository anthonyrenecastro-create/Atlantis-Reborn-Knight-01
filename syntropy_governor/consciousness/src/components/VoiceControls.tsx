import { useEffect } from "react";
import { useVoice } from "../hooks/useVoice";

type VoiceControlsProps = {
  onTranscript: (text: string) => void;
  speechText: string;
};

export default function VoiceControls({ onTranscript, speechText }: VoiceControlsProps) {
  const {
    supported,
    listening,
    speaking,
    transcript,
    error,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    clearTranscript,
  } = useVoice();

  useEffect(() => {
    if (!transcript) {
      return;
    }
    onTranscript(transcript);
  }, [transcript, onTranscript]);

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
      {transcript ? <p className="voice-transcript">Transcript: {transcript}</p> : null}
      {error ? <p className="warning">{error}</p> : null}
    </section>
  );
}
