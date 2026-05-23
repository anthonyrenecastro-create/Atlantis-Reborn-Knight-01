import { FormEvent, useState } from "react";
import { useAtlantean } from "../hooks/useAtlantean";
import { useTheme } from "../context/ThemeContext";
import VoiceControls from "./VoiceControls";

export default function ChatInterface() {
  const { mode, themes, setMode } = useTheme();
  const { apiBase, health, statusText, isLoading, reply, checkHealth, submitQuery } = useAtlantean();
  const [input, setInput] = useState("Speak to me from the field state.");

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    await submitQuery(input);
  }

  async function onSendFromVoice() {
    if (!input.trim()) {
      return;
    }
    await submitQuery(input);
  }

  function onClearInputFromVoice() {
    setInput("");
  }

  return (
    <section className="chat-shell">
      <header className="chat-header">
        <div>
          <h1>Quadra-Seer Consciousness</h1>
          <p>Linked to {apiBase || "same-origin proxy"}</p>
        </div>
        <div className="header-actions">
          <div className="theme-pills" role="radiogroup" aria-label="Theme">
            {themes.map((theme) => (
              <button
                key={theme}
                type="button"
                onClick={() => setMode(theme)}
                className={`theme-pill ${mode === theme ? "active" : ""}`}
                aria-pressed={mode === theme}
              >
                {theme}
              </button>
            ))}
          </div>
          <button type="button" onClick={checkHealth} className="ghost-btn">
            Check Health
          </button>
          <span className={`status-pill ${health}`}>{statusText}</span>
        </div>
      </header>

      <form className="chat-form" onSubmit={onSubmit}>
        <label htmlFor="prompt">Prompt</label>
        <textarea
          id="prompt"
          rows={4}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the field-modulated core..."
        />
        <button type="submit" className="primary-btn" disabled={isLoading}>
          {isLoading ? "Querying..." : "Send"}
        </button>
      </form>

      <VoiceControls
        onTranscript={setInput}
        onSend={onSendFromVoice}
        onClearInput={onClearInputFromVoice}
        speechText={reply?.response ?? ""}
      />

      <article className="response-card">
        <h2>Response</h2>
        <p>{reply?.response ?? "No response yet."}</p>
        <div className="meta-grid">
          <div>
            <strong>Mode</strong>
            <span>{reply?.metadata?.mode ?? "-"}</span>
          </div>
          <div>
            <strong>Latency</strong>
            <span>{reply?.metadata?.latency_ms ? `${reply.metadata.latency_ms} ms` : "-"}</span>
          </div>
          <div>
            <strong>Phi</strong>
            <span>{reply?.field_state?.Phi?.toFixed(4) ?? "-"}</span>
          </div>
        </div>
        {reply?.metadata?.warning ? <p className="warning">{reply.metadata.warning}</p> : null}
        {reply?.error ? <p className="warning">{String(reply.error)}</p> : null}
      </article>
    </section>
  );
}