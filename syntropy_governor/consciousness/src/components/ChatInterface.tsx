import { FormEvent, useEffect, useState } from "react";
import { useAtlantean } from "../hooks/useAtlantean";
import { useTheme } from "../context/ThemeContext";
import VoiceControls from "./VoiceControls";

export default function ChatInterface() {
  const { mode, themes, setMode } = useTheme();
  const {
    apiBase,
    health,
    statusText,
    isLoading,
    reply,
    sovereign,
    checkHealth,
    submitQuery,
    sendFeedback,
    refreshSovereignStatus,
    exportTrainingDataset,
    launchTrainingJob,
    getTrainingJob,
  } = useAtlantean();
  const [input, setInput] = useState("Speak to me from the field state.");
  const [feedbackStatus, setFeedbackStatus] = useState("");
  const [exportStatus, setExportStatus] = useState("");
  const [trainingStatus, setTrainingStatus] = useState("");
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<"idle" | "queued" | "running" | "completed" | "failed">("idle");
  const [trainingLogTail, setTrainingLogTail] = useState("");

  const [epochs, setEpochs] = useState(2);
  const [batchSize, setBatchSize] = useState(8);
  const [seqLen, setSeqLen] = useState(128);
  const [exportLimit, setExportLimit] = useState(1500);

  useEffect(() => {
    void refreshSovereignStatus();
  }, [refreshSovereignStatus]);

  useEffect(() => {
    if (!trainingJobId || (jobState !== "queued" && jobState !== "running")) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const job = await getTrainingJob(trainingJobId);
        setJobState(job.status);
        if (job.log_tail) {
          setTrainingLogTail(job.log_tail);
        }

        if (job.status === "completed") {
          const swapped = job.reload?.status === "reloaded";
          setTrainingStatus(
            `Training complete. Checkpoint: ${job.output_checkpoint ?? "-"}. Hot-swap: ${swapped ? "applied" : "not applied"}.`,
          );
          await refreshSovereignStatus();
          window.clearInterval(timer);
        } else if (job.status === "failed") {
          setTrainingStatus(`Training failed: ${job.error ?? "unknown error"}`);
          window.clearInterval(timer);
        }
      } catch {
        setTrainingStatus("Could not fetch training status.");
        window.clearInterval(timer);
      }
    }, 2000);

    return () => {
      window.clearInterval(timer);
    };
  }, [trainingJobId, jobState, getTrainingJob, refreshSovereignStatus]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    await submitQuery(input);
    await refreshSovereignStatus();
  }

  async function onSendFromVoice() {
    if (!input.trim()) {
      return;
    }
    await submitQuery(input);
    await refreshSovereignStatus();
  }

  function onClearInputFromVoice() {
    setInput("");
  }

  async function onFeedback(event: "user_confirmation" | "user_negative_feedback" | "high_engagement") {
    try {
      await sendFeedback(event, {
        interactionId: reply?.interaction_id,
      });
      setFeedbackStatus("Feedback applied to fields.");
      await refreshSovereignStatus();
    } catch {
      setFeedbackStatus("Feedback failed. Backend may be unavailable.");
    }
  }

  async function onCorrection() {
    const correction = window.prompt("Provide correction for this response:");
    if (!correction || !correction.trim()) {
      return;
    }
    try {
      await sendFeedback("user_correction", {
        interactionId: reply?.interaction_id,
        correction: correction.trim(),
        intensity: 0.8,
      });
      setFeedbackStatus("Correction received and learned.");
      await refreshSovereignStatus();
    } catch {
      setFeedbackStatus("Correction failed to send.");
    }
  }

  async function onExportSovereignData() {
    try {
      const result = await exportTrainingDataset({ limit: 1000 });
      setExportStatus(`Exported ${result.count} rows to ${result.export_path ?? "(no file)"}`);
    } catch {
      setExportStatus("Export failed. Check backend logs.");
    }
  }

  async function onLaunchTraining() {
    setTrainingStatus("Launching training job...");
    setTrainingLogTail("");
    try {
      const job = await launchTrainingJob({
        exportLimit,
        epochs,
        batchSize,
        seqLen,
        autoHotSwap: true,
      });
      setTrainingJobId(job.job_id);
      setJobState(job.status === "queued" ? "queued" : "running");
      setTrainingStatus(`Training job started: ${job.job_id}`);
    } catch {
      setTrainingStatus("Failed to launch training job.");
      setJobState("failed");
    }
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

      <section className="sovereign-panel">
        <div className="sovereign-header">
          <h2>Sovereign Training Panel</h2>
          <button type="button" className="ghost-btn" onClick={() => void refreshSovereignStatus()}>
            Refresh
          </button>
        </div>
        <div className="sovereign-grid">
          <div>
            <strong>Local Only</strong>
            <span>{sovereign?.local_only ? "Yes" : "No"}</span>
          </div>
          <div>
            <strong>Total Queries</strong>
            <span>{String(sovereign?.stats?.queries_total ?? "-")}</span>
          </div>
          <div>
            <strong>Fallback Calls</strong>
            <span>{String(sovereign?.stats?.fallback_calls ?? "-")}</span>
          </div>
          <div>
            <strong>Positive Feedback</strong>
            <span>{String(sovereign?.stats?.positive_feedback ?? "-")}</span>
          </div>
          <div>
            <strong>Negative Feedback</strong>
            <span>{String(sovereign?.stats?.negative_feedback ?? "-")}</span>
          </div>
          <div>
            <strong>Learning Events</strong>
            <span>{String(sovereign?.stats?.learning_events ?? "-")}</span>
          </div>
          <div>
            <strong>Active Model</strong>
            <span>{sovereign?.active_model_path ?? "-"}</span>
          </div>
          <div>
            <strong>Training Job</strong>
            <span>{trainingJobId ?? "-"}</span>
          </div>
          <div>
            <strong>Job State</strong>
            <span>{jobState}</span>
          </div>
        </div>
        <div className="sovereign-grid">
          <div>
            <strong>Epochs</strong>
            <input
              type="number"
              min={1}
              max={20}
              value={epochs}
              onChange={(e) => setEpochs(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
            />
          </div>
          <div>
            <strong>Batch Size</strong>
            <input
              type="number"
              min={1}
              max={64}
              value={batchSize}
              onChange={(e) => setBatchSize(Math.max(1, Math.min(64, Number(e.target.value) || 1)))}
            />
          </div>
          <div>
            <strong>Seq Len</strong>
            <input
              type="number"
              min={16}
              max={512}
              value={seqLen}
              onChange={(e) => setSeqLen(Math.max(16, Math.min(512, Number(e.target.value) || 16)))}
            />
          </div>
          <div>
            <strong>Export Limit</strong>
            <input
              type="number"
              min={10}
              max={10000}
              value={exportLimit}
              onChange={(e) => setExportLimit(Math.max(10, Math.min(10000, Number(e.target.value) || 10)))}
            />
          </div>
        </div>
        <div className="sovereign-actions">
          <button type="button" className="ghost-btn" onClick={onExportSovereignData}>
            Export Sovereign Training Data
          </button>
          <button
            type="button"
            className="primary-btn"
            onClick={onLaunchTraining}
            disabled={jobState === "queued" || jobState === "running"}
          >
            {jobState === "queued" || jobState === "running" ? "Training..." : "Launch Training + Hot-Swap"}
          </button>
        </div>
        {exportStatus ? <p className="warning">{exportStatus}</p> : null}
        {trainingStatus ? <p className="warning">{trainingStatus}</p> : null}
        {trainingLogTail ? <pre className="training-log">{trainingLogTail}</pre> : null}
      </section>

      <VoiceControls
        onTranscript={setInput}
        onSend={onSendFromVoice}
        onClearInput={onClearInputFromVoice}
        speechText={reply?.response ?? ""}
      />

      <article className="response-card">
        <h2>Response</h2>
        <p>{reply?.response ?? "No response yet."}</p>
        <div className="feedback-row">
          <button type="button" className="ghost-btn" onClick={() => onFeedback("user_confirmation")}>
            Helpful
          </button>
          <button type="button" className="ghost-btn" onClick={() => onFeedback("user_negative_feedback")}>
            Off target
          </button>
          <button type="button" className="ghost-btn" onClick={onCorrection}>
            Correct
          </button>
        </div>
        {feedbackStatus ? <p className="warning">{feedbackStatus}</p> : null}
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
          <div>
            <strong>Fallback Used</strong>
            <span>{reply?.metadata?.sovereign?.fallback_used ? "Yes" : "No"}</span>
          </div>
        </div>
        {reply?.metadata?.warning ? <p className="warning">{reply.metadata.warning}</p> : null}
        {reply?.error ? <p className="warning">{String(reply.error)}</p> : null}
      </article>
    </section>
  );
}