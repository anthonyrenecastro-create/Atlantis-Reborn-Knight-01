import { FormEvent, useEffect, useState } from "react";
import { useAtlantean } from "../hooks/useAtlantean";
import { useTheme } from "../context/ThemeContext";
import VoiceControls from "./VoiceControls";

type ResponseVerbosity = "high" | "normal" | "brief";

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
  const [verbosity, setVerbosity] = useState<ResponseVerbosity>("normal");

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
    await submitQuery(input, { verbosity });
    await refreshSovereignStatus();
  }

  async function onSendFromVoice() {
    if (!input.trim()) {
      return;
    }
    await submitQuery(input, { verbosity });
    await refreshSovereignStatus();
  }

  const decision = reply?.decision_output;
  const decisionOptions = decision?.options ?? [];
  const selectedOption = decision?.options?.find((item) => item.id === decision.selected_option);
  const memoryProfile = decision?.memory_trace?.profile;
  const pipelineStages = reply?.metadata?.pipeline?.stages || {};
  const quadraStage = pipelineStages["quadra_seer:final_output_integration"];
  const stateEstimate =
    typeof decision?.state_estimate === "string"
      ? decision.state_estimate
      : decision?.state_estimate
        ? JSON.stringify(decision.state_estimate)
        : "-";

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
        <div className="section-header">
          <h2>Prompt Composer</h2>
          <p>Shape the next field-guided query, then tune the level of detail before sending.</p>
        </div>
        <label htmlFor="prompt">Prompt</label>
        <textarea
          id="prompt"
          rows={4}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the field-modulated core..."
        />
        <label htmlFor="verbosity">Response Verbosity</label>
        <select
          id="verbosity"
          value={verbosity}
          onChange={(e) => setVerbosity(e.target.value as ResponseVerbosity)}
        >
          <option value="high">high</option>
          <option value="normal">normal</option>
          <option value="brief">brief</option>
        </select>
        <button type="submit" className="primary-btn" disabled={isLoading}>
          {isLoading ? "Querying..." : "Send"}
        </button>
      </form>

      <section className="sovereign-panel">
        <div className="sovereign-header">
          <div className="section-header">
            <h2>Sovereign Training Panel</h2>
            <p>Review local-only operating state and launch a hot-swap training cycle from the same surface.</p>
          </div>
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
        <div className="response-header section-header">
          <h2>Response</h2>
          <p>Generated output, reasoning traces, and memory influence summaries for the latest interaction.</p>
        </div>
        <div className="response-body">{reply?.response ?? "No response yet."}</div>
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
        <section className="meta-section">
          <h3>Operational Telemetry</h3>
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
              <strong>Learning Capacity</strong>
              <span>{reply?.field_state?.learning_capacity?.toFixed(4) ?? "-"}</span>
            </div>
            <div>
              <strong>Fallback Used</strong>
              <span>{reply?.metadata?.sovereign?.fallback_used ? "Yes" : "No"}</span>
            </div>
            <div>
              <strong>Intent Confidence</strong>
              <span>{reply?.metadata?.sovereign?.intent_confidence?.toFixed(3) ?? "-"}</span>
            </div>
            <div>
              <strong>Local Quality</strong>
              <span>{reply?.metadata?.sovereign?.local_quality?.toFixed(3) ?? "-"}</span>
            </div>
            <div>
              <strong>Mediator</strong>
              <span>
                {reply?.metadata?.llm_mediator?.gemini_used
                  ? `Gemini (${reply.metadata.llm_mediator.model || "model n/a"})`
                  : "Local stack"}
              </span>
            </div>
            <div>
              <strong>Verbosity</strong>
              <span>{reply?.metadata?.llm_mediator?.verbosity ?? verbosity}</span>
            </div>
            <div>
              <strong>Ledger Session</strong>
              <span>{quadraStage?.session_id ?? reply?.interaction_id ?? "-"}</span>
            </div>
            <div>
              <strong>Ledger Finalizer</strong>
              <span>{quadraStage?.finalizer ?? quadraStage?.status ?? "-"}</span>
            </div>
          </div>
        </section>

        <section className="meta-section">
          <h3>Decision Trace</h3>
          <div className="meta-grid">
          <div>
            <strong>Decision Intent</strong>
            <span>{decision?.intent ?? "-"}</span>
          </div>
          <div>
            <strong>Hypothesis</strong>
            <span>{decision?.hypothesis ?? "-"}</span>
          </div>
          <div>
            <strong>State Estimate</strong>
            <span>{stateEstimate}</span>
          </div>
          <div>
            <strong>Next Action</strong>
            <span>{decision?.next_action ?? "-"}</span>
          </div>
          <div>
            <strong>Expected Signal</strong>
            <span>{decision?.expected_signal ?? "-"}</span>
          </div>
          <div>
            <strong>Selected Confidence</strong>
            <span>{selectedOption?.confidence?.toFixed(3) ?? "-"}</span>
          </div>
          </div>
        </section>

        {decisionOptions.length > 0 ? (
          <div className="options-matrix-wrap">
            <h3>Candidate Confidence Matrix</h3>
            <table className="options-matrix-table">
              <thead>
                <tr>
                  <th>Option</th>
                  <th>Action</th>
                  <th>Confidence</th>
                  <th>Risk</th>
                  <th>Evidence Needed</th>
                  <th>Selected</th>
                </tr>
              </thead>
              <tbody>
                {decisionOptions.map((option, idx) => {
                  const optionKey = option.id ?? `option-${idx}`;
                  const evidence = option.evidence_needed?.join(" | ") || "-";
                  const isSelected = Boolean(option.id && option.id === decision?.selected_option);
                  return (
                    <tr key={optionKey} className={isSelected ? "is-selected" : ""}>
                      <td>{option.id ?? "-"}</td>
                      <td>{option.action ?? "-"}</td>
                      <td>{option.confidence?.toFixed(3) ?? "-"}</td>
                      <td>{option.risk ?? "-"}</td>
                      <td>{evidence}</td>
                      <td>{isSelected ? "yes" : "-"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}

        <section className="meta-section">
          <h3>Memory and Influence</h3>
          <div className="meta-grid">
            <div>
              <strong>Recalled Learned Hint</strong>
              <span>{decision?.memory_trace?.learned_hint ?? "-"}</span>
            </div>
            <div>
              <strong>Memory Match Count</strong>
              <span>{String(memoryProfile?.match_count ?? "-")}</span>
            </div>
            <div>
              <strong>Preferences</strong>
              <span>{memoryProfile?.preferences?.join(" | ") || "-"}</span>
            </div>
            <div>
              <strong>Recurring Topics</strong>
              <span>{memoryProfile?.recurring_topics?.join(" | ") || "-"}</span>
            </div>
            <div>
              <strong>Field Influence</strong>
              <span>{decision?.influences?.field?.weight?.toFixed(3) ?? "-"}</span>
            </div>
            <div>
              <strong>Memory Influence</strong>
              <span>{decision?.influences?.memory?.weight?.toFixed(3) ?? "-"}</span>
            </div>
            <div>
              <strong>Field Lens</strong>
              <span>{decision?.influences?.field?.lens ?? "-"}</span>
            </div>
            <div>
              <strong>HRM Context</strong>
              <span>Not yet exposed in unified payload</span>
            </div>
          </div>
        </section>

        <section className="meta-section">
          <h3>Guardrails and Ledger</h3>
          <div className="meta-grid">
            <div>
              <strong>Guardrails</strong>
              <span>{decision?.guardrails?.join(" | ") || "-"}</span>
            </div>
            <div>
              <strong>Fallback Reason</strong>
              <span>{reply?.metadata?.sovereign?.fallback_reason ?? "-"}</span>
            </div>
            <div>
              <strong>Ledger Log Path</strong>
              <span>{quadraStage?.local_log_path ?? "-"}</span>
            </div>
            <div>
              <strong>Integrity Status</strong>
              <span>{quadraStage?.enabled ? "integrated" : quadraStage?.status ?? "unknown"}</span>
            </div>
          </div>
        </section>
        {reply?.metadata?.warning ? <p className="warning">{reply.metadata.warning}</p> : null}
        {reply?.error ? <p className="warning">{String(reply.error)}</p> : null}
      </article>
    </section>
  );
}