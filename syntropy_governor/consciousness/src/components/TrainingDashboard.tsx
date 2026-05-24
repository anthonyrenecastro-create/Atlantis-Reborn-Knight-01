import { useEffect, useMemo, useState } from "react";
import { useAtlantean, type ModelSwapEvent, type TrainingJob } from "../hooks/useAtlantean";

function formatTime(ts?: number | null) {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleString();
}

export default function TrainingDashboard() {
  const { listTrainingJobs, launchTrainingJob, reloadModel, listModelSwapHistory, refreshSovereignStatus, sovereign } =
    useAtlantean();
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState<string | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [swapHistory, setSwapHistory] = useState<ModelSwapEvent[]>([]);

  const [epochs, setEpochs] = useState(2);
  const [batchSize, setBatchSize] = useState(8);
  const [seqLen, setSeqLen] = useState(128);
  const [exportLimit, setExportLimit] = useState(1500);

  async function refreshJobs() {
    setStatus("loading");
    setError(null);
    try {
      const rows = await listTrainingJobs();
      const swapRows = await listModelSwapHistory(30);
      setJobs(rows);
      setSwapHistory(swapRows);
      await refreshSovereignStatus();
      setStatus("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch training jobs.");
      setStatus("error");
    }
  }

  useEffect(() => {
    void refreshJobs();
    const id = window.setInterval(() => {
      void refreshJobs();
    }, 4000);
    return () => window.clearInterval(id);
  }, []);

  const runningJob = useMemo(
    () => jobs.find((job) => job.status === "running" || job.status === "queued") || null,
    [jobs],
  );

  const selectedJob = useMemo(() => {
    if (!jobs.length) {
      return null;
    }
    if (selectedJobId) {
      return jobs.find((job) => job.id === selectedJobId) || jobs[0];
    }
    return runningJob || jobs[0];
  }, [jobs, selectedJobId, runningJob]);

  async function startTraining() {
    setError(null);
    try {
      await launchTrainingJob({
        exportLimit,
        epochs,
        batchSize,
        seqLen,
        autoHotSwap: true,
      });
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to launch training.");
    }
  }

  async function hotSwap(job: TrainingJob) {
    if (!job.output_checkpoint) {
      setError("No checkpoint path available for this job.");
      return;
    }
    setError(null);
    try {
      await reloadModel(job.output_checkpoint);
      await refreshSovereignStatus();
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Hot-swap failed.");
      await refreshJobs();
    }
  }

  return (
    <section className="training-dashboard">
      <header className="training-header">
        <div>
          <h3>Sovereign Training Dashboard</h3>
          <p>Track jobs, inspect outputs, and hot-swap completed checkpoints.</p>
        </div>
        <button type="button" className="ghost-btn" onClick={() => void refreshJobs()}>
          Refresh
        </button>
      </header>

      <div className="training-grid">
        <div>
          <strong>Active Model</strong>
          <span>{sovereign?.active_model_path ?? "-"}</span>
        </div>
        <div>
          <strong>Running Job</strong>
          <span>{runningJob?.id ?? "none"}</span>
        </div>
        <div>
          <strong>Status</strong>
          <span>{status}</span>
        </div>
      </div>

      <div className="training-controls">
        <label>
          Epochs
          <input
            type="number"
            min={1}
            max={20}
            value={epochs}
            onChange={(e) => setEpochs(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
          />
        </label>
        <label>
          Batch
          <input
            type="number"
            min={1}
            max={64}
            value={batchSize}
            onChange={(e) => setBatchSize(Math.max(1, Math.min(64, Number(e.target.value) || 1)))}
          />
        </label>
        <label>
          Seq Len
          <input
            type="number"
            min={16}
            max={512}
            value={seqLen}
            onChange={(e) => setSeqLen(Math.max(16, Math.min(512, Number(e.target.value) || 16)))}
          />
        </label>
        <label>
          Export Limit
          <input
            type="number"
            min={10}
            max={10000}
            value={exportLimit}
            onChange={(e) => setExportLimit(Math.max(10, Math.min(10000, Number(e.target.value) || 10)))}
          />
        </label>
        <button type="button" className="primary-btn" onClick={() => void startTraining()}>
          Launch Training
        </button>
      </div>

      {error ? <p className="warning">{error}</p> : null}

      <div className="training-jobs-table-wrap">
        <table className="training-jobs-table">
          <thead>
            <tr>
              <th>Job</th>
              <th>Status</th>
              <th>Created</th>
              <th>Finished</th>
              <th>Checkpoint</th>
              <th>Hot-Swap</th>
            </tr>
          </thead>
          <tbody>
            {jobs.length === 0 ? (
              <tr>
                <td colSpan={6}>No jobs yet.</td>
              </tr>
            ) : (
              jobs.map((job) => (
                <tr key={job.id}>
                  <td>
                    <button
                      type="button"
                      className="ghost-btn"
                      onClick={() => setSelectedJobId(job.id)}
                    >
                      {job.id}
                    </button>
                  </td>
                  <td>{job.status}</td>
                  <td>{formatTime(job.created_at)}</td>
                  <td>{formatTime(job.finished_at)}</td>
                  <td title={job.output_checkpoint || ""}>{job.output_checkpoint || "-"}</td>
                  <td>
                    <button
                      type="button"
                      className="ghost-btn"
                      disabled={!job.output_checkpoint || job.status !== "completed"}
                      onClick={() => void hotSwap(job)}
                    >
                      Swap
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {selectedJob ? (
        <section className="training-job-detail">
          <header>
            <h4>Job Detail Drawer</h4>
            <p>Selected: {selectedJob.id}</p>
          </header>
          <div className="training-grid">
            <div>
              <strong>Status</strong>
              <span>{selectedJob.status}</span>
            </div>
            <div>
              <strong>Started</strong>
              <span>{formatTime(selectedJob.started_at)}</span>
            </div>
            <div>
              <strong>Finished</strong>
              <span>{formatTime(selectedJob.finished_at)}</span>
            </div>
            <div>
              <strong>Exit Code</strong>
              <span>{selectedJob.process_exit_code ?? "-"}</span>
            </div>
            <div>
              <strong>Dataset Path</strong>
              <span title={selectedJob.dataset_path || ""}>{selectedJob.dataset_path || "-"}</span>
            </div>
            <div>
              <strong>Checkpoint</strong>
              <span title={selectedJob.output_checkpoint || ""}>{selectedJob.output_checkpoint || "-"}</span>
            </div>
            <div>
              <strong>Auto Reload</strong>
              <span>{selectedJob.reload?.status ?? "-"}</span>
            </div>
            <div>
              <strong>Active Model</strong>
              <span title={selectedJob.reload?.active_model_path || ""}>{selectedJob.reload?.active_model_path || "-"}</span>
            </div>
          </div>
          {selectedJob.error ? <p className="warning">{selectedJob.error}</p> : null}
          {selectedJob.log_tail ? <pre className="training-log">{selectedJob.log_tail}</pre> : null}
        </section>
      ) : null}

      <section className="training-swap-history">
        <header>
          <h4>Swap History</h4>
            <p>Backend-backed checkpoint swap ledger.</p>
        </header>
        {swapHistory.length === 0 ? (
          <p className="warning">No swap events yet.</p>
        ) : (
          <ul>
            {swapHistory.map((event) => (
              <li key={event.id} className={event.status === "success" ? "swap-ok" : "swap-failed"}>
                  [{new Date(event.timestamp * 1000).toLocaleTimeString()}] {event.status.toUpperCase()} | source=
                  {event.source || "unknown"}
                  {event.job_id ? ` | job=${event.job_id}` : ""}
                  {event.model_path ? ` | checkpoint=${event.model_path}` : ""}
                  {event.active_model_path ? ` | active=${event.active_model_path}` : ""}
                {event.error ? ` | error=${event.error}` : ""}
              </li>
            ))}
          </ul>
        )}
      </section>

      {jobs[0]?.log_tail ? <pre className="training-log">{jobs[0].log_tail}</pre> : null}
    </section>
  );
}
