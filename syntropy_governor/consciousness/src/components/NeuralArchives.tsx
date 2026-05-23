import { useEffect, useMemo, useState } from "react";
import { useAtlantean, type AtlanteanFieldState, type SimulationRecord } from "../hooks/useAtlantean";
import FieldHeatmap from "./FieldHeatmap";
import SimulationVisualizer from "./SimulationVisualizer";

function clamp01(n: number) {
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

function makeHeatmap(field: AtlanteanFieldState | null) {
  const phi1 = field?.phi1_mean ?? 0;
  const phi5 = field?.phi5_mean ?? 0;
  const Phi = field?.Phi ?? 0;

  return Array.from({ length: 72 }).map((_, i) => {
    const wave = Math.sin(i * 0.19 + phi1) * 0.55;
    const grain = Math.cos(i * 0.11 + phi5) * 0.35;
    const coherence = (Phi - 0.5) * 0.6;
    return clamp01((wave + grain + coherence + 1) / 2);
  });
}

export default function NeuralArchives() {
  const { fieldState, refreshFields, checkHealth, health, fetchSimulations } = useAtlantean();
  const [history, setHistory] = useState<AtlanteanFieldState[]>([]);
  const [simulations, setSimulations] = useState<SimulationRecord[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [simulationError, setSimulationError] = useState<string | null>(null);
  const [isSimLoading, setIsSimLoading] = useState(false);

  async function refreshSimulationFeed(query: string) {
    setIsSimLoading(true);
    setSimulationError(null);
    try {
      const rows = await fetchSimulations({ query, limit: 120 });
      setSimulations(rows);
    } catch (err) {
      setSimulationError(err instanceof Error ? err.message : "Failed to load simulations");
    } finally {
      setIsSimLoading(false);
    }
  }

  function download(filename: string, content: string, contentType: string) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  function exportJson() {
    download(
      `neural-archives-${Date.now()}.json`,
      JSON.stringify(simulations, null, 2),
      "application/json;charset=utf-8",
    );
  }

  function exportCsv() {
    const header = [
      "id",
      "timestamp",
      "mode",
      "latency_ms",
      "tokens_generated",
      "phi1_mean",
      "phi5_mean",
      "Phi",
      "version",
      "prompt",
      "response",
    ];

    const rows = simulations.map((row) => [
      row.id,
      new Date(row.timestamp * 1000).toISOString(),
      row.mode,
      String(row.latency_ms),
      String(row.tokens_generated),
      String(row.field_state.phi1_mean),
      String(row.field_state.phi5_mean),
      String(row.field_state.Phi),
      String(row.version),
      row.prompt,
      row.response,
    ]);

    const toCell = (value: string) => `"${value.replace(/"/g, '""')}"`;
    const csv = [header.join(","), ...rows.map((cols) => cols.map(toCell).join(","))].join("\n");
    download(`neural-archives-${Date.now()}.csv`, csv, "text/csv;charset=utf-8");
  }

  useEffect(() => {
    let isMounted = true;

    async function bootstrap() {
      await checkHealth();
      const next = await refreshFields();
      if (isMounted && next) {
        setHistory((prev) => [next, ...prev].slice(0, 6));
      }
      await refreshSimulationFeed("");
    }

    void bootstrap();
    return () => {
      isMounted = false;
    };
  }, [checkHealth, refreshFields]);

  useEffect(() => {
    if (!fieldState) {
      return;
    }
    setHistory((prev) => {
      const top = prev[0];
      if (
        top &&
        Math.abs(top.phi1_mean - fieldState.phi1_mean) < 0.0001 &&
        Math.abs(top.phi5_mean - fieldState.phi5_mean) < 0.0001 &&
        Math.abs(top.Phi - fieldState.Phi) < 0.0001
      ) {
        return prev;
      }
      return [fieldState, ...prev].slice(0, 6);
    });
  }, [fieldState]);

  const heatmap = useMemo(() => makeHeatmap(fieldState), [fieldState]);

  return (
    <section className="archives-shell">
      <header className="archives-header">
        <div>
          <h2>Neural Archives</h2>
          <p>Live field memory telemetry and synthetic topology map.</p>
        </div>
        <div className="header-actions">
          <span className={`status-pill ${health}`}>{health === "healthy" ? "Synced" : "Idle"}</span>
          <button type="button" className="ghost-btn" onClick={() => void refreshFields()}>
            Refresh Fields
          </button>
        </div>
      </header>

      <div className="archives-grid">
        <div className="archives-stat">
          <strong>Decision Topology (phi1)</strong>
          <span>{fieldState?.phi1_mean?.toFixed(4) ?? "-"}</span>
        </div>
        <div className="archives-stat">
          <strong>Learning Capacity (phi5)</strong>
          <span>{fieldState?.phi5_mean?.toFixed(4) ?? "-"}</span>
        </div>
        <div className="archives-stat">
          <strong>Global Coherence (Phi)</strong>
          <span>{fieldState?.Phi?.toFixed(4) ?? "-"}</span>
        </div>
      </div>

      <div className="archives-controls">
        <label htmlFor="simulation-search">Search Simulations</label>
        <div className="archives-controls-row">
          <input
            id="simulation-search"
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="mode, prompt, response"
          />
          <button type="button" className="ghost-btn" onClick={() => void refreshSimulationFeed(searchQuery)}>
            {isSimLoading ? "Searching..." : "Search"}
          </button>
          <button type="button" className="ghost-btn" onClick={exportJson} disabled={simulations.length === 0}>
            Export JSON
          </button>
          <button type="button" className="ghost-btn" onClick={exportCsv} disabled={simulations.length === 0}>
            Export CSV
          </button>
        </div>
        {simulationError ? <p className="warning">{simulationError}</p> : null}
      </div>

      <FieldHeatmap title="Field Heatmap" values={heatmap} />

      <SimulationVisualizer simulations={simulations} />

      <div className="archives-history">
        <strong>Recent Backend Simulations</strong>
        <ul>
          {simulations.length === 0 ? (
            <li>No simulation records found.</li>
          ) : (
            simulations.slice(0, 12).map((item) => (
              <li key={item.id}>
                [{new Date(item.timestamp * 1000).toLocaleTimeString()}] {item.mode} | phi1=
                {item.field_state.phi1_mean.toFixed(3)} phi5={item.field_state.phi5_mean.toFixed(3)} Phi=
                {item.field_state.Phi.toFixed(3)} | {item.prompt.slice(0, 56)}
              </li>
            ))
          )}
        </ul>
      </div>
    </section>
  );
}
