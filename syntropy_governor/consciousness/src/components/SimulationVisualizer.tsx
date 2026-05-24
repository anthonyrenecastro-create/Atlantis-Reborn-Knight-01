import { useMemo } from "react";
import type { SimulationRecord } from "../hooks/useAtlantean";

type SimulationVisualizerProps = {
  simulations: SimulationRecord[];
};

function clamp(n: number, min: number, max: number) {
  if (n < min) return min;
  if (n > max) return max;
  return n;
}

function toPlotValue(v: number, min: number, max: number) {
  const denom = max - min;
  if (Math.abs(denom) < 1e-8) return 0.5;
  return clamp((v - min) / denom, 0, 1);
}

export default function SimulationVisualizer({ simulations }: SimulationVisualizerProps) {
  const simulationPoints = useMemo(() => {
    const source =
      simulations.length > 0
        ? [...simulations].reverse().map((row) => row.field_state)
        : [{ phi1_mean: 0, phi5_mean: 0, Phi: 0.5 }];
    const phi1Values = source.map((h) => h.phi1_mean);
    const phi5Values = source.map((h) => h.phi5_mean);
    const phiValues = source.map((h) => h.Phi);

    const phi1Min = Math.min(...phi1Values, -1);
    const phi1Max = Math.max(...phi1Values, 1);
    const phi5Min = Math.min(...phi5Values, -1);
    const phi5Max = Math.max(...phi5Values, 1);
    const phiMin = Math.min(...phiValues, 0);
    const phiMax = Math.max(...phiValues, 1);

    return source.map((sample, idx) => {
      const x = source.length === 1 ? 0 : idx / (source.length - 1);
      return {
        x,
        phi1: toPlotValue(sample.phi1_mean, phi1Min, phi1Max),
        phi5: toPlotValue(sample.phi5_mean, phi5Min, phi5Max),
        Phi: toPlotValue(sample.Phi, phiMin, phiMax),
      };
    });
  }, [simulations]);

  const coherenceSummary = useMemo(() => {
    const source = simulations.length > 0 ? [...simulations].reverse().map((row) => row.field_state.Phi) : [0.5];
    const latest = source[source.length - 1] ?? 0.5;
    const average = source.reduce((sum, value) => sum + value, 0) / source.length;
    const baseline = source.length > 1 ? source[0] : latest;
    const drift = latest - baseline;

    let band = "stabilizing";
    if (latest >= 0.7) {
      band = "resonant";
    } else if (latest < 0.45) {
      band = "fragile";
    }

    return {
      latest,
      average,
      drift,
      band,
    };
  }, [simulations]);

  function toPath(field: "phi1" | "phi5" | "Phi") {
    return simulationPoints
      .map((p, idx) => {
        const x = p.x * 100;
        const y = (1 - p[field]) * 100;
        return `${idx === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }

  return (
    <section className="simulation-shell">
      <header className="simulation-header">
        <h3>Simulation Visualizer</h3>
        <p>Projected trendlines from backend simulation records.</p>
      </header>
      <svg viewBox="0 0 100 100" className="simulation-chart" role="img" aria-label="Field simulation trend">
        <rect x="0" y="0" width="100" height="30" className="coherence-band resonant" />
        <rect x="0" y="30" width="100" height="25" className="coherence-band stabilizing" />
        <rect x="0" y="55" width="100" height="45" className="coherence-band fragile" />
        <polyline points="0,50 100,50" className="simulation-axis" />
        <path d={toPath("phi1")} className="sim-line phi1" />
        <path d={toPath("phi5")} className="sim-line phi5" />
        <path d={toPath("Phi")} className="sim-line phi" />
      </svg>
      <div className="coherence-summary-grid">
        <div>
          <strong>Current Coherence Band</strong>
          <span>{coherenceSummary.band}</span>
        </div>
        <div>
          <strong>Latest Phi</strong>
          <span>{coherenceSummary.latest.toFixed(3)}</span>
        </div>
        <div>
          <strong>Average Phi</strong>
          <span>{coherenceSummary.average.toFixed(3)}</span>
        </div>
        <div>
          <strong>Coherence Drift</strong>
          <span>{coherenceSummary.drift >= 0 ? `+${coherenceSummary.drift.toFixed(3)}` : coherenceSummary.drift.toFixed(3)}</span>
        </div>
      </div>
      <div className="simulation-legend">
        <span className="legend-item phi1">phi1</span>
        <span className="legend-item phi5">phi5</span>
        <span className="legend-item phi">Phi</span>
        <span className="legend-item band-resonant">resonant band</span>
        <span className="legend-item band-stabilizing">stabilizing band</span>
        <span className="legend-item band-fragile">fragile band</span>
      </div>
    </section>
  );
}
