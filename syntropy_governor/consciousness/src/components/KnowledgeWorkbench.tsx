import { ChangeEvent, useEffect, useMemo, useState } from "react";
import mermaid from "mermaid";

type ParsedRow = Record<string, string | number>;

function parseCsv(content: string): ParsedRow[] {
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return [];
  }

  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const row: ParsedRow = {};
    headers.forEach((key, idx) => {
      const raw = (cols[idx] ?? "").trim();
      const maybeNumber = Number(raw);
      row[key] = Number.isFinite(maybeNumber) && raw !== "" ? maybeNumber : raw;
    });
    return row;
  });
}

function extractMermaidBlock(text: string): string | null {
  const match = text.match(/```mermaid\s*([\s\S]*?)```/i);
  return match?.[1]?.trim() || null;
}

function pickNumericSeries(rows: ParsedRow[]) {
  if (rows.length === 0) {
    return { label: "value", values: [] as number[] };
  }

  const keys = Object.keys(rows[0]);
  for (const key of keys) {
    const values = rows
      .map((row) => row[key])
      .filter((value): value is number => typeof value === "number");
    if (values.length > 1) {
      return { label: key, values };
    }
  }

  return { label: "value", values: [] as number[] };
}

export default function KnowledgeWorkbench() {
  const [rawText, setRawText] = useState("");
  const [rows, setRows] = useState<ParsedRow[]>([]);
  const [search, setSearch] = useState("");
  const [parseSummary, setParseSummary] = useState("No file parsed yet.");
  const [mermaidSource, setMermaidSource] = useState(
    "graph TD\n  User[User Prompt] --> Core[Atlantean Core]\n  Core --> Memory[Neural Archives]\n  Core --> Reply[Field Response]",
  );
  const [mermaidSvg, setMermaidSvg] = useState<string>("");
  const [mermaidError, setMermaidError] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;

    async function renderDiagram() {
      setMermaidError(null);
      try {
        mermaid.initialize({ startOnLoad: false, securityLevel: "loose", theme: "dark" });
        const id = `mermaid-${Date.now()}`;
        const { svg } = await mermaid.render(id, mermaidSource);
        if (!canceled) {
          setMermaidSvg(svg);
        }
      } catch (err) {
        if (!canceled) {
          setMermaidError(err instanceof Error ? err.message : "Mermaid rendering failed");
          setMermaidSvg("");
        }
      }
    }

    void renderDiagram();
    return () => {
      canceled = true;
    };
  }, [mermaidSource]);

  async function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) {
      return;
    }

    const text = await file.text();
    setRawText(text);

    const lower = file.name.toLowerCase();
    if (lower.endsWith(".csv")) {
      const parsed = parseCsv(text);
      setRows(parsed);
      setParseSummary(`Parsed CSV with ${parsed.length} rows.`);
    } else if (lower.endsWith(".json")) {
      try {
        const parsed = JSON.parse(text) as unknown;
        if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === "object") {
          setRows(parsed as ParsedRow[]);
          setParseSummary(`Parsed JSON array with ${parsed.length} rows.`);
        } else {
          setRows([]);
          setParseSummary("Parsed JSON object (non-tabular).");
        }
      } catch {
        setRows([]);
        setParseSummary("Invalid JSON file.");
      }
    } else {
      setRows([]);
      setParseSummary(`Parsed text file (${text.length} characters).`);
    }

    const block = extractMermaidBlock(text);
    if (block) {
      setMermaidSource(block);
    }
  }

  const filteredRows = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) {
      return rows;
    }
    return rows.filter((row) => JSON.stringify(row).toLowerCase().includes(query));
  }, [rows, search]);

  const chart = useMemo(() => pickNumericSeries(filteredRows), [filteredRows]);
  const maxValue = Math.max(1, ...chart.values.map((v) => Math.abs(v)));

  return (
    <section className="workbench-shell">
      <header className="workbench-header">
        <h2>Knowledge Workbench</h2>
        <p>Mermaid diagrams, parsed files, and quick charts from uploaded data.</p>
      </header>

      <div className="workbench-grid">
        <article className="workbench-card">
          <h3>File Parsing</h3>
          <div className="workbench-actions">
            <input type="file" onChange={onFileChange} accept=".txt,.md,.csv,.json" />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search parsed rows"
            />
          </div>
          <p className="workbench-note">{parseSummary}</p>
          <details>
            <summary>Raw Text Preview</summary>
            <pre className="workbench-pre">{rawText.slice(0, 3000) || "No content loaded."}</pre>
          </details>
        </article>

        <article className="workbench-card">
          <h3>Mermaid</h3>
          <textarea
            rows={8}
            value={mermaidSource}
            onChange={(e) => setMermaidSource(e.target.value)}
            placeholder="Write Mermaid syntax here"
          />
          {mermaidError ? <p className="warning">{mermaidError}</p> : null}
          <div className="mermaid-preview" dangerouslySetInnerHTML={{ __html: mermaidSvg }} />
        </article>

        <article className="workbench-card">
          <h3>Chart</h3>
          <p className="workbench-note">
            Numeric series: <strong>{chart.label}</strong> ({chart.values.length} points)
          </p>
          <div className="chart-row" role="img" aria-label="Parsed data bar chart">
            {chart.values.length === 0 ? (
              <span className="workbench-note">No numeric series found. Upload CSV/JSON with numeric columns.</span>
            ) : (
              chart.values.slice(0, 24).map((value, idx) => {
                const pct = Math.max(4, (Math.abs(value) / maxValue) * 100);
                return (
                  <div key={`${chart.label}-${idx}`} className="chart-bar-wrap" title={`${value}`}>
                    <span className="chart-bar" style={{ height: `${pct}%` }} />
                  </div>
                );
              })
            )}
          </div>
        </article>
      </div>
    </section>
  );
}
