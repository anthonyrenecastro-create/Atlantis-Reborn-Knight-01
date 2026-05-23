import { useCallback, useMemo, useState } from "react";

export type BackendResponse = {
  response?: string;
  error?: string | boolean;
  metadata?: {
    mode?: string;
    latency_ms?: number;
    warning?: string | null;
  };
  field_state?: {
    phi1_mean?: number;
    phi5_mean?: number;
    Phi?: number;
  };
};

export type AtlanteanFieldState = {
  phi1_mean: number;
  phi5_mean: number;
  Phi: number;
};

export type SimulationRecord = {
  id: string;
  timestamp: number;
  prompt: string;
  response: string;
  mode: string;
  latency_ms: number;
  tokens_generated: number;
  warning?: string | null;
  field_state: AtlanteanFieldState;
  version: number;
};

type HealthState = "unknown" | "healthy" | "down";

const configuredApiBase = (import.meta.env.VITE_API_URL ?? "").trim();
const API_BASE = configuredApiBase;

function normalizeFieldState(data: unknown): AtlanteanFieldState | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const raw = data as {
    phi1_mean?: number;
    phi5_mean?: number;
    Phi?: number;
    phi1?: number;
    phi5?: number;
  };

  const phi1 = typeof raw.phi1_mean === "number" ? raw.phi1_mean : raw.phi1;
  const phi5 = typeof raw.phi5_mean === "number" ? raw.phi5_mean : raw.phi5;
  const Phi = raw.Phi;

  if (typeof phi1 !== "number" || typeof phi5 !== "number" || typeof Phi !== "number") {
    return null;
  }

  return {
    phi1_mean: phi1,
    phi5_mean: phi5,
    Phi,
  };
}

function normalizeSimulationRecord(data: unknown): SimulationRecord | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const raw = data as Record<string, unknown>;
  const field = normalizeFieldState(raw.field_state);
  if (!field) {
    return null;
  }

  return {
    id: typeof raw.id === "string" ? raw.id : `sim-${Date.now()}`,
    timestamp: typeof raw.timestamp === "number" ? raw.timestamp : Date.now() / 1000,
    prompt: typeof raw.prompt === "string" ? raw.prompt : "",
    response: typeof raw.response === "string" ? raw.response : "",
    mode: typeof raw.mode === "string" ? raw.mode : "unknown",
    latency_ms: typeof raw.latency_ms === "number" ? raw.latency_ms : 0,
    tokens_generated: typeof raw.tokens_generated === "number" ? raw.tokens_generated : 0,
    warning: typeof raw.warning === "string" ? raw.warning : null,
    field_state: field,
    version: typeof raw.version === "number" ? raw.version : 0,
  };
}

export function useAtlantean() {
  const [health, setHealth] = useState<HealthState>("unknown");
  const [isLoading, setIsLoading] = useState(false);
  const [reply, setReply] = useState<BackendResponse | null>(null);
  const [fieldState, setFieldState] = useState<AtlanteanFieldState | null>(null);

  const statusText = useMemo(() => {
    if (health === "healthy") return "Backend online";
    if (health === "down") return "Backend unavailable";
    return "Backend status unknown";
  }, [health]);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      setHealth(res.ok ? "healthy" : "down");
      return res.ok;
    } catch {
      setHealth("down");
      return false;
    }
  }, []);

  const refreshFields = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/atlantean/fields`);
      if (!res.ok) {
        return null;
      }
      const data = (await res.json()) as unknown;
      const next = normalizeFieldState(data);
      if (next) {
        setFieldState(next);
      }
      return next;
    } catch {
      return null;
    }
  }, []);

  const submitQuery = useCallback(async (input: string) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/atlantean/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
      });
      const data = (await res.json()) as BackendResponse;
      setReply(data);
      setHealth("healthy");
      const next = normalizeFieldState(data.field_state);
      if (next) {
        setFieldState(next);
      }
      return data;
    } catch (err) {
      const fallback = { error: err instanceof Error ? err.message : "Request failed" };
      setReply(fallback);
      setHealth("down");
      return fallback;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchSimulations = useCallback(async (opts?: { query?: string; limit?: number }) => {
    const query = opts?.query?.trim() ?? "";
    const limit = Math.max(1, Math.min(500, opts?.limit ?? 120));
    const params = new URLSearchParams({ limit: String(limit) });
    if (query) {
      params.set("q", query);
    }

    const res = await fetch(`${API_BASE}/api/atlantean/simulations?${params.toString()}`);
    if (!res.ok) {
      throw new Error(`Simulation fetch failed (${res.status})`);
    }

    const data = (await res.json()) as { simulations?: unknown[] };
    const rows = Array.isArray(data.simulations) ? data.simulations : [];
    return rows.map(normalizeSimulationRecord).filter((row): row is SimulationRecord => row !== null);
  }, []);

  return {
    apiBase: API_BASE,
    health,
    statusText,
    isLoading,
    reply,
    fieldState,
    checkHealth,
    submitQuery,
    refreshFields,
    fetchSimulations,
  };
}
