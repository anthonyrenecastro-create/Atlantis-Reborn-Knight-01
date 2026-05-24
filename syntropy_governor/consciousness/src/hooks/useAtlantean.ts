import { useCallback, useMemo, useState } from "react";

export type BackendResponse = {
  interaction_id?: string;
  response?: string;
  error?: string | boolean;
  decision_output?: {
    schema_version?: string;
    intent?: string;
    state_estimate?: Record<string, unknown> | string;
    hypothesis?: string;
    next_action?: string;
    expected_signal?: string;
    selected_option?: string;
    options?: Array<{
      id?: string;
      action?: string;
      confidence?: number;
      rationale?: string;
      risk?: string;
      evidence_needed?: string[];
    }>;
    guardrails?: string[];
    memory_trace?: {
      used_learned_hint?: boolean;
      learned_hint?: string | null;
      profile?: {
        match_count?: number;
        preferences?: string[];
        recurring_topics?: string[];
      };
    };
    influences?: {
      field?: {
        weight?: number;
        lens?: string;
        depth_bias?: number;
        semantic_delta?: number;
      };
      memory?: {
        weight?: number;
        grounding_line_applied?: boolean;
      };
    };
  };
  metadata?: {
    mode?: string;
    latency_ms?: number;
    warning?: string | null;
    llm_mediator?: {
      provider_requested?: string;
      verbosity?: "high" | "normal" | "brief";
      gemini_attempted?: boolean;
      gemini_used?: boolean;
      model?: string | null;
      error?: string | null;
      mentor_alignment?: number | null;
    };
    sovereign?: {
      local_only?: boolean;
      local_quality?: number;
      local_quality_min?: number;
      intent_confidence?: number;
      clear_intent_min?: number;
      fallback_reason?: string;
      fallback_used?: boolean;
      learning_strength?: number;
    };
    pipeline?: {
      stages?: Record<string, {
        enabled?: boolean;
        status?: string;
        session_id?: string;
        error?: string;
        finalizer?: string;
        local_log_path?: string;
      }>;
    };
  };
  field_state?: {
    phi1_mean?: number;
    phi5_mean?: number;
    Phi?: number;
    learning_capacity?: number;
  };
};

export type AtlanteanFieldState = {
  phi1_mean: number;
  phi5_mean: number;
  Phi: number;
};

export type SovereignStatus = {
  local_only: boolean;
  stats?: Record<string, number | string | null>;
  training_export_path?: string;
  active_model_path?: string | null;
};

export type TrainingJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: number;
  updated_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  params?: Record<string, unknown>;
  dataset_path?: string | null;
  output_checkpoint?: string | null;
  process_exit_code?: number | null;
  reload?: {
    status?: string;
    active_model_path?: string | null;
    error?: string;
  } | null;
  error?: string | null;
  log_tail?: string | null;
};

export type ModelSwapEvent = {
  id: string;
  timestamp: number;
  source?: string | null;
  job_id?: string | null;
  model_path?: string | null;
  status: "success" | "failed";
  active_model_path?: string | null;
  error?: string | null;
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

export type ResponseVerbosity = "high" | "normal" | "brief";

type HealthState = "unknown" | "healthy" | "down";

const configuredApiBase = (import.meta.env.VITE_API_URL ?? "").trim();
const configuredFallback = (import.meta.env.VITE_API_FALLBACK ?? "").trim();
const inferredFallback =
  typeof window !== "undefined" && window.location?.hostname
    ? `${window.location.protocol}//${window.location.hostname}:5001`
    : "http://127.0.0.1:5001";

const API_BASE = configuredApiBase;
const API_FALLBACK = configuredFallback || inferredFallback;

async function fetchApi(path: string, init?: RequestInit): Promise<Response> {
  const primaryUrl = `${API_BASE}${path}`;
  try {
    const primary = await fetch(primaryUrl, init);
    if (primary.ok || API_BASE || !API_FALLBACK || primaryUrl === `${API_FALLBACK}${path}`) {
      return primary;
    }

    const fallback = await fetch(`${API_FALLBACK}${path}`, init);
    return fallback;
  } catch {
    if (!API_FALLBACK || primaryUrl === `${API_FALLBACK}${path}`) {
      throw new Error("Request failed");
    }
    return fetch(`${API_FALLBACK}${path}`, init);
  }
}

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
  const [sovereign, setSovereign] = useState<SovereignStatus | null>(null);

  const statusText = useMemo(() => {
    if (health === "healthy") return "Backend online";
    if (health === "down") return "Backend unavailable";
    return "Backend status unknown";
  }, [health]);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetchApi("/health");
      setHealth(res.ok ? "healthy" : "down");
      return res.ok;
    } catch {
      setHealth("down");
      return false;
    }
  }, []);

  const refreshFields = useCallback(async () => {
    try {
      const res = await fetchApi("/api/atlantean/fields");
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

  const refreshSovereignStatus = useCallback(async () => {
    try {
      const res = await fetchApi("/api/atlantean/status");
      if (!res.ok) {
        return null;
      }
      const data = (await res.json()) as { sovereign?: SovereignStatus };
      if (data.sovereign) {
        setSovereign(data.sovereign);
      }
      return data.sovereign ?? null;
    } catch {
      return null;
    }
  }, []);

  const submitQuery = useCallback(async (input: string, opts?: { verbosity?: ResponseVerbosity }) => {
    setIsLoading(true);
    try {
      const verbosity = opts?.verbosity ?? "normal";
      const res = await fetchApi("/api/atlantean/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input, verbosity }),
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

    const res = await fetchApi(`/api/atlantean/simulations?${params.toString()}`);
    if (!res.ok) {
      throw new Error(`Simulation fetch failed (${res.status})`);
    }

    const data = (await res.json()) as { simulations?: unknown[] };
    const rows = Array.isArray(data.simulations) ? data.simulations : [];
    return rows.map(normalizeSimulationRecord).filter((row): row is SimulationRecord => row !== null);
  }, []);

  const sendFeedback = useCallback(
    async (
      event: "user_confirmation" | "user_negative_feedback" | "user_correction" | "high_engagement",
      opts?: { intensity?: number; interactionId?: string; correction?: string },
    ) => {
      const res = await fetchApi("/api/atlantean/learning-event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event,
          intensity: opts?.intensity ?? 0.6,
          interaction_id: opts?.interactionId,
          correction: opts?.correction,
        }),
      });
      if (!res.ok) {
        throw new Error(`Feedback failed (${res.status})`);
      }
      return (await res.json()) as {
        status: string;
        event: string;
        new_field_state?: AtlanteanFieldState;
      };
    },
    [],
  );

  const exportTrainingDataset = useCallback(
    async (opts?: { limit?: number }) => {
      const limit = Math.max(1, Math.min(5000, opts?.limit ?? 1000));
      const params = new URLSearchParams({
        limit: String(limit),
      });

      const res = await fetchApi(`/api/atlantean/training/export?${params.toString()}`);
      if (!res.ok) {
        throw new Error(`Training export failed (${res.status})`);
      }

      return (await res.json()) as {
        count: number;
        export_path: string | null;
      };
    },
    [],
  );

  const launchTrainingJob = useCallback(
    async (opts?: {
      exportLimit?: number;
      epochs?: number;
      batchSize?: number;
      seqLen?: number;
      lr?: number;
      maxRows?: number;
      device?: string;
      checkpointPath?: string;
      outputPath?: string;
      autoHotSwap?: boolean;
    }) => {
      const payload = {
        export_limit: Math.max(10, Math.min(10000, opts?.exportLimit ?? 1500)),
        epochs: Math.max(1, Math.min(20, opts?.epochs ?? 2)),
        batch_size: Math.max(1, Math.min(64, opts?.batchSize ?? 8)),
        seq_len: Math.max(16, Math.min(512, opts?.seqLen ?? 128)),
        lr: opts?.lr ?? 3e-4,
        max_rows: Math.max(0, Math.min(200000, opts?.maxRows ?? 0)),
        device: opts?.device ?? "cpu",
        checkpoint_path: opts?.checkpointPath,
        output_path: opts?.outputPath,
        auto_hot_swap: opts?.autoHotSwap ?? true,
      };

      const res = await fetchApi("/api/atlantean/training/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Training job launch failed (${res.status}): ${errText}`);
      }

      return (await res.json()) as { job_id: string; status: string };
    },
    [],
  );

  const getTrainingJob = useCallback(async (jobId: string) => {
    const res = await fetchApi(`/api/atlantean/training/jobs/${encodeURIComponent(jobId)}`);
    if (!res.ok) {
      throw new Error(`Training job fetch failed (${res.status})`);
    }
    return (await res.json()) as TrainingJob;
  }, []);

  const listTrainingJobs = useCallback(async () => {
    const res = await fetchApi("/api/atlantean/training/jobs");
    if (!res.ok) {
      throw new Error(`Training jobs list failed (${res.status})`);
    }
    const data = (await res.json()) as { jobs?: TrainingJob[] };
    return Array.isArray(data.jobs) ? data.jobs : [];
  }, []);

  const reloadModel = useCallback(async (modelPath: string) => {
    const res = await fetchApi("/api/atlantean/model/reload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_path: modelPath }),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Model reload failed (${res.status}): ${errText}`);
    }
    return (await res.json()) as { status: string; active_model_path?: string };
  }, []);

  const listModelSwapHistory = useCallback(async (limit = 30) => {
    const res = await fetchApi(`/api/atlantean/model/reload/history?limit=${encodeURIComponent(String(limit))}`);
    if (!res.ok) {
      throw new Error(`Model swap history fetch failed (${res.status})`);
    }
    const data = (await res.json()) as { events?: ModelSwapEvent[] };
    return Array.isArray(data.events) ? data.events : [];
  }, []);

  return {
    apiBase: API_BASE,
    health,
    statusText,
    isLoading,
    reply,
    fieldState,
    sovereign,
    checkHealth,
    submitQuery,
    sendFeedback,
    refreshFields,
    refreshSovereignStatus,
    fetchSimulations,
    exportTrainingDataset,
    launchTrainingJob,
    getTrainingJob,
    listTrainingJobs,
    reloadModel,
    listModelSwapHistory,
  };
}
