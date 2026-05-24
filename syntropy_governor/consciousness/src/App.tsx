import React, { useState, useRef, useCallback, useEffect } from "react";
import { INITIAL_TASKS, generateChartData } from "./constants";
import { ChatMessage, FeedItem, FeedItemType, Screen, WindowInstance, WindowType } from "./types";
import ChatInterface from "./components/ChatInterface";
import NeuralArchives from "./components/NeuralArchives";
import TrainingDashboard from "./components/TrainingDashboard";

type ChatMode = "GENERAL" | "PHYSICS" | "COMPONENT_REPORT";
type ResponseVerbosity = "high" | "normal" | "brief";

type AtlanteanStatus = {
  field_state?: {
    phi1_mean?: number;
    phi5_mean?: number;
    Phi?: number;
  };
  learning_capacity?: number;
  core_brain?: string;
  llm_mediator?: {
    gemini_configured?: boolean;
    gemini_model?: string;
  };
  sovereign?: {
    local_only?: boolean;
    stats?: Record<string, number>;
  };
};

type AtlanteanQueryResponse = {
  response?: string;
  error?: string;
  field_state?: {
    phi1_mean?: number;
    phi5_mean?: number;
    Phi?: number;
  };
  metadata?: {
    mode?: string;
    llm_mediator?: {
      gemini_used?: boolean;
      error?: string | null;
    };
    sovereign?: {
      fallback_used?: boolean;
    };
  };
};

type MemoryHistoryEntry = {
  id: number;
  text: string;
};

type PhysicsSample = {
  t: number;
  phi1: number;
  phi5: number;
  Phi: number;
  learning: number;
};

type SpeechRecognitionResultItem = {
  transcript: string;
  confidence: number;
};

type SpeechRecognitionResultListLike = {
  [index: number]: SpeechRecognitionResultItem;
  length: number;
};

type SpeechRecognitionEventLike = Event & {
  resultIndex: number;
  results: {
    [index: number]: SpeechRecognitionResultListLike;
    length: number;
  };
};

type SpeechRecognitionLike = EventTarget & {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

type WindowWithSpeech = Window & {
  webkitSpeechRecognition?: SpeechRecognitionCtor;
  SpeechRecognition?: SpeechRecognitionCtor;
};

const fileToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = (err) => reject(err);
  });

const useSpeech = () => {
  const selectedVoice = useRef<SpeechSynthesisVoice | null>(null);

  useEffect(() => {
    const chooseVoice = () => {
      const voices = window.speechSynthesis.getVoices();
      selectedVoice.current =
        voices.find((voice) => /Samantha|Google US English|Aria|Neural/i.test(voice.name)) || voices[0] || null;
    };
    chooseVoice();
    window.speechSynthesis.onvoiceschanged = chooseVoice;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  return useCallback((text: string) => {
    if (!text || !("speechSynthesis" in window)) {
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = selectedVoice.current;
    utterance.rate = 1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  }, []);
};

const useVoiceRecognition = ({ onResult }: { onResult: (transcript: string) => void }) => {
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  useEffect(() => {
    const w = window as WindowWithSpeech;
    const RecognitionCtor = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!RecognitionCtor) {
      return;
    }

    const recognition = new RecognitionCtor();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event: SpeechRecognitionEventLike) => {
      const transcript = event.results[event.resultIndex]?.[0]?.transcript?.trim();
      if (transcript) {
        onResult(transcript);
      }
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.stop();
      recognitionRef.current = null;
    };
  }, [onResult]);

  const toggleListening = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) {
      return;
    }
    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      recognition.start();
      setIsListening(true);
    }
  }, [isListening]);

  return { isListening, toggleListening };
};

const AuraAvatar: React.FC = () => <div className="aura-avatar" aria-label="Aura avatar" />;

type TiledPanelProps = {
  win: WindowInstance;
  onClose: (id: number) => void;
  onToggleCollapse: (id: number) => void;
  onToggleExpand: (id: number) => void;
  isCollapsed: boolean;
  isExpanded: boolean;
  children: React.ReactNode;
};

const TiledPanel: React.FC<TiledPanelProps> = ({
  win,
  onClose,
  onToggleCollapse,
  onToggleExpand,
  isCollapsed,
  isExpanded,
  children,
}) => {
  return (
    <section className={`tile-panel ${isCollapsed ? "collapsed" : ""} ${isExpanded ? "expanded" : ""}`}>
      <header className="tile-header">
        <strong>{win.title}</strong>
        <div className="tile-actions">
          <button type="button" className="window-close" onClick={() => onToggleCollapse(win.id)}>
            {isCollapsed ? "+" : "-"}
          </button>
          <button type="button" className="window-close" onClick={() => onToggleExpand(win.id)}>
            {isExpanded ? "[]" : "<>"}
          </button>
          <button type="button" className="window-close" onClick={() => onClose(win.id)}>
            x
          </button>
        </div>
      </header>
      {!isCollapsed ? <div className="tile-body">{children}</div> : null}
    </section>
  );
};

type ChatWindowProps = {
  mode: ChatMode;
  speak: (text: string) => void;
  messages: ChatMessage[];
  isLoading: boolean;
  onSend: (text: string, verbosity: ResponseVerbosity) => void;
};

const ChatWindowComponent: React.FC<ChatWindowProps> = ({ mode, speak, messages, isLoading, onSend }) => {
  const [text, setText] = useState("");
  const [verbosity, setVerbosity] = useState<ResponseVerbosity>("normal");

  const modeDescription =
    mode === "PHYSICS"
      ? "Internal field metrics and dynamics"
      : mode === "COMPONENT_REPORT"
        ? "Component-by-component runtime health"
        : "General dialogue and planning";

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const value = text.trim();
    if (!value) {
      return;
    }
    onSend(value, verbosity);
    setText("");
  };

  return (
    <div className="chat-window">
      <p className="chat-mode-label">{modeDescription}</p>
      <div className="chat-history">
        {messages.map((message) => (
          <div key={message.id} className={`chat-bubble ${message.sender}`}>
            <span>{message.text}</span>
            {message.sender === "aura" ? (
              <button type="button" className="speak-btn" onClick={() => speak(message.text)}>
                Speak
              </button>
            ) : null}
          </div>
        ))}
      </div>
      <form className="chat-input-row" onSubmit={submit}>
        <select
          className="chat-verbosity-select"
          value={verbosity}
          onChange={(event) => setVerbosity(event.target.value as ResponseVerbosity)}
          aria-label="Response verbosity"
        >
          <option value="high">High</option>
          <option value="normal">Normal</option>
          <option value="brief">Brief</option>
        </select>
        <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Message Aura..." />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
};

const PhysicsMetricsWindow: React.FC<{
  fetchStatus: () => Promise<AtlanteanStatus | null>;
  onHotMemory: (text: string) => void;
}> = ({ fetchStatus, onHotMemory }) => {
  const [samples, setSamples] = useState<PhysicsSample[]>([]);
  const pollingRef = useRef(false);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      if (pollingRef.current || document.hidden) {
        return;
      }
      pollingRef.current = true;
      const status = await fetchStatus();
      pollingRef.current = false;
      if (!active || !status) {
        return;
      }
      const sample: PhysicsSample = {
        t: Date.now(),
        phi1: status.field_state?.phi1_mean ?? 0,
        phi5: status.field_state?.phi5_mean ?? 0,
        Phi: status.field_state?.Phi ?? 0,
        learning: status.learning_capacity ?? 0,
      };
      setSamples((prev) => [...prev.slice(-39), sample]);
      onHotMemory(
        `telemetry phi1=${sample.phi1.toFixed(4)} phi5=${sample.phi5.toFixed(4)} Phi=${sample.Phi.toFixed(4)} lc=${(sample.learning * 100).toFixed(2)}%`,
      );
    };

    void poll();
    const id = window.setInterval(() => {
      void poll();
    }, 5000);

    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [fetchStatus, onHotMemory]);

  const latest = samples[samples.length - 1] || { t: Date.now(), phi1: 0, phi5: 0, Phi: 0, learning: 0 };

  const chartPath = (key: keyof PhysicsSample, min: number, max: number) => {
    if (samples.length < 2) {
      return "";
    }
    return samples
      .map((sample, i) => {
        const x = (i / Math.max(1, samples.length - 1)) * 100;
        const value = Number(sample[key]);
        const normalized = (value - min) / Math.max(0.0001, max - min);
        const y = 100 - Math.min(100, Math.max(0, normalized * 100));
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  };

  const metricBar = (label: string, value: number, min = -1, max = 1) => {
    const pct = ((value - min) / Math.max(0.0001, max - min)) * 100;
    return (
      <div className="metric-card">
        <div className="metric-header">
          <span>{label}</span>
          <strong>{value.toFixed(4)}</strong>
        </div>
        <div className="metric-track">
          <div className="metric-fill" style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
        </div>
      </div>
    );
  };

  return (
    <div className="physics-window">
      <div className="metrics-grid">
        {metricBar("phi1", latest.phi1, -1, 1)}
        {metricBar("phi5", latest.phi5, -1, 1)}
        {metricBar("Phi", latest.Phi, -1, 1)}
        {metricBar("learning", latest.learning, 0, 1)}
      </div>

      <div className="physics-chart-shell">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="physics-chart">
          <path d={chartPath("phi1", -1, 1)} className="line-phi1" />
          <path d={chartPath("phi5", -1, 1)} className="line-phi5" />
          <path d={chartPath("Phi", -1, 1)} className="line-Phi" />
          <path d={chartPath("learning", 0, 1)} className="line-learning" />
        </svg>
      </div>
      <p className="physics-legend">Lines: phi1 (blue), phi5 (cyan), Phi (violet), learning (green)</p>
    </div>
  );
};

const ImageGeneratorWindowComponent: React.FC<{ initialPrompt?: string }> = ({ initialPrompt = "" }) => {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [result, setResult] = useState<string>("");

  return (
    <div className="generator-window">
      <label htmlFor="img-prompt">Image Prompt</label>
      <textarea id="img-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} />
      <button
        type="button"
        onClick={() => setResult(`Image task queued: "${prompt.trim() || "Untitled concept"}"`)}
      >
        Generate
      </button>
      {result ? <p>{result}</p> : null}
    </div>
  );
};

const VideoGeneratorWindowComponent: React.FC<{ initialPrompt?: string }> = ({ initialPrompt = "" }) => {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [result, setResult] = useState<string>("");

  return (
    <div className="generator-window">
      <label htmlFor="video-prompt">Video Prompt</label>
      <textarea id="video-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} />
      <button
        type="button"
        onClick={() => setResult(`Video task queued: "${prompt.trim() || "Untitled sequence"}"`)}
      >
        Generate
      </button>
      {result ? <p>{result}</p> : null}
    </div>
  );
};

const BrowserWindowComponent: React.FC<{ query: string }> = ({ query }) => {
  const searchUrl = `https://duckduckgo.com/?q=${encodeURIComponent(query)}`;
  return (
    <div className="browser-window">
      <p>Query: {query}</p>
      <a href={searchUrl} target="_blank" rel="noreferrer">
        Open search results
      </a>
    </div>
  );
};

const DocumentViewerComponent: React.FC<{ file: File }> = ({ file }) => {
  const [src, setSrc] = useState<string>("");

  useEffect(() => {
    let active = true;
    fileToBase64(file)
      .then((value) => {
        if (active) {
          setSrc(value);
        }
      })
      .catch(() => {
        if (active) {
          setSrc("");
        }
      });
    return () => {
      active = false;
    };
  }, [file]);

  return (
    <div className="document-window">
      <p>{file.name}</p>
      {src ? (
        file.type.startsWith("image/") ? (
          <img src={src} alt={file.name} />
        ) : (
          <a href={src} download={file.name}>
            Download preview copy
          </a>
        )
      ) : (
        <p>Preview unavailable.</p>
      )}
    </div>
  );
};

type CommandBarProps = {
  onCommand: (cmd: string, arg: string) => void;
  onFileUpload: (file: File) => void;
  isListening: boolean;
  onToggleListening: () => void;
};

const CommandBar: React.FC<CommandBarProps> = ({ onCommand, onFileUpload, isListening, onToggleListening }) => {
  const [command, setCommand] = useState("");

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = command.trim();
    if (!trimmed) {
      return;
    }
    const [cmd, ...rest] = trimmed.split(" ");
    onCommand(cmd.replace(/^\//, "").toUpperCase(), rest.join(" "));
    setCommand("");
  };

  return (
    <footer className="command-bar">
      <form onSubmit={submit}>
        <input
          value={command}
          onChange={(event) => setCommand(event.target.value)}
          placeholder="/UNIFIED, /FIELD, /TRAINING, /CHAT, /IMAGE <prompt>, /VIDEO <prompt>, /BROWSER <query>"
        />
        <button type="submit">Run</button>
      </form>
      <label className="upload-btn">
        Upload
        <input
          type="file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onFileUpload(file);
            }
            event.target.value = "";
          }}
        />
      </label>
      <button type="button" onClick={onToggleListening}>
        {isListening ? "Listening..." : "Voice"}
      </button>
    </footer>
  );
};

const App: React.FC = () => {
  const [windows, setWindows] = useState<WindowInstance[]>([]);
  const [collapsedPanels, setCollapsedPanels] = useState<Record<number, boolean>>({});
  const [expandedPanelId, setExpandedPanelId] = useState<number | null>(null);
  const [screen] = useState<Screen>(Screen.COMMAND_CONSOLE);
  const [tasks] = useState(INITIAL_TASKS);
  const [feed, setFeed] = useState<FeedItem[]>([
    { id: 1, type: "info", text: "Aetherium interface online." },
  ]);
  const [chartData] = useState(generateChartData());
  const [chatMessagesByMode, setChatMessagesByMode] = useState<Record<ChatMode, ChatMessage[]>>({
    GENERAL: [{ id: 1, sender: "aura", text: "General channel online. Ask anything." }],
    PHYSICS: [{ id: 2, sender: "aura", text: "Physics channel online. Request field metrics or state diagnostics." }],
    COMPONENT_REPORT: [{ id: 3, sender: "aura", text: "Component report channel online. Ask for subsystem health." }],
  });
  const [loadingByMode, setLoadingByMode] = useState<Record<ChatMode, boolean>>({
    GENERAL: false,
    PHYSICS: false,
    COMPONENT_REPORT: false,
  });
  const [hotMemoryHistory, setHotMemoryHistory] = useState<MemoryHistoryEntry[]>([]);
  const [coldMemoryHistory, setColdMemoryHistory] = useState<MemoryHistoryEntry[]>([]);

  const nextId = useRef(20);
  const zIndexCounter = useRef(10);
  const chatWindowModesRef = useRef<Record<number, ChatMode>>({});
  const initializedWindowsRef = useRef(false);
  const speak = useSpeech();

  const pushFeed = useCallback((type: FeedItemType, text: string) => {
    setFeed((prev) => [{ id: Date.now(), type, text }, ...prev].slice(0, 14));
  }, []);

  const addWindow = useCallback((type: WindowType, options: Partial<WindowInstance> = {}) => {
    const id = nextId.current;
    nextId.current += 1;
    zIndexCounter.current += 1;

    const base = {
      id,
      type,
      zIndex: zIndexCounter.current,
      position: options.position ?? { x: 80 + ((id * 28) % 220), y: 80 + ((id * 22) % 140) },
      size: options.size ?? { width: 520, height: 420 },
      title: options.title ?? type.replace(/_/g, " "),
    };

    let win: WindowInstance;
    if (type === "CHAT") {
      win = { ...base, type: "CHAT", title: options.title ?? "Aura Chat" };
      const chatMode = ((options as { chatMode?: ChatMode }).chatMode || "GENERAL") as ChatMode;
      chatWindowModesRef.current[id] = chatMode;
    } else if (type === "UNIFIED_CHAT") {
      win = { ...base, type: "UNIFIED_CHAT", title: options.title ?? "Unified Consciousness" };
    } else if (type === "FIELD_PANEL") {
      win = { ...base, type: "FIELD_PANEL", title: options.title ?? "Field + Simulation Archives" };
    } else if (type === "TRAINING_DASHBOARD") {
      win = { ...base, type: "TRAINING_DASHBOARD", title: options.title ?? "Sovereign Training Dashboard" };
    } else if (type === "IMAGE_GENERATOR") {
      win = {
        ...base,
        type: "IMAGE_GENERATOR",
        title: options.title ?? "Image Generator",
        initialPrompt: (options as { initialPrompt?: string }).initialPrompt,
      };
    } else if (type === "VIDEO_GENERATOR") {
      win = {
        ...base,
        type: "VIDEO_GENERATOR",
        title: options.title ?? "Video Generator",
        initialPrompt: (options as { initialPrompt?: string }).initialPrompt,
      };
    } else if (type === "BROWSER") {
      win = {
        ...base,
        type: "BROWSER",
        title: options.title ?? "Browser",
        initialQuery: (options as { initialQuery?: string }).initialQuery || "Aetherium systems",
      };
    } else {
      const providedFile = (options as { file?: File }).file;
      const fallbackFile = new File([""], "untitled.txt", { type: "text/plain" });
      win = {
        ...base,
        type: "DOCUMENT_VIEWER",
        title: options.title ?? "Document Viewer",
        file: providedFile || fallbackFile,
      };
    }

    setWindows((prev) => [...prev, win]);
    setCollapsedPanels((prev) => ({ ...prev, [id]: false }));
  }, []);

  const addHotMemoryEntry = useCallback((text: string) => {
    setHotMemoryHistory((prev) => [{ id: Date.now(), text }, ...prev].slice(0, 18));
  }, []);

  const addColdMemoryEntry = useCallback((text: string) => {
    setColdMemoryHistory((prev) => [{ id: Date.now(), text }, ...prev].slice(0, 18));
  }, []);

  const getClientGeminiKey = useCallback((): string => {
    try {
      const fromStorage = window.localStorage.getItem("gemini_api_key") || "";
      if (fromStorage.trim()) {
        return fromStorage.trim();
      }
    } catch {
      // ignore localStorage access issues
    }
    const fromEnv = (import.meta.env.VITE_GEMINI_API_KEY || "").trim();
    return fromEnv;
  }, []);

  const appendChat = useCallback((mode: ChatMode, message: ChatMessage) => {
    setChatMessagesByMode((prev) => ({
      ...prev,
      [mode]: [...prev[mode], message],
    }));
  }, []);

  const fetchStatus = useCallback(async (): Promise<AtlanteanStatus | null> => {
    try {
      const res = await fetch("/api/atlantean/status");
      if (!res.ok) {
        return null;
      }
      const data = (await res.json()) as AtlanteanStatus;
      return data;
    } catch {
      return null;
    }
  }, []);

  const formatPhysicsResponse = (status: AtlanteanStatus, userText: string) => {
    const phi1 = status.field_state?.phi1_mean ?? 0;
    const phi5 = status.field_state?.phi5_mean ?? 0;
    const Phi = status.field_state?.Phi ?? 0;
    const learning = status.learning_capacity ?? 0;
    return [
      `Physics metrics for: "${userText}"`,
      `phi1_mean: ${phi1.toFixed(4)}`,
      `phi5_mean: ${phi5.toFixed(4)}`,
      `Phi: ${Phi.toFixed(4)}`,
      `learning_capacity: ${(learning * 100).toFixed(2)}%`,
      `delta(phi5-phi1): ${(phi5 - phi1).toFixed(4)}`,
    ].join("\n");
  };

  const formatComponentResponse = (status: AtlanteanStatus) => {
    const stats = status.sovereign?.stats || {};
    return [
      "System Component Report",
      `core_brain: ${status.core_brain || "unknown"}`,
      `llm_mediator.gemini_configured: ${String(status.llm_mediator?.gemini_configured ?? false)}`,
      `llm_mediator.gemini_model: ${status.llm_mediator?.gemini_model || "n/a"}`,
      `sovereign.local_only: ${String(status.sovereign?.local_only ?? true)}`,
      `queries_total: ${stats.queries_total ?? 0}`,
      `local_calls: ${stats.local_calls ?? 0}`,
      `gemini_calls: ${stats.gemini_calls ?? 0}`,
      `gemini_failures: ${stats.gemini_failures ?? 0}`,
      `fallback_calls: ${stats.fallback_calls ?? 0}`,
    ].join("\n");
  };

  const handleSendMessage = useCallback(
    async (mode: ChatMode, text: string, verbosity: ResponseVerbosity = "normal") => {
      if (loadingByMode[mode]) {
        return;
      }
      const userMessage: ChatMessage = { id: Date.now(), sender: "user", text };
      appendChat(mode, userMessage);
      setLoadingByMode((prev) => ({ ...prev, [mode]: true }));

      try {
        if (mode === "PHYSICS") {
          const status = await fetchStatus();
          if (!status) {
            throw new Error("Status endpoint unavailable");
          }
          const auraText = formatPhysicsResponse(status, text);
          appendChat(mode, { id: Date.now() + 1, sender: "aura", text: auraText });
          addHotMemoryEntry(
            `phi1=${(status.field_state?.phi1_mean ?? 0).toFixed(4)}, phi5=${(status.field_state?.phi5_mean ?? 0).toFixed(4)}, Phi=${(status.field_state?.Phi ?? 0).toFixed(4)}`,
          );
          pushFeed("summary", "Physics channel returned a fresh field-state metric snapshot.");
        } else if (mode === "COMPONENT_REPORT") {
          const status = await fetchStatus();
          if (!status) {
            throw new Error("Status endpoint unavailable");
          }
          const auraText = formatComponentResponse(status);
          appendChat(mode, { id: Date.now() + 1, sender: "aura", text: auraText });
          addColdMemoryEntry(`Component report generated for request: ${text}`);
          pushFeed("summary", "Component report generated from system status.");
        } else {
          const geminiApiKey = getClientGeminiKey();
          const runQuery = async (model?: string) =>
            fetch("/api/atlantean/query", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                input: text,
                verbosity,
                llm_provider: "gemini",
                api_key: geminiApiKey || undefined,
                model,
              }),
            });

          let res = await runQuery();
          let data = (await res.json()) as AtlanteanQueryResponse;

          const needsRetry =
            Boolean(data.metadata?.sovereign?.fallback_used) ||
            /best next move:|say 'deeper'|short answer:/i.test(data.response || "");
          if (needsRetry) {
            res = await runQuery("gemini-2.5-flash");
            data = (await res.json()) as AtlanteanQueryResponse;
          }

          const auraText = data.response || data.error || "No response returned.";
          appendChat(mode, { id: Date.now() + 1, sender: "aura", text: auraText });

          const phi1 = data.field_state?.phi1_mean ?? 0;
          const phi5 = data.field_state?.phi5_mean ?? 0;
          const Phi = data.field_state?.Phi ?? 0;
          addHotMemoryEntry(`chat update -> phi1=${phi1.toFixed(4)}, phi5=${phi5.toFixed(4)}, Phi=${Phi.toFixed(4)}`);
          addColdMemoryEntry(`user="${text.slice(0, 72)}" | response="${auraText.slice(0, 72)}"`);

          if (data.metadata?.llm_mediator?.gemini_used) {
            pushFeed("autonomous", "General chat response mediated through Gemini.");
          } else {
            const mediatorError = data.metadata?.llm_mediator?.error || "";
            if (!geminiApiKey && /not configured|api key|invalid/i.test(mediatorError)) {
              pushFeed("error", "Gemini not available to frontend key path; backend may use local mode.");
            }
            pushFeed("summary", "General chat response completed via local cognitive stack.");
          }
        }

        if (/image/i.test(text)) {
          addWindow("IMAGE_GENERATOR", { initialPrompt: text });
        }
        if (/video/i.test(text)) {
          addWindow("VIDEO_GENERATOR", { initialPrompt: text });
        }
      } catch {
        const fallback = "I encountered a transport error while contacting the cognition backend.";
        appendChat(mode, { id: Date.now() + 1, sender: "aura", text: fallback });
        pushFeed("error", `Backend request failed from ${mode} window.`);
      } finally {
        setLoadingByMode((prev) => ({ ...prev, [mode]: false }));
      }
    },
    [loadingByMode, appendChat, addWindow, addColdMemoryEntry, addHotMemoryEntry, fetchStatus, getClientGeminiKey, pushFeed],
  );

  const closeWindow = (id: number) => {
    if (chatWindowModesRef.current[id]) {
      delete chatWindowModesRef.current[id];
    }
    setCollapsedPanels((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setExpandedPanelId((prev) => (prev === id ? null : prev));
    setWindows((prev) => prev.filter((win) => win.id !== id));
  };

  const togglePanelCollapse = (id: number) => {
    setCollapsedPanels((prev) => {
      const willCollapse = !prev[id];
      if (willCollapse) {
        setExpandedPanelId((current) => (current === id ? null : current));
      }
      return {
        ...prev,
        [id]: willCollapse,
      };
    });
  };

  const togglePanelExpand = (id: number) => {
    setCollapsedPanels((prev) => ({ ...prev, [id]: false }));
    setExpandedPanelId((prev) => (prev === id ? null : id));
  };

  const handleCommand = (cmd: string, arg: string) => {
    switch (cmd) {
      case "CHAT":
        addWindow("CHAT", { title: "General Chat", chatMode: "GENERAL" } as Partial<WindowInstance>);
        pushFeed("info", "Opened Chat window.");
        break;
      case "UNIFIED":
      case "CONSCIOUSNESS":
        addWindow("UNIFIED_CHAT", { title: "Unified Consciousness" } as Partial<WindowInstance>);
        pushFeed("info", "Opened Unified Consciousness window.");
        break;
      case "PHYSICS":
        addWindow("CHAT", { title: "Internal Physics Metrics", chatMode: "PHYSICS" } as Partial<WindowInstance>);
        pushFeed("info", "Opened Physics Metrics window.");
        break;
      case "FIELD":
      case "ARCHIVES":
        addWindow("FIELD_PANEL", { title: "Field + Simulation Archives" } as Partial<WindowInstance>);
        pushFeed("info", "Opened Field Visualization panel.");
        break;
      case "TRAIN":
      case "TRAINING":
        addWindow("TRAINING_DASHBOARD", { title: "Sovereign Training Dashboard" } as Partial<WindowInstance>);
        pushFeed("info", "Opened Training Dashboard.");
        break;
      case "REPORT":
      case "COMPONENT":
        addWindow("CHAT", { title: "Component Functionality Report", chatMode: "COMPONENT_REPORT" } as Partial<WindowInstance>);
        pushFeed("info", "Opened Component Report window.");
        break;
      case "IMAGE":
      case "IMAGE_GENERATOR":
        addWindow("IMAGE_GENERATOR", { initialPrompt: arg });
        pushFeed("autonomous", "Queued image generator task.");
        break;
      case "VIDEO":
      case "VIDEO_GENERATOR":
        addWindow("VIDEO_GENERATOR", { initialPrompt: arg });
        pushFeed("autonomous", "Queued video generator task.");
        break;
      case "BROWSER":
      case "SEARCH":
        addWindow("BROWSER", { initialQuery: arg || "syntropy governor" });
        pushFeed("info", `Opened browser for query: ${arg || "syntropy governor"}`);
        break;
      default:
        pushFeed("error", `Unknown command: ${cmd}`);
    }
  };

  const handleFileUpload = (file: File) => {
    addWindow("DOCUMENT_VIEWER", { file, title: file.name, size: { width: 560, height: 460 } });
    pushFeed("info", `Loaded document: ${file.name}`);
  };

  const handleVoiceResult = useCallback(
    (transcript: string) => {
      pushFeed("autonomous", `Voice command: ${transcript}`);
      void handleSendMessage("GENERAL", transcript, "normal");
    },
    [handleSendMessage, pushFeed],
  );

  const { isListening, toggleListening } = useVoiceRecognition({ onResult: handleVoiceResult });

  useEffect(() => {
    if (initializedWindowsRef.current) {
      return;
    }
    initializedWindowsRef.current = true;

    addWindow("UNIFIED_CHAT", {
      position: { x: 268, y: 54 },
      size: { width: 640, height: 620 },
      title: "Unified Consciousness",
    } as Partial<WindowInstance>);

    addWindow("FIELD_PANEL", {
      position: { x: 924, y: 54 },
      size: { width: 520, height: 420 },
      title: "Field + Simulation Archives",
    } as Partial<WindowInstance>);

    addWindow("TRAINING_DASHBOARD", {
      position: { x: 924, y: 484 },
      size: { width: 520, height: 360 },
      title: "Sovereign Training Dashboard",
    } as Partial<WindowInstance>);

    addWindow("CHAT", {
      position: { x: Math.max(260, window.innerWidth - 980), y: 54 },
      size: { width: 430, height: 620 },
      title: "General Chat",
      chatMode: "GENERAL",
    } as Partial<WindowInstance>);

    addWindow("CHAT", {
      position: { x: Math.max(260, window.innerWidth - 540), y: 54 },
      size: { width: 420, height: 300 },
      title: "Internal Physics Metrics",
      chatMode: "PHYSICS",
    } as Partial<WindowInstance>);

    addWindow("CHAT", {
      position: { x: Math.max(260, window.innerWidth - 540), y: 370 },
      size: { width: 420, height: 304 },
      title: "Component Functionality Report",
      chatMode: "COMPONENT_REPORT",
    } as Partial<WindowInstance>);
  }, [addWindow]);

  const renderWindowContent = (win: WindowInstance) => {
    switch (win.type) {
      case "CHAT": {
        const mode = chatWindowModesRef.current[win.id] || "GENERAL";
        if (mode === "PHYSICS") {
          return <PhysicsMetricsWindow fetchStatus={fetchStatus} onHotMemory={addHotMemoryEntry} />;
        }
        return (
          <ChatWindowComponent
            mode={mode}
            speak={speak}
            messages={chatMessagesByMode[mode]}
            isLoading={loadingByMode[mode]}
            onSend={(text, verbosity) => {
              void handleSendMessage(mode, text, verbosity);
            }}
          />
        );
      }
      case "UNIFIED_CHAT":
        return <ChatInterface />;
      case "FIELD_PANEL":
        return <NeuralArchives />;
      case "TRAINING_DASHBOARD":
        return <TrainingDashboard />;
      case "IMAGE_GENERATOR":
        return <ImageGeneratorWindowComponent initialPrompt={win.initialPrompt} />;
      case "VIDEO_GENERATOR":
        return <VideoGeneratorWindowComponent initialPrompt={win.initialPrompt} />;
      case "BROWSER":
        return <BrowserWindowComponent query={win.initialQuery} />;
      case "DOCUMENT_VIEWER":
        return <DocumentViewerComponent file={win.file} />;
      default:
        return null;
    }
  };

  return (
    <div className="aetherium-app">
      <div className="aetherium-bg" />
      <header className="top-status-bar">
        <div className="brand-wrap">
          <AuraAvatar />
          <div>
            <h1>Aetherium Interface</h1>
            <p>{screen}</p>
          </div>
        </div>
        <div className="kpi-wrap">
          <span>Tasks: {tasks.length}</span>
          <span>Signals: {chartData.length}</span>
          <span>Feed: {feed.length}</span>
        </div>
      </header>

      <aside className="left-rail">
        <h2>Hot/Cold Memory History</h2>
        <div className="memory-history-section">
          <h3>Hot Memory</h3>
          <ul>
            {hotMemoryHistory.map((entry) => (
              <li key={entry.id}>{entry.text}</li>
            ))}
          </ul>
        </div>
        <div className="memory-history-section">
          <h3>Cold Memory</h3>
          <ul>
            {coldMemoryHistory.map((entry) => (
              <li key={entry.id}>{entry.text}</li>
            ))}
          </ul>
        </div>
      </aside>

      <aside className="right-rail">
        <h2>Cognitive Feed</h2>
        <ul>
          {feed.map((item) => (
            <li key={item.id} className={`feed-${item.type}`}>
              {item.text}
            </li>
          ))}
        </ul>
      </aside>

      <main className="desktop-layer">
        {windows.map((win) => (
          <TiledPanel
            key={win.id}
            win={win}
            onClose={closeWindow}
            onToggleCollapse={togglePanelCollapse}
            onToggleExpand={togglePanelExpand}
            isCollapsed={Boolean(collapsedPanels[win.id])}
            isExpanded={expandedPanelId === win.id}
          >
            {renderWindowContent(win)}
          </TiledPanel>
        ))}
      </main>

      <CommandBar onCommand={handleCommand} onFileUpload={handleFileUpload} isListening={isListening} onToggleListening={toggleListening} />
    </div>
  );
};

export default App;