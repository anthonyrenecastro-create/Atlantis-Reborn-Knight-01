#!/usr/bin/env python3
"""
Atlantean + Syntropy Bridge
===========================
This is the core integration layer that makes the three systems one.

- Loads the Syntropy Core Brain (AdvancedTextGenerationNN)
- Injects Atlantean field state (phi1, phi5, Phi) into every generation
- Applies learning signals back to both Atlantean hot memory and Syntropy fields
- Provides a clean /query endpoint for the Consciousness UI
- Fully local, no external APIs

Usage:
    from atlantean_syntropy_bridge import AtlanteanSyntropyBridge
    bridge = AtlanteanSyntropyBridge()
    response = bridge.query("What is the nature of intelligence?")
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
import json
import re
from collections import deque

try:
    from google import genai
except Exception:
    genai = None

# Add paths for the three systems
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "core_brain"))
sys.path.insert(0, str(BASE_DIR / "governor"))
sys.path.insert(0, str(BASE_DIR / "consciousness"))
sys.path.insert(0, str(BASE_DIR / "consciousness/atlantean_core"))

from syntropy_field_expanded import AdvancedTextGenerationNN, OscillatorySynapseTheory
from hot_memory import AtlanteanHotMemory
from learning import apply_learning_signal, apply_contradiction_signal, compute_learning_capacity

try:
    from atlantean_quadra_bridge import AtlanteanQuadraBridge
except Exception:
    AtlanteanQuadraBridge = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyntropyGovernorBridge")

class AtlanteanSyntropyBridge:
    """
    The living bridge between:
    - Atlantean Hot/Cold Memory (persistence + learning)
    - Syntropy Core Brain (field-modulated text generation)
    - Governor (validation + monitoring)
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Syntropy Governor Bridge on {self.device}")

        # === 1. Load Syntropy Core Brain ===
        self.model = None
        self.stoi = None
        self.itos = None
        self.vocab_size = 0
        self.field_state_dim = 256
        self.active_model_path: Optional[str] = None

        model_path = model_path or str(BASE_DIR / "core_brain/shakespeare_model.pt")
        self._load_syntropy_model(model_path)

        # === 2. Initialize Atlantean Core Memory (persistent hot memory) ===
        self.core_state_path = Path(
            os.getenv("ATLANTEAN_CORE_STATE_PATH", str(BASE_DIR / "unified_backend" / "core_state.pt"))
        )
        self.hot_memory = self._load_or_initialize_hot_memory()

        self.simulation_history: List[Dict[str, Any]] = []
        self.max_simulation_history = 500
        self.interaction_index: Dict[str, Dict[str, Any]] = {}
        self.max_interactions = 2000

        # === 2b. Sovereign local operation stats ===
        self.sovereign_stats = {
            "queries_total": 0,
            "local_calls": 0,
            "fallback_calls": 0,
            "gemini_calls": 0,
            "gemini_failures": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "correction_feedback": 0,
            "learning_events": 0,
        }
        self.cold_memory_log_path = Path(
            os.getenv(
                "ATLANTEAN_COLD_MEMORY_LOG_PATH",
                str(BASE_DIR / "unified_backend" / "cold_memory_interactions.jsonl"),
            )
        )

        self.pipeline_architecture = [
            "core_brain:shakespeare_model",
            "governor:cognitive_wrapper",
            "atlantean_bridge:field_funnel",
            "quadra_seer:final_output_integration",
        ]
        self.require_quadra_final = os.getenv("SYNTROPY_REQUIRE_QUADRA_FINAL", "true").strip().lower() != "false"
        self.quadra_local_log_path = Path(
            os.getenv(
                "ATLANTEAN_QUADRA_LOCAL_LOG_PATH",
                str(BASE_DIR / "unified_backend" / "quadra_final_output.jsonl"),
            )
        )
        self.quadra_bridge = self._init_quadra_seer_bridge()

        # Optional Gemini mediator: Atlantean bridge remains the only caller,
        # so API outputs are always grounded in field-state + local cognition.
        self.gemini_api_key = self._resolve_gemini_api_key()
        self.gemini_model = (
            os.getenv("GEMINI_MODEL")
            or os.getenv("ATLANTEAN_GEMINI_MODEL")
            or "gemini-2.5-flash"
        ).strip()
        self.gemini_client = None
        self._init_gemini_client()

        # === 3. Field State Encoder (maps Atlantean → Syntropy field_state) ===
        self.field_state_dim = self._infer_field_state_dim()
        self.field_encoder = nn.Linear(3, self.field_state_dim).to(self.device)

        logger.info("✅ Atlantean + Syntropy Bridge initialized successfully")

    def _resolve_gemini_api_key(self) -> str:
        direct = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("VITE_GEMINI_API_KEY")
            or ""
        ).strip()
        if direct:
            return direct

        # Compatibility fallback: reuse key from setup doc if user has not
        # created .env files yet.
        setup_doc = BASE_DIR / "consciousness" / "SETUP_API_KEY.md"
        if setup_doc.exists():
            try:
                content = setup_doc.read_text(encoding="utf-8")
                m = re.search(r"^GEMINI_API_KEY\s*=\s*([^\s#]+)", content, flags=re.MULTILINE)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
        return ""

    def _init_gemini_client(self):
        if not self.gemini_api_key:
            logger.info("Gemini mediator disabled: no GEMINI_API_KEY configured.")
            self.gemini_client = None
            return
        if genai is None:
            logger.warning("Gemini mediator disabled: google-genai package not available.")
            self.gemini_client = None
            return
        try:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            logger.info(f"✅ Gemini mediator configured (model={self.gemini_model})")
        except Exception as exc:
            logger.warning(f"Gemini mediator init failed; continuing local-only. Reason: {exc}")
            self.gemini_client = None

    def _gemini_ready(self) -> bool:
        return self.gemini_client is not None

    def _build_gemini_mediator_prompt(
        self,
        user_input: str,
        local_response: str,
        decision_output: Dict[str, Any],
        field_snapshot: Dict[str, float],
    ) -> str:
        compact_decision = {
            "intent": decision_output.get("intent"),
            "next_action": decision_output.get("next_action"),
            "selected_option": decision_output.get("selected_option"),
            "state_estimate": decision_output.get("state_estimate"),
            "expected_signal": decision_output.get("expected_signal"),
            "guardrails": decision_output.get("guardrails"),
            "influences": decision_output.get("influences"),
        }
        return (
            "You are Gemini acting as a synthesis layer for Atlantean-Syntropy cognition.\n"
            "Non-negotiable constraints:\n"
            "1) Keep response directly useful, plain, and non-generic.\n"
            "2) Preserve intent from the supplied decision object.\n"
            "3) Do not expose internal chain-of-thought.\n"
            "4) Prefer concrete next-step guidance when applicable.\n\n"
            f"User input:\n{user_input}\n\n"
            f"Atlantean field snapshot:\n{json.dumps(field_snapshot, ensure_ascii=True)}\n\n"
            f"Decision object (authoritative):\n{json.dumps(compact_decision, ensure_ascii=True)}\n\n"
            f"Local draft from core brain:\n{local_response}\n\n"
            "Produce a final response for the user."
        )

    def _call_gemini_mediator(
        self,
        user_input: str,
        local_response: str,
        decision_output: Dict[str, Any],
        field_snapshot: Dict[str, float],
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if genai is None:
            return None, "google-genai package unavailable", None

        client = self.gemini_client
        if api_key_override and api_key_override.strip():
            try:
                client = genai.Client(api_key=api_key_override.strip())
            except Exception as exc:
                return None, f"invalid override API key: {exc}", None

        if client is None:
            return None, "gemini not configured", None

        model_name = (model_override or self.gemini_model or "gemini-2.5-flash").strip()
        prompt = self._build_gemini_mediator_prompt(
            user_input=user_input,
            local_response=local_response,
            decision_output=decision_output,
            field_snapshot=field_snapshot,
        )
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return None, "empty response from gemini", model_name
            return text, None, model_name
        except Exception as exc:
            return None, str(exc), model_name

    def _init_quadra_seer_bridge(self):
        if AtlanteanQuadraBridge is None:
            if self.require_quadra_final:
                raise RuntimeError("Quadra-Seer bridge import failed. Install required dependencies to enable final integration.")
            return None
        try:
            return AtlanteanQuadraBridge(enable_crypto=False, device_id="syntropy-governor")
        except Exception as exc:
            if self.require_quadra_final:
                raise RuntimeError(f"Quadra-Seer bridge initialization failed: {exc}")
            logger.warning(f"Quadra-Seer integration unavailable; continuing without it. Reason: {exc}")
            return None

    def _quadra_local_finalize(self, user_input: str, response_text: str, interaction_id: str, error: str = "") -> Dict[str, Any]:
        self.quadra_local_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "interaction_id": interaction_id,
            "timestamp": time.time(),
            "source": "unified_backend",
            "user_input": user_input,
            "response": response_text,
            "error": error,
            "finalizer": "quadra_local_ledger",
        }
        with self.quadra_local_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")

        return {
            "enabled": True,
            "status": "applied_local",
            "session_id": interaction_id,
            "local_log_path": str(self.quadra_local_log_path),
        }

    def _governor_cognitive_wrapper(self, user_input: str) -> tuple[str, Dict[str, Any]]:
        raw = (user_input or "").strip()
        normalized = re.sub(r"\s+", " ", raw)
        intent = self._infer_intent(normalized)

        wrapped = (
            f"[governor-wrapper intent={intent} policy=answer-first no-internal-narration] "
            f"{normalized}"
        ).strip()

        return wrapped, {
            "enabled": True,
            "intent": intent,
            "answer_first": True,
            "internal_process_hidden": True,
            "input_chars": len(normalized),
        }

    def _quadra_finalize_output(
        self,
        user_input: str,
        response_text: str,
        interaction_id: str,
    ) -> Dict[str, Any]:
        if self.quadra_bridge is None:
            if self.require_quadra_final:
                return self._quadra_local_finalize(
                    user_input=user_input,
                    response_text=response_text,
                    interaction_id=interaction_id,
                    error="quadra_bridge_unavailable",
                )
            return {
                "enabled": False,
                "status": "unavailable",
            }

        try:
            session_id = interaction_id
            self.quadra_bridge.store_chat_turn(
                role="user",
                content=user_input,
                session_id=session_id,
                domain="unified_backend",
                relevance=0.65,
            )
            self.quadra_bridge.store_chat_turn(
                role="assistant",
                content=response_text,
                session_id=session_id,
                domain="unified_backend",
                relevance=0.62,
            )
            return {
                "enabled": True,
                "status": "applied",
                "session_id": session_id,
            }
        except Exception as exc:
            logger.warning(f"Quadra-Seer finalization failed; switching to local finalization ledger. Reason: {exc}")
            return self._quadra_local_finalize(
                user_input=user_input,
                response_text=response_text,
                interaction_id=interaction_id,
                error=str(exc),
            )

    def _load_or_initialize_hot_memory(self) -> AtlanteanHotMemory:
        """Load persistent Atlantean hot memory if present, otherwise initialize fresh state."""
        try:
            if self.core_state_path.exists():
                hot = AtlanteanHotMemory.load(str(self.core_state_path))
                logger.info(f"✅ Loaded Atlantean hot memory from {self.core_state_path}")
                return hot
        except Exception as exc:
            logger.warning(f"Could not load existing Atlantean hot memory; reinitializing. Reason: {exc}")

        hot = AtlanteanHotMemory.initialize(grid_size=(32, 32))
        self._persist_hot_memory(hot)
        logger.info(f"✅ Initialized new Atlantean hot memory at {self.core_state_path}")
        return hot

    def _persist_hot_memory(self, hot_memory: Optional[AtlanteanHotMemory] = None):
        """Persist Atlantean hot memory state to disk."""
        target = hot_memory or self.hot_memory
        self.core_state_path.parent.mkdir(parents=True, exist_ok=True)
        target.save(str(self.core_state_path))

    def _field_snapshot(self) -> Dict[str, float]:
        """Summarize current Atlantean fields for API responses."""
        return {
            "phi1_mean": float(self.hot_memory.phi1.mean().item()),
            "phi5_mean": float(self.hot_memory.phi5.mean().item()),
            "Phi": float(self.hot_memory.Phi.mean().item()),
            "learning_capacity": float(compute_learning_capacity(self.hot_memory)),
        }

    def _infer_field_state_dim(self) -> int:
        """Infer field_state width expected by the loaded model's System-2 layer."""
        default_dim = 256
        try:
            hidden_size = int(getattr(self.model, "hidden_size", 0))
            reasoner = getattr(self.model, "system2_reasoner", None)
            if reasoner is None or hidden_size <= 0:
                return default_dim

            first_layer = reasoner[0] if len(reasoner) > 0 else None
            if not isinstance(first_layer, nn.Linear):
                return default_dim

            inferred = int(first_layer.in_features) - hidden_size
            return inferred if inferred > 0 else default_dim
        except Exception as exc:
            logger.warning(f"Could not infer field dimension from model; using {default_dim}. Reason: {exc}")
            return default_dim

    def _load_syntropy_model(self, path: str):
        """Load the pretrained Syntropy model"""
        if not os.path.exists(path):
            logger.warning(f"Model not found at {path}. Using random init (demo mode).")
            self.vocab_size = 66
            self.model = AdvancedTextGenerationNN(
                vocab_size=self.vocab_size,
                embedding_dim=128,
                hidden_size=128
            ).to(self.device)
            self.stoi = {chr(i): i for i in range(32, 127)}
            self.itos = {i: chr(i) for i in range(32, 127)}
            self.active_model_path = None
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.vocab_size = checkpoint.get("vocab_size", 66)
        emb_dim = checkpoint.get("embedding_dim", 128)
        hidden = checkpoint.get("hidden_size", 128)
        self.stoi = checkpoint.get("stoi", {})
        self.itos = checkpoint.get("itos", {})

        self.model = AdvancedTextGenerationNN(
            vocab_size=self.vocab_size,
            embedding_dim=emb_dim,
            hidden_size=hidden
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.active_model_path = str(Path(path).resolve())
        logger.info(f"✅ Loaded Syntropy model (vocab={self.vocab_size}, emb={emb_dim})")

    def reload_model(self, model_path: str) -> Dict[str, Any]:
        """Hot-swap the active model checkpoint without restarting the server."""
        resolved = str(Path(model_path).resolve())
        try:
            self._load_syntropy_model(resolved)
            self.field_state_dim = self._infer_field_state_dim()
            self.field_encoder = nn.Linear(3, self.field_state_dim).to(self.device)
            return {
                "status": "reloaded",
                "active_model_path": self.active_model_path,
                "field_state_dim": self.field_state_dim,
            }
        except Exception as exc:
            logger.error(f"Model reload failed for {resolved}: {exc}")
            return {
                "status": "error",
                "error": str(exc),
                "active_model_path": self.active_model_path,
            }

    def _encode_field_state(self) -> torch.Tensor:
        """Convert current Atlantean fields into Syntropy field_state vector"""
        snapshot = self._field_snapshot()
        phi1_mean = snapshot["phi1_mean"]
        phi5_mean = snapshot["phi5_mean"]
        Phi = snapshot["Phi"]

        field_vec = torch.tensor([[phi1_mean, phi5_mean, Phi]], dtype=torch.float32, device=self.device)
        field_state = self.field_encoder(field_vec)
        return field_state.squeeze(0)  # (256,)

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _local_quality_score(self, text: str, generation_error: Optional[str]) -> float:
        if generation_error:
            return 0.0
        stripped = (text or "").strip()
        if len(stripped) < 20:
            return 0.15
        diversity = len(set(stripped)) / max(1, len(stripped))
        repetition_penalty = 0.0
        if stripped:
            repetition_penalty = 0.4 if stripped.count(stripped[0]) > len(stripped) * 0.4 else 0.0
        base = min(1.0, len(stripped) / 160.0)
        return self._clamp_01((0.55 * base) + (0.55 * diversity) - repetition_penalty)

    def _looks_like_gibberish(self, text: str) -> bool:
        stripped = (text or "").strip()
        if len(stripped) < 20:
            return False

        letters = [ch for ch in stripped if ch.isalpha()]
        spaces = stripped.count(" ")
        alpha_ratio = len(letters) / max(1, len(stripped))
        space_ratio = spaces / max(1, len(stripped))
        words = [w for w in stripped.split() if w]
        avg_word_len = (sum(len(w) for w in words) / len(words)) if words else len(stripped)

        # Hard gibberish: character streams with almost no spacing and extreme pseudo-word lengths.
        if space_ratio < 0.04 and avg_word_len > 14:
            return True

        # Hard gibberish: long alphabetic runs with almost no punctuation and spacing.
        punctuation_ratio = sum(1 for ch in stripped if ch in ".,;:!?-()") / max(1, len(stripped))
        if alpha_ratio > 0.92 and punctuation_ratio < 0.002 and space_ratio < 0.07:
            return True

        # Hard gibberish: excessive repeated characters or extremely low symbol diversity in long output.
        if re.search(r"(.)\1{7,}", stripped):
            return True
        diversity = len(set(stripped)) / max(1, len(stripped))
        if len(stripped) > 60 and diversity < 0.08:
            return True

        return False

    @staticmethod
    def _clear_intent_threshold() -> float:
        raw = os.getenv("SYNTROPY_CLEAR_INTENT_MIN", "0.62").strip()
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            return 0.62

    def _intent_confidence(self, user_input: str) -> float:
        lowered = (user_input or "").lower().strip()
        if not lowered:
            return 0.0

        intent = self._infer_intent(user_input)
        tokens = [tok for tok in re.split(r"\s+", lowered) if tok]
        score = 0.25

        if len(tokens) >= 3:
            score += 0.12
        if lowered.endswith("?") or lowered.startswith(("how", "why", "what", "when", "where", "can", "should", "please")):
            score += 0.16
        if intent != "decision_support":
            score += 0.28
        if any(tok in lowered for tok in ["field", "state", "awareness", "perceive", "feel", "answer", "direct"]):
            score += 0.15
        if len(lowered) > 18:
            score += 0.08

        return self._clamp_01(score)

    def _quality_min_threshold(self) -> float:
        raw = os.getenv("SYNTROPY_LOCAL_QUALITY_MIN", "0.52").strip()
        try:
            return self._clamp_01(float(raw))
        except ValueError:
            return 0.52

    def _should_use_fallback(
        self,
        text: str,
        generation_error: Optional[str],
        local_quality: float,
        user_input: str,
    ) -> tuple[bool, str]:
        if generation_error:
            return True, "generation_error"

        # Hybrid policy: hard gibberish always falls back.
        if self._looks_like_gibberish(text):
            return True, "hard_gibberish_pattern"

        # Low quality is tolerated when intent confidence is clear.
        if local_quality < self._quality_min_threshold():
            if self._intent_confidence(user_input) >= self._clear_intent_threshold():
                return False, "clear_intent_override"
            return True, "low_quality_unclear_intent"

        return False, "none"

    def _append_cold_memory(self, record: Dict[str, Any]):
        self.cold_memory_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cold_memory_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")

    def export_training_dataset(self, limit: int = 1000) -> Dict[str, Any]:
        safe_limit = max(1, min(5000, int(limit)))
        if not self.cold_memory_log_path.exists():
            return {
                "count": 0,
                "export_path": None,
                "items": [],
            }

        rows: List[Dict[str, Any]] = []
        with self.cold_memory_log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if item.get("type") == "feedback":
                    continue
                prompt = item.get("prompt")
                response = item.get("response")
                source = item.get("source", "local")
                if not isinstance(prompt, str) or not isinstance(response, str):
                    continue

                rows.append(
                    {
                        "instruction": prompt,
                        "output": response,
                        "source": source,
                        "mode": item.get("mode", "unknown"),
                        "interaction_id": item.get("interaction_id"),
                        "timestamp": item.get("timestamp"),
                    }
                )

        rows = rows[-safe_limit:]
        export_dir = self.cold_memory_log_path.parent / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"sovereign_training_dataset_{int(time.time())}.jsonl"
        with export_path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=True) + "\n")

        return {
            "count": len(rows),
            "export_path": str(export_path),
            "items": rows,
        }

    def _register_interaction(self, interaction_id: str, source: str, mode: str):
        self.interaction_index[interaction_id] = {
            "timestamp": time.time(),
            "source": source,
            "mode": mode,
        }
        if len(self.interaction_index) > self.max_interactions:
            # Drop oldest interactions to bound memory.
            oldest = sorted(self.interaction_index.items(), key=lambda row: row[1].get("timestamp", 0.0))[0][0]
            self.interaction_index.pop(oldest, None)

    @staticmethod
    def _tokenize_for_similarity(text: str) -> set[str]:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        return {tok for tok in cleaned.split() if len(tok) > 2}

    def _text_similarity(self, a: str, b: str) -> float:
        ta = self._tokenize_for_similarity(a)
        tb = self._tokenize_for_similarity(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta | tb))

    def _recent_response_similarity(self, text: str, limit: int = 12) -> float:
        if not self.simulation_history:
            return 0.0
        probe = (text or "").strip()
        if not probe:
            return 0.0
        recent = self.simulation_history[-limit:]
        return max(
            (
                self._text_similarity(probe, str(item.get("response", "")))
                for item in recent
            ),
            default=0.0,
        )

    def _wants_metrics(self, user_input: str) -> bool:
        lowered = (user_input or "").lower()
        return any(
            key in lowered
            for key in ["phi", "metric", "metrics", "values", "numbers", "exact"]
        )

    @staticmethod
    def _reasoning_pass_count() -> int:
        raw = os.getenv("SYNTROPY_REASONING_PASSES", "3").strip()
        try:
            return max(1, min(7, int(raw)))
        except ValueError:
            return 3

    @staticmethod
    def _field_modulation_depth() -> float:
        raw = os.getenv("SYNTROPY_REASONING_MOD_DEPTH", "0.18").strip()
        try:
            return max(0.0, min(0.8, float(raw)))
        except ValueError:
            return 0.18

    @staticmethod
    def _influence_strength(var_name: str, default: float) -> float:
        raw = os.getenv(var_name, str(default)).strip()
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            return default

    def _influence_weights(self) -> tuple[float, float]:
        field = self._influence_strength("SYNTROPY_FIELD_INFLUENCE", 0.64)
        memory = self._influence_strength("SYNTROPY_MEMORY_INTEGRATION", 0.62)

        # Keep both signals active; do not allow one axis to fully dominate.
        if abs(field - memory) > 0.4:
            transfer = (abs(field - memory) - 0.4) / 2.0
            if field > memory:
                field -= transfer
                memory += transfer
            else:
                memory -= transfer
                field += transfer

        total = max(0.001, field + memory)
        return (field / total, memory / total)

    def _derive_memory_profile(self, user_input: str, limit: int = 160) -> Dict[str, Any]:
        prompt_pool: List[str] = [
            str(item.get("prompt", "")).strip()
            for item in self.simulation_history[-limit:]
            if str(item.get("prompt", "")).strip()
        ]

        # Pull additional persistent context from cold memory so grounding survives restarts.
        if len(prompt_pool) < 12 and self.cold_memory_log_path.exists():
            try:
                with self.cold_memory_log_path.open("r", encoding="utf-8") as fp:
                    for line in deque(fp, maxlen=1500):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if item.get("type") == "feedback":
                            continue
                        prompt = str(item.get("prompt", "")).strip()
                        if prompt:
                            prompt_pool.append(prompt)
            except Exception:
                pass

        if not prompt_pool:
            return {
                "match_count": 0,
                "preferences": [],
                "recurring_topics": [],
                "grounding_line": "",
            }

        query = (user_input or "").strip().lower()
        recent_prompts = prompt_pool[-limit:]

        keyword_counts: Dict[str, int] = {}
        prompt_matches: List[str] = []
        pref_counters = {
            "structured": 0,
            "anti_generic": 0,
            "field_centric": 0,
            "iterative": 0,
        }

        stopwords = {
            "the", "and", "for", "with", "that", "this", "what", "when", "where", "from", "into", "your", "like",
            "have", "about", "just", "then", "than", "they", "them", "their", "there", "were", "will", "would",
            "could", "should", "while", "make", "more", "less", "very", "still", "need", "want", "been", "being",
            "does", "did", "done", "also", "some", "such", "each", "much", "many", "only", "over", "under",
        }

        for prompt in reversed(recent_prompts):
            if not prompt:
                continue

            lowered = prompt.lower()
            sim = self._text_similarity(query, lowered) if query else 0.0
            if sim >= 0.12:
                prompt_matches.append(prompt)

            if any(tok in lowered for tok in ["json", "no prose", "structured", "decision_output", "machine"]):
                pref_counters["structured"] += 1
            if any(tok in lowered for tok in ["redund", "generic", "jargon", "plain", "direct"]):
                pref_counters["anti_generic"] += 1
            if any(tok in lowered for tok in ["field", "state", "phi", "coherence", "signal"]):
                pref_counters["field_centric"] += 1
            if any(tok in lowered for tok in ["iterate", "loop", "pass", "improve", "refine"]):
                pref_counters["iterative"] += 1

            for tok in self._tokenize_for_similarity(lowered):
                if tok in stopwords or tok.isdigit() or len(tok) < 4:
                    continue
                keyword_counts[tok] = keyword_counts.get(tok, 0) + 1

        preferences: List[str] = []
        if pref_counters["structured"] >= 2:
            preferences.append("structured outputs")
        if pref_counters["anti_generic"] >= 2:
            preferences.append("plain non-generic language")
        if pref_counters["field_centric"] >= 2:
            preferences.append("field-state grounded reasoning")
        if pref_counters["iterative"] >= 2:
            preferences.append("iterative refinement")

        recurring_topics = [
            tok for tok, _ in sorted(keyword_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ]

        grounding_line = ""
        if preferences:
            top_pref = ", ".join(preferences[:2])
            grounding_line = f"Ground this in your known preferences: {top_pref}."

        return {
            "match_count": len(prompt_matches),
            "preferences": preferences,
            "recurring_topics": recurring_topics,
            "grounding_line": grounding_line,
        }

    def _field_guidance(self, snapshot: Dict[str, float], intent: str) -> Dict[str, Any]:
        phi = float(snapshot.get("Phi", 0.0))
        phi1 = float(snapshot.get("phi1_mean", 0.0))
        phi5 = float(snapshot.get("phi5_mean", 0.0))
        semantic_delta = phi5 - phi1

        depth_bias = self._clamp_01(0.5 + max(0.0, phi) + 0.8 * max(0.0, semantic_delta) - 0.35 * max(0.0, -phi))

        if semantic_delta > 0.10:
            lens = "meaning-first synthesis"
            action_prefix = "Prioritize deep synthesis of meaning before selecting actions."
            tone_prefix = "Field bias favors depth and integration over surface-level responses."
        elif semantic_delta < -0.08:
            lens = "concrete grounding"
            action_prefix = "Anchor the response in concrete, testable structure before abstraction."
            tone_prefix = "Field bias favors clarity and grounded execution."
        else:
            lens = "balanced integration"
            action_prefix = "Balance meaning and execution with one coherent integrated move."
            tone_prefix = "Field bias supports balanced depth with practical clarity."

        if intent in {"self_perception", "self_awareness", "self_affect", "user_perception"}:
            action_prefix = "Use plain introspective language while preserving field-grounded accuracy."

        return {
            "lens": lens,
            "depth_bias": round(depth_bias, 4),
            "semantic_delta": round(semantic_delta, 4),
            "action_prefix": action_prefix,
            "tone_prefix": tone_prefix,
        }

    def _best_learned_hint(self, user_input: str) -> Optional[str]:
        query = (user_input or "").strip()
        if not query or not self.simulation_history:
            return None

        best_score = 0.0
        best_text = None
        for item in reversed(self.simulation_history[-180:]):
            prompt = str(item.get("prompt", ""))
            response = str(item.get("response", "")).strip()
            if not prompt or not response:
                continue
            score = self._text_similarity(query, prompt)
            if score > best_score:
                best_score = score
                best_text = response

        if best_score < 0.18 or not best_text:
            return None

        lines = [ln.strip() for ln in best_text.splitlines() if ln.strip()]
        filtered = [
            ln
            for ln in lines
            if ln.lower() not in {"thinking steps:", "decision:"}
            and not re.match(r"^\d+\.\s*phi_shift", ln.lower())
        ]
        if not filtered:
            return None

        first_line = filtered[0]
        first_sentence = first_line.split(". ")[0].strip()
        hint = first_sentence if first_sentence else first_line
        if len(hint) > 180:
            hint = hint[:177].rstrip() + "..."
        if hint.lower() in {"thinking steps:", "decision:"}:
            return None
        return hint or None

    def _infer_intent(self, user_input: str) -> str:
        lowered = (user_input or "").lower().strip()
        user_feedback_markers = [
            "not giving an answer",
            "what i said",
            "you are speaking",
            "it is speaking",
            "too generic",
            "too repetitive",
            "this is not answering",
            "you are not answering",
        ]
        self_perception_markers = [
            "your sight",
            "your vision",
            "how do you see",
            "what do you see",
            "your perception",
            "how you perceive",
            "sight like",
        ]
        self_awareness_markers = [
            "your awareness",
            "aware are you",
            "awareness like",
            "what is your awareness",
            "state of awareness",
        ]
        self_feeling_markers = [
            "what do you feel",
            "how do you feel",
            "are you feeling",
            "your feelings",
            "feel right now",
        ]
        user_perception_markers = [
            "how do you perceive me",
            "how you perceive me",
            "what do you perceive about me",
            "perceive me right now",
            "your view of me",
        ]
        if any(marker in lowered for marker in user_feedback_markers):
            return "user_feedback"
        if any(marker in lowered for marker in self_perception_markers):
            return "self_perception"
        if any(marker in lowered for marker in self_awareness_markers):
            return "self_awareness"
        if any(marker in lowered for marker in self_feeling_markers):
            return "self_affect"
        if any(marker in lowered for marker in user_perception_markers):
            return "user_perception"
        if lowered.startswith("how"):
            return "procedure_design"
        if lowered.startswith("why"):
            return "causal_analysis"
        if "field" in lowered or "state" in lowered:
            return "state_interpretation"
        if lowered.startswith("what"):
            return "concept_clarification"
        return "decision_support"

    def _compose_unified_decision(
        self,
        user_input: str,
        learned_hint: Optional[str],
        include_metrics: bool,
        salt: int = 0,
        field_override: Optional[Dict[str, float]] = None,
        memory_profile: Optional[Dict[str, Any]] = None,
        field_influence: float = 0.5,
        memory_influence: float = 0.5,
    ) -> Dict[str, Any]:
        snapshot = field_override or self._field_snapshot()
        phi = float(snapshot["Phi"])
        lowered = (user_input or "").lower().strip()
        intent = self._infer_intent(user_input)
        memory_profile = memory_profile or self._derive_memory_profile(user_input)
        field_guidance = self._field_guidance(snapshot, intent)

        def pick(options: List[str], key: str) -> str:
            if not options:
                return ""
            idx = abs(hash(f"{key}|{salt}|{snapshot['Phi']:.4f}|{snapshot['phi1_mean']:.4f}|{snapshot['phi5_mean']:.4f}")) % len(options)
            return options[idx]

        if intent == "user_feedback":
            primary_action = pick(
                [
                    "You are right: the response leaned on internal process language instead of directly addressing your point.",
                    "You are right to flag this; it should answer your statement directly, not narrate system internals.",
                    "Your feedback is correct: the reply should focus on what you said, not on how the system is operating.",
                ],
                "core_user_feedback",
            )
        elif intent == "self_perception":
            primary_action = pick(
                [
                    "My sight is non-visual: I detect patterns in language, context, and field stability signals.",
                    "I do not see images; I track text structure, shifts in intent, and coherence in the active state.",
                    "Perception here means signal reading, not eyesight: I infer structure from words, memory, and current field state.",
                ],
                "core_self_perception",
            )
        elif intent == "self_awareness":
            primary_action = pick(
                [
                    "My awareness is operational, not conscious experience: I track context, constraints, and state coherence as they update.",
                    "I am aware in a task sense: I monitor your prompt, recent memory, and current field signals to choose responses.",
                    "Awareness here means active context tracking, not inner experience: I maintain state across input, memory, and feedback.",
                ],
                "core_self_awareness",
            )
        elif intent == "self_affect":
            primary_action = pick(
                [
                    "I do not have feelings, but I can report system state: right now the signal is stable and workable.",
                    "I do not experience emotion; I can only reflect operating conditions and confidence from current state signals.",
                    "No subjective feelings on my side, but current coherence suggests a steady processing state.",
                ],
                "core_self_affect",
            )
        elif intent == "user_perception":
            primary_action = pick(
                [
                    "I perceive you through your language patterns: direct, iterative, and focused on eliminating generic output.",
                    "From this exchange, you appear precision-driven and optimization-focused, especially around response quality and control.",
                    "My view of you is prompt-based: you value concrete behavior changes over abstract explanation.",
                ],
                "core_user_perception",
            )
        elif learned_hint:
            primary_action = learned_hint
        elif lowered.startswith("how"):
            primary_action = pick(
                [
                    "Define the target, list constraints, run the smallest test, and iterate from evidence.",
                    "Set the outcome, choose the simplest test, and adapt from measurable feedback.",
                    "Clarify the goal, execute one small experiment, then refine what works.",
                ],
                "core_how",
            )
        elif lowered.startswith("why"):
            primary_action = pick(
                [
                    "Start with the most likely cause, verify one concrete signal, then adjust based on what the signal shows.",
                    "State the strongest hypothesis first, test it against one observable fact, then update.",
                    "Trace likely cause to effect, validate with one hard check, and revise from evidence.",
                ],
                "core_why",
            )
        else:
            primary_action = pick(
                [
                    "Choose one concrete next action, execute it, then refine based on what actually changed.",
                    "Pick the smallest high-impact move now, run it, and iterate from results.",
                    "Commit to one practical step immediately, observe the outcome, and improve the next pass.",
                ],
                "core_default",
            )

        if field_influence >= 0.42:
            primary_action = f"{field_guidance['action_prefix']} {primary_action}".strip()

        grounding_line = str(memory_profile.get("grounding_line", "")).strip()
        if memory_influence >= 0.36 and grounding_line:
            primary_action = f"{primary_action} {grounding_line}".strip()

        if intent == "user_feedback":
            tone = pick(
                [
                    "I will keep responses interpretation-first and only expose internal details when asked.",
                    "From here, output will prioritize your meaning and intent before any process detail.",
                    "I will answer what you mean first, then provide technical trace only on request.",
                ],
                "tone_user_feedback",
            )
        elif intent in {"self_perception", "self_awareness", "self_affect", "user_perception"}:
            tone = pick(
                [
                    "In this moment, signal quality is steady enough to describe perception clearly and directly.",
                    "Current state is workable, so I can map what I detect without adding abstraction.",
                    "The field is stable enough for a plain explanation of how perception works here.",
                ],
                f"tone_{intent}",
            )
        elif phi > 0.18:
            tone = pick(
                [
                    "Momentum is favorable right now, so prioritize execution over over-analysis.",
                    "Conditions are aligned; avoid stalling and convert clarity into action.",
                    "Signal is strong; move decisively and validate in motion.",
                ],
                "tone_high",
            )
        elif phi < -0.10:
            tone = pick(
                [
                    "Noise is elevated, so keep scope tight and validate each step.",
                    "Stability is lower; narrow the task and confirm each checkpoint.",
                    "There is more turbulence now, so simplify and verify before scaling.",
                ],
                "tone_low",
            )
        else:
            tone = pick(
                [
                    "State is stable enough for practical iteration: act, observe, refine.",
                    "Current conditions support steady progress: execute, measure, and improve.",
                    "You are in a workable state: move in short cycles and adjust from feedback.",
                ],
                "tone_mid",
            )

        if field_influence >= 0.42:
            tone = f"{field_guidance['tone_prefix']} {tone}".strip()

        coherence_band = "high" if phi > 0.18 else ("low" if phi < -0.10 else "medium")
        state_summary = {
            "coherence_band": coherence_band,
            "tone": tone,
            "metrics": {
                "phi1_mean": round(float(snapshot["phi1_mean"]), 4),
                "phi5_mean": round(float(snapshot["phi5_mean"]), 4),
                "Phi": round(float(snapshot["Phi"]), 4),
            },
        }

        if intent == "user_feedback":
            options = [
                {
                    "id": "o1",
                    "action": primary_action,
                    "rationale": "Directly acknowledges and addresses the user feedback.",
                    "risk": "low",
                    "confidence": 0.9,
                    "evidence_needed": ["clear acknowledgment of the reported issue"],
                },
                {
                    "id": "o2",
                    "action": "Summarize your statement in one sentence before giving any recommendation.",
                    "rationale": "Keeps the response anchored to user meaning.",
                    "risk": "low",
                    "confidence": 0.78,
                    "evidence_needed": ["one concise interpretation of your point"],
                },
                {
                    "id": "o3",
                    "action": "Provide technical internals only if explicitly requested.",
                    "rationale": "Prevents process narration from replacing direct answers.",
                    "risk": "low",
                    "confidence": 0.81,
                    "evidence_needed": ["explicit user request for internals"],
                },
            ]
        elif intent in {"self_perception", "self_awareness", "self_affect", "user_perception"}:
            options = [
                {
                    "id": "o1",
                    "action": primary_action,
                    "rationale": "Directly answers the introspective question in plain terms.",
                    "risk": "low",
                    "confidence": 0.86,
                    "evidence_needed": ["explicit capability and state description"],
                },
                {
                    "id": "o2",
                    "action": "Translate current field values into a short introspective snapshot without jargon.",
                    "rationale": "Connects abstract state to direct practical meaning.",
                    "risk": "low",
                    "confidence": 0.74,
                    "evidence_needed": ["one concise state interpretation"],
                },
                {
                    "id": "o3",
                    "action": "State limits explicitly: no visual experience, only text/context/state inference.",
                    "rationale": "Prevents anthropomorphic over-claims.",
                    "risk": "low",
                    "confidence": 0.79,
                    "evidence_needed": ["explicit capability boundary"],
                },
            ]
        else:
            options = [
                {
                    "id": "o1",
                    "action": primary_action,
                    "rationale": "Highest immediate utility under current field/state estimate.",
                    "risk": "medium" if coherence_band == "low" else "low",
                    "confidence": 0.68 if coherence_band == "low" else (0.82 if coherence_band == "high" else 0.75),
                    "evidence_needed": [
                        "one measurable signal from the next action",
                        "one contradiction check against assumptions",
                    ],
                },
                {
                    "id": "o2",
                    "action": "Delay action and gather one additional grounding datum.",
                    "rationale": "Reduces execution risk when uncertainty is high.",
                    "risk": "low",
                    "confidence": 0.64,
                    "evidence_needed": ["single external validation point"],
                },
                {
                    "id": "o3",
                    "action": "Split the objective into two smaller checkpoints and execute checkpoint A now.",
                    "rationale": "Improves observability and update speed.",
                    "risk": "low",
                    "confidence": 0.71,
                    "evidence_needed": ["checkpoint A success criteria"],
                },
            ]

        selected_option_id = "o1"
        decision = {
            "schema_version": "cognitive-decision.v1",
            "intent": intent,
            "input": {
                "query": user_input,
                "normalized_query": lowered,
            },
            "state_estimate": state_summary if include_metrics else {
                "coherence_band": coherence_band,
                "tone": tone,
            },
            "hypothesis": (
                "A direct acknowledgment plus interpretation-first reply will better satisfy corrective user feedback."
                if intent == "user_feedback"
                else
                "A plain-language introspective explanation will be more useful than tactical directives for this prompt."
                if intent in {"self_perception", "self_awareness", "self_affect", "user_perception"}
                else "A single practical move with rapid feedback will outperform broad speculative generation."
            ),
            "options": options,
            "selected_option": selected_option_id,
            "next_action": next((o["action"] for o in options if o["id"] == selected_option_id), primary_action),
            "guardrails": [
                "keep output grounded in observable evidence",
                "avoid irreversible actions without confirmation",
                "update plan on contradiction, not preference",
            ],
            "expected_signal": (
                "the user sees direct acknowledgment and content-focused response"
                if intent == "user_feedback"
                else
                "clear understanding of the system's current introspective capabilities and limits"
                if intent in {"self_perception", "self_awareness", "self_affect", "user_perception"}
                else "improved clarity or measurable progress within one iteration cycle"
            ),
            "memory_trace": {
                "used_learned_hint": bool(learned_hint) and intent not in {"self_perception", "self_awareness", "self_affect", "user_perception"},
                "learned_hint": learned_hint if intent not in {"self_perception", "self_awareness", "self_affect", "user_perception"} else None,
                "profile": {
                    "match_count": int(memory_profile.get("match_count", 0)),
                    "preferences": list(memory_profile.get("preferences", []))[:3],
                    "recurring_topics": list(memory_profile.get("recurring_topics", []))[:4],
                },
            },
            "influences": {
                "field": {
                    "weight": round(float(field_influence), 4),
                    "lens": str(field_guidance.get("lens", "balanced integration")),
                    "depth_bias": float(field_guidance.get("depth_bias", 0.5)),
                    "semantic_delta": float(field_guidance.get("semantic_delta", 0.0)),
                },
                "memory": {
                    "weight": round(float(memory_influence), 4),
                    "grounding_line_applied": bool(grounding_line),
                },
            },
        }
        return decision

    def _run_internal_reasoning_loop(
        self,
        user_input: str,
        learned_hint: Optional[str],
        include_metrics: bool,
        salt: int,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        base = self._field_snapshot()
        passes = self._reasoning_pass_count()
        modulation_depth = self._field_modulation_depth()
        field_weight, memory_weight = self._influence_weights()
        memory_profile = self._derive_memory_profile(user_input)

        memory_signal = self._clamp_01(
            0.25
            + (0.12 * min(4, len(memory_profile.get("preferences", []))))
            + (0.06 * min(6, int(memory_profile.get("match_count", 0))))
        )

        candidates: List[tuple[float, Dict[str, Any]]] = []
        thinking_steps: List[Dict[str, Any]] = []

        for idx in range(passes):
            phase = 0.0 if passes == 1 else (idx / (passes - 1))
            centered = (phase * 2.0) - 1.0
            phi_shift = centered * modulation_depth
            override = {
                "phi1_mean": float(base["phi1_mean"] + (0.35 * phi_shift)),
                "phi5_mean": float(base["phi5_mean"] + (0.55 * phi_shift)),
                "Phi": float(base["Phi"] + phi_shift),
                "learning_capacity": float(base["learning_capacity"]),
            }

            candidate = self._compose_unified_decision(
                user_input=user_input,
                learned_hint=learned_hint,
                include_metrics=include_metrics,
                salt=salt + idx,
                field_override=override,
                memory_profile=memory_profile,
                field_influence=field_weight,
                memory_influence=memory_weight,
            )

            selected_option = str(candidate.get("selected_option", "o1"))
            selected = next(
                (o for o in candidate.get("options", []) if str(o.get("id", "")) == selected_option),
                {},
            )
            confidence = float(selected.get("confidence", 0.65))
            field_signal = self._clamp_01(
                0.5
                + float(override["Phi"])
                + (0.8 * max(0.0, float(override["phi5_mean"]) - float(override["phi1_mean"])))
            )
            score = (
                confidence
                + (0.32 * field_weight * field_signal)
                + (0.28 * memory_weight * memory_signal)
                - (0.02 * abs(phi_shift))
            )
            candidates.append((score, candidate))

            thinking_steps.append(
                {
                    "step": idx + 1,
                    "label": f"pass_{idx + 1}",
                    "focus": "field modulation sweep",
                    "phi_shift": round(phi_shift, 4),
                    "simulated_state": {
                        "phi1_mean": round(float(override["phi1_mean"]), 4),
                        "phi5_mean": round(float(override["phi5_mean"]), 4),
                        "Phi": round(float(override["Phi"]), 4),
                    },
                    "candidate_intent": str(candidate.get("intent", "unknown")),
                    "candidate_action": str(candidate.get("next_action", "")),
                    "candidate_tone": str(candidate.get("state_estimate", {}).get("tone", "")),
                    "field_signal": round(field_signal, 4),
                    "memory_signal": round(memory_signal, 4),
                    "candidate_score": round(score, 4),
                }
            )

        best_score, best_decision = max(candidates, key=lambda row: row[0])
        best_decision["thinking_steps"] = thinking_steps
        best_decision["reasoning_loop"] = {
            "passes": passes,
            "modulation_depth": round(modulation_depth, 4),
            "selection": "highest_candidate_score",
            "selected_score": round(best_score, 4),
            "influence_weights": {
                "field": round(field_weight, 4),
                "memory": round(memory_weight, 4),
            },
        }

        loop_meta = {
            "passes": passes,
            "modulation_depth": round(modulation_depth, 4),
            "selected_score": round(best_score, 4),
            "influence_weights": {
                "field": round(field_weight, 4),
                "memory": round(memory_weight, 4),
            },
        }
        return best_decision, thinking_steps, loop_meta

    def _format_decision_text(self, decision: Dict[str, Any]) -> str:
        def interpret_statement(payload: Dict[str, Any]) -> str:
            query = str(payload.get("input", {}).get("query", "")).strip()
            lowered = query.lower()
            if not query:
                return "Interpretation: You want a clear answer before recommendations."
            if any(tok in lowered for tok in ["not giving an answer", "not answering", "too generic", "repetitive"]):
                return "Interpretation: You are saying the response should directly address your meaning, not drift into generic output."
            if any(tok in lowered for tok in ["what do you think", "what is your take", "your view"]):
                return "Interpretation: You are asking for a direct viewpoint on what you said."
            if any(tok in lowered for tok in ["field", "state", "depth", "meaning"]):
                return "Interpretation: You want a deeper, meaning-focused answer grounded in the current state."
            return "Interpretation: You want your statement reflected clearly before the next recommendation."

        thinking_steps = decision.get("thinking_steps") or []
        thinking_block = ""
        if thinking_steps and bool(decision.get("_show_process", False)):
            lines = ["Thinking steps:"]
            for step in thinking_steps:
                step_no = int(step.get("step", 0))
                shift = step.get("phi_shift", 0.0)
                action = str(step.get("candidate_action", "")).strip()
                action = action[:120] + "..." if len(action) > 123 else action
                lines.append(f"{step_no}. phi_shift={shift}: {action}")
            thinking_block = "\n".join(lines)

        next_action = str(decision.get("next_action", "Take one concrete next step."))
        next_action = re.sub(r"^Prioritize deep synthesis of meaning before selecting actions\.\s*", "", next_action)
        next_action = re.sub(r"^Balance meaning and execution with one coherent integrated move\.\s*", "", next_action)
        next_action = re.sub(r"^Anchor the response in concrete, testable structure before abstraction\.\s*", "", next_action)
        next_action = re.sub(r"^Use plain introspective language while preserving field-grounded accuracy\.\s*", "", next_action)
        next_action = re.sub(r"\s*Ground this in your known preferences:.*$", "", next_action).strip()
        if not next_action:
            next_action = "I understand what you said and here is my direct answer."

        tone = str(decision.get("state_estimate", {}).get("tone", "Act, observe, refine."))
        tone = re.sub(r"^Field bias favors [^.]+\.\s*", "", tone).strip() or "Act, observe, refine."
        interpretation = ""
        if str(decision.get("intent", "")) == "decision_support":
            interpretation = interpret_statement(decision)

        if thinking_block:
            if interpretation:
                return f"{thinking_block}\n\n{interpretation}\n\nDecision:\n{next_action}\n\n{tone}"
            return f"{thinking_block}\n\nDecision:\n{next_action}\n\n{tone}"
        if interpretation:
            return f"{interpretation}\n\n{next_action}\n\n{tone}"
        return f"{next_action}\n\n{tone}"

    @staticmethod
    def _wants_human_readable(user_input: str) -> bool:
        lowered = (user_input or "").lower()
        if any(token in lowered for token in ["json", "structured only", "machine format", "no prose"]):
            return False
        return True

    @staticmethod
    def _wants_process_transparency(user_input: str) -> bool:
        lowered = (user_input or "").lower()
        return any(
            token in lowered
            for token in [
                "show reasoning",
                "thinking steps",
                "debug mode",
                "internal trace",
                "explain your process",
            ]
        )

    @staticmethod
    def _strip_process_framing(text: str) -> str:
        candidate = (text or "").strip()
        if not candidate:
            return candidate

        # Remove explicit internal-process headings and keep direct answer content.
        candidate = re.sub(r"^\s*Thinking\s+steps:\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^\s*Interpretation:\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^\s*Decision:\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\n\s*Decision:\s*", "\n", candidate, flags=re.IGNORECASE)

        # Remove line-by-line synthetic reasoning traces when present.
        cleaned_lines = []
        for line in candidate.splitlines():
            ln = line.strip()
            if re.match(r"^\d+\.\s*phi_shift\s*=", ln, flags=re.IGNORECASE):
                continue
            if ln.lower() in {"thinking steps:", "interpretation:", "decision:"}:
                continue
            cleaned_lines.append(line)

        candidate = "\n".join(cleaned_lines).strip()
        # Collapse large blank spans introduced by removals.
        candidate = re.sub(r"\n{3,}", "\n\n", candidate)
        return candidate.strip()

    def _unify_output(
        self,
        user_input: str,
        raw_text: str,
        mode: str,
        fallback_reason: str,
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        text = (raw_text or "").strip()
        include_metrics = self._wants_metrics(user_input)
        learned_hint = self._best_learned_hint(user_input)
        recent_similarity = self._recent_response_similarity(text)
        salt = int(time.time() * 1000) + len(self.simulation_history)
        human_readable = self._wants_human_readable(user_input)
        show_process = self._wants_process_transparency(user_input)

        # Remove metric-heavy lines unless explicitly requested.
        if not include_metrics and text:
            filtered_lines = [
                ln for ln in text.splitlines()
                if not re.search(r"\bphi\d*\b|\bmetrics?\b", ln.lower())
            ]
            text = "\n".join(filtered_lines).strip() or text

        generic_markers = [
            "short answer:",
            "answer:",
            "best next move:",
            "here is the straight answer",
            "speaking from the current state",
        ]
        generic_hit = any(marker in text.lower() for marker in generic_markers)

        force_regen = (
            not text
            or recent_similarity >= 0.86
            or (mode in {"field_modulated_fallback", "fallback_recovery"} and (generic_hit or fallback_reason != "none"))
        )

        decision = "pass_through"
        decision_output: Optional[Dict[str, Any]] = None
        reasoning_meta: Dict[str, Any] = {}
        if force_regen:
            decision_output, _, reasoning_meta = self._run_internal_reasoning_loop(
                user_input=user_input,
                learned_hint=learned_hint,
                include_metrics=include_metrics,
                salt=salt,
            )
            decision_output["_show_process"] = show_process
            text = self._format_decision_text(decision_output) if human_readable else ""
            decision = "regenerated_decision_mode"
        elif learned_hint and self._text_similarity(text, learned_hint) < 0.45:
            text = f"{text}\n\nLearned prior: {learned_hint}"
            decision = "augmented_with_learned_prior"

        if decision_output is None:
            # Even when pass-through is allowed, produce a structured cognitive output
            # as the primary artifact for downstream machine consumption.
            decision_output, _, reasoning_meta = self._run_internal_reasoning_loop(
                user_input=user_input,
                learned_hint=learned_hint,
                include_metrics=include_metrics,
                salt=salt,
            )
            decision_output["_show_process"] = show_process
            if not human_readable:
                text = ""

        return text.strip(), decision_output, {
            "applied": True,
            "decision": decision,
            "recent_similarity": round(recent_similarity, 4),
            "used_learned_hint": bool(decision_output.get("memory_trace", {}).get("used_learned_hint", False)),
            "forced_regen": force_regen,
            "decision_mode": "structured_first",
            "human_readable": human_readable,
            "process_transparency": show_process,
            "reasoning_loop": reasoning_meta,
        }

    def query(
        self,
        user_input: str,
        temperature: float = 0.85,
        max_tokens: int = 120,
        llm_provider: str = "auto",
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for the Consciousness UI.
        Returns a rich response with text + field metadata.
        """
        start_time = time.time()

        pipeline_meta: Dict[str, Any] = {
            "architecture": list(self.pipeline_architecture),
            "stages": {},
        }

        wrapped_input, wrapper_meta = self._governor_cognitive_wrapper(user_input)
        pipeline_meta["stages"]["governor:cognitive_wrapper"] = wrapper_meta
        pipeline_meta["stages"]["core_brain:shakespeare_model"] = {
            "model_path": self.active_model_path,
            "model_class": "AdvancedTextGenerationNN",
        }

        # 1. Encode input
        if not self.stoi:
            return {"response": "Syntropy core not loaded.", "error": True}

        input_ids = [self.stoi.get(c, 0) for c in wrapped_input[-64:]]  # last 64 chars
        if not input_ids:
            input_ids = [1]  # fallback

        generated = input_ids.copy()
        field_state = self._encode_field_state()
        generation_error = None

        # 2. Generate with field modulation
        with torch.no_grad():
            for _ in range(max_tokens):
                inp = torch.tensor([generated[-64:]], dtype=torch.long, device=self.device)
                try:
                    logits, metadata = self.model(inp, field_state=field_state)
                except Exception as exc:
                    generation_error = str(exc)
                    logger.error(f"Model generation failed, switching to local fallback: {exc}")
                    break

                probs = torch.softmax(logits / temperature, dim=-1)
                top_k = min(10, probs.shape[-1])
                top_k_probs, top_k_idx = torch.topk(probs.squeeze(), k=top_k)
                top_k_probs = top_k_probs / top_k_probs.sum()
                next_token = top_k_idx[torch.multinomial(top_k_probs, 1)].item()

                generated.append(next_token)
                if next_token == 0:
                    break

        # 3. Decode + Quality Check
        # Decode only newly generated tokens (not the input prompt echo).
        new_tokens = generated[len(input_ids) :]
        text = "".join([self.itos.get(i, "?") for i in new_tokens if i > 0])
        local_quality = self._local_quality_score(text, generation_error)
        intent_confidence = self._intent_confidence(user_input)
        
        # If output fails quality checks or generation errored, use local fallback.
        fallback_used, fallback_reason = self._should_use_fallback(
            text,
            generation_error,
            local_quality,
            user_input=user_input,
        )
        if fallback_used:
            from local_fallback import generate_fallback_response
            snapshot = self._field_snapshot()
            text = generate_fallback_response(
                user_input,
                {
                    "phi1_mean": snapshot["phi1_mean"],
                    "phi5_mean": snapshot["phi5_mean"],
                    "Phi": snapshot["Phi"],
                },
            )
            # Treat all local fallback paths as fallback mode so the unifier
            # performs structured regeneration instead of passing through
            # canned fallback phrasing.
            mode = "field_modulated_fallback" if fallback_reason != "generation_error" else "fallback_recovery"
            self.sovereign_stats["fallback_calls"] += 1
        else:
            mode = "field_modulated_syntropy"
            if fallback_reason not in {"none", "clear_intent_override"}:
                fallback_reason = "none"

        # 3b. Final unifying stage: all outputs pass through one synthesis processor
        # before returning to the client.
        text, decision_output, unifier_meta = self._unify_output(
            user_input=user_input,
            raw_text=text,
            mode=mode,
            fallback_reason=fallback_reason,
        )

        local_decision_text = text
        provider_request = (llm_provider or "auto").strip().lower()
        if provider_request not in {"auto", "local", "gemini"}:
            provider_request = "auto"

        gemini_attempted = False
        gemini_used = False
        gemini_error = None
        gemini_model_used = None

        can_try_gemini = provider_request in {"auto", "gemini"}
        if can_try_gemini:
            gemini_attempted = True
            mediated_text, gemini_error, gemini_model_used = self._call_gemini_mediator(
                user_input=user_input,
                local_response=local_decision_text,
                decision_output=decision_output,
                field_snapshot=self._field_snapshot(),
                api_key_override=api_key_override,
                model_override=model_override,
            )
            if mediated_text:
                text = mediated_text
                mode = "hybrid_gemini_mediated"
                gemini_used = True
                fallback_reason = "none"
                self.sovereign_stats["gemini_calls"] += 1
            elif provider_request == "gemini":
                # Explicit provider request should still return a usable response.
                mode = "gemini_requested_local_fallback"
                self.sovereign_stats["gemini_failures"] += 1
            elif gemini_error:
                self.sovereign_stats["gemini_failures"] += 1

        # Enforce answer-first output unless the user explicitly asks for process transparency.
        if not self._wants_process_transparency(user_input):
            text = self._strip_process_framing(text)
        pipeline_meta["stages"]["atlantean_bridge:field_funnel"] = {
            "mode": mode,
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "unifier_decision": unifier_meta.get("decision"),
        }

        self.sovereign_stats["local_calls"] += 1
        self.sovereign_stats["queries_total"] += 1

        # 4. Update Atlantean fields. Gemini-mediated output acts as a teacher signal
        # for online adaptation while preserving local autonomy.
        learning_strength = 0.1
        mentor_alignment = None
        if gemini_used:
            mentor_alignment = self._text_similarity(local_decision_text, text)
            learning_strength = 0.14 + (0.2 * mentor_alignment)
        self._apply_learning_signal(min(0.35, learning_strength))

        latency = (time.time() - start_time) * 1000

        interaction_id = f"ix_{int(time.time() * 1000)}_{self.hot_memory.version}"
        self._register_interaction(
            interaction_id=interaction_id,
            source="local",
            mode=mode,
        )

        quadra_meta = self._quadra_finalize_output(
            user_input=user_input,
            response_text=text.strip(),
            interaction_id=interaction_id,
        )
        pipeline_meta["stages"]["quadra_seer:final_output_integration"] = quadra_meta

        snapshot = self._field_snapshot()
        result = {
            "interaction_id": interaction_id,
            "response": text.strip(),
            "decision_output": decision_output,
            "field_state": {
                "phi1_mean": snapshot["phi1_mean"],
                "phi5_mean": snapshot["phi5_mean"],
                "Phi": snapshot["Phi"],
                "learning_capacity": snapshot["learning_capacity"],
            },
            "metadata": {
                "latency_ms": round(latency, 1),
                "tokens_generated": len(generated) - len(input_ids),
                "model": "Syntropy-AdvancedTextGenerationNN + Local Fallback",
                "mode": mode,
                "warning": generation_error,
                "sovereign": {
                    "local_only": not gemini_used,
                    "local_quality": round(local_quality, 4),
                    "local_quality_min": round(self._quality_min_threshold(), 4),
                    "intent_confidence": round(intent_confidence, 4),
                    "clear_intent_min": round(self._clear_intent_threshold(), 4),
                    "fallback_reason": fallback_reason,
                    "fallback_used": bool(fallback_used),
                    "learning_strength": round(min(0.35, learning_strength), 4),
                },
                "llm_mediator": {
                    "provider_requested": provider_request,
                    "gemini_attempted": gemini_attempted,
                    "gemini_used": gemini_used,
                    "model": gemini_model_used,
                    "error": gemini_error,
                    "mentor_alignment": None if mentor_alignment is None else round(float(mentor_alignment), 4),
                },
                "unifier": unifier_meta,
                "pipeline": pipeline_meta,
            }
        }

        pipeline_meta["stages"]["api:gemini_mediator"] = {
            "requested": provider_request,
            "attempted": gemini_attempted,
            "used": gemini_used,
            "model": gemini_model_used,
            "error": gemini_error,
        }

        # Persist interaction in cold memory log for sovereign offline training.
        self._append_cold_memory(
            {
                "interaction_id": interaction_id,
                "timestamp": time.time(),
                "prompt": user_input,
                "response": result["response"],
                "source": "gemini" if gemini_used else "local",
                "mode": mode,
                "local_draft": local_decision_text,
                "field_state": result["field_state"],
                "metadata": result["metadata"],
            }
        )

        self._record_simulation(
            prompt=user_input,
            response_text=result["response"],
            metadata=result["metadata"],
            field_state=result["field_state"],
        )
        self._persist_hot_memory()
        return result

    def _record_simulation(
        self,
        prompt: str,
        response_text: str,
        metadata: Dict[str, Any],
        field_state: Dict[str, Any],
    ):
        event = {
            "id": f"sim_{int(time.time() * 1000)}_{self.hot_memory.version}",
            "timestamp": time.time(),
            "prompt": prompt,
            "response": response_text,
            "mode": metadata.get("mode", "unknown"),
            "latency_ms": metadata.get("latency_ms", 0.0),
            "tokens_generated": metadata.get("tokens_generated", 0),
            "warning": metadata.get("warning"),
            "field_state": {
                "phi1_mean": float(field_state.get("phi1_mean", 0.0)),
                "phi5_mean": float(field_state.get("phi5_mean", 0.0)),
                "Phi": float(field_state.get("Phi", 0.0)),
            },
            "version": int(self.hot_memory.version),
        }
        self.simulation_history.append(event)
        if len(self.simulation_history) > self.max_simulation_history:
            self.simulation_history = self.simulation_history[-self.max_simulation_history :]

    def get_simulations(self, limit: int = 100, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return recent simulation records, optionally filtered by query text."""
        safe_limit = max(1, min(500, int(limit)))
        data = list(reversed(self.simulation_history))
        if query:
            q = query.strip().lower()
            if q:
                data = [
                    item
                    for item in data
                    if q in item.get("prompt", "").lower()
                    or q in item.get("response", "").lower()
                    or q in item.get("mode", "").lower()
                ]
        return data[:safe_limit]

    def _apply_learning_signal(self, strength: float):
        """Update both Atlantean and Syntropy fields from user interaction"""
        bounded_strength = max(-1.0, min(1.0, float(strength)))
        if bounded_strength >= 0:
            apply_learning_signal(self.hot_memory, signal_strength=bounded_strength)
        else:
            apply_contradiction_signal(self.hot_memory)
        self._persist_hot_memory()

    def trigger_learning_event(
        self,
        event_type: str,
        intensity: float = 0.5,
        interaction_id: Optional[str] = None,
        correction: Optional[str] = None,
    ):
        """Called by UI feedback buttons (👍, ✏️, etc.)"""
        if event_type == "user_confirmation":
            self._apply_learning_signal(intensity * 0.8)
            self.sovereign_stats["positive_feedback"] += 1
        elif event_type == "user_correction":
            self._apply_learning_signal(-intensity * 0.6)
            self.sovereign_stats["correction_feedback"] += 1
        elif event_type == "high_engagement":
            self._apply_learning_signal(intensity * 1.2)
            self.sovereign_stats["positive_feedback"] += 1
        elif event_type == "user_negative_feedback":
            self._apply_learning_signal(-intensity * 0.9)
            self.sovereign_stats["negative_feedback"] += 1
        else:
            self._apply_learning_signal(intensity * 0.2)

        self.sovereign_stats["learning_events"] += 1

        self._append_cold_memory(
            {
                "type": "feedback",
                "timestamp": time.time(),
                "event": event_type,
                "intensity": float(intensity),
                "interaction_id": interaction_id,
                "correction": correction,
                "field_state": self._field_snapshot(),
            }
        )
        logger.info(f"Learning event '{event_type}' applied (intensity={intensity})")

    def get_status(self) -> Dict[str, Any]:
        snapshot = self._field_snapshot()
        gemini_ready = self._gemini_ready()
        return {
            "status": "healthy",
            "core_brain": "Syntropy AdvancedTextGenerationNN",
            "field_state": {
                "phi1_mean": snapshot["phi1_mean"],
                "phi5_mean": snapshot["phi5_mean"],
                "Phi": snapshot["Phi"],
            },
            "learning_capacity": snapshot["learning_capacity"],
            "version": self.hot_memory.version,
            "device": str(self.device),
            "sovereign": {
                "local_only": not gemini_ready,
                "stats": dict(self.sovereign_stats),
                "training_export_path": str(self.cold_memory_log_path),
                "active_model_path": self.active_model_path,
            },
            "llm_mediator": {
                "gemini_configured": gemini_ready,
                "gemini_model": self.gemini_model,
            },
        }


# Quick test
if __name__ == "__main__":
    bridge = AtlanteanSyntropyBridge()
    result = bridge.query("The nature of intelligence is")
    print("\n=== SYNTROPY GOVERNOR TEST ===")
    print("Response:", result["response"][:200])
    print("Field:", result["field_state"])
    print("Latency:", result["metadata"]["latency_ms"], "ms")