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

# Add paths for the three systems
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "core_brain"))
sys.path.insert(0, str(BASE_DIR / "governor"))
sys.path.insert(0, str(BASE_DIR / "consciousness/atlantean_core"))

from syntropy_field_expanded import AdvancedTextGenerationNN, OscillatorySynapseTheory
# from atlantean_quadra_bridge import AtlanteanQuadraBridge  # Will be adapted

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

        model_path = model_path or str(BASE_DIR / "core_brain/shakespeare_model.pt")
        self._load_syntropy_model(model_path)

        # === 2. Initialize Atlantean Memory (simplified for now) ===
        self.atlantean_memory = {
            "phi1": np.random.randn(32, 32).astype(np.float32) * 0.1,
            "phi5": np.random.randn(32, 32).astype(np.float32) * 0.1,
            "Phi": 0.5,
            "version": 0,
            "learning_capacity": 0.3
        }
        self.simulation_history: List[Dict[str, Any]] = []
        self.max_simulation_history = 500

        # === 3. Field State Encoder (maps Atlantean → Syntropy field_state) ===
        self.field_state_dim = self._infer_field_state_dim()
        self.field_encoder = nn.Linear(3, self.field_state_dim).to(self.device)

        logger.info("✅ Atlantean + Syntropy Bridge initialized successfully")

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
        logger.info(f"✅ Loaded Syntropy model (vocab={self.vocab_size}, emb={emb_dim})")

    def _encode_field_state(self) -> torch.Tensor:
        """Convert current Atlantean fields into Syntropy field_state vector"""
        phi1_mean = float(np.mean(self.atlantean_memory["phi1"]))
        phi5_mean = float(np.mean(self.atlantean_memory["phi5"]))
        Phi = float(self.atlantean_memory["Phi"])

        field_vec = torch.tensor([[phi1_mean, phi5_mean, Phi]], dtype=torch.float32, device=self.device)
        field_state = self.field_encoder(field_vec)
        return field_state.squeeze(0)  # (256,)

    def query(self, user_input: str, temperature: float = 0.85, max_tokens: int = 120) -> Dict[str, Any]:
        """
        Main entry point for the Consciousness UI.
        Returns a rich response with text + field metadata.
        """
        start_time = time.time()

        # 1. Encode input
        if not self.stoi:
            return {"response": "Syntropy core not loaded.", "error": True}

        input_ids = [self.stoi.get(c, 0) for c in user_input[-64:]]  # last 64 chars
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
        text = "".join([self.itos.get(i, "?") for i in generated if i > 0])
        
        # If output fails quality checks or generation errored, use local fallback.
        if generation_error or len(text) < 20 or text.count(text[0]) > len(text) * 0.4:
            from local_fallback import generate_fallback_response
            text = generate_fallback_response(user_input, {
                "phi1_mean": float(np.mean(self.atlantean_memory["phi1"])),
                "phi5_mean": float(np.mean(self.atlantean_memory["phi5"])),
                "Phi": float(self.atlantean_memory["Phi"])
            })
            mode = "field_modulated_fallback" if not generation_error else "fallback_recovery"
        else:
            mode = "field_modulated_syntropy"

        # 4. Update Atlantean fields (simple learning signal)
        self._apply_learning_signal(0.1)

        latency = (time.time() - start_time) * 1000

        result = {
            "response": text.strip(),
            "field_state": {
                "phi1_mean": float(np.mean(self.atlantean_memory["phi1"])),
                "phi5_mean": float(np.mean(self.atlantean_memory["phi5"])),
                "Phi": float(self.atlantean_memory["Phi"]),
                "learning_capacity": self.atlantean_memory["learning_capacity"]
            },
            "metadata": {
                "latency_ms": round(latency, 1),
                "tokens_generated": len(generated) - len(input_ids),
                "model": "Syntropy-AdvancedTextGenerationNN + Local Fallback",
                "mode": mode,
                "warning": generation_error
            }
        }

        self._record_simulation(
            prompt=user_input,
            response_text=result["response"],
            metadata=result["metadata"],
            field_state=result["field_state"],
        )
        return result

    def _record_simulation(
        self,
        prompt: str,
        response_text: str,
        metadata: Dict[str, Any],
        field_state: Dict[str, Any],
    ):
        event = {
            "id": f"sim_{int(time.time() * 1000)}_{self.atlantean_memory['version']}",
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
            "version": int(self.atlantean_memory.get("version", 0)),
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
        # Atlantean hot memory update (simplified)
        self.atlantean_memory["phi1"] = np.clip(
            self.atlantean_memory["phi1"] + strength * np.random.randn(32, 32).astype(np.float32) * 0.05,
            -2.0, 2.0
        )
        self.atlantean_memory["phi5"] = np.clip(
            self.atlantean_memory["phi5"] + strength * 0.8 * np.random.randn(32, 32).astype(np.float32) * 0.03,
            -1.5, 1.5
        )
        self.atlantean_memory["Phi"] = np.clip(
            self.atlantean_memory["Phi"] + strength * 0.05, 0.0, 1.0
        )
        self.atlantean_memory["version"] += 1
        self.atlantean_memory["learning_capacity"] = min(0.95, self.atlantean_memory["learning_capacity"] + 0.01)

    def trigger_learning_event(self, event_type: str, intensity: float = 0.5):
        """Called by UI feedback buttons (👍, ✏️, etc.)"""
        if event_type == "user_confirmation":
            self._apply_learning_signal(intensity * 0.8)
        elif event_type == "user_correction":
            self._apply_learning_signal(-intensity * 0.6)
        elif event_type == "high_engagement":
            self._apply_learning_signal(intensity * 1.2)
        logger.info(f"Learning event '{event_type}' applied (intensity={intensity})")

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "core_brain": "Syntropy AdvancedTextGenerationNN",
            "field_state": {
                "phi1_mean": float(np.mean(self.atlantean_memory["phi1"])),
                "phi5_mean": float(np.mean(self.atlantean_memory["phi5"])),
                "Phi": float(self.atlantean_memory["Phi"]),
            },
            "learning_capacity": self.atlantean_memory["learning_capacity"],
            "version": self.atlantean_memory["version"],
            "device": str(self.device)
        }


# Quick test
if __name__ == "__main__":
    bridge = AtlanteanSyntropyBridge()
    result = bridge.query("The nature of intelligence is")
    print("\n=== SYNTROPY GOVERNOR TEST ===")
    print("Response:", result["response"][:200])
    print("Field:", result["field_state"])
    print("Latency:", result["metadata"]["latency_ms"], "ms")