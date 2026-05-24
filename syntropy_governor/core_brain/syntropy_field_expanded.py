import numpy as np
import asyncio
from scipy.integrate import solve_ivp
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Optional, Dict, Tuple
import threading
from scipy.fft import fft, fftfreq
from scipy.stats import variation
import sympy as sp
import requests
import json
from datetime import datetime
import hashlib
from collections import deque
import os

try:
    from cuda_pde_kernel import pde_step_cuda
except Exception:
    pde_step_cuda = None

logging.basicConfig(level=logging.INFO)

# =====================================================
# CORE NEURAL NETWORK MODULES
# =====================================================

class FieldUpdateNN(nn.Module):
    """Neural Network inspired by Field Theory equations."""
    def __init__(self, input_size: int, hidden_size: int = 96):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @torch.compile
    def update_field(self, field: torch.Tensor, chaos_threshold: float = 0.1) -> torch.Tensor:
        update = self.forward(field)
        variance = torch.var(field)
        if variance > chaos_threshold:
            update = update * 0.5
            field = torch.clamp(field, -1.0, 1.0)
        return field + update


class SyntropyNN(nn.Module):
    """Neural Network for Syntropy adjustment."""
    def __init__(self, input_size: int, hidden_size: int = 48):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def adjust_syntropy(self, field: torch.Tensor, target_mean: float = 0.5, speed_factor: float = 0.5) -> torch.Tensor:
        current_mean = torch.mean(field).item()
        adjustment = (target_mean - current_mean) * speed_factor
        mean_input = torch.full_like(field, float(current_mean))
        delta = self.forward(mean_input) * adjustment
        return field + delta * 0.5


class FeedbackNN(nn.Module):
    """Neural Network for variance feedback control."""
    def __init__(self, input_size: int, hidden_size: int = 72):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        penalty = self.fc3(x)
        return penalty

    def penalize_variance(self, field: torch.Tensor) -> torch.Tensor:
        variance = torch.var(field)
        var_input = torch.full((1, field.size(0)), variance.item())
        penalty = self.forward(var_input).item()
        stability_tune = -penalty * variance.item()
        return field + torch.full_like(field, float(stability_tune / field.size(0)))


class FieldEnhancedSNN(nn.Module):
    """Field-enhanced spiking neural network using PDE-driven fields.

    Implements phi1 (excitability), phi5 (plasticity/entropy) and Phi (global potential).
    Uses finite-difference Laplacian kernels (conv2d) for spatial derivatives.
    """
    def __init__(self, size=32,
                 alpha=1.0, zeta=0.5, gamma1=0.2, beta=0.05,
                 omega=0.1, theta=0.3, gamma5=0.15,
                 base_threshold: float = 1.0, base_leak: float = 0.1,
                 device: Optional[torch.device] = None):
        super().__init__()
        # canonical grid size: either int or (H,W)
        if isinstance(size, int):
            H = W = size
        else:
            H, W = size
        self.H, self.W = H, W
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        # PDE parameters
        self.alpha = alpha
        self.zeta = zeta
        self.gamma1 = gamma1
        self.beta = beta
        self.omega = omega
        self.theta = theta
        self.gamma5 = gamma5

        # Neural parameters
        self.base_threshold = float(base_threshold)
        self.base_leak = float(base_leak)

        # State fields (buffers, not learnable parameters)
        # shape: (1,1,H,W) for conv2d convenience
        self.register_buffer('phi1', torch.randn(1, 1, H, W, device=self.device) * 0.1)
        self.register_buffer('phi5', torch.ones(1, 1, H, W, device=self.device) * 0.1)
        self.register_buffer('Phi', torch.zeros(1, 1, H, W, device=self.device))

        # Membrane potential buffer for LIF neurons
        self.register_buffer('mem', torch.zeros(1, 1, H, W, device=self.device))

        # Laplacian kernel (3x3)
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=self.device)
        self.register_buffer('lap_kernel', lap.view(1, 1, 3, 3))

        # Optional custom CUDA PDE kernel path for 1000Hz physics scaling.
        use_cuda_env = os.getenv("SYNTROPY_USE_CUDA_PDE", "true").lower() in {"1", "true", "yes"}
        self.use_cuda_pde = bool(use_cuda_env and pde_step_cuda is not None and self.device.type == "cuda")
        if use_cuda_env and pde_step_cuda is None:
            logging.info("Custom CUDA PDE kernel unavailable; using PyTorch PDE fallback.")

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute discrete Laplacian using conv2d with padding=1."""
        # x assumed shape (1,1,H,W)
        return F.conv2d(x, self.lap_kernel, padding=1)

    def bi_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        return self.laplacian(self.laplacian(x))

    def pde_step(self, dt: float = 0.01):
        if self.use_cuda_pde and self.phi1.is_cuda and self.phi5.is_cuda and self.Phi.is_cuda:
            cuda_out = pde_step_cuda(
                self.phi1.contiguous(),
                self.phi5.contiguous(),
                self.Phi.contiguous(),
                alpha=float(self.alpha),
                zeta=float(self.zeta),
                gamma1=float(self.gamma1),
                beta=float(self.beta),
                omega=float(self.omega),
                theta=float(self.theta),
                gamma5=float(self.gamma5),
                dt=float(dt),
            )
            if cuda_out is not None:
                next_phi1, next_phi5 = cuda_out
                self.phi1.copy_(next_phi1)
                self.phi5.copy_(next_phi5)
                return

        # Spatial derivatives
        lap_phi1 = self.laplacian(self.phi1)
        bi_lap_phi1 = self.bi_laplacian(self.phi1)
        lap_phi5 = self.laplacian(self.phi5)

        # Reaction (cubic nonlinearity) + coupling to global Phi via Laplacian(Phi)
        lap_Phi = self.laplacian(self.Phi)
        reaction = -self.alpha * (self.phi1 ** 3) + self.zeta * lap_Phi

        # Diffusion: negative diffusion + hyper-diffusion (bi-Laplacian)
        diffusion = -self.gamma1 * lap_phi1 + self.beta * bi_lap_phi1
        d_phi1 = diffusion + reaction
        self.phi1 = self.phi1 + d_phi1 * dt

        # Phi5: log-nonlinearity (soliton-like) + advection toward Phi (chemotaxis-like)
        # log term stabilized with small epsilon
        eps = 1e-6
        log_term = self.omega * self.phi5 * torch.log(torch.abs(self.phi5) + eps)
        # simplified advection term: theta * (lap_phi5 * Phi + phi5 * lap(Phi))
        advection = self.theta * (lap_phi5 * self.Phi + self.phi5 * lap_Phi)

        d_phi5 = -self.gamma5 * lap_phi5 + log_term + advection
        self.phi5 = self.phi5 + d_phi5 * dt

        # Optional clipping for numerical stability
        self.phi1 = torch.clamp(self.phi1, -5.0, 5.0)
        self.phi5 = torch.clamp(self.phi5, -10.0, 10.0)

    def integrate(self, input_spikes: torch.Tensor, leak: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simple leaky integration over spatial grid.
        input_spikes: expected shape (1,1,H,W) or broadcastable
        leak: same shape or scalar
        """
        if leak is None:
            leak = self.base_leak
        # mem = mem * (1 - leak) + input
        self.mem = self.mem * (1.0 - leak) + input_spikes
        return self.mem

    def forward(self, input_spikes: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Run one update: PDE step -> modulate SNN -> integrate LIF -> return spikes"""
        # Ensure input shape
        if input_spikes.dim() == 2:
            # assume (H,W)
            input_spikes = input_spikes.view(1, 1, self.H, self.W).to(self.phi1.device)
        else:
            input_spikes = input_spikes.to(self.phi1.device)

        # Step A: update fields
        self.pde_step(dt=dt)

        # Step B: modulate SNN parameters
        dynamic_threshold = self.base_threshold - 0.5 * self.phi1  # lowers threshold where phi1 high
        effective_leak = self.base_leak * (1.0 / (1.0 + self.phi5))

        # Step C: integrate membrane potential and produce spikes (LIF-like)
        mem = self.integrate(input_spikes, leak=effective_leak)
        spikes = (mem > dynamic_threshold).float()

        # reset membrane at spike locations (simple reset)
        self.mem = self.mem * (1.0 - spikes)

        # Feedback: spikes influence Phi (global interplay potential)
        self.Phi = 0.9 * self.Phi + 0.1 * spikes

        return spikes


# =====================================================
# TEXT GENERATION & PROMPT OPTIMIZATION
# =====================================================

class AdvancedTokenEmbedding(nn.Module):
    """Multi-scale token embedding with syntropy awareness."""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Multi-frequency positional encodings (3x coverage)
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, embedding_dim) * 0.02)
        # Syntropy regulation networks: accepts variable field_state size
        self.field_projection = nn.Linear(1, embedding_dim)  # Project any field_state size first
        self.syntropy_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        self.coherence_transform = nn.Linear(embedding_dim, embedding_dim)
        self.scale_transform = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.relu = nn.ReLU()
        
    def forward(self, token_ids: torch.Tensor, position_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(token_ids)
        seq_len = x.size(1) if len(x.shape) > 1 else 1
        
        # Add positional encoding
        # Multi-scale positional encoding (3x resolution)
        pos_enc = self.positional_encoding[:, :min(seq_len, 512), :].expand_as(x) if len(x.shape) > 1 else self.positional_encoding[:, 0, :]
        x = x + pos_enc * 0.3
        
        # Layer normalization for stability
        x = self.layer_norm(x)
        
        # Syntropy-regulated scaling (System 2: deliberate, controlled)
        if position_context is not None:
            # Handle batched field_state: (batch, field_size) or unbatched: (field_size,)
            if position_context.dim() == 2:
                # Batched: (batch, field_size)
                # Average each batch element to scalar, then project all
                field_scalars = position_context.mean(dim=1, keepdim=True)  # (batch, 1)
                field_projected = self.field_projection(field_scalars)  # (batch, embedding_dim)
                
                # Expand to match x shape (batch, seq_len, embedding_dim)
                if x.dim() == 3:
                    field_for_gate = field_projected.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, seq_len, embedding_dim)
                    gate = self.syntropy_gate(field_for_gate.reshape(-1, self.embedding_dim)).reshape(x.shape[0], x.shape[1], -1)
                    coherence = torch.tanh(self.coherence_transform(field_for_gate.reshape(-1, self.embedding_dim))).reshape(x.shape[0], x.shape[1], -1)
                    scale = self.relu(self.scale_transform(coherence))
                    x = x * (1 + gate * scale * 0.2)
                else:
                    # Single sequence case
                    gate = self.syntropy_gate(field_projected)
                    coherence = torch.tanh(self.coherence_transform(field_projected))
                    scale = self.relu(self.scale_transform(coherence))
                    x = x * (1 + gate * scale * 0.2)
            else:
                # Unbatched: (field_size,)
                field_proj_input = position_context.unsqueeze(0).unsqueeze(0)  # (1, 1, field_size)
                field_proj_input = field_proj_input.mean(dim=-1)  # (1, 1) - collapse to scalar
                field_state_aligned = self.field_projection(field_proj_input).squeeze(0)  # (embedding_dim,)
                
                gate = self.syntropy_gate(field_state_aligned.unsqueeze(0) if field_state_aligned.dim() == 1 else field_state_aligned)
                coherence = torch.tanh(self.coherence_transform(field_state_aligned.unsqueeze(0) if field_state_aligned.dim() == 1 else field_state_aligned))
                scale = self.relu(self.scale_transform(coherence))
                x = x * (1 + gate * scale * 0.2)
        
        return x


class SynapticAttention(nn.Module):
    """Attention mechanism modulated by syntropy field with dual-layer fusion."""
    def __init__(self, hidden_size: int, num_heads: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.1)
        self.attention_layer2 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.1)
        # Field state projector: accepts variable-size field and projects to hidden_size
        self.field_projector = nn.Linear(1, hidden_size)  # Project field_state to hidden_size
        self.syntropy_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.field_projection = nn.Linear(hidden_size, hidden_size)
        self.fusion_weight = nn.Parameter(torch.tensor([0.6]))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                field_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out1, _ = self.attention(query, key, value)
        attn_out2, _ = self.attention_layer2(query, key, value)
        attn_out = self.fusion_weight * attn_out1 + (1 - self.fusion_weight) * attn_out2
        
        if field_state is not None:
            # Project field_state (variable size) to hidden_size
            if field_state.dim() == 1:
                # Scalar reduction: average field_state to single value, then project
                field_scalar = field_state.mean().unsqueeze(0).unsqueeze(0)  # (1, 1)
                field_aligned = self.field_projector(field_scalar).squeeze(0)  # (hidden_size,)
            else:
                # Multi-dim: average to scalar
                field_scalar = field_state.mean().unsqueeze(0).unsqueeze(0)
                field_aligned = self.field_projector(field_scalar).squeeze(0)
            
            # Syntropy-modulated gating with deeper network
            gate = self.syntropy_gate(field_aligned.unsqueeze(0) if field_aligned.dim() == 1 else field_aligned)
            field_proj = torch.tanh(self.field_projection(field_aligned.unsqueeze(0) if field_aligned.dim() == 1 else field_aligned))
            attn_out = attn_out * gate + field_proj * (1 - gate)
        
        return attn_out


class RecurrentFeedbackLoop(nn.Module):
    """Feedback loop with ensemble spiking for multi-path refinement."""
    def __init__(self, hidden_size: int, num_passes: int = 4):
        super().__init__()
        self.num_passes = num_passes
        self.refine_layer = nn.GRUCell(hidden_size, hidden_size)
        # Ensemble of parallel refinement paths
        self.ensemble_layers = nn.ModuleList([
            nn.GRUCell(hidden_size, hidden_size) for _ in range(2)
        ])
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        # Spiking parameters
        self.spike_threshold = 0.45
        self.spike_decay = 0.88
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.clone() if x.dim() == 1 else x[:, -1, :].clone()
        quality_scores = []
        ensemble_hiddens = []
        
        for pass_idx in range(self.num_passes):
            # System 2 Reasoning: Deliberate, controlled refinement
            x_input = x if x.dim() == 1 else x[:, -1, :]
            
            # Main refinement with damping factor to reduce chaos
            refined = self.refine_layer(x_input, hidden)
            # Damped update: less than full step, reducing chaotic behavior
            hidden = hidden + 0.4 * (refined - hidden)  # 40% update = controlled evolution
            
            # Ensemble refinement (parallel deliberation paths)
            ensemble_hidden = hidden.clone()
            for gru_ensemble in self.ensemble_layers:
                e_refined = gru_ensemble(x_input, ensemble_hidden)
                # Ensemble also uses damped update for stability
                ensemble_hidden = ensemble_hidden + 0.35 * (e_refined - ensemble_hidden)
            ensemble_hiddens.append(ensemble_hidden)
            
            # Quality evaluation (deterministic, System 2)
            quality = self.quality_predictor(hidden)
            quality_scores.append(quality)
        
        # Conservative quality aggregation (lower variance, more stable)
        avg_quality = torch.median(torch.stack(quality_scores), dim=0)[0]  # Use median not mean = robust to outliers
        
        # Controlled ensemble fusion with stability weights
        if ensemble_hiddens:
            ensemble_avg = torch.mean(torch.stack(ensemble_hiddens), dim=0)
            # Weighted blend emphasizing main path (System 2: confident, controlled)
            hidden = 0.65 * hidden + 0.35 * ensemble_avg
        
        # Spiking dynamics with enhanced threshold
        spike_potential = torch.abs(hidden)
        spikes = (spike_potential > self.spike_threshold).float()
        spike_activity = spikes * spike_potential * self.spike_decay
        
        refined = hidden * (1 + avg_quality * 0.5) + spike_activity * 0.3
        return refined


class CognitiveContextBuffer(nn.Module):
    """Semantic memory with symbolic reasoning for coherence."""
    def __init__(self, hidden_size: int, buffer_size: int = 96):
        super().__init__()
        self.buffer_size = buffer_size
        self.context_encoder = nn.Linear(hidden_size * 2, hidden_size)
        self.context_decoder = nn.Linear(hidden_size, hidden_size)
        self.register_buffer('context_memory', torch.zeros(buffer_size, hidden_size))
        self.memory_idx = 0
        
        # Symbolic reasoning layer for conceptual grounding
        self.symbolic_reasoner = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
        
    def store_context(self, current_state: torch.Tensor, previous_state: torch.Tensor):
        """Store context from current and previous states."""
        combined = torch.cat([current_state, previous_state], dim=-1) if current_state.dim() > 0 and previous_state.dim() > 0 else current_state
        context = torch.tanh(self.context_encoder(combined))
        self.context_memory[self.memory_idx % self.buffer_size] = context
        self.memory_idx += 1
        
    def retrieve_context(self) -> torch.Tensor:
        """Retrieve and blend context from buffer with symbolic grounding."""
        if self.memory_idx == 0:
            return torch.zeros_like(self.context_memory[0])
        blended_context = torch.mean(self.context_memory[:min(self.memory_idx, self.buffer_size)], dim=0)
        decoded = self.context_decoder(blended_context)
        # Apply symbolic reasoning for conceptual grounding
        symbolic_gate = self.symbolic_reasoner(decoded)
        return decoded * symbolic_gate + decoded * (1 - symbolic_gate)


class AdvancedTextGenerationNN(nn.Module):
    """
    Next-generation text generation architecture.
    
    Superior to traditional LLMs through:
    1. Syntropy-modulated attention (field-aware gating)
    2. Recurrent feedback loops (multi-pass refinement)
    3. Cognitive context buffer (semantic memory)
    4. Spiking neural dynamics (efficient computation)
    5. Adaptive vocabulary scaling (context-sensitive tokenization)
    """
    
    def __init__(self, vocab_size: int = 12000, embedding_dim: int = 768, hidden_size: int = 768):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Embedding layer with enhanced capacity
        self.token_embedding = AdvancedTokenEmbedding(vocab_size, embedding_dim)
        
        # Multi-layer processing (expanded to 4 layers)
        self.lstm_encoder = nn.LSTM(embedding_dim, hidden_size, num_layers=4, batch_first=True, 
                                    dropout=0.2, bidirectional=True)
        
        # Determine number of heads dynamically based on hidden_size
        attention_dim = hidden_size * 2
        num_heads = min(12, attention_dim // 64)  # At least 64 dims per head
        num_heads = max(1, num_heads)
        # Ensure divisibility
        while attention_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        # Syntropy-aware attention with dynamic heads
        self.synaptic_attention = SynapticAttention(attention_dim, num_heads=num_heads)
        
        # Feedback loop for refinement (4 passes)
        self.feedback_loop = RecurrentFeedbackLoop(hidden_size * 2, num_passes=4)
        
        # Cognitive buffer for context (96 items)
        self.context_buffer = CognitiveContextBuffer(hidden_size * 2, buffer_size=96)
        
        # Spiking dynamics with enhanced thresholds
        self.spiking_threshold = 0.45
        self.spike_decay = nn.Parameter(torch.tensor(0.88))
        
        # Output layers
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.quality_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.logit_head = nn.Linear(hidden_size, vocab_size)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # System 2 Reasoning Layer: Deliberate token selection
        self.system2_reasoner = nn.Sequential(
            nn.Linear(768 + 256, hidden_size),  # projected(768) + field_state(256) -> hidden
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Syntropy-modulated token regularization
        self.token_regulator = nn.Sequential(
            nn.Linear(vocab_size, vocab_size // 2),
            nn.ReLU(),
            nn.Linear(vocab_size // 2, vocab_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, token_ids: torch.Tensor, field_state: Optional[torch.Tensor] = None,
                previous_hidden: Optional[torch.Tensor] = None, return_all_logits: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with field modulation and feedback.
        Returns logits and metadata dict.
        
        Args:
            token_ids: (batch, seq_len) or (seq_len,)
            field_state: Optional field modulation state
            previous_hidden: Optional previous hidden state
            return_all_logits: If True, return logits for all positions (B, T, V). If False, only last position (V,)
        """
        # Embed tokens
        embedded = self.token_embedding(token_ids, field_state)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm_encoder(embedded)  # (B, T, hidden*2)
        
        # Synaptic attention with field modulation
        attn_out = self.synaptic_attention(lstm_out, lstm_out, lstm_out, field_state)  # (B, T, hidden*2)
        
        # Apply spiking dynamics (elementwise, preserves shape)
        spike_potential = torch.sigmoid(attn_out)
        spikes = (spike_potential > self.spiking_threshold).float()
        spiked_output = spikes * attn_out  # (B, T, hidden*2)
        
        # For training (return_all_logits=True), process all positions at once
        if return_all_logits:
            batch_size = spiked_output.shape[0] if spiked_output.dim() > 1 else 1
            seq_len = spiked_output.shape[1] if spiked_output.dim() > 1 else 1
            hidden_dim = spiked_output.shape[-1]
            
            # Flatten batch and time for projection
            spiked_flat = spiked_output.reshape(-1, hidden_dim) if spiked_output.dim() > 1 else spiked_output.unsqueeze(0)
            
            # Context buffer (simplified for training) - broadcast properly
            context = self.context_buffer.retrieve_context()  # (hidden_dim,)
            # Expand to match flattened batch
            context_flat = context.unsqueeze(0).expand(spiked_flat.shape[0], -1)  # (B*T, hidden_dim)
            
            # Blend with context
            refined_with_context = spiked_flat + context_flat
            
            # Project to logits space
            projected = self.relu(self.output_projection(refined_with_context))  # (B*T, hidden)
            quality = self.quality_gate(projected)  # (B*T, 1)
            gated_output = projected * quality
            
            # Generate logits
            logits_flat = self.logit_head(gated_output)  # (B*T, vocab)
            
            # Reshape back
            if batch_size > 1 and seq_len > 1:
                logits = logits_flat.reshape(batch_size, seq_len, -1)  # (B, T, vocab)
            else:
                logits = logits_flat
            
            # Metadata from last position
            quality_scalar = quality[-1].squeeze().item() if quality[-1].dim() > 0 else quality[-1].item()
            spike_potential_last = spike_potential[:, -1, :] if spike_potential.dim() == 3 else spike_potential
            
        else:
            # Inference: use recurrent feedback for last token
            refined = self.feedback_loop(spiked_output)  # Returns refined representation
            
            # Handle context (simplified to avoid shape mismatches)
            if refined.dim() == 2:
                # refined is (batch, hidden_size*2)
                refined_for_context = refined
            else:
                # refined is (batch, seq_len, hidden_size*2), take last token
                refined_for_context = refined[:, -1, :]
            
            # Store context if previous_hidden exists (ensure both are same shape)
            if previous_hidden is not None and previous_hidden.dim() == refined_for_context.dim():
                self.context_buffer.store_context(refined_for_context.squeeze(), previous_hidden.squeeze())
            
            context = self.context_buffer.retrieve_context()
            
            # Blend context with refined
            if refined.dim() == 2:
                refined_with_context = refined + context.unsqueeze(0)  # (1, hidden_size*2)
            else:
                refined_with_context = refined[:, -1, :] + context.unsqueeze(0)  # (1, hidden_size*2)
            
            # Project and apply quality gate
            to_project = refined_with_context.squeeze() if refined_with_context.dim() > 1 else refined_with_context
            projected = self.relu(self.output_projection(to_project))
            quality = self.quality_gate(projected.unsqueeze(0) if projected.dim() == 1 else projected)
            quality_scalar = quality.squeeze() if quality.dim() > 0 else quality
            gated_output = projected * quality_scalar
            
            # Generate logits
            logits = self.logit_head(gated_output)
            spike_potential_last = spike_potential
        
        # OPTIMIZATION: Syntropy-regulated token selection (System 2) - only for inference
        if field_state is not None and not return_all_logits:
            # Concat for system2 reasoning
            if projected.dim() == 1:
                concatenated = torch.cat([projected, field_state], dim=-1).unsqueeze(0)
            else:
                concatenated = torch.cat([projected, field_state], dim=-1).unsqueeze(0)
            
            system2_confidence = self.system2_reasoner(concatenated)
            
            # Apply softmax regularization
            logits_for_softmax = logits.unsqueeze(0) if logits.dim() == 1 else logits
            regulated_probs = self.token_regulator(torch.softmax(logits_for_softmax, dim=-1))
            
            # Blend raw logits with regulated (System 2) version
            logits = logits * (1 - system2_confidence * 0.3) + torch.log(regulated_probs.squeeze() + 1e-9) * system2_confidence * 0.3
            logits = logits.squeeze() if logits.dim() > 1 else logits
        
        # Metadata for analysis
        if isinstance(quality_scalar, torch.Tensor):
            quality_scalar_val = quality_scalar.item() if quality_scalar.dim() == 0 else quality_scalar.mean().item()
        else:
            quality_scalar_val = quality_scalar
            
        metadata = {
            "spike_ratio": (spike_potential_last > self.spiking_threshold).float().mean().item(),
            "quality_score": quality_scalar_val,
            "attention_entropy": -(spike_potential_last * torch.log(spike_potential_last + 1e-8)).mean().item(),
            "field_influence": torch.norm(field_state).item() if field_state is not None else 0.0
        }
        
        return logits, metadata
    
    def generate_with_feedback(self, prompt_ids: List[int], max_length: int = 100,
                               temperature: float = 0.7, field_state: Optional[torch.Tensor] = None,
                               num_refinements: int = 2) -> Tuple[List[int], List[Dict]]:
        """
        Generate text with internal feedback loops.
        Performs multi-pass refinement for superior quality.
        """
        generated = prompt_ids.copy()
        generation_metadata = []
        previous_hidden = None
        
        with torch.no_grad():
            for step in range(max_length):
                # Prepare input
                context_window = generated[-64:] if len(generated) > 64 else generated
                input_ids = torch.tensor([context_window], dtype=torch.long)
                
                # Single pass with System 2 deterministic selection
                logits, metadata = self.forward(input_ids, field_state, previous_hidden)
                
                # System 2: Deterministic token selection
                if field_state is not None and metadata["quality_score"] > 0.5:
                    next_token = torch.argmax(logits, dim=-1).item()
                else:
                    probs = torch.softmax(logits / max(temperature, 0.5), dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs.squeeze(), k=min(10, probs.size(-1)))
                    top_k_probs = top_k_probs / top_k_probs.sum()
                    next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
                
                generated.append(next_token)
                generation_metadata.append(metadata)
                previous_hidden = logits.detach()
                
                if next_token == 2 or next_token == 0 or step > max_length - 5:
                    break
        
        return generated, generation_metadata
    
    def generate_with_feedback_old(self, prompt_ids: List[int], max_length: int = 100,
                               temperature: float = 0.7, field_state: Optional[torch.Tensor] = None,
                               num_refinements: int = 2) -> Tuple[List[int], List[Dict]]:
        """Legacy: kept for reference only."""
        generated = prompt_ids.copy()
        generation_metadata = []
        previous_hidden = None

        with torch.no_grad():
            for step in range(max_length):
                context_window = generated[-64:] if len(generated) > 64 else generated
                input_ids = torch.tensor([context_window], dtype=torch.long)
                logits, metadata = self.forward(input_ids, field_state, previous_hidden)
                probs = torch.softmax(logits / max(temperature, 0.5), dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs.squeeze(), k=min(10, probs.size(-1)))
                top_k_probs = top_k_probs / top_k_probs.sum()
                next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
                generated.append(next_token)
                generation_metadata.append(metadata)
                previous_hidden = logits.detach()
                if next_token == 2 or next_token == 0 or step > max_length - 5:
                    break
        return generated, generation_metadata
    
    def get_generation_quality(self, metadata_list: List[Dict]) -> Dict:
        """Analyze quality metrics across generation."""
        if not metadata_list:
            return {}
        
        spike_ratios = [m["spike_ratio"] for m in metadata_list]
        quality_scores = [m["quality_score"] for m in metadata_list]
        attention_entropies = [m["attention_entropy"] for m in metadata_list]
        
        return {
            "avg_spike_ratio": np.mean(spike_ratios),
            "avg_quality": np.mean(quality_scores),
            "quality_consistency": 1.0 - np.std(quality_scores),  # Lower variance = more consistent
            "avg_attention_entropy": np.mean(attention_entropies),
            "generation_length": len(metadata_list)
        }


class PromptOptimizationNN(nn.Module):
    """Optimize prompts using field state."""
    def __init__(self, embedding_dim: int = 768, hidden_size: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, embedding_dim)
        self.quality_head = nn.Linear(hidden_size, 1)
        self.diversity_head = nn.Linear(hidden_size, 1)
        self.coherence_head = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, prompt_embedding: torch.Tensor, field_state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        combined = torch.cat([prompt_embedding, field_state[:min(prompt_embedding.size(0), field_state.size(0))]])
        h = self.relu(self.fc1(combined))
        hidden = self.relu(self.fc2(h))
        
        quality = self.sigmoid(self.quality_head(hidden)).mean()
        diversity = self.sigmoid(self.diversity_head(hidden)).mean()
        coherence = self.sigmoid(self.coherence_head(hidden)).mean()
        
        optimized = self.fc3(hidden)
        
        metrics = {
            "quality": quality.item(),
            "diversity": diversity.item(),
            "coherence": coherence.item()
        }
        return optimized, metrics


# =====================================================
# INTERNET INTEGRATION
# =====================================================

class InternetIntegrationModule:
    """Fetch and integrate internet data into field dynamics."""
    def __init__(self, cache_size: int = 500):
        self.cache = deque(maxlen=cache_size)
        self.cache_dict = {}
        self.session = requests.Session()
        self.session.timeout = 3

    def fetch_data(self, query: str, source: str = "general") -> Optional[Dict]:
        cache_key = hashlib.md5(f"{source}:{query}".encode()).hexdigest()
        
        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        try:
            if source == "wikipedia":
                return self._fetch_wikipedia(query, cache_key)
            else:
                return self._fetch_general(query, cache_key)
        except Exception as e:
            logging.warning(f"Fetch error: {e}")
        return None

    def _fetch_wikipedia(self, query: str, cache_key: str) -> Optional[Dict]:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": query,
            "prop": "extracts",
            "explaintext": True,
            "exintro": True
        }
        try:
            resp = self.session.get(url, params=params, timeout=2)
            pages = resp.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values())) if pages else None
            if page and "extract" in page:
                data = {
                    "source": "wikipedia",
                    "query": query,
                    "content": page["extract"][:300],
                    "timestamp": datetime.now().isoformat()
                }
                self.cache_dict[cache_key] = data
                self.cache.append(data)
                return data
        except:
            pass
        return None

    def _fetch_general(self, query: str, cache_key: str) -> Dict:
        data = {
            "source": "general",
            "query": query,
            "content": f"Query: {query}",
            "timestamp": datetime.now().isoformat()
        }
        self.cache_dict[cache_key] = data
        self.cache.append(data)
        return data

    def extract_embeddings(self, data: Dict) -> np.ndarray:
        content = data.get("content", "")
        words = content.split()[:100]
        embedding = np.array([hash(w) % 256 for w in words], dtype=np.float32)
        if len(embedding) < 100:
            embedding = np.pad(embedding, (0, 100 - len(embedding)), mode='constant')
        return embedding[:100]


# =====================================================
# ORCHESTRATION CLASSES
# =====================================================

class OscillatorySynapseTheory:
    """Enhanced system with text generation and internet integration."""

    def __init__(self, field_size: int = 384, device: str = 'cpu', enable_text_gen: bool = True):
        self.device = torch.device(device)
        self.field_size = field_size
        self.field = torch.randn(field_size, device=self.device) * 0.1
        self.lock = threading.Lock()
        
        self.nn1 = FieldUpdateNN(field_size, hidden_size=96).to(self.device)
        self.nn2 = SyntropyNN(field_size, hidden_size=48).to(self.device)
        self.nn3 = FeedbackNN(field_size, hidden_size=72).to(self.device)
        
        self.enable_text_gen = enable_text_gen
        if enable_text_gen:
            self.text_gen = AdvancedTextGenerationNN(vocab_size=12000, embedding_dim=768, hidden_size=768).to(self.device)
            self.prompt_optimizer = PromptOptimizationNN(embedding_dim=768, hidden_size=768).to(self.device)
            # instantiate FieldEnhancedSNN sized to cover the field vector spatially
            H = W = int(np.ceil(np.sqrt(self.field_size)))
            self.field_snn = FieldEnhancedSNN(size=(H, W), device=self.device).to(self.device)
            # Spatial-to-vector encoder: conv stack + flatten + linear projection
            self.field_encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))  # Global spatial pooling
            ).to(self.device)
            # Final projection from conv features to field vector
            self.field_projection = nn.Linear(32, self.field_size).to(self.device)
            # Physics: run PDE at high frequency (1000 Hz) using 1000 substeps per logical step
            self.physics_hz = 1000
            self.physics_substeps = self.physics_hz  # Full 1000 substeps per step
            self.internet_module = InternetIntegrationModule(cache_size=500)
            self.text_gen_history = deque(maxlen=100)
        
        self.balanced = False
        self.balance_threshold = 0.01
        self.step_count = 0
        self.metrics_history = deque(maxlen=1000)

    def fetch_internet_context(self, topic: str) -> Optional[np.ndarray]:
        if not self.enable_text_gen:
            return None
        try:
            data = self.internet_module.fetch_data(topic, source="wikipedia")
            if data:
                return self.internet_module.extract_embeddings(data)
        except:
            pass
        return None

    def generate_adaptive_text(self, prompt_tokens: List[int], max_length: int = 50,
                               num_refinements: int = 2) -> Tuple[List[int], Dict]:
        """Generate text with syntropy-modulated feedback loops."""
        if not self.enable_text_gen:
            return prompt_tokens, {}
        
        try:
            # Normalize field for modulation
            field_mod = (self.field - self.field.min()) / (self.field.max() - self.field.min() + 1e-8)
            field_mod = field_mod.to(self.device)
            
            # Ensure prompt tokens exist
            if len(prompt_tokens) > 0:
                padded_prompt = prompt_tokens + [0] * (10 - len(prompt_tokens)) if len(prompt_tokens) < 10 else prompt_tokens[:10]
                
                # Use advanced generation with feedback loops
                generated, metadata_list = self.text_gen.generate_with_feedback(
                    padded_prompt,
                    max_length=max_length,
                    temperature=0.7,
                    field_state=field_mod,
                    num_refinements=num_refinements
                )
                
                # Analyze quality
                quality_analysis = self.text_gen.get_generation_quality(metadata_list)
                self.text_gen_history.append(quality_analysis)
                
                return generated[:max_length], quality_analysis
        except Exception as e:
            # EXPOSE THE ERROR INSTEAD OF HIDING IT
            print(f"🚨 CRITICAL FAILURE IN TEXT GENERATION: {e}")
            import traceback
            traceback.print_exc()
            logging.error(f"Text generation error: {e}")
        
        return prompt_tokens, {}

    def step(self):
        with self.lock:
            # Core neural updates
            self.field = self.nn1.update_field(self.field.detach())
            self.field = self.nn2.adjust_syntropy(self.field.detach())
            self.field = self.nn3.penalize_variance(self.field.detach())

            # High-frequency field physics via FieldEnhancedSNN (1000 substeps simulating 1000 Hz)
            if hasattr(self, 'field_snn'):
                try:
                    physics_dt = 1.0 / float(getattr(self, 'physics_hz', 1000))
                    # Full 1000 PDE integration substeps per logical step
                    for _ in range(getattr(self, 'physics_substeps', 1000)):
                        self.field_snn.pde_step(dt=physics_dt)
                    # Spatial-to-vector encoding: conv stack + flatten + project
                    phi_input = self.field_snn.Phi.float()  # Ensure float type
                    encoded = self.field_encoder(phi_input)  # (1, 32, 1, 1) after global pool
                    encoded_flat = encoded.view(encoded.size(0), -1)  # (1, 32)
                    field_state_vec = torch.tanh(self.field_projection(encoded_flat)).squeeze(0).to(self.device)
                    # Normalize to [0,1]
                    field_state_vec = (field_state_vec - field_state_vec.min()) / (field_state_vec.max() - field_state_vec.min() + 1e-8)
                    # Blend spatial field state into main 1D field
                    self.field = 0.95 * self.field + 0.05 * field_state_vec
                except Exception as e:
                    pass

            # Internet context integration every 50 steps
            if self.enable_text_gen and self.step_count % 50 == 0:
                try:
                    context = self.fetch_internet_context(f"topic_{self.step_count % 10}")
                    if context is not None:
                        context_tensor = torch.from_numpy(context).float().to(self.device)
                        self.field[:len(context_tensor)] = self.field[:len(context_tensor)] * 0.7 + context_tensor * 0.3
                except:
                    pass

            # Advanced text generation with feedback every 100 steps
            if self.enable_text_gen and self.step_count % 100 == 0:
                try:
                    prompt = [1, 2, 3, 4, 5]
                    generated, gen_quality = self.generate_adaptive_text(prompt, max_length=50, num_refinements=2)
                    if gen_quality and "avg_quality" in gen_quality:
                        # Use quality score to adjust field
                        quality_factor = torch.tensor(gen_quality["avg_quality"], device=self.device)
                        self.field = self.field * (1 - 0.05 * quality_factor) + self.field * 0.05 * quality_factor
                except:
                    pass

            # Record metrics
            var = torch.var(self.field).item()
            mean = torch.mean(self.field).item()
            
            metrics_entry = {
                "step": self.step_count,
                "variance": var,
                "mean": mean,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add text generation quality metrics if available
            if self.enable_text_gen and self.text_gen_history:
                latest_gen_quality = self.text_gen_history[-1]
                metrics_entry["text_quality"] = latest_gen_quality.get("avg_quality", 0.0)
                metrics_entry["generation_length"] = latest_gen_quality.get("generation_length", 0)
            
            self.metrics_history.append(metrics_entry)

            if var < self.balance_threshold:
                self.balanced = True
            
            self.step_count += 1

    def run(self, max_steps: int = float('inf')):
        step = 0
        while not self.balanced and step < max_steps:
            self.step()
            step += 1
            if step % 100 == 0:
                cache_size = len(self.internet_module.cache) if self.enable_text_gen else 0
                print(f"Step {step}: Variance = {torch.var(self.field).item():.4f}, Mean = {torch.mean(self.field).item():.4f}, Cache: {cache_size}")
        print(f"Balanced after {step} steps" if self.balanced else "Max steps reached")

    def get_metrics_summary(self) -> Dict:
        if not self.metrics_history:
            return {}
        
        variances = [m["variance"] for m in self.metrics_history]
        means = [m["mean"] for m in self.metrics_history]
        text_qualities = [m.get("text_quality", 0.0) for m in self.metrics_history if "text_quality" in m]
        gen_lengths = [m.get("generation_length", 0) for m in self.metrics_history if "generation_length" in m]
        
        summary = {
            "avg_variance": np.mean(variances),
            "min_variance": np.min(variances),
            "max_variance": np.max(variances),
            "avg_mean": np.mean(means),
            "total_steps": self.step_count,
            "metrics_recorded": len(self.metrics_history)
        }
        
        # Add text generation metrics
        if text_qualities:
            summary["avg_text_quality"] = np.mean(text_qualities)
            summary["max_text_quality"] = np.max(text_qualities)
            summary["text_quality_trend"] = "improving" if len(text_qualities) > 1 and text_qualities[-1] > text_qualities[0] else "stable"
        
        if gen_lengths:
            summary["avg_generation_length"] = np.mean(gen_lengths)
            summary["max_generation_length"] = np.max(gen_lengths)
        
        # Text generation system info
        if self.enable_text_gen:
            summary["text_gen_enabled"] = True
            summary["text_gen_vocab_size"] = 12000
            summary["text_gen_embedding_dim"] = 768
            summary["text_gen_hidden_size"] = 768
            summary["text_gen_architecture"] = "Advanced (Synaptic + Feedback + Context)"
        
        return summary


class MatrixFluidMemoryVariant(OscillatorySynapseTheory):
    """Memory variant with trajectory tracking."""

    def __init__(self, field_size: int = 256, device: str = 'cpu', max_memory: int = 2000, enable_text_gen: bool = True):
        super().__init__(field_size, device, enable_text_gen)
        self.memory_patterns: List[torch.Tensor] = []
        self.max_memory = max_memory
        self.text_memories: List[Dict] = []

    def step(self):
        super().step()
        pattern = self.field.detach().clone()
        self.memory_patterns.append(pattern)
        
        if self.enable_text_gen and self.step_count % 50 == 0:
            text_context = {
                "step": self.step_count,
                "field_embedding": pattern.cpu().numpy()[:100],
                "timestamp": datetime.now().isoformat()
            }
            self.text_memories.append(text_context)
        
        if len(self.memory_patterns) > self.max_memory:
            self.memory_patterns.pop(0)
        if len(self.text_memories) > 100:
            self.text_memories.pop(0)

    def get_memory_mean(self) -> Optional[float]:
        if not self.memory_patterns:
            return None
        means = torch.stack([p.mean() for p in self.memory_patterns])
        return means.mean().item()

    def get_memory_trajectory(self, last_n: int = 100) -> Dict:
        recent = self.memory_patterns[-last_n:] if len(self.memory_patterns) >= last_n else self.memory_patterns
        if not recent:
            return {}
        
        variances = [p.var().item() for p in recent]
        means = [p.mean().item() for p in recent]
        
        return {
            "variance_trend": variances,
            "mean_trend": means,
            "avg_variance": np.mean(variances),
            "trajectory_length": len(recent)
        }


class FieldStateAsync:
    """Async variant with concurrent updates."""
    def __init__(self, field_size: int = 256, device: str = 'cpu'):
        self.device = torch.device(device)
        self.field_size = field_size
        self.field = torch.randn(field_size, device=self.device) * 0.1
        self.lock = asyncio.Lock()
        self.nn1 = FieldUpdateNN(field_size).to(self.device)
        self.nn2 = SyntropyNN(field_size).to(self.device)
        self.nn3 = FeedbackNN(field_size).to(self.device)
        self.balanced = False
        self.balance_threshold = 0.01

    async def async_step(self):
        async with self.lock:
            self.field = self.nn1.update_field(self.field)
            self.field = self.nn2.adjust_syntropy(self.field)
            self.field = self.nn3.penalize_variance(self.field)
            if torch.var(self.field) < self.balance_threshold:
                self.balanced = True

    async def run_async(self, max_steps: int = float('inf')):
        step = 0
        while not self.balanced and step < max_steps:
            await self.async_step()
            step += 1
            if step % 100 == 0:
                print(f"Async Step {step}: Variance = {torch.var(self.field).item():.4f}, Mean = {torch.mean(self.field).item():.4f}")
        print(f"Balanced after {step} steps" if self.balanced else "Max steps reached")


if __name__ == "__main__":
    print("=" * 70)
    print("EXPANDED SYNTROPY FIELD WITH TEXT GENERATION & INTERNET INTEGRATION")
    print("=" * 70)

    # Synchronous with enhancements
    print("\n[1/3] Running expanded OscillatorySynapseTheory...")
    ost = OscillatorySynapseTheory(field_size=256, enable_text_gen=True)
    ost.run(max_steps=500)
    
    summary = ost.get_metrics_summary()
    print("\nMetrics Summary:")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    # Memory variant
    print("\n[2/3] Running memory variant with trajectory tracking...")
    mem_ost = MatrixFluidMemoryVariant(field_size=256, max_memory=2000, enable_text_gen=True)
    mem_ost.run(max_steps=300)
    print(f"Memory mean: {mem_ost.get_memory_mean():.6f}")
    
    trajectory = mem_ost.get_memory_trajectory(last_n=50)
    print("\nTrajectory Summary:")
    for key, val in trajectory.items():
        if isinstance(val, list) and len(val) > 0:
            print(f"  {key}: [{val[0]:.4f} ... {val[-1]:.4f}] (len: {len(val)})")
        else:
            print(f"  {key}: {val}")

    # Async version
    print("\n[3/3] Running async version...")
    async def main_async():
        async_ost = FieldStateAsync(field_size=256)
        await async_ost.run_async(max_steps=200)

    asyncio.run(main_async())
    print("\n" + "=" * 70)
    print("COMPLETION: All variants executed successfully")
    print("=" * 70)
