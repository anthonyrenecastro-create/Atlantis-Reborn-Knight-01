# hot_memory.py
import torch
import json
import time
import uuid
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Optional

@dataclass
class AtlanteanHotMemory:
    phi1: torch.Tensor   # Excitability field
    phi5: torch.Tensor   # Plasticity / entropy field
    Phi: torch.Tensor    # Global meaning potential
    Theta: Dict          # Learned modulation parameters
    schema_version: int = 1
    identity_fingerprint: Optional[str] = None  # Cryptographic identity
    device_id: Optional[str] = None  # Device/instance identifier
    version: int = 0  # Local version counter for sync
    last_update: float = 0.0  # Timestamp of last modification

    @staticmethod
    def initialize(grid_size=(32, 32), identity=None, device_id=None):
        """
        Initialize a new Atlantean hot memory instance.
        
        Args:
            grid_size: Tuple of (height, width) for field grids
            identity: Optional AtlanteanIdentity for cryptographic signing
            device_id: Optional device identifier (auto-generated if not provided)
            
        Returns:
            AtlanteanHotMemory instance
        """
        fingerprint = identity.fingerprint() if identity else None
        device = device_id or str(uuid.uuid4())

        # Deterministic initialization seed allows reproducible replay from genesis.
        seed_source = f"{fingerprint or 'no_identity'}::{device}"
        seed = int.from_bytes(hashlib.sha256(seed_source.encode('utf-8')).digest()[:8], byteorder='big')
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        return AtlanteanHotMemory(
            phi1=torch.randn(grid_size, generator=generator),
            phi5=torch.ones(grid_size) * 0.1,
            Phi=torch.zeros(1),
            Theta={},
            identity_fingerprint=fingerprint,
            device_id=device,
            version=0,
            last_update=time.time()
        )

    # ---------- Versioned State Management ----------

    def snapshot(self):
        """
        Create a versioned snapshot of the current state.
        
        This is critical for sync: every snapshot is immutable and timestamped.
        
        Returns:
            Dict with all state + metadata
        """
        return {
            "phi1": self.phi1.clone(),
            "phi5": self.phi5.clone(),
            "Phi": self.Phi.clone(),
            "Theta": self.Theta.copy(),
            "device_id": self.device_id,
            "version": self.version,
            "timestamp": self.last_update,
            "identity_fingerprint": self.identity_fingerprint,
            "schema_version": self.schema_version
        }

    def apply_local_update(self):
        """
        Mark that a local update has occurred.
        
        Increments version counter and updates timestamp.
        Call this after any modification to φ₁, φ₅, Φ, or Θ.
        
        CRITICAL: This is how sync knows what changed.
        """
        self.version += 1
        self.last_update = time.time()

    def update_phi1(self, delta):
        """Update excitability field and increment version."""
        self.phi1 += delta
        self.apply_local_update()

    def update_phi5(self, delta):
        """Update plasticity field and increment version."""
        self.phi5 += delta
        self.apply_local_update()

    def update_Phi(self, delta):
        """Update global meaning potential and increment version."""
        self.Phi += delta
        self.apply_local_update()

    def update_Theta(self, key, value):
        """Update modulation parameters and increment version."""
        self.Theta[key] = value
        self.apply_local_update()

    # ---------- Persistence (NON-NEGOTIABLE) ----------

    def save(self, path: str, identity=None):
        """
        Save hot memory state with optional cryptographic signature.
        
        Args:
            path: File path to save to
            identity: Optional AtlanteanIdentity for signing the save
        """
        save_data = {
            "phi1": self.phi1,
            "phi5": self.phi5,
            "Phi": self.Phi,
            "Theta": self.Theta,
            "schema_version": self.schema_version,
            "identity_fingerprint": self.identity_fingerprint,
            "device_id": self.device_id,
            "version": self.version,
            "last_update": self.last_update
        }
        
        # Add signature if identity provided
        if identity:
            state_bytes = json.dumps({
                "phi1": self.phi1.tolist(),
                "phi5": self.phi5.tolist(),
                "Phi": self.Phi.tolist(),
                "Theta": self.Theta
            }, sort_keys=True).encode('utf-8')
            signature = identity.sign(state_bytes)
            save_data["signature"] = signature
        
        torch.save(save_data, path)

    @staticmethod
    def load(path: str, verify_identity=None):
        """
        Load hot memory state with optional signature verification.
        
        Args:
            path: File path to load from
            verify_identity: Optional AtlanteanIdentity to verify signature
            
        Returns:
            AtlanteanHotMemory instance
            
        Raises:
            ValueError: If signature verification fails
        """
        data = torch.load(path)
        
        # Verify signature if requested
        if verify_identity and "signature" in data:
            state_bytes = json.dumps({
                "phi1": data["phi1"].tolist(),
                "phi5": data["phi5"].tolist(),
                "Phi": data["Phi"].tolist(),
                "Theta": data["Theta"]
            }, sort_keys=True).encode('utf-8')
            
            if not verify_identity.verify(state_bytes, data["signature"]):
                raise ValueError("Signature verification failed! Core state may be corrupted.")
        
        return AtlanteanHotMemory(
            phi1=data["phi1"],
            phi5=data["phi5"],
            Phi=data["Phi"],
            Theta=data["Theta"],
            schema_version=data.get("schema_version", 1),
            identity_fingerprint=data.get("identity_fingerprint"),
            device_id=data.get("device_id"),
            version=data.get("version", 0),
            last_update=data.get("last_update", time.time())
        )
