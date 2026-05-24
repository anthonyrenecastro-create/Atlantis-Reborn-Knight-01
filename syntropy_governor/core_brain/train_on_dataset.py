#!/usr/bin/env python3
"""
Sovereign dataset trainer for AdvancedTextGenerationNN.

Input formats supported per JSONL row:
- {"instruction": "...", "output": "..."}
- {"prompt": "...", "response": "..."}
- {"text": "..."}

This script fine-tunes a checkpoint without any external APIs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from syntropy_field_expanded import AdvancedTextGenerationNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train core_brain on local dataset JSONL")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--checkpoint",
        default="core_brain/shakespeare_model.pt",
        help="Input model checkpoint path",
    )
    parser.add_argument(
        "--output",
        default="core_brain/shakespeare_model_sovereign.pt",
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap on rows read")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or cuda")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    return parser.parse_args()


def _row_to_text(row: Dict) -> str:
    if not isinstance(row, dict):
        return ""

    if isinstance(row.get("instruction"), str) and isinstance(row.get("output"), str):
        return f"User: {row['instruction']}\nAssistant: {row['output']}"
    if isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str):
        return f"User: {row['prompt']}\nAssistant: {row['response']}"
    if isinstance(row.get("text"), str):
        return row["text"]
    return ""


def load_text_rows(path: Path, max_rows: int = 0) -> List[str]:
    rows: List[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = _row_to_text(obj).strip()
            if len(text) >= 4:
                rows.append(text)

            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def encode_text(text: str, stoi: Dict[str, int]) -> List[int]:
    unk = stoi.get("?", 0)
    return [stoi.get(ch, unk) for ch in text]


def build_training_tensors(
    texts: Iterable[str],
    stoi: Dict[str, int],
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[List[int]] = []
    ys: List[List[int]] = []

    for text in texts:
        token_ids = encode_text(text, stoi)
        if len(token_ids) < 2:
            continue

        # Sliding windows with stride half of sequence for better reuse.
        stride = max(8, seq_len // 2)
        for start in range(0, len(token_ids) - 1, stride):
            window = token_ids[start : start + seq_len + 1]
            if len(window) < 2:
                continue

            if len(window) < seq_len + 1:
                window = window + [0] * (seq_len + 1 - len(window))

            x = window[:-1]
            y = window[1:]
            xs.append(x)
            ys.append(y)

    if not xs:
        raise ValueError("No trainable samples could be built from dataset.")

    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor


def resolve_checkpoint_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    # Allow invocation from repo root or from core_brain dir.
    repo_candidate = base_dir.parent / path
    if repo_candidate.exists():
        return repo_candidate
    return base_dir / path.name if (base_dir / path.name).exists() else repo_candidate


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    checkpoint_path = resolve_checkpoint_path(base_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab_size = int(checkpoint.get("vocab_size", 66))
    embedding_dim = int(checkpoint.get("embedding_dim", 128))
    hidden_size = int(checkpoint.get("hidden_size", 128))
    stoi = checkpoint.get("stoi", {})
    itos = checkpoint.get("itos", {})

    if not stoi:
        # Basic printable ASCII fallback if checkpoint omitted token maps.
        stoi = {chr(i): i for i in range(32, 127)}
        itos = {i: chr(i) for i in range(32, 127)}

    texts = load_text_rows(dataset_path, max_rows=max(0, args.max_rows))
    if not texts:
        raise ValueError("Dataset is empty or did not contain supported JSONL records.")

    x_train, y_train = build_training_tensors(texts, stoi, seq_len=max(16, args.seq_len))

    model = AdvancedTextGenerationNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device)
    model.train()

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training samples: {len(dataset)}")
    print(f"Device: {device}")

    for epoch in range(max(1, args.epochs)):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_x, return_all_logits=True)

            if logits.dim() == 2:
                # Fallback shape guard.
                logits = logits.view(batch_x.shape[0], batch_x.shape[1], -1)

            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(loader))
        ppl = math.exp(min(20.0, avg_loss))
        print(f"Epoch {epoch + 1}/{args.epochs} - loss={avg_loss:.4f} - ppl={ppl:.2f}")

    out_checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "stoi": stoi,
        "itos": itos,
        "training_source": str(dataset_path),
        "training_rows": len(texts),
        "training_samples": len(dataset),
    }
    torch.save(out_checkpoint, output_path)
    print(f"Saved fine-tuned checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
