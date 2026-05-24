#!/usr/bin/env python3
"""
Prepare UltraChat dataset into JSONL records accepted by core_brain/train_on_dataset.py.

Output format (one row per assistant reply):
- {"instruction": "...", "output": "...", "source": "ultrachat"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UltraChat dataset for sovereign training")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k", help="HF dataset repo")
    parser.add_argument("--split", default="train_sft", help="Dataset split (e.g., train_sft, train_gen)")
    parser.add_argument("--output", default="unified_backend/exports/ultrachat_train.jsonl", help="Output JSONL path")
    parser.add_argument("--max-conversations", type=int, default=2000, help="Maximum conversations to process")
    parser.add_argument("--max-rows", type=int, default=12000, help="Maximum output instruction/output rows")
    return parser.parse_args()


def _normalize_role(role: Optional[str]) -> str:
    if not role:
        return ""
    role = role.lower().strip()
    if role in {"user", "human", "instruction", "prompt"}:
        return "user"
    if role in {"assistant", "bot", "gpt", "model"}:
        return "assistant"
    return role


def _build_pairs(messages: Iterable[dict]) -> List[dict]:
    rows: List[dict] = []
    pending_user: Optional[str] = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = _normalize_role(str(msg.get("role", "")))
        content = str(msg.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user:
            rows.append(
                {
                    "instruction": pending_user,
                    "output": content,
                    "source": "ultrachat",
                }
            )
            pending_user = None

    return rows


def _extract_messages(item: dict) -> Optional[Iterable[dict]]:
    if isinstance(item.get("messages"), list):
        return item["messages"]
    if isinstance(item.get("conversation"), list):
        return item["conversation"]
    if isinstance(item.get("dialog"), list):
        return item["dialog"]
    return None


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package not installed. Run: pip install datasets") from exc

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)

    conv_count = 0
    row_count = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for item in ds:
            messages = _extract_messages(item)
            if not messages:
                continue

            pairs = _build_pairs(messages)
            for row in pairs:
                fp.write(json.dumps(row, ensure_ascii=True) + "\n")
                row_count += 1
                if row_count >= max(1, args.max_rows):
                    break

            conv_count += 1
            if conv_count >= max(1, args.max_conversations) or row_count >= max(1, args.max_rows):
                break

    print(f"Wrote {row_count} rows from {conv_count} conversations to {out_path}")


if __name__ == "__main__":
    main()
