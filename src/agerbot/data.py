"""Carga de corpus y muestreo de ventanas para predicción del siguiente byte."""

from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import ByteTokenizer, CharTokenizer


def load_corpus(
    path: str | Path, tokenizer: ByteTokenizer | CharTokenizer
) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    if not tokens:
        raise ValueError(f"El corpus está vacío: {path}")
    return torch.tensor(tokens, dtype=torch.long)


def split_corpus(tokens: torch.Tensor, train_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction debe estar entre 0.5 y 1.0")
    boundary = int(len(tokens) * train_fraction)
    return tokens[:boundary], tokens[boundary:]


def random_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= context_length:
        raise ValueError(
            f"Se necesitan más de {context_length} tokens; solo hay {len(tokens)}"
        )
    starts = torch.randint(0, len(tokens) - context_length, (batch_size,))
    inputs = torch.stack([tokens[start : start + context_length] for start in starts])
    targets = torch.stack(
        [tokens[start + 1 : start + context_length + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)
