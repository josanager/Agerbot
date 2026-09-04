"""Utilidades compartidas de dispositivo, configuración y checkpoints."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_state_to_float16(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Serializa pesos del modelo en float16 para checkpoints compactos."""
    compact: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if torch.is_floating_point(value):
            compact[key] = value.detach().to(dtype=torch.float16).cpu()
        else:
            compact[key] = value.detach().cpu()
    return compact


def model_state_to_float32(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Carga pesos (posiblemente FP16) como float32 para entrenar/inferir en CPU."""
    restored: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if torch.is_floating_point(value) and value.dtype != torch.float32:
            restored[key] = value.to(dtype=torch.float32)
        else:
            restored[key] = value
    return restored


def save_checkpoint(
    path: Path,
    payload: dict,
    *,
    store_model_float16: bool = True,
    include_optimizer: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_save = dict(payload)
    if store_model_float16 and "model_state" in to_save:
        to_save["model_state"] = model_state_to_float16(to_save["model_state"])
        to_save["model_dtype"] = "float16"
    if not include_optimizer:
        to_save.pop("optimizer_state", None)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(to_save, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    if "model_state" in checkpoint:
        checkpoint["model_state"] = model_state_to_float32(checkpoint["model_state"])
    return checkpoint
