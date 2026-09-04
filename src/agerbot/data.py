"""Carga de corpus, multi-target de respuestas existentes y muestreo LM."""

from __future__ import annotations

import random
import re
from pathlib import Path

import torch

from .tokenizer import TokenizerAny

USER_MARKERS = ("Usuario:", "Pregunta:", "Consulta:", "Chat:", "Interacción:")
ASSISTANT_MARKERS = ("Agerbot:",)


def _strip_marker(line: str, markers: tuple[str, ...]) -> str | None:
    for marker in markers:
        if line.startswith(marker):
            return line[len(marker) :].strip()
    return None


def parse_dialogue_pairs(text: str) -> list[tuple[str, str]]:
    """Extrae pares (usuario, asistente) de marcadores existentes en el corpus."""
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        user = _strip_marker(line, USER_MARKERS)
        if user is not None:
            pending_user = user
            continue
        assistant = _strip_marker(line, ASSISTANT_MARKERS)
        if assistant is not None and pending_user is not None:
            pairs.append((pending_user, assistant))
            pending_user = None
    return pairs


def normalize_user_intent(text: str) -> str:
    """Agrupa paráfrasis cercanas sin inventar texto nuevo."""
    lowered = text.casefold()
    lowered = re.sub(r"[¿?¡!.,;:…\"\'`´]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def normalize_reply_for_dedupe(text: str) -> str:
    """Colapsa respuestas casi idénticas (puntuación / emoji / espacios)."""
    lowered = text.casefold()
    lowered = re.sub(r"[\U0001F300-\U0001FAFF]", " ", lowered)
    lowered = re.sub(r"[¿?¡!.,;:…\"\'`´\-—–()\[\]{}]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if len(a) > 240:
        a = a[:240]
    if len(b) > 240:
        b = b[:240]
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def replies_near_duplicate(a: str, b: str, threshold: float = 0.90) -> bool:
    na, nb = normalize_reply_for_dedupe(a), normalize_reply_for_dedupe(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter = min(len(na), len(nb))
    if shorter < 12:
        return na == nb
    lcs = _lcs_len(na, nb)
    return (lcs / shorter) >= threshold


def dedupe_near_identical_replies(
    replies: list[str], *, threshold: float = 0.90
) -> list[str]:
    """Conserva solo respuestas suficientemente distintas (frases ya existentes)."""
    kept: list[str] = []
    for reply in replies:
        if any(replies_near_duplicate(reply, other, threshold) for other in kept):
            continue
        kept.append(reply)
    return kept


def group_reply_variants(
    pairs: list[tuple[str, str]],
    *,
    dedupe_threshold: float | None = 0.90,
) -> dict[str, dict[str, set[str]]]:
    """Por intención normalizada: usuarios originales y respuestas distintas."""
    groups: dict[str, dict[str, set[str]]] = {}
    for user, reply in pairs:
        key = normalize_user_intent(user)
        if not key or not reply:
            continue
        bucket = groups.setdefault(key, {"users": set(), "replies": set()})
        bucket["users"].add(user)
        if dedupe_threshold is None:
            bucket["replies"].add(reply)
            continue
        # Solo añadir si no es near-dupe de una ya guardada
        if any(
            replies_near_duplicate(reply, existing, dedupe_threshold)
            for existing in bucket["replies"]
        ):
            continue
        bucket["replies"].add(reply)
    return groups


def augment_multitarget_text(
    text: str,
    *,
    seed: int = 0,
    max_extra_turns: int = 4000,
    min_variants: int = 2,
    max_remixes_per_user: int = 2,
    dedupe_near_identical: bool = True,
    dedupe_threshold: float = 0.90,
) -> str:
    """Añade remixes Usuario/Agerbot solo con frases ya presentes en el corpus.

    Para intenciones con varias respuestas distintas, empareja usuarios existentes
    con otras respuestas existentes del mismo grupo (multi-target sin inventar).
    """
    pairs = parse_dialogue_pairs(text)
    threshold = dedupe_threshold if dedupe_near_identical else None
    groups = group_reply_variants(pairs, dedupe_threshold=threshold)
    rng = random.Random(seed)
    extras: list[str] = []
    seen: set[tuple[str, str]] = set(pairs)

    multi_keys = [
        key
        for key, bucket in groups.items()
        if len(bucket["replies"]) >= min_variants
    ]
    rng.shuffle(multi_keys)

    for key in multi_keys:
        if len(extras) >= max_extra_turns:
            break
        bucket = groups[key]
        users = sorted(bucket["users"])
        replies = sorted(bucket["replies"])
        for user in users:
            if len(extras) >= max_extra_turns:
                break
            candidates = [reply for reply in replies if (user, reply) not in seen]
            if not candidates:
                continue
            rng.shuffle(candidates)
            for reply in candidates[: max(1, max_remixes_per_user)]:
                if len(extras) >= max_extra_turns:
                    break
                extras.append(f"Usuario: {user}\nAgerbot: {reply}")
                seen.add((user, reply))

    if not extras:
        return text
    block = "\n\n".join(extras)
    if text.endswith("\n"):
        return text + "\n" + block + "\n"
    return text + "\n\n" + block + "\n"


def load_corpus(
    path: str | Path, tokenizer: TokenizerAny, *, text: str | None = None
) -> torch.Tensor:
    if text is None:
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
