#!/usr/bin/env python3
"""Construye data/processed/agerbot_social_v1.txt enfatizando continuidad social."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_LINE_RE = re.compile(
    r"(?i)(\bsoy\s+agerbot\b|\bagerbot\s+0\.3\.0\b|\bsoy\s+la\s+versi[oó]n\s+0\.3\.0\b|"
    r"\bmi\s+versi[oó]n\s+es\s+la\s+0\.3\.0\b|creativo\s+v5)"
)
HARD_IDENTITY_RE = re.compile(r"(?i)soy\s+agerbot\b|agerbot\s+0\.3\.0")


def blocks_of(text: str) -> list[str]:
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def sanitize_block(block: str) -> str | None:
    lines: list[str] = []
    for line in block.splitlines():
        if line.startswith("Agerbot:") and IDENTITY_LINE_RE.search(line):
            line = "Agerbot: Soy una inteligencia artificial local. ¿En qué te ayudo?"
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if not cleaned.startswith("Usuario:") or "\nAgerbot:" not in cleaned:
        return None
    if HARD_IDENTITY_RE.search(cleaned) and "Soy una inteligencia artificial" not in cleaned:
        return None
    return cleaned


def pair_windows(pairs: list[tuple[str, str]], sizes: tuple[int, ...] = (1, 2, 3, 4)) -> list[str]:
    turns = [f"Usuario: {user}\nAgerbot: {assistant}" for user, assistant in pairs]
    out: list[str] = []
    for width in sizes:
        for index in range(0, max(0, len(turns) - width + 1)):
            out.append("\n".join(turns[index : index + width]))
    return out


def is_social_single(block: str) -> bool:
    if block.count("Usuario:") != 1:
        return False
    answer = block.split("Agerbot:", 1)[-1].strip()
    if len(answer) > 180:
        return False
    lowered = block.casefold()
    return not any(token in lowered for token in ("ingredientes:", "paso 1", "http", "```"))


def harvest(paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for block in blocks_of(path.read_text(encoding="utf-8")):
            cleaned = sanitize_block(block)
            if cleaned is None:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            if cleaned.count("Usuario:") >= 2 or is_social_single(cleaned):
                kept.append(cleaned)
    return kept


def expand_seed(seed_path: Path) -> list[str]:
    if not seed_path.is_file():
        return []
    expanded: list[str] = []
    for block in blocks_of(seed_path.read_text(encoding="utf-8")):
        cleaned = sanitize_block(block)
        if cleaned is None:
            continue
        expanded.append(cleaned)
        lines = cleaned.splitlines()
        pairs: list[tuple[str, str]] = []
        index = 0
        while index < len(lines) - 1:
            if lines[index].startswith("Usuario:") and lines[index + 1].startswith("Agerbot:"):
                user = lines[index].split(":", 1)[1].strip()
                assistant = lines[index + 1].split(":", 1)[1].strip()
                pairs.append((user, assistant))
                index += 2
            else:
                index += 1
        if len(pairs) >= 2:
            expanded.extend(pair_windows(pairs))
    return expanded


def build(seed: int = 20260904) -> tuple[Path, int, int]:
    random.seed(seed)
    processed = ROOT / "data" / "processed"
    raw = ROOT / "data" / "raw"
    out_path = processed / "agerbot_social_v1.txt"
    sources = [
        processed / "agerbot_contexto1024_v10.txt",
        processed / "agerbot_dialogues_v7_2h.txt",
        processed / "creativo_v5.txt",
        raw / "agerbot_dialogues_context_v10.txt",
        raw / "agerbot_dialogues_simple_v7.txt",
        raw / "agerbot_dialogues_simple_v8.txt",
        raw / "agerbot_dialogues_reasoning_v9.txt",
    ]
    blocks = expand_seed(raw / "agerbot_social_seed_v1.txt") + harvest(sources)
    final: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        key = block.casefold()
        if key in seen:
            continue
        seen.add(key)
        final.append(block)
    random.shuffle(final)
    text = "\n\n".join(final) + "\n"
    out_path.write_text(text, encoding="utf-8")
    multi = sum(1 for block in final if block.count("Usuario:") >= 2)
    return out_path, len(text), multi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    path, chars, multi = build(args.seed)
    print(f"wrote {path} chars={chars} multi_turn_blocks={multi}")


if __name__ == "__main__":
    main()
