#!/usr/bin/env python3
"""Paraphrase-augment EXISTING social-v2 only → data/processed/agerbot_learn_v1.txt.

- Dedup near-duplicates
- For each kept block: 1–2 paraphrase variants of Usuario lines (+ light Agerbot paraphrase)
- NO new topics
- Cap ~2–3× v2 size by diversity, not spam
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "agerbot_social_v2.txt"
OUT = ROOT / "data" / "processed" / "agerbot_learn_v1.txt"

# Word/phrase swaps that stay on the same social themes (no new topics).
USER_SWAPS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\bsomos amigos\b", re.I), ["somos amigos", "tú y yo somos amigos", "contamos como amigos"]),
    (re.compile(r"\beres mi amigo\b", re.I), ["eres mi amigo", "te tengo por amigo", "te veo como amigo"]),
    (re.compile(r"\bcompañero de charla\b", re.I), ["compañero de charla", "compañero de conversación"]),
    (re.compile(r"\bme escuchas\b", re.I), ["me escuchas", "me estás escuchando", "de verdad me oyes"]),
    (re.compile(r"\bno cambies de tema\b", re.I), ["no cambies de tema", "quédate en el hilo", "no te salgas del tema"]),
    (re.compile(r"\bhoy estoy triste\b", re.I), ["hoy estoy triste", "hoy ando triste", "me siento triste hoy"]),
    (re.compile(r"\bme siento solo\b", re.I), ["me siento solo", "me siento solitario", "ando solo últimamente"]),
    (re.compile(r"\bestoy ansioso\b", re.I), ["estoy ansioso", "tengo ansiedad", "ando nervioso"]),
    (re.compile(r"\bcuéntame un chiste\b", re.I), ["cuéntame un chiste", "tirame un chiste", "quiero un chiste"]),
    (re.compile(r"\bgracias\b", re.I), ["gracias", "mil gracias", "te lo agradezco"]),
    (re.compile(r"\bde verdad\?\b", re.I), ["de verdad?", "en serio?", "posta?"]),
    (re.compile(r"\bhola\b", re.I), ["hola", "hola hola", "hey"]),
    (re.compile(r"\bbuenos días\b", re.I), ["buenos días", "buen día", "buenos dias"]),
    (re.compile(r"\bbuenas noches\b", re.I), ["buenas noches", "que descanses", "buenas noches"]),
    (re.compile(r"\bno me da risa\b", re.I), ["no me da risa", "no me hizo gracia", "no me reí"]),
]

AGERBOT_SWAPS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\bSí 🙂\b"), ["Sí 🙂", "Sí.", "Claro 🙂"]),
    (re.compile(r"\bTe escucho\b"), ["Te escucho", "Te oigo", "Aquí te escucho"]),
    (re.compile(r"\bAquí estoy\b"), ["Aquí estoy", "Sigo aquí", "Estoy aquí contigo"]),
    (re.compile(r"\bcompañeros de charla\b", re.I), ["compañeros de charla", "compañeros de conversación"]),
    (re.compile(r"\bDe nada\b"), ["De nada", "Para eso estoy", "Cuando quieras"]),
    (re.compile(r"\bPerfecto\b"), ["Perfecto", "Genial", "Vale"]),
    (re.compile(r"\bOk\b"), ["Ok", "Vale", "De acuerdo"]),
    (re.compile(r"\bSin prisa\b"), ["Sin prisa", "Con calma", "A tu ritmo"]),
]


def blocks_of(text: str) -> list[str]:
    return [b.strip() for b in text.split("\n\n") if b.strip()]


def fingerprint(block: str) -> str:
    """Near-dup key: collapse whitespace + casefold + light punctuation strip."""
    t = block.casefold()
    t = re.sub(r"[¿?¡!;,:.…]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha1(t.encode()).hexdigest()


def soft_fingerprint(block: str) -> str:
    """Coarser: only Usuario lines joined."""
    users = []
    for line in block.splitlines():
        if line.startswith("Usuario:"):
            u = line[len("Usuario:") :].strip().casefold()
            u = re.sub(r"[¿?¡!;,:.…]+", " ", u)
            u = re.sub(r"\s+", " ", u).strip()
            users.append(u)
    return "|".join(users)


def apply_swaps(text: str, swaps: list[tuple[re.Pattern[str], list[str]]], rng: random.Random) -> str:
    out = text
    for pattern, options in swaps:
        if pattern.search(out):
            choice = rng.choice(options)
            out = pattern.sub(choice, out, count=1)
    return out


def paraphrase_line(speaker: str, content: str, rng: random.Random) -> str:
    if speaker == "Usuario":
        new = apply_swaps(content, USER_SWAPS, rng)
        # Light punctuation / trailing softness without new topics
        # Soft surface variants (same topic)
        if rng.random() < 0.7:
            prefixes = ["", "oye, ", "eh, ", "bueno… ", "a ver: "]
            suffixes = ["", " eh", " pues", " ahora", ""]
            cand = rng.choice(prefixes) + new + rng.choice(suffixes)
            cand = re.sub(r"\s+", " ", cand).strip()
            # don't invent empty
            if cand:
                new = cand
        elif new == content and rng.random() < 0.5:
            variants = [
                content.rstrip("?") + "?",
                ("¿" + content if not content.startswith("¿") and content.endswith("?") else content),
            ]
            new = rng.choice(variants)
        return new
    # Agerbot: light only
    new = apply_swaps(content, AGERBOT_SWAPS, rng)
    if new == content and rng.random() < 0.35:
        soft = [
            content,
            content.replace(" ✨", "").strip() if "✨" in content else content,
            (content + " 🙂") if "🙂" not in content and len(content) < 70 else content,
        ]
        new = rng.choice(soft)
    return new


def paraphrase_block(block: str, rng: random.Random) -> str:
    lines_out = []
    for line in block.splitlines():
        if line.startswith("Usuario:"):
            content = line[len("Usuario:") :].strip()
            lines_out.append(f"Usuario: {paraphrase_line('Usuario', content, rng)}")
        elif line.startswith("Agerbot:"):
            content = line[len("Agerbot:") :].strip()
            lines_out.append(f"Agerbot: {paraphrase_line('Agerbot', content, rng)}")
        else:
            lines_out.append(line)
    return "\n".join(lines_out)


def dedupe(blocks: list[str]) -> list[str]:
    seen_fp: set[str] = set()
    seen_soft: set[str] = set()
    kept: list[str] = []
    for block in blocks:
        fp = fingerprint(block)
        soft = soft_fingerprint(block)
        if fp in seen_fp:
            continue
        # drop near-dup multi-turn with identical user sequence
        if soft and soft in seen_soft and block.count("Usuario:") >= 2:
            continue
        seen_fp.add(fp)
        if soft:
            seen_soft.add(soft)
        kept.append(block)
    return kept


def build(seed: int = 20260904, max_multiplier: float = 2.6) -> dict:
    rng = random.Random(seed)
    raw = blocks_of(SRC.read_text(encoding="utf-8"))
    base = dedupe(raw)
    augmented: list[str] = list(base)

    # Cap by diversity: up to 2 paraphrases per block, stop at ~max_multiplier × base chars
    base_chars = sum(len(b) for b in base)
    char_budget = int(base_chars * max_multiplier)

    # Shuffle order for which blocks get paraphrases first
    order = list(range(len(base)))
    rng.shuffle(order)
    paraphrases_added = 0
    seen_fp = {fingerprint(b) for b in augmented}

    for idx in order:
        if sum(len(b) for b in augmented) >= char_budget:
            break
        block = base[idx]
        n_vars = 1 if rng.random() < 0.25 else 2
        for _ in range(n_vars):
            if sum(len(b) for b in augmented) >= char_budget:
                break
            variant = paraphrase_block(block, rng)
            fp = fingerprint(variant)
            if fp in seen_fp:
                # try once more with different rng draw
                variant = paraphrase_block(block, rng)
                fp = fingerprint(variant)
            if fp in seen_fp:
                continue
            # Reject only exact duplicates
            if variant == block:
                continue
            seen_fp.add(fp)
            augmented.append(variant)
            paraphrases_added += 1

    rng.shuffle(augmented)
    text = "\n\n".join(augmented) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    return {
        "path": str(OUT),
        "src_blocks": len(raw),
        "deduped_blocks": len(base),
        "out_blocks": len(augmented),
        "paraphrases_added": paraphrases_added,
        "src_chars": len(SRC.read_text(encoding="utf-8")),
        "out_chars": len(text),
        "multiplier": round(len(text) / max(len(SRC.read_text(encoding="utf-8")), 1), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-multiplier", type=float, default=2.8)
    args = parser.parse_args()
    stats = build(args.seed, args.max_multiplier)
    print(
        f"wrote {stats['path']} blocks={stats['out_blocks']} "
        f"(src={stats['src_blocks']} deduped={stats['deduped_blocks']} "
        f"+para={stats['paraphrases_added']}) chars={stats['out_chars']} "
        f"multiplier={stats['multiplier']}"
    )


if __name__ == "__main__":
    main()
