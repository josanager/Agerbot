#!/usr/bin/env python3
"""Construye data/processed/agerbot_agentic_v1.txt mezclando agentic_seed + slice social.

No lanza entrenamiento: solo prepara corpus para un train futuro.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "raw" / "agentic_seed.txt"
DEFAULT_SOCIAL = ROOT / "data" / "processed" / "agerbot_social_v1.txt"
DEFAULT_DST = ROOT / "data" / "processed" / "agerbot_agentic_v1.txt"


def blocks_of(text: str) -> list[str]:
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--social", type=Path, default=DEFAULT_SOCIAL)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument(
        "--social-slice",
        type=int,
        default=120,
        help="Cuántos bloques sociales mezclar (para seguir conversacional).",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    src_text = args.src.read_text(encoding="utf-8")
    agentic_blocks = blocks_of(src_text)
    if not agentic_blocks:
        raise SystemExit(f"Sin bloques en {args.src}")

    social_blocks: list[str] = []
    if args.social.is_file() and args.social_slice > 0:
        rng = random.Random(args.seed)
        all_social = blocks_of(args.social.read_text(encoding="utf-8"))
        # Preferir turnos cortos conversacionales (sin Acción) para no diluir el agentic.
        conversational = [
            b
            for b in all_social
            if "Acción:" not in b and b.count("Usuario:") <= 3 and len(b) < 600
        ]
        pool = conversational or all_social
        k = min(args.social_slice, len(pool))
        social_blocks = rng.sample(pool, k=k)

    merged = agentic_blocks + social_blocks
    rng = random.Random(args.seed + 1)
    rng.shuffle(merged)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text("\n\n".join(merged) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(merged)} blocks "
        f"({len(agentic_blocks)} agentic + {len(social_blocks)} social) "
        f"→ {args.dst.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
