#!/usr/bin/env python3
"""Copia/mezcla data/raw/agentic_seed.txt → data/processed/agerbot_agentic_v1.txt.

No lanza entrenamiento: solo prepara corpus para un train futuro.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "raw" / "agentic_seed.txt"
DEFAULT_DST = ROOT / "data" / "processed" / "agerbot_agentic_v1.txt"


def blocks_of(text: str) -> list[str]:
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument(
        "--append-to",
        type=Path,
        default=None,
        help="Si se indica, añade los bloques al final de este processed existente.",
    )
    args = parser.parse_args()

    src_text = args.src.read_text(encoding="utf-8")
    blocks = blocks_of(src_text)
    if not blocks:
        raise SystemExit(f"Sin bloques en {args.src}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(blocks) + "\n"
    args.dst.write_text(body, encoding="utf-8")
    print(f"Wrote {len(blocks)} blocks → {args.dst.relative_to(ROOT)}")

    if args.append_to is not None:
        target = args.append_to
        existing = target.read_text(encoding="utf-8") if target.is_file() else ""
        merged_blocks = blocks_of(existing)
        seen = set(merged_blocks)
        added = 0
        for block in blocks:
            if block not in seen:
                merged_blocks.append(block)
                seen.add(block)
                added += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n\n".join(merged_blocks) + "\n", encoding="utf-8")
        print(f"Appended {added} new blocks → {target}")


if __name__ == "__main__":
    main()
