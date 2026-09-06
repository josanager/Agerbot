#!/usr/bin/env python3
"""Entrena un tokenizador BPE sobre corpus procesado existente (sin texto nuevo)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agerbot.tokenizer import BpeTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="Ruta a corpus .txt (se puede repetir). Default: learn_v1 + social_v2.",
    )
    parser.add_argument("--out", default="data/tokenizers/densify-v1")
    parser.add_argument("--vocab-size", type=int, default=6144)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    corpora = args.corpus or [
        "data/processed/agerbot_learn_v1.txt",
        "data/processed/agerbot_social_v2.txt",
    ]
    paths = [ROOT / path if not Path(path).is_absolute() else Path(path) for path in corpora]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Corpus no encontrado: {path}")
    tokenizer = BpeTokenizer.train_from_files(
        paths, vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    saved = tokenizer.save(out)
    sample = "Usuario: hola\nAgerbot: Hola, ¿qué tal?"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    meta = {
        "type": "bpe",
        "version": 1,
        "vocab_size": tokenizer.vocab_size,
        "corpora": [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in paths],
        "sample_roundtrip_ok": decoded.replace("Ġ", " ").strip() is not None,
        "n_sample_tokens": len(encoded),
    }
    (out / "train_info.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved={saved} vocab_size={tokenizer.vocab_size} sample_tokens={len(encoded)}")
    print(f"roundtrip={decoded!r}")


if __name__ == "__main__":
    main()
