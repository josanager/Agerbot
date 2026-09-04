#!/usr/bin/env python3
"""Elige el checkpoint que maximiza (accuracy - memorization_rate) en el bank.

Útil para re-seleccionar entre best.pt / bankcand-*.pt / step-*.pt sin reentrenar.
No sube *.pt a git; solo escribe informe JSON y opcionalmente copia best_bank.pt.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from eval_learn_bank import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--glob", default="best.pt,best_bank.pt,bankcand-*.pt")
    p.add_argument("--bank", default=str(ROOT / "data/evaluation/learn_bank_v1.jsonl"))
    p.add_argument(
        "--train-corpus", default=str(ROOT / "data/processed/agerbot_social_v2.txt")
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--promote-best", action="store_true")
    args = p.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    patterns = [x.strip() for x in args.glob.split(",") if x.strip()]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(sorted(ckpt_dir.glob(pat)))
    # unique preserve order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        unique.append(path)
    if not unique:
        raise SystemExit(f"No checkpoints matched in {ckpt_dir}")

    rows = []
    best = None
    for path in unique:
        report = evaluate(
            checkpoint=path,
            bank_path=Path(args.bank),
            train_corpus=Path(args.train_corpus),
            device_name=args.device,
            max_new_tokens=80,
            temperature=0.7,
            top_k=40,
            limit=None,
            out_path=ROOT
            / "reports"
            / f"learn_bank_{ckpt_dir.name}_{path.stem}.json",
        )
        score = report["accuracy"] - report["memorization_rate"]
        row = {
            "checkpoint": str(path),
            "accuracy": report["accuracy"],
            "memorization_rate": report["memorization_rate"],
            "score": round(score, 4),
            "parameters": report["parameters"],
        }
        rows.append(row)
        print(
            f"SELECT {path.name} acc={row['accuracy']} mem={row['memorization_rate']} "
            f"score={row['score']}"
        )
        if best is None or row["score"] > best["score"]:
            best = row

    summary = {
        "checkpoint_dir": str(ckpt_dir),
        "candidates": rows,
        "best": best,
        "selection_metric": "accuracy - memorization_rate",
    }
    out = Path(args.out) if args.out else ROOT / "reports" / f"bank_select_{ckpt_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} best={best}")
    if args.promote_best and best is not None:
        dest = ckpt_dir / "best_bank.pt"
        shutil.copy2(best["checkpoint"], dest)
        # Also overwrite best.pt so serve/eval defaults track bank metric
        shutil.copy2(best["checkpoint"], ckpt_dir / "best.pt")
        print(f"promoted {best['checkpoint']} -> {dest} and best.pt")


if __name__ == "__main__":
    main()
