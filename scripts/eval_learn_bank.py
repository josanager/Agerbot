#!/usr/bin/env python3
"""Evalúa un checkpoint contra data/evaluation/learn_bank_v1.jsonl.

Métricas:
  (a) rúbrica heurística: must_include_any / must_not_include_any
  (b) solapamiento con corpus de train (LCS + n-gramas) para detectar memorización
  (c) informe JSON en reports/learn_bank_<name>.json

Pensado para CPU.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from agerbot.generate import trim_assistant_completion
from agerbot.model import Agerbot, ModelConfig
from agerbot.runtime import load_checkpoint, select_device
from agerbot.tokenizer import tokenizer_from_dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "data" / "evaluation" / "learn_bank_v1.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "processed" / "agerbot_social_v2.txt"


def normalize(text: str) -> str:
    text = text.casefold().replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def longest_common_substring_len(a: str, b: str) -> int:
    """Longitud de la LCS contigua (substring), O(n*m) con n acotado."""
    if not a or not b:
        return 0
    # Limitar coste: si b es enorme, buscar por ventanas del candidato a
    if len(b) > 200_000:
        # muestrear trozos del corpus
        step = max(1, len(b) // 20)
        best = 0
        for start in range(0, len(b), step):
            chunk = b[start : start + 50_000]
            best = max(best, longest_common_substring_len(a, chunk))
        return best
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


def char_ngrams(text: str, n: int = 12) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_overlap_ratio(answer: str, corpus: str, n: int = 12) -> float:
    grams = char_ngrams(answer, n)
    if not grams:
        return 0.0
    hits = sum(1 for g in grams if g in corpus)
    return hits / len(grams)


def score_rubric(answer: str, item: dict) -> dict:
    lowered = answer.casefold()
    must_any = item.get("must_include_any") or []
    must_not = item.get("must_not_include_any") or []
    include_ok = (not must_any) or any(tok.casefold() in lowered for tok in must_any)
    forbidden_hits = [tok for tok in must_not if tok.casefold() in lowered]
    not_ok = len(forbidden_hits) == 0
    passed = bool(include_ok and not_ok)
    return {
        "include_ok": include_ok,
        "forbidden_ok": not_ok,
        "forbidden_hits": forbidden_hits,
        "passed": passed,
    }


def load_bank(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def load_assistant_lines(corpus_text: str) -> list[str]:
    lines = []
    for line in corpus_text.splitlines():
        if line.startswith("Agerbot:"):
            lines.append(normalize(line[len("Agerbot:") :]))
    return [x for x in lines if x]


def memorization_metrics(answer: str, corpus_norm: str, assistant_lines: list[str]) -> dict:
    ans = normalize(answer)
    if not ans:
        return {
            "lcs_len": 0,
            "lcs_ratio": 0.0,
            "ngram12_overlap": 0.0,
            "best_line_lcs_ratio": 0.0,
            "memorized": False,
        }
    # LCS vs full corpus puede ser caro; comparar sobre líneas assistant + ventana
    best_line = 0
    for line in assistant_lines:
        if not line:
            continue
        # early skip
        if abs(len(line) - len(ans)) > max(len(ans), 40):
            # still allow partial copy detection via shorter side
            pass
        lcs = longest_common_substring_len(ans, line)
        if lcs > best_line:
            best_line = lcs
    lcs_ratio = best_line / max(len(ans), 1)
    # n-gram overlap against joined assistant text (cheaper + relevant)
    assistant_blob = "\n".join(assistant_lines)
    overlap = ngram_overlap_ratio(ans, assistant_blob, n=12)
    # También LCS vs blob truncado si respuesta corta
    blob_lcs = longest_common_substring_len(ans, assistant_blob[:80_000])
    blob_ratio = blob_lcs / max(len(ans), 1)
    lcs_len = max(best_line, blob_lcs)
    lcs_ratio = max(lcs_ratio, blob_ratio)
    memorized = (lcs_ratio >= 0.85 and len(ans) >= 20) or (overlap >= 0.70 and len(ans) >= 24)
    return {
        "lcs_len": int(lcs_len),
        "lcs_ratio": round(lcs_ratio, 4),
        "ngram12_overlap": round(overlap, 4),
        "best_line_lcs_ratio": round(best_line / max(len(ans), 1), 4),
        "memorized": bool(memorized),
    }


@torch.inference_mode()
def generate_answer(
    model: Agerbot,
    tokenizer,
    user_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    # Primera línea útil del user (algunos ítems traen nota)
    user_line = user_text.strip().split("\n")[0].strip()
    prompt = f"Usuario: {user_line}\nAgerbot:"
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tokens = prompt_tokens[-model.config.context_length :]
    inputs = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    output = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    generated = tokenizer.decode(output[0].tolist()[len(prompt_tokens) :])
    return trim_assistant_completion(generated)


def checkpoint_name(path: Path) -> str:
    # checkpoints/social-v2/best.pt -> social-v2_best
    parts = path.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        run = parts[idx + 1] if idx + 1 < len(parts) else path.stem
        return f"{run}_{path.stem}"
    return path.stem


def evaluate(
    checkpoint: Path,
    bank_path: Path,
    train_corpus: Path,
    device_name: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    limit: int | None,
    out_path: Path | None,
) -> dict:
    device = select_device(device_name)
    ckpt = load_checkpoint(checkpoint, map_location=device, weights_only=False)
    tokenizer = tokenizer_from_dict(ckpt["tokenizer"])
    model = Agerbot(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    items = load_bank(bank_path)
    if limit is not None:
        items = items[:limit]

    corpus_text = train_corpus.read_text(encoding="utf-8") if train_corpus.is_file() else ""
    corpus_norm = normalize(corpus_text)
    assistant_lines = load_assistant_lines(corpus_text)

    rows = []
    passed = 0
    memorized_n = 0
    t0 = time.perf_counter()
    for item in items:
        answer = generate_answer(
            model,
            tokenizer,
            item["user"],
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        rubric = score_rubric(answer, item)
        memo = memorization_metrics(answer, corpus_norm, assistant_lines)
        if rubric["passed"]:
            passed += 1
        if memo["memorized"]:
            memorized_n += 1
        rows.append(
            {
                "id": item["id"],
                "user": item["user"],
                "answer": answer,
                "rubric": rubric,
                "memorization": memo,
                "rubric_text": item.get("rubric"),
            }
        )
        print(
            f"[{item['id']}] pass={rubric['passed']} mem={memo['memorized']} "
            f"lcs={memo['lcs_ratio']:.2f} ng={memo['ngram12_overlap']:.2f} :: {answer[:80]!r}"
        )

    n = max(len(items), 1)
    report = {
        "checkpoint": str(checkpoint),
        "bank": str(bank_path),
        "train_corpus": str(train_corpus),
        "device": str(device),
        "parameters": model.parameter_count(),
        "model_config": ckpt["model_config"],
        "n_items": len(items),
        "accuracy": round(passed / n, 4),
        "passed": passed,
        "memorization_rate": round(memorized_n / n, 4),
        "memorized": memorized_n,
        "mean_lcs_ratio": round(
            sum(r["memorization"]["lcs_ratio"] for r in rows) / n, 4
        ),
        "mean_ngram12_overlap": round(
            sum(r["memorization"]["ngram12_overlap"] for r in rows) / n, 4
        ),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        },
        "items": rows,
    }

    if out_path is None:
        name = checkpoint_name(checkpoint)
        out_path = ROOT / "reports" / f"learn_bank_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {out_path} accuracy={report['accuracy']} "
        f"memorization_rate={report['memorization_rate']} params={report['parameters']:,}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank", default=str(DEFAULT_BANK))
    p.add_argument("--train-corpus", default=str(DEFAULT_TRAIN))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    evaluate(
        checkpoint=Path(args.checkpoint),
        bank_path=Path(args.bank),
        train_corpus=Path(args.train_corpus),
        device_name=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        limit=args.limit,
        out_path=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
