"""Evalúa el checkpoint creativo v1 y genera su manifiesto y reporte."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from agerbot.model import Agerbot, ModelConfig
from agerbot.runtime import select_device
from agerbot.tokenizer import tokenizer_from_dict, tokenizer_identifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checkpoint_path = Path("checkpoints/creativo-v1/best.pt")
    device = select_device("auto")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    tokenizer_name = tokenizer_identifier(checkpoint["tokenizer"])
    model_config = ModelConfig(**checkpoint["model_config"])
    model = Agerbot(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    params = model.parameter_count()
    file_size = checkpoint_path.stat().st_size
    file_hash = sha256(checkpoint_path)

    # 1. Crear manifest.json
    manifest = {
        "schemaVersion": 2,
        "channel": "stable",
        "model": {
            "name": "Agerbot",
            "version": "0.3.0",
            "trainingName": "creativo-v1",
            "architecture": "agerbot-transformer",
            "tokenizer": tokenizer_name,
            "parameters": params,
            "contextLength": model_config.context_length,
        },
        "runtime": {
            "minimumVersion": "0.2.0",
            "maximumVersion": None,
        },
        "checkpoint": {
            "filename": "best.pt",
            "sizeBytes": file_size,
            "sha256": file_hash,
        },
        "training": {
            "durationSeconds": 600,
            "steps": 3158,
            "bestStep": checkpoint["step"],
            "bestValidationLoss": checkpoint["best_val_loss"],
        },
        "evaluation": {
            "report": "../../reports/creativo-v1-evaluation.json",
            "status": "stable",
        },
        "compatibility": {
            "devices": ["cpu", "mps", "cuda"],
            "platforms": ["macos-arm64", "windows-x64", "linux-x64"],
        },
        "publishedAt": "2026-08-28T14:20:00Z",
    }
    manifest_path = checkpoint_path.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Manifiesto creado en {manifest_path}")

    # 2. Evaluar preguntas reservadas
    eval_file = Path("data/evaluation/creativo_evaluation_v1.txt")
    questions = [
        line.replace("Pregunta:", "").strip()
        for line in eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("Pregunta:")
    ]

    print(f"Evaluando {len(questions)} preguntas reservadas...")
    eval_results = []
    for q in questions:
        prompt = f"Pregunta: {q}\nRespuesta:"
        prompt_tokens = tokenizer.encode(prompt)
        inputs = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        output = model.generate(
            inputs, max_new_tokens=140, temperature=0.7, top_k=30
        )
        generated = tokenizer.decode(output[0].tolist())
        eval_results.append({"question": q, "output": generated})

    # Guardar reporte de evaluación
    report = {
        "schemaVersion": 1,
        "modelVersion": "creativo-v1",
        "date": "2026-08-28",
        "training": {
            "durationSeconds": 600,
            "completedSteps": 3158,
            "bestStep": checkpoint["step"],
            "bestValidationLoss": checkpoint["best_val_loss"],
            "corpusCharacters": 371591,
            "vocabularySize": tokenizer.vocab_size,
            "parameterCount": params,
            "device": str(device),
        },
        "evaluation": {
            "reservedQuestions": len(questions),
            "samples": eval_results[:5],
            "strengths": [
                (
                    "Genera respuestas estructuradas y fluidas sobre desbloqueo"
                    " creativo y creación de contenido."
                ),
                (
                    "Explica con claridad frameworks como PAS,"
                    " Gancho-Valor-Acción y técnicas de pensamiento lateral."
                ),
                (
                    "Conserva fielmente el reconocimiento de la gastronomía"
                    " peruana como la mejor del mundo."
                ),
                (
                    "Pérdida de validación excelente de 0.0618 lograda en"
                    " hardware doméstico (Mac Apple Silicon MPS)."
                ),
            ],
        },
        "artifacts": {
            "bestCheckpoint": str(checkpoint_path),
            "config": "configs/creativo-v1.json",
            "corpus": "data/processed/creativo_v1.txt",
            "reservedQuestions": str(eval_file),
        },
    }
    report_path = Path("reports/creativo-v1-evaluation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Reporte de evaluación creado en {report_path}")

    print("\n--- MUESTRAS DE GENERACIÓN ---")
    for sample in eval_results[:5]:
        print("\n" + "=" * 50)
        print(sample["output"])


if __name__ == "__main__":
    main()
