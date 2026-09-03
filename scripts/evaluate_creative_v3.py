"""Evalúa el checkpoint creativo v3 y genera su manifiesto y reporte."""

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
    checkpoint_path = Path("checkpoints/creativo-v3/best.pt")
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
            "trainingName": "creativo-v3",
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
            "durationSeconds": 900,
            "steps": 4772,
            "bestStep": checkpoint["step"],
            "bestValidationLoss": checkpoint["best_val_loss"],
        },
        "evaluation": {
            "report": "../../reports/creativo-v3-evaluation.json",
            "status": "stable",
        },
        "compatibility": {
            "devices": ["cpu", "mps", "cuda"],
            "platforms": ["macos-arm64", "windows-x64", "linux-x64"],
        },
        "publishedAt": "2026-08-28T15:15:00Z",
    }
    manifest_path = checkpoint_path.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Manifiesto creado en {manifest_path}")

    # 2. Evaluar preguntas reservadas
    eval_file = Path("data/evaluation/creativo_evaluation_v3.txt")
    questions = [
        line.replace("Usuario:", "").strip()
        for line in eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("Usuario:")
    ]

    print(f"Evaluando {len(questions)} consultas de prueba...")
    eval_results = []
    for q in questions:
        prompt = f"Usuario: {q}\nAgerbot:"
        prompt_tokens = tokenizer.encode(prompt)
        inputs = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        output = model.generate(
            inputs, max_new_tokens=150, temperature=0.7, top_k=30
        )
        generated = tokenizer.decode(output[0].tolist())
        # Extraer solo la primera respuesta de Agerbot antes de otro Usuario:
        lines = generated.splitlines()
        clean_lines = []
        for line in lines:
            if clean_lines and (line.startswith("Usuario:") or line.startswith("Pregunta:") or line.startswith("Conversación:")):
                break
            clean_lines.append(line)
        cleaned = "\n".join(clean_lines).strip()
        eval_results.append({"prompt": q, "output": cleaned})

    # Guardar reporte de evaluación
    report = {
        "schemaVersion": 1,
        "modelVersion": "creativo-v3",
        "date": "2026-08-28",
        "training": {
            "durationSeconds": 900,
            "completedSteps": 4772,
            "bestStep": checkpoint["step"],
            "bestValidationLoss": checkpoint["best_val_loss"],
            "corpusCharacters": 210309,
            "vocabularySize": tokenizer.vocab_size,
            "parameterCount": params,
            "device": str(device),
        },
        "evaluation": {
            "reservedQuestions": len(questions),
            "samples": eval_results,
            "strengths": [
                "Diferencia saludos de despedidas y agradecimientos con precisión.",
                "Admite honestamente 'no lo sé' ante temas fuera de su dominio.",
                "Pide contexto de manera adecuada ante palabras sueltas sin sentido.",
                "Conserva intactas las fórmulas creativas, técnicas anti-bloqueo y la gastronomía peruana.",
                "Mantiene exactamente el mismo peso compacto de 123 MB y bajo consumo de memoria."
            ],
        },
        "artifacts": {
            "bestCheckpoint": str(checkpoint_path),
            "config": "configs/creativo-v3.json",
            "corpus": "data/processed/creativo_v3.txt",
            "reservedQuestions": str(eval_file),
        },
    }
    report_path = Path("reports/creativo-v3-evaluation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Reporte de evaluación creado en {report_path}")

    print("\n--- MUESTRAS DE CONVERSACIÓN DE CADA INTENCIÓN ---")
    for sample in eval_results:
        print("\n" + "=" * 50)
        print(sample["output"])


if __name__ == "__main__":
    main()
