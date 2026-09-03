"""Prepara assets verificables de una release de modelo Agerbot; no publica nada."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch

from agerbot.model import Agerbot, ModelConfig
from agerbot.tokenizer import tokenizer_from_dict, tokenizer_identifier

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "tokenizer" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("El checkpoint no contiene tokenizador o configuración de modelo")
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    tokenizer_name = tokenizer_identifier(checkpoint["tokenizer"])
    config = ModelConfig(**checkpoint["model_config"])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("El vocabulario del tokenizador no coincide con el modelo")
    model = Agerbot(config)
    model.load_state_dict(checkpoint["model_state"])
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError("El checkpoint contiene parámetros NaN o infinitos")
    return {
        "tokenizer": tokenizer_name,
        "parameters": model.parameter_count(),
        "contextLength": config.context_length,
    }


def prepare_release(
    checkpoint: Path,
    evaluation: Path,
    version: str,
    output_root: Path,
    published_at: str,
    force: bool = False,
) -> Path:
    if not SEMVER.fullmatch(version):
        raise ValueError("La versión debe seguir Semantic Versioning")
    if not checkpoint.is_file() or not evaluation.is_file():
        raise FileNotFoundError("Falta el checkpoint o el informe de evaluación")
    metadata = inspect_checkpoint(checkpoint)
    evaluation_payload = json.loads(evaluation.read_text(encoding="utf-8"))
    training_name = str(evaluation_payload.get("modelVersion") or checkpoint.parent.name)
    output = output_root / version
    if output.exists():
        if not force:
            raise FileExistsError(f"Ya existe {output}; usa --force para reemplazarlo")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    model_name = f"agerbot-model-{version}.pt"
    evaluation_name = f"agerbot-evaluation-{version}.json"
    model_output = output / model_name
    evaluation_output = output / evaluation_name
    shutil.copy2(checkpoint, model_output)
    shutil.copy2(evaluation, evaluation_output)
    model_hash = sha256(model_output)

    manifest = {
        "schemaVersion": 2,
        "channel": "stable",
        "release": {
            "version": version,
            "tag": f"model-v{version}",
            "publishedAt": published_at,
        },
        "model": {
            "name": "Agerbot",
            "trainingName": training_name,
            "architecture": "agerbot-transformer",
            **metadata,
        },
        "runtime": {"minimumVersion": "0.2.0", "maximumVersion": None},
        "artifact": {
            "assetName": model_name,
            "sizeBytes": model_output.stat().st_size,
            "sha256": model_hash,
        },
        "evaluation": {
            "assetName": evaluation_name,
            "status": "experimental",
        },
        "compatibility": {
            "platforms": ["macos-arm64", "windows-x64"],
            "devices": ["cpu", "mps", "cuda"],
        },
    }
    manifest_output = output / "agerbot-release.json"
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksums = output / "checksums-sha256.txt"
    checksum_lines = [
        f"{sha256(manifest_output)}  {manifest_output.name}",
        f"{model_hash}  {model_output.name}",
        f"{sha256(evaluation_output)}  {evaluation_output.name}",
    ]
    checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--published-at", required=True, help="Fecha ISO-8601 de publicación prevista")
    parser.add_argument("--output", type=Path, default=Path("dist/releases"))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = prepare_release(
        checkpoint=args.checkpoint,
        evaluation=args.evaluation,
        version=args.version,
        output_root=args.output,
        published_at=args.published_at,
        force=args.force,
    )
    print(output)


if __name__ == "__main__":
    main()
