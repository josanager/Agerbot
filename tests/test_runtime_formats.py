import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from agerbot.model import Agerbot, ModelConfig
from agerbot.server import AgerbotRuntime, RuntimeAPIError


class RuntimeFormatTests(unittest.TestCase):
    def create_checkpoint(
        self,
        directory: Path,
        *,
        tokenizer="byte-v1",
        model_vocab_size: int = 256,
        include_tokenizer: bool = True,
        manifest_tokenizer: str | None = None,
    ) -> Path:
        config = ModelConfig(
            vocab_size=model_vocab_size,
            context_length=16,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
        )
        model = Agerbot(config)
        checkpoint = {
            "format_version": 1,
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
        }
        if include_tokenizer:
            checkpoint["tokenizer"] = tokenizer
        path = directory / "model.pt"
        torch.save(checkpoint, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        model_payload = {
            "name": "Agerbot",
            "version": "0.2.0",
            "trainingName": "format-test",
            "parameters": model.parameter_count(),
            "contextLength": config.context_length,
        }
        if manifest_tokenizer is not None:
            model_payload["tokenizer"] = manifest_tokenizer
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "channel": "stable",
                    "model": model_payload,
                    "checkpoint": {
                        "filename": path.name,
                        "sizeBytes": path.stat().st_size,
                        "sha256": digest,
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_loads_legacy_byte_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.create_checkpoint(
                Path(temporary), tokenizer="byte-v1", manifest_tokenizer="byte-v1"
            )
            runtime = AgerbotRuntime(path, "cpu")
            health = runtime.health()
            self.assertEqual(health["runtimeVersion"], "0.2.0")
            self.assertEqual(health["model"]["tokenizer"], "byte-v1")
            self.assertEqual(health["model"]["contextLength"], 16)

    def test_loads_serialized_char_v1_without_refitting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            characters = ["�", "z", "a", " "]
            path = self.create_checkpoint(
                Path(temporary),
                tokenizer={"type": "char", "version": 1, "characters": characters},
                model_vocab_size=len(characters),
                manifest_tokenizer="char-v1",
            )
            runtime = AgerbotRuntime(path, "cpu")
            self.assertEqual(runtime.tokenizer.characters, characters)
            self.assertEqual(runtime.health()["model"]["tokenizer"], "char-v1")

    def test_rejects_unknown_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.create_checkpoint(
                Path(temporary),
                tokenizer={"type": "wordpiece", "version": 1},
                model_vocab_size=4,
            )
            with self.assertRaises(RuntimeAPIError) as context:
                AgerbotRuntime(path, "cpu")
            self.assertEqual(context.exception.code, "tokenizer_unsupported")

    def test_rejects_incompatible_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.create_checkpoint(
                Path(temporary),
                tokenizer={
                    "type": "char",
                    "version": 1,
                    "characters": ["�", "a", "b"],
                },
                model_vocab_size=4,
            )
            with self.assertRaises(RuntimeAPIError) as context:
                AgerbotRuntime(path, "cpu")
            self.assertEqual(context.exception.code, "tokenizer_vocab_mismatch")

    def test_rejects_checkpoint_without_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.create_checkpoint(
                Path(temporary), include_tokenizer=False, model_vocab_size=4
            )
            with self.assertRaises(RuntimeAPIError) as context:
                AgerbotRuntime(path, "cpu")
            self.assertEqual(context.exception.code, "tokenizer_missing")

    def test_rejects_nan_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.create_checkpoint(
                directory, tokenizer="byte-v1", manifest_tokenizer="byte-v1"
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            first_tensor = next(iter(checkpoint["model_state"].values()))
            first_tensor.reshape(-1)[0] = float("nan")
            torch.save(checkpoint, path)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["checkpoint"]["sizeBytes"] = path.stat().st_size
            manifest["checkpoint"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeAPIError) as context:
                AgerbotRuntime(path, "cpu")
            self.assertEqual(context.exception.code, "model_invalid_parameters")


if __name__ == "__main__":
    unittest.main()
