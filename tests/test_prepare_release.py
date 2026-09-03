import json
import tempfile
import unittest
from pathlib import Path

import torch

from agerbot.model import Agerbot, ModelConfig


class PrepareReleaseTests(unittest.TestCase):
    def test_prepares_all_release_assets(self) -> None:
        import importlib.util

        script = Path(__file__).parents[1] / "scripts" / "prepare_release.py"
        spec = importlib.util.spec_from_file_location("prepare_release", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ModelConfig(
                context_length=16, d_model=16, n_heads=4, n_layers=1, dropout=0
            )
            model = Agerbot(config)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "model_config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "tokenizer": "byte-v1",
                },
                checkpoint,
            )
            evaluation = root / "evaluation.json"
            evaluation.write_text(
                json.dumps({"modelVersion": "release-test"}), encoding="utf-8"
            )
            output = module.prepare_release(
                checkpoint=checkpoint,
                evaluation=evaluation,
                version="0.2.0",
                output_root=root / "dist",
                published_at="2026-08-25T00:00:00Z",
            )
            expected = {
                "agerbot-release.json",
                "agerbot-model-0.2.0.pt",
                "agerbot-evaluation-0.2.0.json",
                "checksums-sha256.txt",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            manifest = json.loads((output / "agerbot-release.json").read_text())
            self.assertEqual(manifest["release"]["tag"], "model-v0.2.0")
            self.assertEqual(manifest["artifact"]["sizeBytes"], checkpoint.stat().st_size)
            self.assertEqual(len(manifest["artifact"]["sha256"]), 64)
            self.assertIn("agerbot-model-0.2.0.pt", (output / "checksums-sha256.txt").read_text())


if __name__ == "__main__":
    unittest.main()
