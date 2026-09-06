import tempfile
import unittest
from pathlib import Path

import torch

from agerbot.data import (
    augment_multitarget_text,
    dedupe_near_identical_replies,
    parse_dialogue_pairs,
    replies_near_duplicate,
)
from agerbot.model import Agerbot, ModelConfig
from agerbot.runtime import load_checkpoint, save_checkpoint
from agerbot.tokenizer import BpeTokenizer, tokenizer_from_dict, tokenizer_identifier


SAMPLE = """Usuario: eres mi amigo?
Agerbot: Sí. Amistad de chat: respeto, escucha y presencia.

Usuario: eres mi amigo?
Agerbot: Sí 🙂 Aquí soy tu compañero de charla.

Usuario: hola
Agerbot: Hola, ¿qué tal?
"""


class DensifyV1Tests(unittest.TestCase):
    def test_multitarget_only_uses_existing_replies(self) -> None:
        pairs = set(parse_dialogue_pairs(SAMPLE))
        aug = augment_multitarget_text(SAMPLE, seed=0, max_extra_turns=10)
        for user, reply in parse_dialogue_pairs(aug):
            self.assertTrue(any(u == user for u, _ in pairs) or user in SAMPLE)
            self.assertIn(reply, {r for _, r in pairs})

    def test_bpe_roundtrip_and_checkpoint_fp16(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tok_path = root / "data" / "tokenizers" / "densify-v1"
        if not tok_path.exists():
            self.skipTest("densify-v1 tokenizer not trained")
        tokenizer = BpeTokenizer.from_file(tok_path)
        text = "Usuario: hola\nAgerbot: Hola"
        self.assertIsInstance(tokenizer.decode(tokenizer.encode(text)), str)
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=32,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )
        model = Agerbot(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            save_checkpoint(
                path,
                {
                    "format_version": 1,
                    "model_config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "tokenizer": tokenizer.to_dict(),
                },
                include_optimizer=False,
            )
            raw = torch.load(path, map_location="cpu", weights_only=True)
            first = next(iter(raw["model_state"].values()))
            self.assertEqual(first.dtype, torch.float16)
            loaded = load_checkpoint(path, weights_only=True)
            restored = tokenizer_from_dict(loaded["tokenizer"])
            self.assertEqual(tokenizer_identifier(loaded["tokenizer"]), "bpe-v1")
            self.assertEqual(restored.vocab_size, tokenizer.vocab_size)
            model2 = Agerbot(ModelConfig(**loaded["model_config"]))
            model2.load_state_dict(loaded["model_state"])
            weight = next(model2.parameters())
            self.assertEqual(weight.dtype, torch.float32)


    def test_near_identical_reply_dedupe(self) -> None:
        a = "Sí. Amistad de chat: respeto, escucha y presencia."
        b = "Sí! Amistad de chat: respeto, escucha y presencia 🙂"
        self.assertTrue(replies_near_duplicate(a, b, threshold=0.88))
        kept = dedupe_near_identical_replies([a, b, "Hola distinta respuesta"], threshold=0.88)
        self.assertEqual(len(kept), 2)

    def test_multitarget_remix_cap(self) -> None:
        aug = augment_multitarget_text(
            SAMPLE, seed=1, max_extra_turns=10, max_remixes_per_user=4
        )
        # No inventa respuestas nuevas
        originals = {r for _, r in parse_dialogue_pairs(SAMPLE)}
        for _, reply in parse_dialogue_pairs(aug):
            self.assertIn(reply, originals)


if __name__ == "__main__":
    unittest.main()
