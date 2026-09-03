import unittest

import torch

from agerbot import Agerbot, ByteTokenizer, CharTokenizer, ModelConfig
from agerbot.data import random_batch, split_corpus


class TokenizerTests(unittest.TestCase):
    def test_utf8_round_trip(self) -> None:
        tokenizer = ByteTokenizer()
        text = "¡Hola, pequeña IA! 🤖"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_character_tokenizer_round_trip(self) -> None:
        text = "Ceviche peruano: limón, pescado y ají."
        tokenizer = CharTokenizer.fit(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0
        )
        self.model = Agerbot(self.config)

    def test_forward_and_loss(self) -> None:
        tokens = torch.randint(0, 256, (2, 16))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 16, 256))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_generation_extends_sequence(self) -> None:
        tokens = torch.tensor([[80, 114, 101]], dtype=torch.long)
        generated = self.model.generate(tokens, max_new_tokens=5)
        self.assertEqual(tuple(generated.shape), (1, 8))

    def test_data_split_and_batch(self) -> None:
        tokens = torch.arange(200, dtype=torch.long)
        train, val = split_corpus(tokens, 0.8)
        self.assertEqual((len(train), len(val)), (160, 40))
        inputs, targets = random_batch(train, 3, 16, torch.device("cpu"))
        self.assertEqual(tuple(inputs.shape), (3, 16))
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))


if __name__ == "__main__":
    unittest.main()
