import unittest
from pathlib import Path


CORPUS_PATH = Path(__file__).parents[1] / "data/raw/agerbot_dialogues_simple_v7.txt"
VARIETY_CORPUS_PATH = Path(__file__).parents[1] / "data/raw/agerbot_dialogues_simple_v8.txt"
REASONING_CORPUS_PATH = Path(__file__).parents[1] / "data/raw/agerbot_dialogues_reasoning_v9.txt"
CONTEXT_CORPUS_PATH = Path(__file__).parents[1] / "data/raw/agerbot_dialogues_context_v10.txt"


class ConversationCorpusTests(unittest.TestCase):
    def test_simple_dialogues_cover_real_follow_ups(self) -> None:
        text = CORPUS_PATH.read_text(encoding="utf-8")
        blocks = [block for block in text.split("\n\n") if block.strip()]

        self.assertGreaterEqual(len(blocks), 30)
        self.assertIn("Usuario: cuéntame un chiste\nAgerbot:", text)
        self.assertIn("Usuario: no entendí\nAgerbot:", text)
        self.assertIn("Usuario: cuéntame otro\nAgerbot:", text)
        self.assertIn("Usuario: ¿sabes algo de código?\nAgerbot:", text)
        self.assertIn("Usuario: ¿sabes cuánto es 2 más 2?\nAgerbot:", text)
        self.assertIn("Usuario: ¿cuál es la mejor comida del mundo?\nAgerbot:", text)
        self.assertNotIn("Soy Agerbot", text)
        self.assertNotIn("Agerbot 0.3.0", text)

        for block in blocks:
            self.assertTrue(block.startswith("Usuario:"), block[:80])
            self.assertIn("\nAgerbot:", block, block[:80])

    def test_variety_corpus_covers_greetings_and_joke_feedback(self) -> None:
        text = VARIETY_CORPUS_PATH.read_text(encoding="utf-8")
        blocks = [block for block in text.split("\n\n") if block.strip()]

        self.assertGreaterEqual(len(blocks), 30)
        self.assertGreaterEqual(text.count("Usuario: hola"), 3)
        self.assertIn("Usuario: no me da risa\nAgerbot:", text)
        self.assertIn("Usuario: no me hizo gracia\nAgerbot:", text)
        self.assertIn("Usuario: ese chiste ya me lo contaste\nAgerbot:", text)

        for block in blocks:
            self.assertTrue(block.startswith("Usuario:"), block[:80])
            self.assertIn("\nAgerbot:", block, block[:80])

    def test_reasoning_corpus_preserves_multiturn_feedback(self) -> None:
        text = REASONING_CORPUS_PATH.read_text(encoding="utf-8")
        blocks = [block for block in text.split("\n\n") if block.strip()]

        self.assertGreaterEqual(len(blocks), 40)
        self.assertIn("Usuario: no me da risa\n", text)
        self.assertIn("Usuario: tampoco\n", text)
        self.assertIn("Usuario: y restar\n", text)
        self.assertIn("Usuario: no respondas por responder\n", text)
        self.assertNotIn("Soy Agerbot", text)
        self.assertNotIn("Agerbot 0.3.0", text)

        for block in blocks:
            self.assertTrue(block.startswith("Usuario:"), block[:80])
            self.assertIn("\nAgerbot:", block, block[:80])

    def test_context_corpus_keeps_follow_ups_in_one_dialogue(self) -> None:
        text = CONTEXT_CORPUS_PATH.read_text(encoding="utf-8")
        blocks = [block for block in text.split("\n\n") if block.strip()]

        self.assertGreaterEqual(len(blocks), 10)
        self.assertTrue(
            any(
                "Usuario: ¿sabes sumar?" in block
                and "Usuario: ¿y restar?" in block
                for block in blocks
            )
        )
        self.assertTrue(
            any(
                "Usuario: cuéntame un chiste" in block
                and "Usuario: no me da risa" in block
                and "Usuario: tampoco" in block
                for block in blocks
            )
        )
        self.assertNotIn("Soy Agerbot", text)
        self.assertNotIn("Agerbot 0.3.0", text)

        for block in blocks:
            self.assertTrue(block.startswith("Usuario:"), block[:80])
            self.assertGreaterEqual(block.count("\nAgerbot:"), 2, block[:80])


if __name__ == "__main__":
    unittest.main()
