"""Tests for learn-v1 held-out bank and memorization overlap helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "evaluation" / "learn_bank_v1.jsonl"


class LearnBankFileTests(unittest.TestCase):
    def test_bank_exists_and_has_40_items(self) -> None:
        self.assertTrue(BANK.is_file(), f"missing {BANK}")
        items = [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(items), 40)
        ids = [it["id"] for it in items]
        self.assertEqual(len(ids), len(set(ids)))
        required = {"id", "user", "rubric"}
        for it in items:
            self.assertTrue(required.issubset(it.keys()), it)
            self.assertTrue(it["user"].strip())

    def test_bank_includes_friendship_probes_and_zumba_traps(self) -> None:
        text = BANK.read_text(encoding="utf-8")
        self.assertIn("somos amigos?", text)
        self.assertIn("me quieres como un hermano?", text)
        self.assertIn("joke_trap", text)
        self.assertIn("Zum-ba", text)


class OverlapMetricTests(unittest.TestCase):
    def test_lcs_and_ngram_detect_copy(self) -> None:
        # Import from eval script module path
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "eval_learn_bank", ROOT / "scripts" / "eval_learn_bank.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        a = "sí, aquí soy tu compañero de charla"
        b = "claro. sí, aquí soy tu compañero de charla siempre"
        self.assertGreaterEqual(mod.longest_common_substring_len(a, b), 20)

        corpus = "alpha beta gamma sí, aquí soy tu compañero de charla omega"
        ratio = mod.ngram_overlap_ratio(a, corpus, n=8)
        self.assertGreater(ratio, 0.5)

        unrelated = "el ceviche lleva limón y pescado fresco"
        ratio2 = mod.ngram_overlap_ratio(unrelated, corpus, n=8)
        self.assertLess(ratio2, 0.2)

        memo = mod.memorization_metrics(
            a,
            mod.normalize(corpus),
            [mod.normalize("sí, aquí soy tu compañero de charla")],
        )
        self.assertTrue(memo["memorized"])
        memo2 = mod.memorization_metrics(
            unrelated,
            mod.normalize(corpus),
            [mod.normalize("sí, aquí soy tu compañero de charla")],
        )
        self.assertFalse(memo2["memorized"])


if __name__ == "__main__":
    unittest.main()
