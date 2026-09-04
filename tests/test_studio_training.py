import unittest

from agerbot.server import _ensure_dialogue_format, _merge_training_corpus, TrainingManager


class DialogueWrapTests(unittest.TestCase):
    def test_keeps_usuario_agerbot_blocks(self):
        text = "Usuario: hola\nAgerbot: qué tal\n"
        self.assertIn("Usuario: hola", _ensure_dialogue_format(text))

    def test_wraps_raw_text_as_one_dialogue(self):
        wrapped = _ensure_dialogue_format("el ceviche lleva limón")
        self.assertTrue(wrapped.startswith("Usuario:"))
        self.assertIn("Agerbot: el ceviche lleva limón", wrapped)


class MergeLearnsNotMemorizeTests(unittest.TestCase):
    def test_new_dialogues_are_not_repeated_hundreds_of_times(self):
        base = "Usuario: hola\nAgerbot: hola\n\nUsuario: 2+2\nAgerbot: 4\n"
        new = "Usuario: de qué color es el cielo\nAgerbot: Azul claro de día.\n"
        merged, base_n, new_n = _merge_training_corpus(base, new)
        self.assertGreater(base_n, 0)
        self.assertGreater(new_n, 0)
        copies = merged.count("de qué color es el cielo")
        self.assertEqual(copies, 2)

    def test_raw_paste_is_accepted(self):
        base = "Usuario: hola\nAgerbot: hola\n"
        merged, _, _ = _merge_training_corpus(base, "los gatos duermen muchas horas")
        self.assertIn("los gatos duermen muchas horas", merged)
        self.assertIn("Usuario:", merged)


class SameAgerbotSessionTests(unittest.TestCase):
    def test_start_training_without_server_fails_clearly(self):
        manager = TrainingManager(server=None)
        with self.assertRaises(Exception) as context:
            manager.start_training(
                "Usuario: hola\nAgerbot: hola, qué tal, cómo estás hoy",
                1,
                name="otra_cosa",
            )
        self.assertEqual(context.exception.code, "base_checkpoint_unavailable")
        self.assertEqual(manager.session_name, "")

    def test_wrapped_text_is_still_agerbot_dialogue(self):
        wrapped = _ensure_dialogue_format("dato nuevo sobre el ceviche")
        self.assertIn("Agerbot:", wrapped)
        self.assertNotIn("modelo_", wrapped)


if __name__ == "__main__":
    unittest.main()
