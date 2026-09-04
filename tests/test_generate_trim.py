import unittest

from agerbot.generate import normalize_chat_text, trim_assistant_completion


class GenerateTrimTests(unittest.TestCase):
    def test_normalize_strips_crlf_replacement_source(self) -> None:
        self.assertEqual(normalize_chat_text("hola\r\nmundo\rfin"), "hola\nmundo\nfin")

    def test_trim_stops_at_next_user_turn(self) -> None:
        raw = "Qué bueno verte 🙂\nUsuario: otra cosa\nAgerbot: no debería verse"
        self.assertEqual(trim_assistant_completion(raw), "Qué bueno verte 🙂")

    def test_trim_stops_at_first_newline_for_single_line_turns(self) -> None:
        raw = "Aquí estoy.\nresto que mezcla temas"
        self.assertEqual(trim_assistant_completion(raw), "Aquí estoy.")

    def test_trim_prefers_speaker_marker_before_internal_noise(self) -> None:
        raw = "Sigo contigo.\nAgerbot: eco raro"
        self.assertEqual(trim_assistant_completion(raw), "Sigo contigo.")

    def test_trim_does_not_echo_empty_after_only_markers(self) -> None:
        self.assertEqual(trim_assistant_completion("\nUsuario: hola"), "")


if __name__ == "__main__":
    unittest.main()
