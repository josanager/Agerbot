"""Tokenizadores pequeños, transparentes y sin dependencias externas."""

from __future__ import annotations


class ByteTokenizer:
    """Convierte texto UTF-8 a bytes y viceversa.

    Los 256 bytes posibles forman el vocabulario completo. Esto evita entrenar o
    descargar un tokenizador y permite representar cualquier texto UTF-8.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "byte", "version": 1}


class CharTokenizer:
    """Vocabulario de caracteres aprendido exclusivamente del corpus local."""

    def __init__(self, characters: list[str], *, preserve_order: bool = False) -> None:
        if preserve_order:
            if not characters or characters[0] != "�":
                raise ValueError("El tokenizador char debe reservar � como primer símbolo")
            if any(not isinstance(char, str) or len(char) != 1 for char in characters):
                raise ValueError("Cada entrada del tokenizador char debe ser un carácter")
            if len(set(characters)) != len(characters):
                raise ValueError("El tokenizador char contiene caracteres duplicados")
            self.characters = list(characters)
        else:
            unique = sorted(set(characters))
            self.characters = ["�"] + [char for char in unique if char != "�"]
        self._char_to_id = {char: index for index, char in enumerate(self.characters)}

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("No se puede ajustar el tokenizador a un texto vacío")
        return cls(list(text))

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        unknown = self._char_to_id["�"]
        return [self._char_to_id.get(char, unknown) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self.characters[token] if 0 <= token < len(self.characters) else "�"
            for token in tokens
        )

    def to_dict(self) -> dict:
        return {"type": "char", "version": 1, "characters": self.characters}


def tokenizer_from_dict(config: dict | str) -> ByteTokenizer | CharTokenizer:
    if config == "byte-v1":
        return ByteTokenizer()
    if not isinstance(config, dict):
        raise ValueError("Configuración de tokenizador desconocida")
    tokenizer_type = config.get("type")
    version = config.get("version")
    if tokenizer_type == "byte" and version == 1:
        return ByteTokenizer()
    if tokenizer_type == "char" and version == 1:
        characters = config.get("characters")
        if not isinstance(characters, list):
            raise ValueError("El tokenizador char no contiene su vocabulario")
        return CharTokenizer(characters, preserve_order=True)
    raise ValueError("Configuración de tokenizador desconocida")


def tokenizer_identifier(config: dict | str) -> str:
    if config == "byte-v1":
        return "byte-v1"
    if isinstance(config, dict):
        tokenizer_type = config.get("type")
        version = config.get("version")
        if tokenizer_type in {"byte", "char"} and version == 1:
            return f"{tokenizer_type}-v1"
    raise ValueError("Configuración de tokenizador desconocida")
