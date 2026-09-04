"""Tokenizadores pequeños y transparentes (byte, char, BPE)."""

from __future__ import annotations

from pathlib import Path


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


class BpeTokenizer:
    """BPE byte-level (HuggingFace tokenizers), vocab entrenado solo en corpus local."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        # vocab_size estable: incluye tokens especiales del trainer
        self._vocab_size = int(tokenizer.get_vocab_size())

    @classmethod
    def from_file(cls, path: str | Path) -> "BpeTokenizer":
        from tokenizers import Tokenizer

        tokenizer_path = Path(path)
        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"No se encontró tokenizer BPE en {path}")
        return cls(Tokenizer.from_file(str(tokenizer_path)))

    @classmethod
    def from_json(cls, tokenizer_json: str) -> "BpeTokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_str(tokenizer_json))

    @classmethod
    def train_from_files(
        cls,
        files: list[str | Path],
        *,
        vocab_size: int = 6144,
        min_frequency: int = 2,
    ) -> "BpeTokenizer":
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer

        if vocab_size < 256:
            raise ValueError("vocab_size BPE demasiado pequeño")
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<unk>", "<pad>"],
            show_progress=False,
        )
        paths = [str(Path(path)) for path in files]
        tokenizer.train(files=paths, trainer=trainer)
        return cls(tokenizer)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "tokenizer.json"
        self._tokenizer.save(str(target))
        meta = directory / "meta.json"
        meta.write_text(
            '{"type":"bpe","version":1,"vocab_size":%d}\n' % self.vocab_size,
            encoding="utf-8",
        )
        return target

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text).ids)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def to_dict(self) -> dict:
        return {
            "type": "bpe",
            "version": 1,
            "vocab_size": self.vocab_size,
            "tokenizer_json": self._tokenizer.to_str(),
        }


TokenizerAny = ByteTokenizer | CharTokenizer | BpeTokenizer


def tokenizer_from_dict(config: dict | str) -> TokenizerAny:
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
    if tokenizer_type == "bpe" and version == 1:
        if "tokenizer_json" in config and isinstance(config["tokenizer_json"], str):
            return BpeTokenizer.from_json(config["tokenizer_json"])
        path = config.get("path")
        if isinstance(path, str):
            return BpeTokenizer.from_file(path)
        raise ValueError("El tokenizador bpe necesita tokenizer_json o path")
    raise ValueError("Configuración de tokenizador desconocida")


def tokenizer_identifier(config: dict | str) -> str:
    if config == "byte-v1":
        return "byte-v1"
    if isinstance(config, dict):
        tokenizer_type = config.get("type")
        version = config.get("version")
        if tokenizer_type in {"byte", "char", "bpe"} and version == 1:
            return f"{tokenizer_type}-v1"
    raise ValueError("Configuración de tokenizador desconocida")


def build_tokenizer_from_config(config: dict, corpus_text: str | None = None) -> TokenizerAny:
    """Construye el tokenizador a partir del bloque configs[*].tokenizer."""
    tokenizer_cfg = config.get("tokenizer", {"type": "byte"})
    tokenizer_type = tokenizer_cfg.get("type", "byte")
    if tokenizer_type == "char":
        if corpus_text is None:
            raise ValueError("El tokenizador char requiere el texto del corpus")
        return CharTokenizer.fit(corpus_text)
    if tokenizer_type == "byte":
        return ByteTokenizer()
    if tokenizer_type == "bpe":
        path = tokenizer_cfg.get("path")
        if path:
            return BpeTokenizer.from_file(path)
        if "tokenizer_json" in tokenizer_cfg:
            return BpeTokenizer.from_json(tokenizer_cfg["tokenizer_json"])
        raise ValueError("Config bpe sin path ni tokenizer_json")
    raise ValueError(f"Tokenizador no soportado: {tokenizer_type}")
