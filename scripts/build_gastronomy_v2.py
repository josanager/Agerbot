"""Amplía de forma determinista el corpus v1 sin inventar hechos nuevos."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path


PAIR_PATTERN = re.compile(
    r"Pregunta:\s*(?P<question>.+?)\nRespuesta:\s*(?P<answer>.+?)(?=\n\n|\Z)",
    re.DOTALL,
)

FORMATS = [
    "Pregunta: {question}\nRespuesta: {answer}",
    "Usuario: {question}\nAgerbot: {answer}",
    "Consulta sobre gastronomía peruana: {question}\nRespuesta de Agerbot: {answer}",
    "Una persona pregunta: {question}\nAgerbot responde: {answer}",
    "Tema: cocina peruana. Pregunta: {question}\nRespuesta: {answer}",
    "Conversación gastronómica.\nPregunta: {question}\nRespuesta breve: {answer}",
    "Ayuda culinaria peruana.\nUsuario: {question}\nAsistente: {answer}",
    "Sobre el Perú, alguien consulta: {question}\nLa respuesta es: {answer}",
]


def build(source: Path, destination: Path, repetitions: int, seed: int) -> None:
    text = source.read_text(encoding="utf-8").strip()
    pairs = [
        (match.group("question").strip(), " ".join(match.group("answer").split()))
        for match in PAIR_PATTERN.finditer(text)
    ]
    if not pairs:
        raise ValueError("No se encontraron pares Pregunta/Respuesta")

    randomizer = random.Random(seed)
    augmented: list[str] = []
    for repetition in range(repetitions):
        shuffled = pairs.copy()
        randomizer.shuffle(shuffled)
        for index, (question, answer) in enumerate(shuffled):
            template = FORMATS[(index + repetition) % len(FORMATS)]
            augmented.append(template.format(question=question, answer=answer))
            augmented.append(f"Dato de gastronomía peruana: {answer}")
    randomizer.shuffle(augmented)
    blocks = [text, *augmented]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(
        f"pairs={len(pairs)} characters={destination.stat().st_size} "
        f"output={destination}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/gastronomia_peruana_v1.txt")
    parser.add_argument("--output", default="data/processed/gastronomia_peruana_v2.txt")
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    build(Path(args.source), Path(args.output), args.repetitions, args.seed)


if __name__ == "__main__":
    main()
