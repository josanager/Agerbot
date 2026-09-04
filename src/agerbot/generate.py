"""Carga un checkpoint y genera una continuación de texto."""

from __future__ import annotations

import argparse

import torch

from .model import Agerbot, ModelConfig
from .runtime import load_checkpoint, select_device
from .tokenizer import tokenizer_from_dict

# Marcadores de cambio de turno en el formato de diálogo del corpus.
CHAT_TURN_MARKERS = (
    "\nUsuario:",
    "\nPregunta:",
    "\nConversación:",
    "\nConsulta:",
    "\nChat:",
    "\nInteracción:",
    "\nAgerbot:",
)


def normalize_chat_text(text: str) -> str:
    """Normaliza finales de línea para que \\r no se convierta en �."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def trim_assistant_completion(text: str) -> str:
    """Corta la generación en el siguiente turno, sin dejar eco ni cortes raros.

    El corpus usa una línea por turno, así que un salto de línea suele marcar
    el final de la respuesta. Priorizamos marcadores de hablante por si el
    modelo continúa el diálogo en la misma ráfaga.
    """
    content = normalize_chat_text(text)
    cut_at: int | None = None
    for marker in CHAT_TURN_MARKERS:
        index = content.find(marker)
        if index != -1 and (cut_at is None or index < cut_at):
            cut_at = index
    if cut_at is not None:
        content = content[:cut_at]
    else:
        paragraph = content.find("\n\n")
        if paragraph != -1:
            content = content[:paragraph]
        else:
            newline = content.find("\n")
            if newline != -1:
                content = content[:newline]
    return content.strip()


def generate(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    requested_device: str,
    *,
    completion_only: bool = True,
    trim_chat: bool = True,
) -> str:
    device = select_device(requested_device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = tokenizer_from_dict(checkpoint.get("tokenizer", "byte-v1"))
    model = Agerbot(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt = normalize_chat_text(prompt)
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("El prompt no puede estar vacío")
    inputs = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    stop_token_ids: set[int] | None = None
    if trim_chat:
        newline_ids = set(tokenizer.encode("\n"))
        if newline_ids:
            stop_token_ids = newline_ids
    output = model.generate(
        inputs,
        max_new_tokens,
        temperature,
        top_k,
        stop_token_ids=stop_token_ids,
    )
    token_ids = output[0].tolist()
    if completion_only:
        token_ids = token_ids[len(prompt_tokens) :]
    decoded = tokenizer.decode(token_ids)
    if trim_chat:
        return trim_assistant_completion(decoded)
    return decoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--full-sequence",
        action="store_true",
        help="Incluye el prompt en la salida (comportamiento antiguo).",
    )
    parser.add_argument(
        "--no-trim-chat",
        action="store_true",
        help="No corta en límites de turno Usuario/Agerbot.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        generate(
            args.checkpoint,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.device,
            completion_only=not args.full_sequence,
            trim_chat=not args.no_trim_chat,
        )
    )


if __name__ == "__main__":
    main()
