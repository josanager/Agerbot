"""Carga un checkpoint y genera una continuación de texto."""

from __future__ import annotations

import argparse

import torch

from .model import Agerbot, ModelConfig
from .runtime import select_device
from .tokenizer import tokenizer_from_dict


def generate(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    requested_device: str,
) -> str:
    device = select_device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = tokenizer_from_dict(checkpoint.get("tokenizer", "byte-v1"))
    model = Agerbot(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("El prompt no puede estar vacío")
    inputs = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    output = model.generate(inputs, max_new_tokens, temperature, top_k)
    return tokenizer.decode(output[0].tolist())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
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
        )
    )


if __name__ == "__main__":
    main()
