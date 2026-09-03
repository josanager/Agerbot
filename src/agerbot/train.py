"""CLI de entrenamiento local de Agerbot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .data import load_corpus, random_batch, split_corpus
from .model import Agerbot, ModelConfig
from .runtime import load_json, save_checkpoint, seed_everything, select_device
from .tokenizer import ByteTokenizer, CharTokenizer


@torch.inference_mode()
def estimate_loss(
    model: Agerbot,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for name, tokens in (("train", train_tokens), ("val", val_tokens)):
        values = []
        for _ in range(batches):
            inputs, targets = random_batch(
                tokens, batch_size, model.config.context_length, device
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            values.append(loss.item())
        losses[name] = sum(values) / len(values)
    model.train()
    return losses


def train(config_path: str, resume: str | None = None) -> Path:
    config = load_json(config_path)
    seed_everything(config["seed"])
    device = select_device(config.get("device", "auto"))
    corpus_text = Path(config["data_path"]).read_text(encoding="utf-8")
    tokenizer_type = config.get("tokenizer", {}).get("type", "byte")
    if tokenizer_type == "char":
        tokenizer = CharTokenizer.fit(corpus_text)
    elif tokenizer_type == "byte":
        tokenizer = ByteTokenizer()
    else:
        raise ValueError(f"Tokenizador no soportado: {tokenizer_type}")
    all_tokens = load_corpus(config["data_path"], tokenizer)
    train_tokens, val_tokens = split_corpus(all_tokens, config["train_fraction"])

    model_values = {**config["model"], "vocab_size": tokenizer.vocab_size}
    model_config = ModelConfig(**model_values)
    model = Agerbot(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    start_step = 0
    best_val_loss = float("inf")
    evaluations_without_improvement = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)

    checkpoint_dir = Path(config["checkpoint_dir"])
    latest_path = checkpoint_dir / "latest.pt"
    model.train()
    started = time.perf_counter()
    max_duration = config.get("max_duration_seconds")
    deadline = started + max_duration if max_duration else None
    print(
        f"device={device} parameters={model.parameter_count():,} "
        f"tokens={len(all_tokens):,}"
    )

    for step in range(start_step, config["max_steps"]):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulation = config["gradient_accumulation_steps"]
        for _ in range(accumulation):
            inputs, targets = random_batch(
                train_tokens,
                config["batch_size"],
                model_config.context_length,
                device,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            (loss / accumulation).backward()
            accumulated_loss += loss.item() / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()

        should_evaluate = step == start_step or (step + 1) % config["eval_interval"] == 0
        if should_evaluate:
            losses = estimate_loss(
                model,
                train_tokens,
                val_tokens,
                config["batch_size"],
                config["eval_batches"],
                device,
            )
            elapsed = time.perf_counter() - started
            print(
                f"step={step + 1:04d} train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} elapsed={elapsed:.1f}s"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                evaluations_without_improvement = 0
                best_payload = {
                    "format_version": 1,
                    "step": step,
                    "model_config": model_config.to_dict(),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "training_config": config,
                    "tokenizer": tokenizer.to_dict(),
                    "best_val_loss": best_val_loss,
                }
                save_checkpoint(checkpoint_dir / "best.pt", best_payload)
                print(f"best_checkpoint={checkpoint_dir / 'best.pt'}")
            else:
                evaluations_without_improvement += 1

        should_save = (step + 1) % config["save_interval"] == 0
        if should_save or step + 1 == config["max_steps"]:
            payload = {
                "format_version": 1,
                "step": step,
                "model_config": model_config.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "training_config": config,
                "tokenizer": tokenizer.to_dict(),
                "best_val_loss": best_val_loss,
            }
            save_checkpoint(latest_path, payload)
            save_checkpoint(checkpoint_dir / f"step-{step + 1:06d}.pt", payload)
            print(f"checkpoint={latest_path}")

        patience = config.get("early_stopping_patience")
        if patience and evaluations_without_improvement >= patience:
            print(
                f"early_stop=validation_without_improvement "
                f"evaluations={evaluations_without_improvement}"
            )
            break

        if deadline is not None and time.perf_counter() >= deadline:
            payload = {
                "format_version": 1,
                "step": step,
                "model_config": model_config.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "training_config": config,
                "tokenizer": tokenizer.to_dict(),
                "best_val_loss": best_val_loss,
            }
            save_checkpoint(latest_path, payload)
            save_checkpoint(checkpoint_dir / f"step-{step + 1:06d}.pt", payload)
            print(
                f"time_limit_reached={max_duration}s step={step + 1} "
                f"checkpoint={latest_path}"
            )
            break

    return latest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--resume", help="Checkpoint desde el que continuar")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
