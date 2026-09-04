"""CLI de entrenamiento local de Agerbot."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

from .data import augment_multitarget_text, load_corpus, random_batch, split_corpus
from .model import Agerbot, ModelConfig
from .runtime import load_checkpoint, load_json, save_checkpoint, seed_everything, select_device
from .tokenizer import build_tokenizer_from_config


def _load_eval_learn_bank():
    """Carga scripts/eval_learn_bank.py sin instalarlo como paquete."""
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "eval_learn_bank.py"
    spec = importlib.util.spec_from_file_location("agerbot_eval_learn_bank", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def run_bank_score(
    model: Agerbot,
    tokenizer,
    *,
    bank_path: Path,
    train_corpus: Path,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    limit: int | None,
) -> dict:
    """Evalúa el modelo en memoria contra el bank (sin recargar checkpoint)."""
    eval_mod = _load_eval_learn_bank()
    items = eval_mod.load_bank(bank_path)
    if limit is not None:
        items = items[:limit]
    corpus_text = train_corpus.read_text(encoding="utf-8") if train_corpus.is_file() else ""
    corpus_norm = eval_mod.normalize(corpus_text)
    assistant_lines = eval_mod.load_assistant_lines(corpus_text)
    passed = 0
    memorized_n = 0
    for item in items:
        answer = eval_mod.generate_answer(
            model,
            tokenizer,
            item["user"],
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        rubric = eval_mod.score_rubric(answer, item)
        memo = eval_mod.memorization_metrics(answer, corpus_norm, assistant_lines)
        if rubric["passed"]:
            passed += 1
        if memo["memorized"]:
            memorized_n += 1
    n = max(len(items), 1)
    accuracy = passed / n
    memorization_rate = memorized_n / n
    return {
        "accuracy": round(accuracy, 4),
        "memorization_rate": round(memorization_rate, 4),
        "score": round(accuracy - memorization_rate, 4),
        "passed": passed,
        "memorized": memorized_n,
        "n_items": len(items),
    }


def bank_objective(accuracy: float, memorization_rate: float, mode: str = "acc_minus_mem") -> float:
    """Ranking score for held-out bank checkpoint selection.

    - acc_minus_mem: maximize accuracy - memorization_rate
    - gate_prefer: hard-bonus any (acc>0.475 & mem<0.55); else max acc with soft mem penalty
    """
    if mode == "gate_prefer":
        if accuracy > 0.475 and memorization_rate < 0.55:
            return 10.0 + accuracy - memorization_rate
        mem_pen = max(0.0, memorization_rate - 0.40) * 2.0
        return accuracy - mem_pen
    return accuracy - memorization_rate


def train(config_path: str, resume: str | None = None) -> Path:
    config = load_json(config_path)
    seed_everything(config["seed"])
    device = select_device(config.get("device", "auto"))
    corpus_text = Path(config["data_path"]).read_text(encoding="utf-8")
    multitarget = config.get("multitarget") or {}
    if multitarget.get("enabled", False):
        before = len(corpus_text)
        corpus_text = augment_multitarget_text(
            corpus_text,
            seed=config["seed"],
            max_extra_turns=int(multitarget.get("max_extra_turns", 4000)),
            min_variants=int(multitarget.get("min_variants", 2)),
            max_remixes_per_user=int(multitarget.get("max_remixes_per_user", 2)),
            dedupe_near_identical=bool(multitarget.get("dedupe_near_identical", True)),
            dedupe_threshold=float(multitarget.get("dedupe_threshold", 0.90)),
        )
        print(
            f"multitarget=on chars_before={before:,} chars_after={len(corpus_text):,}"
        )

    tokenizer = build_tokenizer_from_config(config, corpus_text)
    all_tokens = load_corpus(config["data_path"], tokenizer, text=corpus_text)
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
    best_bank_score = float("-inf")
    evaluations_without_improvement = 0
    bank_without_improvement = 0
    if resume:
        checkpoint = load_checkpoint(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        best_bank_score = checkpoint.get("best_bank_score", best_bank_score)

    checkpoint_dir = Path(config["checkpoint_dir"])
    latest_path = checkpoint_dir / "latest.pt"
    model.train()
    started = time.perf_counter()
    max_duration = config.get("max_duration_seconds")
    deadline = started + max_duration if max_duration else None
    store_fp16 = bool(config.get("checkpoint_float16", True))
    bank_cfg = config.get("bank_selection") or {}
    bank_enabled = bool(bank_cfg.get("enabled", False))
    bank_interval = int(bank_cfg.get("interval", 200))
    bank_path = Path(bank_cfg.get("bank_path", "data/evaluation/learn_bank_v1.jsonl"))
    bank_train_corpus = Path(
        bank_cfg.get("train_corpus", "data/processed/agerbot_social_v2.txt")
    )
    bank_limit = bank_cfg.get("limit")
    bank_max_new = int(bank_cfg.get("max_new_tokens", 80))
    bank_temp = float(bank_cfg.get("temperature", 0.7))
    bank_top_k = bank_cfg.get("top_k", 40)
    bank_patience = bank_cfg.get("patience")
    select_by_bank = bank_enabled  # best.pt keyed by bank score when enabled

    print(
        f"device={device} parameters={model.parameter_count():,} "
        f"tokens={len(all_tokens):,} vocab={tokenizer.vocab_size} "
        f"tokenizer={config.get('tokenizer', {}).get('type', 'byte')} "
        f"bank_selection={bank_enabled}"
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
                if not select_by_bank:
                    best_payload = {
                        "format_version": 1,
                        "step": step,
                        "model_config": model_config.to_dict(),
                        "model_state": model.state_dict(),
                        "training_config": config,
                        "tokenizer": tokenizer.to_dict(),
                        "best_val_loss": best_val_loss,
                        "best_bank_score": best_bank_score,
                        "selection_metric": "val_loss",
                    }
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        best_payload,
                        store_model_float16=store_fp16,
                        include_optimizer=False,
                    )
                    print(f"best_checkpoint={checkpoint_dir / 'best.pt'}")
            else:
                evaluations_without_improvement += 1

            # Always keep a val-loss snapshot for diagnostics
            if select_by_bank and losses["val"] <= best_val_loss + 1e-12:
                save_checkpoint(
                    checkpoint_dir / "best_val_loss.pt",
                    {
                        "format_version": 1,
                        "step": step,
                        "model_config": model_config.to_dict(),
                        "model_state": model.state_dict(),
                        "training_config": config,
                        "tokenizer": tokenizer.to_dict(),
                        "best_val_loss": best_val_loss,
                        "selection_metric": "val_loss",
                    },
                    store_model_float16=store_fp16,
                    include_optimizer=False,
                )

        # Held-out bank selection (accuracy - memorization_rate), not val_loss
        should_bank = bank_enabled and (
            (step + 1) % bank_interval == 0 or step == start_step
        )
        if should_bank:
            was_training = model.training
            model.eval()
            metrics = run_bank_score(
                model,
                tokenizer,
                bank_path=bank_path,
                train_corpus=bank_train_corpus,
                device=device,
                max_new_tokens=bank_max_new,
                temperature=bank_temp,
                top_k=int(bank_top_k) if bank_top_k is not None else None,
                limit=int(bank_limit) if bank_limit is not None else None,
            )
            if was_training:
                model.train()
            score_mode = str(bank_cfg.get("score_mode", "acc_minus_mem"))
            objective = bank_objective(
                metrics["accuracy"], metrics["memorization_rate"], mode=score_mode
            )
            metrics = {**metrics, "objective": round(objective, 4), "score_mode": score_mode}
            # Keep legacy 'score' as acc-mem; rank by objective
            print(
                f"bank_eval step={step + 1:04d} acc={metrics['accuracy']} "
                f"mem={metrics['memorization_rate']} score={metrics['score']} "
                f"objective={metrics['objective']} mode={score_mode} "
                f"n={metrics['n_items']}"
            )
            # Lean candidate for post-hoc full-bank reselect
            save_checkpoint(
                checkpoint_dir / f"bankcand-{step + 1:06d}.pt",
                {
                    "format_version": 1,
                    "step": step,
                    "model_config": model_config.to_dict(),
                    "model_state": model.state_dict(),
                    "training_config": config,
                    "tokenizer": tokenizer.to_dict(),
                    "best_val_loss": best_val_loss,
                    "bank_metrics": metrics,
                    "selection_metric": "bank_score",
                },
                store_model_float16=store_fp16,
                include_optimizer=False,
            )
            if metrics.get("objective", metrics["score"]) > best_bank_score:
                best_bank_score = float(metrics.get("objective", metrics["score"]))
                bank_without_improvement = 0
                best_payload = {
                    "format_version": 1,
                    "step": step,
                    "model_config": model_config.to_dict(),
                    "model_state": model.state_dict(),
                    "training_config": config,
                    "tokenizer": tokenizer.to_dict(),
                    "best_val_loss": best_val_loss,
                    "best_bank_score": best_bank_score,
                    "bank_metrics": metrics,
                    "selection_metric": "bank_score",
                }
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    best_payload,
                    store_model_float16=store_fp16,
                    include_optimizer=False,
                )
                save_checkpoint(
                    checkpoint_dir / "best_bank.pt",
                    best_payload,
                    store_model_float16=store_fp16,
                    include_optimizer=False,
                )
                print(
                    f"best_bank_checkpoint={checkpoint_dir / 'best.pt'} "
                    f"score={best_bank_score}"
                )
            else:
                bank_without_improvement += 1

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
                "best_bank_score": best_bank_score,
            }
            save_checkpoint(
                latest_path, payload, store_model_float16=store_fp16, include_optimizer=True
            )
            save_checkpoint(
                checkpoint_dir / f"step-{step + 1:06d}.pt",
                payload,
                store_model_float16=store_fp16,
                include_optimizer=True,
            )
            print(f"checkpoint={latest_path}")

        if select_by_bank and bank_patience:
            if bank_without_improvement >= int(bank_patience):
                print(
                    f"early_stop=bank_without_improvement "
                    f"evaluations={bank_without_improvement} "
                    f"best_bank_score={best_bank_score}"
                )
                break
        else:
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
                "best_bank_score": best_bank_score,
            }
            save_checkpoint(
                latest_path, payload, store_model_float16=store_fp16, include_optimizer=True
            )
            save_checkpoint(
                checkpoint_dir / f"step-{step + 1:06d}.pt",
                payload,
                store_model_float16=store_fp16,
                include_optimizer=True,
            )
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
