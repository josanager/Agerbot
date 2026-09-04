"""Runtime HTTP local de Agerbot para aplicaciones de escritorio y estudio de entrenamiento."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from .data import load_corpus, random_batch, split_corpus
from .generate import normalize_chat_text, trim_assistant_completion
from .model import Agerbot, ModelConfig
from .runtime import save_checkpoint, select_device
from .tokenizer import tokenizer_from_dict, tokenizer_identifier
from .train import estimate_loss
from .web_ui import WEB_UI_HTML

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4318
RUNTIME_VERSION = "0.2.0"
DEFAULT_CHECKPOINT_PATH = "checkpoints/creativo-v5/best.pt"
ACTIVE_CHECKPOINT_FILE = Path("checkpoints/active_checkpoint.json")
MAX_BODY_BYTES = 4 * 1024 * 1024  # 4 MiB para soportar textos de entrenamiento
MAX_MESSAGE_BYTES = 16 * 1024
MAX_HISTORY_ITEMS = 32
MAX_HISTORY_BYTES = 64 * 1024
MAX_CONTEXT_HISTORY_ITEM_CHARS = 320
NEXT_CONTEXT_LENGTH = 1024
REFERENCE_BATCH_SIZE = 16
REFERENCE_CONTEXT_LENGTH = 256
FORBIDDEN_IDENTITY_RE = re.compile(
    r"\b(?:soy\s+agerbot(?:\s+0\.3\.0)?|agerbot\s+0\.3\.0)\b",
    re.IGNORECASE,
)
CONTINUATION_PREFIXES = (
    "otro",
    "otra",
    "cuéntame otro",
    "cuentame otro",
    "cuéntame otra",
    "cuentame otra",
    "más",
    "mas",
    "no entend",
    "no me gust",
    "no me da risa",
    "no me dio risa",
    "no me hizo gracia",
    "no me hace gracia",
    "tampoco",
    "ni eso",
    "nada que ver",
    "no funcionó",
    "pero ",
    "y ",
    "entonces",
    "eso",
    "ese",
    "esa",
    "repít",
    "repit",
)
NEW_QUESTION_PREFIXES = (
    "qué ",
    "que ",
    "quién",
    "quien",
    "cuál",
    "cual",
    "cuánto",
    "cuanto",
    "cómo ",
    "como ",
    "sabes ",
    "eres ",
    "dame ",
    "cuéntame ",
    "cuentame ",
    "escribe ",
    "haz ",
)


class RuntimeAPIError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            }
        }


@dataclass(frozen=True)
class CheckpointManifest:
    model_name: str
    model_version: str
    training_name: str
    tokenizer: str | None
    parameters: int | None
    context_length: int | None
    filename: str
    sha256: str
    size_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_checkpoint_path() -> str:
    """Devuelve el último modelo publicado, conservando el fallback original."""
    explicit = os.environ.get("AGERBOT_CHECKPOINT")
    if explicit:
        return explicit
    try:
        payload = json.loads(ACTIVE_CHECKPOINT_FILE.read_text(encoding="utf-8"))
        checkpoint = payload.get("checkpoint")
        if isinstance(checkpoint, str) and checkpoint.strip():
            candidate = Path(checkpoint).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.is_file():
                return str(candidate)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return DEFAULT_CHECKPOINT_PATH


def _persist_active_checkpoint(checkpoint_path: Path) -> None:
    ACTIVE_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ACTIVE_CHECKPOINT_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"checkpoint": str(checkpoint_path.resolve())}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(ACTIVE_CHECKPOINT_FILE)


def load_checkpoint_manifest(checkpoint_path: Path) -> CheckpointManifest:
    if not checkpoint_path.is_file():
        raise RuntimeAPIError(
            "checkpoint_missing",
            "No se encontró el checkpoint configurado de Agerbot.",
        )
    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeAPIError(
            "checkpoint_manifest_missing",
            "El checkpoint no tiene un manifiesto de verificación.",
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = payload["model"]
        checkpoint = payload["checkpoint"]
        stable_version = str(model["version"])
        training_name = str(model.get("trainingName", stable_version))
        manifest = CheckpointManifest(
            model_name=str(model["name"]),
            model_version=stable_version,
            training_name=training_name,
            tokenizer=str(model["tokenizer"]) if model.get("tokenizer") else None,
            parameters=int(model["parameters"]) if model.get("parameters") is not None else None,
            context_length=int(model["contextLength"]) if model.get("contextLength") is not None else None,
            filename=str(checkpoint["filename"]),
            sha256=str(checkpoint["sha256"]).lower(),
            size_bytes=int(checkpoint["sizeBytes"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeAPIError(
            "checkpoint_invalid", "El manifiesto del checkpoint no es válido."
        ) from error
    stat = checkpoint_path.stat()
    if manifest.filename != checkpoint_path.name or manifest.size_bytes != stat.st_size:
        raise RuntimeAPIError(
            "checkpoint_invalid",
            "El checkpoint no coincide con el tamaño declarado en su manifiesto.",
        )
    if len(manifest.sha256) != 64 or _sha256(checkpoint_path) != manifest.sha256:
        raise RuntimeAPIError(
            "checkpoint_invalid",
            "El checkpoint no coincide con la huella SHA-256 de su manifiesto.",
        )
    return manifest


def _total_memory_bytes() -> int:
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.total_physical)
        except (AttributeError, OSError):
            return 0
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


def _platform_name() -> str:
    return {"darwin": "macOS", "win32": "Windows", "linux": "Linux"}.get(
        sys.platform, platform.system() or "Unknown"
    )


def _training_blocks(text: str) -> list[str]:
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def _normalize_message(text: str) -> str:
    return " ".join(text.casefold().strip(" ¿¡!.,;:").split())


def _sanitize_training_block(block: str) -> str:
    """Elimina respuestas antiguas que revelan la identidad interna del modelo."""
    lines = block.splitlines()
    user_prompt = next(
        (line.split(":", 1)[1].strip() for line in lines if line.startswith("Usuario:")),
        "",
    )
    is_greeting = _normalize_message(user_prompt) in {
        "hola",
        "buenas",
        "hey",
        "qué tal",
        "que tal",
        "buenos días",
        "buenos dias",
        "buenas noches",
    }
    sanitized: list[str] = []
    for line in lines:
        if line.startswith("Agerbot:") and FORBIDDEN_IDENTITY_RE.search(line):
            line = (
                "Agerbot: ¡Hola! Qué gusto leerte. Soy una inteligencia artificial. ¿En qué te ayudo?"
                if is_greeting
                else "Agerbot: Soy una inteligencia artificial. No uso un nombre propio para presentarme."
            )
        sanitized.append(line)
    return "\n".join(sanitized).strip()


def _sanitize_generated_content(content: str) -> str:
    content = re.sub(r"^Agerbot:\s*", "", content.strip(), flags=re.IGNORECASE)
    content = re.sub(
        r"\bSoy\s+Agerbot(?:\s+0\.3\.0)?\b",
        "Soy una inteligencia artificial",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"\bAgerbot\s+0\.3\.0\b", "una inteligencia artificial", content, flags=re.IGNORECASE)
    return content.strip()


def _history_turns(history: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Agrupa el historial en turnos completos para no cortar una respuesta a la mitad."""
    turns: list[list[dict[str, str]]] = []
    index = 0
    while index < len(history):
        if (
            history[index]["role"] == "user"
            and index + 1 < len(history)
            and history[index + 1]["role"] == "assistant"
        ):
            turns.append(history[index : index + 2])
            index += 2
        else:
            turns.append([history[index]])
            index += 1
    return turns


def _needs_recent_history(history: list[dict[str, str]], message: str) -> bool:
    """Evita que un historial irrelevante distraiga al modelo pequeño."""
    if not history:
        return False
    normalized = message.casefold().strip(" ¿¡")
    if normalized.startswith(CONTINUATION_PREFIXES):
        return True
    if normalized.startswith(NEW_QUESTION_PREFIXES):
        return False
    last_item = history[-1]
    return (
        last_item["role"] == "assistant"
        and last_item["content"].rstrip().endswith("?")
        and len(normalized.split()) <= 12
    )


def _resolve_training_corpus(
    checkpoint: dict[str, Any], checkpoint_path: Path, training_name: str
) -> Path:
    candidates: list[Path] = []
    training_config = checkpoint.get("training_config")
    if isinstance(training_config, dict):
        data_path = training_config.get("data_path")
        if isinstance(data_path, str) and data_path.strip():
            candidates.append(Path(data_path).expanduser())

    candidates.append(Path("data/processed") / f"{training_name}.txt")
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if resolved.is_file():
            return resolved.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeAPIError(
        "base_corpus_missing",
        f"No se encontró el corpus del modelo principal ({searched}).",
    )


def _ensure_dialogue_format(text: str) -> str:
    """Si el pegado no trae turnos Usuario/Agerbot, lo envuelve en un diálogo."""
    stripped = text.strip()
    if not stripped:
        return stripped
    if any(line.startswith("Usuario:") for line in stripped.splitlines()):
        return stripped + "\n"
    return (
        "Usuario: Explícame esto con tus palabras, sin copiar el texto tal cual.\n"
        f"Agerbot: {stripped}\n"
    )


def _merge_training_corpus(base_text: str, new_text: str) -> tuple[str, int, int]:
    formatted = _ensure_dialogue_format(new_text)
    base_blocks = [_sanitize_training_block(block) for block in _training_blocks(base_text)]
    new_blocks = [_sanitize_training_block(block) for block in _training_blocks(formatted)]
    new_blocks = [block for block in new_blocks if block]
    if not base_blocks:
        raise RuntimeAPIError(
            "base_corpus_empty", "El corpus del modelo principal está vacío."
        )
    if not new_blocks:
        raise RuntimeAPIError(
            "data_too_short", "El texto nuevo no contiene bloques de entrenamiento válidos."
        )

    # Pocas repeticiones: mezcla con el corpus principal para generalizar,
    # no para que el modelo recote el pegado.
    repetitions = 2
    repeated_new_blocks: list[str] = []
    for _ in range(repetitions):
        shuffled = new_blocks.copy()
        random.shuffle(shuffled)
        repeated_new_blocks.extend(shuffled)

    merged_blocks = base_blocks + repeated_new_blocks
    random.shuffle(merged_blocks)
    merged_text = "\n\n".join(merged_blocks) + "\n"
    return merged_text, len(base_text), len(formatted)


def _build_incremental_model(
    base_checkpoint: dict[str, Any],
    device: torch.device,
    *,
    target_context_length: int = NEXT_CONTEXT_LENGTH,
) -> tuple[Agerbot, ModelConfig, int]:
    """Carga la base y amplía solo la tabla posicional cuando toca."""
    base_config = ModelConfig(**base_checkpoint["model_config"])
    context_length = max(base_config.context_length, target_context_length)
    model_config = ModelConfig(
        **{**base_config.to_dict(), "context_length": context_length}
    )
    model = Agerbot(model_config)
    base_state = base_checkpoint["model_state"]

    if context_length == base_config.context_length:
        model.load_state_dict(base_state)
        return model.to(device), model_config, base_config.context_length

    # La tabla nueva ya viene inicializada para las posiciones adicionales. Se
    # copian las posiciones aprendidas y se conservan intactos todos los demás
    # pesos; así aumentar contexto añade muy pocos parámetros.
    migrated_state = model.state_dict()
    for key, value in base_state.items():
        if key not in migrated_state:
            raise RuntimeAPIError(
                "checkpoint_incompatible",
                f"El checkpoint no contiene el parámetro esperado: {key}.",
            )
        target = migrated_state[key]
        if key == "position_embedding.weight":
            if value.ndim != 2 or target.ndim != 2 or value.shape[1] != target.shape[1]:
                raise RuntimeAPIError(
                    "checkpoint_incompatible",
                    "No se puede ampliar la tabla de posiciones del checkpoint.",
                )
            target[: value.shape[0]].copy_(value.to(dtype=target.dtype))
        elif target.shape != value.shape:
            raise RuntimeAPIError(
                "checkpoint_incompatible",
                f"La forma del parámetro no coincide: {key}.",
            )
        else:
            target.copy_(value.to(dtype=target.dtype))

    model.load_state_dict(migrated_state)
    return model.to(device), model_config, base_config.context_length


def _training_batch_plan(context_length: int) -> tuple[int, int]:
    """Mantiene aproximadamente el mismo uso de activaciones que 16x256."""
    if context_length <= REFERENCE_CONTEXT_LENGTH:
        return REFERENCE_BATCH_SIZE, 1
    batch_size = max(
        1,
        int(
            REFERENCE_BATCH_SIZE
            * (REFERENCE_CONTEXT_LENGTH / context_length) ** 2
        ),
    )
    accumulation = math.ceil(REFERENCE_BATCH_SIZE / batch_size)
    return batch_size, accumulation


class AgerbotRuntime:
    def __init__(self, checkpoint_path: str | Path, requested_device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.manifest = load_checkpoint_manifest(self.checkpoint_path)
        self.device = select_device(requested_device)
        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
            if "tokenizer" not in checkpoint:
                raise RuntimeAPIError(
                    "tokenizer_missing",
                    "El checkpoint no contiene la configuración de su tokenizador.",
                )
            tokenizer_config = checkpoint["tokenizer"]
            try:
                self.tokenizer = tokenizer_from_dict(tokenizer_config)
                self.tokenizer_name = tokenizer_identifier(tokenizer_config)
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeAPIError(
                    "tokenizer_unsupported",
                    "El checkpoint utiliza un tokenizador desconocido o inválido.",
                ) from error
            model_config = ModelConfig(**checkpoint["model_config"])
            if self.tokenizer.vocab_size != model_config.vocab_size:
                raise RuntimeAPIError(
                    "tokenizer_vocab_mismatch",
                    "El vocabulario del tokenizador no coincide con el modelo.",
                )
            if self.manifest.tokenizer and self.manifest.tokenizer != self.tokenizer_name:
                raise RuntimeAPIError(
                    "checkpoint_invalid",
                    "El tokenizador del checkpoint no coincide con su manifiesto.",
                )
            if self.manifest.context_length and self.manifest.context_length != model_config.context_length:
                raise RuntimeAPIError(
                    "checkpoint_invalid",
                    "El contexto del checkpoint no coincide con su manifiesto.",
                )
            self.model = Agerbot(model_config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            if any(not torch.isfinite(parameter).all().item() for parameter in self.model.parameters()):
                raise RuntimeAPIError(
                    "model_invalid_parameters",
                    "El checkpoint contiene parámetros NaN o infinitos.",
                )
            if self.manifest.parameters and self.manifest.parameters != self.model.parameter_count():
                raise RuntimeAPIError(
                    "checkpoint_invalid",
                    "El número de parámetros no coincide con su manifiesto.",
                )
        except RuntimeAPIError:
            raise
        except Exception as error:
            raise RuntimeAPIError(
                "model_unavailable",
                "Agerbot no pudo cargar el checkpoint verificado.",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                retryable=False,
            ) from error
        self._active: dict[str, threading.Event] = {}
        self._active_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "runtimeVersion": RUNTIME_VERSION,
            "model": {
                "name": self.manifest.model_name,
                "version": self.manifest.model_version,
                "trainingName": self.manifest.training_name,
                "loaded": True,
                "parameters": self.model.parameter_count(),
                "parameterCount": self.model.parameter_count(),
                "device": self.device.type,
                "tokenizer": self.tokenizer_name,
                "contextLength": self.model.config.context_length,
            },
        }

    def capabilities(self) -> dict[str, Any]:
        accelerators: list[dict[str, str]] = []
        if torch.backends.mps.is_available():
            accelerators.append(
                {"kind": "mps", "name": "Apple Metal Performance Shaders"}
            )
        if torch.cuda.is_available():
            accelerators.append(
                {"kind": "cuda", "name": torch.cuda.get_device_name(0)}
            )
        return {
            "platform": _platform_name(),
            "architecture": platform.machine(),
            "cpu": {"logicalCores": os.cpu_count() or 1},
            "memory": {"totalBytes": _total_memory_bytes()},
            "accelerators": accelerators,
            "inference": {
                "supported": True,
                "recommendedDevice": self.device.type,
            },
            "training": {
                "supported": True,
                "recommendedDevice": self.device.type,
            },
        }

    def cancel(self, conversation_id: str) -> bool:
        with self._active_lock:
            event = self._active.get(conversation_id)
            if event is None:
                return False
            event.set()
            return True

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        conversation_id, message, history, generation = self._validate_chat(payload)
        cancel_event = threading.Event()
        with self._active_lock:
            if conversation_id in self._active:
                raise RuntimeAPIError(
                    "conversation_busy",
                    "Agerbot ya está generando una respuesta para esta conversación.",
                    status=HTTPStatus.CONFLICT,
                    retryable=True,
                )
            self._active[conversation_id] = cancel_event

        started = time.monotonic()
        try:
            _, prompt_tokens = self._build_context_prompt(history, message)
            inputs = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            newline_ids = set(self.tokenizer.encode("\n"))
            output = self.model.generate(
                inputs,
                max_new_tokens=generation["maxNewTokens"],
                temperature=generation["temperature"],
                top_k=generation["topK"],
                should_stop=cancel_event.is_set,
                stop_token_ids=newline_ids or None,
            )
            if cancel_event.is_set():
                raise RuntimeAPIError(
                    "generation_cancelled",
                    "La generación de Agerbot fue cancelada.",
                    status=HTTPStatus.CONFLICT,
                    retryable=True,
                )
            generated = output[0].tolist()[len(prompt_tokens) :]
            raw_content = self.tokenizer.decode(generated)
            content = _sanitize_generated_content(trim_assistant_completion(raw_content))
            if not content:
                content = "No produje texto en esta ejecución. Inténtalo de nuevo."
            return {
                "conversationId": conversation_id,
                "message": {"role": "assistant", "content": content},
                "usage": {
                    "promptTokens": len(prompt_tokens),
                    "generatedTokens": len(generated),
                    "durationMs": int((time.monotonic() - started) * 1000),
                },
                "model": {
                    "name": self.manifest.model_name,
                    "version": self.manifest.model_version,
                    "device": self.device.type,
                },
            }
        finally:
            with self._active_lock:
                self._active.pop(conversation_id, None)

    def _validate_chat(
        self, payload: dict[str, Any]
    ) -> tuple[str, str, list[dict[str, str]], dict[str, Any]]:
        if not isinstance(payload, dict):
            raise RuntimeAPIError("bad_request", "La petición debe ser un objeto JSON.")
        conversation_id = payload.get("conversationId")
        message = payload.get("message")
        history = payload.get("history", [])
        raw_generation = payload.get("generation", {})
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise RuntimeAPIError("bad_request", "conversationId es obligatorio.")
        if len(conversation_id) > 128:
            raise RuntimeAPIError("bad_request", "conversationId es demasiado largo.")
        if not isinstance(message, str) or not message.strip():
            raise RuntimeAPIError("bad_request", "message es obligatorio.")
        if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise RuntimeAPIError("bad_request", "message supera el límite permitido.")
        if not isinstance(history, list) or len(history) > MAX_HISTORY_ITEMS:
            raise RuntimeAPIError("bad_request", "history supera el límite permitido.")
        clean_history: list[dict[str, str]] = []
        history_bytes = 0
        for item in history:
            if not isinstance(item, dict):
                raise RuntimeAPIError("bad_request", "history contiene una entrada inválida.")
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise RuntimeAPIError("bad_request", "history contiene un rol o texto inválido.")
            history_bytes += len(content.encode("utf-8"))
            clean_history.append({"role": role, "content": content})
        if history_bytes > MAX_HISTORY_BYTES:
            raise RuntimeAPIError("bad_request", "history supera el límite de bytes permitido.")
        if not isinstance(raw_generation, dict):
            raise RuntimeAPIError("bad_request", "generation debe ser un objeto.")
        max_new_tokens = raw_generation.get("maxNewTokens", 256)
        temperature = raw_generation.get("temperature", 0.85)
        top_k = raw_generation.get("topK", 45)
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 512:
            raise RuntimeAPIError("bad_request", "maxNewTokens debe estar entre 1 y 512.")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0.05 <= float(temperature) <= 2.0:
            raise RuntimeAPIError("bad_request", "temperature debe estar entre 0,05 y 2,0.")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 256):
            raise RuntimeAPIError("bad_request", "topK debe estar entre 1 y 256 o ser null.")
        return (
            conversation_id.strip(),
            message.strip(),
            clean_history,
            {
                "maxNewTokens": max_new_tokens,
                "temperature": float(temperature),
                "topK": top_k,
            },
        )

    @staticmethod
    def _build_prompt(history: list[dict[str, str]], message: str) -> str:
        lines: list[str] = []
        for item in history:
            speaker = "Usuario" if item["role"] == "user" else "Agerbot"
            lines.append(f"{speaker}: {normalize_chat_text(item['content']).strip()}")
        lines.append(f"Usuario: {normalize_chat_text(message).strip()}")
        lines.append("Agerbot:")
        return "\n".join(lines)

    def _build_context_prompt(
        self, history: list[dict[str, str]], message: str
    ) -> tuple[str, list[int]]:
        """Construye un prompt que conserva turnos completos dentro del contexto."""
        max_tokens = self.model.config.context_length
        compact_history = []
        if _needs_recent_history(history, message):
            compact_history = [
                {
                    "role": item["role"],
                    "content": item["content"][:MAX_CONTEXT_HISTORY_ITEM_CHARS],
                }
                for item in history
            ]

        safe_message = message
        current_prompt = self._build_prompt([], safe_message)
        current_tokens = self.tokenizer.encode(current_prompt)
        if len(current_tokens) > max_tokens:
            prefix = self.tokenizer.encode("Usuario: ")
            suffix = self.tokenizer.encode("\nAgerbot:")
            available = max_tokens - len(prefix) - len(suffix)
            if available > 0:
                message_tokens = self.tokenizer.encode(safe_message)[:available]
                safe_message = self.tokenizer.decode(message_tokens)
                current_prompt = self._build_prompt([], safe_message)
            else:
                current_tokens = (prefix + suffix)[-max_tokens:]
                return self.tokenizer.decode(current_tokens), current_tokens

        selected: list[dict[str, str]] = []
        for turn in reversed(_history_turns(compact_history)):
            candidate = turn + selected
            candidate_prompt = self._build_prompt(candidate, safe_message)
            if len(self.tokenizer.encode(candidate_prompt)) <= max_tokens:
                selected = candidate
            else:
                # Los turnos se recorren del más reciente al más antiguo. Si el
                # siguiente ya no cabe, los anteriores tampoco deben desplazar
                # el contexto reciente.
                break

        prompt = self._build_prompt(selected, safe_message)
        prompt_tokens = self.tokenizer.encode(prompt)
        return prompt, prompt_tokens


class TrainingManager:
    def __init__(self, server: AgerbotHTTPServer | None = None) -> None:
        self.server = server
        self.lock = threading.Lock()
        self.status = "idle"  # idle, training, completed, failed
        self.session_name = ""
        self.step = 0
        self.max_steps = 0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.elapsed_seconds = 0.0
        self.max_duration_seconds = 0.0
        self.error_message = ""
        self.checkpoint_path = ""
        self.base_checkpoint_path = ""
        self.base_training_name = ""
        self.base_corpus_characters = 0
        self.new_corpus_characters = 0
        self.merged_corpus_characters = 0
        self.logs: list[str] = []
        self._thread: threading.Thread | None = None

    def status_payload(self) -> dict[str, Any]:
        with self.lock:
            percent = 0
            if self.status == "training" and self.max_duration_seconds > 0:
                percent = min(99, int((self.elapsed_seconds / self.max_duration_seconds) * 100))
            elif self.status == "completed":
                percent = 100
            return {
                "status": self.status,
                "sessionName": self.session_name,
                "step": self.step,
                "maxSteps": self.max_steps,
                "trainLoss": round(self.train_loss, 4),
                "valLoss": round(self.val_loss, 4),
                "elapsedSeconds": round(self.elapsed_seconds, 1),
                "maxDurationSeconds": round(self.max_duration_seconds, 1),
                "percent": percent,
                "checkpoint": self.checkpoint_path,
                "baseCheckpoint": self.base_checkpoint_path,
                "baseTrainingName": self.base_training_name,
                "baseCorpusCharacters": self.base_corpus_characters,
                "newCorpusCharacters": self.new_corpus_characters,
                "mergedCorpusCharacters": self.merged_corpus_characters,
                "errorMessage": self.error_message,
                "logs": self.logs[-25:],
            }

    def start_training(self, text: str, duration_minutes: int, name: str = "") -> dict[str, Any]:
        with self.lock:
            if self.status == "training":
                raise RuntimeAPIError("already_training", "Ya hay un entrenamiento en curso.", status=HTTPStatus.CONFLICT)
            
            clean_text = _ensure_dialogue_format(text)
            if len(clean_text.strip()) < 20:
                raise RuntimeAPIError("data_too_short", "El texto es demasiado corto. Escribe al menos un par de diálogos.")

            if self.server is None:
                raise RuntimeAPIError(
                    "base_checkpoint_unavailable",
                    "No se puede entrenar sin un modelo principal activo.",
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )

            # Capturamos la base antes de iniciar el hilo. El modelo nuevo no se
            # convierte en base hasta que termina y el hot-reload ha sido validado.
            base_checkpoint_path = self.server.primary_checkpoint_path
            if not base_checkpoint_path:
                raise RuntimeAPIError(
                    "base_checkpoint_unavailable",
                    "No se encontró el checkpoint del modelo principal.",
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            
            # Siempre es el mismo Agerbot. El nombre opcional solo se anota en el log.
            note = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip()) if name.strip() else ""
            clean_name = "agerbot"
            self.session_name = clean_name
            self.status = "training"
            self.step = 0
            self.train_loss = 0.0
            self.val_loss = 0.0
            self.elapsed_seconds = 0.0
            self.max_duration_seconds = max(30, duration_minutes * 60)
            self.error_message = ""
            self.checkpoint_path = ""
            self.base_checkpoint_path = str(base_checkpoint_path)
            self.base_training_name = "cargando..."
            self.base_corpus_characters = 0
            self.new_corpus_characters = len(clean_text)
            self.merged_corpus_characters = 0
            note_bit = f" ({note})" if note else ""
            self.logs = [f"🚀 Entrenando el mismo Agerbot{note_bit} ({duration_minutes} min)..."]
            
            self._thread = threading.Thread(
                target=self._run_training,
                args=(clean_text, clean_name, self.max_duration_seconds, base_checkpoint_path),
                daemon=True,
            )
            self._thread.start()
            return {"status": "started", "sessionName": clean_name}

    def _run_training(
        self,
        text: str,
        session_name: str,
        max_duration: float,
        base_checkpoint_path: Path,
    ) -> None:
        try:
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            data_file = processed_dir / "agerbot.txt"

            base_manifest = load_checkpoint_manifest(base_checkpoint_path)
            base_checkpoint = torch.load(
                base_checkpoint_path, map_location="cpu", weights_only=True
            )
            base_training_name = base_manifest.training_name
            base_corpus_path = _resolve_training_corpus(
                base_checkpoint, base_checkpoint_path, base_training_name
            )
            base_text = base_corpus_path.read_text(encoding="utf-8")
            merged_text, base_characters, new_characters = _merge_training_corpus(
                base_text, text
            )
            data_file.write_text(merged_text, encoding="utf-8")

            with self.lock:
                self.base_training_name = base_training_name
                self.base_corpus_characters = base_characters
                self.new_corpus_characters = new_characters
                self.merged_corpus_characters = len(merged_text)
                self.logs.append(
                    f"Base principal: {base_training_name} | Corpus anterior: "
                    f"{base_characters:,} caracteres | Datos nuevos: {new_characters:,}"
                )
                self.logs.append(
                    f"Mezcla creada: {len(merged_text):,} caracteres | "
                    "Se conservan los pesos y el tokenizador; la ventana se ajusta si procede."
                )
            
            checkpoint_dir = Path(f"checkpoints/{session_name}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            device = select_device("auto")
            corpus_text = data_file.read_text(encoding="utf-8")
            tokenizer_config = base_checkpoint.get("tokenizer")
            if tokenizer_config is None:
                raise RuntimeAPIError(
                    "tokenizer_missing",
                    "El modelo principal no contiene la configuración de su tokenizador.",
                )
            tokenizer = tokenizer_from_dict(tokenizer_config)
            all_tokens = load_corpus(data_file, tokenizer)
            train_tokens, val_tokens = split_corpus(all_tokens, 0.9)
            base_model_config = ModelConfig(**base_checkpoint["model_config"])
            if tokenizer.vocab_size != base_model_config.vocab_size:
                raise RuntimeAPIError(
                    "tokenizer_vocab_mismatch",
                    "El tokenizador del modelo principal no coincide con su vocabulario.",
                )
            model, model_config, base_context_length = _build_incremental_model(
                base_checkpoint,
                device,
                target_context_length=base_model_config.context_length,
            )
            batch_size, gradient_accumulation = _training_batch_plan(
                model_config.context_length
            )
            base_training_config = base_checkpoint.get("training_config")
            base_learning_rate = (
                float(base_training_config.get("learning_rate", 0.00035))
                if isinstance(base_training_config, dict)
                else 0.00035
            )
            # Los ajustes encadenados habían reducido la tasa hasta volver
            # simbólico el aprendizaje. Conservamos un ajuste pequeño, pero
            # suficiente para que los nuevos turnos de conversación se aprendan.
            learning_rate = min(max(base_learning_rate, 0.00008), 0.00015)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=0.05
            )
            
            started = time.perf_counter()
            deadline = started + max_duration
            best_val_loss = float("inf")
            best_checkpoint_path = checkpoint_dir / "best.pt"
            
            with self.lock:
                self.logs.append(
                    f"Hardware: {device} | Parámetros conservados: "
                    f"{model.parameter_count():,} | Tokens mezclados: {len(all_tokens):,}"
                )
                if model_config.context_length > base_context_length:
                    self.logs.append(
                        f"Contexto ampliado: {base_context_length} → "
                        f"{model_config.context_length} | Lote: {batch_size} | "
                        f"Acumulación: {gradient_accumulation} (memoria controlada)"
                    )
                self.logs.append(
                    f"Ajuste incremental: learning rate {learning_rate:.6f} | "
                    f"Tokenizador {tokenizer_identifier(tokenizer.to_dict())}"
                )
            
            step = 0
            model.train()
            while time.perf_counter() < deadline and step < 100000:
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                for _ in range(gradient_accumulation):
                    inputs, targets = random_batch(
                        train_tokens, batch_size, model_config.context_length, device
                    )
                    _, loss = model(inputs, targets)
                    assert loss is not None
                    (loss / gradient_accumulation).backward()
                    accumulated_loss += loss.item() / gradient_accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                step += 1
                now = time.perf_counter()
                elapsed = now - started
                
                if step % 25 == 0 or step == 1:
                    with self.lock:
                        self.step = step
                        self.train_loss = accumulated_loss
                        self.elapsed_seconds = elapsed

                if step % 150 == 0 or step == 1:
                    losses = estimate_loss(
                        model, train_tokens, val_tokens, batch_size, 10, device
                    )
                    with self.lock:
                        self.val_loss = losses["val"]
                        self.train_loss = losses["train"]
                        self.logs.append(f"Paso {step:04d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | Tiempo: {elapsed:.0f}s")
                    
                    if losses["val"] < best_val_loss:
                        best_val_loss = losses["val"]
                        payload = {
                            "format_version": 1,
                            "step": step,
                            "model_config": model_config.to_dict(),
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "tokenizer": tokenizer.to_dict(),
                            "training_config": {
                                "mode": "incremental",
                                "base_checkpoint": str(base_checkpoint_path),
                                "base_training_name": base_training_name,
                                "base_corpus": str(base_corpus_path),
                                "new_data_characters": new_characters,
                                "merged_corpus_characters": len(merged_text),
                                "learning_rate": learning_rate,
                            },
                            "best_val_loss": best_val_loss,
                        }
                        save_checkpoint(best_checkpoint_path, payload)
            
            if not best_checkpoint_path.exists():
                payload = {
                    "format_version": 1,
                    "step": step,
                    "model_config": model_config.to_dict(),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "training_config": {
                        "mode": "incremental",
                        "base_checkpoint": str(base_checkpoint_path),
                        "base_training_name": base_training_name,
                        "base_corpus": str(base_corpus_path),
                        "new_data_characters": new_characters,
                        "merged_corpus_characters": len(merged_text),
                        "learning_rate": learning_rate,
                    },
                    "best_val_loss": best_val_loss if best_val_loss != float("inf") else 0.0,
                }
                save_checkpoint(best_checkpoint_path, payload)
            
            manifest = {
                "schemaVersion": 2,
                "channel": "stable",
                "model": {
                    "name": "Agerbot",
                    "version": "custom",
                    "trainingName": "agerbot",
                    "architecture": "agerbot-transformer",
                    "tokenizer": tokenizer_identifier(tokenizer.to_dict()),
                    "parameters": model.parameter_count(),
                    "contextLength": model_config.context_length,
                },
                "runtime": {"minimumVersion": "0.2.0", "maximumVersion": None},
                "checkpoint": {
                    "filename": "best.pt",
                    "sizeBytes": best_checkpoint_path.stat().st_size,
                    "sha256": _sha256(best_checkpoint_path),
                },
                "training": {
                    "durationSeconds": int(time.perf_counter() - started),
                    "steps": step,
                    "bestValidationLoss": best_val_loss,
                    "mode": "incremental",
                    "baseTrainingName": base_training_name,
                    "baseCheckpoint": str(base_checkpoint_path),
                    "baseCorpusCharacters": base_characters,
                    "newCorpusCharacters": new_characters,
                    "mergedCorpusCharacters": len(merged_text),
                },
                "publishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            
            if self.server:
                self.server.reload_runtime(best_checkpoint_path, promote_to_primary=True)
            
            with self.lock:
                self.status = "completed"
                self.checkpoint_path = str(best_checkpoint_path)
                self.elapsed_seconds = max_duration
                self.logs.append(f"🎉 Agerbot actualizado ({model.parameter_count():,} parámetros, {best_checkpoint_path.stat().st_size / (1024*1024):.1f} MB). Pregúntale en el chat con otras palabras.")
        except Exception as e:
            with self.lock:
                self.status = "failed"
                self.error_message = str(e)
                self.logs.append(f"❌ Error en entrenamiento: {e}")


class AgerbotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], runtime: AgerbotRuntime) -> None:
        super().__init__(address, AgerbotRequestHandler)
        self.runtime = runtime
        # Esta es la línea principal de aprendizaje. Se mantiene separada del
        # runtime publicado para que un entrenamiento fallido nunca cambie la
        # base de la siguiente sesión.
        self.primary_checkpoint_path = runtime.checkpoint_path
        self.requested_device = runtime.device.type
        self.training_manager = TrainingManager(self)

    def reload_runtime(
        self, checkpoint_path: str | Path, *, promote_to_primary: bool = False
    ) -> None:
        new_runtime = AgerbotRuntime(checkpoint_path, self.requested_device)
        self.runtime = new_runtime
        if promote_to_primary:
            self.primary_checkpoint_path = new_runtime.checkpoint_path
            _persist_active_checkpoint(new_runtime.checkpoint_path)


class AgerbotRequestHandler(BaseHTTPRequestHandler):
    server: AgerbotHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT.value)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            if self.path in {"/", "/index.html", "/ui"}:
                self._write_html(HTTPStatus.OK, WEB_UI_HTML)
            elif self.path == "/v1/health":
                self._write_json(HTTPStatus.OK, self.server.runtime.health())
            elif self.path == "/v1/capabilities":
                self._write_json(HTTPStatus.OK, self.server.runtime.capabilities())
            elif self.path == "/v1/train/status":
                self._write_json(HTTPStatus.OK, self.server.training_manager.status_payload())
            else:
                raise RuntimeAPIError(
                    "not_found", "La ruta solicitada no existe.", status=HTTPStatus.NOT_FOUND
                )
        except RuntimeAPIError as error:
            self._write_json(error.status, error.payload())

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/v1/chat":
                self._write_json(HTTPStatus.OK, self.server.runtime.chat(payload))
            elif self.path == "/v1/train/start":
                text = payload.get("data", "")
                duration = int(payload.get("durationMinutes", 15))
                name = str(payload.get("name", ""))
                res = self.server.training_manager.start_training(text, duration, name)
                self._write_json(HTTPStatus.OK, res)
            elif self.path == "/v1/model/reload":
                ckpt = payload.get("checkpoint")
                if not isinstance(ckpt, str) or not ckpt:
                    raise RuntimeAPIError("bad_request", "checkpoint es obligatorio.")
                self.server.reload_runtime(ckpt, promote_to_primary=True)
                self._write_json(HTTPStatus.OK, {"status": "reloaded", "checkpoint": ckpt})
            elif self.path == "/v1/chat/cancel":
                conversation_id = payload.get("conversationId")
                if not isinstance(conversation_id, str) or not conversation_id:
                    raise RuntimeAPIError("bad_request", "conversationId es obligatorio.")
                requested = self.server.runtime.cancel(conversation_id)
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "conversationId": conversation_id,
                        "cancelRequested": requested,
                    },
                )
            else:
                raise RuntimeAPIError(
                    "not_found", "La ruta solicitada no existe.", status=HTTPStatus.NOT_FOUND
                )
        except RuntimeAPIError as error:
            self._write_json(error.status, error.payload())
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                RuntimeAPIError(
                    "internal_error",
                    f"Agerbot encontró un error interno: {e}",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    retryable=True,
                ).payload(),
            )

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as error:
            raise RuntimeAPIError("bad_request", "Content-Length no es válido.") from error
        if length <= 0:
            raise RuntimeAPIError("bad_request", "La petición no contiene JSON.")
        if length > MAX_BODY_BYTES:
            raise RuntimeAPIError(
                "body_too_large", "La petición supera el límite de tamaño.", status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            )
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeAPIError("bad_request", "El cuerpo JSON no es válido.") from error
        if not isinstance(payload, dict):
            raise RuntimeAPIError("bad_request", "La petición debe ser un objeto JSON.")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _write_html(self, status: HTTPStatus, html_content: str) -> None:
        body = html_content.encode("utf-8")
        try:
            self.send_response(status.value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=_configured_checkpoint_path(),
    )
    parser.add_argument("--host", default=os.environ.get("AGERBOT_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGERBOT_PORT", DEFAULT_PORT)))
    parser.add_argument("--device", default=os.environ.get("AGERBOT_DEVICE", "auto"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.host != DEFAULT_HOST:
        print("bad_host: Agerbot solo puede escuchar en 127.0.0.1.", file=sys.stderr)
        raise SystemExit(2)
    try:
        runtime = AgerbotRuntime(args.checkpoint, args.device)
        server = AgerbotHTTPServer((args.host, args.port), runtime)
    except RuntimeAPIError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(2) from error
    except OSError as error:
        print(f"port_unavailable: No se pudo usar 127.0.0.1:{args.port} ({error.strerror}).", file=sys.stderr)
        raise SystemExit(2) from error

    def stop_server(signum: int, frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
