import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import torch

from agerbot import ByteTokenizer
from agerbot.model import Agerbot, ModelConfig
from agerbot.server import (
    AgerbotRuntime,
    RuntimeAPIError,
    _build_incremental_model,
    _needs_recent_history,
    _sanitize_generated_content,
    _training_batch_plan,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


class RuntimeServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.checkpoint = cls.directory / "test.pt"
        config = ModelConfig(
            context_length=16, d_model=16, n_heads=4, n_layers=1, dropout=0.0
        )
        model = Agerbot(config)
        torch.save(
            {
                "format_version": 1,
                "tokenizer": "byte-v1",
                "model_config": config.to_dict(),
                "model_state": model.state_dict(),
            },
            cls.checkpoint,
        )
        digest = hashlib.sha256(cls.checkpoint.read_bytes()).hexdigest()
        (cls.directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "model": {"name": "Agerbot", "version": "test"},
                    "checkpoint": {
                        "filename": cls.checkpoint.name,
                        "sizeBytes": cls.checkpoint.stat().st_size,
                        "sha256": digest,
                    },
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.port = free_port()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agerbot.server",
                "--checkpoint",
                str(self.checkpoint),
                "--port",
                str(self.port),
                "--device",
                "cpu",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.fail(f"El servidor terminó al iniciar: {self.process.stderr.read()}")
            try:
                request_json(f"http://127.0.0.1:{self.port}/v1/health")
                return
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        self.fail("El servidor no quedó listo")

    def tearDown(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.process.stderr is not None:
            self.process.stderr.close()

    def test_health_capabilities_and_chat(self) -> None:
        health = request_json(f"http://127.0.0.1:{self.port}/v1/health")
        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["model"]["name"], "Agerbot")
        self.assertTrue(health["model"]["loaded"])

        capabilities = request_json(
            f"http://127.0.0.1:{self.port}/v1/capabilities"
        )
        self.assertTrue(capabilities["inference"]["supported"])
        self.assertGreaterEqual(capabilities["cpu"]["logicalCores"], 1)

        reply = request_json(
            f"http://127.0.0.1:{self.port}/v1/chat",
            {
                "conversationId": "agerbot-local",
                "message": "Hola",
                "history": [],
                "generation": {"maxNewTokens": 2, "temperature": 0.8, "topK": 20},
            },
        )
        self.assertEqual(reply["conversationId"], "agerbot-local")
        self.assertEqual(reply["message"]["role"], "assistant")
        self.assertTrue(reply["message"]["content"])
        self.assertEqual(reply["usage"]["generatedTokens"], 2)

    def test_invalid_request_has_structured_error(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as context:
            request_json(f"http://127.0.0.1:{self.port}/v1/chat", {})
        payload = json.load(context.exception)
        self.assertEqual(payload["error"]["code"], "bad_request")
        self.assertFalse(payload["error"]["retryable"])

    def test_context_keeps_recent_turns_intact(self) -> None:
        runtime = object.__new__(AgerbotRuntime)
        runtime.tokenizer = ByteTokenizer()
        runtime.model = type("ModelStub", (), {"config": ModelConfig(context_length=160)})()
        history = [
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "respuesta antigua " + "x" * 150},
            {"role": "user", "content": "cuéntame un chiste"},
            {"role": "assistant", "content": "¿Por qué el libro sonrió? Porque tenía una buena historia."},
        ]

        prompt, tokens = runtime._build_context_prompt(history, "cuéntame otro")

        self.assertLessEqual(len(tokens), 160)
        self.assertTrue(prompt.startswith("Usuario: cuéntame un chiste"))
        self.assertTrue(prompt.endswith("Usuario: cuéntame otro\nAgerbot:"))
        self.assertIn("¿Por qué el libro sonrió?", prompt)
        self.assertNotIn("respuesta antigua", prompt)

    def test_new_question_does_not_inherit_unrelated_history(self) -> None:
        runtime = object.__new__(AgerbotRuntime)
        runtime.tokenizer = ByteTokenizer()
        runtime.model = type("ModelStub", (), {"config": ModelConfig(context_length=128)})()
        history = [
            {"role": "user", "content": "cuéntame un chiste"},
            {"role": "assistant", "content": "¿Por qué el ordenador fue al médico?"},
        ]

        prompt, tokens = runtime._build_context_prompt(history, "¿sabes algo de código?")

        self.assertEqual(prompt, "Usuario: ¿sabes algo de código?\nAgerbot:")
        self.assertLessEqual(len(tokens), 128)

    def test_feedback_about_joke_keeps_recent_context(self) -> None:
        history = [
            {"role": "user", "content": "cuéntame un chiste"},
            {"role": "assistant", "content": "¿Por qué cruzó la gallina?"},
        ]
        self.assertTrue(_needs_recent_history(history, "no me da risa"))
        self.assertTrue(_needs_recent_history(history, "tampoco"))

    def test_generated_identity_is_sanitized(self) -> None:
        content = _sanitize_generated_content(
            "¡Hola! Soy Agerbot 0.3.0. ¿En qué te ayudo?"
        )
        self.assertEqual(
            content, "¡Hola! Soy una inteligencia artificial. ¿En qué te ayudo?"
        )

    def test_incremental_context_expansion_preserves_old_positions(self) -> None:
        base_config = ModelConfig(
            context_length=4, d_model=8, n_heads=2, n_layers=1, dropout=0.0
        )
        base_model = Agerbot(base_config)
        base_positions = base_model.position_embedding.weight.detach().clone()
        model, model_config, old_context = _build_incremental_model(
            {"model_config": base_config.to_dict(), "model_state": base_model.state_dict()},
            torch.device("cpu"),
            target_context_length=8,
        )

        self.assertEqual((old_context, model_config.context_length), (4, 8))
        self.assertTrue(
            torch.equal(model.position_embedding.weight[:4], base_positions)
        )
        self.assertEqual(_training_batch_plan(256), (16, 1))
        self.assertEqual(_training_batch_plan(512), (4, 4))
        self.assertEqual(_training_batch_plan(1024), (1, 16))

    def test_second_runtime_reports_port_conflict(self) -> None:
        duplicate = subprocess.run(
            [
                sys.executable,
                "-m",
                "agerbot.server",
                "--checkpoint",
                str(self.checkpoint),
                "--port",
                str(self.port),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("port_unavailable", duplicate.stderr)


class RuntimeFailureTests(unittest.TestCase):
    def test_missing_checkpoint_is_clear(self) -> None:
        with self.assertRaises(RuntimeAPIError) as context:
            AgerbotRuntime("/definitely/missing/agerbot.pt", "cpu")
        self.assertEqual(context.exception.code, "checkpoint_missing")

    def test_manifest_hash_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            path.write_bytes(b"not a checkpoint")
            (path.parent / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": {"name": "Agerbot", "version": "bad"},
                        "checkpoint": {
                            "filename": "model.pt",
                            "sizeBytes": len(path.read_bytes()),
                            "sha256": "0" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeAPIError) as context:
                AgerbotRuntime(path, "cpu")
            self.assertEqual(context.exception.code, "checkpoint_invalid")

    def test_cancel_stops_active_generation(self) -> None:
        class SlowModel:
            config = ModelConfig(context_length=16)

            def generate(self, tokens, max_new_tokens, temperature, top_k, should_stop, stop_token_ids=None):
                for _ in range(100):
                    if should_stop():
                        return tokens
                    time.sleep(0.01)
                return tokens

            def parameter_count(self):
                return 0

        runtime = object.__new__(AgerbotRuntime)
        runtime.model = SlowModel()
        runtime.tokenizer = __import__("agerbot").ByteTokenizer()
        runtime.device = torch.device("cpu")
        runtime.agentic_default = False
        runtime.tool_runtime = None
        runtime.manifest = type("M", (), {"model_name": "Agerbot", "model_version": "test"})()
        runtime._active = {}
        runtime._active_lock = threading.Lock()
        result: list[str] = []

        def generate() -> None:
            try:
                runtime.chat({"conversationId": "cancel-me", "message": "genera una respuesta"})
            except RuntimeAPIError as error:
                result.append(error.code)

        worker = threading.Thread(target=generate)
        worker.start()
        deadline = time.monotonic() + 2
        while "cancel-me" not in runtime._active and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(runtime.cancel("cancel-me"))
        worker.join(timeout=2)
        self.assertEqual(result, ["generation_cancelled"])


if __name__ == "__main__":
    unittest.main()
