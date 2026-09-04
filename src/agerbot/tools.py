"""Herramientas agenticas de Agerbot: lectura segura + comandos en allowlist."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_MAX_FILE_BYTES = 32 * 1024
DEFAULT_MAX_LIST_ENTRIES = 200
DEFAULT_MAX_RESULT_CHARS = 4_000
DEFAULT_CMD_TIMEOUT_SECONDS = 5

# Lecturas / inspección segura: se ejecutan sin confirmación.
SAFE_TOOLS = frozenset({"list_dir", "read_file", "run_cmd"})

# Comandos permitidos en run_cmd (solo el binario base; sin pipes/redir).
ALLOWED_COMMANDS = frozenset({"pwd", "ls", "date", "whoami", "uname"})

# Herramientas que mutan el sistema (v1: ninguna expuesta; ruta de confirm lista).
MUTATING_TOOLS = frozenset({"write_file", "run_shell", "delete_path"})

ACCION_RE = re.compile(
    r"(?m)^\s*Acci[oó]n:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})?\s*$"
)


class ToolError(Exception):
    """Error controlado al ejecutar o parsear una herramienta."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    raw_line: str = ""


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str
    needs_confirm: bool = False
    code: str = "ok"


@dataclass
class ToolRuntime:
    """Ejecuta herramientas acotadas al workspace del proyecto."""

    workspace: Path
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_list_entries: int = DEFAULT_MAX_LIST_ENTRIES
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    cmd_timeout_seconds: int = DEFAULT_CMD_TIMEOUT_SECONDS
    allow_mutating: bool = False

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ToolError("workspace_missing", f"Workspace inexistente: {self.workspace}")

    def resolve_path(self, raw: str | None) -> Path:
        """Resuelve una ruta relativa al workspace y la mantiene dentro."""
        if raw is None or str(raw).strip() == "":
            raw = "."
        candidate = Path(str(raw))
        if candidate.is_absolute():
            target = candidate.resolve()
        else:
            target = (self.workspace / candidate).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as error:
            raise ToolError(
                "path_outside_workspace",
                f"Ruta fuera del workspace: {raw}",
            ) from error
        return target

    def list_dir(self, path: str = ".", **_: Any) -> str:
        target = self.resolve_path(path)
        if not target.exists():
            raise ToolError("not_found", f"No existe: {path}")
        if not target.is_dir():
            raise ToolError("not_a_directory", f"No es un directorio: {path}")
        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if len(entries) >= self.max_list_entries:
                entries.append("… (lista truncada)")
                break
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.name}{suffix}")
        rel = target.relative_to(self.workspace)
        header = f"{rel}/" if str(rel) != "." else "./"
        body = "\n".join(entries) if entries else "(vacío)"
        return f"{header}\n{body}"

    def read_file(self, path: str, **_: Any) -> str:
        target = self.resolve_path(path)
        if not target.exists():
            raise ToolError("not_found", f"No existe: {path}")
        if not target.is_file():
            raise ToolError("not_a_file", f"No es un archivo: {path}")
        size = target.stat().st_size
        if size > self.max_file_bytes:
            raise ToolError(
                "file_too_large",
                f"Archivo demasiado grande ({size} B > {self.max_file_bytes} B).",
            )
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolError("not_text", f"No es texto UTF-8: {path}") from error
        return text

    def run_cmd(self, cmd: str, confirm: bool = False, **_: Any) -> str:
        if not isinstance(cmd, str) or not cmd.strip():
            raise ToolError("bad_args", "run_cmd requiere cmd no vacío.")
        try:
            parts = shlex.split(cmd.strip())
        except ValueError as error:
            raise ToolError("bad_args", f"cmd inválido: {error}") from error
        if not parts:
            raise ToolError("bad_args", "cmd vacío.")
        binary = Path(parts[0]).name
        if binary not in ALLOWED_COMMANDS:
            raise ToolError(
                "command_not_allowed",
                f"Comando no permitido: {binary}. Allowlist: {', '.join(sorted(ALLOWED_COMMANDS))}.",
            )
        # Bloquear metacaracteres peligrosos ya rotos por shlex, pero refuerzo.
        forbidden = {"|", ";", "&&", "||", ">", "<", "`", "$(", "\n"}
        if any(token in forbidden for token in parts):
            raise ToolError("command_not_allowed", "Metacaracteres de shell no permitidos.")
        # Solo cwd = workspace; sin shell.
        try:
            completed = subprocess.run(
                parts,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.cmd_timeout_seconds,
                check=False,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": os.environ.get("HOME", ""),
                    "LANG": os.environ.get("LANG", "C.UTF-8"),
                },
            )
        except subprocess.TimeoutExpired as error:
            raise ToolError("timeout", f"Comando agotó el tiempo: {cmd}") from error
        stdout = (completed.stdout or "").rstrip()
        stderr = (completed.stderr or "").rstrip()
        pieces = []
        if stdout:
            pieces.append(stdout)
        if stderr:
            pieces.append(f"[stderr]\n{stderr}")
        if completed.returncode != 0:
            pieces.append(f"[exit {completed.returncode}]")
        return "\n".join(pieces) if pieces else "(sin salida)"

    def execute(self, call: ToolCall) -> ToolResult:
        name = call.name
        args = dict(call.args)
        confirm = bool(args.pop("confirm", False))

        if name in MUTATING_TOOLS and not (confirm or self.allow_mutating):
            return ToolResult(
                name=name,
                ok=False,
                output=(
                    f"La acción '{name}' muta el sistema. "
                    "Reenvía con \"confirm\": true para autorizarla."
                ),
                needs_confirm=True,
                code="needs_confirm",
            )

        handlers: dict[str, Callable[..., str]] = {
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "run_cmd": self.run_cmd,
        }
        handler = handlers.get(name)
        if handler is None:
            return ToolResult(
                name=name,
                ok=False,
                output=f"Herramienta desconocida: {name}",
                code="unknown_tool",
            )

        if name not in SAFE_TOOLS and not confirm:
            return ToolResult(
                name=name,
                ok=False,
                output=f"'{name}' requiere confirmación explícita.",
                needs_confirm=True,
                code="needs_confirm",
            )

        try:
            if name == "run_cmd":
                output = handler(**args, confirm=confirm)
            else:
                output = handler(**args)
            return ToolResult(
                name=name,
                ok=True,
                output=self._truncate(output),
                code="ok",
            )
        except ToolError as error:
            return ToolResult(
                name=name,
                ok=False,
                output=error.message,
                code=error.code,
            )
        except TypeError as error:
            return ToolResult(
                name=name,
                ok=False,
                output=f"Argumentos inválidos: {error}",
                code="bad_args",
            )
        except OSError as error:
            return ToolResult(
                name=name,
                ok=False,
                output=f"Error del sistema: {error}",
                code="os_error",
            )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_result_chars:
            return text
        return text[: self.max_result_chars] + "\n… (resultado truncado)"


def parse_accion_line(line: str) -> ToolCall | None:
    """Parsea una línea 'Acción: tool {json}'."""
    match = ACCION_RE.match(line.strip())
    if not match:
        return None
    name = match.group(1)
    raw_args = (match.group(2) or "").strip()
    args: dict[str, Any] = {}
    if raw_args:
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as error:
            raise ToolError("bad_json", f"JSON inválido en Acción: {error}") from error
        if not isinstance(parsed, dict):
            raise ToolError("bad_json", "Los argumentos de Acción deben ser un objeto JSON.")
        args = parsed
    return ToolCall(name=name, args=args, raw_line=line.strip())


def extract_acciones(text: str) -> list[ToolCall]:
    """Extrae todas las líneas Acción válidas de un texto (modelo o usuario)."""
    calls: list[ToolCall] = []
    for line in text.splitlines():
        try:
            call = parse_accion_line(line)
        except ToolError:
            continue
        if call is not None:
            calls.append(call)
    return calls


def format_resultado(result: ToolResult) -> str:
    status = "ok" if result.ok else result.code
    return f"Resultado: [{result.name}/{status}]\n{result.output}"


def run_acciones(
    text: str,
    runtime: ToolRuntime,
    *,
    max_calls: int = 3,
) -> tuple[list[ToolResult], str]:
    """Ejecuta hasta max_calls acciones halladas en text; devuelve resultados y bloque Resultado."""
    calls = extract_acciones(text)[:max_calls]
    results: list[ToolResult] = []
    blocks: list[str] = []
    for call in calls:
        result = runtime.execute(call)
        results.append(result)
        blocks.append(format_resultado(result))
    return results, "\n\n".join(blocks)


@dataclass
class AgenticTrace:
    loops: list[dict[str, Any]] = field(default_factory=list)

    def add(self, source: str, calls: list[ToolCall], results: list[ToolResult]) -> None:
        self.loops.append(
            {
                "source": source,
                "actions": [
                    {"name": c.name, "args": c.args, "raw": c.raw_line} for c in calls
                ],
                "results": [
                    {
                        "name": r.name,
                        "ok": r.ok,
                        "code": r.code,
                        "needsConfirm": r.needs_confirm,
                        "output": r.output,
                    }
                    for r in results
                ],
            }
        )

    def flat_results_text(self) -> str:
        parts: list[str] = []
        for loop in self.loops:
            for result in loop["results"]:
                status = "ok" if result["ok"] else result["code"]
                parts.append(f"[{result['name']}/{status}] {result['output']}")
        return "\n\n".join(parts)
