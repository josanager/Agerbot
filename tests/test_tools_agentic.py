"""Pruebas del sandbox agentico, mutaciones, auto mode y allowlist."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agerbot.tools import (
    AUTO_MUTATING_TOOLS,
    MUTATING_TOOLS,
    ToolCall,
    ToolError,
    ToolRuntime,
    extract_acciones,
    format_resultado,
    parse_accion_line,
    run_acciones,
)


class AccionParsingTests(unittest.TestCase):
    def test_parse_list_dir(self) -> None:
        call = parse_accion_line('Acción: list_dir {"path":"."}')
        assert call is not None
        self.assertEqual(call.name, "list_dir")
        self.assertEqual(call.args, {"path": "."})

    def test_parse_without_args(self) -> None:
        call = parse_accion_line("Acción: list_dir")
        assert call is not None
        self.assertEqual(call.name, "list_dir")
        self.assertEqual(call.args, {})

    def test_parse_accent_variant(self) -> None:
        call = parse_accion_line('Accion: run_cmd {"cmd":"pwd"}')
        assert call is not None
        self.assertEqual(call.name, "run_cmd")

    def test_parse_move_file(self) -> None:
        call = parse_accion_line('Acción: move_file {"src":"a.txt","dst":"b.txt"}')
        assert call is not None
        self.assertEqual(call.name, "move_file")
        self.assertEqual(call.args["src"], "a.txt")

    def test_extract_multiple(self) -> None:
        text = (
            "Hola\n"
            'Acción: list_dir {"path":"."}\n'
            'Acción: read_file {"path":"README.md"}\n'
        )
        calls = extract_acciones(text)
        self.assertEqual([c.name for c in calls], ["list_dir", "read_file"])

    def test_bad_json_raises(self) -> None:
        with self.assertRaises(ToolError):
            parse_accion_line("Acción: list_dir {no-json}")


class ToolSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "README.md").write_text("# Hola\nLínea 2\n", encoding="utf-8")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "note.txt").write_text("nota", encoding="utf-8")
        self.rt = ToolRuntime(workspace=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_list_dir(self) -> None:
        out = self.rt.list_dir(".")
        self.assertIn("README.md", out)
        self.assertIn("sub/", out)

    def test_read_file(self) -> None:
        out = self.rt.read_file("README.md")
        self.assertIn("# Hola", out)

    def test_path_escape_rejected(self) -> None:
        with self.assertRaises(ToolError) as ctx:
            self.rt.read_file("../outside.txt")
        self.assertEqual(ctx.exception.code, "path_outside_workspace")

    def test_absolute_outside_rejected(self) -> None:
        with self.assertRaises(ToolError) as ctx:
            self.rt.list_dir("/etc")
        self.assertEqual(ctx.exception.code, "path_outside_workspace")

    def test_file_too_large(self) -> None:
        big = self.root / "big.txt"
        big.write_bytes(b"x" * 100)
        tiny = ToolRuntime(workspace=self.root, max_file_bytes=10)
        with self.assertRaises(ToolError) as ctx:
            tiny.read_file("big.txt")
        self.assertEqual(ctx.exception.code, "file_too_large")

    def test_run_cmd_pwd(self) -> None:
        result = self.rt.execute(ToolCall("run_cmd", {"cmd": "pwd"}))
        self.assertTrue(result.ok)
        self.assertIn(str(self.root), result.output)

    def test_run_cmd_date_whoami_uname_ls(self) -> None:
        for cmd in ("date", "whoami", "uname", "ls"):
            result = self.rt.execute(ToolCall("run_cmd", {"cmd": cmd}))
            self.assertTrue(result.ok, msg=f"{cmd}: {result.output}")

    def test_allowlist_rejects_rm(self) -> None:
        result = self.rt.execute(ToolCall("run_cmd", {"cmd": "rm -rf /"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "command_not_allowed")

    def test_allowlist_rejects_curl(self) -> None:
        result = self.rt.execute(ToolCall("run_cmd", {"cmd": "curl https://example.com"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "command_not_allowed")

    def test_allowlist_rejects_sudo(self) -> None:
        result = self.rt.execute(ToolCall("run_cmd", {"cmd": "sudo ls"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "command_not_allowed")

    def test_mutating_needs_confirm_by_default(self) -> None:
        self.assertIn("write_file", MUTATING_TOOLS)
        self.assertIn("write_file", AUTO_MUTATING_TOOLS)
        result = self.rt.execute(ToolCall("write_file", {"path": "x", "content": "y"}))
        self.assertTrue(result.needs_confirm)
        self.assertEqual(result.code, "needs_confirm")
        self.assertFalse((self.root / "x").exists())

    def test_run_acciones_formats_resultado(self) -> None:
        text = 'Acción: list_dir {"path":"."}'
        results, block = run_acciones(text, self.rt)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok)
        self.assertTrue(block.startswith("Resultado:"))
        self.assertIn("README.md", block)

    def test_format_resultado(self) -> None:
        result = self.rt.execute(ToolCall("list_dir", {"path": "sub"}))
        rendered = format_resultado(result)
        self.assertIn("[list_dir/ok]", rendered)


class MutationSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.txt").write_text("alpha", encoding="utf-8")
        (self.root / "README.md").write_text("# hi\n", encoding="utf-8")
        self.rt = ToolRuntime(workspace=self.root)
        self.auto = ToolRuntime(workspace=self.root, auto=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_move_needs_confirm_without_auto(self) -> None:
        result = self.rt.execute(ToolCall("move_file", {"src": "a.txt", "dst": "b.txt"}))
        self.assertTrue(result.needs_confirm)
        self.assertTrue((self.root / "a.txt").exists())
        self.assertFalse((self.root / "b.txt").exists())

    def test_move_with_confirm(self) -> None:
        result = self.rt.execute(
            ToolCall("move_file", {"src": "a.txt", "dst": "b.txt", "confirm": True})
        )
        self.assertTrue(result.ok, result.output)
        self.assertFalse((self.root / "a.txt").exists())
        self.assertEqual((self.root / "b.txt").read_text(encoding="utf-8"), "alpha")

    def test_move_auto_mode(self) -> None:
        result = self.auto.execute(ToolCall("move_file", {"src": "a.txt", "dst": "b.txt"}))
        self.assertTrue(result.ok, result.output)
        self.assertEqual((self.root / "b.txt").read_text(encoding="utf-8"), "alpha")

    def test_move_path_escape_dst_rejected(self) -> None:
        result = self.auto.execute(
            ToolCall("move_file", {"src": "a.txt", "dst": "../outside.txt"})
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "path_outside_workspace")
        self.assertTrue((self.root / "a.txt").exists())

    def test_copy_needs_confirm_without_auto(self) -> None:
        result = self.rt.execute(ToolCall("copy_file", {"src": "a.txt", "dst": "c.txt"}))
        self.assertTrue(result.needs_confirm)
        self.assertFalse((self.root / "c.txt").exists())

    def test_copy_auto_mode(self) -> None:
        result = self.auto.execute(ToolCall("copy_file", {"src": "a.txt", "dst": "c.txt"}))
        self.assertTrue(result.ok, result.output)
        self.assertTrue((self.root / "a.txt").exists())
        self.assertEqual((self.root / "c.txt").read_text(encoding="utf-8"), "alpha")

    def test_copy_escape_rejected(self) -> None:
        result = self.auto.execute(
            ToolCall("copy_file", {"src": "a.txt", "dst": "/tmp/escaped.txt"})
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "path_outside_workspace")

    def test_write_needs_confirm_without_auto(self) -> None:
        result = self.rt.execute(
            ToolCall("write_file", {"path": "new.txt", "content": "hola"})
        )
        self.assertTrue(result.needs_confirm)
        self.assertFalse((self.root / "new.txt").exists())

    def test_write_with_confirm(self) -> None:
        result = self.rt.execute(
            ToolCall(
                "write_file",
                {"path": "new.txt", "content": "hola", "confirm": True},
            )
        )
        self.assertTrue(result.ok, result.output)
        self.assertEqual((self.root / "new.txt").read_text(encoding="utf-8"), "hola")

    def test_write_auto_mode(self) -> None:
        result = self.auto.execute(
            ToolCall("write_file", {"path": "auto.txt", "content": "auto"})
        )
        self.assertTrue(result.ok, result.output)
        self.assertEqual((self.root / "auto.txt").read_text(encoding="utf-8"), "auto")

    def test_write_size_cap(self) -> None:
        tiny = ToolRuntime(workspace=self.root, auto=True, max_write_bytes=4)
        result = tiny.execute(
            ToolCall("write_file", {"path": "big.txt", "content": "12345"})
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "file_too_large")

    def test_mkdir_needs_confirm_without_auto(self) -> None:
        result = self.rt.execute(ToolCall("mkdir", {"path": "docs"}))
        self.assertTrue(result.needs_confirm)
        self.assertFalse((self.root / "docs").exists())

    def test_mkdir_auto_mode(self) -> None:
        result = self.auto.execute(ToolCall("mkdir", {"path": "docs"}))
        self.assertTrue(result.ok, result.output)
        self.assertTrue((self.root / "docs").is_dir())

    def test_mkdir_nested_auto(self) -> None:
        result = self.auto.execute(ToolCall("mkdir", {"path": "tmp/logs"}))
        self.assertTrue(result.ok, result.output)
        self.assertTrue((self.root / "tmp" / "logs").is_dir())

    def test_delete_always_needs_confirm_even_in_auto(self) -> None:
        result = self.auto.execute(ToolCall("delete_file", {"path": "a.txt"}))
        self.assertTrue(result.needs_confirm)
        self.assertTrue((self.root / "a.txt").exists())

    def test_delete_with_confirm(self) -> None:
        result = self.auto.execute(
            ToolCall("delete_file", {"path": "a.txt", "confirm": True})
        )
        self.assertTrue(result.ok, result.output)
        self.assertFalse((self.root / "a.txt").exists())

    def test_delete_rejects_directory(self) -> None:
        (self.root / "folder").mkdir()
        result = self.rt.execute(
            ToolCall("delete_file", {"path": "folder", "confirm": True})
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "not_a_file")
        self.assertTrue((self.root / "folder").is_dir())

    def test_move_into_dir_auto(self) -> None:
        (self.root / "docs").mkdir()
        result = self.auto.execute(
            ToolCall("move_file", {"src": "README.md", "dst": "docs"})
        )
        self.assertTrue(result.ok, result.output)
        self.assertFalse((self.root / "README.md").exists())
        self.assertTrue((self.root / "docs" / "README.md").is_file())

    def test_run_acciones_move_with_confirm_flag_in_json(self) -> None:
        text = 'Acción: mkdir {"path":"out","confirm":true}'
        results, block = run_acciones(text, self.rt)
        self.assertTrue(results[0].ok)
        self.assertIn("out", block)
        self.assertTrue((self.root / "out").is_dir())


if __name__ == "__main__":
    unittest.main()
