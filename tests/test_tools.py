"""Tests for agenticflow.tools."""

from pathlib import Path

from agenticflow.tools.base import ToolResult
from agenticflow.tools.code_exec import RunCommandTool
from agenticflow.tools.file_io import ListDirectoryTool, ReadFileTool, WriteFileTool


class TestToolResult:
    def test_to_api_content(self):
        r = ToolResult(output="hello")
        content = r.to_api_content()
        assert content["type"] == "text"
        assert content["text"] == "hello"


class TestReadFileTool:
    def test_read_existing(self, tmp_path: Path):
        (tmp_path / "hello.txt").write_text("world")
        tool = ReadFileTool(tmp_path)
        result = tool.execute(path="hello.txt")
        assert not result.is_error
        assert result.output == "world"

    def test_read_missing(self, tmp_path: Path):
        tool = ReadFileTool(tmp_path)
        result = tool.execute(path="missing.txt")
        assert result.is_error
        assert "not found" in result.output.lower()

    def test_traversal_blocked(self, tmp_path: Path):
        tool = ReadFileTool(tmp_path)
        result = tool.execute(path="../../etc/passwd")
        assert result.is_error

    def test_sibling_prefix_traversal_blocked(self, tmp_path: Path):
        tool = ReadFileTool(tmp_path / "ws")
        result = tool.execute(path="../ws2/secret.txt")
        assert result.is_error

    def test_to_anthropic_dict(self, tmp_path: Path):
        tool = ReadFileTool(tmp_path)
        d = tool.to_anthropic_dict()
        assert d["name"] == "read_file"
        assert "input_schema" in d
        assert d["input_schema"]["required"] == ["path"]


class TestWriteFileTool:
    def test_write_new_file(self, tmp_path: Path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute(path="out.txt", content="hello")
        assert not result.is_error
        assert (tmp_path / "out.txt").read_text() == "hello"

    def test_write_creates_dirs(self, tmp_path: Path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute(path="sub/dir/out.txt", content="nested")
        assert not result.is_error
        assert (tmp_path / "sub" / "dir" / "out.txt").read_text() == "nested"

    def test_traversal_blocked(self, tmp_path: Path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute(path="../../evil.txt", content="bad")
        assert result.is_error

    def test_sibling_prefix_traversal_blocked(self, tmp_path: Path):
        tool = WriteFileTool(tmp_path / "ws")
        result = tool.execute(path="../ws2/evil.txt", content="bad")
        assert result.is_error


class TestListDirectoryTool:
    def test_list_root(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b").mkdir()
        tool = ListDirectoryTool(tmp_path)
        result = tool.execute(path=".")
        assert not result.is_error
        assert "a.txt" in result.output
        assert "b/" in result.output

    def test_list_empty(self, tmp_path: Path):
        tool = ListDirectoryTool(tmp_path)
        result = tool.execute(path=".")
        assert "empty" in result.output.lower()

    def test_list_missing(self, tmp_path: Path):
        tool = ListDirectoryTool(tmp_path)
        result = tool.execute(path="nope")
        assert result.is_error

    def test_list_sibling_prefix_traversal_blocked(self, tmp_path: Path):
        tool = ListDirectoryTool(tmp_path / "ws")
        result = tool.execute(path="../ws2")
        assert result.is_error


class TestRunCommandTool:
    def test_echo(self, tmp_path: Path):
        tool = RunCommandTool(tmp_path)
        result = tool.execute(command="echo hello")
        assert not result.is_error
        assert "hello" in result.output

    def test_failing_command(self, tmp_path: Path):
        tool = RunCommandTool(tmp_path)
        result = tool.execute(command="false")
        assert result.is_error

    def test_timeout(self, tmp_path: Path):
        tool = RunCommandTool(tmp_path, timeout=1)
        result = tool.execute(command="sleep 10")
        assert result.is_error
        assert "timed out" in result.output.lower()
