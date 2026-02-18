"""Tests for the git tool."""

from pathlib import Path

from agenticflow.tools.git import GitTool


class TestGitTool:
    def test_schema(self, tmp_path: Path):
        tool = GitTool(tmp_path)
        d = tool.to_anthropic_dict()
        assert d["name"] == "git"
        assert "args" in d["input_schema"]["properties"]

    def test_init_and_status(self, tmp_path: Path):
        tool = GitTool(tmp_path)
        # Init a repo first
        result = tool.execute(args="init")
        assert not result.is_error
        assert "Initialized" in result.output or "Reinitialized" in result.output

        # Status should work now
        result = tool.execute(args="status")
        assert not result.is_error

    def test_blocked_subcommand(self, tmp_path: Path):
        tool = GitTool(tmp_path)
        result = tool.execute(args="push origin main")
        assert result.is_error
        assert "not allowed" in result.output

    def test_injection_like_args_are_blocked(self, tmp_path: Path):
        tool = GitTool(tmp_path)
        result = tool.execute(args="status; echo pwned")
        assert result.is_error
        assert "not allowed" in result.output

    def test_empty_args(self, tmp_path: Path):
        tool = GitTool(tmp_path)
        result = tool.execute(args="")
        assert result.is_error
        assert "No git arguments" in result.output

    def test_add_and_commit(self, tmp_path: Path):
        tool = GitTool(tmp_path)
        tool.execute(args="init")

        # Create a file
        (tmp_path / "test.txt").write_text("hello")

        result = tool.execute(args="add test.txt")
        assert not result.is_error

        result = tool.execute(args='commit -m "initial"')
        assert not result.is_error

    def test_log(self, tmp_path: Path):
        tool = GitTool(tmp_path)
        tool.execute(args="init")
        (tmp_path / "test.txt").write_text("hello")
        tool.execute(args="add test.txt")
        tool.execute(args='commit -m "initial"')

        result = tool.execute(args="log --oneline")
        assert not result.is_error
        assert "initial" in result.output

    def test_diff(self, tmp_path: Path):
        tool = GitTool(tmp_path)
        tool.execute(args="init")
        (tmp_path / "test.txt").write_text("hello")
        tool.execute(args="add test.txt")
        tool.execute(args='commit -m "initial"')

        (tmp_path / "test.txt").write_text("world")
        result = tool.execute(args="diff")
        assert not result.is_error
        assert "world" in result.output or "hello" in result.output
