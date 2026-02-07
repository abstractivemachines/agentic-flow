"""Tests for the shell tool."""

from pathlib import Path

from agenticflow.tools.shell import ShellTool


class TestShellTool:
    def test_schema(self, tmp_path: Path):
        tool = ShellTool(tmp_path)
        d = tool.to_anthropic_dict()
        assert d["name"] == "shell"
        assert "command" in d["input_schema"]["properties"]
        assert "timeout" in d["input_schema"]["properties"]
        assert "env" in d["input_schema"]["properties"]

    def test_echo(self, tmp_path: Path):
        tool = ShellTool(tmp_path)
        result = tool.execute(command="echo hello world")
        assert not result.is_error
        assert "hello world" in result.output

    def test_env_vars(self, tmp_path: Path):
        tool = ShellTool(tmp_path)
        result = tool.execute(command="echo $MY_VAR", env={"MY_VAR": "test_value"})
        assert not result.is_error
        assert "test_value" in result.output

    def test_timeout(self, tmp_path: Path):
        tool = ShellTool(tmp_path, timeout=60)
        result = tool.execute(command="sleep 10", timeout=1)
        assert result.is_error
        assert "timed out" in result.output.lower()

    def test_failing_command(self, tmp_path: Path):
        tool = ShellTool(tmp_path)
        result = tool.execute(command="false")
        assert result.is_error

    def test_cwd_is_workspace(self, tmp_path: Path):
        tool = ShellTool(tmp_path)
        result = tool.execute(command="pwd")
        assert not result.is_error
        assert str(tmp_path.resolve()) in result.output

    def test_output_truncation(self, tmp_path: Path):
        tool = ShellTool(tmp_path)
        # Generate very long output
        result = tool.execute(command="python3 -c \"print('x' * 100000)\"")
        assert not result.is_error
        assert len(result.output) <= 50_100  # MAX_OUTPUT_CHARS + some overhead
