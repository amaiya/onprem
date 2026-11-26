"""
Tests for how AgentExecutor copies custom tool code into the tools directory.

These tests do not need an LLM, network access, or patchpal.
"""

import importlib.util
import textwrap

from onprem.pipelines.agent.base import AgentExecutor


TOOL_MODULE = '''
import json
from os.path import (
    basename,
    join,
)
from os.path import \\
    dirname

from onprem.utils import batch_list

PREFIX = "result"
SEPARATOR = "-"
LABEL = PREFIX + SEPARATOR
UNUSED = "not used by the tool"


def my_tool(path: str) -> str:
    """Return a JSON label for a path."""
    return json.dumps({LABEL: join(dirname(path), basename(path))})
'''


def _write_tool(tmp_path):
    """Load `my_tool` from a real .py file and write it out with AgentExecutor."""
    module_file = tmp_path / "my_tools.py"
    module_file.write_text(TOOL_MODULE)
    spec = importlib.util.spec_from_file_location("my_tools", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    executor = AgentExecutor(model="openai/gpt-5-mini", custom_tools=[module.my_tool], verbose=False)
    tools_dir = tmp_path / "tools"
    executor._write_custom_tools(tools_dir)
    return (tools_dir / "my_tool.py").read_text()


def _run_tool(tool_source, *args):
    namespace = {}
    exec(compile(tool_source, "my_tool.py", "exec"), namespace)
    return namespace["my_tool"](*args)


def test_multiline_imports_are_copied_whole(tmp_path):
    tool_source = _write_tool(tmp_path)
    # Must be valid Python (a cut-off "from x import (" line is a SyntaxError).
    compile(tool_source, "my_tool.py", "exec")
    assert "from os.path import basename, join" in tool_source
    assert "from os.path import dirname" in tool_source
    assert "onprem" not in tool_source


def test_module_constants_used_by_tool_are_copied(tmp_path):
    tool_source = _write_tool(tmp_path)
    assert 'PREFIX = "result"' in tool_source
    assert 'LABEL = PREFIX + SEPARATOR' in tool_source
    assert "UNUSED" not in tool_source
    assert _run_tool(tool_source, "/a/b.txt") == '{"result-": "/a/b.txt"}'
