"""Agent executor that wraps patchpal-sandbox for sandboxed autonomous execution."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

__all__ = ['AgentExecutor']


class AgentExecutor:
    """
    Sandboxed agent executor using patchpal-sandbox.

    This executor is a Python wrapper around patchpal-sandbox, providing:
    - Resource isolation and security via Docker/Podman containers
    - Support for both cloud and local LLMs
    - Custom tool integration via ~/.patchpal/tools/
    - API key management via .env files
    
    Args:
        model (str): LiteLLM model identifier (e.g., 'anthropic/claude-sonnet-4-5', 'ollama_chat/gpt-oss-120b')
        agent_type (str): Type of agent ('function_calling' or 'react')
        max_iterations (int): Maximum number of autopilot iterations
        env_file (str): Path to .env file with API keys
        sandbox (bool): Run in container sandbox (default: False). Set True for isolated/secure execution.
        image (str): Container image to use (default: python:3.11-slim) [sandbox only]
        network (str): Network mode ('bridge', 'host', 'none') [sandbox only]
        memory (str): Memory limit (e.g., '2g', '4g') [sandbox only]
        cpus (float): CPU limit (e.g., 2, 4) [sandbox only]
        enabled_tools (list): List of tool names to enable. If None, uses DEFAULT_TOOLS:
                             ['read_file', 'read_lines', 'edit_file', 'write_file', 
                              'grep', 'find', 'run_shell', 'web_search', 'web_fetch']
                             Pass an empty list [] to use all available patchpal tools.
        completion_promise (str): String that signals task completion (default: 'COMPLETE')
        verbose (bool): Print container output in real-time
    """
    
    # Default tools for code-focused agent tasks
    DEFAULT_TOOLS = [
        'read_file',      # Read complete files
        'read_lines',     # Read specific line ranges
        'edit_file',      # Edit files via find/replace
        'write_file',     # Write complete files
        'grep',           # Search for patterns in files
        'find',           # Find files by glob pattern
        'run_shell',      # Execute shell commands
        'web_search',     # Search the web for information
        'web_fetch',      # Fetch content from URLs
    ]

    def __init__(
        self,
        model: str,
        agent_type: str = "function_calling",
        max_iterations: int = 50,
        env_file: Optional[str] = None,
        sandbox: bool = False,
        image: str = "python:3.11-slim",
        network: str = "bridge",
        memory: Optional[str] = None,
        cpus: Optional[float] = None,
        enabled_tools: Optional[List[str]] = None,
        completion_promise: str = "COMPLETE",
        verbose: bool = False,
    ):
        # Normalize ollama/ to ollama_chat/ for LiteLLM compatibility
        if model.startswith("ollama/"):
            model = model.replace("ollama/", "ollama_chat/", 1)

        self.model = model
        self.agent_type = agent_type
        self.max_iterations = max_iterations
        self.env_file = env_file
        self.sandbox = sandbox
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        # Use default tools if none specified
        self.enabled_tools = enabled_tools if enabled_tools is not None else self.DEFAULT_TOOLS.copy()
        self.completion_promise = completion_promise
        self.verbose = verbose

    @classmethod
    def print_default_tools(cls):
        """
        Pretty-print the default tools available in AgentExecutor.
        
        This shows the tools that are used when enabled_tools=None (the default).
        Users can customize by passing a different list to enabled_tools parameter.
        """
        print("=" * 70)
        print("AgentExecutor Default Tools")
        print("=" * 70)
        print("\nThese tools are used by default when enabled_tools=None:\n")
        
        tool_descriptions = {
            'read_file': 'Read complete file contents',
            'read_lines': 'Read specific line ranges from files',
            'edit_file': 'Edit files via find/replace',
            'write_file': 'Write complete file contents',
            'grep': 'Search for patterns in files',
            'find': 'Find files by glob pattern',
            'run_shell': 'Execute shell commands',
            'web_search': 'Search the web for information',
            'web_fetch': 'Fetch and read content from URLs',
        }
        
        for i, tool in enumerate(cls.DEFAULT_TOOLS, 1):
            desc = tool_descriptions.get(tool, 'No description available')
            print(f"  {i:2d}. {tool:15s} - {desc}")
        
        print("\n" + "=" * 70)
        print("Customization Examples:")
        print("=" * 70)
        print("\n# Use defaults (recommended):")
        print("executor = AgentExecutor(model='openai/gpt-4o-mini')")
        print("\n# No Shell Access or Web Access:")
        print("executor = AgentExecutor(")
        print("    model='openai/gpt-5-mini',")
        print("    enabled_tools=['read_file', 'read_lines', 'write_file', 'edit_file', 'grep', 'find']")
        print(")")
        print("\n# Web Research:")
        print("executor = AgentExecutor(")
        print("    model='openai/gpt-5-mini',")
        print("    enabled_tools=['web_search', 'web_fetch'] # only web tools")
        print(")")
        print()

    def _build_sandbox_command(self, task_file: Path) -> List[str]:
        """Build the patchpal-sandbox command."""
        cmd = ['patchpal-sandbox']
        
        # Add sandbox options
        if self.image != "python:3.11-slim":
            cmd.extend(['--image', self.image])
        
        if self.network == 'host':
            cmd.append('--host-network')
        elif self.network == 'none':
            cmd.append('--no-network')
        # 'bridge' is default, no flag needed
        
        if self.memory:
            cmd.extend(['--memory', self.memory])
        
        if self.cpus:
            cmd.extend(['--cpus', str(self.cpus)])
        
        if self.env_file:
            cmd.extend(['--env-file', self.env_file])
        
        # Separator between sandbox args and patchpal args
        cmd.append('--')
        
        # Add patchpal autopilot command
        cmd.append('autopilot')
        cmd.extend(['--model', self.model])
        cmd.extend(['--prompt-file', str(task_file)])
        cmd.extend(['--completion-promise', self.completion_promise])
        cmd.extend(['--max-iterations', str(self.max_iterations)])
        
        return cmd
    
    def _build_direct_command(self, task_file: Path) -> List[str]:
        """Build the direct patchpal-autopilot command (no sandbox)."""
        cmd = ['patchpal-autopilot']
        cmd.extend(['--model', self.model])
        cmd.extend(['--prompt-file', str(task_file)])
        cmd.extend(['--completion-promise', self.completion_promise])
        cmd.extend(['--max-iterations', str(self.max_iterations)])

        return cmd
    
    def run(
        self,
        task: str,
        working_dir: Optional[str] = None,
    ) -> str:
        """
        Run the agent on a given task in a sandboxed environment.
        
        Args:
            task (str): The task description/prompt
            working_dir (str): Directory to run in (default: current directory)
            
        Returns:
            str: The agent's response/output
            
        Raises:
            RuntimeError: If agent execution fails
            FileNotFoundError: If patchpal-sandbox is not installed
        """
        # Save current directory
        original_dir = os.getcwd()
        
        try:
            # Change to working directory if specified
            if working_dir:
                working_path = Path(working_dir)
                working_path.mkdir(parents=True, exist_ok=True)
                os.chdir(working_dir)
            
            # Create task file in current directory
            task_file = Path('task_prompt.md')
            
            # Add completion promise instruction
            task_with_completion = (
                f"{task}\n\n"
                f"When you have successfully completed the task, output the following exactly:\n"
                f"<promise>{self.completion_promise}</promise>"
            )
            task_file.write_text(task_with_completion)
            
            try:
                # Set environment variables for agent configuration
                env = os.environ.copy()

                # Skip autopilot confirmation prompt (we're running non-interactively)
                env['PATCHPAL_AUTOPILOT_CONFIRMED'] = 'true'

                # Disable streaming output unless verbose mode is enabled
                if not self.verbose:
                    env['PATCHPAL_STREAM_OUTPUT'] = 'false'

                # Set agent type
                if self.agent_type == "react":
                    env['PATCHPAL_REACT_MODE'] = 'true'

                # Set enabled tools if specified
                if self.enabled_tools:
                    env['PATCHPAL_ENABLED_TOOLS'] = ','.join(self.enabled_tools)

                # Load .env file if specified (for non-sandboxed execution)
                if self.env_file and not self.sandbox:
                    # Load environment variables from .env file
                    env_path = Path(self.env_file)
                    if env_path.exists():
                        with open(env_path) as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#') and '=' in line:
                                    key, value = line.split('=', 1)
                                    env[key.strip()] = value.strip()

                # Build command (sandbox or direct)
                if self.sandbox:
                    cmd = self._build_sandbox_command(task_file)
                else:
                    cmd = self._build_direct_command(task_file)

                # Run patchpal-sandbox
                if self.verbose:
                    # Real-time output
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    
                    output_lines = []
                    for line in process.stdout:
                        print(line, end='')
                        output_lines.append(line)
                    
                    process.wait()
                    
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"Agent execution failed with return code {process.returncode}"
                        )
                    
                    return ''.join(output_lines)
                else:
                    # Capture all output
                    result = subprocess.run(
                        cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=3600,  # 1 hour timeout
                    )
                    
                    if result.returncode != 0:
                        # Check if it's actually an error or just warnings
                        # Podman/pip warnings go to stderr but aren't fatal
                        output = result.stdout + result.stderr
                        
                        # Look for actual error indicators
                        is_error = any(phrase in output.lower() for phrase in [
                            'error:', 'exception:', 'traceback', 'command not found',
                            'no such file', 'permission denied', 'connection refused'
                        ])
                        
                        if is_error or not result.stdout.strip():
                            # Real error or no output at all
                            error_msg = result.stderr or result.stdout
                            raise RuntimeError(f"Agent execution failed: {error_msg}")
                        # else: warnings only, treat as success
                    
                    return result.stdout
                    
            except FileNotFoundError:
                if self.sandbox:
                    raise FileNotFoundError(
                        "patchpal-sandbox command not found. Install PatchPal with: pip install patchpal"
                    )
                else:
                    raise FileNotFoundError(
                        "patchpal command not found. Install PatchPal with: pip install patchpal"
                    )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Agent execution timed out after 1 hour")
            finally:
                # Clean up task file
                if task_file.exists():
                    task_file.unlink()
        
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    # Convenience methods for common scenarios
    def run_with_minimal_tools(self, task: str, working_dir: Optional[str] = None) -> str:
        """
        Run task with only essential file/shell tools (no web tools).
        
        Useful for faster execution with smaller models or offline environments.
        """
        original_tools = self.enabled_tools
        self.enabled_tools = ['read_file', 'read_lines', 'write_file', 'edit_file', 'run_shell']
        try:
            return self.run(task, working_dir=working_dir)
        finally:
            self.enabled_tools = original_tools
    
    def run_with_websearch(self, task: str, working_dir: Optional[str] = None) -> str:
        """
        Run task with ONLY web tools (web_search, web_fetch).
        
        Useful for research/information gathering tasks without file system access.
        """
        original_tools = self.enabled_tools
        self.enabled_tools = ['web_search', 'web_fetch']
        try:
            return self.run(task, working_dir=working_dir)
        finally:
            self.enabled_tools = original_tools
    
    @classmethod
    def for_ollama(
        cls,
        model: str,
        agent_type: str = "react",
        **kwargs
    ):
        r"""
        Create an executor configured for Ollama models.

        Uses host network mode by default. Works when Ollama runs on the host:
        - Linux/WSL2 with mirrored networking: localhost:11434 works via --host-network ✅
        - macOS/Windows Docker Desktop: Set OLLAMA_API_BASE=http://host.docker.internal:11434
        - Podman: Set OLLAMA_API_BASE=http://host.containers.internal:11434

        For WSL2 mirrored networking, add to C:\Users\<username>\.wslconfig:
          [wsl2]
          networkingMode=mirrored

        The OLLAMA_API_BASE environment variable is automatically passed to the
        container by patchpal-sandbox (all OLLAMA_* variables are forwarded).

        Args:
            model (str): Ollama model name (e.g., 'ollama_chat/llama3.1')
            agent_type (str): 'react' for models without function calling,
                            'function_calling' for models with native function calling
            **kwargs: Additional AgentExecutor arguments

        Returns:
            AgentExecutor configured for Ollama

        Example:
            # Linux/WSL2 with mirrored networking (uses --host-network)
            executor = AgentExecutor.for_ollama('ollama_chat/llama3.1')
            result = executor.run("Analyze the codebase")

            # macOS/Windows Docker Desktop
            import os
            os.environ['OLLAMA_API_BASE'] = 'http://host.docker.internal:11434'
            executor = AgentExecutor.for_ollama('ollama_chat/llama3.1')

            # Podman
            import os
            os.environ['OLLAMA_API_BASE'] = 'http://host.containers.internal:11434'
            executor = AgentExecutor.for_ollama('ollama_chat/llama3.1')

            # Remote Ollama on another machine
            import os
            os.environ['OLLAMA_API_BASE'] = 'http://192.168.1.100:11434'
            executor = AgentExecutor.for_ollama('ollama_chat/llama3.1')
        """
        return cls(
            model=model,
            agent_type=agent_type,
            network='host',
            **kwargs
        )
    
    @classmethod
    def for_openai(
        cls,
        model: str = "openai/gpt-5.2-codex",
        env_file: Optional[str] = None,
        sandbox: bool = False,
        **kwargs
    ):
        """
        Create an executor configured for OpenAI models.
        
        Args:
            model (str): OpenAI model name
            env_file (str): Path to .env with OPENAI_API_KEY
            sandbox (bool): Run in container sandbox (default: False)
            **kwargs: Additional AgentExecutor arguments
        
        Returns:
            AgentExecutor configured for OpenAI
        """
        return cls(
            model=model,
            env_file=env_file,
            sandbox=sandbox,
            agent_type='function_calling',
            **kwargs
        )
    
    @classmethod
    def for_anthropic(
        cls,
        model: str = "anthropic/claude-sonnet-4-5",
        env_file: Optional[str] = None,
        sandbox: bool = False,
        **kwargs
    ):
        """
        Create an executor configured for Anthropic models.
        
        Args:
            model (str): Anthropic model name
            env_file (str): Path to .env with ANTHROPIC_API_KEY
            sandbox (bool): Run in container sandbox (default: False)
            **kwargs: Additional AgentExecutor arguments
        
        Returns:
            AgentExecutor configured for Anthropic
        """
        return cls(
            model=model,
            env_file=env_file,
            sandbox=sandbox,
            agent_type='function_calling',
            **kwargs
        )
