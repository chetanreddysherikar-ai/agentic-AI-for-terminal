"""Code execution tools: run Python, optionally run shell commands."""
from __future__ import annotations

import subprocess
import sys

from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.security import SecurityError, SecurityPolicy


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated {len(text) - limit} chars]"


def execution_tools(settings: Settings, policy: SecurityPolicy) -> list[BaseTool]:
    """Build code-execution tools bound to the workspace and policy."""

    @tool
    def run_python(code: str, timeout: int = 60) -> str:
        """Execute a Python snippet in a subprocess and return stdout/stderr.

        Use this for calculations, data wrangling, testing code, and inspecting
        the environment. The snippet runs in the workspace directory with the
        project virtualenv interpreter. API keys and other credentials are
        stripped from the subprocess environment, so the code cannot read them.

        Args:
            code: Python source to execute.
            timeout: Maximum seconds to wait before killing the process.
        """
        timeout = min(max(int(timeout), 1), 300)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(settings.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                env=policy.sanitized_env(),
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: execution timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001 - surface the real reason to the model
            return f"ERROR: {type(exc).__name__}: {exc}"

        parts = []
        if proc.stdout.strip():
            parts.append(proc.stdout.rstrip())
        if proc.stderr.strip():
            parts.append(f"[stderr]\n{proc.stderr.rstrip()}")
        if proc.returncode != 0:
            parts.append(f"[exit code {proc.returncode}]")
        output = "\n".join(parts) or "(no output)"
        policy.audit("run_python", bytes=len(code), timeout=timeout, exit=proc.returncode)
        return policy.redact(_truncate(output, settings.max_output_chars))

    tools: list[BaseTool] = [run_python]

    if settings.allow_shell:

        @tool
        def run_shell(command: str, timeout: int = 60) -> str:
            """Run a shell command in the workspace and return its output.

            Destructive commands (rm -rf, mkfs, piping downloads into a shell,
            and similar) are rejected by the security policy.

            Args:
                command: Command line to execute.
                timeout: Maximum seconds to wait before killing the process.
            """
            try:
                policy.check_shell(command)
            except SecurityError as exc:
                policy.audit("run_shell_blocked", command=command)
                return f"ERROR: {exc}"

            timeout = min(max(int(timeout), 1), 300)
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(settings.workspace),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                    env=policy.sanitized_env(),
                )
            except subprocess.TimeoutExpired:
                return f"ERROR: command timed out after {timeout}s"
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {type(exc).__name__}: {exc}"

            parts = []
            if proc.stdout.strip():
                parts.append(proc.stdout.rstrip())
            if proc.stderr.strip():
                parts.append(f"[stderr]\n{proc.stderr.rstrip()}")
            if proc.returncode != 0:
                parts.append(f"[exit code {proc.returncode}]")
            output = "\n".join(parts) or "(no output)"
            policy.audit("run_shell", command=command, exit=proc.returncode)
            return policy.redact(_truncate(output, settings.max_output_chars))

        tools.append(run_shell)

    return tools
