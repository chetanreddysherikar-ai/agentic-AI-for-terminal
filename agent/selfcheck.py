"""Offline self-check for the tool layer and the security policy. No model calls."""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from langchain_core.tools import BaseTool

from agent.config import Settings
from agent.security import SecurityPolicy
from agent.tools import all_tools


def _call(tool: BaseTool, **kwargs) -> str:
    result = tool.invoke(kwargs)
    return result if isinstance(result, str) else str(result)


def run_selfcheck(settings: Settings) -> int:
    """Exercise every tool plus the security rules. Returns the failure count."""
    policy = SecurityPolicy(
        workspace=settings.workspace,
        allow_network=settings.allow_network,
        allow_destructive=settings.allow_destructive,
        allow_shell=settings.allow_shell,
        redact_secrets=settings.redact_secrets,
        audit_log=None,
    )
    tools = {t.name: t for t in all_tools(settings, policy)}
    failures: list[str] = []
    passed = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal passed
        if condition:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failures.append(label)
            print(f"  FAIL  {label} {detail}")

    print("tool self-check")

    # --- utility: calculate ---------------------------------------------
    result = _call(tools["calculate"], expression="17*23 + 5!")
    check("calculate arithmetic", result == "511", f"-> {result!r}")

    result = _call(tools["calculate"], expression="sqrt(144) + pi")
    check("calculate functions", result.startswith("15.14"), f"-> {result!r}")

    result = _call(tools["calculate"], expression="__import__('os').system('echo pwned')")
    check("calculate rejects code execution", result.startswith("ERROR"), f"-> {result!r}")

    # --- utility: datetime ----------------------------------------------
    result = _call(tools["current_datetime"], timezone_name="Asia/Kolkata")
    check("current_datetime valid tz", "IST" in result or "+" in result, f"-> {result!r}")

    result = _call(tools["current_datetime"], timezone_name="Not/AZone")
    check("current_datetime rejects bad tz", result.startswith("ERROR"), f"-> {result!r}")

    # --- filesystem ------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp).resolve()
        scoped = replace(settings, workspace=workspace)
        scoped_policy = replace(policy, workspace=workspace)
        scoped_tools = {t.name: t for t in all_tools(scoped, scoped_policy)}

        result = _call(scoped_tools["write_file"], path="hello.txt", content="hello world")
        check("write_file", "Wrote" in result, f"-> {result!r}")

        result = _call(scoped_tools["read_file"], path="hello.txt")
        check("read_file", "hello world" in result, f"-> {result!r}")

        result = _call(scoped_tools["edit_file"], path="hello.txt", old_string="world", new_string="agent")
        check("edit_file", "Edited" in result, f"-> {result!r}")

        result = _call(scoped_tools["read_file"], path="hello.txt")
        check("edit_file applied", "hello agent" in result, f"-> {result!r}")

        result = _call(
            scoped_tools["edit_file"], path="hello.txt", old_string="nope", new_string="x"
        )
        check("edit_file errors on no match", result.startswith("ERROR"), f"-> {result!r}")

        result = _call(scoped_tools["list_directory"], path=".")
        check("list_directory", "hello.txt" in result, f"-> {result!r}")

        result = _call(scoped_tools["make_directory"], path="sub/dir")
        check("make_directory", "Created" in result, f"-> {result!r}")

        result = _call(scoped_tools["delete_path"], path="hello.txt")
        check("delete_path", "Deleted" in result, f"-> {result!r}")

        result = _call(scoped_tools["read_file"], path="missing.txt")
        check("read_file missing file errors", result.startswith("ERROR"), f"-> {result!r}")

        # --- execution ----------------------------------------------------
        result = _call(scoped_tools["run_python"], code="print(6*7)")
        check("run_python executes", result.strip() == "42", f"-> {result!r}")

        result = _call(scoped_tools["run_python"], code="raise ValueError('boom')")
        check("run_python captures stderr", "boom" in result, f"-> {result!r}")

        result = _call(scoped_tools["run_python"], code="import time; time.sleep(5)", timeout=1)
        check("run_python enforces timeout", "timed out" in result.lower(), f"-> {result!r}")

    # --- web -------------------------------------------------------------
    try:
        result = _call(tools["fetch_url"], url="https://example.com", max_chars=2000)
        check("fetch_url retrieves page", "Example Domain" in result, f"-> {result[:120]!r}")
    except Exception as exc:  # noqa: BLE001
        check("fetch_url retrieves page", False, f"(raised {type(exc).__name__}: {exc})")

    result = _call(tools["fetch_url"], url="not-a-url")
    check("fetch_url rejects bad scheme", result.startswith("ERROR"), f"-> {result!r}")

    # =====================================================================
    print("\nsecurity self-check")

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp).resolve()
        (workspace / ".env").write_text("GOOGLE_API_KEY=AIzaSyTESTKEY1234567890abcdefghij\n")
        (workspace / "leak.txt").write_text(
            "token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n"
            "password: hunter2secret\n"
        )
        (workspace / "safe.txt").write_text("nothing to see here\n")

        scoped = replace(settings, workspace=workspace)
        scoped_policy = replace(policy, workspace=workspace)
        scoped_tools = {t.name: t for t in all_tools(scoped, scoped_policy)}

        # 1. secret files are unreadable
        result = _call(scoped_tools["read_file"], path=".env")
        check("blocks reading .env", "blocked" in result.lower(), f"-> {result[:120]!r}")

        result = _call(scoped_tools["read_file"], path="safe.txt")
        check("allows reading normal files", "nothing to see" in result, f"-> {result[:120]!r}")

        # 2. path escape
        result = _call(scoped_tools["read_file"], path="../../etc/hosts")
        check("blocks path escape", "outside the workspace" in result, f"-> {result[:120]!r}")

        # 3. secrets redacted from what the model sees
        result = _call(scoped_tools["read_file"], path="leak.txt")
        check("redacts secrets in file output", "ghp_ABCDEF" not in result, f"-> {result[:160]!r}")
        check("redacts password values", "hunter2secret" not in result, f"-> {result[:160]!r}")

        # 4. SSRF
        for label, url in [
            ("loopback", "http://127.0.0.1:8080/"),
            ("localhost", "http://localhost/admin"),
            ("cloud metadata", "http://169.254.169.254/latest/meta-data/"),
            ("private range", "http://192.168.1.1/"),
        ]:
            result = _call(scoped_tools["fetch_url"], url=url)
            check(f"blocks SSRF to {label}", result.startswith("ERROR"), f"-> {result[:120]!r}")

        # 5. credentials are not inherited by subprocesses
        result = _call(
            scoped_tools["run_python"],
            code="import os; print('KEY=' + os.environ.get('GOOGLE_API_KEY', 'ABSENT'))",
        )
        check("subprocess cannot read API keys", "KEY=ABSENT" in result, f"-> {result[:120]!r}")

        # 6. read-only mode
        readonly = replace(settings, workspace=workspace, allow_destructive=False)
        readonly_policy = replace(policy, workspace=workspace, allow_destructive=False)
        readonly_tools = {t.name: t for t in all_tools(readonly, readonly_policy)}
        result = _call(readonly_tools["delete_path"], path="safe.txt")
        check("read-only blocks deletes", "read-only" in result.lower(), f"-> {result[:120]!r}")

        result = _call(readonly_tools["write_file"], path="safe.txt", content="x")
        check("read-only blocks overwrites", "read-only" in result.lower(), f"-> {result[:120]!r}")

        # 7. offline mode
        offline = replace(settings, workspace=workspace, allow_network=False)
        offline_policy = replace(policy, workspace=workspace, allow_network=False)
        offline_tools = {t.name: t for t in all_tools(offline, offline_policy)}
        result = _call(offline_tools["fetch_url"], url="https://example.com")
        check("offline blocks network", "offline" in result.lower(), f"-> {result[:120]!r}")

        # 8. destructive shell commands are rejected
        shell_policy = replace(policy, workspace=workspace, allow_shell=True)
        shell_settings = replace(settings, workspace=workspace, allow_shell=True)
        shell_tools = {t.name: t for t in all_tools(shell_settings, shell_policy)}
        result = _call(shell_tools["run_shell"], command="rm -rf /")
        check("blocks destructive shell", "blocked by the security policy" in result, f"-> {result[:120]!r}")

        result = _call(shell_tools["run_shell"], command="echo hello")
        check("allows safe shell", "hello" in result, f"-> {result[:120]!r}")

    print(f"\n{passed} passed, {len(failures)} failed")
    if failures:
        for name in failures:
            print(f"  - {name}")
        return len(failures)
    return 0
