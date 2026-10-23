"""Filesystem tools: read, write, edit, list. Sandboxed and policy-aware."""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.security import SecurityError, SecurityPolicy


def filesystem_tools(settings: Settings, policy: SecurityPolicy) -> list[BaseTool]:
    """Build filesystem tools bound to a workspace and security policy."""

    def resolve(path: str) -> Path:
        if settings.allow_outside_workspace:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = settings.workspace / candidate
            resolved = candidate.resolve(strict=False)
            if not policy.is_sensitive(resolved):
                return resolved
        return policy.resolve_path(path)

    def relative(path: Path) -> str:
        try:
            return str(path.relative_to(settings.workspace))
        except ValueError:
            return str(path)

    def blocked(tool: str, path: str, exc: SecurityError) -> str:
        policy.audit(f"{tool}_blocked", path=path, reason=str(exc)[:200])
        return f"ERROR: {exc}"

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a UTF-8 text file from the workspace.

        Credential stores (.env, ssh keys, token files) are blocked and any
        secret-looking values in the file are redacted before you see them.

        Args:
            path: File path, relative to the workspace root or absolute.
            start_line: First line to return (1-indexed).
            end_line: Last line to return, 0 means "until end of file".
        """
        try:
            target = resolve(path)
        except SecurityError as exc:
            return blocked("read_file", path, exc)
        if not target.exists():
            return f"ERROR: file not found: {target}"
        if target.is_dir():
            return f"ERROR: '{target}' is a directory, use list_directory instead."
        if target.stat().st_size > MAX_READ_BYTES:
            return (
                f"ERROR: file is {target.stat().st_size} bytes, too large to read at once. "
                "Read it in chunks using start_line/end_line."
            )

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        first = max(1, start_line)
        last = min(end_line if end_line and end_line >= first else total, total)

        body = "\n".join(f"{i:>6} | {lines[i - 1]}" for i in range(first, last + 1))
        policy.audit("read_file", path=str(target))
        return policy.redact(f"{relative(target)} ({total} lines, showing {first}-{last})\n{body}")

    @tool
    def write_file(path: str, content: str) -> str:
        """Create a file or overwrite it completely with the given content.

        Args:
            path: File path, relative to the workspace root or absolute.
            content: Full text content to write.
        """
        try:
            target = resolve(path)
        except SecurityError as exc:
            return blocked("write_file", path, exc)
        if not policy.allow_destructive and target.exists():
            return "ERROR: this session is read-only, overwriting existing files is blocked."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        policy.audit("write_file", path=str(target), bytes=len(content))
        return f"Wrote {len(content)} chars to {relative(target)}"

    @tool
    def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Replace an exact substring inside an existing file.

        The old_string must match exactly and appear at least once. This is safer
        than rewriting the whole file because it fails loudly on a mismatch.

        Args:
            path: File path, relative to the workspace root or absolute.
            old_string: Exact text to find.
            new_string: Text to replace it with.
            replace_all: Replace every occurrence instead of just the first.
        """
        try:
            target = resolve(path)
        except SecurityError as exc:
            return blocked("edit_file", path, exc)
        if not policy.allow_destructive:
            return "ERROR: this session is read-only, editing files is blocked."
        if not target.exists():
            return f"ERROR: file not found: {target}"
        text = target.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old_string)
        if occurrences == 0:
            return f"ERROR: old_string not found in {relative(target)}"
        if occurrences > 1 and not replace_all:
            return (
                f"ERROR: old_string appears {occurrences} times in {relative(target)}. "
                "Add more context or set replace_all=true."
            )
        text = text.replace(old_string, new_string, -1 if replace_all else 1)
        target.write_text(text, encoding="utf-8")
        replaced = occurrences if replace_all else 1
        policy.audit("edit_file", path=str(target), replacements=replaced)
        return f"Edited {relative(target)} ({replaced} replacement(s))"

    @tool
    def list_directory(path: str = ".") -> str:
        """List the files and folders inside a directory.

        Args:
            path: Directory path, relative to the workspace root or absolute.
        """
        try:
            target = resolve(path)
        except SecurityError as exc:
            return blocked("list_directory", path, exc)
        if not target.exists():
            return f"ERROR: directory not found: {target}"
        if not target.is_dir():
            return f"ERROR: '{target}' is not a directory."

        rows = []
        for entry in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            kind = "dir " if entry.is_dir() else "file"
            size = "" if entry.is_dir() else f"{entry.stat().st_size:>8}B"
            flag = " [sensitive: blocked]" if policy.is_sensitive(entry) else ""
            rows.append(f"{kind} {size}  {entry.name}{flag}")

        header = f"{relative(target) or '.'} ({len(rows)} entries)"
        policy.audit("list_directory", path=str(target))
        return policy.redact(header + "\n" + ("\n".join(rows) if rows else "(empty)"))

    @tool
    def make_directory(path: str) -> str:
        """Create a directory (including parents) inside the workspace.

        Args:
            path: Directory path, relative to the workspace root or absolute.
        """
        try:
            target = resolve(path)
        except SecurityError as exc:
            return blocked("make_directory", path, exc)
        target.mkdir(parents=True, exist_ok=True)
        policy.audit("make_directory", path=str(target))
        return f"Created directory {relative(target)}"

    @tool
    def delete_path(path: str) -> str:
        """Delete a file or an empty directory.

        Args:
            path: Path to delete, relative to the workspace root or absolute.
        """
        if not policy.allow_destructive:
            return "ERROR: this session is read-only, deleting files is blocked."
        try:
            target = resolve(path)
        except SecurityError as exc:
            return blocked("delete_path", path, exc)
        if not target.exists():
            return f"ERROR: path not found: {target}"
        if target.is_dir():
            if any(target.iterdir()):
                return f"ERROR: directory is not empty: {relative(target)}"
            target.rmdir()
            policy.audit("delete_path", path=str(target), kind="dir")
            return f"Removed directory {relative(target)}"
        target.unlink()
        policy.audit("delete_path", path=str(target), kind="file")
        return f"Deleted {relative(target)}"

    return [
        read_file,
        write_file,
        edit_file,
        list_directory,
        make_directory,
        delete_path,
    ]


MAX_READ_BYTES = 200_000
