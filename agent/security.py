"""Security policy for the agent.

Defends against the things that actually go wrong with a tool-using agent:

1. **Path escape** - tools resolve paths inside the workspace only.
2. **Secret exposure** - sensitive files are off limits and secret-looking
   strings are redacted from anything the model reads.
3. **SSRF** - the agent cannot reach loopback, private or cloud-metadata hosts.
4. **Credential leakage to subprocesses** - API keys are stripped from the
   environment handed to `run_python` / `run_shell`.
5. **Unbounded damage** - destructive operations can be gated or disabled.
6. **Audit trail** - every sensitive action is recorded.
"""
from __future__ import annotations

import fnmatch
import ipaddress
import json
import os
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

BLOCKED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "metadata",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
}

SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.example",
    ".netrc",
    ".pgpass",
    ".htpasswd",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
    "authorized_keys",
    "secrets.yaml",
    "secrets.yml",
    "secrets.json",
}

SENSITIVE_GLOBS = (
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "*.jks",
    "*.keystore",
    "*.env",
    ".env.*",
    "*.kdbx",
    "*.ppk",
)

SENSITIVE_DIRS = {".ssh", ".aws", ".gnupg", ".gpg", ".kube", ".docker", ".azure"}

SECRET_ENV_VARS = {
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "SERPER_API_KEY",
    "BRAVE_API_KEY",
    "EXA_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
}

_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"),
     "[PRIVATE KEY REDACTED]"),
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"), "AIza[REDACTED]"),
    (re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}"), "AQ.[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA[REDACTED]"),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), "ASIA[REDACTED]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "[JWT REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "sk-[REDACTED]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"), "[GITHUB TOKEN REDACTED]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"), "[SLACK TOKEN REDACTED]"),
    (re.compile(r"(?i)\b(authorization\s*:\s*(?:bearer|basic)\s+)\S+"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)\b(google_api_key|gemini_api_key|openai_api_key|anthropic_api_key|"
            r"aws_secret_access_key|aws_access_key_id|tavily_api_key|serper_api_key|"
            r"api_key|api_secret|client_secret|access_token|private_key|password|passwd)"
            r"\b(\s*[:=]\s*)[\"']?([^\s\"',;]{4,})"
        ),
        r"\1\2[REDACTED]",
    ),
)

SHELL_DENY_PATTERNS = (
    r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]",
    r"\bmkfs\b",
    r"\bdd\s+if=.*of=/dev/",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bformat\s+[a-zA-Z]:",
    r"\bdel\s+/[a-zA-Z]",
    r"\bRemove-Item\b.*-Recurse.*-Force",
    r":\(\)\s*\{",
    r"\bchmod\s+777\s+/",
    r"\bcurl\b[^|]*\|\s*(ba)?sh",
    r"\bwget\b[^|]*\|\s*(ba)?sh",
    r"\bgit\s+push\b",
    r"\bnpm\s+publish\b",
    r"\bpip\s+install\b",
)


class SecurityError(Exception):
    """Raised when an action violates the security policy."""


@dataclass
class SecurityPolicy:
    """Runtime security rules for a single agent session."""

    workspace: Path
    allow_network: bool = True
    allow_destructive: bool = True
    allow_shell: bool = False
    redact_secrets: bool = True
    audit_log: Path | None = None
    extra_sensitive_globs: tuple[str, ...] = field(default_factory=tuple)

    # ---------------------------------------------------------------- paths
    def resolve_path(self, path: str, *, allow_sensitive: bool = False) -> Path:
        """Resolve a path inside the workspace, rejecting escapes."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:  # pragma: no cover - defensive
            raise SecurityError(f"cannot resolve path '{path}': {exc}") from exc

        if not resolved.is_relative_to(self.workspace):
            raise SecurityError(
                f"path '{resolved}' is outside the workspace '{self.workspace}'. "
                "Blocked to prevent escaping the sandbox."
            )
        if not allow_sensitive and self.is_sensitive(resolved):
            raise SecurityError(
                f"path '{resolved.name}' looks like a secret store and is blocked. "
                "Copy the value you need into a normal file first."
            )
        return resolved

    def is_sensitive(self, path: Path) -> bool:
        """True when the path points at credentials or secret material."""
        name = path.name
        if name.lower() in SENSITIVE_NAMES:
            return True
        for pattern in (*SENSITIVE_GLOBS, *self.extra_sensitive_globs):
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                return True
        for part in path.parts:
            if part.lower() in SENSITIVE_DIRS:
                return True
        return False

    # --------------------------------------------------------------- secrets
    def redact(self, text: str) -> str:
        """Mask secret-looking strings so the model never sees them."""
        if not self.redact_secrets or not text:
            return text
        redacted = text
        for pattern, replacement in _REDACTIONS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    def sanitized_env(self) -> dict[str, str]:
        """Environment for child processes, with credentials removed."""
        env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV_VARS}
        env["AGENT_SECURITY_SANDBOXED"] = "1"
        return env

    # ----------------------------------------------------------------- audit
    def audit(self, event: str, **details: object) -> None:
        """Append a redacted audit record."""
        if not self.audit_log:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "detail": self.redact(json.dumps(details, default=str)),
        }
        try:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            pass

    # --------------------------------------------------------------- network
    def check_url(self, url: str) -> None:
        """Block requests to loopback, private and cloud-metadata hosts."""
        if not self.allow_network:
            raise SecurityError(
                "network access is disabled for this session (--offline)."
            )
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise SecurityError("only http:// and https:// URLs are allowed.")
        host = (parsed.hostname or "").lower()
        if not host:
            raise SecurityError("URL has no host.")
        if host in BLOCKED_HOSTS:
            raise SecurityError(f"host '{host}' is blocked (SSRF protection).")

        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror as exc:
            raise SecurityError(f"cannot resolve host '{host}': {exc}") from exc

        for info in infos:
            address = info[4][0]
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                raise SecurityError(
                    f"host '{host}' resolves to a non-public address ({address}). Blocked."
                )

    def check_shell(self, command: str) -> None:
        """Reject shell commands that look destructive."""
        lowered = command.lower()
        for pattern in SHELL_DENY_PATTERNS:
            if re.search(pattern, lowered):
                raise SecurityError(
                    f"command blocked by the security policy: matches '{pattern}'. "
                    "Run it yourself if you really mean it."
                )
