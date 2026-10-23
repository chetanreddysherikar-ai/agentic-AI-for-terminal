"""Runtime configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Gemini's free tier is capped at ~20 requests per day *per model*, and the
# cap is tracked per model id. These give the agent a second, third and
# fourth chance when the primary model runs out of quota.
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = (
    "google_genai:gemini-3.1-flash-lite",
    "google_genai:gemini-flash-lite-latest",
    "google_genai:gemma-4-26b-a4b-it",
)
load_dotenv(PROJECT_ROOT / ".env")

TRUTHY = {"1", "true", "yes", "on"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in TRUTHY


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "")).strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Everything the agent needs to boot."""

    name: str
    model: str
    fallback_models: tuple[str, ...]
    workspace: Path
    session_db: Path
    max_steps: int
    temperature: float
    allow_shell: bool
    allow_outside_workspace: bool
    request_timeout: int
    max_output_chars: int
    verbose: bool
    allow_network: bool
    allow_destructive: bool
    redact_secrets: bool
    require_approval: bool
    audit_log: Path

    @classmethod
    def from_env(cls) -> "Settings":
        workspace = Path(os.getenv("AGENT_WORKSPACE") or PROJECT_ROOT).expanduser().resolve()
        db_raw = os.getenv("AGENT_SESSION_DB")
        session_db = (
            Path(db_raw).expanduser().resolve()
            if db_raw
            else (PROJECT_ROOT / ".agent" / "sessions.db").resolve()
        )
        session_db.parent.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)

        audit_raw = os.getenv("AGENT_AUDIT_LOG")
        audit_log = (
            Path(audit_raw).expanduser().resolve()
            if audit_raw
            else (PROJECT_ROOT / ".agent" / "audit.log").resolve()
        )

        fallbacks_raw = os.getenv("AGENT_MODEL_FALLBACKS", "").strip()
        if fallbacks_raw:
            fallback_models = tuple(
                part.strip()
                for part in fallbacks_raw.replace(";", ",").split(",")
                if part.strip()
            )
        else:
            fallback_models = DEFAULT_FALLBACK_MODELS

        return cls(
            name=os.getenv("AGENT_NAME", "CRS"),
            model=os.getenv("AGENT_MODEL", "google_genai:gemini-3.5-flash-lite"),
            fallback_models=fallback_models,
            workspace=workspace,
            session_db=session_db,
            max_steps=env_int("AGENT_MAX_STEPS", 25),
            temperature=env_float("AGENT_TEMPERATURE", 0.2),
            allow_shell=env_bool("AGENT_ALLOW_SHELL", False),
            allow_outside_workspace=env_bool("AGENT_ALLOW_OUTSIDE_WORKSPACE", False),
            request_timeout=env_int("AGENT_REQUEST_TIMEOUT", 30),
            max_output_chars=env_int("AGENT_MAX_OUTPUT_CHARS", 20000),
            verbose=env_bool("AGENT_VERBOSE", False),
            allow_network=env_bool("AGENT_ALLOW_NETWORK", True),
            allow_destructive=env_bool("AGENT_ALLOW_DESTRUCTIVE", True),
            redact_secrets=env_bool("AGENT_REDACT_SECRETS", True),
            require_approval=env_bool("AGENT_REQUIRE_APPROVAL", True),
            audit_log=audit_log,
        )

    def describe(self) -> str:
        return (
            f"model={self.model} workspace={self.workspace} "
            f"db={self.session_db} shell={self.allow_shell}"
        )
