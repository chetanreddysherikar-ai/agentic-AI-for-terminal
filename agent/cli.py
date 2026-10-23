"""Interactive CLI for the agent."""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from agent.builder import APPROVAL_GATED_TOOLS, build_agent, build_policy
from agent.config import Settings
from agent.memory import SessionStore

BANNER = r"""
  ___    ___   ____        {name}  -  local AI agent
 / __|  / _ \ |  _ \       model:     {model}
| (__  | |_) || |_) |      fallback:  {fallbacks}
 \___| |  _ < | ___) |      session:   {session}
      |_| \_\|____/        workspace: {workspace}
"""

HELP = """
Commands
  /help            show this help
  /tools           list available tools
  /policy          show the active security policy
  /sessions        list stored sessions
  /new             start a new session (clears context)
  /history         show the current session's messages
  /model           show the active model
  /clear           delete the current session from the database
  /exit, /quit     leave the agent

Just type a task and press Enter.
"""


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _preview(value: Any, limit: int = 220) -> str:
    if isinstance(value, dict):
        value = ", ".join(f"{k}={v!r}" for k, v in value.items())
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + " ..."


def friendly_error(exc: BaseException) -> str:
    """Turn raw provider errors into something actionable."""
    blob = str(exc)
    detail = f"{type(exc).__name__}: {exc}"

    if "429" in blob or "RESOURCE_EXHAUSTED" in blob:
        delay = re.search(r"Please retry in ([0-9a-zA-Z.]+)", blob)
        metric = re.search(r"quotaId': '([A-Za-z_]+)", blob)
        limit = re.search(r"limit: (\d+)", blob)
        period = "?"
        if metric:
            if "PerDay" in metric.group(1):
                period = "day"
            elif "PerMinute" in metric.group(1):
                period = "minute"
        return (
            "Gemini quota exhausted (429).\n"
            f"  metric : {metric.group(1) if metric else 'unknown'}\n"
            f"  limit  : {limit.group(1) if limit else '?'} requests per {period}\n"
            f"  resets : {delay.group(1) if delay else 'unknown'}\n"
            "  fix    : use a paid key, switch model with --model, or wait for the reset.\n"
            f"detail: {detail[:240]}"
        )

    if "401" in blob or "403" in blob or "API key" in blob:
        return (
            "Gemini rejected the API key. Check GOOGLE_API_KEY in .env.\n"
            f"detail: {detail[:300]}"
        )
    if "503" in blob or "UNAVAILABLE" in blob:
        return (
            "Gemini is overloaded right now (503). Try again in a minute, or use a\n"
            "different model with --model.\n"
            f"detail: {detail[:300]}"
        )
    return detail[:600]


class AgentRunner:
    """Wraps the compiled agent and renders its output for a terminal."""

    def __init__(
        self,
        agent,
        store: SessionStore,
        session: str,
        stream: bool = True,
        auto_approve: bool = False,
        policy=None,
    ) -> None:
        self.agent = agent
        self.store = store
        self.session = session
        self.stream = stream
        self.auto_approve = auto_approve
        self.policy = policy
        self.config = store.config(session)
        self._fallback_text = ""

    # ------------------------------------------------------------------ run
    def run(self, user_input: str) -> str:
        payload: Any = {"messages": [{"role": "user", "content": user_input}]}
        produced = False
        self._fallback_text = ""

        while True:
            interrupted = False
            for mode, chunk in self.agent.stream(
                payload, config=self.config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    message, _metadata = chunk
                    if isinstance(message, AIMessageChunk):
                        text = _text_of(message.content)
                        if text:
                            produced = True
                            sys.stdout.write(text)
                            sys.stdout.flush()
                elif mode == "updates":
                    if isinstance(chunk, dict) and "__interrupt__" in chunk:
                        interrupted = True
                        payload = self._resume_command(chunk["__interrupt__"])
                    else:
                        self._render_updates(chunk)
            if not interrupted:
                break

        if not produced:
            if self._fallback_text.strip():
                sys.stdout.write(self._fallback_text + "\n")
            else:
                print(
                    "\n[no response produced - the model call was rate limited, "
                    "failed, or hit the step limit]",
                    flush=True,
                )
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()
        return ""

    # ------------------------------------------------------------ approval
    def _resume_command(self, interrupts: Any) -> Command:
        """Ask the user to approve pending tool calls and build the resume command."""
        payload = None
        for item in interrupts or ():
            payload = getattr(item, "value", item)
            break
        if not isinstance(payload, dict):
            return Command(resume={"decisions": []})

        actions = payload.get("action_requests", []) or []
        decisions: list[dict] = []
        approve_all = False
        reject_all = False

        for action in actions:
            name = action.get("name", "tool")
            args = action.get("args", {})
            print(f"\n[approval required] {name}({_preview(args)})", flush=True)
            if approve_all:
                decisions.append({"type": "approve"})
                self._audit("approval_granted", name, args)
                continue
            if reject_all:
                decisions.append({"type": "reject", "message": "Rejected by the user."})
                self._audit("approval_rejected", name, args)
                continue
            if self.auto_approve:
                decisions.append({"type": "approve"})
                self._audit("approval_granted", name, args)
                continue
            try:
                answer = input("  approve? [y]es / [n]o / [a]ll / [r]eject all: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in {"a", "all"}:
                approve_all = True
                decisions.append({"type": "approve"})
                self._audit("approval_granted", name, args)
            elif answer in {"r", "reject", "all_no"}:
                reject_all = True
                decisions.append({"type": "reject", "message": "Rejected by the user."})
                self._audit("approval_rejected", name, args)
            elif answer in {"y", "yes"}:
                decisions.append({"type": "approve"})
                self._audit("approval_granted", name, args)
            else:
                decisions.append(
                    {"type": "reject", "message": "Rejected by the user. Do not retry."}
                )
                self._audit("approval_rejected", name, args)

        return Command(resume={"decisions": decisions})

    def _audit(self, event: str, name: str, args: Any) -> None:
        if self.policy is not None:
            self.policy.audit(event, tool=name, args=args)

    # ------------------------------------------------------------ rendering
    def _render_updates(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            for message in update.get("messages", []) or []:
                if node == "model":
                    calls = getattr(message, "tool_calls", []) or []
                    for call in calls:
                        print(
                            f"\n[tool] {call.get('name')}({_preview(call.get('args'))})",
                            flush=True,
                        )
                    if not calls:
                        text = _text_of(getattr(message, "content", ""))
                        if text.strip():
                            self._fallback_text = text
                elif node == "tools":
                    name = getattr(message, "name", "tool")
                    print(f"[result] {name}: {_preview(message.content)}", flush=True)

    def history(self) -> str:
        state = self.agent.get_state(self.config)
        rows = []
        for message in (state.values or {}).get("messages", []) or []:
            kind = getattr(message, "type", "?")
            content = _text_of(getattr(message, "content", ""))
            if kind == "tool":
                content = f"[{getattr(message, 'name', 'tool')}] {_preview(content)}"
            rows.append(f"{kind:>10} | {content}")
        return "\n".join(rows) if rows else "(empty)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings_name = os.getenv("AGENT_NAME", "CRS")
    parser = argparse.ArgumentParser(
        prog="crs",
        description=f"{settings_name} - a local, tool-using AI agent.",
    )
    parser.add_argument("task", nargs="*", help="Run a single task and exit.")
    parser.add_argument("--session", default="default", help="Session name (memory is per session).")
    parser.add_argument("--model", default=None, help="Override the model, e.g. gemini-2.5-pro.")
    parser.add_argument("--workspace", default=None, help="Directory the agent is allowed to work in.")
    parser.add_argument("--shell", action="store_true", help="Enable the run_shell tool.")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming.")
    parser.add_argument("--list-sessions", action="store_true", help="List stored sessions and exit.")
    parser.add_argument("--clear-session", action="store_true", help="Delete the current session and exit.")
    parser.add_argument("--print-tools", action="store_true", help="List the tool schema and exit.")
    parser.add_argument(
        "--selfcheck", action="store_true", help="Run the offline tool self-check and exit."
    )
    parser.add_argument(
        "--demo",
        default=None,
        help="Run a scripted demo with a mock model (no API key, no quota).",
    )
    parser.add_argument(
        "--list-demos", action="store_true", help="List the available demos and exit."
    )
    security = parser.add_argument_group("security")
    security.add_argument(
        "--read-only", action="store_true", help="Block writes, edits and deletes."
    )
    security.add_argument("--offline", action="store_true", help="Block all outbound network access.")
    security.add_argument(
        "--no-approve", action="store_true", help="Do not ask for approval before risky tools."
    )
    security.add_argument(
        "--no-redact", action="store_true", help="Do not redact secrets from tool output."
    )
    security.add_argument("--audit-log", default=None, help="Path of the audit log file.")
    security.add_argument(
        "--yes", action="store_true", help="Auto-approve every gated tool call (scripting only)."
    )
    return parser.parse_args(argv)


def _apply_flags(settings: Settings, args: argparse.Namespace) -> Settings:
    if args.model:
        settings = replace(settings, model=args.model)
    if args.workspace:
        workspace = Path(args.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        settings = replace(settings, workspace=workspace)
    if args.shell:
        settings = replace(settings, allow_shell=True)
    if args.read_only:
        settings = replace(settings, allow_destructive=False)
    if args.offline:
        settings = replace(settings, allow_network=False)
    if args.no_approve:
        settings = replace(settings, require_approval=False)
    if args.no_redact:
        settings = replace(settings, redact_secrets=False)
    if args.audit_log:
        settings = replace(settings, audit_log=Path(args.audit_log).expanduser().resolve())
    return settings


def _policy_summary(settings: Settings) -> str:
    policy = build_policy(settings)
    return (
        f"workspace        {policy.workspace}\n"
        f"network          {'allowed' if policy.allow_network else 'BLOCKED'}\n"
        f"writes/deletes   {'allowed' if policy.allow_destructive else 'BLOCKED'}\n"
        f"shell tool       {'enabled' if policy.allow_shell else 'disabled'}\n"
        f"secret redaction {'on' if policy.redact_secrets else 'off'}\n"
        f"approval gate    {'on (' + ', '.join(APPROVAL_GATED_TOOLS) + ')' if settings.require_approval else 'off'}\n"
        f"audit log        {policy.audit_log}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = _apply_flags(Settings.from_env(), args)

    store = SessionStore(settings.session_db)

    demo_model = None
    if args.demo:
        from agent.mock import DEMOS, build_demo_model

        if args.demo not in DEMOS:
            print(f"unknown demo '{args.demo}'. available: {', '.join(DEMOS)}", file=sys.stderr)
            return 2
        demo_model = build_demo_model(args.demo)
        print(f"[demo: {args.demo}] {DEMOS[args.demo]['description']}\n")

    if args.list_demos:
        from agent.mock import DEMOS

        for name, spec in DEMOS.items():
            print(f"- {name}: {spec['description']}")
        return 0

    if args.selfcheck:
        from agent.selfcheck import run_selfcheck

        return 1 if run_selfcheck(settings) else 0

    if args.list_sessions:
        threads = store.threads()
        print("\n".join(threads) if threads else "(no stored sessions)")
        return 0

    if args.clear_session:
        store.clear(args.session)
        print(f"Cleared session '{args.session}'.")
        return 0

    agent, tools = build_agent(settings, checkpointer=store.saver, model=demo_model)

    if args.print_tools:
        for t in tools:
            print(f"- {t.name}: {t.description}")
        return 0

    runner = AgentRunner(
        agent,
        store,
        session=args.session,
        stream=not args.no_stream,
        auto_approve=args.yes,
        policy=build_policy(settings),
    )

    if args.task or args.demo:
        task = " ".join(args.task) if args.task else f"run the {args.demo} demo"
        try:
            runner.run(task)
        except Exception as exc:  # noqa: BLE001
            print(f"\nERROR: {friendly_error(exc)}", file=sys.stderr)
            return 1
        return 0

    print(
        BANNER.format(
            name=settings.name,
            model=settings.model,
            fallbacks=(
                ", ".join(m.split(":")[-1] for m in settings.fallback_models) or "none"
            ),
            session=args.session,
            workspace=settings.workspace,
        )
    )
    print(_policy_summary(settings))
    print("\nType /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            print("bye")
            return 0
        if user_input == "/help":
            print(HELP)
            continue
        if user_input == "/tools":
            for t in tools:
                print(f"- {t.name}: {_preview(t.description, 160)}")
            continue
        if user_input == "/policy":
            print(_policy_summary(settings))
            continue
        if user_input == "/sessions":
            print("\n".join(store.threads()) or "(no stored sessions)")
            continue
        if user_input == "/model":
            print(settings.model)
            continue
        if user_input == "/new":
            runner.session = store.new_session()
            runner.config = store.config(runner.session)
            print(f"New session: {runner.session}")
            continue
        if user_input == "/history":
            print(runner.history())
            continue
        if user_input == "/clear":
            store.clear(runner.session)
            print(f"Cleared session '{runner.session}'.")
            continue

        try:
            runner.run(user_input)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as exc:  # noqa: BLE001
            print(f"\nERROR: {friendly_error(exc)}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
