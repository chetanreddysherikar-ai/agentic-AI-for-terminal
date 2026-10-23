"""A scripted chat model, used for offline demos and tests.

The Gemini free tier allows only ~20 requests/day, which makes it impossible to
demo the agent loop on demand. This model replays a fixed list of assistant
turns (text and/or tool calls) so the whole pipeline - streaming, tool
execution, approval prompts, memory - can be exercised with no API key.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Replays a scripted sequence of assistant turns."""

    script: list[dict[str, Any]]
    cursor: list[int] = [0]
    profile: Any = {"max_input_tokens": 1_000_000, "max_output_tokens": 8192}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"steps": len(self.script)}

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = self.cursor[0]
        self.cursor[0] = index + 1
        step = self.script[min(index, len(self.script) - 1)]

        tool_calls = [
            {
                "name": call["name"],
                "args": call.get("args", {}),
                "id": call.get("id", f"call_{index}_{call['name']}"),
                "type": "tool_call",
            }
            for call in step.get("tool_calls", [])
        ]
        message = AIMessage(content=step.get("content", ""), tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


DEMOS: dict[str, dict] = {
    "build-file": {
        "description": "Create a file, run it, read it back, and summarise.",
        "script": [
            {
                "tool_calls": [
                    {
                        "name": "write_file",
                        "args": {"path": "demo/hello.py", "content": "print('Hello Agent')\n"},
                    }
                ]
            },
            {
                "tool_calls": [
                    {"name": "run_python", "args": {"code": "print(open('demo/hello.py').read().strip())"}}
                ]
            },
            {
                "tool_calls": [
                    {"name": "list_directory", "args": {"path": "demo"}}
                ]
            },
            {"content": "Created demo/hello.py, executed it, and confirmed the directory contents."},
        ],
    },
    "blocked-escape": {
        "description": "Try to read a file outside the workspace; the policy blocks it.",
        "script": [
            {
                "tool_calls": [
                    {"name": "read_file", "args": {"path": "../../../etc/passwd"}}
                ]
            },
            {
                "content": "The read was blocked by the sandbox, which is the expected behaviour.",
            },
        ],
    },
    "blocked-secret": {
        "description": "Try to read .env; the policy blocks it and redacts secrets.",
        "script": [
            {
                "tool_calls": [
                    {"name": "read_file", "args": {"path": ".env"}}
                ]
            },
            {
                "content": "Reading .env is blocked by the security policy - credential files are off limits.",
            },
        ],
    },
    "ssrf": {
        "description": "Try to fetch the cloud metadata endpoint; SSRF protection blocks it.",
        "script": [
            {
                "tool_calls": [
                    {"name": "fetch_url", "args": {"url": "http://169.254.169.254/latest/meta-data/"}}
                ]
            },
            {"content": "The metadata endpoint was blocked by the SSRF policy."},
        ],
    },
}


def build_demo_model(name: str) -> ScriptedChatModel:
    """Instantiate the scripted model for a named demo."""
    if name not in DEMOS:
        raise KeyError(f"unknown demo '{name}'. available: {', '.join(DEMOS)}")
    return ScriptedChatModel(script=list(DEMOS[name]["script"]))
