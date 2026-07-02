"""
Data structures shared across the agent layer.

Kept as plain dataclasses (no pydantic dependency) so this package has
zero new third-party requirements until an LLM client is actually wired
in. The shapes below intentionally mirror the OpenAI/Gemini-compatible
chat-completions + tool-calling format, so swapping ``StubLLMClient`` for
a real Gemini client later requires no changes to ``agent.py`` or
``tools.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single turn in the conversation, OpenAI/Gemini chat-format shaped.

    `content` is typed as Any rather than str: for role="assistant"/"user"
    it's always plain text, but for role="tool" it carries the tool's
    structured (JSON-serializable) result so callers don't have to
    re-parse JSON. `to_dict()` normalizes it to a string for any wire
    format that expects one.
    """

    role: Role
    content: Any
    # Populated on assistant messages that request tool execution.
    tool_calls: list["ToolCall"] = field(default_factory=list)
    # Populated on role="tool" messages: which call this result answers.
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the plain dict shape most chat-completion APIs expect."""
        content = (
            self.content
            if isinstance(self.content, str)
            else json.dumps(self.content)
        )
        out: dict[str, Any] = {"role": self.role, "content": content}
        if self.tool_calls:
            out["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class ToolCall:
    """A request from the LLM to execute a specific tool with arguments."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    """The outcome of executing a ToolCall, fed back to the LLM as context."""

    tool_call_id: str
    name: str
    content: Any  # JSON-serializable payload (dict/list/str)
    error: str | None = None


@dataclass
class LLMResponse:
    """What an LLMClient.chat() call returns."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class AgentResponse:
    """What MovieAgent.handle_message() returns to the UI layer."""

    reply: str
    # Structured recommendation payload, if any tool produced one — lets
    # the UI render poster cards instead of parsing them out of text.
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)
