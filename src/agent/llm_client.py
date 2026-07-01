"""
Pluggable LLM client interface.

`LLMClient` is the seam where Qwen gets integrated later. Everything in
`agent.py` is written against this interface, not against any concrete
provider — so wiring up Qwen will mean adding a `QwenLLMClient` subclass
here and changing one line where `MovieAgent` is constructed. No other
file in the project needs to change.

`StubLLMClient` is today's implementation: a small rule-based
"understander" that inspects the user's message with simple keyword
matching and decides which tool (if any) to call. It exists purely so
the full agent pipeline — message in, tool call, tool execution,
response out — can be exercised end-to-end right now, before any real
model is connected.
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod

from src.agent.schemas import LLMResponse, Message, ToolCall


class LLMClient(ABC):
    """Abstract interface every LLM backend (stub or Qwen) must implement."""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """Given conversation history and available tool schemas, return
        either a final text reply, or one or more tool calls to execute.

        Args:
            messages: Full conversation so far, oldest first, including
                the system prompt as messages[0].
            tools: Qwen/OpenAI-compatible tool schemas (see
                ``src.agent.tools.get_tool_schemas``).

        Returns:
            An LLMResponse with `content` and/or `tool_calls` populated.
        """
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """Rule-based placeholder standing in for Qwen.

    This is intentionally simple and NOT meant to be a good NLU system —
    its only job is to prove out the agent architecture (tool-calling
    loop, tool execution, response formatting) so that dropping in a real
    Qwen-backed client later is a mechanical swap, not a redesign.
    """

    # Very small set of trigger phrases for "the user wants recommendations".
    _RECOMMEND_TRIGGERS = (
        "recommend",
        "similar to",
        "like ",
        "suggest",
        "more movies like",
    )

    def chat(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        last_user_message = self._last_user_content(messages)

        if last_user_message is None:
            return LLMResponse(
                content="I didn't receive a message to respond to."
            )

        text = last_user_message.strip()
        lower = text.lower()

        # If we're being handed tool results (role="tool" at the end of
        # `messages`), this is the second half of the loop: summarize.
        if messages and messages[-1].role == "tool":
            return self._summarize_tool_result(messages[-1])

        if any(trigger in lower for trigger in self._RECOMMEND_TRIGGERS):
            movie_title = self._extract_movie_title(text)
            if movie_title:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=str(uuid.uuid4()),
                            name="recommend_movies",
                            arguments={"movie_title": movie_title},
                        )
                    ],
                )
            return LLMResponse(
                content=(
                    "Sure — which movie should I base recommendations on? "
                    "Try something like: 'recommend movies like Inception'."
                )
            )

        return LLMResponse(
            content=(
                "I can recommend movies similar to one you name — try "
                "asking, for example, 'suggest movies like The Dark Knight'."
            )
        )

    @staticmethod
    def _last_user_content(messages: list[Message]) -> str | None:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return None

    @staticmethod
    def _extract_movie_title(text: str) -> str | None:
        """Pull a movie title out of phrases like 'like X', 'similar to X'."""
        patterns = [
            r"(?:similar to|like|recommend(?:ations)? for)\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().strip("?.! ")
                if candidate:
                    return candidate
        return None

    @staticmethod
    def _summarize_tool_result(tool_message: Message) -> LLMResponse:
        content = tool_message.content

        if isinstance(content, dict) and content.get("error"):
            return LLMResponse(content=content["error"])

        if tool_message.name == "recommend_movies" and isinstance(content, dict):
            source = content.get("source_movie", "your pick")
            return LLMResponse(content=f"Here's what I found based on '{source}':")

        if tool_message.name == "search_movies" and isinstance(content, dict):
            matches = content.get("matches") or []
            if matches:
                return LLMResponse(
                    content="Closest matches: " + ", ".join(matches)
                )
            return LLMResponse(content="I couldn't find a close match for that.")

        return LLMResponse(content="Here's what I found.")
