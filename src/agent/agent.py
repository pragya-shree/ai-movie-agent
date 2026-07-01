"""
Agent orchestrator.

`MovieAgent` implements the standard tool-calling loop:

    1. Send conversation history + tool schemas to the LLMClient.
    2. If the LLM asks for a tool call, execute it against the existing,
       unmodified recommendation engine (via `src.agent.tools`).
    3. Feed the tool result back to the LLMClient for a final reply.
    4. Return an AgentResponse (text + structured recommendation data)
       to the UI layer.

This loop is written entirely against the `LLMClient` abstract interface
(`src.agent.llm_client.LLMClient`), so it is unaffected by which
implementation is passed in — `StubLLMClient` today, `QwenLLMClient`
later.
"""

from __future__ import annotations

import pandas as pd

from src.agent.llm_client import LLMClient
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.schemas import AgentResponse, Message, ToolCall
from src.agent.tools import build_tools, execute_tool

MAX_TOOL_ITERATIONS = 3


class MovieAgent:
    """Conversational front-end for the movie recommendation engine."""

    def __init__(self, llm_client: LLMClient, movies: pd.DataFrame, similarity):
        self._llm = llm_client
        self._tools = build_tools(movies, similarity)
        self._tool_schemas = [tool.to_schema() for tool in self._tools]

    def handle_message(
        self, user_message: str, history: list[Message] | None = None
    ) -> AgentResponse:
        """Process one user message and return the agent's response.

        Args:
            user_message: The new message from the user.
            history: Prior conversation turns (excluding the system
                prompt), oldest first. Pass None for a fresh conversation.

        Returns:
            AgentResponse with a text reply and any structured
            recommendation data produced by tool calls.
        """
        messages: list[Message] = [Message(role="system", content=SYSTEM_PROMPT)]
        messages.extend(history or [])
        messages.append(Message(role="user", content=user_message))

        recommendations: list[dict] = []
        tool_calls_made: list[str] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._llm.chat(messages, self._tool_schemas)

            if not response.tool_calls:
                return AgentResponse(
                    reply=response.content,
                    recommendations=recommendations,
                    tool_calls_made=tool_calls_made,
                )

            # Record the assistant's tool-call request in history, then
            # execute each requested tool and append its result.
            messages.append(
                Message(role="assistant", content="", tool_calls=response.tool_calls)
            )

            for tool_call in response.tool_calls:
                tool_calls_made.append(tool_call.name)
                result_message = self._run_tool(tool_call)
                messages.append(result_message)

                if tool_call.name == "recommend_movies" and isinstance(
                    result_message.content, dict
                ):
                    recommendations.extend(
                        result_message.content.get("recommendations", [])
                    )

        # Safety valve: if the LLM keeps requesting tools past the
        # iteration cap, return whatever we've gathered rather than loop
        # forever.
        return AgentResponse(
            reply="I found some results, but had trouble wrapping up cleanly.",
            recommendations=recommendations,
            tool_calls_made=tool_calls_made,
        )

    def _run_tool(self, tool_call: ToolCall) -> Message:
        """Execute a tool call and package the outcome as a tool Message.

        `content` stays structured (dict/list), not a JSON string — the
        stub LLM and MovieAgent both consume it directly, and
        `Message.to_dict()` handles JSON-encoding it only when/if a real
        wire-format API call needs it.
        """
        try:
            content = execute_tool(tool_call.name, tool_call.arguments, self._tools)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the LLM
            content = {"error": str(exc)}

        return Message(
            role="tool",
            content=content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )
