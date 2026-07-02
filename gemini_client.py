"""
Gemini API client.

Handles all outbound requests to Google Gemini, via the official
Google Gen AI SDK (`google-genai`, imported as `google.genai`). This is
the single place in the project that knows how to talk to Gemini —
nothing else should import `google.genai` directly.

This module replaces the previous `qwen_client.py`. It's a straight
provider swap: same responsibilities, same public shape, same fail-soft
behavior — only the backend changed, from Qwen (DashScope, called via
raw `requests`) to Gemini (called via Google's official SDK). Nothing
elsewhere in the app needs to know which provider is behind it.

Two layers are provided, same as before:

- `get_gemini_response(user_message, context="")` — the simple,
  synchronous, text-in/text-out function used by the chat UI in app.py.
  (This is the direct replacement for the old `get_qwen_response`.)
- `call_gemini_chat(messages, tools=None)` — a lower-level building
  block (OpenAI-style messages/tools in, a normalized dict out) kept for
  parity with the previous client and for any future multi-turn/tool-
  calling agent work. `get_gemini_response` is implemented in terms of
  it.

Nothing here touches the recommendation engine, the similarity matrix,
or `src.recommender` — this module only produces conversational text.
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import get_gemini_api_key, GEMINI_MODEL_NAME

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly, concise assistant for a movie recommendation app. "
    "Help the user find movies they'll like."
)

# Fallback strings returned instead of raising, so a chat UI never
# crashes on a missing key or network hiccup — same fail-soft convention
# as src/poster_service.py's placeholder URLs (and the old qwen_client.py).
ERROR_NO_API_KEY = (
    "The Gemini API key isn't configured, so I can't respond right now. "
    "Set GEMINI_API_KEY in your .env (local) or Streamlit secrets (cloud)."
)
ERROR_REQUEST_FAILED = (
    "I couldn't reach Gemini just now — please try again in a moment."
)
ERROR_MALFORMED_RESPONSE = (
    "I got an unexpected response from Gemini and couldn't parse a reply."
)


class GeminiAPIError(Exception):
    """Raised by call_gemini_chat() on request failure.

    get_gemini_response() catches this and returns a fallback string
    instead of raising, but callers that need the raw response may want
    to handle this themselves.
    """


def _messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[str, list[types.Content]]:
    """Translate OpenAI-style messages into Gemini's expected shapes.

    Gemini takes a `system_instruction` (a single string) separately
    from the turn-by-turn `contents`, and uses "model" rather than
    "assistant" as the role name for prior AI turns. Any "system"-role
    messages are concatenated into the system instruction; everything
    else becomes a Content entry.
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []

    role_map = {"user": "user", "assistant": "model", "tool": "user"}

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            if content:
                system_parts.append(str(content))
            continue

        text = content if isinstance(content, str) else str(content)
        contents.append(
            types.Content(role=role_map.get(role, "user"), parts=[types.Part(text=text)])
        )

    return "\n\n".join(system_parts), contents


def call_gemini_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    model: str = GEMINI_MODEL_NAME,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """Low-level call to Gemini via the official google-genai SDK.

    Args:
        messages: OpenAI-style message dicts, e.g.
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}].
        tools: Optional tool schemas. NOTE: unlike the old Qwen client,
            these are not translated/forwarded yet — accepted here only
            for interface parity with the previous client, so any future
            agent work has a stable call signature to build on.
        model: Gemini model name (default from src.config.GEMINI_MODEL_NAME).
        temperature: Sampling temperature.
        max_output_tokens: Max tokens to generate in the reply.

    Returns:
        A normalized dict shaped like {"choices": [{"message":
        {"content": <text>}}]} — the same shape the previous Qwen client
        returned, so nothing downstream needs to change.

    Raises:
        GeminiAPIError: if the API key is missing or the request fails
            for any reason (network error, timeout, API error).
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise GeminiAPIError(ERROR_NO_API_KEY)

    system_instruction, contents = _messages_to_gemini(messages)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction or DEFAULT_SYSTEM_PROMPT,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        text = (response.text or "").strip()
    except genai_errors.APIError as exc:
        # Covers non-2xx API responses (auth failures, bad model name,
        # rate limits, etc.) — the SDK's own error type.
        raise GeminiAPIError(ERROR_REQUEST_FAILED) from exc
    except Exception as exc:  # noqa: BLE001 - network/transport errors, etc.
        raise GeminiAPIError(ERROR_REQUEST_FAILED) from exc

    if not text:
        raise GeminiAPIError(ERROR_MALFORMED_RESPONSE)

    return {"choices": [{"message": {"content": text}}]}


def get_gemini_response(user_message: str, context: str = "") -> str:
    """Get a single conversational reply from Gemini for a user message.

    Direct replacement for the old `get_qwen_response(user_message,
    context="")` — same signature, same behavior, same fail-soft
    guarantees. Everything in app.py that called `get_qwen_response`
    only needs its import (and call sites) updated to this name.

    Args:
        user_message: The user's chat message.
        context: Optional extra context to steer the reply — e.g. prior
            conversation summary, or details about the movie currently
            being discussed. Sent as the system instruction; falls back
            to DEFAULT_SYSTEM_PROMPT when empty.

    Returns:
        Gemini's reply text, or a friendly fallback string if the API
        key is missing or the request fails for any reason. Never raises.
    """
    messages = [
        {"role": "system", "content": context or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        data = call_gemini_chat(messages)
    except GeminiAPIError as exc:
        return str(exc)

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        return ERROR_MALFORMED_RESPONSE
