"""
AI-agent architecture for the Movie Recommender.

This package wraps the existing (unmodified) recommendation engine —
``src.recommender``, ``src.poster_service``, ``src.data_loader`` — behind
an agent orchestration layer, so that a conversational LLM (Gemini) can be
plugged in later as a pure drop-in replacement for ``StubLLMClient``.

Nothing in here changes the recommendation algorithm. This package only
adds a decision/orchestration layer *around* it:

    user message
        -> MovieAgent.handle_message()
        -> LLMClient.chat()            (stub today, Gemini later)
        -> tool_calls?  -> execute_tool()  -> src.recommender / src.poster_service
        -> AgentResponse

Public API:
    MovieAgent        - orchestrator (see agent.py)
    StubLLMClient      - rule-based placeholder LLM (see llm_client.py)
    LLMClient           - abstract interface Gemini will implement (see llm_client.py)
    get_tool_schemas    - Gemini/OpenAI-style function-calling schemas (see tools.py)
"""

from src.agent.agent import MovieAgent
from src.agent.llm_client import LLMClient, StubLLMClient
from src.agent.tools import get_tool_schemas

__all__ = [
    "MovieAgent",
    "LLMClient",
    "StubLLMClient",
    "get_tool_schemas",
]
