"""
Tool layer: exposes the existing, unmodified recommendation engine to the
agent as a set of callable "tools" with Qwen/OpenAI-compatible
function-calling schemas.

IMPORTANT: this module does not implement any recommendation logic. It
only wraps ``src.recommender.recommend`` and ``src.poster_service.fetch_poster``
so an LLM (via tool calling) can invoke them. The algorithm itself is
never touched here.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Callable

import pandas as pd

from src.poster_service import fetch_poster
from src.recommender import recommend


@dataclass
class ToolDefinition:
    """A single agent-callable tool: schema + the function it dispatches to."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema, Qwen/OpenAI "function" format
    handler: Callable[..., Any]

    def to_schema(self) -> dict[str, Any]:
        """Qwen/OpenAI-compatible function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _search_movies(query: str, movies: pd.DataFrame, limit: int = 5) -> list[str]:
    """Fuzzy-match a user-typed title against the known movie catalog.

    This does not affect recommendation quality or ranking — it only
    helps translate free-text chat input ("recommend me something like
    spiderman") into an exact title the (unmodified) recommender expects.
    """
    titles = movies["title"].tolist()
    lower_map = {t.lower(): t for t in titles}

    query_lower = query.lower().strip()

    # Exact (case-insensitive) match first.
    if query_lower in lower_map:
        return [lower_map[query_lower]]

    # Substring matches next (e.g. "spider" -> "Spider-Man").
    substring_hits = [t for t in titles if query_lower in t.lower()]
    if substring_hits:
        return substring_hits[:limit]

    # Fall back to fuzzy matching for typos.
    return get_close_matches(query, titles, n=limit, cutoff=0.5)


def _recommend_movies(
    movie_title: str, movies: pd.DataFrame, similarity
) -> dict[str, Any]:
    """Tool handler for the 'recommend_movies' tool.

    Calls the existing, unmodified `recommend()` algorithm and enriches
    the result with poster URLs via the existing `fetch_poster()`. Raises
    a ValueError (caught by the executor) if the title isn't recognized,
    so the agent can ask the user to clarify or pick from suggestions.
    """
    if movie_title not in movies["title"].values:
        suggestions = _search_movies(movie_title, movies)
        raise ValueError(
            f"'{movie_title}' was not found in the catalog. "
            f"Closest matches: {suggestions or 'none found'}."
        )

    names, movie_ids = recommend(movie_title, movies, similarity)
    posters = [fetch_poster(mid) for mid in movie_ids]

    return {
        "source_movie": movie_title,
        "recommendations": [
            {"title": name, "movie_id": mid, "poster_url": poster}
            for name, mid, poster in zip(names, movie_ids, posters)
        ],
    }


def build_tools(movies: pd.DataFrame, similarity) -> list[ToolDefinition]:
    """Construct the tool registry, bound to the already-loaded data.

    Binding `movies`/`similarity` here (rather than having tools reload
    them) keeps a single, cached copy of the pickled artifacts in memory
    and keeps this module free of any data-loading responsibility — that
    stays owned by ``src.data_loader``.
    """
    return [
        ToolDefinition(
            name="search_movies",
            description=(
                "Search the movie catalog for titles matching a free-text "
                "query. Use this when the user's wording doesn't exactly "
                "match a catalog title, to find candidates before calling "
                "recommend_movies."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text movie title or partial title.",
                    }
                },
                "required": ["query"],
            },
            handler=lambda query: {"matches": _search_movies(query, movies)},
        ),
        ToolDefinition(
            name="recommend_movies",
            description=(
                "Get the top 5 movies most similar to a given movie title, "
                "using the precomputed content-based similarity model. The "
                "title must exactly match a title in the catalog — use "
                "search_movies first if unsure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "movie_title": {
                        "type": "string",
                        "description": "Exact catalog title to base recommendations on.",
                    }
                },
                "required": ["movie_title"],
            },
            handler=lambda movie_title: _recommend_movies(
                movie_title, movies, similarity
            ),
        ),
    ]


def get_tool_schemas(movies: pd.DataFrame, similarity) -> list[dict[str, Any]]:
    """Return Qwen/OpenAI-compatible tool schemas for the LLM's tools=[...] param."""
    return [tool.to_schema() for tool in build_tools(movies, similarity)]


def execute_tool(
    tool_name: str, arguments: dict[str, Any], tools: list[ToolDefinition]
) -> Any:
    """Dispatch a tool call by name, returning its result or raising.

    Raises:
        KeyError: if `tool_name` doesn't match any registered tool.
        Exception: whatever the underlying handler raises (e.g. ValueError
            for an unrecognized movie title) — the caller (agent.py) is
            responsible for turning this into a user-facing ToolResult.
    """
    by_name = {tool.name: tool for tool in tools}
    if tool_name not in by_name:
        raise KeyError(f"Unknown tool: '{tool_name}'")

    return by_name[tool_name].handler(**arguments)
