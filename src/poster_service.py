"""
Poster-fetching service using the TMDB API.

Behavior is unchanged from the original implementation: given a TMDB
movie ID, fetch its poster URL, falling back to a placeholder image if
the poster is missing or the request fails for any reason. Error
handling has been made more specific (missing key, timeout, bad status
code, malformed response) but the observable output — a poster URL or a
placeholder URL — is identical to before.
"""

import requests
import streamlit as st

from src.config import (
    PLACEHOLDER_ERROR_URL,
    PLACEHOLDER_POSTER_URL,
    TMDB_BASE_URL,
    get_tmdb_api_key,
)

REQUEST_TIMEOUT_SECONDS = 5


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    """Fetch a movie poster URL from TMDB for the given movie ID.

    Args:
        movie_id: The TMDB movie ID to fetch the poster for.

    Returns:
        A full poster image URL, or a placeholder URL if the poster is
        unavailable, the API key is missing, or the request fails.
    """
    api_key = get_tmdb_api_key()

    if not api_key:
        # No key configured — fail gracefully rather than crash the app.
        return PLACEHOLDER_ERROR_URL

    try:
        url = f"{TMDB_BASE_URL}/{movie_id}"
        response = requests.get(
            url,
            params={"api_key": api_key, "language": "en-US"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")
        if not poster_path:
            return PLACEHOLDER_POSTER_URL

        return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except requests.exceptions.RequestException:
        # Covers timeouts, connection errors, and non-2xx status codes.
        return PLACEHOLDER_ERROR_URL
    except (ValueError, KeyError):
        # Covers malformed/unexpected JSON responses.
        return PLACEHOLDER_ERROR_URL
