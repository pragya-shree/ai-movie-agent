"""
Centralized configuration for the Movie Recommender app.

Resolves the TMDB API key from, in order of priority:
1. Streamlit secrets (``st.secrets``) — used on Streamlit Community Cloud.
2. Environment variables loaded from a local ``.env`` file — used for
   local development.

No API key is ever hardcoded in this file or anywhere else in the project.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Project root (one level up from this src/ package).
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load variables from a local .env file, if present. This is a no-op on
# Streamlit Cloud, where .env files are not deployed and st.secrets is
# used instead.
load_dotenv(BASE_DIR / ".env")

# Data file locations (unchanged, at project root, as required).
MOVIES_PKL_PATH: Path = BASE_DIR / "movies_dict.pkl"
SIMILARITY_PKL_PATH: Path = BASE_DIR / "similarity.pkl"

# TMDB API configuration.
TMDB_BASE_URL: str = "https://api.themoviedb.org/3/movie"
PLACEHOLDER_POSTER_URL: str = "https://via.placeholder.com/500x750?text=No+Image"
PLACEHOLDER_ERROR_URL: str = "https://via.placeholder.com/500x750?text=Error"

# Qwen API configuration.
#
# NOT YET USED. This mirrors get_tmdb_api_key() below so that wiring up
# a real QwenLLMClient later is a drop-in: the key-resolution pattern
# (Streamlit secrets first, then .env) already exists and is tested via
# the TMDB path. src.agent currently only uses StubLLMClient, which
# never calls get_qwen_api_key() or reaches the network.
QWEN_MODEL_NAME: str = "qwen-plus"


def get_qwen_api_key() -> str | None:
    """Return the Qwen API key, or None if it isn't configured anywhere.

    Same resolution order as get_tmdb_api_key(): Streamlit secrets first,
    then the QWEN_API_KEY environment variable. Not called anywhere yet
    — present so the AI-agent architecture is complete ahead of actual
    Qwen integration.
    """
    try:
        if "QWEN_API_KEY" in st.secrets:
            return str(st.secrets["QWEN_API_KEY"])
    except Exception:
        pass

    return os.environ.get("QWEN_API_KEY")


def get_tmdb_api_key() -> str | None:
    """Return the TMDB API key, or None if it isn't configured anywhere.

    Checks Streamlit secrets first (Streamlit Cloud deployment), then
    falls back to the ``TMDB_API_KEY`` environment variable (local
    development via .env). Never raises if the key is missing — callers
    are expected to handle a None key gracefully.
    """
    try:
        if "TMDB_API_KEY" in st.secrets:
            return str(st.secrets["TMDB_API_KEY"])
    except Exception:
        # st.secrets raises if no secrets.toml exists at all — that's
        # fine, we just fall through to the environment variable.
        pass

    return os.environ.get("TMDB_API_KEY")
