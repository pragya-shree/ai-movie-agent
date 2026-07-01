"""
Data loading utilities for the Movie Recommender app.

Loads the precomputed movie metadata and similarity matrix from disk.
These pickle files are treated as immutable model artifacts — this
module only reads them, it never regenerates or modifies them.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MOVIES_PKL_PATH, SIMILARITY_PKL_PATH


@st.cache_data(show_spinner=False)
def load_movies(path: Path = MOVIES_PKL_PATH) -> pd.DataFrame:
    """Load the movies dataset from a pickle file into a DataFrame.

    Args:
        path: Path to the movies_dict.pkl file.

    Returns:
        A DataFrame with (at least) 'movie_id' and 'title' columns.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
        RuntimeError: If the file exists but cannot be unpickled.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Movies data file not found at '{path}'. "
            "Make sure movies_dict.pkl is present in the project root."
        )

    try:
        with open(path, "rb") as f:
            movies_dict = pickle.load(f)
        return pd.DataFrame(movies_dict)
    except Exception as exc:
        raise RuntimeError(f"Failed to load movies data from '{path}': {exc}") from exc


@st.cache_data(show_spinner=False)
def load_similarity_matrix(path: Path = SIMILARITY_PKL_PATH):
    """Load the precomputed cosine similarity matrix from a pickle file.

    Args:
        path: Path to the similarity.pkl file.

    Returns:
        A 2D numpy array of pairwise similarity scores.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
        RuntimeError: If the file exists but cannot be unpickled.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Similarity matrix file not found at '{path}'. "
            "Make sure similarity.pkl is present in the project root."
        )

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load similarity matrix from '{path}': {exc}"
        ) from exc
