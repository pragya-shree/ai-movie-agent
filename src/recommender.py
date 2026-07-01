"""
Content-based movie recommendation logic.

This module contains the exact same algorithm as the original app: for a
given movie title, look up its row in the precomputed cosine similarity
matrix, sort all other movies by similarity score, and return the top 5.

No changes have been made to the algorithm itself — only its location.
"""

import pandas as pd


def recommend(movie: str, movies: pd.DataFrame, similarity) -> tuple[list[str], list[int]]:
    """Return the top 5 movies most similar to the given movie.

    Args:
        movie: The exact title of the movie to base recommendations on,
            as it appears in the 'title' column of `movies`.
        movies: DataFrame with 'movie_id' and 'title' columns, indexed
            the same way as the rows/columns of `similarity`.
        similarity: Precomputed 2D similarity matrix (e.g. cosine
            similarity), where similarity[i][j] is the similarity score
            between movies at DataFrame index i and j.

    Returns:
        A tuple of (recommended_titles, recommended_movie_ids), each a
        list of 5 items, ordered from most to least similar.
    """
    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1],
    )[1:6]

    recommended_titles: list[str] = []
    recommended_ids: list[int] = []

    for i in movies_list:
        recommended_ids.append(movies.iloc[i[0]].movie_id)
        recommended_titles.append(movies.iloc[i[0]].title)

    return recommended_titles, recommended_ids
