"""
Genre-to-seed-movie mapping, used to auto-answer bare genre requests
("horror", "suggest a comedy") without asking the user to name a movie.

Every title below was verified against the project's actual
movies_dict.pkl (4806 movies) before being added here — none are
guessed or invented. The recommendation flow only ever calls the
existing, unmodified recommend() with one of these titles; this module
never returns movie results itself, only picks which title to seed
recommend() with.
"""

import re

# genre keyword -> exact catalog title (must match movies['title'] verbatim,
# since it gets passed straight into recommender.recommend()).
GENRE_SEEDS: dict[str, str] = {
    "action": "Mad Max: Fury Road",
    "horror": "The Conjuring",
    "thriller": "Se7en",
    "comedy": "The Hangover",
    "romance": "The Notebook",
    "drama": "The Shawshank Redemption",
    "science fiction": "The Matrix",
    "sci-fi": "The Matrix",
    "fantasy": "Harry Potter and the Philosopher's Stone",
    "family": "The Lion King",
    "animation": "Toy Story",
    "crime": "The Godfather",
    "war": "Saving Private Ryan",
    "superhero": "The Avengers",
    "adventure": "Pirates of the Caribbean: The Curse of the Black Pearl",
}

# Longest keyword first, so "science fiction" is tried before it could
# ever be shadowed by a shorter overlapping key.
_GENRE_KEYWORDS_BY_LENGTH = sorted(GENRE_SEEDS, key=len, reverse=True)


def detect_genre(text: str) -> str | None:
    """Return the matched genre keyword if `text` mentions one, else None.

    Whole-word matching (not bare substring) so e.g. "war" doesn't
    false-positive inside an unrelated word. Deliberately simple, same
    spirit as the existing keyword-based is_movie_related() check.
    """
    lowered = text.lower()
    for genre in _GENRE_KEYWORDS_BY_LENGTH:
        pattern = r"(?<![a-z0-9])" + re.escape(genre) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            return genre
    return None


def seed_movie_for_genre(genre: str) -> str:
    """Look up the seed movie title for an already-detected genre."""
    return GENRE_SEEDS[genre]
