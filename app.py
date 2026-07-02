"""
Movie Recommender — Streamlit entry point.

Two experiences live here:
- A real chat interface, combining Gemini (conversational language) with
  the existing ML recommendation engine (actual movie picks). Gemini
  never invents titles; it only narrates whatever the ML engine already
  returned, or chats normally when the user isn't asking about movies.
- The original Classic Recommender (dropdown + button), left exactly as
  it was, so the untouched ML pipeline is always directly demonstrable.

All business logic still lives in src/ (data loading, the recommender,
poster fetching) — untouched by this step — and in gemini_client.py,
which replaced qwen_client.py as the sole LLM provider client. Everything
below is UI + a small amount of glue: detecting whether a message is
movie-related, finding which catalog movie it refers to, and combining
the two outputs. None of that glue changed when the provider did.
"""

import re
from difflib import get_close_matches

import streamlit as st

from gemini_client import get_gemini_response
from src.data_loader import load_movies, load_similarity_matrix
from src.poster_service import fetch_poster
from src.recommender import recommend

st.set_page_config(page_title="MovieMind AI", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e0e0e;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# 📦 Load Data
# ---------------------------
try:
    movies = load_movies()
    similarity = load_similarity_matrix()
except (FileNotFoundError, RuntimeError) as exc:
    st.error(f"⚠️ Could not load required data: {exc}")
    st.stop()

CATALOG_TITLES: list[str] = movies["title"].tolist()

# Titles shorter than this are excluded from fuzzy/substring matching —
# very short titles (e.g. a single-letter title) otherwise match as
# trivial substrings/near-matches of unrelated text (e.g. inside
# "comedy", "movie"). Used by is_movie_related() and find_movie_in_text().
MIN_TITLE_LENGTH_FOR_MATCHING = 3
FUZZY_CANDIDATE_TITLES = [t for t in CATALOG_TITLES if len(t) >= MIN_TITLE_LENGTH_FOR_MATCHING]


# ---------------------------
# 🎬 HERO SECTION (TOP)
# ---------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #E50914; font-size: 42px;'>
        MOVIEMIND AI
    </h1>
    <p style='text-align: center; color: #bbbbbb; font-size: 16px;'>
        Chat about movies — powered by Gemini + a Machine Learning recommender
    </p>
    <hr style="border: 1px solid #333;">
    """,
    unsafe_allow_html=True,
)


def render_recommendation_cards(names: list[str], posters: list[str]) -> None:
    """Render poster cards. Shared by the chat and classic tabs so both
    surfaces present the ML engine's output identically."""
    cols = st.columns(5)
    for i in range(len(names)):
        with cols[i]:
            st.markdown(
                """
                <div style="
                    background-color: #1a1a1a;
                    padding: 10px;
                    border-radius: 12px;
                    text-align: center;
                    height: 100%;
                ">
                """,
                unsafe_allow_html=True,
            )

            st.image(posters[i], width="stretch")

            st.markdown(
                f"""
                <p style="
                    color: white;
                    font-size: 14px;
                    font-weight: 600;
                    margin-top: 8px;
                ">
                {names[i]}
                </p>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# 🧠 INTENT DETECTION (kept simple, per spec)
# ---------------------------
MOVIE_KEYWORDS = (
    "movie", "movies", "film", "films", "watch", "recommend",
    "recommendation", "recommendations", "suggest", "suggestion",
    "similar", "similar to", "like watching", "cinema", "genre",
    "sequel", "franchise",
    # Genres — a bare genre mention ("something funny" aside, explicit
    # genre words) is still a strong movie-intent signal even without
    # one of the trigger words above.
    "action", "comedy", "thriller", "drama", "horror", "romance",
)

# Greetings / small talk that must ALWAYS go straight to Gemini, never
# through title detection. This is a whole-message check (after
# normalizing case/punctuation) rather than a keyword-in-text check —
# deliberately, so it can't misfire on a real movie-related sentence
# that merely opens with a pleasantry (e.g. "Hi, recommend me a movie"
# should still get recommendations, not be swallowed by a greeting match).
GREETING_PHRASES = {
    "hi", "hii", "hiya", "hello", "hey", "heyy", "yo", "sup", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "how are you today",
    "hows it going", "how's it going", "whats up", "what's up",
    "thanks", "thank you", "bye", "goodbye", "see you", "ok", "okay",
}


def _normalize_for_smalltalk(text: str) -> str:
    """Lowercase, drop simple punctuation, collapse whitespace.

    Used only for the greeting check — deliberately conservative (it
    doesn't touch letters), so it just turns "Hello!!" / "Hello." /
    "  Hello " into a plain "hello" for a clean set lookup.
    """
    cleaned = re.sub(r"[!?.,]+", "", text.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def is_greeting_or_smalltalk(text: str) -> bool:
    """Is this message *just* a greeting/pleasantry, with no other content?

    This is what actually fixes "Hello" being misread as the movie
    "Hellboy": greetings are checked and bypassed BEFORE any fuzzy
    title matching ever runs, rather than relying on a fuzzy-match
    cutoff to reject them after the fact.
    """
    return _normalize_for_smalltalk(text) in GREETING_PHRASES


# Fuzzy-match confidence thresholds (difflib ratio, 0-1). Raised from
# their previous values after discovering they let short greetings and
# common phrases slip through as "confident" title matches — e.g.
# "hello" vs. "Hellboy" scores 0.83, and "good morning" vs. "Good
# Morning, Vietnam" scores 0.73, both of which cleared the old cutoffs.
TRIGGER_PHRASE_FUZZY_CUTOFF = 0.6   # after "like X" / "similar to X"
BARE_TITLE_FUZZY_CUTOFF = 0.85      # matching the whole raw message
MIN_MESSAGE_LENGTH_FOR_BARE_FUZZY = 4  # skip fuzzy matching very short messages


def is_movie_related(text: str) -> bool:
    """Cheap, keyword-based intent check: is this message about movies?

    Deliberately simple (per spec): checks for common movie-chat trigger
    words only. Greetings are excluded up front so a short pleasantry
    can never be classified as movie-related just because it happens to
    resemble a title (that used to be handled by a fuzzy-match fallback
    here, which was the actual source of the "Hello" -> "Hellboy" bug —
    title detection now lives solely in find_movie_in_text(), with its
    own, stricter confidence thresholds).
    """
    if is_greeting_or_smalltalk(text):
        return False

    lowered = text.lower()
    return any(keyword in lowered for keyword in MOVIE_KEYWORDS)


def _title_appears_in(text_lower: str, title: str) -> bool:
    """Whole-word(ish) containment check: does `title` appear in `text_lower`?

    Uses boundary look-arounds instead of plain `in` so a short title
    isn't falsely "found" inside an unrelated longer word (e.g. without
    this, a one-letter title would match inside "comedy" or "movie").
    """
    if len(title) < MIN_TITLE_LENGTH_FOR_MATCHING:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(title.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


def find_movie_in_text(text: str) -> str | None:
    """Try to identify which catalog movie a free-text message refers to.

    Tries, in order: (1) a catalog title appearing verbatim in the text,
    (2) a title following a trigger phrase like "like X" or "similar to
    X", (3) a fuzzy match on the whole message. Returns an exact catalog
    title (safe to pass into recommend()) or None if nothing matched.

    Greetings/small talk never reach any of this — see
    is_greeting_or_smalltalk(), checked by the caller (and defensively
    here too, in case this is ever called on its own).
    """
    if is_greeting_or_smalltalk(text):
        return None

    lowered = text.lower()

    # 1) Direct (whole-word) containment — longest match wins so
    # "Spider-Man" isn't shadowed by a shorter unrelated title also
    # present in the text. Exact titles ("Interstellar", "Avatar", "The
    # Dark Knight") are always caught here, with no fuzzy cutoff
    # involved at all — this step is unaffected by the threshold changes
    # below.
    contained = [title for title in CATALOG_TITLES if _title_appears_in(lowered, title)]
    if contained:
        return max(contained, key=len)

    # 2) Trigger-phrase extraction, then fuzzy-match the fragment. The
    # trigger phrase itself ("like"/"similar to"/...) is already a
    # strong movie-intent signal, so this cutoff can stay a bit more
    # permissive than step 3's.
    for pattern in (r"(?:similar to|like|recommend(?:ations)? for|based on)\s+(.+)",):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            fragment = match.group(1).strip(" ?.!\"'")
            fuzzy = get_close_matches(
                fragment, FUZZY_CANDIDATE_TITLES, n=1, cutoff=TRIGGER_PHRASE_FUZZY_CUTOFF
            )
            if fuzzy:
                return fuzzy[0]

    # 3) Last resort: fuzzy-match the whole message (handles bare titles,
    # e.g. the user just typing "Interstellar" with nothing else). No
    # trigger phrase to lean on here, so: skip very short messages
    # entirely, and require a high confidence ratio — this is what
    # rejects "hello" (0.83 vs. "Hellboy") and "good morning" (0.73 vs.
    # "Good Morning, Vietnam") without needing to hardcode those titles.
    stripped = text.strip()
    if len(stripped) < MIN_MESSAGE_LENGTH_FOR_BARE_FUZZY:
        return None

    fuzzy = get_close_matches(stripped, FUZZY_CANDIDATE_TITLES, n=1, cutoff=BARE_TITLE_FUZZY_CUTOFF)
    return fuzzy[0] if fuzzy else None



# ---------------------------
# 🤝 COMBINING GEMINI + THE ML ENGINE
# ---------------------------
CHAT_PERSONA = (
    "You are MovieMind AI, a friendly, concise movie-chat assistant. "
    "Keep replies short (2-4 sentences) and natural."
)


def build_recommendation_reply(user_message: str, movie_title: str) -> tuple[str, list[dict]]:
    """Get real recommendations from the ML engine, then have Gemini narrate them.

    The ML engine (unmodified recommend()) decides *which* movies and in
    *what order* — Gemini is only given that already-decided list and asked
    to phrase a reply, so it never has the chance to invent a title.
    """
    names, movie_ids = recommend(movie_title, movies, similarity)
    posters = [fetch_poster(movie_id) for movie_id in movie_ids]
    recommendations = [
        {"title": name, "poster_url": poster}
        for name, poster in zip(names, posters)
    ]

    context = (
        f"{CHAT_PERSONA} The ML recommendation engine (content-based, cosine "
        f"similarity) matched the user's message to the movie '{movie_title}' "
        f"and produced exactly these recommendations, in this order: "
        f"{', '.join(names)}. Write a short, warm reply presenting them. "
        "Do not mention any movie title that isn't in that list, and do not "
        "reorder or add to it — the poster cards below your reply will show "
        "them, so you don't need to describe each one in detail."
    )
    reply = get_gemini_response(user_message, context=context)
    return reply, recommendations


def build_clarifying_reply(user_message: str) -> str:
    """Movie-related message, but no catalog title could be identified."""
    context = (
        f"{CHAT_PERSONA} The user seems to want a movie recommendation, but "
        "no specific movie title could be matched in the catalog. Ask them, "
        "in one short sentence, which movie to base recommendations on."
    )
    return get_gemini_response(user_message, context=context)


def build_general_reply(user_message: str) -> str:
    """Not movie-related — just have a normal conversation."""
    return get_gemini_response(user_message, context=CHAT_PERSONA)


def handle_chat_message(user_message: str) -> tuple[str, list[dict]]:
    """Combine intent detection + ML engine + Gemini into one reply.

    Order matters here:
    1. Greetings/small talk always go straight to Gemini — this is the
       fix for "Hello" being misread as the movie "Hellboy".
    2. Otherwise, try to find a confident movie title in the message
       (covers both "recommend movies like Interstellar" and someone
       just typing "Interstellar" alone).
    3. If no title was found but the message still looks movie-related
       by keyword (e.g. "I want an action movie"), ask which movie.
    4. Otherwise, it's just normal conversation.

    Returns (reply_text, recommendations) — recommendations is empty
    unless the ML engine actually produced some.
    """
    if is_greeting_or_smalltalk(user_message):
        return build_general_reply(user_message), []

    movie_title = find_movie_in_text(user_message)
    if movie_title is not None:
        return build_recommendation_reply(user_message, movie_title)

    if is_movie_related(user_message):
        return build_clarifying_reply(user_message), []

    return build_general_reply(user_message), []


def render_assistant_message(content: str, recommendations: list[dict]) -> None:
    """Render one assistant turn, used for both history replay and live replies.

    Per spec: when the ML engine produced recommendations, the reply is
    shown under a "🧠 AI Explanation" heading, followed by a "🎬
    Recommended Movies" heading over the poster cards. Plain chat (no
    recommendations) is shown as-is, with no headings.
    """
    if recommendations:
        st.markdown("#### 🧠 AI Explanation")
        st.write(content)
        st.markdown("#### 🎬 Recommended Movies")
        names = [r["title"] for r in recommendations]
        posters = [r["poster_url"] for r in recommendations]
        render_recommendation_cards(names, posters)
    else:
        st.write(content)


# ---------------------------
# 🗂️ TABS
# ---------------------------
tab_chat, tab_classic = st.tabs(["💬 Chat with MovieMind AI", "🎯 Classic Recommender"])

# ---------------------------
# 💬 CHAT INTERFACE (Gemini + ML engine, combined)
# ---------------------------
with tab_chat:
    st.caption(
        "Chat naturally — ask general questions, or ask for recommendations "
        "(e.g. \"movies like Inception\") and the ML engine will pick the "
        "titles while Gemini writes the reply."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict] = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message["content"], message.get("recommendations") or [])
            else:
                st.write(message["content"])

    user_input = st.chat_input("Try: “recommend movies like Inception”")

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "recommendations": []}
        )
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking... 🤖"):
            reply, recommendations = handle_chat_message(user_input)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply, "recommendations": recommendations}
        )
        with st.chat_message("assistant"):
            render_assistant_message(reply, recommendations)

# ---------------------------
# 🎯 CLASSIC RECOMMENDER (unchanged behavior)
# ---------------------------
with tab_classic:
    st.markdown("### 🎥 Search Movie")
    selected_movie_name = st.selectbox(
        "🎥 Select a movie",
        movies["title"].values,
        label_visibility="collapsed",
    )

    if st.button("🎬 Show Recommendations"):

        with st.spinner("Fetching best recommendations... 🎯"):
            names, movie_ids = recommend(selected_movie_name, movies, similarity)
            posters = [fetch_poster(movie_id) for movie_id in movie_ids]

        st.markdown("## 🍿 Recommended for You")
        render_recommendation_cards(names, posters)
