"""
Movie Recommender — Streamlit entry point.

This file is intentionally thin: it configures the page, loads data via
src.data_loader, generates recommendations via src.recommender, fetches
posters via src.poster_service, and renders the UI. All business logic
lives in the src/ package.
"""

import streamlit as st

from src.agent import MovieAgent, StubLLMClient
from src.agent.schemas import Message
from src.data_loader import load_movies, load_similarity_matrix
from src.poster_service import fetch_poster
from src.recommender import recommend

st.set_page_config(page_title="Movie Recommender", layout="wide")

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


# ---------------------------
# 🎬 HERO SECTION (TOP)
# ---------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #E50914; font-size: 42px;'>
        MOVIE RECOMMENDER
    </h1>
    <p style='text-align: center; color: #bbbbbb; font-size: 16px;'>
        Discover movies based on your taste using Machine Learning
    </p>
    <hr style="border: 1px solid #333;">
    """,
    unsafe_allow_html=True,
)


def render_recommendation_cards(names: list[str], posters: list[str]) -> None:
    """Render poster cards. Shared by the classic and agent tabs so both
    surfaces present tool/algorithm output identically."""
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


tab_classic, tab_agent = st.tabs(["🎯 Classic Recommender", "🤖 AI Agent (Beta)"])

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

# ---------------------------
# 🤖 AI AGENT (BETA) — architecture preview, no Qwen integration yet
# ---------------------------
with tab_agent:
    st.caption(
        "This chat is powered by a rule-based stand-in for the LLM "
        "(`StubLLMClient`), not Qwen yet. It exercises the full agent "
        "architecture — tool calling into the same recommendation engine "
        "used by the Classic tab — so the Qwen integration will be a "
        "drop-in swap rather than a redesign."
    )

    if "agent" not in st.session_state:
        st.session_state.agent = MovieAgent(StubLLMClient(), movies, similarity)
    if "agent_history" not in st.session_state:
        st.session_state.agent_history: list[Message] = []

    for message in st.session_state.agent_history:
        if message.role in ("user", "assistant") and message.content:
            with st.chat_message(message.role):
                st.write(message.content)

    user_input = st.chat_input(
        "Try: “recommend movies like Inception”"
    )

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking... 🤖"):
            response = st.session_state.agent.handle_message(
                user_input, st.session_state.agent_history
            )

        st.session_state.agent_history.append(Message(role="user", content=user_input))
        st.session_state.agent_history.append(
            Message(role="assistant", content=response.reply)
        )

        with st.chat_message("assistant"):
            st.write(response.reply)
            if response.recommendations:
                names = [r["title"] for r in response.recommendations]
                posters = [r["poster_url"] for r in response.recommendations]
                render_recommendation_cards(names, posters)
