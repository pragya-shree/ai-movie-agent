# 🎬 Movie Recommendation System

A machine learning-based movie recommender that suggests similar movies and displays posters using the TMDB API. Built with Python and Streamlit.

---
## Project Preview

A smart movie recommendation system that suggests similar movies using machine learning and shows real posters using TMDB API.

👉 Try it live here: https://moviesystem-mnznjrkqmbj5pnfubpmnos.streamlit.app/

---

## 📌 Features
- 🎯 Content-based movie recommendations
- 🎬 Movie posters using TMDB API
- ⚡ Fast similarity-based engine
- 🌐 Streamlit web app
- 📊 ML similarity model

---

## 🧠 Tech Stack
- Python
- Pandas
- NumPy
- Streamlit
- Requests
- TMDB API

---

## 🧠 How It Works

1. Movies are converted into feature vectors using NLP techniques
2. Cosine similarity is used to find similar movies (precomputed, stored in `similarity.pkl`)
3. Top matching movies are recommended
4. Posters are fetched using TMDB API
5. Results are displayed using Streamlit UI

## 📊 Machine Learning Approach

- Technique: Content-Based Filtering
- Vectorization: CountVectorizer / TF-IDF (done offline, prior to this app)
- Similarity Metric: Cosine Similarity

## 📁 Project Structure

```
movie_system/
├── app.py                       # Streamlit entry point — Classic tab (unchanged)
│                                 # + Chat tab (Gemini + ML engine combined)
├── movies_dict.pkl              # Precomputed movie metadata (unchanged)
├── similarity.pkl               # Precomputed cosine similarity matrix (unchanged)
├── requirements.txt
├── .env.example                 # Template for local TMDB_API_KEY + GEMINI_API_KEY
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example     # Template for Streamlit Cloud secrets
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized config, secrets resolution
│   ├── data_loader.py            # Cached pickle loading
│   ├── recommender.py            # Core recommendation algorithm (UNCHANGED)
│   ├── poster_service.py         # TMDB poster fetching (UNCHANGED)
│   └── agent/                    # AI-agent architecture (new)
│       ├── __init__.py           # Public exports
│       ├── schemas.py            # Message / ToolCall / AgentResponse dataclasses
│       ├── tools.py               # Wraps recommender/poster as callable tools
│       ├── llm_client.py          # LLMClient interface + StubLLMClient
│       ├── prompts.py             # System prompt (ready for Gemini)
│       └── agent.py               # MovieAgent: the tool-calling orchestration loop
└── README.md
```

### 🤖 AI-Agent Architecture (chat tab now live on Gemini)

The app now has two tabs:

- **🎯 Classic Recommender** — exactly the original dropdown → button →
  poster-grid flow. Byte-for-byte the same algorithm, same code path.
- **💬 Chat with MovieMind AI** — a chat interface that calls
  `gemini_client.get_gemini_response()` for conversational replies, and
  the *same* `recommend()` / `fetch_poster()` functions the Classic tab
  uses for actual movie picks. Gemini narrates; it never invents titles.

The `src/agent/` package (`MovieAgent`, `LLMClient`, `StubLLMClient`,
tool-calling schemas) documented in the file tree above is a separate,
more elaborate tool-calling architecture built earlier in this project's
history. It is not currently wired into `app.py`'s chat tab, which
integrates Gemini directly instead. It's left in place as a reusable
building block if a full tool-calling agent loop is wanted later — its
`LLMClient` interface would accept a `GeminiLLMClient` implementation
the same way it was designed to accept a Qwen one.

`get_gemini_api_key()` in `src/config.py` and the `GEMINI_API_KEY`
entries in `.env.example` / `secrets.toml.example` are what the chat
tab actually reads from, following the same resolution pattern as the
TMDB key (Streamlit secrets first, then local `.env`).

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/pragya-shree/movie_system.git
cd movie_system
pip install -r requirements.txt

# Set up your TMDB API key locally:
cp .env.example .env
# then edit .env and set TMDB_API_KEY=your_real_key

streamlit run app.py
```

---

## 🔑 API Setup (TMDB)

Get an API key: https://www.themoviedb.org/settings/api

**Local development:** copy `.env.example` to `.env` and set `TMDB_API_KEY`.

**Streamlit Community Cloud:** open your app's Settings → Secrets, and paste in the contents of `.streamlit/secrets.toml.example` with your real key:

```toml
TMDB_API_KEY = "your_real_key_here"
```

No API key is ever hardcoded in the source code.

---

## 🔑 API Setup (Gemini)

Get an API key: https://aistudio.google.com/apikey

**Local development:** add `GEMINI_API_KEY` to your `.env` (see `.env.example`).

**Streamlit Community Cloud:** add it alongside `TMDB_API_KEY` in Settings → Secrets:

```toml
GEMINI_API_KEY = "your_real_key_here"
```

Without this key configured, the Chat tab still runs — `gemini_client.py`
fails soft and shows a friendly "API key isn't configured" message
instead of crashing. The Classic Recommender tab doesn't need this key
at all; it never calls Gemini.

---

## 🌐 Links
- Live App: https://moviesystem-mnznjrkqmbj5pnfubpmnos.streamlit.app/
- GitHub: https://github.com/pragya-shree/movie_system
- TMDB API: https://www.themoviedb.org/settings/api

---

## ⚖️ Model Validity & Limitations

This movie recommendation system is based on a **content-based filtering approach**, where movies are recommended by measuring similarity between feature vectors (such as genres, keywords, overview text, and tags).

### ✔️ On what basis recommendations are valid:
- Movies are compared using **cosine similarity**
- Similarity is calculated from text-based features like genres, keywords, and movie overview
- The system recommends movies with the **highest similarity scores**

### ⚠️ Limitations:
- The model does not understand real-world movie meaning or context
- It only works on **mathematical/text similarity**, not human-level understanding
- Recommendations may sometimes include movies that feel unrelated to humans
- Accuracy depends heavily on the quality of input features
- It does not include user preferences or watch history (no collaborative filtering)

### 💡 Conclusion:
This system provides **reasonably good recommendations based on feature similarity**, but it is **not 100% accurate** and can be improved using advanced techniques like TF-IDF tuning, embeddings, or hybrid recommendation systems.

## ✨ Future Improvements
- Wire the `src/agent/` tool-calling architecture (`MovieAgent`) up to
  Gemini via a `GeminiLLMClient`, for multi-turn tool-calling instead of
  the current direct-call chat integration
- Netflix-style UI
- Search feature
- Faster recommendations
- Mobile optimization

---

## 👨‍💻 Author
Pragya Shree

---

⭐ If you like this project, give it a star!
