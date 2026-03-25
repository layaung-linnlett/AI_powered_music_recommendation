"""Mood-based Streamlit UI for the Music Mood Classifier.

Users describe their mood in text or pick a quick-vibe button. The app
maps the mood to one of six genre categories and recommends playlists.
The audio feature slider mode is available as a secondary tab.

Run with:
    streamlit run ui-mood/app.py
"""

# ==== Standard Library Imports ====
import pickle
import sys
import urllib.parse
from pathlib import Path

# ==== Third-Party Imports ====
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==== Constants ====
MODEL_PATH: Path = PROJECT_ROOT / "models" / "final_model.pkl"
PIPELINE_PATH: Path = PROJECT_ROOT / "models" / "preprocessor.pkl"
ENCODER_PATH: Path = PROJECT_ROOT / "models" / "label_encoder.pkl"

BATCH_MAX_ROWS: int = 500

# (emoji, hex_color, tagline)
GENRE_STYLE: dict[str, tuple] = {
    "acoustic":    ("🍂", "#D97706", "Warm, quiet, and real"),
    "alternative": ("🌙", "#6D28D9", "Moody, layered, and raw"),
    "dance":       ("✨", "#DB2777", "Feel it in your feet"),
    "electronic":  ("⚡", "#0891B2", "Pure energy, no lyrics needed"),
    "heavy":       ("🔥", "#B91C1C", "Loud and unapologetically so"),
    "vocal":       ("🎤", "#C2410C", "The words are the music"),
}

PLAYLISTS: dict[str, list[dict]] = {
    "acoustic": [
        {"name": "Peaceful Piano",       "desc": "Soft instrumentals for quiet moments",       "query": "peaceful piano ambient playlist"},
        {"name": "Coffee Shop Acoustic", "desc": "Gentle singer-songwriter for slow mornings",  "query": "coffee shop acoustic playlist"},
        {"name": "Deep Focus",           "desc": "Ambient sounds to help you concentrate",      "query": "deep focus study ambient playlist"},
        {"name": "Rainy Day Folk",       "desc": "Folk and country for grey, reflective days",  "query": "rainy day folk acoustic playlist"},
    ],
    "alternative": [
        {"name": "Indie Essentials", "desc": "The best of indie rock and dream pop",     "query": "indie rock essentials playlist"},
        {"name": "90s Alt Rock",     "desc": "Grunge and alternative classics",           "query": "90s alternative grunge rock playlist"},
        {"name": "Road Trip Rock",   "desc": "Guitar-driven tracks for the open road",    "query": "road trip rock driving playlist"},
        {"name": "Psychedelic Haze", "desc": "Layered, hazy, and hypnotic",               "query": "psychedelic rock playlist"},
    ],
    "dance": [
        {"name": "Today's Top Hits", "desc": "The biggest pop tracks right now",           "query": "todays top hits pop playlist"},
        {"name": "Summer Vibes",     "desc": "Sun-drenched latin pop and tropical beats",  "query": "summer latin pop vibes playlist"},
        {"name": "Party Anthems",    "desc": "High-energy dance floor bangers",            "query": "party anthems dance floor playlist"},
        {"name": "K-Pop Central",    "desc": "Irresistible K-Pop and J-Pop hits",         "query": "kpop jpop hits playlist"},
    ],
    "electronic": [
        {"name": "Electronic Focus", "desc": "Deep house and ambient for concentration",   "query": "electronic focus deep house playlist"},
        {"name": "Late Night EDM",   "desc": "Festival-ready drops and big room sets",     "query": "late night edm festival playlist"},
        {"name": "Synthwave Drive",  "desc": "Retro-future synths for night drives",       "query": "synthwave retrowave night drive playlist"},
        {"name": "Drum and Bass",    "desc": "High-tempo breaks and rolling basslines",    "query": "drum and bass dnb playlist"},
    ],
    "heavy": [
        {"name": "Metal Classics", "desc": "The definitive heavy metal hall of fame",      "query": "metal classics rock playlist"},
        {"name": "Gym Fuel",       "desc": "Hard rock and metal for maximum effort",       "query": "gym workout metal rock playlist"},
        {"name": "Punk Rush",      "desc": "Fast, loud, and over in three minutes",        "query": "punk rock energy playlist"},
        {"name": "Modern Metal",   "desc": "New-wave metal and heavy alternative",         "query": "modern metal heavy alternative playlist"},
    ],
    "vocal": [
        {"name": "Hip-Hop Essentials", "desc": "The tracks that defined the genre",       "query": "hip hop rap essentials playlist"},
        {"name": "Lofi Hip-Hop",       "desc": "Lo-fi beats and laid-back flows",          "query": "lofi hip hop chill beats playlist"},
        {"name": "R&B Mood",           "desc": "Smooth soul for late evenings",            "query": "rnb soul mood playlist"},
        {"name": "Lyrical Bars",       "desc": "Wordsmiths and storytellers at their best","query": "lyrical rap storytelling playlist"},
    ],
}

QUICK_MOODS: list[tuple[str, str]] = [
    ("😊 Happy",      "I'm feeling happy and upbeat"),
    ("😔 Sad",        "I'm feeling sad and a bit down"),
    ("⚡ Energetic",  "I'm feeling energetic and pumped"),
    ("😌 Chill",      "I want something calm and relaxing"),
    ("😤 Fired Up",   "I'm feeling intense and motivated"),
    ("🎤 Expressive", "I want something lyric-heavy and expressive"),
    ("🌙 Moody",      "I'm feeling moody and introspective"),
    ("🎉 Party",      "I want to dance and celebrate"),
]

MOOD_KEYWORDS: dict[str, list[str]] = {
    "acoustic": [
        "calm", "relax", "peaceful", "quiet", "sad", "melanchol", "nostalgic",
        "soft", "gentle", "cozy", "study", "sleep", "meditat", "slow",
        "folk", "piano", "acoustic", "rain", "coffee", "morning", "tired",
        "lonely", "reflective", "introspective", "mellow", "serene", "down",
        "blue", "tearful", "heartbroken", "chill out",
    ],
    "alternative": [
        "indie", "angs", "creative", "complex", "road trip", "driving",
        "grunge", "different", "unique", "arty", "layered", "bittersweet",
        "restless", "searching", "alternative", "edgy", "guitar", "rock",
    ],
    "dance": [
        "happy", "party", "danc", "fun", "excit", "celebrat", "joyful", "upbeat",
        "euphoric", "summer", "tropical", "pop", "positive", "good vibes",
        "jumping", "moving", "groove", "bouncy", "festive", "wedding", "birthday",
        "smile", "giddy", "cheerful",
    ],
    "electronic": [
        "focus", "work", "gaming", "club", "night", "rave",
        "techno", "edm", "house", "beat", "bass", "electric",
        "productive", "concentrate", "running", "synth", "workout", "exercise",
    ],
    "heavy": [
        "angry", "intense", "powerful", "aggressive", "motivat", "pump",
        "metal", "hard", "strong", "rage", "gym", "beast", "hardcore",
        "scream", "loud", "furious", "adrenaline", "fired up", "frustrated",
        "lift", "sprint",
    ],
    "vocal": [
        "rap", "hip hop", "hiphop", "lyric", "word", "story", "flow", "verse",
        "spoken", "poetic", "r&b", "rnb", "soul", "express",
        "rhyme", "bars", "freestyle", "trap", "expressive",
    ],
}

FEATURE_CONFIG: dict[str, tuple] = {
    "popularity":       ("Popularity",        0,     100,    50,     1,    "Spotify popularity score (0=obscure, 100=viral)"),
    "duration_ms":      ("Duration (ms)",     0,     600000, 210000, 1000, "Track length in milliseconds"),
    "explicit":         ("Explicit",          0,     1,      0,      1,    "1 if explicit lyrics, 0 otherwise"),
    "danceability":     ("Danceability",      0.0,   1.0,    0.6,    0.01, "How suitable for dancing (0=low, 1=high)"),
    "energy":           ("Energy",            0.0,   1.0,    0.7,    0.01, "Perceptual intensity (0=calm, 1=intense)"),
    "key":              ("Key",               0,     11,     5,      1,    "Musical key (0=C, 1=C#, ..., 11=B)"),
    "loudness":         ("Loudness (dB)",    -60.0,  0.0,   -7.0,    0.1,  "Average loudness in decibels"),
    "mode":             ("Mode",              0,     1,      1,      1,    "1=major, 0=minor"),
    "speechiness":      ("Speechiness",      0.0,   1.0,    0.05,   0.01, "Presence of spoken words"),
    "acousticness":     ("Acousticness",     0.0,   1.0,    0.2,    0.01, "Confidence the track is acoustic"),
    "instrumentalness": ("Instrumentalness", 0.0,   1.0,    0.0,    0.01, "Probability of no vocals"),
    "liveness":         ("Liveness",         0.0,   1.0,    0.12,   0.01, "Probability of live recording"),
    "valence":          ("Valence",          0.0,   1.0,    0.5,    0.01, "Musical positiveness (0=sad, 1=happy)"),
    "tempo":            ("Tempo (BPM)",      0.0,   250.0,  120.0,  0.5,  "Estimated beats per minute"),
    "time_signature":   ("Time Signature",   1,     7,      4,      1,    "Beats per bar"),
}

CUSTOM_CSS: str = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: linear-gradient(160deg, #f3f0ff 0%, #fff0f9 45%, #f0f7ff 100%);
    min-height: 100vh;
}
.block-container {
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hero */
.hero { text-align: center; padding: 36px 0 12px; }
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin: 0 0 10px;
}
.hero-sub { font-size: 1.05rem; color: #6B7280; font-weight: 400; margin: 0; }

/* Mood section */
.mood-label { font-size: 1rem; font-weight: 600; color: #374151; margin: 24px 0 12px; text-align: center; }
.or-row {
    display: flex; align-items: center; gap: 12px;
    margin: 14px 0; color: #9CA3AF; font-size: 0.85rem; font-weight: 500;
}
.or-row::before, .or-row::after { content: ''; flex: 1; height: 1px; background: #E5E7EB; }

/* Text input */
.stTextInput > div > div > input {
    border-radius: 22px !important;
    border: 1.5px solid #E5E7EB !important;
    padding: 14px 22px !important;
    font-size: 1rem !important;
    background: rgba(255,255,255,0.9) !important;
    box-shadow: none !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
    outline: none !important;
}

/* Buttons */
.stButton > button {
    border-radius: 22px !important;
    font-weight: 600 !important;
    border: 1.5px solid #E5E7EB !important;
    background: rgba(255,255,255,0.85) !important;
    color: #374151 !important;
    padding: 8px 12px !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s, background 0.2s !important;
    backdrop-filter: blur(8px) !important;
}
.stButton > button:hover {
    border-color: #7c3aed !important;
    background: rgba(124,58,237,0.05) !important;
    color: #7c3aed !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: white !important;
    border: none !important;
    padding: 10px 28px !important;
    font-size: 0.95rem !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88 !important;
    color: white !important;
}

/* Genre result card */
.genre-card {
    border-radius: 22px;
    padding: 32px 28px 24px;
    color: white;
    text-align: center;
    margin: 24px 0 8px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.g-emoji { font-size: 3.2rem; display: block; margin-bottom: 6px; }
.g-name  { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; text-transform: capitalize; display: block; margin-bottom: 4px; }
.g-tag   { font-size: 1rem; opacity: 0.88; font-weight: 400; display: block; margin-bottom: 20px; }
.conf-row { margin: 6px 0; }
.conf-head {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: rgba(255,255,255,0.85);
    margin-bottom: 3px; font-weight: 500; text-transform: capitalize;
}
.conf-track { background: rgba(255,255,255,0.2); border-radius: 4px; height: 5px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 4px; background: rgba(255,255,255,0.75); }

/* Playlist cards */
.pl-heading { font-size: 1.1rem; font-weight: 700; color: #111827; letter-spacing: -0.01em; margin: 24px 0 12px; }
.pl-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 16px 20px;
    border: 1px solid rgba(240,240,255,0.9);
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.pl-info h4 { font-size: 0.95rem; font-weight: 600; color: #111827; margin: 0 0 3px; }
.pl-info p  { font-size: 0.82rem; color: #6B7280; margin: 0; }
.pl-btn {
    display: inline-block;
    background: #1DB954;
    color: white !important;
    padding: 7px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-decoration: none !important;
    white-space: nowrap;
    flex-shrink: 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.65);
    border-radius: 14px;
    padding: 4px;
    backdrop-filter: blur(8px);
}
.stTabs [data-baseweb="tab"] { border-radius: 10px; font-weight: 500; font-size: 0.9rem; padding: 8px 20px; }
.stTabs [aria-selected="true"] { background: white !important; box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; }
</style>
"""


# ==== Loaders ====

@st.cache_resource(show_spinner="Loading model...")
def load_artefacts() -> tuple:
    """Load and cache the model, preprocessing pipeline, and label encoder.

    Returns:
        A tuple of (model, pipeline, label_encoder).
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, pipeline, le


# ==== Core Logic ====

def map_mood_to_genre(text: str) -> tuple[str, dict[str, float]]:
    """Map a natural language mood description to the closest genre.

    Scores each genre by counting keyword matches in the input, then
    normalises the scores to sum to 1. Defaults to 'dance' with uniform
    scores if no keywords match.

    Args:
        text: Free-text mood description from the user.

    Returns:
        A tuple of (best_genre, normalised_scores).
    """
    text_lower = text.lower()
    raw: dict[str, int] = {g: 0 for g in MOOD_KEYWORDS}
    for genre, keywords in MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                raw[genre] += 1
    total = sum(raw.values())
    if total == 0:
        default = 1.0 / len(raw)
        return "dance", {g: default for g in raw}
    normalised = {g: raw[g] / total for g in raw}
    best = max(normalised, key=normalised.get)
    return best, normalised


def predict_single(record: dict, model, pipeline, le) -> tuple[str, dict[str, float] | None]:
    """Run inference on a single audio feature record.

    Args:
        record: Dict mapping feature name to its numeric value.
        model: Fitted LightGBM classifier.
        pipeline: Fitted sklearn preprocessing pipeline.
        le: Fitted LabelEncoder.

    Returns:
        A tuple of (predicted_label, scores) where scores maps each genre to
        a probability float, or None if predict_proba is unavailable.
    """
    df = pd.DataFrame([record])
    X_scaled = pipeline.transform(df)
    label_int = model.predict(X_scaled)[0]
    genre = le.inverse_transform([label_int])[0]
    scores = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        scores = {name: float(p) for name, p in zip(le.classes_, proba)}
    return genre, scores


def predict_batch(df_raw: pd.DataFrame, model, pipeline, le) -> pd.DataFrame:
    """Run batch inference on a DataFrame of audio features.

    Args:
        df_raw: DataFrame with audio feature columns.
        model: Fitted classifier.
        pipeline: Fitted preprocessing pipeline.
        le: Fitted LabelEncoder.

    Returns:
        df_raw with appended predicted_genre and confidence columns.
    """
    feature_cols = list(FEATURE_CONFIG.keys())
    for col in feature_cols:
        if col not in df_raw.columns:
            df_raw[col] = 0
    X = df_raw[feature_cols].copy()
    X_scaled = pipeline.transform(X)
    y_pred = model.predict(X_scaled)
    result = df_raw.copy()
    result["predicted_genre"] = le.inverse_transform(y_pred)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        result["confidence"] = (proba.max(axis=1) * 100).round(1)
    return result


# ==== HTML Builders ====

def _genre_card_html(genre: str, scores: dict[str, float] | None) -> str:
    """Return HTML for a genre result card with confidence bars.

    Args:
        genre: The predicted genre name.
        scores: Dict of genre to probability, or None.

    Returns:
        HTML string for the styled genre card.
    """
    emoji, color, tagline = GENRE_STYLE[genre]
    bars = ""
    if scores:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        bars = "".join(
            f"""<div class="conf-row">
                  <div class="conf-head">
                    <span>{name}</span><span>{score * 100:.1f}%</span>
                  </div>
                  <div class="conf-track">
                    <div class="conf-fill" style="width:{score * 100:.1f}%"></div>
                  </div>
                </div>"""
            for name, score in sorted_scores
        )
    return f"""
    <div class="genre-card" style="background: linear-gradient(135deg, {color}ee, {color}aa);">
        <span class="g-emoji">{emoji}</span>
        <span class="g-name">{genre}</span>
        <span class="g-tag">{tagline}</span>
        {bars}
    </div>"""


def _playlists_html(genre: str) -> str:
    """Return HTML for playlist recommendation cards for a given genre.

    Args:
        genre: The genre to render playlists for.

    Returns:
        HTML string containing the playlist section heading and all cards.
    """
    emoji, _, _ = GENRE_STYLE[genre]
    cards = "".join(
        f"""<div class="pl-card">
              <div class="pl-info">
                <h4>{pl['name']}</h4>
                <p>{pl['desc']}</p>
              </div>
              <a href="https://open.spotify.com/search/{urllib.parse.quote(pl['query'])}"
                 target="_blank" class="pl-btn">&#9654; Listen</a>
            </div>"""
        for pl in PLAYLISTS[genre]
    )
    return f'<div class="pl-heading">{emoji} Playlists for you</div>{cards}'


# ==== Tab Renderers ====

def _mood_tab() -> None:
    """Render the mood chat tab with quick-pick buttons and free-text input."""
    if "quick_mood" not in st.session_state:
        st.session_state.quick_mood = ""

    st.markdown('<p class="mood-label">Pick a vibe...</p>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (label, description) in enumerate(QUICK_MOODS):
        with cols[i % 4]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_mood = description

    st.markdown('<div class="or-row">or tell me how you feel</div>', unsafe_allow_html=True)

    user_text = st.text_input(
        label="mood_input",
        placeholder="e.g. I'm feeling nostalgic and want something soft...",
        label_visibility="collapsed",
    )

    if user_text:
        st.session_state.quick_mood = ""

    active = user_text or st.session_state.quick_mood
    if active:
        genre, scores = map_mood_to_genre(active)
        st.markdown(_genre_card_html(genre, scores), unsafe_allow_html=True)
        st.markdown(_playlists_html(genre), unsafe_allow_html=True)


def _advanced_tab(model, pipeline, le) -> None:
    """Render the audio feature slider prediction mode.

    Args:
        model: Fitted LightGBM classifier.
        pipeline: Fitted sklearn preprocessing pipeline.
        le: Fitted LabelEncoder.
    """
    st.markdown("Adjust the sliders to describe a track, then click **Predict**.")
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    record: dict = {}
    for idx, (key, cfg) in enumerate(FEATURE_CONFIG.items()):
        label, lo, hi, default, step, help_text = cfg
        with columns[idx % 3]:
            if isinstance(step, int):
                record[key] = st.slider(label, int(lo), int(hi), int(default), step, help=help_text)
            else:
                record[key] = st.slider(label, float(lo), float(hi), float(default), float(step), help=help_text)

    if st.button("Predict Genre", type="primary"):
        genre, scores = predict_single(record, model, pipeline, le)
        st.markdown(_genre_card_html(genre, scores), unsafe_allow_html=True)
        st.markdown(_playlists_html(genre), unsafe_allow_html=True)


def _batch_tab(model, pipeline, le) -> None:
    """Render the batch CSV upload mode.

    Args:
        model: Fitted LightGBM classifier.
        pipeline: Fitted sklearn preprocessing pipeline.
        le: Fitted LabelEncoder.
    """
    st.markdown(
        f"Upload a CSV with up to **{BATCH_MAX_ROWS} rows**. "
        "The file must include the audio feature columns. "
        "Missing columns are filled with zeros."
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
        if len(df_raw) > BATCH_MAX_ROWS:
            st.warning(f"File has {len(df_raw)} rows. Only the first {BATCH_MAX_ROWS} will be used.")
            df_raw = df_raw.head(BATCH_MAX_ROWS)
        with st.spinner("Predicting..."):
            result_df = predict_batch(df_raw, model, pipeline, le)
        st.success(f"Done. {len(result_df)} predictions ready.")
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            label="Download predictions as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="genre_predictions.csv",
            mime="text/csv",
        )


# ==== Entry Point ====

def main() -> None:
    """Main entry point for the mood Streamlit app."""
    st.set_page_config(
        page_title="Music Mood",
        page_icon="🎵",
        layout="centered",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <p class="hero-title">What are you in the mood for?</p>
        <p class="hero-sub">Tell us how you feel and we'll find your sound.</p>
    </div>
    """, unsafe_allow_html=True)

    model, pipeline, le = None, None, None
    try:
        model, pipeline, le = load_artefacts()
    except FileNotFoundError as exc:
        st.warning(
            f"Model file not found ({exc}). "
            "The mood tab still works without it. "
            "Run `python -m src.model_training` to enable the audio feature mode."
        )

    tab_mood, tab_advanced, tab_batch = st.tabs(["🎧  Mood", "🎛  Audio Features", "📂  Batch"])

    with tab_mood:
        _mood_tab()

    with tab_advanced:
        if model is None:
            st.info("Model not loaded. Run the training pipeline to enable this mode.")
        else:
            _advanced_tab(model, pipeline, le)

    with tab_batch:
        if model is None:
            st.info("Model not loaded. Run the training pipeline to enable this mode.")
        else:
            _batch_tab(model, pipeline, le)


if __name__ == "__main__":
    main()
